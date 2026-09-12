"""Natively binary layers.

The spine of the direction, measured rather than assumed in Phase 0: **consistency
beats precision.** Four components in a partially binarised model had NEGATIVE
in-situ cost, i.e. restoring them to bf16 made the model worse. A full-precision
component feeding binarised consumers is worse than a consistent binary one, so the
object to build is a transformer designed in Hamming space, not a real-valued one
with quantisation bolted on.

Phase 0 also priced the purity question. Per-output-channel scales are worth
**0.4358 bpb** (2.0127 with them, 2.4485 without, dense 1.7571) and cost one float
per row, about 1% of the byte budget. They stay, they are counted, and section 3.9
of the plan amends the definition accordingly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SignSTE(torch.autograd.Function):
    """sign() with a clipped straight-through estimator.

    The clip matters: without it, latent weights drift arbitrarily far from the
    decision boundary and stop responding to gradient, which is the classic BNN
    failure. The estimator is standard and is NOT a contribution; section 4.6's
    update rule is.
    """

    @staticmethod
    def forward(ctx, x, clip):
        ctx.save_for_backward(x)
        ctx.clip = clip
        s = torch.sign(x)
        return torch.where(s == 0, torch.ones_like(s), s)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= ctx.clip).to(g.dtype), None


def sign_ste(x, clip=1.0):
    return _SignSTE.apply(x, clip)


class BinaryLinear(nn.Module):
    """y = (sign(W) * alpha) @ (sign(x) * beta).

    alpha is one float per OUTPUT channel, beta one float per token, both the
    L1-optimal XNOR-Net choice mean|.|. `binarise_acts=False` gives the W1A16 rung,
    which changes no operation and exists only as a ladder step.

    The latent weight is fp during Phase 1 so the architecture can be tested with the
    optimiser held fixed. Phase 3 replaces it with the 1-bit + counter rule; merging
    the two would make a failure uninterpretable.
    """

    def __init__(self, in_features, out_features, bias=False, binarise_acts=True,
                 weight_scale="row", act_scale="token", clip=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.binarise_acts = binarise_acts
        self.weight_scale, self.act_scale, self.clip = weight_scale, act_scale, clip
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def reset_parameters(self):
        # Same constraint as BinaryEmbedding: stay inside the clip window. The usual
        # 1/sqrt(fan_in) is far inside it for realistic widths, but clamp anyway so a
        # narrow layer cannot silently produce dead bits.
        nn.init.normal_(self.weight, mean=0.0,
                        std=min(1.0 / math.sqrt(self.in_features), self.clip / 3.0))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def binary_weight(self):
        w = sign_ste(self.weight, self.clip)
        if self.weight_scale == "row":
            w = w * self.weight.detach().abs().mean(dim=1, keepdim=True)
        return w

    def forward(self, x):
        if self.binarise_acts:
            xb = sign_ste(x, self.clip)
            if self.act_scale == "token":
                xb = xb * x.detach().abs().mean(dim=-1, keepdim=True)
        else:
            xb = x
        return F.linear(xb, self.binary_weight().to(xb.dtype), self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"acts={'1b' if self.binarise_acts else 'fp'}, "
                f"w_scale={self.weight_scale}, a_scale={self.act_scale}")


class BinaryEmbedding(nn.Module):
    """A learned 1-bit table: V x D bits plus one float per row.

    Section 4.4's primary arm. At V=32,768 and D=1024 that is 4.2 MB against 33.5 MB
    for an fp16 table at d=512, so 8x smaller while twice as wide. Every token keeps
    a private full-width row, so nothing about the function is approximated: the same
    argument as section 3.5 makes for the head.
    """

    def __init__(self, num_embeddings, embedding_dim, scale="row", clip=1.0):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.scale, self.clip = scale, clip
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def reset_parameters(self):
        # The magnitude of a latent weight has no effect on the forward pass, which
        # sees only its sign; its only role is inertia for the STE. So it must be
        # initialised INSIDE the clip window, or the weight is born dead: outside it
        # the STE passes no gradient and the bit can never flip again. Initialising
        # at std=1.0 against clip=1.0 left 10.19% of weights dead at step 0 and
        # frozen there, which is exactly what O6's dead-bit diagnostic is for.
        nn.init.normal_(self.weight, mean=0.0, std=self.clip / 3.0)

    def forward(self, idx):
        rows = self.weight[idx]
        out = sign_ste(rows, self.clip)
        if self.scale == "row":
            out = out * rows.detach().abs().mean(dim=-1, keepdim=True)
        return out

    def extra_repr(self):
        return f"V={self.num_embeddings}, D={self.embedding_dim}, scale={self.scale}"


def _swap(parent, name, new):
    setattr(parent, name, new)


@torch.no_grad()
def binarise_model_(model, binarise_acts=True, linear=True, embeddings=True,
                    skip=(), weight_scale="row", act_scale="token", clip=1.0,
                    verbose=False):
    """Replace nn.Linear with BinaryLinear and nn.Embedding with BinaryEmbedding.

    In place, preserving the trained weights as the latent values. `skip` holds
    substrings of module names to leave alone, which is how the Phase 1 ladder is
    built: the in-situ cost table (LEARNINGS) orders which ones come off last.
    """
    swapped = []
    for mod_name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if any(s in full for s in skip):
                continue
            if linear and isinstance(child, nn.Linear):
                new = BinaryLinear(child.in_features, child.out_features,
                                   bias=child.bias is not None,
                                   binarise_acts=binarise_acts,
                                   weight_scale=weight_scale, act_scale=act_scale,
                                   clip=clip)
                new = new.to(child.weight.device, child.weight.dtype)
                # Rescale the donor weights into the clip window. Only signs matter to
                # the forward pass, so this is function-preserving up to the per-row
                # scale, and it stops inherited magnitudes from being born dead.
                w = child.weight
                sc = w.abs().max().clamp(min=1e-12)
                new.weight.copy_(w * (clip / 3.0) / sc)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
                _swap(mod, child_name, new)
                swapped.append((full, "Linear"))
            elif embeddings and isinstance(child, nn.Embedding):
                new = BinaryEmbedding(child.num_embeddings, child.embedding_dim,
                                      scale=weight_scale, clip=clip)
                new = new.to(child.weight.device, child.weight.dtype)
                w = child.weight
                sc = w.abs().max().clamp(min=1e-12)
                new.weight.copy_(w * (clip / 3.0) / sc)
                _swap(mod, child_name, new)
                swapped.append((full, "Embedding"))
    if verbose:
        for f, k in swapped:
            print(f"  binarised {k:<10} {f}")
    return swapped
