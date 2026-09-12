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
        """weight_scale is "row" | "none" | "threshold".

        "threshold" is section 3.3's arm, and it is not an arbitrary substitution.
        A POSITIVE per-channel scale is provably redundant wherever its output is
        consumed by another sign(), because sign(alpha*z) == sign(z) for alpha > 0.
        So alpha can only be load-bearing where the value meets something
        scale-sensitive: the residual accumulation, the attention scores, and the
        logits. This mode drops the multiplicative alpha and learns an ADDITIVE
        per-channel threshold instead, which is byte-identical (one float per
        channel) and asks whether the scale's real job was thresholding.
        Phase 0 measured the target it has to beat: removing scales outright costs
        0.4358 bpb (2.4485 against 2.0127, dense 1.7571).
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.binarise_acts = binarise_acts
        self.weight_scale, self.act_scale, self.clip = weight_scale, act_scale, clip
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # LEARNABLE per-output-channel scale, not a detached function of |W|.
        #
        # Deriving it as mean|W| kills any zero-initialised layer permanently: the
        # scale is 0, so the binary weight is 0, the output is 0, and because the
        # scale is detached the gradient to the latent weight is 0 too. nanochat
        # zero-initialises every c_proj (the residual-branch projection of every
        # block, so each block starts as identity), which meant 2 dead matmuls per
        # layer, 24 of ~48 at depth 12, frozen from step 0 and unrecoverable.
        #
        # Section 3.9 of the plan already amended the definition to permit one float
        # per output channel, so making it learnable costs nothing that was not
        # already being paid and lets a zero-init layer grow its own scale back.
        # SHAPE IS 1-D ON PURPOSE. As (out_features, 1) this is 2-D, and
        # GPT.setup_optimizer routes every ndim==2 parameter to Muon
        # (gpt.py:11290, catch-all at :11343). That breaks it three ways:
        #   1. Newton-Schulz orthogonalises the update, and the only orthogonal
        #      factor of an (N,1) matrix is the unit column, so every channel gets
        #      an identical update magnitude. A per-channel scale whose channels
        #      cannot differ in the update is not a per-channel scale.
        #   2. optim.py:448 multiplies the LR by max(1, shape[-2]/shape[-1])**0.5,
        #      which is 32x at out_features=1024.
        #   3. Muon applies weight decay (gpt.py:11460) while every AdamW group in
        #      setup_optimizer uses 0.0, so the scale is dragged toward exp(0)=1.
        # 1-D routes to struct_adamw_params instead. Never fired before because
        # binary.py had only run through standalone probes, never base_train.
        self.log_alpha = nn.Parameter(torch.zeros(out_features)) \
            if weight_scale == "row" else None
        self.theta = nn.Parameter(torch.zeros(out_features)) \
            if weight_scale == "threshold" else None
        # ONE scalar per layer alongside the per-channel threshold. Without it the
        # arm is not the experiment it claims to be: theta is additive and cannot
        # control output magnitude, so sign(W)@x comes out ~sqrt(in_features) too
        # large (measured std 3.42 against row-scale's 0.54) and the arm would fail
        # for a scale reason rather than an information one. A single scalar is
        # 1 float per LAYER against out_features per layer, so the byte comparison
        # against "row" is unchanged, and it isolates the real question: does the
        # multiplicative scale need to be PER CHANNEL, or is per-channel
        # thresholding plus one global gain enough?
        self.log_g = nn.Parameter(torch.zeros(())) \
            if weight_scale == "threshold" else None

    def reset_parameters(self):
        # Same constraint as BinaryEmbedding: stay inside the clip window. The usual
        # 1/sqrt(fan_in) is far inside it for realistic widths, but clamp anyway so a
        # narrow layer cannot silently produce dead bits.
        nn.init.normal_(self.weight, mean=0.0,
                        std=min(1.0 / math.sqrt(self.in_features), self.clip / 3.0))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.set_scale_from_weight()

    SCALE_FLOOR = 1e-4

    def binary_weight(self):
        w = sign_ste(self.weight, self.clip)
        if self.log_alpha is not None:
            # ADDITIVE floor, not a clamp. The forward is w = sign(W) * alpha, so
            # dL/dW is proportional to alpha: if a scale reaches zero the LATENT
            # WEIGHT stops receiving gradient and the layer is dead forever. This is
            # the same structural failure as the zero-initialised c_proj bug, but
            # reached by TRAINING rather than by init, and it is what killed R5: the
            # loss went 10.96 -> 7.28 -> 8.27 -> 10.04 and then pinned at 10.397207,
            # which is exactly ln(32768), i.e. a constant-logit model.
            #
            # SCALE_FLOOR was only ever applied in set_scale_from_weight, so nothing
            # stopped log_alpha running to -inf during training. A torch.clamp would
            # not fix it either: at the clamp the gradient is zero and the scale is
            # stuck. Adding the floor keeps alpha >= SCALE_FLOOR unconditionally, so
            # the latent weight keeps a gradient path even if the scale itself stalls.
            alpha = self.log_alpha.exp() + self.SCALE_FLOOR
            w = w * alpha.unsqueeze(-1)
        return w

    @torch.no_grad()
    def set_scale_from_weight(self):
        """Initialise alpha (or the threshold arm's global gain) from mean|W|."""
        if self.log_g is not None:
            g = self.weight.abs().mean().clamp_min(self.SCALE_FLOOR)
            self.log_g.copy_(g.log())
        if self.log_alpha is None:
            return
        a = self.weight.abs().mean(dim=1).clamp_min(self.SCALE_FLOOR)
        self.log_alpha.copy_(a.log())

    def forward(self, x):
        if self.binarise_acts:
            xb = sign_ste(x, self.clip)
            if self.act_scale == "token":
                xb = xb * x.detach().abs().mean(dim=-1, keepdim=True)
        else:
            xb = x
        y = F.linear(xb, self.binary_weight().to(xb.dtype), self.bias)
        if self.log_g is not None:
            y = y * (self.log_g.exp() + self.SCALE_FLOOR).to(y.dtype)
        if self.theta is not None:
            y = y + self.theta.to(y.dtype)
        return y

    def flops_per_token(self):
        """Same shape of matmul as the nn.Linear it replaced: 2 FLOPs per MAC.

        Binarisation changes what an operation COSTS, not how many there are, which
        is the whole reason section 3.1 replaces the FLOPs axis. Reporting anything
        else here would smuggle the claim into the baseline accounting.
        """
        return 2 * self.in_features * self.out_features

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

    def flops_per_token(self):
        """Zero: this is a lookup, exactly as nn.Embedding is.

        GPT.estimate_flops calls this on any wte that is not a bare nn.Embedding,
        because a CODED input embedding is a matmul rather than a gather and the 6N
        proxy would misprice it. A learned 1-bit table is still a gather, so the
        answer is 0 and the parameters are excluded from the proxy as usual.
        """
        return 0

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
                # A zero row carries no sign information at all, so give it random
                # signs rather than the constant +1 that sign(0) would produce.
                zero_rows = (w.abs().sum(dim=1) == 0)
                if zero_rows.any():
                    w = w.clone()
                    w[zero_rows] = torch.randn_like(w[zero_rows]) * (clip / 3.0)
                sc = w.abs().max().clamp(min=1e-12)
                new.weight.copy_(w * (clip / 3.0) / sc)
                # Preserve "starts as a no-op" by giving those rows the floor scale
                # instead of a zero one, which keeps the block near identity at init
                # while leaving the layer trainable.
                new.set_scale_from_weight()
                if zero_rows.any():
                    new.log_alpha.data[zero_rows] = math.log(new.SCALE_FLOOR)
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
