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
from torch.utils.checkpoint import checkpoint
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
def nativise_model_(model, config, tau=1.0, hard=False, resid_width=1, verbose=False):
    """Replace every gpt.Block in transformer.h with a BinaryBlock.

    This is the CONSISTENCY change (plan section 4.2/4.3), distinct from the WIDTH
    change. binarise_model_ quantises the operands of a real-valued design and
    leaves softmax attention and the fp residual in place; this replaces the
    operations themselves. O5's measured argument for it: four components had
    NEGATIVE in-situ cost, so an fp component feeding binarised consumers was worse
    than a consistent binary one.

    Call this INSTEAD of binarise_model_ for the body, then binarise_model_ with
    linear=False for the interfaces, or just let this run first and binarise the
    remaining Linear/Embedding modules afterwards.
    """
    swapped = []
    h = model.transformer.h
    if resid_width <= 0:
        # Derived, not typed. The residual carries 2*n_layer branch contributions on
        # top of a saturated +-1 embedding, and the two failure modes bracket the
        # answer tightly. Measured at depth 8, agreement between the block stack's
        # output sign and its input:
        #
        #   width  1 -> 49.8%   destroyed: majority-of-two with a constant tiebreak
        #   width  4 -> 78.4%   signal survives and layers can still change it
        #   width 16 -> 100.0%  frozen: no branch can move a saturated channel
        #
        # n_layer/2 lands in the middle of that and scales the right way with depth.
        resid_width = max(2, len(h) // 2)
    for i in range(len(h)):
        blk = BinaryBlock(config, i, tau=tau, hard=hard, resid_width=resid_width)
        blk = blk.to(next(model.parameters()).device, next(model.parameters()).dtype)
        blk.reset_parameters()
        h[i] = blk
        swapped.append((f"transformer.h.{i}", "Block->BinaryBlock"))
    if verbose:
        for f, k in swapped:
            print(f"  nativised {k:<22} {f}")
    return swapped


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


# ---------------------------------------------------------------------------
# Natively binary operations (plan sections 4.2 and 4.3)
#
# Everything above this line QUANTISES a real-valued design: BinaryLinear computes
# sign(W) @ sign(x), which is the same operation a linear layer computes, with
# coarser operands. Phase 1 measured what that costs: +0.62 bpb at d=512, growing
# with data.
#
# Section 3.8 v3 says why. A binary neuron summing n terms emits n+1 levels, so its
# output carries log2(n) bits: 9.0 at n=512 against an fp neuron's ~16. The
# information dies at the ACCUMULATION, not at the weights, and the remedy is width.
# A native binary transformer is not d=512 with 1-bit weights, it is d=8192 BITS at
# the same memory.
#
# These are the operations that make width usable: bundling instead of fp addition,
# Hamming retrieval instead of softmax averaging.
# ---------------------------------------------------------------------------


def bundle(x, dim=0, keepdim=False):
    """Majority vote: the VSA superposition operator, and the binary residual add.

    Bundling is how a set of hypervectors becomes one that is similar to all of them.
    Its capacity is O(D / log D) items before recall degrades, which is why width is
    the resource and why bundling and width have to be introduced together.

    Ties at an even count resolve to +1 rather than 0, because a zero would leave a
    hole in the sign that every downstream popcount reads as "no information".

    The sign goes through the SAME straight-through estimator as everything else. A
    hard torch.where here severs the graph outright: the first version of this
    function did exactly that and the whole block returned a tensor with no grad_fn.
    The clip window must also scale with the number of bundled items, because a sum
    of m terms in {-1,+1} lands in [-m, m] and a clip of 1.0 would zero the gradient
    for every element except exact ties.
    """
    m = x.shape[dim] if dim is not None else x.numel()
    s = x.sum(dim=dim, keepdim=keepdim)
    return sign_ste(s, clip=float(max(m, 1)))


class BundledResidual(nn.Module):
    """Residual accumulation as majority bundling, with an accumulator-width knob.

    An fp residual stream re-introduces exactly the precision the rest of the model
    gave up, and O5 measured that mixing hurts: four components had NEGATIVE in-situ
    cost, so an fp component feeding binarised consumers was worse than a consistent
    binary one.

    `width` is the plan's section 4.3 knob. width=1 is pure majority bundling, the
    VSA operator. Larger widths keep a bounded integer accumulator and re-binarise
    against a learned per-channel threshold, trading strict binarity for the ability
    to carry a magnitude across a few layers. Normalisation is absent by
    construction: sign is scale-free, so section 3.3's threshold absorbs what
    RMSNorm was doing.
    """

    def __init__(self, n_embd, width=1, clip=1.0):
        super().__init__()
        self.width, self.clip = width, clip
        self.threshold = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x, branch):
        if self.width <= 1:
            # A majority vote with a LEARNED bias, not a plain one. The first version
            # called bundle() directly here, so self.threshold received no gradient
            # at all and the layer had a dead parameter. A biased majority is also
            # what section 3.3 means by the threshold absorbing normalisation.
            s = x + branch
            return sign_ste(s - self.threshold, clip=2.0)
        # width > 1 is a BOUNDED ACCUMULATOR, and the first version was a no-op. Both
        # x and branch are +-1, so (x + branch).clamp(-width, width) never left
        # [-2, 2] and every width behaved exactly like width 1. Worse, it signed the
        # result, so the stream was re-binarised at all 2*n_layer sublayers and no
        # magnitude could survive even one of them. That is the accumulation
        # bandwidth of section 3.8 destroyed by an implementation detail.
        #
        # The accumulator is carried in [-1, 1] with resolution 1/width rather than
        # as a raw integer in [-width, width]. They are the same object scaled, and
        # the scaled form means every downstream sign_ste keeps its clip of 1.0 and
        # so keeps passing gradient, instead of zeroing it everywhere outside a
        # window it no longer matches. The threshold biases the branch, which is what
        # gives it a gradient; section 3.3's point that sign() is scale-free is why
        # the scaling is free to choose.
        return (x + (branch - self.threshold) / self.width).clamp(-1.0, 1.0)

    def extra_repr(self):
        return f"width={self.width} ({'majority bundling' if self.width <= 1 else 'int accumulator'})"


class HammingAttention(nn.Module):
    """Attention as Hamming-radius retrieval, not softmax-weighted averaging.

    scores = popcount(XNOR(q, k)) = n - 2*hamming, computed as an ordinary matmul
    over +-1 (they are the same number; see the identity below). What changes is
    everything after: no softmax over fp scores, no 1/sqrt(d), no fp value average.
    Keys above a learned threshold are retrieved and their values BUNDLED by
    majority, which is Kanerva's sparse distributed memory read.

        <q, k> = 2*agreements - n = n - 2*hamming(q, k)

    A hard mask has no gradient, so training uses a temperature-annealed softmax over
    the INTEGER scores as a differentiable surrogate and anneals toward the hard
    threshold. Inference uses the hard path, which is the one a popcount kernel runs.
    `tau -> 0` recovers hard top-k retrieval.
    """

    def __init__(self, n_embd, n_head, tau=1.0, hard=False, clip=1.0, chunk=256):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.tau, self.hard, self.clip, self.chunk = tau, hard, clip, chunk
        # One threshold per head: the Hamming radius at which a key counts as a match.
        self.radius = nn.Parameter(torch.zeros(n_head))

    def forward(self, q, k, v):
        """Chunked over query blocks.

        The unchunked version materialises the full (B, H, T, T) score tensor, which
        FlashAttention exists specifically to avoid. At B=128, T=2048, d=3584 that is
        28 GB in bf16 plus 56 GB for the fp32 softmax: 84 GB for ONE layer on an 80 GB
        card. Chunking the queries caps the FORWARD peak at (B, H, chunk, T) and changes
        nothing about the result.

        Chunking alone does NOT cap the TRAINING peak, which is the trap this hit on an
        H100. Softmax backward needs its own output and the value einsum needs the same
        weights, so autograd saves `w` for every chunk, and the chunks sum back to the
        full (B, H, T, T) matrix in fp32: at B=64, H=28, T=2048 that is 30 GB for ONE
        layer, which is exactly the tensor chunking was supposed to avoid. Each chunk is
        therefore recomputed in backward instead of stored, which makes the saved memory
        O(chunk) as intended at roughly 30% more attention FLOPs.
        """
        B, T, H, D = q.shape
        qb, kb, vb = sign_ste(q, self.clip), sign_ste(k, self.clip), sign_ste(v, self.clip)
        radius = self.radius.view(1, -1, 1, 1)
        chunk = self.chunk if self.chunk > 0 else T
        recompute = self.training and torch.is_grad_enabled() and chunk < T
        outs = []
        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            args = (qb[:, start:stop], kb, vb, radius, start, stop, T, D)
            if recompute:
                outs.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                outs.append(self._chunk(*args))
        return sign_ste(torch.cat(outs, dim=1), self.clip)

    def _chunk(self, qc, kb, vb, radius, start, stop, T, D):
        # popcount as an integer matmul; a b1 kernel computes the same numbers
        scores = torch.einsum("bthd,bshd->bhts", qc, kb)
        pos = torch.arange(start, stop, device=qc.device).unsqueeze(1)
        keys = torch.arange(T, device=qc.device).unsqueeze(0)
        allowed = keys <= pos
        scores = scores.masked_fill(~allowed.view(1, 1, stop - start, T), float("-inf"))
        # The 0.5 makes an exact tie resolve the same way in both branches. A +-1 dot
        # product over an even D is even, so scores == 0 is common, and it means
        # exactly half the bits agree: no evidence. The hard branch excluded it and
        # sigmoid half-voted it, which alone put 4.9% of output bits between the two.
        # Subtracting half a step changes nothing for integer scores under `> 0` and
        # sends the tie to zero weight as tau falls.
        gate = scores - radius.to(scores.dtype) * D - 0.5
        if self.hard:
            w = (gate > 0).to(vb.dtype)
        else:
            # A Kanerva read gates each key INDEPENDENTLY against a radius and bundles
            # everything inside it. Softmax is the opposite operation: it makes keys
            # compete for a fixed unit of weight, so it does not converge to the hard
            # path as tau falls, and the soft and hard branches were computing
            # different functions rather than sharper and blunter versions of one.
            # A sigmoid gate is the same function as the hard threshold, softened.
            #
            # The temperature is divided by sqrt(D) because these are +-1 dot products
            # over D terms, so their standard deviation IS sqrt(D). This is the same
            # 1/sqrt(head_dim) that ordinary attention applies, and dropping it was not
            # purification: section 3.3 argues sign() is scale-free so RMSNorm has
            # nothing to do, which is true of the residual stream and false of a gate,
            # where the scale relative to the temperature is the entire operation.
            # Without it, tau=1.0 attended 1.8 keys out of 256.
            w = torch.sigmoid(gate.float() / (max(self.tau, 1e-6) * math.sqrt(D))).to(vb.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bhts,bshd->bthd", w, vb)

    def extra_repr(self):
        return f"heads={self.n_head}, head_dim={self.head_dim}, tau={self.tau}, hard={self.hard}"


class BinaryKVFFN(nn.Module):
    """The FFN as a binary associative memory: bind against learned keys, bundle values.

    An FFN already IS a key-value memory (Geva et al., "Transformer Feed-Forward
    Layers Are Key-Value Memories"); `W2 . act(W1 x)` scores the input against the
    rows of W1 and mixes the rows of W2 by those scores. Written natively in Hamming
    space that becomes: match against binary keys by popcount, gate, bundle the
    matching binary values. Same object, operations native to the medium.
    """

    def __init__(self, n_embd, n_slots, tau=1.0, hard=False, clip=1.0):
        super().__init__()
        self.keys = nn.Parameter(torch.empty(n_slots, n_embd))
        self.values = nn.Parameter(torch.empty(n_slots, n_embd))
        self.threshold = nn.Parameter(torch.zeros(n_slots))
        self.tau, self.hard, self.clip = tau, hard, clip
        self.n_slots, self.n_embd = n_slots, n_embd

    def reset_parameters(self):
        std = min(1.0 / math.sqrt(self.n_embd), self.clip / 3.0)
        nn.init.normal_(self.keys, 0.0, std)
        nn.init.normal_(self.values, 0.0, std)
        nn.init.zeros_(self.threshold)

    def forward(self, x):
        xb = sign_ste(x, self.clip)
        kb = sign_ste(self.keys, self.clip)
        vb = sign_ste(self.values, self.clip)
        scores = F.linear(xb, kb.to(xb.dtype)) - self.threshold.to(xb.dtype) - 0.5  # ties, as above
        if self.hard:
            w = (scores > 0).to(x.dtype)
        else:
            # Same two corrections as HammingAttention, and they matter more here
            # because there are 4d slots rather than T keys. Softmax at tau=1.0 over
            # scores with standard deviation sqrt(d)=22.6 retrieved 1.4 effective slots
            # out of 2048: the associative memory was a lookup table returning one
            # stored vector, which is the opposite of the bundling this module exists
            # to do. Independent sigmoid gates at the score's own scale restore it.
            w = torch.sigmoid(scores.float() / (max(self.tau, 1e-6) * math.sqrt(self.n_embd)))
        w = (w / w.sum(-1, keepdim=True).clamp_min(1.0)).to(vb.dtype)
        return sign_ste(w @ vb, self.clip)

    def extra_repr(self):
        return f"d={self.n_embd}, slots={self.n_slots}, tau={self.tau}, hard={self.hard}"


class BinaryBlock(nn.Module):
    """A transformer block with no floating-point operations in its data path.

    Drop-in for gpt.Block: same forward signature, so binarise_model_ can swap it.

    What differs from binarising a dense block:
      - attention is Hamming retrieval and majority bundling, not softmax averaging
      - the FFN is a binary associative memory, not two projections and a nonlinearity
      - the residual is majority bundling, not an fp add
      - there is NO normalisation. sign() is scale-free, so RMSNorm has nothing to do
        that the learned thresholds do not already do (plan section 3.3)

    Rotary is applied to the fp projections BEFORE binarisation: a rotation of a +-1
    vector is not +-1, so the order matters and position information would otherwise
    be destroyed by the sign.
    """

    def __init__(self, config, layer_idx, tau=1.0, hard=False, resid_width=1):
        super().__init__()
        d = config.n_embd
        self.n_head = max(1, d // 128)
        self.head_dim = d // self.n_head
        self.layer_idx = layer_idx
        self.c_q = BinaryLinear(d, d)
        self.c_k = BinaryLinear(d, d)
        self.c_v = BinaryLinear(d, d)
        self.c_proj = BinaryLinear(d, d)
        self.attn = HammingAttention(d, self.n_head, tau=tau, hard=hard,
                                     clip=config.binary_clip,
                                     chunk=int(getattr(config, "binary_attn_chunk", 256)))
        self.mlp = BinaryKVFFN(d, 4 * d, tau=tau, hard=hard, clip=config.binary_clip)
        self.resid_attn = BundledResidual(d, width=resid_width, clip=config.binary_clip)
        self.resid_mlp = BundledResidual(d, width=resid_width, clip=config.binary_clip)

    def reset_parameters(self):
        for m in (self.c_q, self.c_k, self.c_v, self.c_proj, self.mlp):
            m.reset_parameters()

    def init_weights(self):
        self.reset_parameters()

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None,
                eet_frozen_kv=False, frozen_k=None, frozen_v=None):
        assert kv_cache is None, "BinaryBlock has no KV cache path; training only for now"
        B, T, D = x.shape
        xb = sign_ste(x, self.attn.clip)
        q = self.c_q(xb).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(xb).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(xb).view(B, T, self.n_head, self.head_dim)
        if cos_sin is not None:
            from nanochat.gpt import apply_rotary_emb
            cos, sin = cos_sin
            # Rotate BEFORE the sign inside HammingAttention: rotating +-1 does not
            # give +-1, so binarising first would discard the position signal.
            q = apply_rotary_emb(q, cos[:T], sin[:T])
            k = apply_rotary_emb(k, cos[:T], sin[:T])
        if ve is not None:
            # Value embeddings enter by bundling, which is how a hypervector is added
            # to another in this algebra.
            v = bundle(torch.stack((v, ve.view(B, T, self.n_head, self.head_dim)), 0), dim=0)
        a = self.attn(q, k, v).reshape(B, T, D)
        # onto the STREAM, not onto xb: accumulating onto the signed copy threw the
        # incoming accumulator away at every block and made resid_width unreachable.
        x = self.resid_attn(x, self.c_proj(a))
        x = self.resid_mlp(x, self.mlp(x))
        return x
