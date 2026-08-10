"""MoL: Mixture of Layers with Hybrid Attention (Ternovtsii & Bilak, 2026).

Faithful reimplementation of arXiv:2605.09516v1, "Mixture of Layers with Hybrid
Attention: Parallel Thin Blocks for Sparse Transformer Compute", for use as a
baseline against MST.

THERE IS NO OFFICIAL CODE RELEASE. All 14 pages contain no repository link, no
code-availability statement, no algorithm listing and no pseudocode. Everything
here is recovered from Eq (1), Eq (2), sections 2 to 4, and the configuration
tables, then validated against their published parameter counts (see
tests/test_mol.py, which pins Table 1's 85.3M and Table 5's 0.61B active).

The architecture, in their notation:

    ThinBlock_i(x) = W_up,i . ( Block_thin,i(W_down,i . x) - W_down,i . x )   (1)
    SplitStage(x)  = x + (1/k) . sum_{i in top-k} w_i . ThinBlock_i(x)       (2)

A split stage replaces one full-width transformer block with N thin blocks at
d_thin << d_model. S of them are "shared" (always active on every token, full
softmax attention, providing global context); the remaining N-S are "routed",
selected top-k per token, and attend only over the tokens routed to them. Their
notation for this is S+KofN.

WHY THIS EXISTS. MoL wraps every thin block in its own W_down/W_up pair, so its
plumbing cost is D/(D + 6*d_thin) of a wrapped block, which grows without bound
as blocks narrow: their own section 2.3 quotes 40% at d_thin=256, 57% at 128 and
73% at 64. MST partitions the residual stream instead and pays (N+8)/(13N+8),
which is independent of width. That contrast is the point of the comparison, and
scripts/p09_projection_overhead.py derives it from num_scaling_params() alone.

NOT YET IMPLEMENTED: Gated DeltaNet in the routed blocks, which is their headline
1+3of15 configuration. Their section 5.3 reports a dense DeltaNet control matching
dense softmax within 0.01 PPL, so the structural gain is not the attention swap,
and they run a "MoL all-softmax" control themselves. But their Table 2 does price
DeltaNet at 0.85 PPL inside MoL at d_thin=256, so until `mol_routed_attn` grows a
'deltanet' arm we are not running their best configuration. See OPEN_QUESTIONS.md.
"""

import math
from dataclasses import fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Block, GPTConfig, Linear, has_ve, norm

# Geometry that MoL sets deliberately on the shadow config, plus the architecture
# switches. Everything else Block/CausalSelfAttention/MLP read must be at its
# GPTConfig default, or a thin block silently stops being the dense baseline at
# reduced width and the whole head-to-head is void.
_SHADOW_OVERRIDDEN = frozenset({
    'n_embd', 'n_head', 'n_kv_head', 'p34_ffn_mult',
    'use_mol', 'use_mst', 'use_eet',
})


# Flags that on their own change what module Block/CausalSelfAttention/MLP builds,
# mapped to the value at which they are OFF.
#
# Checked against the OFF value, NOT against the GPTConfig default. base_train sets
# some fields to sentinels that differ from the dataclass default while meaning
# "inactive" (`p23_std_moe_topk=-1` is the one that took down a whole sweep), so a
# default-comparison guard false-positives on a perfectly ordinary run.
_BLOCK_GATES = {
    'dense_intermediate_ln': 0,
    'p18_dynamic_activation': 0, 'p18_layer_drop': 0.0, 'p18_mixture_norm': 0,
    'p18_per_channel_scale': 0,
    'p19_attn_logit_bias': 0, 'p19_head_importance': 0, 'p19_residual_gate': 0,
    'p19_spectral_reparam': 0, 'p19_ve_bias': 0, 'p19_weight_noise': 0.0,
    'p20_adwi': 0, 'p20_dgcr_branches': 0, 'p20_hrcs_scale': 0,
    'p20_lrcfb_branches': 0, 'p20_lswr_scale': 0, 'p20_mone_experts': 0,
    'p20_ncea_branches': 0,
    'p21_per_attn': 0, 'p21_per_experts': 0,
    'p22_attn_moe_route': 'none',
    'p23_std_moe_experts': 0,
    'p34_ffn_last_depth': 1, 'p34_ffn_no_ffn_replacement': 'none',
    'p34_ffn_schedule': '',
    'p36_swiglu_ffn': 0,
}

# Read only inside a branch that one of the gates above has already opened, so their
# value cannot affect a thin block while every gate is off. Listed explicitly rather
# than ignored, so that test_mol.py's coverage check can prove every knob gpt.Block
# reads is either gated or a gate.
_BLOCK_GATED_PARAMS = frozenset({
    'p20_dgcr_aux_weight',
    'p20_lrcfb_learned', 'p20_lrcfb_narrow', 'p20_lrcfb_topk',
    'p20_lswr_planes',
    'p20_mone_frozen', 'p20_mone_narrow', 'p20_mone_topk',
    'p20_ncea_eps',
    'p21_per_learned', 'p21_per_topk',
    'p23_quantile_route', 'p23_std_moe_aux_weight', 'p23_std_moe_topk',
})


def _block_config_knobs():
    """Every config field Block, CausalSelfAttention and MLP branch on."""
    return tuple(sorted(set(_BLOCK_GATES) | _BLOCK_GATED_PARAMS))


def make_thin_config(config):
    """Clone `config` down to d_thin so gpt.Block builds a genuine narrow block.

    MoL specifies "a complete transformer block at reduced dimensionality". The
    fairest reading of that is literally this repo's dense block, so MoL's thin
    block and our dense arm share an implementation and no quality difference can
    be blamed on two different block definitions.

    d_head is pinned to 64 at every width, which is their stated rule (section 2.2:
    d_model=1024 uses 16 heads, d_thin=256 uses 4 heads) and happens to be MST's G1.
    """
    d_thin = int(config.mol_thin_dim)
    head_dim = int(getattr(config, 'mol_head_dim', 64)) or 64
    assert d_thin % head_dim == 0, (
        f"mol_thin_dim ({d_thin}) must be divisible by the head dim ({head_dim}); "
        f"MoL pins d_head across all block widths")
    n_head = d_thin // head_dim

    bad = []
    for name, off in _BLOCK_GATES.items():
        if name in _SHADOW_OVERRIDDEN or not hasattr(config, name):
            continue
        got = getattr(config, name)
        if isinstance(off, str):
            enabled = str(got).strip() != off
        else:
            enabled = bool(got) and got != off
        if enabled:
            bad.append(f"{name}={got!r} (off={off!r})")
    assert not bad, (
        "MoL builds its thin blocks from gpt.Block via a shadow config, which "
        "inherits every research flag. These are ENABLED, so a thin block would no "
        "longer match the dense baseline and the comparison would be meaningless: "
        + ", ".join(bad))

    return replace(
        config,
        n_embd=d_thin,
        n_head=n_head,
        n_kv_head=n_head,
        p34_ffn_mult=float(config.mol_ffn_mult),
        use_mol=False, use_mst=False, use_eet=False,
    )


class ThinBlock(nn.Module):
    """Eq (1): W_up . ( Block_thin(W_down . x) - W_down . x ).

    The subtraction strips the block's own inner residual, so a ThinBlock emits
    only the delta its block computed. gpt.Block already adds `x` internally at
    both the attention and the MLP branch, so `Block(h) - h` is exactly
    attn_out + mlp_out with no other change.
    """

    def __init__(self, config, thin_config, layer_idx):
        super().__init__()
        D = config.n_embd
        d = thin_config.n_embd
        self.w_down = Linear(D, d, bias=False)
        self.w_up = Linear(d, D, bias=False)
        self.block = Block(thin_config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None):
        h = self.w_down(x)
        out = self.block(h, ve, cos_sin, window_size, kv_cache,
                         token_active=token_active)
        return self.w_up(out - h)


class SplitStage(nn.Module):
    """Eq (2), extended with the shared blocks of section 3.2.

    S shared blocks run on every token with unrestricted attention. The remaining
    N-S routed blocks are selected top-k per token by a softmax router and attend
    only over the tokens routed to them, which is the "dense restricted attention"
    their section 2.3 defines sparse dispatch to be equivalent to.
    """

    def __init__(self, config, thin_config, layer_idx):
        super().__init__()
        self.n_blocks = int(config.mol_n_blocks)
        self.n_shared = int(config.mol_n_shared)
        self.topk = int(config.mol_topk)
        self.n_routed = self.n_blocks - self.n_shared
        assert self.n_shared >= 0 and self.n_routed >= 0, \
            f"mol_n_shared ({self.n_shared}) must not exceed mol_n_blocks ({self.n_blocks})"
        assert 1 <= self.topk <= max(1, self.n_routed) or self.n_routed == 0, \
            f"mol_topk ({self.topk}) must be in [1, n_routed={self.n_routed}]"

        self.blocks = nn.ModuleList([
            ThinBlock(config, thin_config, layer_idx) for _ in range(self.n_blocks)
        ])
        # Block is constructed with the STAGE index because resolve_ffn_schedule indexes
        # a list of length n_layer with it. But CausalSelfAttention reuses layer_idx as
        # its KV-cache slot, so leaving it would make all n_blocks blocks of a stage
        # alias the same cache. Retarget the slot only.
        for j, tb in enumerate(self.blocks):
            tb.block.attn.layer_idx = layer_idx * self.n_blocks + j
        # Router covers the routed blocks only. Their S+KofN notation is explicit
        # that 1+3of15 "selects 3 from 14 routed blocks", so the shared block is
        # not a router candidate.
        self.router = Linear(config.n_embd, self.n_routed, bias=False) \
            if self.n_routed > 0 else None
        self.aux_weight = float(config.mol_router_aux)
        self.per_block_ve = bool(getattr(config, 'mol_per_block_ve', 0))
        self.d_thin = int(config.mol_thin_dim)
        self.dispatch = bool(getattr(config, 'mol_dispatch', 1))
        self.capacity_factor = float(getattr(config, 'mol_capacity_factor', 1.0))
        self._last_load = None      # (n_routed,) fraction of tokens per routed block
        self._last_drop = None

    def _route(self, x):
        """Softmax scores, top-k mask, and the CV^2 balance loss.

        Returns (weights, mask) both (B, T, n_routed); weights are zero off the
        top-k so the caller can multiply without gathering.
        """
        logits = self.router(norm(x))
        probs = F.softmax(logits.float(), dim=-1)
        topk_idx = probs.topk(self.topk, dim=-1).indices
        mask = torch.zeros_like(probs).scatter_(-1, topk_idx, 1.0)
        weights = probs * mask

        # Section 2.2: "a coefficient-of-variation (CV^2) loss on per-block routing
        # weights, weighted by alpha=0.05, following standard MoE practice". Shazeer's
        # importance loss: CV^2 of the per-block summed routing weight.
        importance = probs.sum(dim=(0, 1))
        cv2 = importance.var(unbiased=False) / importance.mean().pow(2).clamp_min(1e-12)
        self._last_load = mask.mean(dim=(0, 1)).detach()
        return weights.to(x.dtype), mask.to(torch.bool), self.aux_weight * cv2

    def _capacity(self, T):
        """Per-block token capacity, their §2.3 fixed-size dispatch buffer."""
        return max(1, min(T, int(math.ceil(T * self.topk / self.n_routed
                                           * self.capacity_factor))))

    def _routed_dispatched(self, x, ve, cos_sin, kv_cache, weights, mask):
        """§2.3 sparse dispatch: gather each block's tokens, run, scatter back.

        This is the DEFAULT path, and not only for speed. The masked alternative
        builds a (B, 1, T, T) attention mask per routed block, which at T=2048 with
        14 blocks over 8 layers is 112 masked SDPA calls per forward, each
        materialising ~134MB. It is unrunnable at real sequence lengths and it made
        torch.compile hang on the 120-block graph.

        Gathering in POSITION ORDER is what makes plain causal attention over the
        compact buffer exactly equal to restricted attention over the full sequence:
        token t attends to earlier tokens that also chose this block, and to nothing
        else. No mask is needed at all. Selection takes the first K tokens that chose
        the block rather than the K highest-scoring, which keeps it causal (the same
        reason MST's _ffn_dispatched orders by position).
        """
        B, T, D = x.shape
        K = self._capacity(T)
        dev = x.device

        # (B, n_routed, T) -> earlier positions win the topk, so selection is causal.
        m = mask.permute(0, 2, 1).to(torch.float32)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        top_val, top_idx = (m * (T - pos)).topk(K, dim=-1)
        order = top_idx.argsort(dim=-1)
        top_idx = top_idx.gather(-1, order)              # ascending position
        keep = top_val.gather(-1, order) > 0             # padding slots are dropped
        # Kept as a tensor and never compared or cast here: `float(...)` and `m.sum() > 0`
        # both force a device sync, which torch.compile reports as a graph break and
        # which would land in the middle of the routed-block loop. compute_diagnostics()
        # does the conversion, outside the compiled region.
        if not torch.compiler.is_compiling():
            self._last_drop = (m.sum(-1).clamp(max=K).sum()
                               / m.sum().clamp_min(1.0)).detach()

        cos, sin = cos_sin
        hd = cos.shape[-1]
        # Rotary must use the ORIGINAL positions, not the compact ones. Gathered ONCE
        # for every routed block rather than inside the loop: the previous form was
        # cos.expand(B,...).gather(...) per block, which materialised a (B,K,1,hd)
        # tensor 2*n_routed times per layer (224 kernels per forward at 1+3of15 x L=8).
        cos_g = cos[0, :, 0, :][top_idx]          # (B, n_routed, K, hd)
        sin_g = sin[0, :, 0, :][top_idx]
        acc = torch.zeros_like(x)
        bidx = torch.arange(B, device=dev).unsqueeze(1)                # (B, 1)
        for i in range(self.n_routed):
            idx = top_idx[:, i]                                        # (B, K)
            gx = idx.unsqueeze(-1).expand(-1, -1, D)
            xi = x.gather(1, gx)                                       # (B, K, D)
            cs = (cos_g[:, i].unsqueeze(2), sin_g[:, i].unsqueeze(2))
            ve_i = self._ve_for(ve, self.n_shared + i)
            vi = None if ve_i is None else ve_i.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, ve_i.shape[-1])).contiguous()
            yi = self.blocks[self.n_shared + i](xi, vi, cs, (-1, 0), kv_cache)
            wi = weights[..., i].gather(1, idx) * keep[:, i].to(weights.dtype)
            # index_put_ with a (B,K) index beats scatter_add_ with a (B,K,D) one by
            # ~1.4x measured; scatter is the dominant cost of the dispatch path.
            # Equivalent here because topk returns distinct indices within a block, and
            # duplicated padding slots are zeroed by `keep` so they accumulate zero.
            acc.index_put_((bidx, idx), yi * wi.unsqueeze(-1), accumulate=True)
        return acc

    def _ve_for(self, ve, j):
        """Block j's slice of the value-embedding table.

        With mol_per_block_ve the table is (n_blocks * d_thin) wide and each block
        reads its own slice, so every block sees a different vector. Without it the
        table is d_thin wide and every block sees the same one.
        """
        if ve is None or not self.per_block_ve:
            return ve
        return ve[..., j * self.d_thin:(j + 1) * self.d_thin]

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        out = x
        # Shared blocks: always active, every token, unrestricted attention.
        for j in range(self.n_shared):
            out = out + self.blocks[j](x, self._ve_for(ve, j), cos_sin, window_size,
                                       kv_cache)

        if self.n_routed == 0:
            return out, x.new_zeros(())

        weights, mask, aux = self._route(x)
        if self.dispatch:
            acc = self._routed_dispatched(x, ve, cos_sin, kv_cache, weights, mask)
        else:
            acc = torch.zeros_like(x)
            for i in range(self.n_routed):
                # Reference path. Correct but quadratic in T and mask-heavy; kept so
                # the dispatched path has something to be checked against.
                y = self.blocks[self.n_shared + i](
                    x, self._ve_for(ve, self.n_shared + i), cos_sin, window_size,
                    kv_cache, token_active=mask[..., i])
                acc = acc + weights[..., i:i + 1] * y
        return out + acc / self.topk, aux


class MoL(nn.Module):
    """MoL model. Interface-compatible with GPT and MST."""

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        D = config.n_embd
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1)
                             // pad_vocab_size_to) * pad_vocab_size_to
        self._padded_vocab_size = padded_vocab_size

        thin_config = make_thin_config(config)
        self.thin_config = thin_config
        self.n_blocks = int(config.mol_n_blocks)
        self.d_thin = thin_config.n_embd
        self.thin_n_head = thin_config.n_head
        self.head_dim = self.d_thin // thin_config.n_head

        self.window_sizes = self._compute_window_sizes(config)

        self.wte = nn.Embedding(padded_vocab_size, D)
        self.stages = nn.ModuleList([
            SplitStage(config, thin_config, i) for i in range(config.n_layer)
        ])
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Value embeddings at the thin width, so a thin block's attention can read
        # them exactly as the dense block does at full width.
        # Value embeddings. MoL's paper has none (it is a nanochat/modded-nanogpt
        # component), but our dense baseline and MST both carry them, so MoL must be
        # able to use them or the comparison is rigged against it.
        #
        # mol_per_block_ve=0: one d_thin-wide table, every block reads the same vector
        #   and differentiates only through its own gate. This is the MST-*plain*
        #   equivalent.
        # mol_per_block_ve=1: an (n_blocks * d_thin)-wide table, block j reads slice j.
        #   This is the G3 equivalent, and G3 is worth 0.0059 bpb to MST at L=16, so
        #   without this option MoL is handicapped by exactly the component we just
        #   measured to matter.
        #
        # The two are NOT equal in cost, and that asymmetry is a real architectural
        # consequence rather than unfairness: MST's per-stream table is N*d = D wide,
        # while MoL's per-block table is n_blocks * d_thin, which at 1+3of15 is 3.75x D.
        # Run both and report MoL's better arm.
        self.per_block_ve = bool(getattr(config, 'mol_per_block_ve', 0))
        ve_dim = self.n_blocks * self.d_thin if self.per_block_ve else self.d_thin
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, ve_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.lm_head = Linear(D, padded_vocab_size, bias=False)

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self._last_aux_loss = None
        print0(f"[MoL] {config.mol_n_shared}+{config.mol_topk}of{config.mol_n_blocks} "
               f"d_thin={self.d_thin} ({self.thin_n_head} heads x {self.head_dim}) "
               f"ffn_mult={config.mol_ffn_mult} routed_attn={config.mol_routed_attn}")

    # ---- scaffolding shared with GPT/MST ------------------------------------
    def _compute_window_sizes(self, config):
        """Per-layer sliding window, matching base GPT's SSSSL pattern."""
        n = config.n_layer
        return [(-1, 0) if (i + 1) % 5 == 0 or i == n - 1 else (1024, 0) for i in range(n)]

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return (cos.to(COMPUTE_DTYPE)[None, :, None, :],
                sin.to(COMPUTE_DTYPE)[None, :, None, :])

    def init_weights(self):
        self.wte.to(dtype=torch.float32)
        torch.nn.init.normal_(self.wte.weight, std=1.0)
        self.lm_head.to(dtype=torch.float32)
        torch.nn.init.normal_(self.lm_head.weight, std=0.001)
        for emb in self.value_embeds.values():
            emb.to(dtype=torch.float32)
            torch.nn.init.normal_(emb.weight, std=1.0)

        D = self.config.n_embd
        for stage in self.stages:
            if stage.router is not None:
                torch.nn.init.uniform_(stage.router.weight, -D ** -0.5, D ** -0.5)
            for tb in stage.blocks:
                # Structured layers want per-factor init (Qiu et al. 2024). Down is a
                # fan-in-scaled projection; up is zero so a fresh stage is exactly the
                # identity, matching how MST zero-inits its residual branches.
                torch.nn.init.uniform_(tb.w_down.weight, -D ** -0.5, D ** -0.5)
                torch.nn.init.zeros_(tb.w_up.weight)
                for p in tb.block.parameters():
                    if p.ndim >= 2:
                        fan_out, fan_in = p.shape[0], p.shape[1]
                        torch.nn.init.uniform_(p, -fan_in ** -0.5, fan_in ** -0.5)
                    else:
                        torch.nn.init.zeros_(p)

        self.resid_lambdas.data.fill_(1.0)
        self.x0_lambdas.data.fill_(0.1)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    @property
    def transformer(self):
        return nn.ModuleDict({"h": self.stages, "wte": self.wte})

    # Generation with a KV cache is NOT implemented for MoL, and it is not a small
    # gap. Routed blocks use restricted attention, so each block's cache must hold
    # only the tokens routed to that block, at their original positions, and a new
    # token joins some blocks' caches and not others. Letting every block attend over
    # the full cache would run a different model from the trained one and silently
    # produce wrong samples, which is worse than not sampling. base_train checks this
    # flag and skips its periodic sample step.
    supports_kv_cache_generation = False

    @property
    def kv_cache_config(self):
        """KVCache constructor kwargs. Consumed as **model.kv_cache_config, so this
        has to be a property, and the key names are KVCache's, not GPTConfig's.

        One cache slot per thin block per stage, mirroring MST's n_layer * N_subs."""
        return {
            "num_heads": self.thin_n_head,
            "head_dim": self.head_dim,
            "v_head_dim": self.head_dim,
            "num_layers": self.config.n_layer * self.n_blocks,
        }

    @property
    def max_seq_len(self):
        return self.config.sequence_len

    # ---- forward -------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T
        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            cos, sin = self._precompute_rotary_embeddings(new_len, self.head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        x = self.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x

        aux = x.new_zeros(())
        for i, stage in enumerate(self.stages):
            ve = self.value_embeds[str(i)](idx).to(COMPUTE_DTYPE) \
                if str(i) in self.value_embeds else None
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x, stage_aux = stage(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            aux = aux + stage_aux
        self._last_aux_loss = aux.detach()

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits.float() / 30.0)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=loss_reduction)
        if self.training and self.config.mol_router_aux > 0:
            loss = loss + aux.to(loss.dtype)
        return loss

    # ---- accounting ----------------------------------------------------------
    def _param_groups(self):
        """(shared_block_params, routed_block_params, projection_params) per stage.

        Projection params are counted separately because MoL's plumbing fraction is
        the whole point of the comparison; see scripts/p09_projection_overhead.py.
        """
        shared = routed = proj = 0
        for stage in self.stages:
            for j, tb in enumerate(stage.blocks):
                p_proj = tb.w_down.weight.numel() + tb.w_up.weight.numel()
                p_blk = sum(p.numel() for p in tb.block.parameters())
                proj += p_proj
                if j < stage.n_shared:
                    shared += p_proj + p_blk
                else:
                    routed += p_proj + p_blk
        return shared, routed, proj

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        ve = sum(p.numel() for n, p in self.named_parameters() if 'value_embed' in n)
        matrices = total - self.wte.weight.numel() - ve \
            - self.resid_lambdas.numel() - self.x0_lambdas.numel() \
            - self.lm_head.weight.numel()
        shared, routed, proj = self._param_groups()
        n_routed = max(1, self.stages[0].n_routed)
        active_frac = self.stages[0].topk / n_routed if self.stages[0].n_routed else 0.0
        return {
            'total': total,
            'transformer_matrices': matrices,
            'lm_head': self.lm_head.weight.numel(),
            'embedding': self.wte.weight.numel() + ve,
            'active': total - int(routed * (1.0 - active_frac)),
            'projections': proj,
            'projection_fraction': proj / max(1, shared + routed),
        }

    def estimate_flops(self):
        """Matches MST/GPT convention exactly: 6*(matmul params) + attention.

        Two things are specific to MoL and easy to get wrong in its favour or
        against it:

          * routed blocks attend only over the tokens routed to them, roughly
            T*k/n_routed of the sequence, so their attention term uses that reduced
            length rather than T. Omitting this would understate their architecture.
          * active_flops discounts the n_routed-k unselected blocks. On the masked
            path that is a claim about what a sparse kernel *could* skip, which is
            the exact class of bug that showed up in mst.py's Monarch accounting, so
            it is asserted in tests/test_mol.py rather than trusted.
        """
        nparams = sum(p.numel() for p in self.parameters())
        ve_numel = sum(p.numel() for n, p in self.named_parameters() if 'value_embed' in n)
        nparams_exclude = (self.wte.weight.numel() + ve_numel
                           + self.resid_lambdas.numel() + self.x0_lambdas.numel())
        matmul_flops = 6 * (nparams - nparams_exclude)

        t = self.config.sequence_len
        nh, hd = self.thin_n_head, self.head_dim
        s0 = self.stages[0]
        frac = (s0.topk / s0.n_routed) if s0.n_routed else 0.0

        attn_total = attn_active = 0
        for i, stage in enumerate(self.stages):
            window = self.window_sizes[i][0]
            eff = t if window < 0 else min(window, t)
            per_block = 12 * nh * hd * eff
            attn_total += stage.n_blocks * per_block
            attn_active += stage.n_shared * per_block
            # A routed block runs on a `frac` share of tokens, and each of those
            # attends over a `frac` share of the keys: quadratic saving, not linear.
            attn_active += stage.n_routed * per_block * frac * frac

        total_flops = matmul_flops + attn_total

        _, routed_params, _ = self._param_groups()
        inactive = int(routed_params * (1.0 - frac))
        active_flops = total_flops - 6 * inactive - (attn_total - attn_active)
        active_params = nparams - inactive
        return total_flops, int(active_flops), active_params

    def compute_diagnostics(self):
        out = {}
        for i, stage in enumerate(self.stages):
            if stage._last_load is None:
                continue
            load = stage._last_load.float()
            ideal = stage.topk / max(1, stage.n_routed)
            out[f'mol_load_min_L{i}'] = float(load.min() / ideal)
            out[f'mol_load_max_L{i}'] = float(load.max() / ideal)
            p = load / load.sum().clamp_min(1e-9)
            out[f'mol_entropy_L{i}'] = float(-(p * p.clamp_min(1e-9).log()).sum())
        if self._last_aux_loss is not None:
            out['mol_aux'] = float(self._last_aux_loss)
        drops = [s._last_drop for s in self.stages if s._last_drop is not None]
        if drops:
            out['mol_keep_frac'] = float(min(d.min() for d in drops))
        return out

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2,
                        matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95),
                        scalar_lr=0.5, disable_mu_p=False, mu_p_scale_override=-1.0,
                        gate_lr_scale=0.3, **_unused):
        """Deliberately identical in structure to MST.setup_optimizer.

        MoL's own recipe is AdamW at 3e-4 cosine (their §4.2), but the head-to-head
        has to hold the optimizer fixed or any bpb difference is confounded. So MoL
        gets exactly the treatment MST gets: the same muP scaling, the same group
        kinds and LRs, and Muon on every 2D matrix.

        Note that MST's block-diagonal Muon has no MoL analogue to add, and needs
        none: MST stacks its N streams into one (N*out, in) tensor and must
        orthogonalise per block by hand, whereas MoL's thin blocks are already
        separate parameters, so Muon is per-block for free. `mol_block_lr_scale`
        exists so the per-block LR question can be ablated rather than assumed;
        their paper specifies no such scaling, so it defaults to 1.0.
        """
        from nanochat.optim import MuonAdamW, DistMuonAdamW
        from nanochat.common import get_dist_info

        _ddp, _rank, _local_rank, world_size = get_dist_info()

        embed_params = list(self.wte.parameters())
        unembed_params = list(self.lm_head.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        covered = {id(p) for p in embed_params + unembed_params + value_embeds_params
                   + resid_params + x0_params}

        matrix_params, scalar_params = [], []
        for p in self.parameters():
            if id(p) in covered or not p.requires_grad:
                continue
            (matrix_params if p.ndim >= 2 else scalar_params).append(p)

        model_dim = self.config.n_embd
        if mu_p_scale_override > 0.0:
            s = mu_p_scale_override
        elif disable_mu_p:
            s = 1.0
        else:
            s = (model_dim / 768) ** -0.5
        print0(f"[MoL] muP LR scaling 1/sqrt({model_dim}/768) = {s:.6f}")

        block_scale = float(getattr(self.config, 'mol_block_lr_scale', 1.0))
        param_groups = [
            dict(kind='adamw', params=unembed_params, lr=unembedding_lr * s,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embed_params, lr=embedding_lr * s,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * s,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=scalar_params, lr=embedding_lr * s,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon stacks a group's tensors, so a group must be shape-homogeneous. Same
        # bucketing MST does for its non-stacked matrices. MoL needs it more: W_down
        # is (d, D) and W_up is (D, d), which are different shapes at every stage.
        for shape in sorted({tuple(p.shape) for p in matrix_params}):
            param_groups.append(dict(
                kind='muon', params=[p for p in matrix_params if tuple(p.shape) == shape],
                lr=matrix_lr * block_scale, momentum=0.95, ns_steps=5, beta2=0.95,
                weight_decay=weight_decay,
            ))
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        cls = DistMuonAdamW if world_size > 1 else MuonAdamW
        # No **kwargs forwarding. base_train passes architecture-specific extras such
        # as gate_lr_scale (RemixedLinear) that MuonAdamW does not accept; MST absorbs
        # them in its signature the same way and never forwards them.
        optimizer = cls(param_groups)
        # base_train's LR schedule reads group['initial_lr'] (base_train.py:1938, 2354),
        # so every architecture has to stamp it; MST does the same at the end of its
        # setup_optimizer.
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        return optimizer

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        rng = torch.Generator(device=self.get_device()).manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.get_device())
        for _ in range(max_tokens):
            logits = self(ids[:, -self.config.sequence_len:])[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=rng)
            ids = torch.cat([ids, nxt], dim=1)
            yield nxt.item()
