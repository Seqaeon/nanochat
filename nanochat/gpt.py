"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (optionally combined with learned absolute positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    """Configuration for :class:`GPT` and optional research branches.

    Router defaults are explicit to keep notebook experiments reproducible:
    ``router_context_window=-1`` (full context), ``router_causal=True``,
    ``router_num_heads=4``, ``router_num_queries=8``, ``router_n_layers=2``,
    and ``router_use_vocab_prior=False``.
    """
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

    # Research branches
    use_moe: bool = False
    use_perm: bool = False
    moe_num_experts: int = 8
    moe_router_dim: int = 64
    moe_embed_dim: int = 64
    dropout: float = 0.0
    use_remix_linear: bool = False
    remix_context_dim: int = 64
    # If > 0, overrides remix_context_dim with n_embd // remix_context_dim_ratio.
    # Recommended: 8 (gives 96-dim context for 768-dim model, 128-dim for 1024-dim, etc.)
    # Set to 0 to use the fixed remix_context_dim value instead.
    remix_context_dim_ratio: int = 6
    remix_basis_size: int = 64
    # Rank of the low-rank output gate. Smaller r = more stable at long T.
    # r=8 works well from T=64 to T=2048. Increase to 16 only if needed.
    remix_output_gate_rank: int = 16
    remixed_linear_kwargs: dict | None = None
    use_pos_embed: bool = False
    moe_use_abs_pos_embed: bool = False

    # Fix 1A: per-layer context updaters for remix_linear (zero-init deltas applied at each block)
    use_layer_context: bool = True
    # Fix 1B: auto-scale basis_size to max(remix_basis_size, in_features // 4) to prevent rank bottleneck
    scale_basis_size: bool = True
    # Fix 1D: PermutationMoE expert mode — 'full' (original D×D), 'low_rank', or 'factored'
    perm_expert_mode: str = 'low_rank'
    # Fix 1D: rank for low_rank mode (rank = max(8, base_embed_dim // perm_rank_ratio))
    #          or block_size for factored mode (num_blocks = base_embed_dim // perm_rank)
    perm_rank: int = 16

    # Shared context-aware router defaults used by embedding/context branches
    router_context_window: int = -1  # -1 = full context. For sequence_len > 512, set this to
                                      # 256-512 to prevent the GlobalContextManager's attention
                                      # from becoming a bottleneck (O(T²)) and diluting the
                                      # context signal. The per-layer residual-stream updaters
                                      # now carry the long-range signal instead.
    router_causal: bool = True
    router_num_heads: int = 4
    router_num_queries: int = 16
    router_n_layers: int = 2
    router_use_vocab_prior: bool = False

    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSSL"


# Used by notebooks to validate kwargs passed to GPTConfig.
RESEARCH_ALLOWED_KEYS = {
    "use_moe", "use_perm",
    "moe_num_experts", "moe_router_dim", "moe_embed_dim", "dropout",
    "use_remix_linear", "remix_context_dim", "remix_context_dim_ratio", "remix_basis_size", "remix_output_gate_rank", "remixed_linear_kwargs",
    "use_pos_embed", "moe_use_abs_pos_embed",
    "router_context_window", "router_causal", "router_num_heads",
    "router_num_queries", "router_n_layers", "router_use_vocab_prior",
    # Fix 1A
    "use_layer_context",
    # Fix 1B
    "scale_basis_size",
    # Fix 1D
    "perm_expert_mode", "perm_rank",
}


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)


class ImprovedContextAwareRouter(nn.Module):
    """Context-aware router used by research embedding branches.

    Defaults: ``context_window=-1``, ``causal=True``, ``num_heads=4``,
    ``num_queries=8``, ``n_layers=1``, ``use_vocab_prior=False``.
    """
    def __init__(
        self,
        vocab_size,
        num_experts,
        router_dim,
        full_embed_dim,
        context_window=-1,
        causal=True,
        num_heads=4,
        num_queries=8,
        n_layers=1,
        use_vocab_prior=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.context_window = context_window
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = router_dim // num_heads
        self.n_layers = n_layers
        self.use_vocab_prior = use_vocab_prior

        self.embed_proj = Linear(full_embed_dim, router_dim, bias=True)
        self.qkv_proj = Linear(router_dim, 3 * router_dim, bias=False)
        self.out_proj = Linear(router_dim, router_dim, bias=True)
        self.ln = nn.LayerNorm(router_dim)
        self.routing_queries = nn.Parameter(torch.randn(num_queries, router_dim))
        self.temperature_predictor = Linear(router_dim, 1, bias=True)
        self.expert_proj = Linear(router_dim, num_experts, bias=True)
        self.cross_expert_proj = Linear(router_dim, num_experts, bias=True)
        self.alpha_gate = Linear(router_dim, 1, bias=True)
        if use_vocab_prior:
            self.vocab_routing_bias = nn.Embedding(vocab_size, num_experts)

    def _create_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        if self.context_window == -1:
            return pos_diff > 0 if self.causal else torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if self.causal:
            return (pos_diff > 0) | (pos_diff < -self.context_window)
        window_half = self.context_window // 2
        return torch.abs(pos_diff) > window_half

    def _multi_head_attention(self, x):
        # Fix 1C: SDPA + QK-norm. Mirrors the main model for stability at scale.
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # QK-norm stabilises attention logit scale at larger router_dims
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Use fused SDPA when causal+full-context (most common case)
        is_full_causal = self.causal and (self.context_window == -1 or self.context_window >= seq_len)
        if is_full_causal:
            context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Windowed or non-causal: fall back to explicit masked attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = self._create_mask(seq_len, x.device).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
            context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.router_dim)
        return self.out_proj(context) + x

    def _cross_attention(self, queries, context):
        """Causal cross-attention from learned routing queries into the token context.

        Position t can only attend to context[0..t] (causal prefix aggregation).

        Uses SDPA (Flash Attention when available) rather than an explicit
        (B, T, Q, T) attention matrix.  The old explicit-matmul approach triggered
        a Triton pointwise kernel with XBLOCK=8192 at T=2048, exceeding Triton's
        4096 limit and crashing torch.compile.

        Loops over the Q routing queries (Q is small, typically 8-16).
        Each SDPA call uses shape (B, 1, T, D): well within kernel limits and
        handled by Flash Attention's O(T) memory algorithm.

        Returns (B, T, D): causally-masked per-position summary.
        """
        batch_size, seq_len, dim = context.shape
        queries = queries.to(dtype=context.dtype)  # (Q, D)

        # Keys/Values: (B, 1, T, D) — same context for every routing query.
        # unsqueeze(1) on a contiguous (B,T,D) tensor is itself contiguous.
        k = context.unsqueeze(1)  # (B, 1, T, D)

        outputs = []
        for q_vec in queries.unbind(0):  # iterate over Q routing queries
            # q: (B, 1, T, D) — same routing query at every target position.
            # .contiguous() lets SDPA pick Flash Attention when available.
            q = q_vec.view(1, 1, 1, dim).expand(batch_size, 1, seq_len, dim).contiguous()
            # is_causal=True: target position t attends only to keys 0..t.
            out = F.scaled_dot_product_attention(q, k, k, is_causal=True)  # (B, 1, T, D)
            outputs.append(out)

        # (B, Q, T, D) → mean across Q routing queries → (B, T, D)
        return torch.cat(outputs, dim=1).mean(dim=1)

    def forward(self, full_embeds, input_ids=None):
        dtype = full_embeds.dtype
        x = self.embed_proj(full_embeds)
        for _ in range(self.n_layers):
            x = self._multi_head_attention(x)
        # Fix: ensure x matches LN's parameter dtype to avoid RuntimeError in generation/inference
        x = self.ln(x.to(dtype=self.ln.weight.dtype)).to(dtype=dtype)
        self_attn_logits = self.expert_proj(x)
        cross_attn_logits = self.cross_expert_proj(self._cross_attention(self.routing_queries, x))
        alpha = torch.sigmoid(self.alpha_gate(x))
        expert_logits = alpha * self_attn_logits + (1 - alpha) * cross_attn_logits
        if self.use_vocab_prior and input_ids is not None:
            expert_logits = expert_logits + self.vocab_routing_bias(input_ids)
        adaptive_temp = torch.sigmoid(self.temperature_predictor(x)) * 2.0 + 0.1
        adaptive_temp = torch.clamp(adaptive_temp, min=1e-6)
        return expert_logits, adaptive_temp, self.expert_proj.weight


class DirectContextualEmbedding(nn.Module):
    """Direct contextual embedding with context-aware routing over K learned expert codes.

    Fix 1E: Routes among ``num_experts`` (K, typically 8) learned direction vectors rather
    than misusing num_experts=dim. Each token's embedding = seed + router-weighted expert code.
    This is parameter-efficient (K×dim extra params) and conceptually correct per the paper.
    """

    def __init__(
        self,
        vocab_size,
        dim,
        num_experts,
        router_dim,
        context_window,
        dropout=0.0,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
    ):
        super().__init__()
        self.seed_embeddings = nn.Embedding(vocab_size, dim)
        # K learned expert code vectors (context-independent, shared across vocab)
        # Small: K × dim params (e.g. 8 × 256 = 2048)
        self.expert_codes = nn.Parameter(torch.zeros(num_experts, dim))
        self.router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=num_experts,   # correctly K, not dim
            router_dim=router_dim,
            full_embed_dim=dim,
            context_window=context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        seeds = self.seed_embeddings(input_ids)                              # (B, T, dim)
        expert_logits, adaptive_temp, _ = self.router(seeds, input_ids)     # (B, T, K)
        expert_weights = F.softmax(expert_logits / adaptive_temp, dim=-1)   # (B, T, K)
        # Context-modulated perturbation: weighted sum of K expert codes
        # Ensure codes match weights/seeds dtype for einsum and addition
        expert_codes = self.expert_codes.to(dtype=seeds.dtype)
        perturbation = torch.einsum('btk,kd->btd', expert_weights.to(dtype=seeds.dtype), expert_codes)  # (B, T, dim)
        output = seeds + perturbation
        return self.out_norm(output), {'expert_weights': expert_weights}


class PermutationMoE(nn.Module):
    """Permutation MoE embedding with configurable expert router defaults.

    Defaults to learned absolute positional embeddings (``moe_use_abs_pos_embed=True``).

    Fix 1D: Supports three expert modes controlled by ``perm_expert_mode``:
    - ``'full'``:     Original D×D selection matrix per expert (O(D²) cost).
    - ``'low_rank'``: D×rank selection weights over learned rank basis vectors (O(D·rank)).
                     rank = max(8, base_embed_dim // perm_rank) where perm_rank acts as divisor.
    - ``'factored'``: Block-diagonal — D split into (D//perm_rank) blocks of perm_rank each.
                     Independent perm_rank×perm_rank permutation per block (O(D·perm_rank)).
    Fix 1H: adaptive_temp is clamped to [0.5, 2.1] to prevent routing collapse from near-zero temps.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        base_embed_dim,
        num_experts=8,
        router_dim=64,
        selection_mode='soft',
        allow_replacement=True,
        dropout=0.0,
        router_context_window=-1,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
        moe_use_abs_pos_embed=True,
        perm_expert_mode='low_rank',
        perm_rank=16,
    ):
        super().__init__()
        self.base_embed_dim = base_embed_dim
        self.num_experts = num_experts
        self.selection_mode = selection_mode
        self.allow_replacement = allow_replacement
        self.perm_expert_mode = perm_expert_mode
        self.embeddings = nn.Embedding(vocab_size, base_embed_dim)
        self.position_embeddings = nn.Embedding(block_size, base_embed_dim) if moe_use_abs_pos_embed else None

        D = base_embed_dim
        if perm_expert_mode == 'full':
            # Original: each expert outputs D×D logits (O(D²) cost)
            self.perm_rank = D
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, D * D, bias=False),
                ) for _ in range(num_experts)
            ])
        elif perm_expert_mode == 'low_rank':
            # Fix 1D low_rank: D×rank selection weights + learned (rank, D) basis per expert
            # rank = max(8, D // perm_rank)  — perm_rank acts as a divisor, e.g. perm_rank=16 ⇒ rank=D//16
            r = max(8, D // perm_rank)
            self.perm_rank = r
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, D * r, bias=False),  # D*rank outputs
                ) for _ in range(num_experts)
            ])
            # Learned basis: (K, rank, D) — K sets of rank D-dimensional basis vectors
            self.perm_basis = nn.ParameterList([
                nn.Parameter(torch.randn(r, D) * (D ** -0.5))
                for _ in range(num_experts)
            ])
        elif perm_expert_mode == 'factored':
            # Fix 1D factored (block-diagonal): block_size=perm_rank, num_blocks=D//perm_rank
            # perm_rank must evenly divide D; if not, round to nearest divisor.
            bs = perm_rank
            while D % bs != 0 and bs > 1:
                bs -= 1
            nb = D // bs
            self.perm_rank = bs   # block_size stored in perm_rank attr
            self.perm_num_blocks = nb
            self.dim_selectors = nn.ModuleList([
                nn.Sequential(
                    Linear(D, router_dim, bias=False),
                    nn.LayerNorm(router_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    Linear(router_dim, nb * bs * bs, bias=False),  # nb independent bs×bs matrices
                ) for _ in range(num_experts)
            ])
        else:
            raise ValueError(f"Unknown perm_expert_mode: {perm_expert_mode!r}. Use 'full', 'low_rank', or 'factored'.")

        self.expert_router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=num_experts,
            router_dim=router_dim,
            full_embed_dim=base_embed_dim,
            context_window=router_context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )
        self.ln = nn.LayerNorm(base_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, input_ids):
        B, T = input_ids.shape
        D = self.base_embed_dim
        embeds = self.embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(T, device=input_ids.device)
            positions = torch.clamp(positions, max=self.position_embeddings.num_embeddings - 1)
            embeds = embeds + self.position_embeddings(positions)

        expert_outputs = []
        for expert_idx in range(self.num_experts):
            raw = self.dim_selectors[expert_idx](embeds)  # shape depends on mode

            if self.perm_expert_mode == 'full':
                selection_logits = raw.view(B, T, D, D).clamp(-30, 30)
                if self.selection_mode == 'hard' and not self.allow_replacement:
                    selection_weights = F.gumbel_softmax(selection_logits, tau=self.temperature, hard=True, dim=-1)
                elif self.selection_mode == 'hard':
                    selection_weights = F.gumbel_softmax(selection_logits, tau=self.temperature, hard=False, dim=-1)
                else:
                    selection_weights = F.softmax(selection_logits / self.temperature, dim=-1)
                selected = torch.einsum('bloi,bli->blo', selection_weights, embeds)

            elif self.perm_expert_mode == 'low_rank':
                r = self.perm_rank
                # (B, T, D, r) selection weights: for each output dim, weights over r basis vectors
                sel_w = F.softmax(raw.view(B, T, D, r).clamp(-30, 30) / self.temperature, dim=-1)
                # Project embeds onto rank-r basis: (B, T, r) scalar projections
                basis = self.perm_basis[expert_idx]  # (r, D)
                basis_proj = torch.einsum('bti,ri->btr', embeds, basis)  # (B, T, r)
                # selected[b,t,d] = sum_r( sel_w[b,t,d,r] * basis_proj[b,t,r] )
                selected = torch.einsum('btdr,btr->btd', sel_w, basis_proj)  # (B, T, D)

            else:  # factored
                bs = self.perm_rank
                nb = self.perm_num_blocks
                selection_logits = raw.view(B, T, nb, bs, bs).clamp(-30, 30)
                selection_weights = F.softmax(selection_logits / self.temperature, dim=-1)  # (B, T, nb, bs, bs)
                embeds_blocked = embeds.view(B, T, nb, bs)
                selected = torch.einsum('btnoi,btni->btno', selection_weights, embeds_blocked).reshape(B, T, D)

            expert_outputs.append(selected)

        expert_outputs = torch.stack(expert_outputs, dim=2)  # (B, T, K, D)
        expert_logits, adaptive_temp, _ = self.expert_router(embeds, input_ids)
        # Fix 1H: clamp adaptive_temp from below to prevent near-zero softmax collapse
        adaptive_temp = adaptive_temp.clamp(min=0.5)
        expert_weights = F.softmax(expert_logits / (self.temperature * adaptive_temp), dim=-1)
        expert_weights = self.dropout(expert_weights)
        # expert_outputs is (B, T, K, D), expert_weights is (B, T, K)
        # Match dtypes for final summation
        output = (expert_weights.to(dtype=expert_outputs.dtype).unsqueeze(-1) * expert_outputs).sum(dim=2)
        return self.ln(output), {'expert_weights': expert_weights}


class GlobalContextManager(nn.Module):
    """Global context manager built on the context-aware router.

    Defaults: ``router_num_heads=4``, ``router_num_queries=8``,
    ``router_n_layers=2``, ``router_use_vocab_prior=False``.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        router_dim=64,
        context_window=128,
        router_causal=True,
        router_num_heads=4,
        router_num_queries=8,
        router_n_layers=2,
        router_use_vocab_prior=False,
    ):
        super().__init__()
        self.router = ImprovedContextAwareRouter(
            vocab_size=vocab_size,
            num_experts=router_dim,
            router_dim=router_dim,
            full_embed_dim=d_model,
            context_window=context_window,
            causal=router_causal,
            num_heads=router_num_heads,
            num_queries=router_num_queries,
            n_layers=router_n_layers,
            use_vocab_prior=router_use_vocab_prior,
        )

    def forward(self, x_embeds, input_ids=None):
        context_logits, _, _ = self.router(x_embeds, input_ids)
        return F.layer_norm(context_logits, context_logits.shape[-1:])


class RemixedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_dim, basis_size=64, remixed_linear_kwargs=None, scale_basis=True):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        # Fix 1B: prevent rank bottleneck — but basis_size should compress relative to
        # the *smaller* of in/out (template_mixing is out_features x basis_size, so using
        # in_features // 4 when in_features >> out_features makes the layer *more* expensive
        # than dense: e.g. c_proj 3072->768 would get basis_size=768 = full-rank output).
        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // 4)
        self.basis_size = basis_size
        self.use_basis_gate = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.use_context = remixed_linear_kwargs.get('use_context', True)

        self.basis = Linear(in_features, basis_size, bias=False)
        self.template_mixing = nn.Parameter(torch.randn(out_features, basis_size))
        self.ln_basis = nn.LayerNorm(basis_size)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if self.use_context:
            # --- Gate network design ---
            #
            # PROBLEM AT LONG SEQUENCES:
            # output_modulator = Linear(ctx_dim, out_features) has ctx_dim × out_features
            # parameters (e.g. 128 × 3072 = 393K for c_fc). At T=2048, positions 0-~100
            # have weak causal context, so their context_state is near-zero. Those positions
            # produce near-random output gates that MULTIPLY the pre_output, corrupting
            # gradients for basis and template_mixing everywhere. The dense baseline has
            # no such corruption — this is why RemixedLinear degrades at long T.
            #
            # FIX 1: Low-rank output gate.
            # Instead of predicting out_features independent gate values per position,
            # predict r << out_features coefficients and apply them to r learned gate
            # basis vectors: gate = sigmoid(coeffs @ gate_basis), where
            #   coeffs  = Linear(ctx_dim, r)        — small, stable to train
            #   gate_basis = Parameter(r, out_features)  — learned gate directions
            # This reduces gate noise by forcing it onto a low-dimensional manifold.
            # Gradient of gate_basis weights is summed over T positions, reducing variance.
            # Gradient of coeffs weights is summed over out_features directions per position,
            # amplifying signal by r.
            #
            # FIX 2: Centered gate activation for output gate.
            # sigmoid starts at 0.88 (with bias=2.0) but the identity operation is 1.0.
            # Using gate = 1 + tanh(scale * coeffs @ gate_basis) centers the gate around
            # 1.0 and allows it to range in [0, 2], providing both attenuation AND
            # amplification. Gradient at init = sech²(0) = 1.0 vs sigmoid'(2.0) ≈ 0.10.
            # 10× better gradient flow for the output gate at initialization.
            basis_hidden = max(context_dim // 2, min(basis_size, context_dim * 2))
            self.basis_modulator = nn.Sequential(
                Linear(context_dim, basis_hidden, bias=True),
                nn.GELU(),
                Linear(basis_hidden, basis_size, bias=True),
            )
            # Low-rank output gate: ctx_dim → r coefficients, r basis vectors → out_features
            # r is a small constant (default 8), making this O(ctx_dim*r + r*out_features)
            # rather than O(ctx_dim*out_features).
            r = remixed_linear_kwargs.get('output_gate_rank', 8)
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.zeros(r, out_features))
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)  # learnable scale, starts small

    def forward(self, x, context_state):
        dtype = x.dtype
        h_basis = self.ln_basis(self.basis(x).to(dtype=self.ln_basis.weight.dtype)).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)

            # Basis gate: sigmoid, works fine since basis_size is small
            gate_basis = torch.sigmoid(self.basis_modulator(ctx)).to(dtype=dtype) if self.use_basis_gate else torch.ones_like(h_basis)

            # Output gate: LOW-RANK + CENTERED ACTIVATION
            # gate = 1 + tanh(scale * coeffs @ gate_basis_vectors)
            # - Starts at 1.0 (identity) instead of sigmoid's 0.5-0.88
            # - Gradient at init = sech²(0)*scale = scale, controllable
            # - Constrained to low-dim manifold, much more stable at long T
            if self.use_output_gate:
                coeffs = self.output_gate_coeffs(ctx)                           # (B, T, r)
                gate_logits = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))  # (B, T, out_features)
                gate_out = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gate_logits)
            else:
                gate_out = None
        else:
            gate_basis = torch.ones_like(h_basis)
            gate_out = None

        h_gated = (h_basis * gate_basis).to(dtype=dtype)
        pre_output = F.linear(h_gated, self.template_mixing.to(dtype=dtype))

        if gate_out is not None:
            pre_output = pre_output * gate_out

        return (pre_output + self.bias.to(dtype=dtype)).to(dtype=dtype)


class RemixedFeedForward(nn.Module):
    """Feedforward path using RemixedLinear in the base MLP framework."""
    def __init__(self, config):
        super().__init__()
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        self.c_fc = RemixedLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)
        self.c_proj = RemixedLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)

    def forward(self, x, context_state):
        x = self.c_fc(x, context_state)
        x = F.relu(x).square()
        x = self.c_proj(x, context_state)
        return x


class RemixedMultiAttention(nn.Module):
    """Attention path using RemixedLinear for Q/K/V/proj while preserving GPT attention logic."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.ve_gate_channels = min(self.n_embd, 32)
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        self.c_q = RemixedLinear(self.n_embd, self.n_head * self.head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)
        self.c_k = RemixedLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)
        self.c_v = RemixedLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)
        self.c_proj = RemixedLinear(self.n_embd, self.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        B, T, C = x.size()
        q = self.c_q(x, context_state).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x, context_state).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x, context_state).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y, context_state)


class RemixedBlock(nn.Module):
    """Base block structure with remixed attention + remixed feedforward."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = RemixedMultiAttention(config, layer_idx)
        self.ffwd = RemixedFeedForward(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, context_state)
        x = x + self.ffwd(norm(x), context_state)
        return x


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(self.n_embd, 32)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        self.use_moe = config.use_moe
        self.use_perm = config.use_perm
        self.moe_num_experts = config.moe_num_experts
        self.moe_router_dim = config.moe_router_dim
        self.moe_embed_dim = config.moe_embed_dim
        self.use_remix_linear = config.use_remix_linear
        # Auto-scale context_dim with model width if ratio is set
        if config.use_remix_linear and getattr(config, 'remix_context_dim_ratio', 0) > 0:
            config.remix_context_dim = max(config.remix_context_dim, config.n_embd // config.remix_context_dim_ratio)
            print0(f"remix_context_dim auto-scaled to {config.remix_context_dim} (n_embd={config.n_embd} // ratio={config.remix_context_dim_ratio})")
        # Auto-cap router_context_window at long sequences.
        # The GlobalContextManager runs O(T²) causal attention.  At T=2048 with the default
        # router_context_window=-1 (full context), this produces a highly diluted context signal
        # AND is very expensive.  Cap it automatically so the residual-stream context_updaters
        # carry the long-range signal instead (they read from norm(x) which is well-conditioned).
        if config.use_remix_linear and config.router_context_window == -1 and config.sequence_len > 512:
            config.router_context_window = 256
            print0(f"router_context_window auto-capped to 256 (sequence_len={config.sequence_len} > 512). "
                   f"Set router_context_window explicitly to override.")
        self.remix_context_dim = config.remix_context_dim
        self.remix_basis_size = config.remix_basis_size
        self.use_pos_embed = config.use_pos_embed
        if config.remixed_linear_kwargs is None:
            self.remixed_linear_kwargs = dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        else:
            self.remixed_linear_kwargs = config.remixed_linear_kwargs
        # Wire output_gate_rank into kwargs so RemixedLinear picks it up
        self.remixed_linear_kwargs['output_gate_rank'] = getattr(config, 'remix_output_gate_rank', 8)
        config.remixed_linear_kwargs = self.remixed_linear_kwargs
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        block_cls = RemixedBlock if self.use_remix_linear else Block
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([block_cls(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        if self.use_pos_embed:
            self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)
        self.embedding_model = None
        if self.use_moe:
            if self.use_perm:
                self.embedding_model = PermutationMoE(
                    vocab_size=padded_vocab_size,
                    block_size=config.sequence_len,
                    base_embed_dim=self.moe_embed_dim,
                    num_experts=self.moe_num_experts,
                    router_dim=self.moe_router_dim,
                    selection_mode='soft',
                    allow_replacement=True,
                    dropout=config.dropout,
                    router_context_window=config.router_context_window,
                    router_causal=config.router_causal,
                    router_num_heads=config.router_num_heads,
                    router_num_queries=config.router_num_queries,
                    router_n_layers=config.router_n_layers,
                    router_use_vocab_prior=config.router_use_vocab_prior,
                    moe_use_abs_pos_embed=config.moe_use_abs_pos_embed,
                    # Fix 1D: expert mode selection
                    perm_expert_mode=getattr(config, 'perm_expert_mode', 'low_rank'),
                    perm_rank=getattr(config, 'perm_rank', 16),
                )
            else:
                self.embedding_model = DirectContextualEmbedding(
                    vocab_size=padded_vocab_size,
                    dim=self.moe_embed_dim,
                    # Fix 1E: pass correct num_experts (K) and router_dim
                    num_experts=self.moe_num_experts,
                    router_dim=self.moe_router_dim,
                    context_window=config.router_context_window,
                    dropout=config.dropout,
                    router_causal=config.router_causal,
                    router_num_heads=config.router_num_heads,
                    router_num_queries=config.router_num_queries,
                    router_n_layers=config.router_n_layers,
                    router_use_vocab_prior=config.router_use_vocab_prior,
                )
            assert self.moe_embed_dim == config.n_embd, "moe_embed_dim must match n_embd"
        self.context_manager = None
        self.context_updaters = None
        if self.use_remix_linear:
            self.context_manager = GlobalContextManager(
                vocab_size=padded_vocab_size,
                d_model=config.n_embd,
                router_dim=self.remix_context_dim,
                context_window=config.router_context_window,
                router_causal=config.router_causal,
                router_num_heads=config.router_num_heads,
                router_num_queries=config.router_num_queries,
                router_n_layers=config.router_n_layers,
                router_use_vocab_prior=config.router_use_vocab_prior,
            )
            # Fix 1A: per-layer context updaters — zero-init Linears that add a layer-specific
            # delta to the base context, allowing each block to condition on its own residual state.
            if getattr(config, 'use_layer_context', True):
                # Residual-stream-grounded context updaters.
                #
                # Old design: updaters mapped context_dim → context_dim, so the context
                # signal was purely self-referential. At long sequences (2048+) this meant
                # the context could never recover from the GlobalContextManager's diluted
                # initial signal — it just looped on itself getting no new information.
                #
                # New design: updaters project from n_embd → context_dim.  At each layer
                # they read directly from the transformer's residual stream x, grounding
                # the context signal in actual processed representations regardless of
                # sequence length.  This completely bypasses the GlobalContextManager's
                # long-sequence dilution problem.
                #
                # Each updater: Linear(n_embd, ctx_hidden) → GELU → Linear(ctx_hidden, ctx_dim)
                # Second layer is zero-init (in init_weights) so the initial delta is zero
                # and training starts from the static-context baseline.
                ctx_dim = self.remix_context_dim
                ctx_hidden = max(ctx_dim * 2, ctx_dim)
                self.context_updaters = nn.ModuleList([
                    nn.Sequential(
                        Linear(config.n_embd, ctx_hidden, bias=True),
                        nn.GELU(),
                        Linear(ctx_hidden, ctx_dim, bias=True),
                    )
                    for _ in range(config.n_layer)
                ])
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Fix 3 (Bug 3): learnable per-layer blend gate on base_context contribution.
        # Init to -4 → sigmoid(-4) ≈ 0.018, so context_state ≈ delta at init.
        # base_context can only grow its influence as training finds it useful.
        # Only created when use_remix_linear is active (base_context is non-None).
        # 1D tensor → routes to AdamW automatically (p.ndim < 2 in setup_optimizer).
        if self.use_remix_linear:
            self.base_ctx_scales = nn.Parameter(torch.full((config.n_layer,), -4.0))
        else:
            self.base_ctx_scales = None
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """
        # Research branches are also on to_empty() storage and need explicit init.
        def _init_research_module(mod: nn.Module):
            # We use a non-recursive approach to avoid overwriting specialized inits
            for sub in mod.modules():
                if isinstance(sub, RemixedLinear):
                    torch.nn.init.orthogonal_(sub.basis.weight)
                    torch.nn.init.kaiming_normal_(sub.template_mixing)
                    torch.nn.init.zeros_(sub.bias)
                    if sub.use_context:
                        # basis_modulator: zero-init ALL linear weights so that at init
                        # first_linear(ctx) = 0·ctx + 0 = 0 → GELU(0) = 0 → second_linear(0) = 0 + 2.0 = 2.0
                        # → gate_basis = sigmoid(2.0) ≈ 0.88 for ALL positions, independent of context quality.
                        # Fix 1 (Bug 1): previously xavier-init on first linear caused wildly varying gates
                        # at early positions for long sequences where base_context is noisy.
                        # GELU'(0) = 0.5 ≠ 0, so gradients flow and the first layer learns normally from step 1.
                        linears = [m for m in sub.basis_modulator.modules() if isinstance(m, (Linear, nn.Linear))]
                        for m in linears:
                            torch.nn.init.zeros_(m.weight)  # zero BOTH linears (not xavier)
                            if m.bias is not None: torch.nn.init.zeros_(m.bias)
                        if linears: torch.nn.init.constant_(linears[-1].bias, 2.0)
                        # Low-rank output gate: xavier for coeffs, zero for basis
                        # gate_basis zeros => gate_logits=0 => tanh(0)=0 => gate=1.0 at init
                        # This guarantees identity behavior at initialization regardless of T
                        torch.nn.init.xavier_uniform_(sub.output_gate_coeffs.weight)
                        torch.nn.init.zeros_(sub.output_gate_coeffs.bias)
                        torch.nn.init.zeros_(sub.output_gate_basis)
                        torch.nn.init.constant_(sub.output_gate_scale, 0.1)
                    continue  # Skip further processing of this module's sub-components here

                if isinstance(sub, ImprovedContextAwareRouter):
                    torch.nn.init.normal_(sub.routing_queries, mean=0.0, std=sub.router_dim ** -0.5)
                    # expert_proj: normal init (per-position self-attention logits, well-behaved)
                    torch.nn.init.normal_(sub.expert_proj.weight, mean=0.0, std=0.02)
                    if sub.expert_proj.bias is not None: torch.nn.init.zeros_(sub.expert_proj.bias)
                    # Fix 2 (Bug 2): zero-init cross_expert_proj to disable the non-causal global-mean
                    # cross-attention path at init. _cross_attention returns the mean over all T tokens
                    # (leaks future tokens, signal dilutes as mean of 2048 things → near-constant).
                    # Zero-init lets training decide if this path is ever useful; alpha_gate will
                    # initially route all weight to the causal self_attn_logits path.
                    torch.nn.init.zeros_(sub.cross_expert_proj.weight)
                    if sub.cross_expert_proj.bias is not None: torch.nn.init.zeros_(sub.cross_expert_proj.bias)
                    
                    # Projections and gates (Xavier or Normal)
                    for m in [sub.embed_proj, sub.out_proj, sub.temperature_predictor, sub.alpha_gate]:
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: torch.nn.init.zeros_(m.bias)
                    if sub.use_vocab_prior:
                        torch.nn.init.normal_(sub.vocab_routing_bias.weight, mean=0.0, std=0.02)
                    continue

                # Fallback for remaining research linear layers/embeddings
                if isinstance(sub, (Linear, nn.Linear)):
                    # Check if already initialized by a parent (RemixedLinear/Router)
                    # In this simplified logic, we just check if it's a direct child of research
                    torch.nn.init.xavier_uniform_(sub.weight)
                    if sub.bias is not None:
                        torch.nn.init.zeros_(sub.bias)
                elif isinstance(sub, nn.Embedding):
                    torch.nn.init.normal_(sub.weight, mean=0.0, std=0.02)
                elif isinstance(sub, (nn.LayerNorm, nn.RMSNorm)):
                    if getattr(sub, 'weight', None) is not None:
                        torch.nn.init.ones_(sub.weight)
                    if getattr(sub, 'bias', None) is not None:
                        torch.nn.init.zeros_(sub.bias)

                # Buffers also need explicit initialization if to_empty was used
                if hasattr(sub, 'temperature'):
                    # sub.temperature is a buffer, not a parameter, so we set its value directly
                    sub.temperature.fill_(1.0)

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if "wpe" in self.transformer:
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            if isinstance(block, RemixedBlock):
                _init_research_module(block)
                if block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        if self.embedding_model is not None:
            _init_research_module(self.embedding_model)
            # Fix 1E: zero-init expert_codes so initial behavior = standard seed embedding
            if hasattr(self.embedding_model, 'expert_codes'):
                torch.nn.init.zeros_(self.embedding_model.expert_codes)
        if self.context_manager is not None:
            _init_research_module(self.context_manager)
        # Fix 1A: zero-init the *last* linear in each context_updater MLP so initial
        # behaviour is identical to static context (delta starts at zero)
        if self.context_updaters is not None:
            for updater in self.context_updaters:
                # updater is now a Sequential(Linear, GELU, Linear)
                linears = [m for m in updater.modules() if isinstance(m, (Linear, nn.Linear))]
                # Xavier init first layer, zero-init last layer for identity-start
                if len(linears) >= 1:
                    torch.nn.init.xavier_uniform_(linears[0].weight)
                    if linears[0].bias is not None:
                        torch.nn.init.zeros_(linears[0].bias)
                if len(linears) >= 2:
                    torch.nn.init.zeros_(linears[-1].weight)
                    if linears[-1].bias is not None:
                        torch.nn.init.zeros_(linears[-1].bias)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = 256
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    @property
    def max_seq_len(self):
        """Property for use by external evaluation scripts (like core_eval.py)."""
        return self.config.sequence_len

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        wpe_numel = self.transformer.wpe.weight.numel() if "wpe" in self.transformer else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + wpe_numel + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters()) if "wpe" in self.transformer else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        research = 0
        if self.embedding_model is not None:
            research += sum(p.numel() for p in self.embedding_model.parameters())
        if self.context_manager is not None:
            research += sum(p.numel() for p in self.context_manager.parameters())
        if self.context_updaters is not None:
            research += sum(p.numel() for p in self.context_updaters.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if self.base_ctx_scales is not None:
            scalars += self.base_ctx_scales.numel()
        total = wte + wpe + value_embeds + lm_head + transformer_matrices + research + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'wpe': wpe,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': research,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, disable_mu_p=False, mu_p_scale_override=-1.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        candidate_matrix_params = list(self.transformer.h.parameters())
        if self.context_manager is not None:
            candidate_matrix_params += list(self.context_manager.parameters())
        if self.embedding_model is not None:
            candidate_matrix_params += list(self.embedding_model.parameters())
        # Fix 1A: context_updaters are n_embd×context_dim matrices — route to Muon
        if self.context_updaters is not None:
            candidate_matrix_params += list(self.context_updaters.parameters())

        # Muon is only defined for matrix-like tensors (>=2D).
        # Remixed/research branches introduce 1D params (biases/LN scales), route those to AdamW.
        matrix_params = [p for p in candidate_matrix_params if p.ndim >= 2]
        research_adamw_params = [p for p in candidate_matrix_params if p.ndim < 2]

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        if "wpe" in self.transformer:
            embedding_params += list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Fix 3: base_ctx_scales is 1D → AdamW. Include in scalar_params so parameter count holds.
        base_ctx_params = [self.base_ctx_scales] if self.base_ctx_scales is not None else []
        assert len(list(self.parameters())) == len(matrix_params) + len(research_adamw_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(base_ctx_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model, μP-style).
        # Can be disabled via disable_mu_p=True so raw flags are used directly (e.g. for research models
        # that may not satisfy the μP assumptions).
        if mu_p_scale_override > 0.0:
            dmodel_lr_scale = mu_p_scale_override
            print0(f"μP LR scaling OVERRIDDEN to {dmodel_lr_scale:.6f}")
        elif disable_mu_p:
            dmodel_lr_scale = 1.0
            print0(f"μP LR scaling DISABLED — using raw LR flags directly (model_dim={model_dim})")
        else:
            dmodel_lr_scale = (model_dim / 768) ** -0.5
            print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=research_adamw_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            # Fix 3: base_ctx_scales — 1D scalar per layer, same schedule as other scalars
            *([dict(kind='adamw', params=base_ctx_params, lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0)] if base_ctx_params else []),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T

        if T_total > self.cos.size(1):
            # Dynamic cache growth: double the cache or use T_total, whichever is larger
            new_len = max(T_total, self.cos.size(1) * 2)
            print0(f"Growing rotary embeddings cache from {self.cos.size(1)} to {new_len}")
            head_dim = self.config.n_embd // self.config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            # Re-register buffers to update their size (persistent=False as in __init__)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        if self.embedding_model is None:
            x = self.transformer.wte(idx) # embed current token
        else:
            x, _ = self.embedding_model(idx)
        if "wpe" in self.transformer:
            assert T_total <= self.config.sequence_len, f"use_pos_embed=True requires sequence <= {self.config.sequence_len}, got {T_total}"
            positions = torch.arange(T0, T_total, device=idx.device)
            x = x + self.transformer.wpe(positions)
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        # Fix 1A: compute base context once from initial embedding.
        # At long sequences the GlobalContextManager's attention over 2048 raw token
        # embeddings produces a diluted signal.  The per-layer context_updaters now
        # re-ground the context from the residual stream x at each block, so the
        # base_context only needs to provide a coarse initialisation.
        base_context = self.context_manager(x, idx) if self.context_manager is not None else None
        context_state = base_context
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            if self.use_remix_linear:
                if self.context_updaters is not None:
                    # Read from the residual stream x (projected to context_dim), not from
                    # context_state itself.  This re-grounds the context signal at every
                    # layer from actual transformer representations, fixing the long-sequence
                    # dilution problem where the self-referential context loop carried no
                    # new information beyond what the GlobalContextManager produced at t=0.
                    delta = self.context_updaters[i](norm(x))
                    # Fix 3 (Bug 3): blend base_context in with a learned per-layer gate.
                    # At init: sigmoid(-4) ≈ 0.018, so context_state ≈ delta.
                    # base_context's influence grows only as training finds it useful,
                    # preventing the noisy diluted GlobalContextManager signal from
                    # dominating at early training steps with long sequences.
                    if base_context is not None:
                        bc_weight = torch.sigmoid(self.base_ctx_scales[i].to(dtype=delta.dtype))
                        context_state = bc_weight * base_context + delta
                    else:
                        context_state = delta
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, context_state=context_state)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 20 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
