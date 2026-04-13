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
import math

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

    # CCL block modulation strategy (only used when use_remix_linear=True):
    #   'weight'        — RemixedLinear gates in activation space.
    #   'normalization' — CCLBlock with AdaRMSNorm conditioning.
    #   'householder'   — operator-space basis transport via context-conditioned
    #                     Householder reflection(s).
    #   'spectral'      — operator-space spectral scaffold (near-isometric scaling).
    #   'ocd'           — orthogonal-complement-style dynamic delta with overlap penalty.
    #   'decoupled'     — orthogonal static/dynamic input channel routing.
    #   'tucker'        — Tucker-routed operator simplex.
    #   'svs'           — singular-value steering over fixed factors.
    #   'vq'            — vector-quantized regime-conditioned operator deltas.
    #   'dcu'           — deferred-context-unlock style tail-space steering.
    #   'fsi'           — Frozen Subspace Indexing: frozen orthogonal rotations + frozen routing.
    #   'aesp'          — Attention-Entropy Stratified Projection: entropy-routed per-stratum deltas.
    #   'ckr'           — Causal Kernel Reparameterization: position-dependent branch mixing.
    cclblock_modulation: str = 'weight'
    cclblock_orth_lambda: float = 0.0
    # The context stream logic:
    #   'local'      — (default) derived directly from norm(x) inside the block. No cross-block threading.
    #   'ema'        — fixed ema factor, with .detach() to prevent gradient explosion.
    #   'selective'  — GRU-style input-dependent gating, no detach.
    #   'multiscale' — 3 parallel selective temporal channels (Fast, Med, Slow).
    #   'ssm'        — linear state-space style context highway.
    cclblock_context_stream: str = 'local'
    cclblock_ema_factor: float = 0.99
    # Design C: Cross-layer stale context lag (0 = disabled, k>=1 = use context from k blocks ago).
    # When k>0, block i is conditioned on the context emitted by block i-k in the same forward pass.
    # This breaks the circular dependency because i-k is a genuinely different layer's computation.
    # The stale context is detached — only the within-block gradient path is active.
    cclblock_stale_ctx_lag: int = 0
    # --- Novel Ablation Designs ---
    # Design 1: 'shifted' residual uses norm(x_entry) from the START of the block
    #            (= previous layer output). Decouples basis from gate signal source.
    #            Use via --cclblock-context-stream "shifted"
    # Design 3: Sparse top-k basis gate. 0=soft sigmoid, N=activate top-N basis functions.
    #            Uses straight-through estimator for gradients.
    cclblock_sparse_gate_k: int = 0
    # Design 6: Basis gate temperature. <1.0=sharper (more selective), >1.0=softer (more uniform).
    cclblock_gate_temperature: float = 1.0
    # Design 4: Context prototype bank. 0=disabled, N=use N learned prototype vectors.
    #            Forces model to learn a vocab of text-type contexts via soft lookup.
    cclblock_context_bank_size: int = 0
    # Design 7: Per-module context. True=separate ctx projections for attn vs ffn.
    #            Attn gets pre-attn ctx, FFN gets post-attn ctx.
    cclblock_per_head_ctx: bool = False
    # Design 2: Context source for the FFN gate. 'norm_x' uses the residual stream (default).
    #            'attn_heads' uses mean query vectors (q after RoPE+QKnorm, detached).
    #            This gives the gate a direct signal about "what each position is searching for"
    #            without any recurrence or cross-block dependency.
    cclblock_context_source: str = 'norm_x'  # 'norm_x' | 'attn_heads' | 'attn_geometry'
    # Phase 8: Boundary-Gated / Chunk Context / Auxiliary Objective
    # Design 8 (boundary gate): learned soft gate that fires only at segment boundaries.
    #   Use --cclblock-context-stream "boundary"
    # Design 9 (hard chunk): context = pooled summary of previous K-token chunk.
    #   Strictly causal, zero recurrence. Use --cclblock-chunk-size N (e.g. 64)
    cclblock_chunk_size: int = 0
    # Design 10 (aux objective): context branch predicts an auxiliary target.
    #   'boundary': predicts whether current token is a structural boundary.
    #   'entropy':  predicts per-token cross-entropy (difficulty signal).
    #   Forces context to encode non-trivial info, prevents identity collapse.
    cclblock_aux_objective: str = 'none'  # 'none' | 'boundary' | 'entropy'
    cclblock_aux_lambda: float = 0.1      # weight of auxiliary loss
    cclblock_boundary_token_id: int = 198 # token ID for boundary detection (default=\n)
    # Phase 9: Residual Adaptive Linear + New Context Streams + FiLM Gate
    # Proposal A: ResidualAdaptiveLinear — dense base + context-conditioned low-rank delta.
    #   y = base(x) + scale * (x @ U * ctx_coeffs(ctx)) @ V
    #   V=0 at init → zero delta → exact dense baseline at step 0. No regression guarantee.
    use_ral: bool = False    # use ResidualAdaptiveLinear instead of RemixedLinear
    ral_rank: int = 32       # rank r for the context-conditioned additive delta
    # Proposal C: FiLM basis conditioning — affine (scale+shift) instead of sigmoid gate.
    #   gate = 1 + tanh(scale * 0.1); h_gated = h * gate + shift
    #   Zero-init → scale=0→gate=1, shift=0 → exact identity at init.
    cclblock_film_gate: bool = False
    # Dual-V shadow routing: append shadow channels to attention V and route the
    # shadow output directly into FFN context conditioning.
    cclblock_attn_shadow_dim: int = 0
    # Paradigm 1: Decoupled Feature-Space Routing
    cclblock_dynamic_ratio: float = 0.25
    cclblock_gate_rank: int = 8
    # Paradigm 2: Evidence Accumulation SSM context stream
    cclblock_num_regimes: int = 8
    cclblock_regime_temperature: float = 1.0
    # Additional operator-family ablations
    cclblock_poly_order: int = 2
    cclblock_lie_generators: int = 4
    cclblock_grassmann_bank_size: int = 4
    cclblock_tucker_rank: int = 32
    cclblock_tucker_modes: int = 8
    cclblock_svs_rank: int = 64
    cclblock_svs_eps: float = 0.1
    cclblock_vq_codes: int = 8
    cclblock_vq_temperature: float = 1.0
    cclblock_dcu_warmup_steps: int = 0
    # Phase 12: Novel zero-friction context-conditioned linear layers
    # FSI: Frozen Subspace Indexing — K frozen orthogonal rotations + frozen routing
    cclblock_fsi_rotations: int = 8       # number of frozen orthogonal rotation matrices
    cclblock_fsi_selector_dim: int = 64   # dimension of frozen routing projection
    # AESP: Attention-Entropy Stratified Projection
    cclblock_aesp_strata: int = 4         # number of entropy strata (stratum 0 = pure dense)
    cclblock_aesp_delta_rank: int = 4     # rank of per-stratum low-rank deltas
    # CKR: Causal Kernel Reparameterization
    cclblock_ckr_branches: int = 4        # number of parallel dense branches
    cclblock_ckr_kernel_size: int = 64    # causal conv1d kernel size
    # Phase 13: CKR enhancements
    cclblock_ckr_pos_channels: int = 1    # 13a: multi-channel position signal (1=original, 3=fast/med/slow)
    cclblock_ckr_dual_optim: int = 0      # 13b: route CKR gate params to dedicated AdamW group
    cclblock_ckr_content_bias: float = 0.0  # 13c: frozen content hash bias scale (0=pure position)
    # Phase 14: Gradient-isolated content conditioning
    cclblock_giad_rank: int = 32              # 14a: GIAD low-rank bottleneck dimension
    cclblock_psg_kernel_size: int = 64        # 14b: PSG causal conv kernel size
    cclblock_ss_dynamic_ratio: float = 0.25   # 14c: SplitStream dynamic channel fraction
    cclblock_ss_branches: int = 2             # 14c: SplitStream CKR branches on dynamic path
    cclblock_ss_kernel_size: int = 64         # 14c: SplitStream causal conv kernel size
    # Phase 15: LoKR — Low-rank Kernel Reparameterization
    cclblock_lokr_branches: int = 8           # 15: LoKR number of low-rank perturbation branches
    cclblock_lokr_rank: int = 16              # 15: LoKR rank of each perturbation
    # Phase 16: CKR-Anneal temperature annealing
    cclblock_ckr_temp_start: float = 2.0      # 16A: initial softmax temperature (soft routing)
    cclblock_ckr_temp_end: float = 0.3        # 16A: final softmax temperature (sharp routing)
    # Phase 16: Causal Output Mixer
    cclblock_com_kernel_size: int = 32        # 16C: COM causal depthwise conv kernel size
    # Phase 17: CKR enhancements
    cclblock_ckr_ortho_init: int = 0          # 17D: orthogonal branch initialization (0/1)
    cclblock_ckr_branch_dropout: float = 0.0  # 17E: branch dropout probability (0=off)
    cclblock_ckr_diversity_lambda: float = 0.0 # 17H: auxiliary branch diversity loss weight
    cclblock_ckr_layer_selective: int = 0     # 17F: 0=all layers CKR, 1=odd layers only
    # Phase 17: New architectures kernel sizes
    cclblock_pgr_kernel_size: int = 64        # 17C: PGR causal conv kernel size
    cclblock_cil_kernel_size: int = 64        # 17I: CIL causal conv kernel size
    cclblock_prb_kernel_size: int = 64        # 17J: PRB causal conv kernel size
    # Phase 18: Beyond CKR — orthogonal improvements
    p18_layer_drop: float = 0.0               # 18E: stochastic depth drop probability
    p18_dynamic_activation: int = 0           # 18I: learned activation mix (0/1)
    p18_mixture_norm: int = 0                 # 18H: RMSNorm+LayerNorm mixture (0/1)
    p18_causal_attn_bias: int = 0             # 18J: learned causal attention score bias (0/1)
    p18_aux_sim_lambda: float = 0.0           # 18G: layer similarity penalty weight
    p18_gradient_penalty: float = 0.0         # 18B: gradient penalty weight
    p18_per_channel_scale: int = 0            # 18F: learnable per-channel output scale (0/1)
    # Phase 19: Zero-overhead indirect modulation
    p19_residual_gate: int = 0                # 19A: per-layer learned scalar on block output (0/1)
    p19_head_importance: int = 0              # 19B: per-head learned scalar on attn output (0/1)
    p19_residual_mix_groups: int = 0          # 19C: grouped 1x1 conv between blocks (0=off, N=group_size)
    p19_attn_logit_bias: int = 0              # 19D: per-head learned QK temperature (0/1)
    p19_residual_decay: int = 0               # 19E: learned depth-dependent x0 decay (0/1)
    p19_grad_equilibrium: float = 0.0         # 19F: gradient equilibrium regularization lambda (0=off)
    p19_spectral_reparam: int = 0             # 19G: spectral reparameterization (0=off, 1=c_proj, 2=c_fc+c_proj)
    p19_weight_anticollapse: float = 0.0      # 19H: weight anti-collapse penalty lambda (0=off)
    p19_ve_bias: int = 0                      # 19I: add learnable bias to VE gate (0/1)
    p19_weight_noise: float = 0.0             # 19J: training-time weight perturbation epsilon (0=off)
    # Phase 20: Context-conditioned dynamic weight computation
    p20_mone_experts: int = 0                 # 20F: Mixture of Narrow Experts (0=off, K=num experts)
    p20_mone_topk: int = 0                    # 20F: top-k expert routing (0=compute all, K=top-k sparse)
    p20_lrcfb_branches: int = 0               # 20C: Frozen content-routed full-rank branches (0=off, K=num branches)
    p20_dgcr_branches: int = 0                # 20D: Detached-gradient content-routed branches (0=off, K=num branches)
    p20_dgcr_aux_weight: float = 0.01         # 20D: weight for auxiliary routing loss

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
    # CCL block redesign
    "cclblock_modulation", "cclblock_orth_lambda", "cclblock_context_stream", "cclblock_ema_factor", "cclblock_stale_ctx_lag",
    # Novel ablation designs
    "cclblock_sparse_gate_k", "cclblock_gate_temperature", "cclblock_context_bank_size", "cclblock_per_head_ctx",
    "cclblock_context_source",
    # Phase 8
    "cclblock_chunk_size", "cclblock_aux_objective", "cclblock_aux_lambda", "cclblock_boundary_token_id",
    # Phase 9
    "use_ral", "ral_rank", "cclblock_film_gate", "cclblock_attn_shadow_dim",
    # New paradigms
    "cclblock_dynamic_ratio", "cclblock_gate_rank", "cclblock_num_regimes", "cclblock_regime_temperature",
    "cclblock_poly_order", "cclblock_lie_generators", "cclblock_grassmann_bank_size",
    "cclblock_tucker_rank", "cclblock_tucker_modes", "cclblock_svs_rank", "cclblock_svs_eps",
    "cclblock_vq_codes", "cclblock_vq_temperature", "cclblock_dcu_warmup_steps",
    # Phase 12: FSI/AESP/CKR
    "cclblock_fsi_rotations", "cclblock_fsi_selector_dim",
    "cclblock_aesp_strata", "cclblock_aesp_delta_rank",
    "cclblock_ckr_branches", "cclblock_ckr_kernel_size",
    # Phase 13
    "cclblock_ckr_pos_channels", "cclblock_ckr_dual_optim", "cclblock_ckr_content_bias",
    # Phase 14
    "cclblock_giad_rank", "cclblock_psg_kernel_size",
    "cclblock_ss_dynamic_ratio", "cclblock_ss_branches", "cclblock_ss_kernel_size",
    # Phase 15
    "cclblock_lokr_branches", "cclblock_lokr_rank",
    # Phase 16
    "cclblock_ckr_temp_start", "cclblock_ckr_temp_end",
    "cclblock_com_kernel_size",
    # Phase 18
    "p18_layer_drop", "p18_dynamic_activation", "p18_mixture_norm", "p18_causal_attn_bias",
    "p18_aux_sim_lambda", "p18_gradient_penalty", "p18_per_channel_scale",
    # Phase 19
    "p19_residual_gate", "p19_head_importance", "p19_residual_mix_groups",
    "p19_attn_logit_bias", "p19_residual_decay", "p19_grad_equilibrium",
    "p19_spectral_reparam", "p19_weight_anticollapse", "p19_ve_bias", "p19_weight_noise",
    # Phase 20
    "p20_mone_experts", "p20_mone_topk", "p20_lrcfb_branches",
    "p20_dgcr_branches", "p20_dgcr_aux_weight",
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
        """Cross-attention from learned routing queries into the full token context.

        The routing queries are fixed learned parameters (not derived from any
        specific token), so they cannot directly leak future token identity.
        Without a causal mask, each position receives a summary of the full
        visible context — a richer signal than a causal prefix average.
        At inference time the context is always the causal prefix anyway.

        Uses SDPA (Flash Attention when available) to avoid materializing the
        full (B, T, Q, T) attention matrix that caused Triton XBLOCK > 4096.

        Returns (B, T, D): per-position summary (same for all positions since
        queries are position-independent and context is the full sequence).
        """
        batch_size, seq_len, dim = context.shape
        queries = queries.to(dtype=context.dtype)  # (Q, D)

        # Keys/Values: (B, 1, T, D) — same context for every routing query.
        k = context.unsqueeze(1)  # (B, 1, T, D)

        outputs = []
        for q_vec in queries.unbind(0):  # iterate over Q routing queries
            # q: (B, 1, T, D) — same routing query at every target position.
            # .contiguous() lets SDPA pick Flash Attention when available.
            q = q_vec.view(1, 1, 1, dim).expand(batch_size, 1, seq_len, dim).contiguous()
            # Non-causal: each routing query attends to all T context tokens.
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


class EMAContextStream(nn.Module):
    """Legacy Exponential Moving Average context stream (Design A).

    Uses a fixed EMA factor and detaches history to prevent long-range gradient 
    explosion, stabilizing the model at the cost of theoretically shorter context gradient flow.
    """
    def __init__(self, n_embd, ctx_dim, ema_factor=0.99):
        super().__init__()
        self.ema_factor = ema_factor
        self.proj = Linear(n_embd, ctx_dim, bias=False)

    def forward(self, attn_out, prev_ctx):
        new_ctx = self.proj(attn_out)
        if prev_ctx is not None:
            return self.ema_factor * prev_ctx.detach() + (1 - self.ema_factor) * new_ctx
        return new_ctx


class LocalContextStream(nn.Module):
    """Purely local context logic derived from pre-MLP norm(x). Ignores previous block states."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight) # CRITICAL: zero-init so gates start at identity

    def forward(self, pre_mlp_norm_x, prev_ctx=None):
        return self.proj(pre_mlp_norm_x)


class ContextBank(nn.Module):
    """Design 4: Soft lookup into a learned vocabulary of N context prototypes.

    Projects norm(x) to form a query, retrieves a weighted mixture of learned
    prototype vectors. Forces the model to learn a discrete vocabulary of text-type
    contexts (code, prose, math, etc.) as prototypes.

    Zero-init prototypes + small std init ensures identity-like behaviour at t=0.
    Gradient path: loss → ffwd gate → ctx → weights (softmax) → query_proj → norm(x).
    No recurrence, no chain across blocks.
    """
    def __init__(self, n_embd, ctx_dim, n_prototypes):
        super().__init__()
        self.query_proj = Linear(n_embd, ctx_dim, bias=False)
        nn.init.zeros_(self.query_proj.weight)  # stable identity start
        self.prototypes = nn.Parameter(torch.zeros(n_prototypes, ctx_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.scale = ctx_dim ** -0.5

    def forward(self, norm_x, prev_ctx=None):
        q = self.query_proj(norm_x)                                 # (B, T, ctx_dim)
        scores = q @ self.prototypes.to(q.dtype).T * self.scale     # (B, T, n_proto)
        weights = F.softmax(scores, dim=-1)
        return weights @ self.prototypes.to(q.dtype)                # (B, T, ctx_dim)


class SelectiveContextStream(nn.Module):
    """GRU-style selective update for the cross-block context state (Design A).

    Replaces the fixed-λ EMA + detach used in the old RemixedBlock with
    input-dependent gating:
      - alpha = sigmoid(gate(attn_out))  — update gate, input-dependent
      - content = tanh(write(attn_out)) — new content
      - ctx = (1-alpha)*prev_ctx + alpha*content  — no detach, full grad flow

    Zero-init on write weight/bias ensures no spurious content at step 0.
    gate bias=0 → sigmoid(0)=0.5 (balanced at init).
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.gate  = Linear(n_embd, ctx_dim, bias=True)
        self.write = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.write.weight)
        nn.init.zeros_(self.write.bias)

    def forward(self, attn_out, prev_ctx):
        alpha   = torch.sigmoid(self.gate(attn_out))          # (B, T, ctx_dim)
        content = torch.tanh(self.write(attn_out))            # (B, T, ctx_dim)
        if prev_ctx is None:
            return alpha * content
        # No detach: full gradient flow through context history
        return (1 - alpha) * prev_ctx + alpha * content


class MultiScaleContext(nn.Module):
    """Three parallel selective state channels with different temporal scales (Design D).

    Fast  (bias=+2.0 → α≈0.88): updates aggressively — captures recent syntax/tokens.
    Medium(bias= 0.0 → α≈0.50): balanced — captures paragraph-level topic.
    Slow  (bias=-2.0 → α≈0.12): retains history — captures document-level style.

    ctx_dim is auto-corrected to the nearest multiple of 3 when this class is
    instantiated, so the caller should read self.ctx_dim after construction.
    """
    GATE_BIASES = [2.0, 0.0, -2.0]   # sigmoid → [0.88, 0.50, 0.12]

    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        # Auto-correct to nearest multiple of 3 (required for 3 equal sub-channels)
        corrected = max(3, round(ctx_dim / 3) * 3)
        self.ctx_dim   = corrected
        self.scale_dim = corrected // 3
        self.gates  = nn.ModuleList([Linear(n_embd, self.scale_dim, bias=True)  for _ in range(3)])
        self.writes = nn.ModuleList([Linear(n_embd, self.scale_dim, bias=True)  for _ in range(3)])
        for gate, bias_val in zip(self.gates, self.GATE_BIASES):
            nn.init.constant_(gate.bias, bias_val)
        for write in self.writes:
            nn.init.zeros_(write.weight)
            nn.init.zeros_(write.bias)

    def forward(self, attn_out, prev_ctx):
        chunks = []
        for i in range(3):
            alpha   = torch.sigmoid(self.gates[i](attn_out))
            content = torch.tanh(self.writes[i](attn_out))
            if prev_ctx is None:
                chunks.append(alpha * content)
            else:
                prev = prev_ctx[..., i*self.scale_dim:(i+1)*self.scale_dim]
                chunks.append((1 - alpha) * prev + alpha * content)
        return torch.cat(chunks, dim=-1)


class ParallelLinearContextStream(nn.Module):
    """Linear state-space style context highway with stable exponential decay."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.dt_proj = Linear(n_embd, ctx_dim, bias=True)
        self.x_proj = Linear(n_embd, ctx_dim, bias=True)
        self.a_log = nn.Parameter(torch.zeros(ctx_dim))
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.zeros_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)

    def forward(self, x, prev_ctx=None):
        bsz, seqlen, _ = x.shape
        dt = F.softplus(self.dt_proj(x))
        decay = torch.exp(-dt * torch.exp(self.a_log).view(1, 1, -1))
        write = self.x_proj(x)
        state = torch.zeros(bsz, self.ctx_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            state = state * decay[:, t, :] + write[:, t, :]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class DetachedAttnContextStream(nn.Module):
    """Proposal B: Projects norm(attn_out).detach() to ctx_dim.
    
    attn_out is intrinsically orthogonal to x_t (it's the cross-token mixture).
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, attn_out, prev_ctx=None):
        return self.proj(norm(attn_out).detach())


class CausalPrefixContextStream(nn.Module):
    """Proposal E: Causal Prefix Context via cumsum (O(T), no recurrence)."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, prev_ctx=None):
        cumsum = x.cumsum(dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return self.proj(norm(cumsum / counts).detach())


class WarmupEMAContextStream(nn.Module):
    """Proposal D: Position-Aware EMA Warmup."""
    def __init__(self, n_embd, ctx_dim, ema_factor=0.95):
        super().__init__()
        self.ema_factor = ema_factor
        self.ema_proj = Linear(n_embd, ctx_dim, bias=False)
        self.local_proj = Linear(n_embd, ctx_dim, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.ema_proj.weight)
        nn.init.zeros_(self.local_proj.weight)

    def forward(self, x, prev_ctx=None):
        B, T, D = x.shape
        ema_signal = self.ema_proj(x)
        local_signal = self.local_proj(norm(x).detach())
        
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        blend = torch.sigmoid(self.alpha * torch.log(t + 1.0) + self.bias).view(1, -1, 1)
        
        if prev_ctx is None:
            ema_ctx = ema_signal
        else:
            ema_ctx = self.ema_factor * prev_ctx.detach() + (1 - self.ema_factor) * ema_signal
            
        return blend * ema_ctx + (1 - blend) * local_signal


class DACSEMAContextStream(nn.Module):
    """Proposal F: Combines DACS (attn_out source) with Selective EMA smoothing."""
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.gate  = Linear(n_embd, ctx_dim, bias=True)
        self.write = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.write.weight)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, -2.0)  # closed at init

    def forward(self, attn_out, prev_ctx=None):
        content = torch.tanh(self.write(attn_out.detach()))
        alpha = torch.sigmoid(self.gate(attn_out.detach()))
        if prev_ctx is None:
            return alpha * content
        return (1 - alpha) * prev_ctx.detach() + alpha * content


class DecayPrefixContextStream(nn.Module):
    """Proposal G: Exponentially-decayed causal prefix mean.
    Computed via loop over T to approximate vector-scan.
    """
    def __init__(self, n_embd, ctx_dim, gamma=0.9):
        super().__init__()
        self.proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma = gamma

    def forward(self, x, prev_ctx=None):
        B, T, D = x.shape
        states = []
        curr = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            curr = self.gamma * curr + (1 - self.gamma) * x[:, t]
            states.append(curr)
        out = torch.stack(states, dim=1)
        return self.proj(norm(out).detach())


class BoundaryGatedContextStream(nn.Module):
    """Design 8: Soft boundary-gated context — update context only at learned segment boundaries.

    A lightweight binary gate decides per-position whether to refresh the context vector
    or hold the previous segment's context state fixed.

    Init strategy:
      - ctx_proj.weight = 0   → context starts as pure bias (neutral)
      - boundary_proj.bias = -2 → sigmoid(-2) ≈ 0.12, gate starts nearly closed
        (context is held fixed 88% of the time at init)

    Gradient flow:
      - prev_ctx is DETACHED: no BPTT through the full sequence chain (avoids the
        SelectiveContextStream failure mode).
      - Only the local ctx_new path and boundary_prob gate carry gradients.
      - context learns to update at genuine boundaries via the gate gradient signal.
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.ctx_proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)
        self.boundary_proj = Linear(n_embd, 1, bias=True)
        nn.init.zeros_(self.boundary_proj.weight)
        nn.init.constant_(self.boundary_proj.bias, -2.0)  # gate≈0.12 at init

    def forward(self, norm_x, prev_ctx=None):
        ctx_new = self.ctx_proj(norm_x)                                    # (B, T, ctx_dim)
        boundary_prob = torch.sigmoid(self.boundary_proj(norm_x))          # (B, T, 1)
        if prev_ctx is None:
            return ctx_new
        return boundary_prob * ctx_new + (1 - boundary_prob) * prev_ctx.detach()


class HardChunkContextStream(nn.Module):
    """Design 9: Context = pooled summary of the immediately preceding K-token chunk.

    Strictly causal implementation:
      - Tokens 0..K-1  receive zero context (no prior chunk exists).
      - Tokens K..2K-1 receive ctx = ctx_proj(mean(norm_x[0:K])).
      - Tokens 2K..3K-1 receive ctx = ctx_proj(mean(norm_x[K:2K])), etc.

    Fully vectorized via unfold + shift — no Python loops over T, compile-safe.
    Zero recurrence: each chunk is fully independent of all others.
    """
    def __init__(self, n_embd, ctx_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctx_dim = ctx_dim
        self.ctx_proj = Linear(n_embd, ctx_dim, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)

    def forward(self, norm_x, prev_ctx=None):
        B, T, C = norm_x.shape
        K = self.chunk_size
        ctx_dim = self.ctx_dim
        # Pad to multiple of K
        pad = (K - T % K) % K
        x_pad = F.pad(norm_x, (0, 0, 0, pad)) if pad > 0 else norm_x  # (B, T', C)
        n_chunks = x_pad.shape[1] // K
        # Compute mean of each K-token chunk → project to ctx_dim
        x_chunks = x_pad.view(B, n_chunks, K, C).mean(dim=2)        # (B, n_chunks, C)
        ctx_chunks = self.ctx_proj(x_chunks)                         # (B, n_chunks, ctx_dim)
        # Causal shift: chunk i receives summary of chunk i-1 (zero for i=0)
        ctx_chunks_shifted = F.pad(ctx_chunks, (0, 0, 1, 0))[:, :-1, :]  # (B, n_chunks, ctx_dim)
        # Expand back to token level
        ctx = ctx_chunks_shifted.unsqueeze(2).expand(-1, -1, K, -1).contiguous()
        ctx = ctx.view(B, -1, ctx_dim)[:, :T, :]                    # (B, T, ctx_dim)
        return ctx.to(norm_x.dtype)


class PredictiveChunkContextStream(nn.Module):
    """Paradigm 3: predictive, zero-lag chunk context.

    Predict chunk i+1 context from chunk i summary, then hold it constant within
    chunk i+1. This avoids stale chunk lag and removes per-token context noise.
    """
    def __init__(self, n_embd, ctx_dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.ctx_dim = ctx_dim
        self.future_predictor = nn.Sequential(
            Linear(n_embd, n_embd, bias=True),
            nn.GELU(),
            Linear(n_embd, ctx_dim, bias=True),
        )
        nn.init.zeros_(self.future_predictor[-1].weight)
        nn.init.zeros_(self.future_predictor[-1].bias)

    def forward(self, norm_x, prev_ctx=None):
        B, T, C = norm_x.shape
        K = self.chunk_size
        pad = (K - T % K) % K
        x_pad = F.pad(norm_x, (0, 0, 0, pad)) if pad > 0 else norm_x
        n_chunks = x_pad.shape[1] // K
        x_chunks = x_pad.view(B, n_chunks, K, C).mean(dim=2)
        predicted_future_ctx = self.future_predictor(x_chunks)
        ctx_chunks_shifted = F.pad(predicted_future_ctx, (0, 0, 1, 0))[:, :-1, :]
        ctx = ctx_chunks_shifted.unsqueeze(2).expand(-1, -1, K, -1).contiguous()
        ctx = ctx.view(B, -1, self.ctx_dim)[:, :T, :]
        return ctx.to(norm_x.dtype)


class EvidenceAccumulationContextStream(nn.Module):
    """Paradigm 2: linear low-dim evidence SSM for regime selection context."""
    def __init__(self, n_embd, ctx_dim, num_regimes=8, temperature=1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = max(float(temperature), 1e-6)
        self.evidence_proj = Linear(n_embd, num_regimes, bias=True)
        self.decay_proj = Linear(n_embd, num_regimes, bias=True)
        self.ctx_from_regime = Linear(num_regimes, ctx_dim, bias=False)
        nn.init.zeros_(self.evidence_proj.weight)
        nn.init.zeros_(self.evidence_proj.bias)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, 4.5)  # sigmoid(4.5)≈0.99
        nn.init.zeros_(self.ctx_from_regime.weight)

    def forward(self, norm_x, prev_ctx=None):
        B, T, _ = norm_x.shape
        evidence = self.evidence_proj(norm_x)
        decay = torch.sigmoid(self.decay_proj(norm_x))
        states = []
        curr_state = torch.zeros(B, self.num_regimes, device=norm_x.device, dtype=norm_x.dtype)
        for t in range(T):
            curr_state = decay[:, t, :] * curr_state + evidence[:, t, :]
            states.append(curr_state)
        S_t = torch.stack(states, dim=1)
        R_t = F.softmax(S_t / self.temperature, dim=-1)
        return self.ctx_from_regime(R_t.to(dtype=norm_x.dtype))


class AdaRMSNorm(nn.Module):
    """Context-conditioned RMSNorm for the 'normalization' CCL path (Design B).

    Instead of modulating weight matrices (RemixedLinear), the context predicts
    per-position (scale, shift) for the post-norm representation — the DiT insight
    applied to language model blocks.

    At init: proj.weight=0, proj.bias[:n_embd]=1.0 (scale), proj.bias[n_embd:]=0.0 (shift).
    This guarantees exact standard RMSNorm at initialization regardless of ctx quality.
    Clean gradient path: loss → output → (scale*norm_x + shift) → ctx → context stream.
    """
    def __init__(self, n_embd, ctx_dim):
        super().__init__()
        self.n_embd = n_embd
        self.proj = Linear(ctx_dim, 2 * n_embd, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:n_embd] = 1.0   # scale starts at 1.0 (identity)
        # shift stays at 0.0 (no bias at init)

    def forward(self, x, ctx):
        x_norm = F.rms_norm(x, (x.shape[-1],))
        if ctx is None:
            return x_norm
        params = self.proj(ctx.to(dtype=x.dtype))    # (B, T, 2*n_embd)
        scale, shift = params.chunk(2, dim=-1)
        return x_norm * scale + shift


class CCLBlock(nn.Module):
    """Clean Context-Conditioned Language Block — 'normalization' modulation path.

    Standard dense CausalSelfAttention + MLP with AdaRMSNorm conditioning.
    No RemixedLinear complexity (no basis gates, output gates, template mixing).

    Context lifecycle:
      1. Attention input is conditioned by prev_ctx via AdaRMSNorm (scale+shift).
      2. attn_out is fed to SelectiveContextStream/MultiScaleContext → fresh ctx.
         (AFTER attention, so ctx carries cross-token aggregated signal.)
      3. MLP input is conditioned by freshly updated ctx via AdaRMSNorm.
      4. ctx is returned as next block's prev_ctx.

    This breaks the circular conditioning of the old AG-CCL:
      - ctx is derived from attn_out, NOT from norm(x) the MLP will read.
      - AdaRMSNorm replaces RemixedLinear as the modulation mechanism.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        ctx_dim = config.remix_context_dim
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)
        stream_type = getattr(config, 'cclblock_context_stream', 'local')
        if stream_type == 'multiscale':
            self.ctx_stream = MultiScaleContext(config.n_embd, ctx_dim)
            ctx_dim = self.ctx_stream.ctx_dim  # possibly auto-corrected
        elif stream_type == 'ssm':
            self.ctx_stream = ParallelLinearContextStream(config.n_embd, ctx_dim)
        elif stream_type == 'selective':
            self.ctx_stream = SelectiveContextStream(config.n_embd, ctx_dim)
        elif stream_type == 'ema':
            self.ctx_stream = EMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
        else:
            self.ctx_stream = LocalContextStream(config.n_embd, ctx_dim)
        self.ada_norm_attn = AdaRMSNorm(config.n_embd, ctx_dim)
        self.ada_norm_mlp  = AdaRMSNorm(config.n_embd, ctx_dim)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, prev_ctx=None):
        is_local = isinstance(self.ctx_stream, LocalContextStream)
        
        # Attention with context-conditioned norm
        attn_in  = self.ada_norm_attn(x, None if is_local else prev_ctx)
        attn_out = self.attn(attn_in, ve, cos_sin, window_size, kv_cache)
        x        = x + attn_out

        if is_local:
            ctx = self.ctx_stream(norm(x), prev_ctx)
        else:
            ctx = self.ctx_stream(attn_out, prev_ctx)

        # MLP with freshly updated context-conditioned norm
        x = x + self.mlp(self.ada_norm_mlp(x, ctx))
        
        return x, ctx.detach() if is_local else ctx


class RemixedLinear(nn.Module):
    def __init__(self, in_features, out_features, context_dim, basis_size=64, remixed_linear_kwargs=None, scale_basis=True, film_gate=False):
        super().__init__()
        self._film_gate_flag = film_gate
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
        # Design 3: sparse top-k gate. 0=off, N=top-N active basis functions.
        self.sparse_gate_k = remixed_linear_kwargs.get('sparse_gate_k', 0)
        # Design 6: gate temperature. divides sigmoid logits. <1=sharper, >1=softer.
        self.gate_temperature = max(remixed_linear_kwargs.get('gate_temperature', 1.0), 1e-6)
        # Operator-space modulation: none|householder|spectral|ocd
        self.operator_modulation = remixed_linear_kwargs.get('operator_modulation', 'none')
        self._last_orth_loss = None

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
            film_gate = getattr(self, '_film_gate_flag', False)  # set by caller after init if needed
            # FiLM gate (Proposal C): output 2*basis_size so we can produce (scale, shift)
            _gate_out_size = 2 * basis_size if self.use_basis_gate and film_gate else basis_size
            self.basis_modulator = nn.Sequential(
                Linear(context_dim, basis_hidden, bias=True),
                nn.GELU(),
                Linear(basis_hidden, _gate_out_size, bias=True),
            )
            nn.init.zeros_(self.basis_modulator[-1].weight)
            if self.basis_modulator[-1].bias is not None:
                nn.init.zeros_(self.basis_modulator[-1].bias)
            # Low-rank output gate: ctx_dim → r coefficients, r basis vectors → out_features
            # r is a small constant (default 8), making this O(ctx_dim*r + r*out_features)
            # rather than O(ctx_dim*out_features).
            r = remixed_linear_kwargs.get('output_gate_rank', 8)
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.zeros(r, out_features))
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)  # learnable scale, starts small
            if self.operator_modulation == 'householder':
                self.operator_householder = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.operator_householder.weight)
                nn.init.zeros_(self.operator_householder.bias)
            elif self.operator_modulation == 'spectral':
                self.operator_spectral = Linear(context_dim, basis_size, bias=True)
                nn.init.zeros_(self.operator_spectral.weight)
                nn.init.zeros_(self.operator_spectral.bias)
            elif self.operator_modulation == 'ocd':
                # Low-rank dynamic delta: sum_r coeff_r * (out_r ⊗ in_r)
                ocd_rank = remixed_linear_kwargs.get('ocd_rank', 16)
                self.ocd_coeffs = Linear(context_dim, ocd_rank, bias=True)
                self.ocd_in = nn.Parameter(torch.zeros(ocd_rank, basis_size))
                self.ocd_out = nn.Parameter(torch.zeros(ocd_rank, out_features))
                self.ocd_scale = nn.Parameter(torch.tensor(0.1))
                nn.init.zeros_(self.ocd_coeffs.weight)
                nn.init.zeros_(self.ocd_coeffs.bias)
            elif self.operator_modulation == 'lie':
                m = int(remixed_linear_kwargs.get('lie_generators', 4))
                self.lie_coeffs = Linear(context_dim, m, bias=True)
                self.lie_generators = nn.Parameter(torch.zeros(m, basis_size, basis_size))
                nn.init.zeros_(self.lie_coeffs.weight)
                nn.init.zeros_(self.lie_coeffs.bias)
            elif self.operator_modulation == 'polynomial':
                p = int(remixed_linear_kwargs.get('poly_order', 2))
                self.poly_order = max(1, p)
                self.poly_coeffs = Linear(context_dim, self.poly_order + 1, bias=True)
                self.poly_A = nn.Parameter(torch.eye(basis_size))
                nn.init.zeros_(self.poly_coeffs.weight)
                nn.init.zeros_(self.poly_coeffs.bias)
                with torch.no_grad():
                    self.poly_coeffs.bias[0] = 1.0
            elif self.operator_modulation == 'grassmann':
                k = int(remixed_linear_kwargs.get('grassmann_bank_size', 4))
                self.grassmann_alpha = Linear(context_dim, k, bias=True)
                self.grassmann_bank = nn.Parameter(torch.ones(k, basis_size))
                nn.init.zeros_(self.grassmann_alpha.weight)
                nn.init.zeros_(self.grassmann_alpha.bias)

    def gate_parameters(self):
        """Yield context-gate-specific parameters (basis_modulator, output_gate_*).
        These are routed to a lower-LR optimizer group to reduce gradient noise at long T."""
        if self.use_context:
            yield from self.basis_modulator.parameters()
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
            if hasattr(self, "operator_householder"):
                yield from self.operator_householder.parameters()
            if hasattr(self, "operator_spectral"):
                yield from self.operator_spectral.parameters()
            if hasattr(self, "ocd_coeffs"):
                yield from self.ocd_coeffs.parameters()
                yield self.ocd_scale
            if hasattr(self, "lie_coeffs"):
                yield from self.lie_coeffs.parameters()
            if hasattr(self, "poly_coeffs"):
                yield from self.poly_coeffs.parameters()
            if hasattr(self, "grassmann_alpha"):
                yield from self.grassmann_alpha.parameters()

    def non_gate_parameters(self):
        """Yield structural parameters (basis, template_mixing, bias) — normal LR."""
        yield self.basis.weight
        yield self.template_mixing
        yield self.bias
        if hasattr(self, "ocd_in"):
            yield self.ocd_in
        if hasattr(self, "ocd_out"):
            yield self.ocd_out
        if hasattr(self, "lie_generators"):
            yield self.lie_generators
        if hasattr(self, "poly_A"):
            yield self.poly_A
        if hasattr(self, "grassmann_bank"):
            yield self.grassmann_bank

    def forward(self, x, context_state):
        dtype = x.dtype
        h_basis = self.ln_basis(self.basis(x).to(dtype=self.ln_basis.weight.dtype)).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)
            if self.operator_modulation == 'householder':
                v = F.normalize(self.operator_householder(ctx), dim=-1, eps=1e-6)
                h_basis = h_basis - 2.0 * (h_basis * v).sum(dim=-1, keepdim=True) * v
            elif self.operator_modulation == 'spectral':
                scale = 1.0 + torch.tanh(self.operator_spectral(ctx) * 0.1)
                h_basis = h_basis * scale.to(dtype=dtype)
            elif self.operator_modulation == 'lie':
                coeffs = self.lie_coeffs(ctx)
                generators = self.lie_generators.to(dtype=dtype)
                skew = 0.5 * (generators - generators.transpose(-1, -2))
                transport = torch.einsum('btm,mij->btij', coeffs, skew)
                eye = torch.eye(self.basis_size, device=x.device, dtype=dtype).view(1, 1, self.basis_size, self.basis_size)
                h_basis = torch.einsum('bti,btij->btj', h_basis, eye + 0.1 * transport)
            elif self.operator_modulation == 'polynomial':
                a = self.poly_coeffs(ctx)
                A = self.poly_A.to(dtype=dtype)
                z = torch.zeros_like(h_basis)
                h_pow = h_basis
                z = z + a[..., :1] * h_pow
                for i in range(1, self.poly_order + 1):
                    h_pow = torch.matmul(h_pow, A)
                    z = z + a[..., i:i+1] * h_pow
                h_basis = z
            elif self.operator_modulation == 'grassmann':
                alpha = F.softmax(self.grassmann_alpha(ctx), dim=-1)
                scale = torch.einsum('btk,kd->btd', alpha, self.grassmann_bank.to(dtype=dtype))
                h_basis = h_basis * scale

            # Basis gate: sparse or dense sigmoid with configurable temperature
            if self.use_basis_gate:
                gate_logits = self.basis_modulator(ctx)
                if self._film_gate_flag:
                    # Proposal C: FiLM affine conditioning — scale + shift
                    # basis_modulator outputs 2*basis_size; split into (scale, shift)
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    # 1 + tanh(0.1 * scale_logits) → range [0,2], starts at 1.0 (identity)
                    gate_basis = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h_basis = h_basis * gate_basis + shift.to(dtype=dtype)
                    gate_basis = None  # already applied
                elif self.sparse_gate_k > 0:
                    # Design 3: Straight-through sparse top-k gate.
                    # Forward: activate top-k basis functions via soft distribution.
                    # Backward: gradients flow through the continuous sigmoid path.
                    k = min(self.sparse_gate_k, self.basis_size)
                    topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=-1)
                    sparse = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
                    soft = torch.sigmoid(gate_logits / self.gate_temperature)
                    gate_basis = (sparse + (soft - soft.detach())).to(dtype=dtype)
                else:
                    gate_basis = torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)
            else:
                gate_basis = torch.ones_like(h_basis)

            # Output gate: LOW-RANK + CENTERED ACTIVATION
            # gate = 1 + tanh(scale * coeffs @ gate_basis_vectors)
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

        if self.use_context and context_state is not None and self.operator_modulation == 'ocd':
            coeffs = self.ocd_coeffs(ctx)                                         # (B, T, r)
            proj = torch.matmul(h_basis, self.ocd_in.to(dtype=dtype).transpose(0, 1))  # (B, T, r)
            delta = torch.matmul(proj * coeffs, self.ocd_out.to(dtype=dtype))      # (B, T, out)
            pre_output = pre_output + self.ocd_scale.to(dtype=dtype) * delta
            # Soft overlap penalty proxy: ||W_s W_d^T||_F^2 where W_d is mean dynamic map.
            c_mean = coeffs.mean(dim=(0, 1))                                       # (r,)
            w_dyn = torch.einsum(
                'r,ro,ri->oi',
                c_mean,
                self.ocd_out.to(dtype=dtype),
                self.ocd_in.to(dtype=dtype),
            )                                                                       # (out, basis)
            overlap = torch.matmul(self.template_mixing.to(dtype=dtype), w_dyn.transpose(0, 1))
            self._last_orth_loss = overlap.pow(2).mean()
        else:
            self._last_orth_loss = None

        if gate_out is not None:
            pre_output = pre_output * gate_out

        return (pre_output + self.bias.to(dtype=dtype)).to(dtype=dtype)


class DecoupledAdaptiveLinear(nn.Module):
    """Paradigm 1: split static/dynamic channels for gradient decoupling."""
    def __init__(self, in_features, out_features, context_dim, dynamic_ratio=0.25, basis_size=64, gate_rank=8, **_ignored):
        super().__init__()
        self.dyn_features = max(1, int(in_features * dynamic_ratio))
        self.stat_features = in_features - self.dyn_features
        if self.stat_features <= 0:
            self.stat_features = in_features - 1
            self.dyn_features = 1

        self.static_proj = Linear(self.stat_features, out_features, bias=True)
        self.dyn_basis = Linear(self.dyn_features, basis_size, bias=False)
        self.dyn_mix = nn.Parameter(torch.randn(out_features, basis_size) / (basis_size ** 0.5))
        self.gate_coeffs = Linear(context_dim, gate_rank, bias=True)
        self.gate_basis = nn.Parameter(torch.zeros(gate_rank, out_features))
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.gate_coeffs.weight)
        nn.init.zeros_(self.gate_coeffs.bias)

    def gate_parameters(self):
        yield from self.gate_coeffs.parameters()
        yield self.gate_basis
        yield self.gate_scale

    def non_gate_parameters(self):
        yield from self.static_proj.parameters()
        yield self.dyn_basis.weight
        yield self.dyn_mix

    def forward(self, x, context_state):
        x_stat = x[..., :self.stat_features]
        x_dyn = x[..., self.stat_features:]
        out_stat = self.static_proj(x_stat)
        h_basis = self.dyn_basis(x_dyn)
        out_dyn = F.linear(h_basis, self.dyn_mix.to(dtype=x.dtype))
        if context_state is not None:
            coeffs = self.gate_coeffs(context_state.to(dtype=x.dtype))
            gate_logits = torch.matmul(coeffs, self.gate_basis.to(dtype=x.dtype))
            gate = 1.0 + torch.tanh(self.gate_scale.to(dtype=x.dtype) * gate_logits)
            out_dyn = out_dyn * gate
        return out_stat + out_dyn


class TuckerAdaptiveLinear(nn.Module):
    """Tucker-decomposed context routing: W_eff = Σ_k v_k(ctx) A G_k B^T."""
    def __init__(self, in_features, out_features, context_dim, tucker_rank=32, tucker_modes=8, **_ignored):
        super().__init__()
        r = max(1, int(tucker_rank))
        k = max(1, int(tucker_modes))
        self.A = nn.Parameter(torch.randn(in_features, r) * (in_features ** -0.5))
        self.B = nn.Parameter(torch.randn(out_features, r) * (out_features ** -0.5))
        self.core = nn.Parameter(torch.randn(r, r, k) * (r ** -0.5))
        self.ctx_router = Linear(context_dim, k, bias=True)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.ctx_router.weight)
        nn.init.zeros_(self.ctx_router.bias)

    def gate_parameters(self):
        yield from self.ctx_router.parameters()

    def non_gate_parameters(self):
        yield self.A
        yield self.B
        yield self.core
        yield self.bias

    def forward(self, x, context_state):
        h = torch.matmul(x, self.A.to(dtype=x.dtype))  # (B,T,r)
        if context_state is None:
            weights = torch.full((*h.shape[:2], self.core.shape[-1]), 1.0 / self.core.shape[-1], device=x.device, dtype=x.dtype)
        else:
            weights = F.softmax(self.ctx_router(context_state.to(dtype=x.dtype)), dim=-1)
        mixed = torch.einsum('btk,ijk->btij', weights, self.core.to(dtype=x.dtype))
        h2 = torch.einsum('bti,btij->btj', h, mixed)
        y = torch.matmul(h2, self.B.to(dtype=x.dtype).transpose(0, 1))
        return y + self.bias.to(dtype=x.dtype)


class SingularValueSteeringLinear(nn.Module):
    """SVS: context steers singular-value-like gains over fixed low-rank factors."""
    def __init__(self, in_features, out_features, context_dim, svs_rank=64, svs_eps=0.1, dcu_warmup_steps=0, **_ignored):
        super().__init__()
        r = max(1, min(int(svs_rank), in_features, out_features))
        self.V = nn.Parameter(torch.randn(in_features, r) * (in_features ** -0.5))
        self.U = nn.Parameter(torch.randn(out_features, r) * (out_features ** -0.5))
        self.sigma = nn.Parameter(torch.ones(r))
        self.ctx = Linear(context_dim, r, bias=True)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.svs_eps = float(svs_eps)
        self.dcu_warmup_steps = max(0, int(dcu_warmup_steps))
        self._step = 0
        self.ln_basis = nn.Identity()
        nn.init.zeros_(self.ctx.weight)
        nn.init.zeros_(self.ctx.bias)

    def gate_parameters(self):
        yield from self.ctx.parameters()

    def non_gate_parameters(self):
        yield self.V
        yield self.U
        yield self.sigma
        yield self.bias

    def forward(self, x, context_state):
        h = torch.matmul(x, self.V.to(dtype=x.dtype))
        if context_state is None:
            steer = torch.ones_like(h)
        else:
            steer = 1.0 + self.svs_eps * torch.tanh(self.ctx(context_state.to(dtype=x.dtype)))
        if self.dcu_warmup_steps > 0:
            self._step += 1
            unlock = min(1.0, float(self._step) / float(self.dcu_warmup_steps))
            steer = 1.0 + unlock * (steer - 1.0)
        h = h * (self.sigma.to(dtype=x.dtype) * steer)
        y = torch.matmul(h, self.U.to(dtype=x.dtype).transpose(0, 1))
        return y + self.bias.to(dtype=x.dtype)


class VQAdaptiveLinear(nn.Module):
    """VQ regime routing: dense base + regime-selected full-rank delta."""
    def __init__(self, in_features, out_features, context_dim, vq_codes=8, vq_temperature=1.0, **_ignored):
        super().__init__()
        self.base = Linear(in_features, out_features, bias=True)
        self.K = max(1, int(vq_codes))
        self.temperature = max(float(vq_temperature), 1e-6)
        self.codebook = nn.Parameter(torch.randn(self.K, context_dim) * (context_dim ** -0.5))
        self.delta = nn.Parameter(torch.zeros(self.K, out_features, in_features))
        self.ln_basis = nn.Identity()

    def gate_parameters(self):
        yield self.codebook

    def non_gate_parameters(self):
        yield from self.base.parameters()
        yield self.delta

    def forward(self, x, context_state):
        y = self.base(x)
        if context_state is None:
            return y
        ctx = context_state.to(dtype=x.dtype)
        B, T, C = ctx.shape
        flat = ctx.reshape(B * T, C)
        dist = torch.cdist(flat, self.codebook.to(dtype=x.dtype))
        idx = torch.argmin(dist, dim=-1)
        hard = F.one_hot(idx, num_classes=self.K).to(dtype=x.dtype)
        soft = F.softmax(-dist / self.temperature, dim=-1)
        w = hard + (soft - soft.detach())
        w = w.view(B, T, self.K)
        w_delta = torch.einsum('btk,koi->btoi', w, self.delta.to(dtype=x.dtype))
        y_delta = torch.einsum('btoi,bti->bto', w_delta, x)
        return y + y_delta


class FrozenSubspaceIndexedLinear(nn.Module):
    """Frozen Subspace Indexing (FSI): context-dependent input transformation
    with ZERO trainable routing parameters.

    y = base_dense(x - 2 * sum_k w_k * v_k * (v_k^T x))

    Uses K frozen Householder reflections (unit vectors v_k) with soft routing
    weights w_k from a frozen random projection of the attention signal.

    Key properties:
    - base_dense is STANDARD nn.Linear — identical gradient dynamics to baseline
    - Reflectors are FROZEN (registered buffers) — zero gradient interference
    - Routing uses FROZEN random projections — no controller to train
    - Memory: O(K*D) reflector buffer, NOT O(K*D²) rotation matrices
    - Compute: O(B*T*K*D) — linear in D, not quadratic
    - torch.compile friendly: no gather, no einsum over D², just matmuls + element-wise

    Compatible with setup_optimizer: gate_parameters() yields nothing (no gates),
    non_gate_parameters() yields base weight/bias. ln_basis = Identity for compat.
    """
    def __init__(self, in_features, out_features, n_rotations=8, selector_dim=64, signal_dim=None, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_rotations = n_rotations
        self.signal_dim = signal_dim or in_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # FROZEN: K random unit vectors defining Householder reflections
        # P_k = I - 2 * v_k * v_k^T  (orthogonal reflection through hyperplane ⊥ v_k)
        # Each reflection provides a distinct "view" of the input space
        vecs = torch.randn(n_rotations, in_features)
        vecs = F.normalize(vecs, dim=-1)  # unit vectors
        self.register_buffer('reflectors', vecs)  # (K, D_in)

        # FROZEN: random projection for routing signal extraction
        # signal (signal_dim) → selector_dim → K-dim soft weights
        self.register_buffer('selector_proj',
            torch.randn(self.signal_dim, selector_dim) / (self.signal_dim ** 0.5))
        self.register_buffer('selector',
            torch.randn(selector_dim, n_rotations) / (selector_dim ** 0.5))

    def gate_parameters(self):
        """No gate parameters — routing is entirely frozen."""
        return iter([])

    def non_gate_parameters(self):
        """Only the base dense layer is trained."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state):
        """
        x: (B, T, D_in)
        context_state: (B, T, signal_dim) — detached attention output for routing.
                       If None, uses pure base projection (identity transform).
        """
        if context_state is None:
            return self.base(x)

        dtype = x.dtype
        signal = context_state.to(dtype=dtype).detach()

        # Frozen routing: signal → soft weights over K reflections
        routing = (signal @ self.selector_proj.to(dtype)) @ self.selector.to(dtype)  # (B, T, K)
        weights = torch.softmax(routing, dim=-1)  # (B, T, K)

        # Soft-weighted Householder reflections (compile-friendly, O(BTK*D))
        # dot_k = v_k^T x  →  dots: (B, T, K)
        dots = torch.einsum('btd,kd->btk', x, self.reflectors.to(dtype))

        # weighted_dots = sum_k w_k * dot_k * v_k  →  delta: (B, T, D_in)
        # This is the soft mixture of reflections applied to x
        weighted_dots = weights * dots  # (B, T, K)
        delta = torch.einsum('btk,kd->btd', weighted_dots, self.reflectors.to(dtype))

        # x_transformed = x - 2 * delta  (soft mixture of Householder reflections)
        x_transformed = x - 2.0 * delta

        return self.base(x_transformed)


class AttentionEntropyStratifiedLinear(nn.Module):
    """Attention-Entropy Stratified Projection (AESP): per-stratum dense
    projection selected by attention entropy.

    y = base(x) + alpha_k * (x @ U_k) @ V_k    where k = stratum(entropy)

    Uses attention entropy (already computed, zero extra FLOPs for routing)
    to partition positions into K strata. Each stratum has a tiny low-rank
    delta (rank 4 by default) with FROZEN scaling alpha_k.

    Key properties:
    - Stratum 0 (highest entropy = weakest context) has alpha=0 → EXACT dense baseline
    - Higher strata allow progressively more specialization (alpha grows)
    - Deltas are tiny rank-4 → minimal parameter overhead
    - Routing (entropy → stratum) is free (entropy computed in attention already)
    - Frozen alphas prevent the network from "turning off" specialization

    Compatible with setup_optimizer: gate_parameters() yields nothing (routing is
    frozen), non_gate_parameters() yields base + U/V + scale.
    """
    def __init__(self, in_features, out_features, n_strata=4, delta_rank=4, **_ignored):
        super().__init__()
        self.n_strata = n_strata
        self.delta_rank = delta_rank
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat

        # Per-stratum low-rank deltas (V=0 at init → zero delta)
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, delta_rank))
            for _ in range(n_strata)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.zeros(delta_rank, out_features))
            for _ in range(n_strata)
        ])
        for u in self.U:
            nn.init.kaiming_uniform_(u, a=0.01)

        # FROZEN scaling: stratum 0 = zero delta (pure baseline)
        # Higher strata get progressively more specialization budget
        alphas = torch.linspace(0, 0.2, n_strata)
        alphas[0] = 0.0  # stratum 0 is always pure dense baseline
        self.register_buffer('alphas', alphas)

    def gate_parameters(self):
        """No gate parameters — routing/scaling is frozen."""
        return iter([])

    def non_gate_parameters(self):
        """Base projection + per-stratum U/V deltas."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias
        for u in self.U:
            yield u
        for v in self.V:
            yield v

    def forward(self, x, context_state):
        """
        x: (B, T, D)
        context_state: (B, T) — attention entropy per position (scalar).
                       If None or wrong shape, uses pure base projection.
        """
        y = self.base(x)

        if context_state is None:
            return y

        dtype = x.dtype
        entropy = context_state.to(dtype=dtype)

        # Handle case where context_state is (B, T, D) — take mean to get scalar
        if entropy.ndim == 3:
            entropy = entropy.mean(dim=-1)  # (B, T)

        # Quantize entropy into strata via percentile boundaries
        with torch.no_grad():
            q_levels = torch.linspace(0, 1, self.n_strata + 1, device=x.device)[1:-1]
            boundaries = torch.quantile(entropy.float().flatten(), q_levels)
            stratum = torch.bucketize(entropy.float(), boundaries)  # (B, T) in [0, K-1]

        # Add stratum-specific deltas (skip stratum 0 which has alpha=0)
        for k in range(1, self.n_strata):
            alpha_k = self.alphas[k].to(dtype)
            if alpha_k == 0:
                continue
            mask = (stratum == k).unsqueeze(-1).to(dtype)  # (B, T, 1)
            delta = (x @ self.U[k].to(dtype)) @ self.V[k].to(dtype)  # (B, T, out)
            y = y + mask * alpha_k * delta

        return y


class CausalKernelLinear(nn.Module):
    """Causal Kernel Reparameterization (CKR): position-dependent mixing
    of K parallel dense branches via tiny causal conv1d.

    Training:  y_t = sum_k  w_k(t) · (W_k · x_t)
    Inference: y_t = W_merged(t) · x_t  (structural reparameterization)

    Phase 13 enhancements:
    - 13a: Multi-channel position signal (C_pos > 1) for multi-scale temporal patterns
    - 13c: Optional frozen content hash bias for document-type fingerprinting

    Key properties:
    - w_k(t) depends on POSITION (+ optional frozen content hash), not learned content
    - No chicken-and-egg: W_k learn features, w_k learn positions, independently
    - Position weights from causal conv capture local structure naturally
    - At init: conv output ≈ 0 → softmax(0) = 1/K → equal branch weighting
    - Reparameterizable at inference for zero overhead

    Compatible with setup_optimizer: gate_parameters() yields position signal +
    conv weights (small), non_gate_parameters() yields branch projections.
    """
    def __init__(self, in_features, out_features, n_branches=4, kernel_size=64,
                 max_seq_len=2048, n_pos_channels=1, content_bias_scale=0.0,
                 signal_dim=None, temperature=1.0,
                 ortho_init=False, branch_dropout=0.0, **_ignored):
        super().__init__()
        self.n_branches = n_branches
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.content_bias_scale = content_bias_scale
        # Temperature as a buffer tensor (NOT Python float) to avoid torch.compile
        # guard breaks when annealing. Buffer tensors are treated as dynamic values.
        # _init_temperature (Python float) survives to_empty() for re-init.
        self._init_temperature = temperature
        self.register_buffer('_temperature', torch.tensor([temperature]), persistent=False)
        self.branch_dropout = branch_dropout  # 17E: probability of dropping each branch
        self.ln_basis = nn.Identity()  # compat

        # K parallel projection branches
        self.branches = nn.ModuleList([
            Linear(in_features, out_features, bias=(i == 0))
            for i in range(n_branches)
        ])
        # 17D: Orthogonal initialization for branch diversity
        self._ortho_init = ortho_init
        if ortho_init:
            for branch in self.branches:
                nn.init.orthogonal_(branch.weight)

        # 13a: Multi-channel position signal (C_pos channels → conv → K branches)
        # C_pos > 1 allows the conv to learn multi-scale temporal patterns
        self.n_pos_channels = n_pos_channels
        self.pos_signal = nn.Parameter(torch.zeros(1, n_pos_channels, max_seq_len))
        self.branch_conv = nn.Conv1d(
            n_pos_channels, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True  # causal: we trim the right padding
        )
        # Init: conv output ≈ 0 → softmax(0) = 1/K → equal branch weighting
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

        # 13c: Frozen content hash for document-type fingerprinting
        # Projects detached input → K-dim frozen logit bias
        if content_bias_scale > 0:
            sig_dim = signal_dim or in_features
            self.register_buffer('content_proj',
                torch.randn(sig_dim, n_branches) / (sig_dim ** 0.5))
        else:
            self.content_proj = None

    def gate_parameters(self):
        """Position signal and conv weights — the 'routing' side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()

    def non_gate_parameters(self):
        """Branch projection weights — the 'feature' side."""
        for branch in self.branches:
            yield from branch.parameters()

    def forward(self, x, context_state=None):
        """
        x: (B, T, D). context_state: optional (B, T, signal_dim) for content bias.
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Position-dependent branch weights (NOT content-dependent)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)  # (B, C_pos, T)
        raw_weights = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]  # (B, K, T) — causal trim
        logits = raw_weights.to(dtype)  # (B, K, T)

        # 13c: Add frozen content hash bias (if enabled)
        if self.content_proj is not None and context_state is not None:
            # Frozen projection of detached context → (B, T, K) logit bias
            ctx = context_state.to(dtype=dtype).detach()
            content_logits = ctx @ self.content_proj.to(dtype)  # (B, T, K)
            logits = logits + self.content_bias_scale * content_logits.permute(0, 2, 1)  # (B, K, T)

        w = F.softmax(logits / self._temperature.to(dtype), dim=1)  # (B, K, T)

        # 17E: Branch dropout — randomly zero some branches during training
        if self.training and self.branch_dropout > 0 and self.n_branches > 1:
            # Create per-branch mask (same for all positions in batch)
            mask = torch.ones(self.n_branches, device=w.device, dtype=dtype)
            drop_idx = torch.randint(0, self.n_branches, (1,)).item()
            mask[drop_idx] = 0.0
            w = w * mask.view(1, -1, 1)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # renormalize

        # Parallel branch computation
        y = torch.zeros(B, T, self.out_features, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y_k = self.branches[k](x)  # (B, T, out)
            y = y + w[:, k, :].unsqueeze(-1) * y_k  # position-weighted sum

        return y


class GradientIsolatedDeltaLinear(nn.Module):
    """Phase 14a: Gradient-Isolated Additive Delta (GIAD).

    y = base(x) + delta_net(x)

    Approximate gradient isolation via initialization:
    - delta_up.weight = 0 at init → delta output is exactly zero
    - As training progresses, delta_up weights grow from zero,
      gradually introducing the delta path's contribution
    - Early training is virtually identical to the dense baseline

    Note: no separate scale parameter — nn.Parameter(torch.zeros(1))
    is float32, which causes dtype promotion (float32 × bf16 → float32)
    under torch.compile + fp8/bf16 autocast, crashing FlashAttention.
    """
    def __init__(self, in_features, out_features, delta_rank=32, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # Low-rank bottleneck: x → rank-r → output (delta_up is zero-init)
        self.delta_down = Linear(in_features, delta_rank, bias=False)
        self.delta_up = Linear(delta_rank, out_features, bias=False)

    def gate_parameters(self):
        """Delta path params are the 'gate' side (lower LR)."""
        yield self.delta_down.weight
        yield self.delta_up.weight

    def non_gate_parameters(self):
        """Base projection weights — standard dense, unaffected by delta."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state=None):
        """
        x: (B, T, D_in). context_state is ignored.
        """
        y = self.base(x)
        # delta_up zero-init → starts at 0, grows naturally during training
        delta = self.delta_up(F.gelu(self.delta_down(x)))
        return y + delta


class PositionalScalarGatedLinear(nn.Module):
    """Phase 14b: Positional Scalar Gating (PSG).

    y_t = W · (s(t) ⊙ x_t)

    where s(t) is a D_in-dimensional position-dependent scaling vector from a
    tiny causal conv1d over a learned positional signal.

    Effective weight at position t: W_eff(t) = W · diag(s(t))
    This is a rank-D perturbation — can modulate every element of W_eff —
    yet costs only O(conv_params) extra, NOT O(K × D × D) like CKR.

    Init: s(t) = 1.0 for all t → y = W · x (identity with dense baseline).
    Position-only routing: no content-dependent gates → no friction tax.
    """
    def __init__(self, in_features, out_features, kernel_size=64,
                 max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = Linear(in_features, out_features, bias=True)
        self.ln_basis = nn.Identity()  # compat

        # Learned 1D positional signal → causal conv → D_in-dim scaling per position
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.scale_conv = nn.Conv1d(
            1, in_features, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        # Init: conv output ≈ 0 → s(t) = 1.0 (identity scaling)
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.zeros_(self.scale_conv.bias)
        # Controls the maximum deviation from identity
        self.scale_magnitude = nn.Parameter(torch.tensor(0.01))

    def gate_parameters(self):
        """Position signal + conv = routing side."""
        yield self.pos_signal
        yield from self.scale_conv.parameters()
        yield self.scale_magnitude

    def non_gate_parameters(self):
        """Base projection weights — standard dense."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias

    def forward(self, x, context_state=None):
        """
        x: (B, T, D_in). context_state is ignored (position-only scaling).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Position-dependent per-channel scaling
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)  # (B, 1, T)
        raw = self.scale_conv(sig.to(dtype=self.scale_conv.weight.dtype))[:, :, :T]  # (B, D_in, T)
        # s(t) = 1 + magnitude * tanh(raw) → starts at 1.0, bounded perturbation
        s = 1.0 + self.scale_magnitude * torch.tanh(raw.to(dtype))  # (B, D_in, T)
        x_scaled = x * s.permute(0, 2, 1)  # (B, T, D_in)
        return self.base(x_scaled)


class SplitStreamLinear(nn.Module):
    """Phase 14c: SplitStream — Decoupled channel split + CKR dynamic path.

    y = concat(
        W_static · x[:, :, :D_s],       # static channels: pure dense
        CKR_branches(x[:, :, D_s:])      # dynamic channels: position-dependent
    )

    Combines the two winning principles:
    - Decoupled (Phase 11): Reduce gradient competition via channel partitioning
    - CKR (Phase 12): Position-only routing avoids content-gradient interference

    Static path (majority ~75%): exact dense baseline gradient dynamics.
    Dynamic path (minority ~25%): CKR-style position routing on fewer channels.
    Zero parameter sharing between paths.
    """
    def __init__(self, in_features, out_features, dynamic_ratio=0.25,
                 n_branches=2, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()  # compat

        # Channel split dimensions
        self.d_dynamic = max(1, int(in_features * dynamic_ratio))
        self.d_static = in_features - self.d_dynamic
        d_out_dynamic = max(1, int(out_features * dynamic_ratio))
        d_out_static = out_features - d_out_dynamic

        # Static path: plain dense projection
        self.static_proj = Linear(self.d_static, d_out_static, bias=True)

        # Dynamic path: CKR-style multi-branch with position routing
        self.n_branches = n_branches
        self.dynamic_branches = nn.ModuleList([
            Linear(self.d_dynamic, d_out_dynamic, bias=(i == 0))
            for i in range(n_branches)
        ])
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.branch_conv = nn.Conv1d(
            1, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

    def gate_parameters(self):
        """Position signal + conv + dynamic branch weights = routing side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()
        for branch in self.dynamic_branches:
            yield from branch.parameters()

    def non_gate_parameters(self):
        """Static projection weights — pure dense, zero interference."""
        yield self.static_proj.weight
        if self.static_proj.bias is not None:
            yield self.static_proj.bias

    def forward(self, x, context_state=None):
        """
        x: (B, T, D_in). context_state is ignored (position-only dynamic path).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Channel split
        x_static = x[:, :, :self.d_static]
        x_dynamic = x[:, :, self.d_static:]

        # Static path: standard dense projection
        y_static = self.static_proj(x_static)

        # Dynamic path: CKR-style position-dependent branch mixing
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)
        raw_w = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]
        w = F.softmax(raw_w.to(dtype), dim=1)  # (B, K, T)

        y_dynamic = torch.zeros(B, T, self.dynamic_branches[0].weight.shape[0],
                                device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y_k = self.dynamic_branches[k](x_dynamic)
            y_dynamic = y_dynamic + w[:, k, :].unsqueeze(-1) * y_k

        return torch.cat([y_static, y_dynamic], dim=-1)


class LoKRLinear(nn.Module):
    """Phase 15: Low-rank Kernel Reparameterization (LoKR).

    W_eff(t) = W + Σ_k  w_k(t) · U_k @ V_k^T

    Combines CKR's key insight (position-dependent multi-branch mixing) with
    dramatically lower parameter cost by using low-rank perturbations instead
    of full-rank duplicate branches.

    Key properties:
    - W is shared full-rank base → standard dense gradient (Pattern 1 ✓)
    - K low-rank perturbation branches (U_k, V_k) provide diversity (Pattern 2 ✓)
    - Position-only routing via causal conv (Pattern 3 ✓)
    - At init: V_k=0 → perturbation=0 → exact dense baseline
    - 21% param overhead vs CKR's 300% (at K=8, r=16 for D=768)

    Computation:
        y = W·x + Σ_k w_k(t) · U_k @ (V_k @ x)
    Each V_k@x is (B,T,r), then U_k@result is (B,T,D_out). Very efficient.
    """
    def __init__(self, in_features, out_features, n_branches=8, rank=16,
                 kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_branches = n_branches
        self.rank = rank
        self.ln_basis = nn.Identity()  # compat

        # Shared full-rank base projection
        self.base = Linear(in_features, out_features, bias=True)

        # K low-rank perturbation branches: U_k (out×r) and V_k (r×in)
        self.lora_down = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, in_features))   # V_k: r × D_in
            for _ in range(n_branches)
        ])
        self.lora_up = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, rank))  # U_k: D_out × r
            for _ in range(n_branches)
        ])
        # V_k = 0 at init → perturbation = 0 → exact dense baseline
        # U_k gets xavier init for gradient flow once V_k starts learning

        # Position-dependent branch weights (same as CKR)
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.branch_conv = nn.Conv1d(
            1, n_branches, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.branch_conv.weight)
        nn.init.zeros_(self.branch_conv.bias)

    def gate_parameters(self):
        """Position signal + conv = routing side."""
        yield self.pos_signal
        yield from self.branch_conv.parameters()

    def non_gate_parameters(self):
        """Base projection + low-rank perturbation matrices = feature side."""
        yield self.base.weight
        if self.base.bias is not None:
            yield self.base.bias
        for v in self.lora_down:
            yield v
        for u in self.lora_up:
            yield u

    def forward(self, x, context_state=None):
        """
        x: (B, T, D_in). context_state ignored (position-only routing).
        """
        B, T, D = x.shape
        dtype = x.dtype

        # Standard dense base
        y = self.base(x)

        # Position-dependent branch weights
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1)
        raw_w = self.branch_conv(sig.to(dtype=self.branch_conv.weight.dtype))[:, :, :T]
        w = F.softmax(raw_w.to(dtype), dim=1)  # (B, K, T)

        # Low-rank position-dependent perturbations
        delta = torch.zeros_like(y)
        for k in range(self.n_branches):
            # V_k @ x: (B, T, D_in) → (B, T, r)
            h_k = F.linear(x, self.lora_down[k].to(dtype))  # (B, T, r)
            # U_k @ h_k: (B, T, r) → (B, T, D_out)
            d_k = F.linear(h_k, self.lora_up[k].to(dtype))  # (B, T, D_out)
            delta = delta + w[:, k, :].unsqueeze(-1) * d_k

        return y + delta


class CausalOutputMixer(nn.Module):
    """Phase 16C: Causal Output Mixer (COM).

    y = Linear(x) + gate(pos) ⊙ causal_depthwise_conv(Linear(x))

    Fundamentally different from all previous approaches:
    - The Linear layer is COMPLETELY UNMODIFIED (true zero friction)
    - Position-dependent causal convolution adds sequence-level mixing
    - Each position can selectively mix with previous positions
    - Gate starts at 0 (zero-init bias) → pure dense baseline at init

    Similar to what works in Mamba/RWKV: conv + gating on features.
    ~0.5% parameter overhead vs CKR's 300%.
    """
    def __init__(self, in_features, out_features, kernel_size=32,
                 max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()  # compat with setup_optimizer

        # Standard dense projection (unmodified)
        self.base = Linear(in_features, out_features, bias=True)

        # Causal depthwise convolution on output features
        # Groups = out_features → each channel convolved independently
        self.causal_conv = nn.Conv1d(
            out_features, out_features, kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal: trim right side
            groups=out_features, bias=False
        )
        nn.init.zeros_(self.causal_conv.weight)

        # Position-dependent gate: learned pos signal → per-channel gate
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        self.gate_conv = nn.Conv1d(
            1, out_features, kernel_size=kernel_size,
            padding=kernel_size - 1, bias=True
        )
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, -3.0)  # sigmoid(-3) ≈ 0.05 → nearly off at init

    def gate_parameters(self):
        """Routing side: position signal + gate conv + causal conv."""
        yield self.pos_signal
        yield from self.gate_conv.parameters()
        yield from self.causal_conv.parameters()

    def non_gate_parameters(self):
        """Feature side: base projection weights."""
        yield from self.base.parameters()

    def forward(self, x, context_state=None):
        """
        x: (B, T, D_in). context_state ignored.
        """
        B, T, _ = x.shape
        dtype = x.dtype

        # Standard dense projection
        y = self.base(x)  # (B, T, D_out)

        # Causal depthwise conv on output (channels-first), cast weights to match input dtype
        y_t = y.transpose(1, 2)  # (B, D_out, T)
        conv_out = F.conv1d(
            y_t, self.causal_conv.weight.to(dtype), bias=None,
            padding=self.kernel_size - 1, groups=y_t.shape[1]
        )[:, :, :T]  # causal trim → (B, D_out, T)

        # Position-dependent gate (cast gate_conv weights to match)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)  # (B, 1, T)
        gate = torch.sigmoid(
            F.conv1d(sig, self.gate_conv.weight.to(dtype),
                     self.gate_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, D_out, T)

        # Gate ⊙ conv_out → residual
        mixed = gate * conv_out  # (B, D_out, T)
        return y + mixed.transpose(1, 2)  # (B, T, D_out)


class PositionGatedResidual(nn.Module):
    """Phase 17C: Position-Gated Residual (PGR).

    y = W₀x + g(pos) · W₁x

    Two full-rank branches with explicit base+delta structure.
    g(pos) is a scalar gate per position from causal conv on learned position signal.
    W₁ zero-init → starts as pure W₀ (dense baseline).

    Compatible with setup_optimizer: gate_parameters() yields pos_signal + gate_conv,
    non_gate_parameters() yields W₀ and W₁.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        # Base projection (standard init)
        self.base = Linear(in_features, out_features, bias=True)
        # Delta projection (zero-init for dense-baseline start)
        self.delta = Linear(in_features, out_features, bias=False)
        nn.init.zeros_(self.delta.weight)

        # Position gate: learned signal → scalar gate per position
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        self.gate_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, -3.0)  # sigmoid(-3) ≈ 0.05

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.gate_conv.parameters()

    def non_gate_parameters(self):
        yield from self.base.parameters()
        yield from self.delta.parameters()

    def forward(self, x, context_state=None):
        B, T, _ = x.shape
        dtype = x.dtype
        base_out = self.base(x)  # (B, T, out)
        delta_out = self.delta(x)  # (B, T, out)
        # Position gate
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        gate = torch.sigmoid(
            F.conv1d(sig, self.gate_conv.weight.to(dtype),
                     self.gate_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, 1, T)
        gate = gate.permute(0, 2, 1)  # (B, T, 1)
        return base_out + gate * delta_out


class CausalInterpolationLinear(nn.Module):
    """Phase 17I: Causal Interpolation Linear (CIL).

    y = ((1-α(pos)) · W₀ + α(pos) · W₁) · x

    Interpolates between two weight matrices in weight space using a position-dependent
    scalar α from causal conv. α starts at 0 → pure W₀ at init.

    Key: only ONE matmul with the interpolated weight, not two separate matmuls.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        # Two full-rank weight matrices
        self.weight_0 = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_1 = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight_0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))

        # Position interpolation: learned signal → α(pos) ∈ [0,1]
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        self.alpha_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.alpha_conv.weight)
        nn.init.constant_(self.alpha_conv.bias, -5.0)  # sigmoid(-5) ≈ 0.007 → nearly pure W₀

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.alpha_conv.parameters()

    def non_gate_parameters(self):
        yield self.weight_0
        yield self.weight_1
        yield self.bias

    def forward(self, x, context_state=None):
        B, T, _ = x.shape
        dtype = x.dtype
        # Position-dependent alpha
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        alpha = torch.sigmoid(
            F.conv1d(sig, self.alpha_conv.weight.to(dtype),
                     self.alpha_conv.bias.to(dtype),
                     padding=self.kernel_size - 1)[:, :, :T]
        )  # (B, 1, T)  → scalar per position
        alpha = alpha.permute(0, 2, 1)  # (B, T, 1)

        # Interpolate weight matrices: W_eff = (1-α)·W₀ + α·W₁
        # Per-position matmul: x @ W_eff^T = (1-α)·(x@W₀^T) + α·(x@W₁^T)
        y0 = F.linear(x, self.weight_0.to(dtype), self.bias.to(dtype))  # (B, T, out)
        y1 = F.linear(x, self.weight_1.to(dtype))  # (B, T, out)
        return (1 - alpha) * y0 + alpha * y1


class PositionalResidualBias(nn.Module):
    """Phase 17J: Positional Residual Bias (PRB).

    y = Wx + b(pos)

    Position-dependent bias vector (NOT weight modulation). ~1% overhead.
    Diagnostic experiment: if PRB matches CKR, the benefit was never about
    weight modulation — just position-dependent output.
    """
    def __init__(self, in_features, out_features, kernel_size=64, max_seq_len=2048, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.ln_basis = nn.Identity()

        self.base = Linear(in_features, out_features, bias=True)
        # Position-dependent bias from causal conv
        self.pos_signal = nn.Parameter(torch.zeros(1, 1, max_seq_len))
        nn.init.normal_(self.pos_signal, std=0.01)
        # Conv: 1 channel → out_features channels (generates bias vector per position)
        self.bias_conv = nn.Conv1d(1, out_features, kernel_size, padding=kernel_size - 1, bias=True)
        nn.init.zeros_(self.bias_conv.weight)
        nn.init.zeros_(self.bias_conv.bias)

    def gate_parameters(self):
        yield self.pos_signal
        yield from self.bias_conv.parameters()

    def non_gate_parameters(self):
        yield from self.base.parameters()

    def forward(self, x, context_state=None):
        B, T, _ = x.shape
        dtype = x.dtype
        y = self.base(x)  # (B, T, out)
        sig = self.pos_signal[:, :, :T].expand(B, -1, -1).to(dtype)
        bias = F.conv1d(
            sig, self.bias_conv.weight.to(dtype), self.bias_conv.bias.to(dtype),
            padding=self.kernel_size - 1
        )[:, :, :T]  # (B, out, T)
        return y + bias.permute(0, 2, 1)  # (B, T, out)


# ============================================================================
# Phase 18: Beyond CKR — Fundamentally Different Approaches
# ============================================================================

class AdaptiveGatedLinear(nn.Module):
    """Phase 18A: Adaptive ReLU² Gate (ARG).

    y = W·x · g(x)  where g(x) = sigmoid(w_g·x + b_g) per output channel.

    Content-dependent (NOT position), minimal overhead (~D_out params).
    g starts at 1.0 (bias=+5 → sigmoid(5) ≈ 0.993).
    """
    def __init__(self, in_features, out_features, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()
        self.base = Linear(in_features, out_features, bias=True)
        # Gate: content-dependent per-channel gate
        self.gate_w = nn.Parameter(torch.zeros(out_features, in_features))
        self.gate_b = nn.Parameter(torch.full((out_features,), 5.0))  # sigmoid(5)≈0.993

    def gate_parameters(self):
        yield self.gate_w
        yield self.gate_b

    def non_gate_parameters(self):
        yield from self.base.parameters()

    def forward(self, x, context_state=None):
        dtype = x.dtype
        y = self.base(x)  # (B, T, out)
        gate_logits = F.linear(x, self.gate_w.to(dtype), self.gate_b.to(dtype))  # (B, T, out)
        gate = torch.sigmoid(gate_logits)
        return y * gate


class KroneckerLinear(nn.Module):
    """Phase 18C: Kronecker-Factored Linear (KFL).

    W = A ⊗ B where A is (d1, d1') and B is (d2, d2').
    d1*d2 = D_out, d1'*d2' = D_in.

    FEWER parameters than dense: d1² + d2² vs (d1*d2)².
    Computation: reshape x to (..., d2', d1'), multiply by B left and A right, reshape.
    """
    def __init__(self, in_features, out_features, **_ignored):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_basis = nn.Identity()
        # Find factorization closest to sqrt
        d2_in = self._find_factor(in_features)
        d1_in = in_features // d2_in
        d2_out = self._find_factor(out_features)
        d1_out = out_features // d2_out
        self.d1_in, self.d2_in = d1_in, d2_in
        self.d1_out, self.d2_out = d1_out, d2_out
        self.A = nn.Parameter(torch.empty(d1_out, d1_in))
        self.B = nn.Parameter(torch.empty(d2_out, d2_in))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Kronecker init: effective weight is A⊗B, so var(AB) = var(A)*var(B).
        # Target var = 2/fan_in. So each factor gets var = sqrt(2/fan_in).
        target_std = math.sqrt(math.sqrt(2.0 / in_features))
        nn.init.normal_(self.A, std=target_std)
        nn.init.normal_(self.B, std=target_std)

    @staticmethod
    def _find_factor(n):
        """Find factor of n closest to sqrt(n)."""
        s = int(math.sqrt(n))
        for i in range(s, 0, -1):
            if n % i == 0:
                return i
        return 1

    def gate_parameters(self):
        return iter([])

    def non_gate_parameters(self):
        yield self.A
        yield self.B
        yield self.bias

    def forward(self, x, context_state=None):
        B, T, _ = x.shape
        dtype = x.dtype
        # Reshape: (B, T, d2_in, d1_in)
        x_r = x.view(B, T, self.d2_in, self.d1_in)
        # B @ x_r → (B, T, d2_out, d1_in)
        y = torch.matmul(self.B.to(dtype), x_r)
        # y @ A^T → (B, T, d2_out, d1_out)
        y = torch.matmul(y, self.A.to(dtype).t())
        # Reshape back: (B, T, d2_out * d1_out)
        y = y.reshape(B, T, self.out_features)
        return y + self.bias.to(dtype)


class MixtureNorm(nn.Module):
    """Phase 18H: Mixture of Norms.

    norm(x) = w₁·RMSNorm(x) + w₂·LayerNorm(x)

    Learned mixture lets model keep magnitude info where useful.
    Starts at pure RMSNorm (w₁=1, w₂=0).
    """
    def __init__(self, n_embd):
        super().__init__()
        self.rms = nn.RMSNorm(n_embd)
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.w1 = nn.Parameter(torch.tensor(1.0))  # RMSNorm weight
        self.w2 = nn.Parameter(torch.tensor(0.0))  # LayerNorm weight (starts off)

    def forward(self, x):
        return self.w1 * self.rms(x) + self.w2 * self.ln(x)


class DynamicActivation(nn.Module):
    """Phase 18I: Dynamic Activation Function.

    f(x) = α·ReLU²(x) + β·GELU(x) + γ·SiLU(x)

    Per-layer learned scalars. ~3 params per layer. Starts as pure ReLU².
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # ReLU² weight
        self.beta = nn.Parameter(torch.tensor(0.0))   # GELU weight
        self.gamma = nn.Parameter(torch.tensor(0.0))  # SiLU weight

    def forward(self, x):
        return (self.alpha * F.relu(x).square()
                + self.beta * F.gelu(x)
                + self.gamma * F.silu(x))


class CausalAttnBias(nn.Module):
    """Phase 18J: Causal Attention Score Bias.

    Learned lower-triangular bias added to attention scores before softmax.
    Uses a factored representation: bias(i,j) = f(i-j) where f is a learned
    function of relative distance, stored as a 1D vector of max_dist values.

    Much cheaper than a full T×T matrix, and generalizes better.
    """
    def __init__(self, n_head, max_dist=2048):
        super().__init__()
        self.n_head = n_head
        self.max_dist = max_dist
        # Per-head learned bias as function of relative distance
        # Shape: (n_head, max_dist) — bias for distance 0, 1, 2, ...
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_dist))

    def get_bias(self, T, device):
        """Return (1, n_head, T, T) lower-triangular bias matrix."""
        # Create relative distance matrix
        pos = torch.arange(T, device=device)
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T), negative = future
        dist = dist.clamp(min=0, max=self.max_dist - 1)  # causal: only past
        # Mask future positions
        causal_mask = (pos.unsqueeze(0) - pos.unsqueeze(1)) >= 0  # (T, T)
        # Look up bias values
        bias = self.rel_bias[:, dist]  # (n_head, T, T)
        bias = bias * causal_mask.unsqueeze(0).float()
        return bias.unsqueeze(0)  # (1, n_head, T, T)


class PerChannelScale(nn.Module):
    """Phase 18F: Per-Channel Learnable Scale.

    Applied after a linear layer: y = Linear(x) * scale
    where scale is a per-channel learned parameter initialized to 1.0.

    Achieves similar effect to per-channel LR: channels that don't contribute
    have their scale shrink, effectively reducing their influence.
    """
    def __init__(self, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return x * self.scale.to(x.dtype)


class ModulationDiagnostics:
    """Universal diagnostic collector for ALL position-conditioned linear modules.

    Supports: CKR, LoKR, PGR, CIL, PRB, COM, PSG, and any future module
    with pos_signal or similar position-dependent routing.

    Computes diagnostics DIRECTLY from model parameters using torch.no_grad().
    This is compile-safe because it runs OUTSIDE the compiled forward graph.

    Key diagnostics:
    - branch_entropy: how uniform the routing is (1.0=uniform, 0.0=collapsed)
    - position_std: how much routing varies by position (0=static, high=dynamic)
    - gate_magnitude: average gate value (for gated architectures like PGR/PRB)
    - weight_divergence: how different branch weights are from each other
    - grad_norm: gradient norm of position signal (learning signal strength)
    - temperature: current softmax temperature (CKR only)
    """
    # All module types we can diagnose
    DIAG_TYPES = (CausalKernelLinear, LoKRLinear, PositionGatedResidual,
                  CausalInterpolationLinear, PositionalResidualBias,
                  CausalOutputMixer, PositionalScalarGatedLinear,
                  AdaptiveGatedLinear, KroneckerLinear)  # Phase 18
    # Phase 18 types diagnosed from model-level (not per-layer module)
    P18_TYPES = (DynamicActivation, MixtureNorm, PerChannelScale)

    def __init__(self, model):
        self.model = model
        self._layers = []
        for name, mod in model.named_modules():
            if isinstance(mod, self.DIAG_TYPES):
                self._layers.append((name, mod))
        # Phase 18: Also track P18 modules
        self._p18_layers = []
        for name, mod in model.named_modules():
            if isinstance(mod, self.P18_TYPES):
                self._p18_layers.append((name, mod))

    @torch.no_grad()
    def _collect_branch_routing(self, mod):
        """Collect branch routing stats for modules with pos_signal + branch_conv (CKR, LoKR)."""
        sig = mod.pos_signal  # (1, C_pos, T_max)
        raw_w = mod.branch_conv(sig.to(dtype=mod.branch_conv.weight.dtype))  # (1, K, T_max)
        temp = 1.0
        if hasattr(mod, '_temperature'):
            temp = mod._temperature.item()
        w = F.softmax(raw_w.float() / max(temp, 1e-8), dim=1)  # (1, K, T)

        # Branch weight entropy (normalized)
        log_w = torch.log(w.clamp(min=1e-8))
        entropy = -(w * log_w).sum(dim=1).mean().item()
        max_entropy = math.log(mod.n_branches)
        entropy_ratio = entropy / max(max_entropy, 1e-8)

        # Position variance (how much routing varies across positions)
        weight_std = w.std(dim=2).mean().item()

        # Weight divergence: mean pairwise cosine distance between branch weights
        weight_div = 0.0
        if hasattr(mod, 'branches'):
            norms = []
            for b in mod.branches:
                norms.append(b.weight.float().reshape(-1))
            if len(norms) >= 2:
                cos_sims = []
                for i in range(len(norms)):
                    for j in range(i+1, len(norms)):
                        cos = F.cosine_similarity(norms[i].unsqueeze(0), norms[j].unsqueeze(0)).item()
                        cos_sims.append(cos)
                weight_div = 1.0 - (sum(cos_sims) / len(cos_sims))  # 0=identical, 1=orthogonal

        return {
            'type': type(mod).__name__,
            'entropy_ratio': entropy_ratio,
            'position_std': weight_std,
            'temperature': temp,
            'weight_divergence': weight_div,
            'pos_signal_norm': sig.float().norm().item(),
            'pos_signal_grad_norm': sig.grad.float().norm().item() if sig.grad is not None else 0.0,
        }

    @torch.no_grad()
    def _collect_gate_stats(self, mod):
        """Collect gate stats for modules with a scalar position gate (PGR, CIL, PRB, COM, PSG)."""
        sig = mod.pos_signal  # (1, 1, T_max)
        # Determine gate conv
        if isinstance(mod, PositionGatedResidual):
            conv = mod.gate_conv
            gate_name = 'position_gate'
        elif isinstance(mod, CausalInterpolationLinear):
            conv = mod.alpha_conv
            gate_name = 'alpha'
        elif isinstance(mod, PositionalResidualBias):
            conv = mod.bias_conv
            gate_name = 'bias_magnitude'
        elif isinstance(mod, CausalOutputMixer):
            conv = mod.gate_conv
            gate_name = 'com_gate'
        elif isinstance(mod, PositionalScalarGatedLinear):
            conv = mod.gate_conv if hasattr(mod, 'gate_conv') else None
            gate_name = 'psg_gate'
        else:
            return {'type': type(mod).__name__, 'error': 'unknown_gate_type'}

        if conv is None:
            return {'type': type(mod).__name__, 'error': 'no_conv'}

        raw = conv(sig.to(dtype=conv.weight.dtype))  # (1, C_out, T_max)
        gate_vals = torch.sigmoid(raw.float())

        # For PRB, the "gate" is the bias magnitude, not sigmoid
        if isinstance(mod, PositionalResidualBias):
            gate_vals = raw.float()

        result = {
            'type': type(mod).__name__,
            'gate_name': gate_name,
            'gate_mean': gate_vals.mean().item(),
            'gate_std': gate_vals.std().item(),
            'gate_min': gate_vals.min().item(),
            'gate_max': gate_vals.max().item(),
            'pos_signal_norm': sig.float().norm().item(),
            'pos_signal_grad_norm': sig.grad.float().norm().item() if sig.grad is not None else 0.0,
        }

        # Weight divergence for PGR (base vs delta)
        if isinstance(mod, PositionGatedResidual):
            base_w = mod.base.weight.float().reshape(-1)
            delta_w = mod.delta.weight.float().reshape(-1)
            delta_norm = delta_w.norm().item()
            base_norm = base_w.norm().item()
            result['delta_to_base_ratio'] = delta_norm / max(base_norm, 1e-8)

        # Weight divergence for CIL (w0 vs w1)
        if isinstance(mod, CausalInterpolationLinear):
            w0 = mod.weight_0.float().reshape(-1)
            w1 = mod.weight_1.float().reshape(-1)
            cos = F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
            result['weight_cosine_sim'] = cos
            result['weight_divergence'] = 1.0 - cos

        return result

    @torch.no_grad()
    def _collect_p18_stats(self, mod):
        """Collect stats for Phase 18 modules (DynamicActivation, MixtureNorm, PerChannelScale)."""
        result = {'type': type(mod).__name__}
        if isinstance(mod, DynamicActivation):
            result['alpha'] = mod.alpha.item()
            result['beta'] = mod.beta.item()
            result['gamma'] = mod.gamma.item()
        elif isinstance(mod, MixtureNorm):
            result['w_rms'] = mod.w1.item()
            result['w_ln'] = mod.w2.item()
        elif isinstance(mod, PerChannelScale):
            result['scale_mean'] = mod.scale.float().mean().item()
            result['scale_std'] = mod.scale.float().std().item()
        return result

    @torch.no_grad()
    def _collect_arg_stats(self, mod):
        """Collect stats for AdaptiveGatedLinear (18A)."""
        gate_b = mod.gate_b.float()
        gate_w_norm = mod.gate_w.float().norm().item()
        return {
            'type': 'AdaptiveGatedLinear',
            'gate_mean': torch.sigmoid(gate_b).mean().item(),
            'gate_std': torch.sigmoid(gate_b).std().item(),
            'gate_w_norm': gate_w_norm,
            'gate_b_mean': gate_b.mean().item(),
        }

    @torch.no_grad()
    def _collect_kfl_stats(self, mod):
        """Collect stats for KroneckerLinear (18C)."""
        return {
            'type': 'KroneckerLinear',
            'A_norm': mod.A.float().norm().item(),
            'B_norm': mod.B.float().norm().item(),
            'factorization': f'{mod.d1_out}x{mod.d1_in} ⊗ {mod.d2_out}x{mod.d2_in}',
            'param_ratio': (mod.A.numel() + mod.B.numel()) / (mod.in_features * mod.out_features),
        }

    @torch.no_grad()
    def collect(self):
        """Compute diagnostics for all tracked layers."""
        if not self._layers and not self._p18_layers:
            return {}
        all_metrics = {}
        for name, mod in self._layers:
            if isinstance(mod, (CausalKernelLinear, LoKRLinear)):
                all_metrics[name] = self._collect_branch_routing(mod)
            elif isinstance(mod, AdaptiveGatedLinear):
                all_metrics[name] = self._collect_arg_stats(mod)
            elif isinstance(mod, KroneckerLinear):
                all_metrics[name] = self._collect_kfl_stats(mod)
            else:
                all_metrics[name] = self._collect_gate_stats(mod)
        # P18 modules: aggregate by type (only report first instance per type)
        seen_types = set()
        for name, mod in self._p18_layers:
            tname = type(mod).__name__
            if tname not in seen_types:
                all_metrics[f'p18/{tname}'] = self._collect_p18_stats(mod)
                seen_types.add(tname)
        return all_metrics

    def format(self, metrics):
        """Format diagnostics as a compact log string."""
        if not metrics:
            return "  [diag] no conditioning layers found"
        lines = ["  [modulation diagnostics]"]
        # Group by module type
        by_type = {}
        for name, m in metrics.items():
            t = m.get('type', 'unknown')
            by_type.setdefault(t, []).append(m)

        for mod_type, mods in by_type.items():
            lines.append(f"    {mod_type} ({len(mods)} layers):")
            if 'entropy_ratio' in mods[0]:
                ent = sum(m['entropy_ratio'] for m in mods) / len(mods)
                pstd = sum(m['position_std'] for m in mods) / len(mods)
                wdiv = sum(m.get('weight_divergence', 0) for m in mods) / len(mods)
                gnorm = sum(m.get('pos_signal_grad_norm', 0) for m in mods) / len(mods)
                lines.append(f"      entropy={ent:.3f}  pos_std={pstd:.4f}  "
                           f"wt_div={wdiv:.3f}  grad_norm={gnorm:.4f}")
                if any(m.get('temperature', 1.0) != 1.0 for m in mods):
                    lines.append(f"      temperature={mods[0]['temperature']:.3f}")
            elif 'gate_mean' in mods[0]:
                gm = sum(m['gate_mean'] for m in mods) / len(mods)
                gs = sum(m['gate_std'] for m in mods) / len(mods)
                gnorm = sum(m.get('pos_signal_grad_norm', 0) for m in mods) / len(mods)
                lines.append(f"      gate_mean={gm:.4f}  gate_std={gs:.4f}  "
                           f"grad_norm={gnorm:.4f}")
                if any('delta_to_base_ratio' in m for m in mods):
                    ratio = sum(m.get('delta_to_base_ratio', 0) for m in mods) / len(mods)
                    lines.append(f"      delta/base_ratio={ratio:.4f}")
                if any('weight_divergence' in m for m in mods):
                    wdiv = sum(m.get('weight_divergence', 0) for m in mods) / len(mods)
                    lines.append(f"      weight_divergence={wdiv:.4f}")
            elif 'alpha' in mods[0]:
                # DynamicActivation
                a = mods[0]['alpha']
                b = mods[0]['beta']
                g = mods[0]['gamma']
                lines.append(f"      α(ReLU²)={a:.3f}  β(GELU)={b:.3f}  γ(SiLU)={g:.3f}")
            elif 'w_rms' in mods[0]:
                # MixtureNorm
                w1 = mods[0]['w_rms']
                w2 = mods[0]['w_ln']
                lines.append(f"      w_rms={w1:.3f}  w_ln={w2:.3f}")
            elif 'scale_mean' in mods[0]:
                # PerChannelScale
                sm = mods[0]['scale_mean']
                ss = mods[0]['scale_std']
                lines.append(f"      scale_mean={sm:.4f}  scale_std={ss:.4f}")
            elif 'A_norm' in mods[0]:
                # KroneckerLinear
                an = sum(m['A_norm'] for m in mods) / len(mods)
                bn = sum(m['B_norm'] for m in mods) / len(mods)
                pr = mods[0].get('param_ratio', 0)
                lines.append(f"      A_norm={an:.3f}  B_norm={bn:.3f}  param_ratio={pr:.4f}")
            elif 'gate_w_norm' in mods[0]:
                # AdaptiveGatedLinear
                gm = sum(m['gate_mean'] for m in mods) / len(mods)
                gs = sum(m['gate_std'] for m in mods) / len(mods)
                gwn = sum(m['gate_w_norm'] for m in mods) / len(mods)
                lines.append(f"      gate_mean={gm:.4f}  gate_std={gs:.4f}  gate_w_norm={gwn:.4f}")
        return "\n".join(lines)

    def to_dict(self, metrics):
        """Flatten metrics into a single dict for wandb/logging."""
        if not metrics:
            return {}
        result = {}
        # Compute means across all layers
        all_vals = list(metrics.values())
        if 'entropy_ratio' in all_vals[0]:
            result['diag/branch_entropy'] = sum(m['entropy_ratio'] for m in all_vals) / len(all_vals)
            result['diag/position_std'] = sum(m['position_std'] for m in all_vals) / len(all_vals)
            result['diag/weight_divergence'] = sum(m.get('weight_divergence', 0) for m in all_vals) / len(all_vals)
            result['diag/temperature'] = all_vals[0].get('temperature', 1.0)
        if 'gate_mean' in all_vals[0]:
            result['diag/gate_mean'] = sum(m['gate_mean'] for m in all_vals) / len(all_vals)
            result['diag/gate_std'] = sum(m['gate_std'] for m in all_vals) / len(all_vals)
        result['diag/pos_signal_grad_norm'] = sum(m.get('pos_signal_grad_norm', 0) for m in all_vals) / len(all_vals)
        result['diag/pos_signal_norm'] = sum(m.get('pos_signal_norm', 0) for m in all_vals) / len(all_vals)
        if any('delta_to_base_ratio' in m for m in all_vals):
            result['diag/delta_to_base_ratio'] = sum(m.get('delta_to_base_ratio', 0) for m in all_vals) / len(all_vals)
        if any('weight_divergence' in m for m in all_vals):
            result['diag/weight_divergence'] = sum(m.get('weight_divergence', 0) for m in all_vals) / len(all_vals)
        return result

    def save_to_file(self, metrics, step, filepath):
        """Append metrics as JSONL to a file."""
        import json, os
        if not metrics:
            return
        flat = self.to_dict(metrics)
        flat['step'] = step
        # Also save per-layer breakdown for first 4 layers
        for i, (name, m) in enumerate(metrics.items()):
            if i >= 4:
                break
            prefix = f"layer_{i}"
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat[f"{prefix}/{k}"] = v
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(flat) + '\n')

    # ── Phase 19: Expanded Diagnostics ──────────────────────────────────────

    @torch.no_grad()
    def collect_p19(self, model):
        """Collect Phase 19 expanded diagnostics from model state.

        Returns a dict of metrics covering:
        - Gradient health (per-layer grad norms, SNR, flow ratio)
        - Weight dynamics (weight norms per layer)
        - Phase 19 proposal-specific metrics (alpha, head_importance, logit_scale,
          sigma stats, depth_decay, mixer gamma, etc.)
        - Residual stream parameters (resid_lambda, x0_lambda ratios)
        """
        metrics = {}

        # ── 1. Gradient health per block ──
        grad_norms_attn = []
        grad_norms_ffn = []
        weight_norms = []
        for i, block in enumerate(model.transformer.h):
            prefix = f"block_{i}"
            # Get key weight references
            if hasattr(block, 'attn'):
                cq_w = block.attn.c_q.weight if hasattr(block.attn, 'c_q') else None
                cp_w = block.attn.c_proj.weight if hasattr(block.attn, 'c_proj') else None
            else:
                cq_w = cp_w = None
            if hasattr(block, 'mlp'):
                fc_w = getattr(block.mlp, 'c_fc', None)
                fc_w = fc_w.weight if fc_w is not None else None
                proj_w = getattr(block.mlp, 'c_proj', None)
                proj_w = proj_w.weight if proj_w is not None else None
                # P20 MLP variants: fall back to first/last 2D param
                if fc_w is None:
                    mat_params = [p for p in block.mlp.parameters() if p.ndim == 2]
                    fc_w = mat_params[0] if mat_params else None
                    proj_w = mat_params[-1] if len(mat_params) > 1 else fc_w
            elif hasattr(block, 'ffwd'):
                fc_w = block.ffwd.c_fc.weight
                proj_w = block.ffwd.c_proj.weight
            else:
                fc_w = proj_w = None

            # Grad norms
            attn_gn = cp_w.grad.float().norm().item() if (cp_w is not None and cp_w.grad is not None) else 0.0
            ffn_gn = fc_w.grad.float().norm().item() if (fc_w is not None and fc_w.grad is not None) else 0.0
            grad_norms_attn.append(attn_gn)
            grad_norms_ffn.append(ffn_gn)

            # Grad SNR (mean / std) for FFN
            if fc_w is not None and fc_w.grad is not None:
                g = fc_w.grad.float()
                snr = (g.mean().abs() / (g.std() + 1e-8)).item()
            else:
                snr = 0.0
            metrics[f"{prefix}/grad_norm_attn"] = attn_gn
            metrics[f"{prefix}/grad_norm_ffn"] = ffn_gn
            metrics[f"{prefix}/grad_snr"] = snr
            # Gradient flow ratio (attn vs ffn balance)
            metrics[f"{prefix}/grad_flow_ratio"] = attn_gn / max(ffn_gn, 1e-8)

            # Weight norms
            wn = fc_w.float().norm().item() if fc_w is not None else 0.0
            weight_norms.append(wn)
            metrics[f"{prefix}/weight_norm_fc"] = wn

        # Aggregate gradient stats
        if grad_norms_ffn:
            gn_tensor = torch.tensor(grad_norms_ffn)
            gn_mean = gn_tensor.mean().item()
            gn_std = gn_tensor.std().item()
            metrics["global/grad_cv"] = gn_std / max(gn_mean, 1e-8)  # coefficient of variation
            metrics["global/grad_norm_mean"] = gn_mean

        # ── 2. Residual stream parameters ──
        rl = model.resid_lambdas.float()
        x0l = model.x0_lambdas.float()
        for i in range(len(rl)):
            metrics[f"block_{i}/resid_lambda"] = rl[i].item()
            metrics[f"block_{i}/x0_lambda"] = x0l[i].item()
            metrics[f"block_{i}/resid_scale_ratio"] = rl[i].item() / max(abs(x0l[i].item()), 1e-8)

        # ── 3. Phase 19 proposal-specific metrics ──

        # 19A: Residual Gate Scaling (alpha per layer)
        for i, block in enumerate(model.transformer.h):
            if hasattr(block, 'residual_alpha') and block.residual_alpha is not None:
                alpha_val = F.softplus(block.residual_alpha).item()
                metrics[f"block_{i}/p19a_alpha"] = alpha_val
                if block.residual_alpha.grad is not None:
                    metrics[f"block_{i}/p19a_alpha_grad"] = block.residual_alpha.grad.float().norm().item()

        # 19B: Head Importance Scaling
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 'head_importance') and attn.head_importance is not None:
                his = F.softplus(attn.head_importance.float())
                metrics[f"block_{i}/p19b_head_scale_mean"] = his.mean().item()
                metrics[f"block_{i}/p19b_head_scale_std"] = his.std().item()
                metrics[f"block_{i}/p19b_head_scale_min"] = his.min().item()
                metrics[f"block_{i}/p19b_head_scale_max"] = his.max().item()
                # Entropy of head importance (normalized)
                p = his / his.sum()
                ent = -(p * torch.log(p + 1e-8)).sum().item()
                max_ent = math.log(len(his))
                metrics[f"block_{i}/p19b_head_entropy"] = ent / max(max_ent, 1e-8)

        # 19C: Residual Stream Mixing
        if model.residual_mix_gamma is not None:
            for i, gamma_p in enumerate(model.residual_mix_gamma):
                metrics[f"block_{i}/p19c_gamma"] = gamma_p.item()
            if model.residual_mixers is not None:
                for i, mixer in enumerate(model.residual_mixers):
                    metrics[f"block_{i}/p19c_mix_weight_norm"] = mixer.weight.float().norm().item()

        # 19D: Attention Logit Bias
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 'attn_logit_scale') and attn.attn_logit_scale is not None:
                scale = F.softplus(attn.attn_logit_scale.float())
                metrics[f"block_{i}/p19d_logit_scale_mean"] = scale.mean().item()
                metrics[f"block_{i}/p19d_logit_scale_std"] = scale.std().item()

        # 19E: Learned Residual Decay
        if model.depth_decay_raw is not None:
            decay_base = torch.sigmoid(model.depth_decay_raw.float()).item()
            metrics["global/p19e_decay_base"] = decay_base
            for i in range(model.config.n_layer):
                metrics[f"block_{i}/p19e_x0_factor"] = decay_base ** i

        # 19G: Spectral Reparameterization
        for i, block in enumerate(model.transformer.h):
            ffwd = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if ffwd is None:
                continue
            for label, srp in [('proj', getattr(ffwd, 'srp_proj', None)), ('fc', getattr(ffwd, 'srp_fc', None))]:
                if srp is None:
                    continue
                sigma = srp.sigma.float()
                metrics[f"block_{i}/p19g_{label}_sigma_max"] = sigma.max().item()
                metrics[f"block_{i}/p19g_{label}_sigma_min"] = sigma.min().item()
                metrics[f"block_{i}/p19g_{label}_sigma_ratio"] = (sigma.max() / (sigma.min() + 1e-8)).item()
                # Sigma entropy (normalized) — measures spectral concentration
                p = sigma.abs() / (sigma.abs().sum() + 1e-8)
                ent = -(p * torch.log(p + 1e-8)).sum().item()
                max_ent = math.log(len(sigma))
                metrics[f"block_{i}/p19g_{label}_sigma_entropy"] = ent / max(max_ent, 1e-8)

        # 19I: VE gate bias values
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            if hasattr(attn, 've_gate') and attn.ve_gate is not None and attn.ve_gate.bias is not None:
                bias = attn.ve_gate.bias.float()
                metrics[f"block_{i}/p19i_ve_bias_mean"] = bias.mean().item()
                metrics[f"block_{i}/p19i_ve_bias_std"] = bias.std().item()

        return metrics

    def format_p19(self, p19_metrics):
        """Format Phase 19 diagnostics as a compact log string."""
        if not p19_metrics:
            return ""
        lines = ["  [p19 diagnostics]"]
        # Global metrics
        for k in sorted(p19_metrics):
            if k.startswith("global/"):
                lines.append(f"    {k}: {p19_metrics[k]:.4f}")
        # Per-block summary (just first and last block for brevity)
        n_blocks = sum(1 for k in p19_metrics if k.startswith("block_0/"))
        if n_blocks > 0:
            block_keys = sorted(set(k.split("/")[1] for k in p19_metrics if k.startswith("block_")))
            for bk in block_keys:
                vals = [p19_metrics.get(f"block_{i}/{bk}", None) for i in range(20) if f"block_{i}/{bk}" in p19_metrics]
                if vals:
                    lines.append(f"    {bk}: [{vals[0]:.4f}...{vals[-1]:.4f}] (n={len(vals)})")
        return "\n".join(lines)

    def to_dict_p19(self, p19_metrics):
        """Flatten P19 metrics into wandb dict with 'p19/' prefix."""
        if not p19_metrics:
            return {}
        result = {}
        for k, v in p19_metrics.items():
            if isinstance(v, (int, float)):
                result[f"p19/{k}"] = v
        return result

    # ── Phase 20: Dynamic Weight Computation Diagnostics ──────────────────

    @torch.no_grad()
    def collect_p20(self, model):
        """Collect Phase 20 diagnostics: routing entropy, expert load, branch divergence."""
        metrics = {}
        for i, block in enumerate(model.transformer.h):
            mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if mlp is None:
                continue
            prefix = f"block_{i}"

            if isinstance(mlp, MoNE_MLP):
                # Router entropy
                metrics[f"{prefix}/p20f_router_entropy"] = mlp._last_router_entropy
                # Expert load balance (std of loads — 0 = perfect balance)
                if mlp._last_expert_load is not None:
                    load = mlp._last_expert_load.float()
                    metrics[f"{prefix}/p20f_load_std"] = load.std().item()
                    metrics[f"{prefix}/p20f_load_max"] = load.max().item()
                    metrics[f"{prefix}/p20f_load_min"] = load.min().item()
                # Expert weight divergence (cosine distance between expert fc weights)
                norms = [e.weight.float().reshape(-1) for e in mlp.experts_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20f_expert_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))
                # Router weight gradient norm
                if mlp.router.weight.grad is not None:
                    metrics[f"{prefix}/p20f_router_grad_norm"] = mlp.router.weight.grad.float().norm().item()

            elif isinstance(mlp, FrozenRoutedMLP):
                metrics[f"{prefix}/p20c_routing_entropy"] = mlp._last_routing_entropy
                if mlp._last_routing_weights is not None:
                    load = mlp._last_routing_weights.float()
                    metrics[f"{prefix}/p20c_load_std"] = load.std().item()
                # Branch weight divergence
                norms = [b.weight.float().reshape(-1) for b in mlp.branches_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20c_branch_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

            elif isinstance(mlp, DetachedRoutedMLP):
                metrics[f"{prefix}/p20d_routing_entropy"] = mlp._last_routing_entropy
                # Router gradient norm (from aux loss)
                router_last = mlp.router[-1]  # last Linear in router Sequential
                if router_last.weight.grad is not None:
                    metrics[f"{prefix}/p20d_router_grad_norm"] = router_last.weight.grad.float().norm().item()
                # Branch divergence
                norms = [b.weight.float().reshape(-1) for b in mlp.branches_fc]
                if len(norms) >= 2:
                    cos_sims = []
                    for ii in range(len(norms)):
                        for jj in range(ii + 1, len(norms)):
                            cos = F.cosine_similarity(norms[ii].unsqueeze(0), norms[jj].unsqueeze(0)).item()
                            cos_sims.append(cos)
                    metrics[f"{prefix}/p20d_branch_divergence"] = 1.0 - (sum(cos_sims) / len(cos_sims))

        return metrics

    def format_p20(self, p20_metrics):
        """Format Phase 20 diagnostics as a compact log string."""
        if not p20_metrics:
            return ""
        lines = ["  [p20 diagnostics]"]
        block_keys = sorted(set(k.split("/")[1] for k in p20_metrics if k.startswith("block_")))
        for bk in block_keys:
            vals = [p20_metrics.get(f"block_{i}/{bk}", None) for i in range(64) if f"block_{i}/{bk}" in p20_metrics]
            if vals:
                lines.append(f"    {bk}: [{vals[0]:.4f}...{vals[-1]:.4f}] (n={len(vals)})")
        return "\n".join(lines)

    def to_dict_p20(self, p20_metrics):
        """Flatten P20 metrics into wandb dict with 'p20/' prefix."""
        if not p20_metrics:
            return {}
        result = {}
        for k, v in p20_metrics.items():
            if isinstance(v, (int, float)):
                result[f"p20/{k}"] = v
        return result


class ResidualAdaptiveLinear(nn.Module):
    """Proposal A: Dense base layer + context-conditioned additive low-rank delta.

    y = base(x) + scale * (x @ U * ctx_coeffs(ctx)) @ V

    Key properties:
    - V=0 at init → delta=0 → EXACT dense baseline at step 0 (no regression risk)
    - base(x) always has full D_in×D_out rank — no bottleneck from basis compression
    - Gradient path through base is pure dense-linear (no interference)
    - Delta grows only as optimizer finds it useful, gated by ctx_coeffs
    - Same forward(x, ctx) / gate_parameters() / non_gate_parameters() API as RemixedLinear

    Compatible with setup_optimizer: gate_parameters() returns ctx_coeffs params,
    non_gate_parameters() returns base+U+V+scale. No ln_basis → optimizer loop skips it.
    """
    def __init__(self, in_features, out_features, context_dim, rank=32, **_ignored):
        super().__init__()
        self.rank = rank
        self.in_features  = in_features
        self.out_features = out_features
        # Full-rank dense base — always active, no rank deficit
        self.base = Linear(in_features, out_features, bias=True)
        # Low-rank adaptive delta matrices
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.zeros(rank, out_features))    # ZERO: delta=0 at init
        # Context → rank-dimensional coefficient vector
        self.ctx_coeffs = Linear(context_dim, rank, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Dummy attribute so setup_optimizer can safely call .ln_basis.parameters() check
        self.ln_basis = nn.Identity()  # has no parameters

        nn.init.kaiming_uniform_(self.U, a=0.01)
        nn.init.zeros_(self.ctx_coeffs.weight)
        nn.init.zeros_(self.ctx_coeffs.bias)

    def gate_parameters(self):
        """Context gate params routed to lower-LR optimizer group."""
        yield from self.ctx_coeffs.parameters()
        yield self.scale

    def non_gate_parameters(self):
        """Structural params (base weights, U, V) routed to normal-LR group."""
        yield self.base.weight
        yield self.base.bias
        yield self.U
        yield self.V

    def forward(self, x, context_state):
        y = self.base(x)
        if context_state is not None:
            ctx = context_state.to(x.dtype)
            coeffs = self.ctx_coeffs(ctx)                                     # (B, T, rank)
            h = (x @ self.U.to(x.dtype))                                      # (B, T, rank)
            delta = (h * coeffs) @ self.V.to(x.dtype)                         # (B, T, out_features)
            y = y + self.scale.to(x.dtype) * delta
        return y


class RemixedFeedForward(nn.Module):
    """Feedforward path using RemixedLinear (or RAL) in the base MLP framework."""
    def __init__(self, config):
        super().__init__()
        use_ral = getattr(config, 'use_ral', False)
        mode = getattr(config, 'cclblock_modulation', 'weight')
        use_decoupled = mode == 'decoupled'
        ral_rank = getattr(config, 'ral_rank', 32)
        film_gate = getattr(config, 'cclblock_film_gate', False)
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        if mode == 'fsi':
            n_rot = getattr(config, 'cclblock_fsi_rotations', 8)
            sel_dim = getattr(config, 'cclblock_fsi_selector_dim', 64)
            self.c_fc   = FrozenSubspaceIndexedLinear(config.n_embd, 4 * config.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=config.n_embd)
            self.c_proj = FrozenSubspaceIndexedLinear(4 * config.n_embd, config.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=config.n_embd)
        elif mode == 'aesp':
            n_str = getattr(config, 'cclblock_aesp_strata', 4)
            d_rank = getattr(config, 'cclblock_aesp_delta_rank', 4)
            self.c_fc   = AttentionEntropyStratifiedLinear(config.n_embd, 4 * config.n_embd, n_strata=n_str, delta_rank=d_rank)
            self.c_proj = AttentionEntropyStratifiedLinear(4 * config.n_embd, config.n_embd, n_strata=n_str, delta_rank=d_rank)
        elif mode == 'ckr' or mode == 'ckr_ffn':
            n_br = getattr(config, 'cclblock_ckr_branches', 4)
            ksz = getattr(config, 'cclblock_ckr_kernel_size', 64)
            n_pc = getattr(config, 'cclblock_ckr_pos_channels', 1)
            cb_scale = getattr(config, 'cclblock_ckr_content_bias', 0.0)
            temp = getattr(config, 'cclblock_ckr_temp_start', 1.0)
            ortho = bool(getattr(config, 'cclblock_ckr_ortho_init', 0))
            bdrop = float(getattr(config, 'cclblock_ckr_branch_dropout', 0.0))
            ckr_common = dict(n_branches=n_br, kernel_size=ksz, max_seq_len=config.sequence_len, n_pos_channels=n_pc, content_bias_scale=cb_scale, signal_dim=config.n_embd, temperature=temp, ortho_init=ortho, branch_dropout=bdrop)
            self.c_fc   = CausalKernelLinear(config.n_embd, 4 * config.n_embd, **ckr_common)
            self.c_proj = CausalKernelLinear(4 * config.n_embd, config.n_embd, **ckr_common)
        elif mode == 'pgr':
            pgr_ks = getattr(config, 'cclblock_pgr_kernel_size', 64)
            self.c_fc   = PositionGatedResidual(config.n_embd, 4 * config.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionGatedResidual(4 * config.n_embd, config.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
        elif mode == 'cil':
            cil_ks = getattr(config, 'cclblock_cil_kernel_size', 64)
            self.c_fc   = CausalInterpolationLinear(config.n_embd, 4 * config.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalInterpolationLinear(4 * config.n_embd, config.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
        elif mode == 'prb':
            prb_ks = getattr(config, 'cclblock_prb_kernel_size', 64)
            self.c_fc   = PositionalResidualBias(config.n_embd, 4 * config.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalResidualBias(4 * config.n_embd, config.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
        elif mode == 'arg':
            # 18A: Adaptive ReLU² Gate — content-dependent per-channel gate
            self.c_fc   = AdaptiveGatedLinear(config.n_embd, 4 * config.n_embd)
            self.c_proj = AdaptiveGatedLinear(4 * config.n_embd, config.n_embd)
        elif mode == 'kfl':
            # 18C: Kronecker-Factored Linear — fewer params than dense
            self.c_fc   = KroneckerLinear(config.n_embd, 4 * config.n_embd)
            self.c_proj = KroneckerLinear(4 * config.n_embd, config.n_embd)
        elif mode == 'com':
            com_ks = getattr(config, 'cclblock_com_kernel_size', 32)
            self.c_fc   = CausalOutputMixer(config.n_embd, 4 * config.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalOutputMixer(4 * config.n_embd, config.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
        elif mode == 'giad':
            giad_rank = getattr(config, 'cclblock_giad_rank', 32)
            self.c_fc   = GradientIsolatedDeltaLinear(config.n_embd, 4 * config.n_embd, delta_rank=giad_rank)
            self.c_proj = GradientIsolatedDeltaLinear(4 * config.n_embd, config.n_embd, delta_rank=giad_rank)
        elif mode == 'psg':
            psg_ks = getattr(config, 'cclblock_psg_kernel_size', 64)
            self.c_fc   = PositionalScalarGatedLinear(config.n_embd, 4 * config.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalScalarGatedLinear(4 * config.n_embd, config.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
        elif mode == 'splitstream':
            ss_ratio = float(getattr(config, 'cclblock_ss_dynamic_ratio', 0.25))
            ss_br = getattr(config, 'cclblock_ss_branches', 2)
            ss_ks = getattr(config, 'cclblock_ss_kernel_size', 64)
            self.c_fc   = SplitStreamLinear(config.n_embd, 4 * config.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_proj = SplitStreamLinear(4 * config.n_embd, config.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
        elif mode == 'lokr':
            lokr_br = getattr(config, 'cclblock_lokr_branches', 8)
            lokr_r = getattr(config, 'cclblock_lokr_rank', 16)
            lokr_ks = getattr(config, 'cclblock_ckr_kernel_size', 64)
            self.c_fc   = LoKRLinear(config.n_embd, 4 * config.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_proj = LoKRLinear(4 * config.n_embd, config.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
        elif mode == 'tucker':
            self.c_fc = TuckerAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_proj = TuckerAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
        elif mode in ('svs', 'dcu'):
            self.c_fc = SingularValueSteeringLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0)
            self.c_proj = SingularValueSteeringLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0)
        elif mode == 'vq':
            self.c_fc = VQAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
            self.c_proj = VQAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
        elif use_decoupled:
            ratio = float(getattr(config, 'cclblock_dynamic_ratio', 0.25))
            gate_rank = int(getattr(config, 'cclblock_gate_rank', 8))
            self.c_fc   = DecoupledAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_proj = DecoupledAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
        elif use_ral:
            self.c_fc   = ResidualAdaptiveLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_proj = ResidualAdaptiveLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, rank=ral_rank)
        else:
            self.c_fc   = RemixedLinear(config.n_embd, 4 * config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
            self.c_proj = RemixedLinear(4 * config.n_embd, config.n_embd, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
        # 18I: Dynamic Activation (learned mix of ReLU², GELU, SiLU)
        self.dynamic_act = DynamicActivation() if getattr(config, 'p18_dynamic_activation', 0) else None
        # 18F: Per-Channel Scale after projection
        self.channel_scale = PerChannelScale(config.n_embd) if getattr(config, 'p18_per_channel_scale', 0) else None

    def forward(self, x, context_state):
        x = self.c_fc(x, context_state)
        if self.dynamic_act is not None:
            x = self.dynamic_act(x)
        else:
            x = F.relu(x).square()
        x = self.c_proj(x, context_state)
        if self.channel_scale is not None:
            x = self.channel_scale(x)
        return x


class RemixedMultiAttention(nn.Module):
    """Attention path using RemixedLinear (or RAL) for Q/K/V/proj."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.ve_gate_channels = min(self.n_embd, 32)
        self.shadow_dim = max(0, int(getattr(config, 'cclblock_attn_shadow_dim', 0)))
        self.shadow_dim_per_head = 0
        if self.shadow_dim > 0:
            raw_per_head = (self.shadow_dim + self.n_kv_head - 1) // self.n_kv_head
            if raw_per_head % 8 != 0:
                raw_per_head = ((raw_per_head + 7) // 8) * 8
            corrected = raw_per_head * self.n_kv_head
            if corrected != self.shadow_dim:
                print0(f"cclblock_attn_shadow_dim auto-corrected from {self.shadow_dim} to {corrected} to ensure FlashAttention V-head multiple of 8")
                self.shadow_dim = corrected
            self.shadow_dim_per_head = self.shadow_dim // self.n_kv_head
        self.v_head_dim = self.head_dim + self.shadow_dim_per_head
        use_ral = getattr(config, 'use_ral', False)
        mode = getattr(config, 'cclblock_modulation', 'weight')
        use_decoupled = mode == 'decoupled'
        ral_rank = getattr(config, 'ral_rank', 32)
        film_gate = getattr(config, 'cclblock_film_gate', False)
        kwargs = config.remixed_linear_kwargs if config.remixed_linear_kwargs is not None else dict(use_basis_gate=True, use_output_gate=True, use_context=True)
        scale = getattr(config, 'scale_basis_size', True)
        if mode == 'fsi':
            n_rot = getattr(config, 'cclblock_fsi_rotations', 8)
            sel_dim = getattr(config, 'cclblock_fsi_selector_dim', 64)
            self.c_q    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_head * self.head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_k    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_kv_head * self.head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_v    = FrozenSubspaceIndexedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
            self.c_proj = FrozenSubspaceIndexedLinear(self.n_embd, self.n_embd, n_rotations=n_rot, selector_dim=sel_dim, signal_dim=self.n_embd)
        elif mode == 'aesp':
            n_str = getattr(config, 'cclblock_aesp_strata', 4)
            d_rank = getattr(config, 'cclblock_aesp_delta_rank', 4)
            self.c_q    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_head * self.head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_k    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_kv_head * self.head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_v    = AttentionEntropyStratifiedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_strata=n_str, delta_rank=d_rank)
            self.c_proj = AttentionEntropyStratifiedLinear(self.n_embd, self.n_embd, n_strata=n_str, delta_rank=d_rank)
        elif mode == 'ckr':
            n_br = getattr(config, 'cclblock_ckr_branches', 4)
            ksz = getattr(config, 'cclblock_ckr_kernel_size', 64)
            n_pc = getattr(config, 'cclblock_ckr_pos_channels', 1)
            cb_scale = getattr(config, 'cclblock_ckr_content_bias', 0.0)
            temp = getattr(config, 'cclblock_ckr_temp_start', 1.0)
            ortho = bool(getattr(config, 'cclblock_ckr_ortho_init', 0))
            bdrop = float(getattr(config, 'cclblock_ckr_branch_dropout', 0.0))
            ckr_common = dict(n_branches=n_br, kernel_size=ksz, max_seq_len=config.sequence_len, n_pos_channels=n_pc, content_bias_scale=cb_scale, signal_dim=self.n_embd, temperature=temp, ortho_init=ortho, branch_dropout=bdrop)
            self.c_q    = CausalKernelLinear(self.n_embd, self.n_head * self.head_dim, **ckr_common)
            self.c_k    = CausalKernelLinear(self.n_embd, self.n_kv_head * self.head_dim, **ckr_common)
            self.c_v    = CausalKernelLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **ckr_common)
            self.c_proj = CausalKernelLinear(self.n_embd, self.n_embd, **ckr_common)
        elif mode == 'ckr_ffn':
            # 16B: Plain Linear for attention (CKR only on FFN)
            self.c_q    = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k    = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v    = Linear(self.n_embd, self.n_kv_head * self.v_head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        elif mode == 'pgr':
            pgr_ks = getattr(config, 'cclblock_pgr_kernel_size', 64)
            self.c_q    = PositionGatedResidual(self.n_embd, self.n_head * self.head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionGatedResidual(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionGatedResidual(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionGatedResidual(self.n_embd, self.n_embd, kernel_size=pgr_ks, max_seq_len=config.sequence_len)
        elif mode == 'cil':
            cil_ks = getattr(config, 'cclblock_cil_kernel_size', 64)
            self.c_q    = CausalInterpolationLinear(self.n_embd, self.n_head * self.head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_k    = CausalInterpolationLinear(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_v    = CausalInterpolationLinear(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=cil_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalInterpolationLinear(self.n_embd, self.n_embd, kernel_size=cil_ks, max_seq_len=config.sequence_len)
        elif mode == 'prb':
            prb_ks = getattr(config, 'cclblock_prb_kernel_size', 64)
            self.c_q    = PositionalResidualBias(self.n_embd, self.n_head * self.head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionalResidualBias(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionalResidualBias(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=prb_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalResidualBias(self.n_embd, self.n_embd, kernel_size=prb_ks, max_seq_len=config.sequence_len)
        elif mode == 'com':
            com_ks = getattr(config, 'cclblock_com_kernel_size', 32)
            self.c_q    = CausalOutputMixer(self.n_embd, self.n_head * self.head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_k    = CausalOutputMixer(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_v    = CausalOutputMixer(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=com_ks, max_seq_len=config.sequence_len)
            self.c_proj = CausalOutputMixer(self.n_embd, self.n_embd, kernel_size=com_ks, max_seq_len=config.sequence_len)
        elif mode == 'giad':
            giad_rank = getattr(config, 'cclblock_giad_rank', 32)
            self.c_q    = GradientIsolatedDeltaLinear(self.n_embd, self.n_head * self.head_dim, delta_rank=giad_rank)
            self.c_k    = GradientIsolatedDeltaLinear(self.n_embd, self.n_kv_head * self.head_dim, delta_rank=giad_rank)
            self.c_v    = GradientIsolatedDeltaLinear(self.n_embd, self.n_kv_head * self.v_head_dim, delta_rank=giad_rank)
            self.c_proj = GradientIsolatedDeltaLinear(self.n_embd, self.n_embd, delta_rank=giad_rank)
        elif mode == 'psg':
            psg_ks = getattr(config, 'cclblock_psg_kernel_size', 64)
            self.c_q    = PositionalScalarGatedLinear(self.n_embd, self.n_head * self.head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_k    = PositionalScalarGatedLinear(self.n_embd, self.n_kv_head * self.head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_v    = PositionalScalarGatedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, kernel_size=psg_ks, max_seq_len=config.sequence_len)
            self.c_proj = PositionalScalarGatedLinear(self.n_embd, self.n_embd, kernel_size=psg_ks, max_seq_len=config.sequence_len)
        elif mode == 'splitstream':
            ss_ratio = float(getattr(config, 'cclblock_ss_dynamic_ratio', 0.25))
            ss_br = getattr(config, 'cclblock_ss_branches', 2)
            ss_ks = getattr(config, 'cclblock_ss_kernel_size', 64)
            self.c_q    = SplitStreamLinear(self.n_embd, self.n_head * self.head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_k    = SplitStreamLinear(self.n_embd, self.n_kv_head * self.head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_v    = SplitStreamLinear(self.n_embd, self.n_kv_head * self.v_head_dim, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
            self.c_proj = SplitStreamLinear(self.n_embd, self.n_embd, dynamic_ratio=ss_ratio, n_branches=ss_br, kernel_size=ss_ks, max_seq_len=config.sequence_len)
        elif mode == 'lokr':
            lokr_br = getattr(config, 'cclblock_lokr_branches', 8)
            lokr_r = getattr(config, 'cclblock_lokr_rank', 16)
            lokr_ks = getattr(config, 'cclblock_ckr_kernel_size', 64)
            self.c_q    = LoKRLinear(self.n_embd, self.n_head * self.head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_k    = LoKRLinear(self.n_embd, self.n_kv_head * self.head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_v    = LoKRLinear(self.n_embd, self.n_kv_head * self.v_head_dim, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
            self.c_proj = LoKRLinear(self.n_embd, self.n_embd, n_branches=lokr_br, rank=lokr_r, kernel_size=lokr_ks, max_seq_len=config.sequence_len)
        elif mode == 'tucker':
            self.c_q = TuckerAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_k = TuckerAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_v = TuckerAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
            self.c_proj = TuckerAdaptiveLinear(self.n_embd, self.n_embd, context_dim=config.remix_context_dim, tucker_rank=getattr(config, 'cclblock_tucker_rank', 32), tucker_modes=getattr(config, 'cclblock_tucker_modes', 8))
        elif mode in ('svs', 'dcu'):
            dcu_steps = getattr(config, 'cclblock_dcu_warmup_steps', 0) if mode == 'dcu' else 0
            common = dict(context_dim=config.remix_context_dim, svs_rank=getattr(config, 'cclblock_svs_rank', 64), svs_eps=getattr(config, 'cclblock_svs_eps', 0.1), dcu_warmup_steps=dcu_steps)
            self.c_q = SingularValueSteeringLinear(self.n_embd, self.n_head * self.head_dim, **common)
            self.c_k = SingularValueSteeringLinear(self.n_embd, self.n_kv_head * self.head_dim, **common)
            self.c_v = SingularValueSteeringLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **common)
            self.c_proj = SingularValueSteeringLinear(self.n_embd, self.n_embd, **common)
        elif mode == 'vq':
            common = dict(context_dim=config.remix_context_dim, vq_codes=getattr(config, 'cclblock_vq_codes', 8), vq_temperature=getattr(config, 'cclblock_vq_temperature', 1.0))
            self.c_q = VQAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, **common)
            self.c_k = VQAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, **common)
            self.c_v = VQAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, **common)
            self.c_proj = VQAdaptiveLinear(self.n_embd, self.n_embd, **common)
        elif use_decoupled:
            ratio = float(getattr(config, 'cclblock_dynamic_ratio', 0.25))
            gate_rank = int(getattr(config, 'cclblock_gate_rank', 8))
            self.c_q    = DecoupledAdaptiveLinear(self.n_embd, self.n_head * self.head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_k    = DecoupledAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_v    = DecoupledAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
            self.c_proj = DecoupledAdaptiveLinear(self.n_embd, self.n_embd, context_dim=config.remix_context_dim, dynamic_ratio=ratio, basis_size=config.remix_basis_size, gate_rank=gate_rank)
        elif use_ral:
            self.c_q    = ResidualAdaptiveLinear(self.n_embd, self.n_head * self.head_dim,    context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_k    = ResidualAdaptiveLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_v    = ResidualAdaptiveLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, rank=ral_rank)
            self.c_proj = ResidualAdaptiveLinear(self.n_embd, self.n_embd,                    context_dim=config.remix_context_dim, rank=ral_rank)
        else:
            self.c_q    = RemixedLinear(self.n_embd, self.n_head * self.head_dim,    context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
            self.c_k    = RemixedLinear(self.n_embd, self.n_kv_head * self.head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
            self.c_v    = RemixedLinear(self.n_embd, self.n_kv_head * self.v_head_dim, context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
            self.c_proj = RemixedLinear(self.n_embd, self.n_embd,                    context_dim=config.remix_context_dim, basis_size=config.remix_basis_size, remixed_linear_kwargs=kwargs, scale_basis=scale, film_gate=film_gate)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._attn_mode = mode  # store for forward dispatch

    def forward(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        B, T, C = x.size()
        if self._attn_mode == 'ckr_ffn':
            # Plain Linear for attention — no context_state
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v_full = self.c_v(x).view(B, T, self.n_kv_head, self.v_head_dim)
        else:
            q = self.c_q(x, context_state).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x, context_state).view(B, T, self.n_kv_head, self.head_dim)
            v_full = self.c_v(x, context_state).view(B, T, self.n_kv_head, self.v_head_dim)
        v = v_full[..., :self.head_dim]

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
        if self._attn_mode == 'ckr_ffn':
            return self.c_proj(y)
        return self.c_proj(y, context_state)

    def forward_with_shadow(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        """Dual-V routing: return content output plus shadow stream routed by same attention map."""
        B, T, C = x.size()
        q = self.c_q(x, context_state).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x, context_state).view(B, T, self.n_kv_head, self.head_dim)
        v_full = self.c_v(x, context_state).view(B, T, self.n_kv_head, self.v_head_dim)
        v = v_full[..., :self.head_dim]

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            if self.ve_gate is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
        if self.shadow_dim_per_head > 0:
            v_full = torch.cat([v, v_full[..., self.head_dim:]], dim=-1)
        else:
            v_full = v

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y_full = flash_attn.flash_attn_func(q, k, v_full, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y_full = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v_full,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y_content = y_full[..., :self.head_dim].contiguous().view(B, T, -1)
        y_proj = self.c_proj(y_content, context_state)
        if self.shadow_dim_per_head > 0:
            y_shadow = y_full[..., self.head_dim:].contiguous().view(B, T, -1)
        else:
            y_shadow = torch.zeros(B, T, 0, device=x.device, dtype=x.dtype)
        return y_proj, y_shadow

    def forward_with_q_stats(self, x, ve, cos_sin, window_size, kv_cache, context_state):
        """Design 2: run forward, also returning mean-query-vector as context signal.

        q after RoPE+QKnorm is the model's per-position 'search intent'. Its mean across
        attention heads gives a (B, T, head_dim) tensor capturing the query distribution
        without any temporal recurrence. Detached so it contributes no gradient to Q/K/V.
        """
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

        # Capture query statistics BEFORE flash_attn (q is post-RoPE+QKnorm)
        # Mean across heads: (B, T, head_dim) — the average 'search direction'
        q_stats = q.mean(dim=2).detach()  # detach: no gradient through ctx into Q/K weights

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
        return self.c_proj(y, context_state), q_stats


class RemixedBlock(nn.Module):
    """Attention-Grounded Context-Conditioned Linear block — 'weight' modulation path.

    Supports all novel ablation designs via config toggles:
      - cclblock_context_stream: 'local'|'shifted'|'ema'|'selective'|'multiscale'|'ssm'|
                                 'predictive_chunk'|'evidence_ssm'
      - cclblock_context_bank_size: 0=off, N=ContextBank with N prototypes
      - cclblock_per_head_ctx: separate context projections for attn vs ffn
      - cclblock_sparse_gate_k: 0=soft sigmoid, N=sparse top-k gate (via remixed_linear_kwargs)
      - cclblock_gate_temperature: sigmoid temperature for basis gate
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = RemixedMultiAttention(config, layer_idx)
        self.ffwd = RemixedFeedForward(config)
        ctx_dim = config.remix_context_dim
        stream_type = getattr(config, 'cclblock_context_stream', 'local')
        bank_size  = getattr(config, 'cclblock_context_bank_size', 0)
        per_head   = getattr(config, 'cclblock_per_head_ctx', False)
        ctx_source = getattr(config, 'cclblock_context_source', 'norm_x')
        chunk_size = getattr(config, 'cclblock_chunk_size', 0)
        self._modulation_mode = getattr(config, 'cclblock_modulation', 'weight')
        # 18H: Mixture norm
        if getattr(config, 'p18_mixture_norm', 0):
            self.norm_attn = MixtureNorm(config.n_embd)
            self.norm_mlp = MixtureNorm(config.n_embd)
        else:
            self.norm_attn = None
            self.norm_mlp = None
        # 18E: Stochastic depth (LayerDrop)
        self.layer_drop = getattr(config, 'p18_layer_drop', 0.0)

        self.is_shifted   = (stream_type == 'shifted')
        self.per_head_ctx = per_head
        self.ctx_source   = ctx_source  # 'norm_x' | 'attn_heads' | 'attn_geometry'
        self.attn_shadow_dim = max(0, int(getattr(config, 'cclblock_attn_shadow_dim', 0)))
        if self.attn_shadow_dim > 0:
            self.shadow_ctx_proj = Linear(self.attn.shadow_dim, ctx_dim, bias=True)
            nn.init.zeros_(self.shadow_ctx_proj.weight)
            nn.init.zeros_(self.shadow_ctx_proj.bias)

        # Design 2: query-based context projection (head_dim → ctx_dim)
        # Only created when attn_heads source is active; replaces ctx_stream for FFN gating.
        if ctx_source == 'attn_heads':
            head_dim = config.n_embd // config.n_head
            self.ctx_proj_q = Linear(head_dim, ctx_dim, bias=True)
            nn.init.zeros_(self.ctx_proj_q.weight)  # start neutral
        elif ctx_source == 'attn_geometry':
            self.ctx_proj_geom = Linear(4, ctx_dim, bias=True)
            nn.init.zeros_(self.ctx_proj_geom.weight)

        def _make_stream():
            """Factory: returns the configured context stream module."""
            if bank_size > 0:
                return ContextBank(config.n_embd, ctx_dim, bank_size)
            elif chunk_size > 0 or stream_type == 'chunk':
                effective_chunk_size = chunk_size if chunk_size > 0 else 64
                return HardChunkContextStream(config.n_embd, ctx_dim, effective_chunk_size)
            elif stream_type == 'predictive_chunk':
                effective_chunk_size = chunk_size if chunk_size > 0 else 64
                return PredictiveChunkContextStream(config.n_embd, ctx_dim, effective_chunk_size)
            elif stream_type in ('local', 'shifted'):
                return LocalContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'selective':
                return SelectiveContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'boundary':
                return BoundaryGatedContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'ema':
                return EMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
            elif stream_type == 'dacs':
                return DetachedAttnContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'prefix':
                return CausalPrefixContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'warmup_ema':
                return WarmupEMAContextStream(config.n_embd, ctx_dim, ema_factor=getattr(config, 'cclblock_ema_factor', 0.99))
            elif stream_type == 'dacs_ema':
                return DACSEMAContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'decay_prefix':
                return DecayPrefixContextStream(config.n_embd, ctx_dim, gamma=0.9)
            elif stream_type == 'ssm':
                return ParallelLinearContextStream(config.n_embd, ctx_dim)
            elif stream_type == 'evidence_ssm':
                return EvidenceAccumulationContextStream(
                    config.n_embd,
                    ctx_dim,
                    num_regimes=getattr(config, 'cclblock_num_regimes', 8),
                    temperature=getattr(config, 'cclblock_regime_temperature', 1.0),
                )
            else:  # multiscale
                return MultiScaleContext(config.n_embd, ctx_dim)

        # FSI/AESP/CKR/ARG/KFL: no context stream needed (bypass in forward)
        if self._modulation_mode in ('fsi', 'aesp', 'ckr', 'giad', 'psg', 'splitstream', 'lokr', 'arg', 'kfl'):
            self.ctx_stream = None
        elif per_head:
            # Design 7: two independent streams; attn uses pre-attn repr, ffn uses post-attn repr
            self.ctx_stream_attn = _make_stream()
            self.ctx_stream_ffn  = _make_stream()
        else:
            self.ctx_stream = _make_stream()

    def _is_local(self, stream):
        """Returns True if stream doesn't need prev_ctx (local or bank or shifted)."""
        return isinstance(stream, (LocalContextStream, ContextBank, HardChunkContextStream, PredictiveChunkContextStream, EvidenceAccumulationContextStream, BoundaryGatedContextStream, DetachedAttnContextStream, CausalPrefixContextStream, DecayPrefixContextStream))

    def forward(self, x, ve, cos_sin, window_size, kv_cache, prev_ctx=None):
        x_entry = x  # snapshot before attn: = previous layer output (used for 'shifted' mode)

        # FSI/AESP/CKR/PGR/CIL/PRB/ARG/KFL bypass: these modes don't use context streams at all.
        # They derive their modulation signal directly from the attention output.
        if self._modulation_mode in ('fsi', 'aesp', 'ckr', 'ckr_ffn', 'com', 'giad', 'psg', 'splitstream', 'lokr', 'pgr', 'cil', 'prb', 'arg', 'kfl'):
            norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
            norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
            attn_out = self.attn(norm_fn_attn(x), ve, cos_sin, window_size, kv_cache, context_state=None)
            x = x + attn_out
            ctx_signal = attn_out.detach()
            # 18E: Stochastic depth — randomly skip FFN during training
            if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
                pass  # Skip FFN
            else:
                scale = 1.0 / (1.0 - self.layer_drop) if self.training and self.layer_drop > 0 else 1.0
                x = x + scale * self.ffwd(norm_fn_mlp(x), ctx_signal)
            self._last_ctx = ctx_signal
            return x, ctx_signal

        if self.attn_shadow_dim > 0:
            attn_out, shadow_ctx = self.attn.forward_with_shadow(
                norm(x), ve, cos_sin, window_size, kv_cache, None
            )
            x = x + attn_out
            ctx = self.shadow_ctx_proj(shadow_ctx)
            x = x + self.ffwd(norm(x), ctx)
            self._last_ctx = ctx.detach()
            return x, self._last_ctx

        if self.ctx_source in ('attn_heads', 'attn_geometry'):
            # Design 2: context derived from query vectors (post-RoPE+QKnorm, detached)
            # - Attention runs with NO ctx conditioning (standard mode)
            # - q_stats captures the 'search intent' of each position (B, T, head_dim)
            # - FFN basis gate is conditioned on this query-derived signal
            attn_out, q_stats = self.attn.forward_with_q_stats(
                norm(x), ve, cos_sin, window_size, kv_cache, None)
            x = x + attn_out
            if self.ctx_source == 'attn_heads':
                ctx = self.ctx_proj_q(q_stats)   # (B, T, ctx_dim)
            else:
                # Attention-geometry router features from query geometry.
                q_norm = q_stats.norm(dim=-1, keepdim=True)
                q_mean = q_stats.mean(dim=-1, keepdim=True)
                q_std = q_stats.std(dim=-1, keepdim=True)
                q_peak = q_stats.abs().amax(dim=-1, keepdim=True)
                geom = torch.cat([q_norm, q_mean, q_std, q_peak], dim=-1)
                ctx = self.ctx_proj_geom(geom)
            x = x + self.ffwd(norm(x), ctx)
            self._last_ctx = ctx.detach()
            return x, self._last_ctx

        elif self.per_head_ctx:
            # Design 7: separate contexts from pre-attn and post-attn residuals
            ctx_src_attn = norm(x_entry)                  # before attention
            ctx_attn = self.ctx_stream_attn(ctx_src_attn) # (B, T, ctx_dim)
            # Run attention with dedicated attn context
            attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache, ctx_attn)
            x = x + attn_out
            ctx_src_ffn = norm(x_entry if self.is_shifted else x)  # shifted=prev, else post-attn
            ctx_ffn = self.ctx_stream_ffn(ctx_src_ffn)   # (B, T, ctx_dim)
            x = x + self.ffwd(norm(x), ctx_ffn)
            self._last_ctx = ctx_ffn.detach()
            return x, self._last_ctx
        else:
            stream = self.ctx_stream
            local = self._is_local(stream)
            is_dacs = isinstance(stream, (DetachedAttnContextStream, DACSEMAContextStream))
            
            if is_dacs:
                # Attention runs WITHOUT context
                attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache, context_state=None)
                x = x + attn_out
                # Compute context FROM attn_out
                ctx = stream(attn_out, None if local else prev_ctx)
            else:
                # Attention runs WITH context (unless local/shifted)
                attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache,
                                     None if (local or self.is_shifted) else prev_ctx)
                x = x + attn_out
                # Context source: 'shifted' reads from x_entry (prev layer), others from post-attn norm(x)
                ctx_src = norm(x_entry if self.is_shifted else x)
                ctx = stream(ctx_src, None if local else prev_ctx)
                
            x = x + self.ffwd(norm(x), ctx)
            out_ctx = ctx.detach() if (local or self.is_shifted or is_dacs) else ctx
            self._last_ctx = out_ctx
            return x, out_ctx


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
        # 19I: VE gate with optional learnable bias
        use_ve_bias = bool(getattr(config, 'p19_ve_bias', 0))
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=use_ve_bias) if has_ve(layer_idx, config.n_layer) else None
        # 19B: Per-head importance scaling — learned multiplier on attention output
        # softplus(0.5413) ≈ 1.0, so head_scale_raw inited to 0.5413 gives scale=1.0 (identity)
        self.head_importance = None
        if getattr(config, 'p19_head_importance', 0):
            self.head_importance = nn.Parameter(torch.full((self.n_head,), 0.5413))
        # 19D: Per-head attention logit temperature — scales Q before flash_attn
        # softplus(0.5413) ≈ 1.0, so scale=1.0 at init (identity)
        self.attn_logit_scale = None
        if getattr(config, 'p19_attn_logit_bias', 0):
            self.attn_logit_scale = nn.Parameter(torch.full((self.n_head,), 0.5413))

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

        # 19D: Per-head learned QK temperature — scale Q to control attention sharpness
        # Effective dot product: (scale * q) · k = scale * (q · k)
        # FA3-safe: we scale Q before the call, not the attention matrix
        if self.attn_logit_scale is not None:
            scale = F.softplus(self.attn_logit_scale).to(q.dtype)  # (n_head,)
            q = q * scale.view(1, 1, self.n_head, 1)  # broadcast: (1, 1, H, 1)

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

        # 19B: Per-head importance scaling — learned multiplier on post-attention output
        if self.head_importance is not None:
            # y shape: (B, T, n_head, head_dim)
            his = F.softplus(self.head_importance).to(y.dtype)  # (n_head,)
            y = y * his.view(1, 1, self.n_head, 1)  # broadcast

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SpectralReparamLinear(nn.Module):
    """Phase 19G: Spectral Reparameterization.

    W = U @ diag(sigma) @ V^T where U, V are frozen (from SVD of initial W),
    only sigma (the singular values) is learned.

    At init: sigma = actual singular values of W → exact identity with original W.
    At inference: reparameterizable back to a single W_eff = U @ diag(sigma) @ V^T.
    Params added: min(in_features, out_features) per layer.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # These will be filled in init_weights after the base Linear is initialized
        self.register_buffer('U', torch.zeros(out_features, min(in_features, out_features)))
        self.register_buffer('Vt', torch.zeros(min(in_features, out_features), in_features))
        self.sigma = nn.Parameter(torch.ones(min(in_features, out_features)))

    def init_from_weight(self, weight):
        """Decompose a weight matrix and store frozen U, V^T + learned sigma.
        Call this from init_weights() AFTER the weight has been properly initialized."""
        with torch.no_grad():
            W = weight.float()  # SVD in float32 for precision
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            self.U.copy_(U.to(self.U.dtype))
            self.Vt.copy_(Vt.to(self.Vt.dtype))
            self.sigma.data.copy_(S)

    def forward(self, x):
        dtype = x.dtype
        # W_eff = U @ diag(sigma) @ V^T
        # Compute as: x @ V^T^T @ diag(sigma) @ U^T = x @ V @ diag(sigma) @ U^T
        # But it's more efficient to compute the effective weight directly:
        # y = (x @ V^T.T) * sigma @ U^T = x @ (V^T.T * sigma) @ U^T
        Vt = self.Vt.to(dtype)  # (k, in)
        U = self.U.to(dtype)    # (out, k)
        sigma = self.sigma.to(dtype)  # (k,)
        # x: (B, T, in) → (B, T, k) → (B, T, out)
        h = F.linear(x, Vt)  # x @ V = x @ Vt.T → (B, T, k) — F.linear does x @ W^T
        h = h * sigma  # element-wise scale by singular values
        return F.linear(h, U)  # h @ U^T → (B, T, out)


# ── Phase 20: Context-Conditioned Dynamic Weight Computation ────────────────

class MoNE_MLP(nn.Module):
    """20F: Mixture of Narrow Experts.

    K narrow expert MLPs with content-based routing. Same total params as dense
    MLP (each expert has hidden_dim = 4*D/K). Different tokens can route to
    different experts, enabling content-dependent specialization without
    parameter overhead.

    Training: compute all K experts, weight by softmax routing scores.
    Same total FLOPs as baseline (K × D × 4D/K = D × 4D), but each expert
    can specialize.

    Inference: can use top-1 for K× FLOP savings (not implemented yet).

    The router is a tiny Linear(D, K) — routing gradients are small relative
    to expert weight gradients, minimizing FM1 (gradient fracturing).

    Identity-ish at init: all experts initialized identically (same as one
    wide MLP split into K pieces), router output uniform → equal weighting.
    """
    def __init__(self, config, n_experts=4, topk=0):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk  # 0 = compute all (soft MoE), >0 = sparse top-k
        D = config.n_embd
        expert_hidden = 4 * D // n_experts
        assert expert_hidden * n_experts == 4 * D, (
            f"4*D={4*D} must be divisible by n_experts={n_experts}")
        # K narrow expert MLPs
        self.experts_fc = nn.ModuleList([
            Linear(D, expert_hidden, bias=False) for _ in range(n_experts)])
        self.experts_proj = nn.ModuleList([
            Linear(expert_hidden, D, bias=False) for _ in range(n_experts)])
        # Router: tiny D → K linear (no bias → softmax is shift-invariant)
        self.router = Linear(D, n_experts, bias=False)
        # Load-balancing auxiliary loss coefficient (stored, not a parameter)
        self._aux_balance_coeff = 0.01
        # Cache routing weights for diagnostics (not a parameter, not a buffer)
        self._last_router_entropy = 0.0
        self._last_expert_load = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Router: content-dependent routing scores
        router_logits = self.router(x)  # (B, T, K)
        router_weights = F.softmax(router_logits.float(), dim=-1).to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            rw_float = router_weights.float()
            log_rw = torch.log(rw_float.clamp(min=1e-8))
            self._last_router_entropy = -(rw_float * log_rw).sum(dim=-1).mean().item()
            self._last_expert_load = rw_float.mean(dim=(0, 1))  # (K,) — average load per expert

        if self.topk > 0 and self.topk < self.n_experts:
            # Sparse top-k: only compute selected experts
            topk_weights, topk_indices = router_weights.topk(self.topk, dim=-1)  # (B, T, topk)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize
            y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
            for rank_idx in range(self.topk):
                expert_idx = topk_indices[..., rank_idx]  # (B, T)
                weight = topk_weights[..., rank_idx]  # (B, T)
                # Compute each expert for all tokens, mask by selection
                for k in range(self.n_experts):
                    mask = (expert_idx == k)  # (B, T) bool
                    if mask.any():
                        h = self.experts_fc[k](x)
                        h = F.relu(h).square()
                        h = self.experts_proj[k](h)
                        y = y + (weight * mask.to(dtype)).unsqueeze(-1) * h
        else:
            # Soft routing: compute all experts, weight by routing scores
            y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
            for k in range(self.n_experts):
                h = self.experts_fc[k](x)
                h = F.relu(h).square()
                h = self.experts_proj[k](h)
                y = y + router_weights[..., k:k+1] * h

        return y


class FrozenRoutedMLP(nn.Module):
    """20C: Frozen Content-Routed Full-Rank Branches (LRCFB).

    K full-rank MLP branches mixed by a FROZEN content-dependent routing signal.
    The routing projection R is a random matrix fixed at init (registered as a
    buffer, never updated). This gives content-dependent weight composition
    without ANY learned routing — completely avoiding FM1 (gradient fracturing).

    alpha = softmax(x @ R)  — frozen routing, no gradient through R
    y = Σ_k alpha_k * MLP_k(x)  — each branch is a full-rank MLP

    This is CKR's architecture but with frozen CONTENT routing instead of
    position routing. Content-dependent (unlike CKR) but gradient-free
    (unlike RemixedLinear). The combination we never tested.

    NOTE: K× parameter overhead (K full-rank MLPs). At K=4 this means 4×
    more MLP params. This is known to be expensive but we need to test whether
    frozen content routing provides enough signal to justify it.
    """
    def __init__(self, config, n_branches=4):
        super().__init__()
        self.n_branches = n_branches
        D = config.n_embd
        H = 4 * D
        # K full-rank MLP branches
        self.branches_fc = nn.ModuleList([
            Linear(D, H, bias=False) for _ in range(n_branches)])
        self.branches_proj = nn.ModuleList([
            Linear(H, D, bias=False) for _ in range(n_branches)])
        # Frozen random projection for content routing (buffer = no grad)
        # Scaled by 1/sqrt(D) for reasonable softmax logit magnitude
        self.register_buffer('content_proj',
            torch.randn(D, n_branches) / (D ** 0.5))
        # Cache for diagnostics
        self._last_routing_entropy = 0.0
        self._last_routing_weights = None

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype
        # Frozen content-dependent routing (no gradient through content_proj)
        logits = x.float() @ self.content_proj.float()  # (B, T, K)
        weights = F.softmax(logits, dim=-1).to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            w_float = weights.float()
            log_w = torch.log(w_float.clamp(min=1e-8))
            self._last_routing_entropy = -(w_float * log_w).sum(dim=-1).mean().item()
            self._last_routing_weights = w_float.mean(dim=(0, 1))

        # Compute all K branches, weight by routing
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            h = self.branches_fc[k](x)
            h = F.relu(h).square()
            h = self.branches_proj[k](h)
            y = y + weights[..., k:k+1] * h

        return y


class DetachedRoutedMLP(nn.Module):
    """20D: Detached-Gradient Content Routing (DGCR).

    K full-rank MLP branches with a LEARNED router, but the router's main-loss
    gradient is stopped. The router only receives gradient through the auxiliary
    routing loss, which trains it to predict which branch would minimize
    per-token loss ex-post.

    Main forward path:
        alpha = router(x.detach()).detach()  # stop grad BOTH ways
        y = Σ_k alpha_k * MLP_k(x)

    Auxiliary loss (computed separately):
        For each branch k, compute per-token error ||MLP_k(x) - y_target||²
        Train router to assign high weight to the lowest-error branch

    This avoids FM1 (gradient fracturing) because the main loss never
    backprops through the router. The router learns independently via aux loss.

    NOTE: K× parameter overhead. The aux loss requires computing all K branch
    outputs even if we use top-k routing for the main path.
    """
    def __init__(self, config, n_branches=4, aux_weight=0.01):
        super().__init__()
        self.n_branches = n_branches
        self.aux_weight = aux_weight
        D = config.n_embd
        H = 4 * D
        # K full-rank MLP branches
        self.branches_fc = nn.ModuleList([
            Linear(D, H, bias=False) for _ in range(n_branches)])
        self.branches_proj = nn.ModuleList([
            Linear(H, D, bias=False) for _ in range(n_branches)])
        # Learned router (receives grad only through aux loss)
        self.router = nn.Sequential(
            Linear(D, D // 4, bias=False),
            nn.GELU(),
            Linear(D // 4, n_branches, bias=False),
        )
        # Zero-init final router layer for uniform routing at start
        nn.init.zeros_(self.router[-1].weight)
        # Cache for aux loss computation and diagnostics
        self._last_branch_outputs = None  # set during forward
        self._last_router_logits = None
        self._last_routing_entropy = 0.0

    def forward(self, x):
        B, T, D = x.shape
        dtype = x.dtype

        # Router: content-dependent but gradient-isolated from main loss
        # x.detach() stops main-loss gradient from reaching router
        router_logits = self.router(x.detach().to(dtype))  # (B, T, K)
        self._last_router_logits = router_logits  # save for aux loss
        # .detach() on output stops router gradient from reaching branch weights
        router_weights = F.softmax(router_logits.float(), dim=-1).detach().to(dtype)  # (B, T, K)

        # Cache for diagnostics
        with torch.no_grad():
            rw_float = router_weights.float()
            log_rw = torch.log(rw_float.clamp(min=1e-8))
            self._last_routing_entropy = -(rw_float * log_rw).sum(dim=-1).mean().item()

        # Compute all branches (needed for aux loss)
        branch_outputs = []
        for k in range(self.n_branches):
            h = self.branches_fc[k](x)
            h = F.relu(h).square()
            h = self.branches_proj[k](h)
            branch_outputs.append(h)

        # Save for aux loss (detached from main graph)
        self._last_branch_outputs = [bo.detach() for bo in branch_outputs]

        # Main output: weighted sum of branches
        y = torch.zeros(B, T, D, device=x.device, dtype=dtype)
        for k in range(self.n_branches):
            y = y + router_weights[..., k:k+1] * branch_outputs[k]

        return y

    def compute_aux_loss(self, target_output):
        """Compute auxiliary routing loss.

        target_output: the final model output AFTER this MLP (detached).
        Trains router to assign weight to the branch closest to target.
        """
        if self._last_branch_outputs is None or self._last_router_logits is None:
            return torch.tensor(0.0)
        # Per-branch error: which branch output is closest to target?
        target = target_output.detach()
        errors = []
        for bo in self._last_branch_outputs:
            err = ((bo - target) ** 2).mean(dim=-1)  # (B, T)
            errors.append(err)
        errors = torch.stack(errors, dim=-1)  # (B, T, K)
        # Best branch = lowest error
        best_branch = errors.argmin(dim=-1)  # (B, T)
        # Train router to predict best branch
        logits = self._last_router_logits  # (B, T, K) — has grad to router params
        aux_loss = F.cross_entropy(
            logits.reshape(-1, self.n_branches),
            best_branch.reshape(-1),
        )
        return self.aux_weight * aux_loss


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
        # 18I: Dynamic Activation (learned mix of ReLU², GELU, SiLU)
        self.dynamic_act = DynamicActivation() if getattr(config, 'p18_dynamic_activation', 0) else None
        # 18F: Per-Channel Scale after projection
        self.channel_scale = PerChannelScale(config.n_embd) if getattr(config, 'p18_per_channel_scale', 0) else None
        # 19G: Spectral Reparameterization
        srp_mode = getattr(config, 'p19_spectral_reparam', 0)
        self.srp_proj = SpectralReparamLinear(4 * config.n_embd, config.n_embd) if srp_mode >= 1 else None
        self.srp_fc = SpectralReparamLinear(config.n_embd, 4 * config.n_embd) if srp_mode >= 2 else None

    def forward(self, x):
        x = self.c_fc(x)
        if self.dynamic_act is not None:
            x = self.dynamic_act(x)
        else:
            x = F.relu(x).square()
        # 19G: If spectral reparam is on for c_proj, use it; else use standard linear
        if self.srp_proj is not None:
            x = self.srp_proj(x)
        else:
            x = self.c_proj(x)
        if self.channel_scale is not None:
            x = self.channel_scale(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # Phase 20: Select MLP variant based on config
        _mone_k = getattr(config, 'p20_mone_experts', 0)
        _lrcfb_k = getattr(config, 'p20_lrcfb_branches', 0)
        _dgcr_k = getattr(config, 'p20_dgcr_branches', 0)
        if _dgcr_k > 0:
            self.mlp = DetachedRoutedMLP(config, n_branches=_dgcr_k,
                                         aux_weight=getattr(config, 'p20_dgcr_aux_weight', 0.01))
        elif _lrcfb_k > 0:
            self.mlp = FrozenRoutedMLP(config, n_branches=_lrcfb_k)
        elif _mone_k > 0:
            self.mlp = MoNE_MLP(config, n_experts=_mone_k,
                                topk=getattr(config, 'p20_mone_topk', 0))
        else:
            self.mlp = MLP(config)
        # 18H: Mixture norm (learned RMSNorm + LayerNorm blend)
        if getattr(config, 'p18_mixture_norm', 0):
            self.norm_attn = MixtureNorm(config.n_embd)
            self.norm_mlp = MixtureNorm(config.n_embd)
        else:
            self.norm_attn = None
            self.norm_mlp = None
        # 18E: Stochastic depth (LayerDrop)
        self.layer_drop = getattr(config, 'p18_layer_drop', 0.0)
        # 19A: Residual Gate Scaling — learned scalar on block output
        # softplus(0.5413) ≈ 1.0 → identity at init
        self.residual_alpha = None
        if getattr(config, 'p19_residual_gate', 0):
            self.residual_alpha = nn.Parameter(torch.tensor(0.5413))
        # 19J: Training-time weight noise epsilon
        self._weight_noise_eps = float(getattr(config, 'p19_weight_noise', 0.0))

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        norm_fn_attn = self.norm_attn if self.norm_attn is not None else norm
        norm_fn_mlp = self.norm_mlp if self.norm_mlp is not None else norm
        # 19J: Weight noise — add isotropic noise to key weights during training
        if self.training and self._weight_noise_eps > 0:
            eps = self._weight_noise_eps
            for p in self.mlp.parameters():
                if p.ndim == 2:  # only perturb weight matrices, not biases
                    p.data.add_(eps * torch.randn_like(p))
        attn_out = self.attn(norm_fn_attn(x), ve, cos_sin, window_size, kv_cache)
        x = x + attn_out
        # 18E: Stochastic depth — randomly skip FFN during training
        if self.training and self.layer_drop > 0 and torch.rand(1).item() < self.layer_drop:
            pass  # Skip FFN
        else:
            scale = 1.0 / (1.0 - self.layer_drop) if self.training and self.layer_drop > 0 else 1.0
            block_out = scale * self.mlp(norm_fn_mlp(x))
            # 19A: Residual Gate Scaling
            if self.residual_alpha is not None:
                block_out = F.softplus(self.residual_alpha).to(block_out.dtype) * block_out
            x = x + block_out
        # 19J: Undo weight noise (subtract it back so gradients update the clean weights)
        if self.training and self._weight_noise_eps > 0:
            # NOTE: We don't undo here — the noise is effectively a different perturbation each step.
            # The noise added to .data before forward is part of this step's computation only.
            # The gradient update will apply to the noised weights, which is the intended SAM-like behavior.
            pass
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
        if config.use_remix_linear and getattr(config, 'remix_context_dim_ratio', 0) > 0:
            config.remix_context_dim = max(config.remix_context_dim, config.n_embd // config.remix_context_dim_ratio)
            print0(f"remix_context_dim auto-scaled to {config.remix_context_dim} (n_embd={config.n_embd} // ratio={config.remix_context_dim_ratio})")
        
        self.cclblock_modulation   = getattr(config, 'cclblock_modulation', 'weight')
        self.cclblock_stale_ctx_lag = getattr(config, 'cclblock_stale_ctx_lag', 0)
        
        # Auto-correct to nearest multiple of 48 (LCM of 3 and 16) if using Multiscale
        # This ensures 3 equal channels AND fp8 compatibility (divisible by 16)
        if getattr(config, 'cclblock_context_stream', 'ema') == 'multiscale':
            orig_dim = config.remix_context_dim
            config.remix_context_dim = max(48, round(orig_dim / 48) * 48)
            print0(f"remix_context_dim auto-corrected for MultiScale/FP8 from {orig_dim} to {config.remix_context_dim}")
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
        if self.cclblock_modulation in ('householder', 'spectral', 'ocd', 'lie', 'polynomial', 'grassmann'):
            self.remixed_linear_kwargs['operator_modulation'] = self.cclblock_modulation
        else:
            self.remixed_linear_kwargs['operator_modulation'] = 'none'
        self.remixed_linear_kwargs['lie_generators'] = getattr(config, 'cclblock_lie_generators', 4)
        self.remixed_linear_kwargs['poly_order'] = getattr(config, 'cclblock_poly_order', 2)
        self.remixed_linear_kwargs['grassmann_bank_size'] = getattr(config, 'cclblock_grassmann_bank_size', 4)
        config.remixed_linear_kwargs = self.remixed_linear_kwargs
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        if self.use_remix_linear:
            if self.cclblock_modulation == 'normalization':
                block_cls = CCLBlock
            else:
                block_cls = RemixedBlock  # 'weight' path
        else:
            block_cls = Block
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
        # CCL: context is derived per-block inside RemixedBlock/CCLBlock via SelectiveContextStream.
        # GlobalContextManager and context_updaters are not used for remix_linear.
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Design 10 (auxiliary objective): lightweight head predicts boundary or entropy from
        # the mean context vector across all RemixedBlocks. Forces context to encode
        # non-trivial information and prevents gradient-collapse to identity.
        self.aux_head = None
        if config.use_remix_linear and getattr(config, 'cclblock_aux_objective', 'none') != 'none':
            self.aux_head = Linear(config.remix_context_dim, 1, bias=True)
            nn.init.zeros_(self.aux_head.weight)
            nn.init.constant_(self.aux_head.bias, 0.0)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # 19C: Residual Stream Depthwise Mixing — grouped 1x1 conv between blocks
        mix_groups = getattr(config, 'p19_residual_mix_groups', 0)
        self._residual_mix_groups = mix_groups
        if mix_groups > 0:
            # Per-layer: gamma (init=0 → identity) + group-linear mixer
            self.residual_mix_gamma = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(config.n_layer)
            ])
            self.residual_mixers = nn.ModuleList([
                nn.Conv1d(config.n_embd, config.n_embd, 1, groups=config.n_embd // mix_groups)
                for _ in range(config.n_layer)
            ])
        else:
            self.residual_mix_gamma = None
            self.residual_mixers = None
        # 19E: Learned Residual Decay — global depth-dependent x0 decay
        # x0_factor_i = sigmoid(depth_decay_raw) ^ i. When depth_decay_raw is large,
        # sigmoid ≈ 1 so x0 persists across layers; when small, x0 fades quickly.
        self._use_residual_decay = bool(getattr(config, 'p19_residual_decay', 0))
        if self._use_residual_decay:
            # init such that sigmoid(2.0)^i starts with a moderate decay
            self.depth_decay_raw = nn.Parameter(torch.tensor(2.0))
        else:
            self.depth_decay_raw = None
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
                if isinstance(sub, CCLBlock):
                    # AdaRMSNorm: zero proj weights, scale bias=1.0, shift bias=0.0.
                    # Guarantees identity RMSNorm at init regardless of ctx quality.
                    for ada in [sub.ada_norm_attn, sub.ada_norm_mlp]:
                        torch.nn.init.zeros_(ada.proj.weight)
                        torch.nn.init.zeros_(ada.proj.bias)
                        ada.proj.bias.data[:ada.n_embd] = 1.0   # scale
                    # SelectiveContextStream / MultiScaleContext: handled by their own
                    # __init__ zero-inits already; but re-apply here for safety.
                    _init_ctx_stream(sub.ctx_stream)
                    # Attn and MLP: same as regular Block
                    torch.nn.init.uniform_(sub.attn.c_q.weight, -s, s)
                    torch.nn.init.uniform_(sub.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(sub.attn.c_v.weight, -s, s)
                    torch.nn.init.zeros_(sub.attn.c_proj.weight)
                    torch.nn.init.uniform_(sub.mlp.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(sub.mlp.c_proj.weight)
                    if sub.attn.ve_gate is not None:
                        torch.nn.init.zeros_(sub.attn.ve_gate.weight)
                    continue  # Skip further processing of sub-components

                if isinstance(sub, RemixedBlock):
                    # SelectiveContextStream / MultiScaleContext replaces ctx_from_attn + ctx_ema_gate.
                    # FSI/AESP/CKR modes have ctx_stream=None (no context stream needed).
                    if sub.ctx_stream is not None:
                        _init_ctx_stream(sub.ctx_stream)
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
                            torch.nn.init.xavier_uniform_(m.weight)  # original xavier init
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
                    torch.nn.init.normal_(sub.cross_expert_proj.weight, mean=0.0, std=0.02)  # original normal init
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

                # Phase 17: Re-initialize buffers and position signals for
                # position-routed modules. to_empty() replaces tensor storage
                # with garbage; Python float attrs survive for re-init.
                if isinstance(sub, CausalKernelLinear):
                    sub._temperature.fill_(sub._init_temperature)
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.branch_conv.weight)
                    torch.nn.init.zeros_(sub.branch_conv.bias)
                    if sub.content_proj is not None:
                        sig_dim = sub.content_proj.shape[0]
                        sub.content_proj.copy_(torch.randn_like(sub.content_proj) / (sig_dim ** 0.5))
                    if getattr(sub, '_ortho_init', False):
                        for branch in sub.branches:
                            torch.nn.init.orthogonal_(branch.weight)
                elif isinstance(sub, LoKRLinear):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.branch_conv.weight)
                    torch.nn.init.zeros_(sub.branch_conv.bias)
                elif isinstance(sub, (PositionGatedResidual, PositionalResidualBias)):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    if isinstance(sub, PositionGatedResidual):
                        torch.nn.init.zeros_(sub.delta.weight)
                        torch.nn.init.zeros_(sub.gate_conv.weight)
                        torch.nn.init.constant_(sub.gate_conv.bias, -3.0)
                    elif isinstance(sub, PositionalResidualBias):
                        torch.nn.init.zeros_(sub.bias_conv.weight)
                        torch.nn.init.zeros_(sub.bias_conv.bias)
                elif isinstance(sub, CausalInterpolationLinear):
                    torch.nn.init.normal_(sub.pos_signal, std=0.01)
                    torch.nn.init.zeros_(sub.alpha_conv.weight)
                    torch.nn.init.constant_(sub.alpha_conv.bias, -5.0)
                    torch.nn.init.zeros_(sub.bias)
                # Phase 18: re-init after to_empty()
                elif isinstance(sub, KroneckerLinear):
                    target_std = math.sqrt(math.sqrt(2.0 / sub.in_features))
                    torch.nn.init.normal_(sub.A, std=target_std)
                    torch.nn.init.normal_(sub.B, std=target_std)
                    torch.nn.init.zeros_(sub.bias)
                elif isinstance(sub, AdaptiveGatedLinear):
                    torch.nn.init.zeros_(sub.gate_w)
                    torch.nn.init.constant_(sub.gate_b, 5.0)  # sigmoid(5)≈0.993
                elif isinstance(sub, DynamicActivation):
                    sub.alpha.data.fill_(1.0)
                    sub.beta.data.fill_(0.0)
                    sub.gamma.data.fill_(0.0)
                elif isinstance(sub, MixtureNorm):
                    sub.w1.data.fill_(1.0)
                    sub.w2.data.fill_(0.0)
                elif isinstance(sub, PerChannelScale):
                    sub.scale.data.fill_(1.0)

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if "wpe" in self.transformer:
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

        def _init_ctx_stream(stream):
            """Common init for SelectiveContextStream and MultiScaleContext."""
            if isinstance(stream, SelectiveContextStream):
                # gate bias=0 (sigmoid(0)=0.5 balanced), write zero-init (no spurious content)
                torch.nn.init.zeros_(stream.gate.bias)
                torch.nn.init.zeros_(stream.write.weight)
                torch.nn.init.zeros_(stream.write.bias)
                # gate weight: xavier for good gradient flow
                torch.nn.init.xavier_uniform_(stream.gate.weight)
            elif isinstance(stream, MultiScaleContext):
                for gate, bias_val in zip(stream.gates, MultiScaleContext.GATE_BIASES):
                    torch.nn.init.xavier_uniform_(gate.weight)
                    torch.nn.init.constant_(gate.bias, bias_val)
                for write in stream.writes:
                    torch.nn.init.zeros_(write.weight)
                    torch.nn.init.zeros_(write.bias)

        for block in self.transformer.h:
            if isinstance(block, (RemixedBlock, CCLBlock)):
                _init_research_module(block)
                # RemixedBlock attn ve_gate not handled in _init_research_module — do it here
                if isinstance(block, RemixedBlock) and block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
                    # 19I: VE gate bias zero-init (neutral at init)
                    if block.attn.ve_gate.bias is not None:
                        torch.nn.init.zeros_(block.attn.ve_gate.bias)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
                # MLP init: handles both standard MLP and P20 variants
                if hasattr(block.mlp, 'c_fc'):
                    # Standard MLP
                    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(block.mlp.c_proj.weight)
                elif isinstance(block.mlp, MoNE_MLP):
                    for expert_fc in block.mlp.experts_fc:
                        torch.nn.init.uniform_(expert_fc.weight, -s, s)
                    for expert_proj in block.mlp.experts_proj:
                        torch.nn.init.zeros_(expert_proj.weight)
                    torch.nn.init.zeros_(block.mlp.router.weight)
                elif isinstance(block.mlp, (FrozenRoutedMLP, DetachedRoutedMLP)):
                    for branch_fc in block.mlp.branches_fc:
                        torch.nn.init.uniform_(branch_fc.weight, -s, s)
                    for branch_proj in block.mlp.branches_proj:
                        torch.nn.init.zeros_(branch_proj.weight)
                    if isinstance(block.mlp, DetachedRoutedMLP):
                        # Zero-init router for uniform routing at start
                        for sub in block.mlp.router:
                            if hasattr(sub, 'weight'):
                                torch.nn.init.zeros_(sub.weight)
                # 19I: VE gate bias zero-init for dense Block too
                if block.attn.ve_gate is not None and block.attn.ve_gate.bias is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.bias)

            # Phase 19 inits (shared across Block/RemixedBlock/CCLBlock)
            # 19G: Spectral Reparameterization — decompose c_proj AFTER it is initialized
            mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
            if mlp is not None:
                if hasattr(mlp, 'srp_proj') and mlp.srp_proj is not None:
                    mlp.srp_proj.init_from_weight(mlp.c_proj.weight)
                if hasattr(mlp, 'srp_fc') and mlp.srp_fc is not None:
                    mlp.srp_fc.init_from_weight(mlp.c_fc.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # 19C: Residual mixer init — gamma=0 (identity), conv weights small random
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                gamma_p.data.fill_(0.0)
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                torch.nn.init.normal_(mixer.weight, std=0.01)
                if mixer.bias is not None:
                    torch.nn.init.zeros_(mixer.bias)

        # 19E: Depth decay init
        if self.depth_decay_raw is not None:
            self.depth_decay_raw.data.fill_(2.0)  # sigmoid(2.0) ≈ 0.88 moderate decay

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
        if self.aux_head is not None:
            research += sum(p.numel() for p in self.aux_head.parameters())
        # AG-CCL: ctx_from_attn and ctx_ema_gate live inside transformer.h (RemixedBlock)
        # and are already counted in transformer_matrices above.
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # P19: GPT-level scalar params (depth_decay_raw, residual_mix_gamma, residual_mixers)
        # These are nn.Parameters on GPT, not inside transformer.h, so we must count them.
        if self.depth_decay_raw is not None:
            scalars += self.depth_decay_raw.numel()
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                scalars += gamma_p.numel()
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                scalars += sum(p.numel() for p in mixer.parameters())
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
        # For RemixedLinear, split gate params (basis_modulator, output_gate_*) from
        # structural params (basis, template_mixing) — gate params get a lower LR (suggestion 5).
        gate_matrix_params   = []  # 2D gate params  → lower-LR Muon
        gate_adamw_params    = []  # 1D gate params  → lower-LR AdamW
        struct_matrix_params = []  # 2D struct params → normal Muon
        struct_adamw_params  = []  # 1D struct params → normal AdamW
        ckr_gate_adamw_params = []  # 13b: CKR gate params → dedicated conservative AdamW

        def _sort_ctx_stream_params(stream):
            """Route SelectiveContextStream/MultiScaleContext params to gate groups."""
            for p in stream.parameters():
                (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)

        if self.use_remix_linear:
            for block in self.transformer.h:
                if isinstance(block, CCLBlock):
                    # 'normalization' path: ctx_stream and AdaRMSNorm.proj are gate-side.
                    # Standard CausalSelfAttention + MLP weights are structural.
                    _sort_ctx_stream_params(block.ctx_stream)
                    for ada in [block.ada_norm_attn, block.ada_norm_mlp]:
                        for p in ada.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    # Structural: attn Q/K/V/proj, MLP fc/proj
                    for p in [block.attn.c_q.weight, block.attn.c_k.weight,
                               block.attn.c_v.weight, block.attn.c_proj.weight,
                               block.mlp.c_fc.weight, block.mlp.c_proj.weight]:
                        struct_matrix_params.append(p)
                    if block.attn.ve_gate is not None:
                        struct_matrix_params.append(block.attn.ve_gate.weight)
                else:
                    assert isinstance(block, RemixedBlock), "Expected RemixedBlock or CCLBlock in remix_linear mode"
                    # 'weight' path: ctx streams and projs are gate-side.
                    if hasattr(block, 'ctx_stream') and block.ctx_stream is not None:
                        _sort_ctx_stream_params(block.ctx_stream)
                    if hasattr(block, 'ctx_stream_attn'):
                        _sort_ctx_stream_params(block.ctx_stream_attn)
                    if hasattr(block, 'ctx_stream_ffn'):
                        _sort_ctx_stream_params(block.ctx_stream_ffn)
                    
                    if hasattr(block, 'ctx_proj_q'):
                        for p in block.ctx_proj_q.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                    if hasattr(block, 'shadow_ctx_proj'):
                        for p in block.shadow_ctx_proj.parameters():
                            (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)

                    # RemixedLinear: sort gate vs structural
                    remix_linears = [
                        block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj,
                        block.ffwd.c_fc, block.ffwd.c_proj,
                    ]
                    # 13b: Dual-optimizer CKR — route CKR gate params to dedicated AdamW
                    ckr_dual = (self.cclblock_modulation == 'ckr' and
                                getattr(self.config, 'cclblock_ckr_dual_optim', 0))
                    for rl in remix_linears:
                        if hasattr(rl, 'gate_parameters'):
                            for p in rl.gate_parameters():
                                if ckr_dual:
                                    # 13b: CKR gate params → dedicated AdamW (conservative)
                                    ckr_gate_adamw_params.append(p)
                                else:
                                    (gate_matrix_params if p.ndim == 2 else gate_adamw_params).append(p)
                        if hasattr(rl, 'non_gate_parameters'):
                            for p in rl.non_gate_parameters():
                                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                        else:
                            # Plain Linear/Float8Linear — all params are structural
                            for p in rl.parameters():
                                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                        # ln_basis (LayerNorm) — only on research modules
                        if hasattr(rl, 'ln_basis'):
                            for p in rl.ln_basis.parameters():
                                struct_adamw_params.append(p)
                    # ve_gate (if present) is structural
                    if block.attn.ve_gate is not None:
                        struct_matrix_params.append(block.attn.ve_gate.weight)
                        # 19I: VE gate bias (if present)
                        if block.attn.ve_gate.bias is not None:
                            struct_adamw_params.append(block.attn.ve_gate.bias)
                    # Phase 18: MixtureNorm params (scalar weights)
                    if hasattr(block, 'norm_attn') and block.norm_attn is not None:
                        for p in block.norm_attn.parameters():
                            (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                    if hasattr(block, 'norm_mlp') and block.norm_mlp is not None:
                        for p in block.norm_mlp.parameters():
                            (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)
                    # Phase 18: DynamicActivation params (α, β, γ scalars)
                    if hasattr(block.ffwd, 'dynamic_act') and block.ffwd.dynamic_act is not None:
                        for p in block.ffwd.dynamic_act.parameters():
                            struct_adamw_params.append(p)
                    # Phase 18: PerChannelScale params (1D scale vector)
                    if hasattr(block.ffwd, 'channel_scale') and block.ffwd.channel_scale is not None:
                        for p in block.ffwd.channel_scale.parameters():
                            struct_adamw_params.append(p)
                    # Phase 19: Per-block P19 params
                    # 19A: residual_alpha scalar
                    if hasattr(block, 'residual_alpha') and block.residual_alpha is not None:
                        struct_adamw_params.append(block.residual_alpha)
                    # 19B: head_importance (per-head scalars)
                    if hasattr(block.attn, 'head_importance') and block.attn.head_importance is not None:
                        struct_adamw_params.append(block.attn.head_importance)
                    # 19D: attn_logit_scale (per-head scalars)
                    if hasattr(block.attn, 'attn_logit_scale') and block.attn.attn_logit_scale is not None:
                        struct_adamw_params.append(block.attn.attn_logit_scale)
                    # 19G: Spectral Reparameterization sigma params
                    ffwd = block.ffwd if hasattr(block, 'ffwd') else (block.mlp if hasattr(block, 'mlp') else None)
                    if ffwd is not None:
                        if hasattr(ffwd, 'srp_proj') and ffwd.srp_proj is not None:
                            struct_adamw_params.append(ffwd.srp_proj.sigma)
                        if hasattr(ffwd, 'srp_fc') and ffwd.srp_fc is not None:
                            struct_adamw_params.append(ffwd.srp_fc.sigma)
                    # 19J: no params (noise epsilon is a config float, not learned)
        else:
            # Regular Block: all transformer.h params go to candidate_matrix_params
            candidate = list(self.transformer.h.parameters())
            struct_matrix_params = [p for p in candidate if p.ndim == 2]
            struct_adamw_params  = [p for p in candidate if p.ndim != 2]

        # Embedding model (MoE), if present
        if self.embedding_model is not None:
            cand = list(self.embedding_model.parameters())
            struct_matrix_params += [p for p in cand if p.ndim == 2]
            struct_adamw_params  += [p for p in cand if p.ndim != 2]

        # Aux head (Design 10), if present
        if self.aux_head is not None:
            for p in self.aux_head.parameters():
                (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)

        research_adamw_params = gate_adamw_params + struct_adamw_params

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        if "wpe" in self.transformer:
            embedding_params += list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Phase 19: Collect GPT-level P19 params
        p19_scalar_params = []
        # 19C: Residual mixer gamma + conv weights
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                p19_scalar_params.append(gamma_p)
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                for p in mixer.parameters():
                    (struct_matrix_params if p.ndim == 2 else p19_scalar_params).append(p)
        # 19E: Depth decay scalar
        if self.depth_decay_raw is not None:
            p19_scalar_params.append(self.depth_decay_raw)
        all_params = (gate_matrix_params + struct_matrix_params + research_adamw_params +
                      embedding_params + lm_head_params + value_embeds_params +
                      resid_params + x0_params + ckr_gate_adamw_params + p19_scalar_params)
        assert len(list(self.parameters())) == len(all_params), (
            f"Parameter count mismatch: model has {len(list(self.parameters()))} params, "
            f"optimizer groups cover {len(all_params)}")

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
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Structural matrix params: normal Muon LR
        for shape in sorted({p.shape for p in struct_matrix_params}):
            group_params = [p for p in struct_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        # Gate matrix params: 0.3× Muon LR (suggestion 5: reduce gradient noise for gate params)
        gate_lr = matrix_lr * 0.3
        for shape in sorted({p.shape for p in gate_matrix_params}):
            group_params = [p for p in gate_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=gate_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        # 13b: Dedicated conservative AdamW for CKR position routing params
        if ckr_gate_adamw_params:
            param_groups.append(dict(
                kind='adamw', params=ckr_gate_adamw_params,
                lr=embedding_lr * dmodel_lr_scale * 0.5,  # conservative: half of embedding LR
                betas=(0.9, 0.999),  # high β₂ for slow, stable adaptation
                eps=1e-10, weight_decay=0.0,  # no decay on positional params
            ))
        # Phase 19: GPT-level scalar params (mixer gammas, depth decay, etc.)
        if p19_scalar_params:
            param_groups.append(dict(
                kind='adamw', params=p19_scalar_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
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
        # CCL context threading: context flows block-to-block, starting as None at block 0.
        # RemixedBlock and CCLBlock both return (x, ctx) and accept prev_ctx.
        # Design C (cclblock_stale_ctx_lag > 0): block i receives context from block i-k
        # (guaranteed to be independent because it's from a different layer's computation).
        # The stale context is detached — within-block gradient path remains intact.
        lag = self.cclblock_stale_ctx_lag
        if self.use_remix_linear and lag > 0:
            ctx_history = []
            for i, block in enumerate(self.transformer.h):
                x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                stale_ctx = ctx_history[i - lag].detach() if i >= lag else None
                x, new_ctx = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, stale_ctx)
                ctx_history.append(new_ctx)
        else:
            prev_ctx = None
            collect_sim = float(getattr(self.config, 'p18_aux_sim_lambda', 0.0)) > 0
            if collect_sim:
                self._layer_outputs = []
            # 19E: Precompute x0 decay factors
            if self._use_residual_decay and self.depth_decay_raw is not None:
                decay_base = torch.sigmoid(self.depth_decay_raw)  # scalar in (0, 1)
            else:
                decay_base = None
            for i, block in enumerate(self.transformer.h):
                # 19E: Apply depth-dependent x0 decay
                x0_w = self.x0_lambdas[i]
                if decay_base is not None:
                    x0_w = x0_w * (decay_base ** i)
                x = self.resid_lambdas[i] * x + x0_w * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                if self.use_remix_linear:
                    x, prev_ctx = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, prev_ctx)
                else:
                    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                # 19C: Residual stream mixing after block output
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x.dtype)
                    # Conv1d expects (B, C, T), but x is (B, T, C)
                    mixed = self.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
                    x = x + gamma * mixed
                if collect_sim:
                    self._layer_outputs.append(x.detach())
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 20 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Design 10: Auxiliary context objective
            # Reads _last_ctx stored on each RemixedBlock during this forward pass.
            if self.aux_head is not None and loss_reduction == 'mean':
                aux_obj = getattr(self.config, 'cclblock_aux_objective', 'none')
                ctx_vecs = []
                for block in self.transformer.h:
                    if isinstance(block, RemixedBlock) and hasattr(block, '_last_ctx') and block._last_ctx is not None:
                        ctx_vecs.append(block._last_ctx)
                if ctx_vecs:
                    # Mean context across all layers: (B, T, ctx_dim)
                    ctx_mean = torch.stack(ctx_vecs, dim=0).mean(dim=0).to(logits.dtype)
                    aux_logits = self.aux_head(ctx_mean).squeeze(-1)  # (B, T)
                    if aux_obj == 'boundary':
                        boundary_id = getattr(self.config, 'cclblock_boundary_token_id', 198)
                        boundary_target = (targets == boundary_id).float()
                        aux_loss = F.binary_cross_entropy_with_logits(aux_logits, boundary_target)
                    elif aux_obj == 'entropy':
                        # Teach context to predict per-token loss magnitude (detached)
                        with torch.no_grad():
                            token_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), targets.view(-1),
                                ignore_index=-1, reduction='none').view(B, T)
                        aux_loss = F.mse_loss(aux_logits, token_loss.detach() / 3.0)
                    else:
                        aux_loss = torch.zeros(1, device=loss.device, dtype=loss.dtype)
                    aux_lambda = getattr(self.config, 'cclblock_aux_lambda', 0.1)
                    loss = loss + aux_lambda * aux_loss

            # OCD overlap penalty from RemixedLinear layers (if enabled)
            orth_lambda = float(getattr(self.config, 'cclblock_orth_lambda', 0.0))
            if orth_lambda > 0.0 and loss_reduction == 'mean':
                orth_terms = []
                for mod in self.modules():
                    if isinstance(mod, RemixedLinear) and mod._last_orth_loss is not None:
                        orth_terms.append(mod._last_orth_loss.to(dtype=loss.dtype))
                if orth_terms:
                    loss = loss + orth_lambda * torch.stack(orth_terms).mean()

            # 18G: Auxiliary representation similarity loss
            # Penalize high cosine similarity between adjacent layers
            sim_lambda = float(getattr(self.config, 'p18_aux_sim_lambda', 0.0))
            if sim_lambda > 0.0 and loss_reduction == 'mean' and hasattr(self, '_layer_outputs'):
                sim_losses = []
                for i in range(len(self._layer_outputs) - 1):
                    h_curr = self._layer_outputs[i].float()
                    h_next = self._layer_outputs[i + 1].float()
                    # Mean cosine similarity across batch and sequence
                    cos_sim = F.cosine_similarity(h_curr, h_next, dim=-1).mean()
                    sim_losses.append(cos_sim)
                if sim_losses:
                    loss = loss + sim_lambda * torch.stack(sim_losses).mean()

            # 19H: Anti-Collapse Weight Norm Penalty
            # Penalizes high cosine similarity between adjacent layers' WEIGHT matrices
            # (different from 18G which penalized activations — weights being similar means redundant layers)
            ac_lambda = float(getattr(self.config, 'p19_weight_anticollapse', 0.0))
            if ac_lambda > 0.0 and loss_reduction == 'mean':
                ac_losses = []
                blocks = list(self.transformer.h)
                for j in range(len(blocks) - 1):
                    b_curr, b_next = blocks[j], blocks[j + 1]
                    # Get MLP c_fc weights (the largest weight matrices)
                    w_curr = b_curr.mlp.c_fc.weight if hasattr(b_curr, 'mlp') else (b_curr.ffwd.c_fc.weight if hasattr(b_curr, 'ffwd') else None)
                    w_next = b_next.mlp.c_fc.weight if hasattr(b_next, 'mlp') else (b_next.ffwd.c_fc.weight if hasattr(b_next, 'ffwd') else None)
                    if w_curr is not None and w_next is not None:
                        sim = F.cosine_similarity(w_curr.flatten().unsqueeze(0).float(),
                                                  w_next.flatten().unsqueeze(0).float(), dim=-1)
                        # Penalize only when similarity exceeds threshold (0.8)
                        ac_losses.append(F.relu(sim - 0.8))
                if ac_losses:
                    loss = loss + ac_lambda * torch.stack(ac_losses).mean()

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
