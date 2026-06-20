"""
Modular Sub-Transformer (MST) Architecture.

Each transformer layer contains N parallel sub-transformers operating at
dimension d = D/N. Sub-transformers can specialize in different aspects
of the representation through configurable input partitioning, routing,
FFN structure, layer-to-layer transitions, and final output combination.

Design axes (from modular_transformer_experiment_plan.md):
  Axis 1 (Input):      How sub-transformers receive initial tokens
  Axis 2 (Routing):    How outputs are combined/selected
  Axis 3 (FFN):        Internal FFN structure (d->4d->d vs d->4d)
  Axis 4 (Transition): How outputs flow between layers
  Axis 5 (Final):      How final layer produces vocabulary logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.gpt import Linear, apply_rotary_emb, norm, GPTConfig, has_ve
from nanochat.flash_attention import flash_attn


class SubTransformerAttention(nn.Module):
    """Self-attention for a single sub-transformer at dimension d.

    Supports expanded QKV projections: if head_dim > d//n_head, Q/K/V are
    projected to n_head*head_dim then back to d, keeping residual at d.
    Includes RoPE and QK-norm, matching the base GPT attention style.
    """

    def __init__(self, d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx=0):
        super().__init__()
        self.d = d
        self.n_head = n_head
        self.head_dim = head_dim
        self.qkv_dim = n_head * head_dim   # may be > d when head_dim is expanded
        self.layer_idx = layer_idx
        self.sub_layer_idx = sub_layer_idx  # unique cache slot index = layer_idx * N_subs + sub_idx
        assert d % n_head == 0 or head_dim > 0, f"sub_dim {d} not divisible by n_head {n_head}"

        self.c_q = Linear(d, self.qkv_dim, bias=False)
        self.c_k = Linear(d, self.qkv_dim, bias=False)
        self.c_v = Linear(d, self.qkv_dim, bias=False)
        self.c_proj = Linear(self.qkv_dim, d, bias=False)

        # Value embeddings (ResFormer-style): alternating layers, last always included.
        # Gate maps from input (dim d) to per-head scales (n_head), VE is reshaped into head space.
        self.ve_gate_channels = min(d, 32)
        self.ve_gate = Linear(self.ve_gate_channels, n_head, bias=False) if has_ve(layer_idx, n_layer) else None

    def forward(self, x, cos_sin, ve=None, window_size=(-1, 0), kv_cache=None, total_sub_layers=1):
        """x: (B, T, d), cos_sin: (cos, sin) for RoPE, ve: optional (B, T, d) value embedding.
        kv_cache: optional KVCache for inference. total_sub_layers: n_layer * N_subs."""
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head.
        # VE is at dim d; repeat-interleave to fill head_dim if head_dim > d//n_head.
        if ve is not None and self.ve_gate is not None:
            base_head_dim = self.d // self.n_head
            if self.head_dim > base_head_dim:
                # Tile VE heads to match the expanded head_dim
                repeats = self.head_dim // base_head_dim
                ve_heads = ve.view(B, T, self.n_head, base_head_dim).repeat_interleave(repeats, dim=-1)
            else:
                ve_heads = ve.view(B, T, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_head)
            v = v + gate.unsqueeze(-1) * ve_heads

        cos, sin = cos_sin
        # Handle progressive merge: different levels have different head_dim
        # but share pre-computed rotary embeddings at max head_dim.
        half_hd = self.head_dim // 2
        cos_dim = cos.shape[-1]
        if cos_dim >= half_hd:
            # cos is at max head_dim — slice down to this level's head_dim
            cos = cos[..., :half_hd]
            sin = sin[..., :half_hd]
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        else:
            # cos is smaller than this head_dim — apply partial RoPE
            # Rotate first 2*cos_dim dims, pass rest through unchanged
            rot_dim = cos_dim * 2
            q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rot = apply_rotary_emb(q_rot, cos, sin)
            k_rot = apply_rotary_emb(k_rot, cos, sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            # Training: standard causal flash attention with optional sliding window
            y = flash_attn.flash_attn_func(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                causal=True, window_size=window_size
            )
        else:
            # Inference: use flash_attn_with_kvcache — updates cache in-place
            k_cache, v_cache = kv_cache.get_layer_cache(self.sub_layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position counter after the last sub-layer processes
            if self.sub_layer_idx == total_sub_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, self.qkv_dim)
        return self.c_proj(y)


class SubTransformerFFN(nn.Module):
    """FFN for a sub-transformer. Two modes:
    - 'standard': d -> 4d -> d (with relu² activation)
    - 'no_downproj': d -> 4d (output is 4d, aggregation happens in expanded space)
    - 'linear':      d -> d (single projection, no expansion — minimal FFN)
    """

    def __init__(self, d, mode='standard', inner_dim=0):
        super().__init__()
        self.mode = mode
        if mode == 'linear':
            self.c_fc = Linear(d, d, bias=False)
            self.c_proj = None
        else:
            actual_inner = inner_dim if inner_dim > 0 else 4 * d
            self.c_fc = Linear(d, actual_inner, bias=False)
            if mode == 'standard':
                self.c_proj = Linear(actual_inner, d, bias=False)
            else:
                self.c_proj = None  # no down-projection

    @property
    def out_dim(self):
        return self.c_fc.out_features if self.mode == 'no_downproj' else self.c_fc.in_features

    def forward(self, x):
        x = self.c_fc(x)
        if self.mode == 'linear':
            return x  # simple linear mixing, no nonlinearity
        x = F.relu(x).square()
        if self.c_proj is not None:
            x = self.c_proj(x)
        return x


class SubTransformerBlock(nn.Module):
    """A single sub-transformer block: pre-norm attention + FFN with residuals.
    When sub_layers > 1, contains multiple internal attention+FFN layers."""

    def __init__(self, d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx=0,
                 ffn_mode='standard', ffn_inner_dim=0, sub_layers=1):
        super().__init__()
        self.sub_layers = sub_layers
        self.attn = SubTransformerAttention(d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx)
        self.ffn = SubTransformerFFN(d, mode=ffn_mode, inner_dim=ffn_inner_dim)
        # no_downproj: FFN outputs 4d — need a block-level projection back to d for the residual.
        # This decouples the expansion (d→4d + relu²) from compression (plain 4d→d linear),
        # which is the inductive bias difference being tested vs standard mode.
        self.ffn_out_proj = Linear(4 * d, d, bias=False) if ffn_mode == 'no_downproj' else None
        self.d = d

        # SL1: Additional internal layers for multi-layer subs
        if sub_layers > 1:
            self.extra_attns = nn.ModuleList([
                SubTransformerAttention(d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx)
                for _ in range(sub_layers - 1)
            ])
            self.extra_ffns = nn.ModuleList([
                SubTransformerFFN(d, mode=ffn_mode, inner_dim=ffn_inner_dim)
                for _ in range(sub_layers - 1)
            ])
            if ffn_mode == 'no_downproj':
                self.extra_ffn_out_projs = nn.ModuleList([
                    Linear(4 * d, d, bias=False) for _ in range(sub_layers - 1)
                ])
            else:
                self.extra_ffn_out_projs = None

    def forward(self, x, cos_sin, ve=None, window_size=(-1, 0), kv_cache=None, total_sub_layers=1):
        """x: (B, T, d). Returns (B, T, d)."""
        # First internal layer
        x = x + self.attn(norm(x), cos_sin, ve=ve, window_size=window_size,
                          kv_cache=kv_cache, total_sub_layers=total_sub_layers)
        if not getattr(self, '_skip_ffn', False):
            if self.ffn.mode in ('standard', 'linear'):
                x = x + self.ffn(norm(x))
            else:  # no_downproj
                x = x + self.ffn_out_proj(self.ffn(norm(x)))

        # Additional internal layers (SL1)
        if self.sub_layers > 1:
            for i in range(self.sub_layers - 1):
                x = x + self.extra_attns[i](norm(x), cos_sin, ve=ve, window_size=window_size,
                                             kv_cache=kv_cache, total_sub_layers=total_sub_layers)
                if not getattr(self, '_skip_ffn', False):
                    if self.ffn.mode in ('standard', 'linear'):
                        x = x + self.extra_ffns[i](norm(x))
                    elif self.extra_ffn_out_projs is not None:
                        x = x + self.extra_ffn_out_projs[i](self.extra_ffns[i](norm(x)))
        return x


class MSTInputLayer(nn.Module):
    """Axis 1: How sub-transformers receive initial token representations.

    Modes:
      'fixed_slice':    Slice D-dim embedding into N chunks of d each.
      'learned_proj':   Each sub has a learned W_i in R^{D x d}.
      'rotated_slice':  Apply orthogonal rotation R then slice.
      'per_sub_embed':  N separate embedding tables (requires vocab_size).
      'stem':           Shared small transformer on d-dim embeddings.
    """

    def __init__(self, D, d, N, mode='fixed_slice', vocab_size=0,
                 rotated_slice_learned=False, stem_depth=2, n_head=4):
        super().__init__()
        self.D = D
        self.d = d
        self.N = N
        self.mode = mode

        if mode == 'learned_proj':
            self.projections = nn.ModuleList([
                Linear(D, d, bias=False) for _ in range(N)
            ])
        elif mode == 'rotated_slice':
            if rotated_slice_learned:
                self.rotation = nn.Parameter(torch.eye(D))
            else:
                # Random orthogonal matrix (frozen)
                Q, _ = torch.linalg.qr(torch.randn(D, D))
                self.register_buffer('rotation', Q)
        elif mode == 'per_sub_embed':
            assert vocab_size > 0, "per_sub_embed requires vocab_size"
            self.sub_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, d) for _ in range(N)
            ])
        elif mode == 'stem':
            # Small shared transformer on d-dim before distributing
            self.stem_proj = Linear(D, d, bias=False)
            self.stem_blocks = nn.ModuleList([
                SubTransformerBlock(d, n_head, d // n_head, layer_idx=i, n_layer=stem_depth)
                for i in range(stem_depth)
            ])

    def forward(self, x, input_ids=None):
        """x: (B, T, D) full embedding. Returns list of N tensors (B, T, d)."""
        B, T, D = x.shape

        if self.mode == 'fixed_slice':
            return [x[..., i*self.d:(i+1)*self.d] for i in range(self.N)]
        elif self.mode == 'learned_proj':
            return [proj(x) for proj in self.projections]
        elif self.mode == 'rotated_slice':
            rot = self.rotation.to(x.dtype)
            x_rot = x @ rot.T
            return [x_rot[..., i*self.d:(i+1)*self.d] for i in range(self.N)]
        elif self.mode == 'per_sub_embed':
            assert input_ids is not None
            return [emb(input_ids) for emb in self.sub_embeddings]
        elif self.mode == 'stem':
            # Project D -> d, run small transformer, then replicate
            h = self.stem_proj(x)
            # Need cos_sin for stem blocks - will be passed separately
            return [h.clone() for _ in range(self.N)]
        else:
            raise ValueError(f"Unknown input mode: {self.mode}")

    def forward_with_cos_sin(self, x, cos_sin, input_ids=None):
        """Version that passes cos_sin to stem blocks if needed."""
        if self.mode == 'stem':
            h = self.stem_proj(x)
            for block in self.stem_blocks:
                h = block(h, cos_sin)
            return [h.clone() for _ in range(self.N)]
        return self.forward(x, input_ids)


class MSTRouter(nn.Module):
    """Axis 2: How sub-transformer outputs are combined.

    Modes:
      'soft_weighted':  Soft weighted sum of all N sub outputs (all active).
      'topk_hard':      Hard top-k selection with load balancing aux loss.
      'sequence_path':  Sequence-level path routing (one path per sequence).
    """

    def __init__(self, d, N, D, mode='soft_weighted', topk=4, aux_weight=0.01, diversity_weight=0.0):
        super().__init__()
        self.d = d
        self.N = N
        self.D = D
        self.mode = mode
        self.topk = topk
        self.aux_weight = aux_weight
        self.diversity_weight = diversity_weight
        self._last_entropy = None
        self._last_balance = None
        self._last_aux_loss = None

        if mode in ('soft_weighted', 'topk_hard'):
            # Router: project aggregated repr to N scores
            self.router = Linear(d, N, bias=False)
        elif mode == 'sequence_path':
            self.router = Linear(d, N, bias=False)

    def forward(self, sub_outputs):
        """sub_outputs: list of N tensors (B, T, d). Returns (B, T, d) + aux_loss."""
        stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
        B, T, N, d = stacked.shape

        # Router input: mean across sub-transformers
        router_input = stacked.mean(dim=2)  # (B, T, d)
        logits = self.router(router_input)  # (B, T, N)

        if self.mode == 'soft_weighted':
            weights = F.softmax(logits, dim=-1)  # (B, T, N)
            output = (weights.unsqueeze(-1) * stacked).sum(dim=2)  # (B, T, d)
        elif self.mode == 'topk_hard':
            k = min(self.topk, N)
            topk_vals, topk_idx = logits.topk(k, dim=-1)
            weights = F.softmax(topk_vals, dim=-1)  # (B, T, k)
            # Gather selected sub outputs
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, d)
            selected = torch.gather(stacked, 2, topk_idx_exp)  # (B, T, k, d)
            output = (weights.unsqueeze(-1) * selected).sum(dim=2)
        elif self.mode == 'sequence_path':
            # Sequence-level: average logits across T, then softmax
            seq_logits = logits.mean(dim=1)  # (B, N)
            weights = F.softmax(seq_logits, dim=-1)  # (B, N)
            output = (weights.unsqueeze(1).unsqueeze(-1) * stacked).sum(dim=2)
        else:
            raise ValueError(f"Unknown routing mode: {self.mode}")

        # Track diagnostics — store as tensors (no .item()) to avoid graph break.
        # Call .item() only at logging time via last_entropy / last_balance properties.
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            self._last_entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()  # tensor
            load = probs.mean(dim=(0, 1))  # (N,)
            self._last_balance = load.min() / (load.max() + 1e-8)               # tensor

        # Load balance aux loss (switch transformer style)
        aux_loss = torch.tensor(0.0, device=output.device)
        if self.training and self.mode in ('soft_weighted', 'topk_hard'):
            probs_mean = F.softmax(logits, dim=-1).mean(dim=(0, 1))  # (N,)
            # Fraction of tokens routed to each expert
            if self.mode == 'topk_hard':
                dispatch = torch.zeros(N, device=output.device)
                dispatch.scatter_add_(0, topk_idx.reshape(-1), torch.ones(B*T*self.topk, device=output.device))
                dispatch = dispatch / (B * T)
            else:
                dispatch = probs_mean
            aux_loss = self.aux_weight * N * (dispatch * probs_mean).sum()

        # Cosine diversity penalty: penalize pairwise cosine similarity of sub outputs
        # to encourage sub-transformers to learn distinct, complementary representations.
        # Uses mean-pooled representations to avoid O(B*T*N*N) memory.
        if self.training and self.diversity_weight > 0.0:
            mean_repr = stacked.mean(dim=(0, 1))        # (N, d) — mean repr per sub
            normed = F.normalize(mean_repr, dim=-1)      # (N, d)
            sim = normed @ normed.T                       # (N, N)
            # Mean off-diagonal similarity
            mask = ~torch.eye(N, device=sim.device, dtype=torch.bool)
            diversity_loss = sim.masked_select(mask).mean()
            aux_loss = aux_loss + self.diversity_weight * diversity_loss

        self._last_aux_loss = aux_loss

        return output, aux_loss


class MSTTransition(nn.Module):
    """Axis 4: How outputs flow between layers.

    Modes:
      'parallel':             Direct pass — sub i feeds sub i in next layer.
      'aggregate_distribute': Router combines -> per-sub projection redistributes.
      'cross_attend':         Each sub cross-attends to all other subs (lightweight).
      'concat_proj':          Concatenate all N sub outputs, project down, redistribute.
      'free_for_all':         Each sub dynamically routes its output to target subs.
      'micro_attention':      T1 — N-way self-attention over sub outputs. O(N²×d) per position.
    """

    def __init__(self, d, N, D, mode='parallel', diversity_weight=0.0,
                 routing_mode='soft_weighted', routing_topk=0):
        super().__init__()
        self.mode = mode
        self.d = d
        self.N = N

        if mode == 'aggregate_distribute':
            self.router = MSTRouter(d, N, D, mode=routing_mode, diversity_weight=diversity_weight)
            self.distribute = nn.ModuleList([
                Linear(d, d, bias=False) for _ in range(N)
            ])
        elif mode == 'cross_attend':
            # Lightweight: each sub gets a weighted sum of all other subs
            self.cross_weights = nn.Parameter(torch.zeros(N, N))
        elif mode == 'concat_proj':
            # Concatenate all N sub outputs (N*d), project to d, then redistribute.
            # Preserves cross-sub information that weighted sum destroys.
            self.concat_down = Linear(N * d, d, bias=False)
            self.distribute = nn.ModuleList([
                Linear(d, d, bias=False) for _ in range(N)
            ])
        elif mode == 'free_for_all':
            # Each sub has a router: (B, T, d) → (B, T, N) logits over target subs.
            # Creates a dynamic, input-dependent (N_sender, N_target) wiring pattern.
            self.sub_routers = nn.ModuleList([
                Linear(d, N, bias=False) for _ in range(N)
            ])
            self.aux_weight = 0.01  # light load balance for target utilization
            self.routing_topk = routing_topk  # 0 = soft (all targets), >0 = hard topk
            self.temperature = 1.0  # set by MSTLayer from config
            self._last_route_weights = None  # stored for diagnostics
        elif mode == 'micro_attention':
            # T1: Self-attention over N sub-transformer outputs per position.
            # Treats N subs as a tiny "sequence" and runs single-head attention.
            # Q, K, V projections: d → d. Cost: O(N² × d × seq_len) — negligible.
            self.ma_q = Linear(d, d, bias=False)
            self.ma_k = Linear(d, d, bias=False)
            self.ma_v = Linear(d, d, bias=False)
            self.ma_proj = Linear(d, d, bias=False)
            self._last_attn_weights = None  # N×N attention pattern for diagnostics
        elif mode == 'micro_attention_shared_kv':
            # T1b: Shared-KV micro-attention. Each sub has its own Q but K,V are shared.
            # Encourages subs to specialize via different "questions" into same "memory".
            # Per-sub Q: N separate d→d projections. Shared K,V: 1 each.
            self.ma_qs = nn.ModuleList([Linear(d, d, bias=False) for _ in range(N)])
            self.ma_k = Linear(d, d, bias=False)  # shared
            self.ma_v = Linear(d, d, bias=False)  # shared
            self.ma_proj = Linear(d, d, bias=False)
            self._last_attn_weights = None

    def forward(self, sub_outputs):
        """sub_outputs: list of N tensors (B, T, d).
        Returns: list of N tensors (B, T, d) for next layer, aux_loss."""
        if self.mode == 'parallel':
            return sub_outputs, torch.tensor(0.0, device=sub_outputs[0].device)
        elif self.mode == 'aggregate_distribute':
            aggregated, aux_loss = self.router(sub_outputs)  # (B, T, d)
            return [proj(aggregated) for proj in self.distribute], aux_loss
        elif self.mode == 'cross_attend':
            stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
            weights = F.softmax(self.cross_weights, dim=-1)  # (N, N)
            # Each sub i gets weighted combo of all subs
            mixed = torch.einsum('ij,btjd->btid', weights.to(stacked.dtype), stacked)
            return [mixed[:, :, i] for i in range(self.N)], torch.tensor(0.0, device=stacked.device)
        elif self.mode == 'concat_proj':
            concatenated = torch.cat(sub_outputs, dim=-1)  # (B, T, N*d)
            aggregated = self.concat_down(concatenated)     # (B, T, d)
            return [proj(aggregated) for proj in self.distribute], torch.tensor(0.0, device=aggregated.device)
        elif self.mode == 'free_for_all':
            N = self.N
            stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
            # Each sender sub produces routing logits over N target subs
            route_logits = torch.stack([
                router(sub_out) for router, sub_out in zip(self.sub_routers, sub_outputs)
            ], dim=2)  # (B, T, N_sender, N_target)
            # Apply topk masking if configured (each sender picks top-k targets)
            if self.routing_topk > 0 and self.routing_topk < N:
                _, topk_idx = torch.topk(route_logits, self.routing_topk, dim=-1)
                mask = torch.zeros_like(route_logits).scatter_(-1, topk_idx, 1.0)
                route_logits = route_logits.masked_fill(mask == 0, -1e9)
            route_weights = F.softmax(route_logits / self.temperature, dim=-1)  # (B, T, N_sender, N_target)
            self._last_route_weights = route_weights.detach()  # store for diagnostics
            # Target sub j receives: sum_i(weights[..., i, j] * sub_i_output)
            mixed = torch.einsum('btij,btid->btjd', route_weights, stacked)  # (B, T, N_target, d)
            # Light load balance: ensure targets are utilized roughly equally
            aux_loss = torch.tensor(0.0, device=stacked.device)
            if self.training:
                target_load = route_weights.mean(dim=(0, 1, 2))  # (N_target,) avg fraction to each target
                aux_loss = self.aux_weight * N * (target_load * target_load).sum()
            return [mixed[:, :, j] for j in range(N)], aux_loss
        elif self.mode == 'micro_attention':
            # T1: Self-attention over N sub-transformer outputs per token position.
            # Stack: (B, T, N, d). Attention over the N dimension.
            N = self.N
            stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
            B, T, _, d = stacked.shape
            # Project to Q, K, V
            q = self.ma_q(stacked)  # (B, T, N, d)
            k = self.ma_k(stacked)  # (B, T, N, d)
            v = self.ma_v(stacked)  # (B, T, N, d)
            # Scaled dot-product attention over the N dimension
            # (B, T, N, d) × (B, T, d, N) → (B, T, N, N)
            scale = d ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, T, N, N)
            attn_weights = F.softmax(attn_weights, dim=-1)
            # Store attention pattern for diagnostics
            if not self.training:
                self._last_attn_weights = attn_weights.detach().mean(dim=(0, 1))  # (N, N)
            else:
                with torch.no_grad():
                    self._last_attn_weights = attn_weights.detach().mean(dim=(0, 1))
            # Apply attention: (B, T, N, N) × (B, T, N, d) → (B, T, N, d)
            attended = torch.matmul(attn_weights, v)
            # Output projection
            out = self.ma_proj(attended)  # (B, T, N, d)
            return [out[:, :, j] for j in range(N)], torch.tensor(0.0, device=stacked.device)
        elif self.mode == 'micro_attention_shared_kv':
            # T1b: Per-sub Q, shared K/V. Each sub asks different questions of shared memory.
            N = self.N
            stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
            B, T, _, d = stacked.shape
            # Shared K, V from all sub outputs
            k = self.ma_k(stacked)  # (B, T, N, d) — shared weights, N different keys
            v = self.ma_v(stacked)  # (B, T, N, d) — shared weights, N different values
            # Per-sub Q: each sub has its own query projection
            q = torch.stack([wq(sub_outputs[i]) for i, wq in enumerate(self.ma_qs)], dim=2)  # (B, T, N, d)
            # Attention: each sub's Q attends over all N keys
            scale = d ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, T, N, N)
            attn_weights = F.softmax(attn_weights, dim=-1)
            with torch.no_grad():
                self._last_attn_weights = attn_weights.detach().mean(dim=(0, 1))
            attended = torch.matmul(attn_weights, v)  # (B, T, N, d)
            out = self.ma_proj(attended)
            return [out[:, :, j] for j in range(N)], torch.tensor(0.0, device=stacked.device)
        else:
            raise ValueError(f"Unknown transition mode: {self.mode}")


class MSTFinalHead(nn.Module):
    """Axis 5: How the final layer produces vocabulary logits.

    Modes:
      'aggregate_proj':   Weighted sum of sub outputs -> single lm_head projection.
      'weighted_logits':  Each sub has independent lm_head; weighted sum of logits.
      'concat_proj':      Concatenate all N sub outputs, project to D, then lm_head.
    """

    def __init__(self, d, N, D, vocab_size, mode='aggregate_proj', diversity_weight=0.0,
                 routing_mode='soft_weighted', routing_topk=0):
        super().__init__()
        self.mode = mode
        self.d = d
        self.N = N
        self.D = D
        self.routing_topk = routing_topk

        if mode == 'aggregate_proj':
            self.agg_router = MSTRouter(d, N, D, mode=routing_mode, diversity_weight=diversity_weight)
            self.proj = Linear(d, D, bias=False)  # d -> D before lm_head
        elif mode == 'weighted_logits':
            self.sub_heads = nn.ModuleList([
                Linear(d, vocab_size, bias=False) for _ in range(N)
            ])
            self.head_weights = nn.Parameter(torch.ones(N) / N)
        elif mode == 'concat_proj':
            # Concatenate all N sub outputs (N*d), project directly to D.
            self.proj = Linear(N * d, D, bias=False)
            # If topk > 0, add a selection router to pick which subs to include
            if routing_topk > 0 and routing_topk < N:
                self.select_router = Linear(d, N, bias=False)
        else:
            raise ValueError(f"Unknown final mode: {mode}")

    def forward(self, sub_outputs, lm_head=None):
        """sub_outputs: list of N tensors (B, T, d).
        Returns logits (B, T, vocab_size) and aux_loss."""
        if self.mode == 'aggregate_proj':
            aggregated, aux_loss = self.agg_router(sub_outputs)  # (B, T, d)
            h = self.proj(aggregated)  # (B, T, D)
            logits = lm_head(h)
            return logits, aux_loss
        elif self.mode == 'weighted_logits':
            weights = F.softmax(self.head_weights, dim=0)
            logits = sum(w * head(sub_out)
                         for w, head, sub_out
                         in zip(weights, self.sub_heads, sub_outputs))
            return logits, torch.tensor(0.0, device=logits.device)
        elif self.mode == 'concat_proj':
            stacked = torch.stack(sub_outputs, dim=2)  # (B, T, N, d)
            if self.routing_topk > 0 and self.routing_topk < self.N and hasattr(self, 'select_router'):
                # Select top-k subs per token, zero out the rest
                mean_repr = stacked.mean(dim=2)  # (B, T, d)
                sel_logits = self.select_router(mean_repr)  # (B, T, N)
                _, topk_idx = torch.topk(sel_logits, self.routing_topk, dim=-1)
                mask = torch.zeros_like(sel_logits).scatter_(-1, topk_idx, 1.0)  # (B, T, N)
                stacked = stacked * mask.unsqueeze(-1)  # zero out non-selected subs
            concatenated = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)  # (B, T, N_actual*d)
            # Handle progressive merge: fewer subs than N, pad to expected N*d
            expected_dim = self.N * self.d
            if concatenated.shape[-1] < expected_dim:
                pad = torch.zeros(*concatenated.shape[:-1], expected_dim - concatenated.shape[-1],
                                  device=concatenated.device, dtype=concatenated.dtype)
                concatenated = torch.cat([concatenated, pad], dim=-1)
            h = self.proj(concatenated)  # (B, T, D)
            logits = lm_head(h)
            return logits, torch.tensor(0.0, device=stacked.device)


class DenseSubLayer(nn.Module):
    """Full-width D-dim layer for hybrid dense-MST mode.

    Concatenates N sub inputs to D, runs standard attention+FFN at full width,
    then slices back to N sub outputs. Uses same head_dim as sub-transformers
    so RoPE embeddings can be reused.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        D = config.n_embd
        d = config.mst_sub_dim
        N = config.mst_n_subs
        n_layer = config.n_layer
        # Use same head_dim as sub-transformers to reuse precomputed RoPE
        sub_head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // config.n_head
        dense_n_head = D // sub_head_dim
        self.N = N
        self.d = d
        self.mode = 'dense'  # for transition residual check

        self.attn = SubTransformerAttention(D, dense_n_head, sub_head_dim,
                                             layer_idx, n_layer, sub_layer_idx=layer_idx)
        self.ffn_up = Linear(D, 4 * D, bias=False)
        self.ffn_down = Linear(4 * D, D, bias=False)

    def forward(self, sub_inputs, cos_sin, sub_ves=None, window_size=(-1, 0),
                kv_cache=None, total_sub_layers=1):
        """Same interface as MSTLayer — takes/returns list of N tensors (B,T,d)."""
        x = torch.cat(sub_inputs, dim=-1)  # (B, T, D)
        # Attention with residual
        x = x + self.attn(norm(x), cos_sin, window_size=window_size)
        # FFN with residual
        h = F.relu(self.ffn_up(norm(x))).square()
        x = x + self.ffn_down(h)
        # Slice back to N sub outputs
        sub_outputs = [x[..., i*self.d:(i+1)*self.d] for i in range(self.N)]
        return sub_outputs, torch.tensor(0.0, device=x.device)


class MSTLayer(nn.Module):
    """One full MST layer: N parallel sub-transformer blocks + transition."""

    def __init__(self, config, layer_idx, diversity_weight=0.0):
        super().__init__()
        d = config.mst_sub_dim
        N = config.mst_n_subs
        D = config.n_embd
        n_head = config.n_head
        n_layer = config.n_layer

        self.N = N
        self.layer_idx = layer_idx
        self.sub_dropout = config.mst_sub_dropout
        head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // n_head
        self.sub_blocks = nn.ModuleList([
            SubTransformerBlock(d, n_head, head_dim, layer_idx, n_layer,
                                sub_layer_idx=layer_idx * N + j,
                                ffn_mode=config.mst_ffn_mode,
                                ffn_inner_dim=config.mst_ffn_inner_dim,
                                sub_layers=config.mst_sub_layers)
            for j in range(N)
        ])

        # Cross-sub KV sharing: all subs share K,V projections (Q remains per-sub)
        # Saves (N-1) * 2 * d * (n_head*head_dim) params
        if config.mst_cross_sub_kv:
            ref_k = self.sub_blocks[0].attn.c_k
            ref_v = self.sub_blocks[0].attn.c_v
            for block in self.sub_blocks[1:]:
                block.attn.c_k = ref_k
                block.attn.c_v = ref_v

        # Shared FFN: single D-width FFN replaces per-sub FFNs.
        # Concat sub outputs → D → inner_dim → D → slice back.
        # One forward pass instead of N, captures cross-sub interactions.
        if config.mst_ffn_shared_up:
            D = config.n_embd
            inner_dim = config.mst_ffn_inner_dim if config.mst_ffn_inner_dim > 0 else 4 * d
            self.shared_ffn_up = Linear(D, inner_dim, bias=False)
            self.shared_ffn_down = Linear(inner_dim, D, bias=False)
            for block in self.sub_blocks:
                block._skip_ffn = True  # sub blocks only do attention
        else:
            self.shared_ffn_up = None

        # Transition: skip at non-transition layers (transition_every > 1)
        use_transition = (layer_idx % config.mst_transition_every == 0)
        if use_transition:
            self.transition = MSTTransition(d, N, D, mode=config.mst_transition_mode,
                                             diversity_weight=diversity_weight,
                                             routing_mode=config.mst_routing_mode,
                                             routing_topk=config.mst_routing_topk)
            # Wire FFA temperature
            if hasattr(self.transition, 'temperature'):
                self.transition.temperature = config.mst_ffa_temperature
        else:
            self.transition = MSTTransition(d, N, D, mode='parallel')

    def forward(self, sub_inputs, cos_sin, sub_ves=None, window_size=(-1, 0),
                sub_window_sizes=None, kv_cache=None, total_sub_layers=1):
        """sub_inputs: list of N tensors (B, T, d).
        sub_ves: optional list of N tensors (B, T, d) value embeddings.
        sub_window_sizes: optional list of N tuples for per-sub window sizes (multi-scale).
        Returns: list of N tensors (B, T, d), aux_loss."""
        if sub_ves is not None:
            sub_outputs = [
                block(h, cos_sin, ve=ve,
                      window_size=sub_window_sizes[j] if sub_window_sizes else window_size,
                      kv_cache=kv_cache, total_sub_layers=total_sub_layers)
                for j, (block, h, ve) in enumerate(zip(self.sub_blocks, sub_inputs, sub_ves))
            ]
        else:
            sub_outputs = [
                block(h, cos_sin,
                      window_size=sub_window_sizes[j] if sub_window_sizes else window_size,
                      kv_cache=kv_cache, total_sub_layers=total_sub_layers)
                for j, (block, h) in enumerate(zip(self.sub_blocks, sub_inputs))
            ]

        # Shared FFN: concat to D, one FFN pass, slice back
        if self.shared_ffn_up is not None:
            x = torch.cat(sub_outputs, dim=-1)  # (B, T, D)
            h = F.relu(self.shared_ffn_up(norm(x))).square()  # (B, T, inner_dim)
            ffn_out = self.shared_ffn_down(h)  # (B, T, D)
            d = sub_outputs[0].shape[-1]
            for i in range(self.N):
                sub_outputs[i] = sub_outputs[i] + ffn_out[..., i*d:(i+1)*d]

        # Sub dropout: randomly zero entire sub outputs during training
        if self.training and self.sub_dropout > 0:
            drop_mask = torch.rand(self.N, device=sub_outputs[0].device) < self.sub_dropout
            # Don't drop ALL subs
            if drop_mask.all():
                drop_mask[torch.randint(self.N, (1,))] = False
            for i in range(self.N):
                if drop_mask[i]:
                    sub_outputs[i] = sub_outputs[i] * 0  # in-place-compatible, keeps graph but no extra alloc

        # Transition with optional residual + pre-norm.
        # Residual helps bottleneck transitions (aggdist, cross_attend, concat_proj)
        # by preserving sub identity through the skip path.
        # FFA creates deliberately mixed representations — residual dilutes that and hurts.
        _USE_TRANSITION_RESIDUAL = {'aggregate_distribute', 'cross_attend', 'concat_proj', 'micro_attention', 'micro_attention_shared_kv'}
        mode = self.transition.mode
        if mode == 'parallel':
            return self.transition(sub_outputs)
        elif mode in _USE_TRANSITION_RESIDUAL:
            normed = [norm(h) for h in sub_outputs]
            transitioned, aux_loss = self.transition(normed)
            return [h + t for h, t in zip(sub_outputs, transitioned)], aux_loss
        else:
            # free_for_all: no residual, but still pre-norm for routing stability
            normed = [norm(h) for h in sub_outputs]
            return self.transition(normed)


# ---------------------------------------------------------------------------
# Compile-optimized batched MST layer
# ---------------------------------------------------------------------------
# The key insight: instead of N separate SubTransformerBlock modules (each with
# its own weight matrices and separate forward calls → N× kernel launches),
# store all weights as fused (N, out, in) parameter tensors and process them
# with a single batched operation via torch.einsum or torch.bmm.
#
# The entire sub-state flows as a (B, T, N, d) tensor — no Python lists, no
# stacking/unstacking. torch.compile sees clean tensor ops and fuses them.
# ---------------------------------------------------------------------------

def _batched_linear(x, weight):
    """Batched linear: x (B, T, N, in) @ weight (N, out, in).T → (B, T, N, out).

    Uses bmm for efficiency: reshape to (N, B*T, in) @ (N, in, out) → (N, B*T, out).
    """
    B, T, N, d_in = x.shape
    # (B, T, N, d_in) → (N, B*T, d_in)
    x_r = x.permute(2, 0, 1, 3).reshape(N, B * T, d_in)
    # weight is (N, d_out, d_in) → transpose to (N, d_in, d_out)
    y = torch.bmm(x_r, weight.to(dtype=x.dtype).transpose(1, 2))  # (N, B*T, d_out)
    # (N, B*T, d_out) → (B, T, N, d_out)
    return y.view(N, B, T, -1).permute(1, 2, 0, 3)


class BatchedMSTLayer(nn.Module):
    """Compile-optimized MST layer: all N sub-transformers processed via batched ops.

    Supports:
      - Batched attention (Q/K/V/proj as (N, out, in) weight tensors)
      - Batched FFN (fc/proj as (N, out, in) weight tensors)
      - Per-sub sliding window attention (loop over N for flash_attn, N is static)
      - Batched aggregate_distribute transition
      - Value embeddings with batched VE gates

    Input/output: (B, T, N, d) tensor — no Python lists.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        d = config.mst_sub_dim
        N = config.mst_n_subs
        D = config.n_embd
        n_head = config.n_head
        n_layer = config.n_layer
        head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // n_head
        qkv_dim = n_head * head_dim

        self.N = N
        self.d = d
        self.n_head = n_head
        self.head_dim = head_dim
        self.qkv_dim = qkv_dim
        self.layer_idx = layer_idx

        # --- Batched attention weights: stored as 2D (N*out, in) for Muon compat ---
        # Reshaped to (N, out, in) in forward via .view()
        self.c_q_w = nn.Parameter(torch.empty(N * qkv_dim, d))
        self.c_k_w = nn.Parameter(torch.empty(N * qkv_dim, d))
        self.c_v_w = nn.Parameter(torch.empty(N * qkv_dim, d))
        self.c_proj_w = nn.Parameter(torch.empty(N * d, qkv_dim))

        # --- Batched FFN weights: standard d → 4d → d ---
        inner = 4 * d
        self._inner = inner
        self.fc_w = nn.Parameter(torch.empty(N * inner, d))
        self.fc_proj_w = nn.Parameter(torch.empty(N * d, inner))

        # --- Batched VE gates: stored as 2D (N*n_head, ve_gate_channels) ---
        self.ve_gate_channels = min(d, 32)
        self._has_ve = has_ve(layer_idx, n_layer)
        if self._has_ve:
            self.ve_gate_w = nn.Parameter(torch.empty(N * n_head, self.ve_gate_channels))
        else:
            self.ve_gate_w = None

        # --- Transition weights (mode-dependent) ---
        self._transition_mode = config.mst_transition_mode
        if config.mst_transition_mode == 'aggregate_distribute':
            # Router: project mean sub repr to N logits
            self.router_w = nn.Parameter(torch.empty(N, d))  # (N, d) — Linear(d, N), already 2D
            # Distribute: stored as 2D (N*d, d) for Muon compat
            self.distribute_w = nn.Parameter(torch.empty(N * d, d))
        elif config.mst_transition_mode == 'free_for_all':
            # Per-sub routers: each sender sub routes to N target subs
            # Weight shape: (N_sender, N_target, d) → stored as 2D (N*N, d) for Muon
            self.ffa_router_w = nn.Parameter(torch.empty(N * N, d))
            self._ffa_temperature = config.mst_ffa_temperature
            self._ffa_topk = config.mst_routing_topk
        elif config.mst_transition_mode == 'micro_attention':
            # Shared Q/K/V/proj for self-attention over the N subs per position
            # These are shared across subs (not per-sub), so they're regular 2D (d, d)
            self.ma_q_w = nn.Parameter(torch.empty(d, d))
            self.ma_k_w = nn.Parameter(torch.empty(d, d))
            self.ma_v_w = nn.Parameter(torch.empty(d, d))
            self.ma_proj_w = nn.Parameter(torch.empty(d, d))
        self.aux_weight = config.mst_routing_aux_weight

    def forward(self, sub_states, cos_sin, ve=None, window_sizes=None,
                kv_cache=None, total_sub_layers=1):
        """
        Args:
            sub_states: (B, T, N, d) batched sub-transformer states
            cos_sin: (cos, sin) for RoPE
            ve: optional (B, T, d) value embedding (shared across subs)
            window_sizes: list of N (left, right) tuples for per-sub sliding window
            kv_cache: optional KVCache for inference
            total_sub_layers: n_layer * N for cache slot indexing

        Returns: (B, T, N, d), aux_loss scalar tensor
        """
        B, T, N, d = sub_states.shape
        cos, sin = cos_sin

        # Reshape stored 2D weights (N*out, in) → 3D (N, out, in) for batched ops
        c_q_w = self.c_q_w.view(N, self.qkv_dim, d)
        c_k_w = self.c_k_w.view(N, self.qkv_dim, d)
        c_v_w = self.c_v_w.view(N, self.qkv_dim, d)
        c_proj_w = self.c_proj_w.view(N, d, self.qkv_dim)
        fc_w = self.fc_w.view(N, self._inner, d)
        fc_proj_w = self.fc_proj_w.view(N, d, self._inner)
        distribute_w = self.distribute_w.view(N, d, d) if self._transition_mode == 'aggregate_distribute' else None
        ve_gate_w = self.ve_gate_w.view(N, self.n_head, self.ve_gate_channels) if self.ve_gate_w is not None else None

        # ==================== ATTENTION ====================
        # Pre-norm
        x = norm(sub_states)  # (B, T, N, d) — RMSNorm on last dim

        # Batched Q, K, V projections: (B, T, N, d) @ (N, qkv, d).T → (B, T, N, qkv)
        q = _batched_linear(x, c_q_w)
        k = _batched_linear(x, c_k_w)
        v = _batched_linear(x, c_v_w)

        # Value embedding: shared VE across all subs, per-sub gating
        if ve is not None and ve_gate_w is not None:
            # ve: (B, T, d) → (B, T, 1, n_head, head_dim) for broadcasting
            ve_heads = ve.view(B, T, 1, self.n_head, self.head_dim)
            # Gate: x[..., :ve_gc] → (B, T, N, ve_gc) @ ve_gate_w (N, n_head, ve_gc).T
            gate_in = x[..., :self.ve_gate_channels]  # (B, T, N, ve_gc)
            # einsum: (B,T,N,gc) × (N,H,gc) → (B,T,N,H)
            gates = torch.einsum('btng,nhg->btnh',
                                 gate_in, ve_gate_w.to(dtype=gate_in.dtype))
            gates = 2.0 * torch.sigmoid(gates)  # (B, T, N, n_head)
            # Apply to v: reshape v to (B,T,N,H,hd), add gated VE
            v_heads = v.view(B, T, N, self.n_head, self.head_dim)
            v_heads = v_heads + gates.unsqueeze(-1) * ve_heads
            v = v_heads.reshape(B, T, N, self.qkv_dim)

        # Per-sub flash attention (loop — N is static, compile unrolls it)
        # Each sub may have a different sliding window size.
        half_hd = self.head_dim // 2
        cos_slice = cos[..., :half_hd]
        sin_slice = sin[..., :half_hd]

        attn_results = []
        for j in range(N):
            qj = q[:, :, j].view(B, T, self.n_head, self.head_dim)
            kj = k[:, :, j].view(B, T, self.n_head, self.head_dim)
            vj = v[:, :, j].view(B, T, self.n_head, self.head_dim)

            # RoPE + QK-norm
            qj = apply_rotary_emb(qj, cos_slice, sin_slice)
            kj = apply_rotary_emb(kj, cos_slice, sin_slice)
            qj, kj = norm(qj), norm(kj)

            ws = window_sizes[j] if window_sizes is not None else (-1, 0)
            if kv_cache is None:
                yj = flash_attn.flash_attn_func(
                    qj.to(torch.bfloat16), kj.to(torch.bfloat16), vj.to(torch.bfloat16),
                    causal=True, window_size=ws
                )
            else:
                sub_layer_idx = self.layer_idx * N + j
                k_cache, v_cache = kv_cache.get_layer_cache(sub_layer_idx)
                yj = flash_attn.flash_attn_with_kvcache(
                    qj.to(torch.bfloat16), k_cache, v_cache,
                    k=kj.to(torch.bfloat16), v=vj.to(torch.bfloat16),
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True, window_size=ws,
                )
                if self.layer_idx == (total_sub_layers // N) - 1 and j == N - 1:
                    kv_cache.advance(T)
            attn_results.append(yj.reshape(B, T, self.qkv_dim))

        y = torch.stack(attn_results, dim=2)  # (B, T, N, qkv_dim)

        # Batched output projection
        attn_out = _batched_linear(y, c_proj_w)  # (B, T, N, d)

        # Attention residual
        sub_states = sub_states + attn_out

        # ==================== FFN ====================
        x = norm(sub_states)
        h = _batched_linear(x, fc_w)              # (B, T, N, 4d)
        h = F.relu(h).square()                      # relu²
        ffn_out = _batched_linear(h, fc_proj_w)    # (B, T, N, d)
        sub_states = sub_states + ffn_out

        # ==================== TRANSITION ====================
        # Pre-norm for transition
        x = norm(sub_states)  # (B, T, N, d)
        aux_loss = sub_states.new_zeros(())

        if self._transition_mode == 'aggregate_distribute':
            # Router: mean across subs → logits
            router_input = x.mean(dim=2)  # (B, T, d)
            # router_w is (N, d), used as weight for Linear(d, N)
            logits = F.linear(router_input, self.router_w.to(dtype=router_input.dtype))  # (B, T, N)
            weights = F.softmax(logits, dim=-1)  # (B, T, N)

            # Weighted sum: (B, T, N) × (B, T, N, d) → (B, T, d)
            aggregated = (weights.unsqueeze(-1) * x).sum(dim=2)  # (B, T, d)

            # Batched distribute: (B, T, d) → (B, T, N, d) via N projections
            # distribute_w viewed as: (N, d, d)
            # einsum: (B,T,d) × (N,d_out,d_in) → (B,T,N,d_out)
            dist_w_3d = distribute_w.to(dtype=aggregated.dtype)
            distributed = torch.einsum('btd,nod->btno', aggregated, dist_w_3d)

            # Transition residual (aggregate_distribute uses residual)
            sub_states = sub_states + distributed

            # Aux loss: load balance (soft_weighted)
            if self.training:
                probs_mean = weights.mean(dim=(0, 1))  # (N,)
                aux_loss = self.aux_weight * N * (probs_mean * probs_mean).sum()

        elif self._transition_mode == 'free_for_all':
            # FFA: each sender sub independently routes to all target subs
            # ffa_router_w viewed as (N_sender, N_target, d)
            ffa_w = self.ffa_router_w.view(N, N, d).to(dtype=x.dtype)

            # Per-sub routing logits: (B,T,N_sender,d) × (N_sender,N_target,d) → (B,T,N_s,N_t)
            route_logits = torch.einsum('btid,ijd->btij', x, ffa_w)

            # Compute soft routing weights (always needed for gradient flow)
            soft_weights = F.softmax(route_logits / self._ffa_temperature, dim=-1)  # (B,T,N_s,N_t)

            if self._ffa_topk == 1:
                # Hard routing with straight-through estimator (STE)
                # Forward: one-hot argmax — each sender routes to exactly 1 target
                # Backward: gradients flow through soft probabilities
                hard_idx = soft_weights.argmax(dim=-1, keepdim=True)  # (B,T,N_s,1)
                hard_weights = torch.zeros_like(soft_weights).scatter_(-1, hard_idx, 1.0)
                route_weights = soft_weights + (hard_weights - soft_weights).detach()  # STE
            elif self._ffa_topk > 1 and self._ffa_topk < N:
                # Soft top-k: mask non-top-k logits, renormalize via softmax
                _, topk_idx = torch.topk(route_logits, self._ffa_topk, dim=-1)
                mask = torch.zeros_like(route_logits).scatter_(-1, topk_idx, 1.0)
                masked_logits = route_logits.masked_fill(mask == 0, -1e9)
                route_weights = F.softmax(masked_logits / self._ffa_temperature, dim=-1)
            else:
                # Fully soft (topk=0): all targets weighted
                route_weights = soft_weights

            # Target sub j receives weighted sum of all senders
            # (B,T,N_sender,N_target) × (B,T,N_sender,d) → (B,T,N_target,d)
            mixed = torch.einsum('btij,btid->btjd', route_weights, x)

            # FFA: NO residual — creates deliberately mixed representations
            sub_states = mixed

            # Aux loss: load balance on target utilization
            if self.training:
                target_load = route_weights.mean(dim=(0, 1, 2))  # (N_target,)
                aux_loss = self.aux_weight * N * (target_load * target_load).sum()

        elif self._transition_mode == 'micro_attention':
            # Self-attention over the N subs at each token position
            # Shared Q/K/V projections: (B,T,N,d) @ (d,d) → (B,T,N,d)
            ma_q = F.linear(x, self.ma_q_w.to(dtype=x.dtype))  # (B,T,N,d)
            ma_k = F.linear(x, self.ma_k_w.to(dtype=x.dtype))  # (B,T,N,d)
            ma_v = F.linear(x, self.ma_v_w.to(dtype=x.dtype))  # (B,T,N,d)

            # Scaled dot-product attention over the N dimension
            # (B,T,N,d) × (B,T,d,N) → (B,T,N,N)
            scale = d ** -0.5
            attn_weights = torch.matmul(ma_q, ma_k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)  # (B,T,N,N)

            # Apply attention: (B,T,N,N) × (B,T,N,d) → (B,T,N,d)
            attended = torch.matmul(attn_weights, ma_v)

            # Output projection
            out = F.linear(attended, self.ma_proj_w.to(dtype=attended.dtype))  # (B,T,N,d)

            # Micro-attention uses residual
            sub_states = sub_states + out

        return sub_states, aux_loss

    @torch.no_grad()
    def init_weights(self):
        """Initialize weights matching the per-sub initialization pattern."""
        sub_s = 1.0 / (self.d ** 0.5)
        N, d = self.N, self.d
        # View 2D stored weights as 3D for per-sub init
        c_q = self.c_q_w.view(N, self.qkv_dim, d)
        c_k = self.c_k_w.view(N, self.qkv_dim, d)
        c_v = self.c_v_w.view(N, self.qkv_dim, d)
        c_proj = self.c_proj_w.view(N, d, self.qkv_dim)
        fc = self.fc_w.view(N, self._inner, d)
        fc_proj = self.fc_proj_w.view(N, d, self._inner)
        ve_gate = self.ve_gate_w.view(N, self.n_head, self.ve_gate_channels) if self.ve_gate_w is not None else None

        for j in range(N):
            nn.init.uniform_(c_q[j], -sub_s, sub_s)
            nn.init.uniform_(c_k[j], -sub_s, sub_s)
            nn.init.uniform_(c_v[j], -sub_s, sub_s)
            nn.init.zeros_(c_proj[j])
            # FFN
            nn.init.uniform_(fc[j], -sub_s, sub_s)
            nn.init.zeros_(fc_proj[j])
            # VE gate
            if ve_gate is not None:
                nn.init.uniform_(ve_gate[j], -sub_s, sub_s)

        # Transition-specific init
        if self._transition_mode == 'aggregate_distribute':
            dist = self.distribute_w.view(N, d, d)
            for j in range(N):
                nn.init.uniform_(dist[j], -sub_s, sub_s)
            # Router: zero init (soft at start)
            nn.init.zeros_(self.router_w)
        elif self._transition_mode == 'free_for_all':
            # FFA routers: zero init so routing starts uniform
            nn.init.zeros_(self.ffa_router_w)
        elif self._transition_mode == 'micro_attention':
            # Shared Q/K/V: uniform init, proj: zero init (residual-friendly)
            nn.init.uniform_(self.ma_q_w, -sub_s, sub_s)
            nn.init.uniform_(self.ma_k_w, -sub_s, sub_s)
            nn.init.uniform_(self.ma_v_w, -sub_s, sub_s)
            nn.init.zeros_(self.ma_proj_w)


def _can_use_batched_layer(config):
    """Check if the config is compatible with BatchedMSTLayer."""
    return (
        config.mst_ffn_mode == 'standard'
        and config.mst_transition_mode in ('aggregate_distribute', 'free_for_all', 'micro_attention')
        and config.mst_sub_layers == 1
        and not config.mst_ffn_shared_up
        and not config.mst_cross_sub_kv
        and config.mst_sub_dropout == 0.0
        and (config.mst_head_dim == 0 or config.mst_head_dim == config.mst_sub_dim // config.n_head)
        and not config.mst_progressive_merge
        and not config.mst_hybrid_dense
        and not config.mst_delta_residual
    )


class MST(nn.Module):
    """Modular Sub-Transformer model.

    Compatible interface with GPT: forward(idx, targets, kv_cache, loss_reduction).
    """

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        D = config.n_embd
        d = config.mst_sub_dim
        N = config.mst_n_subs
        assert D == N * d, f"n_embd ({D}) must equal mst_n_subs ({N}) * mst_sub_dim ({d})"

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self._padded_vocab_size = padded_vocab_size

        # Sliding window attention (matching base GPT's SSSSL pattern)
        self.window_sizes = self._compute_window_sizes(config)

        # W1: Multi-scale per-sub windows — each sub attends at a different context scale.
        # Geometrically spaced from local to global. Gives subs natural specialization signal.
        self.multi_scale_windows = bool(config.mst_multi_scale_windows)
        if self.multi_scale_windows:
            import math
            # Geometric spacing: e.g. N=4 → [64, 256, 1024, full]
            # N=8 → [32, 64, 128, 256, 512, 1024, 1536, full]
            min_window = 32
            max_window = config.sequence_len  # full causal
            self.sub_window_sizes = []
            for j in range(N):
                if j == N - 1:
                    # Last sub always gets full causal attention
                    self.sub_window_sizes.append((-1, 0))
                else:
                    # Geometric spacing from min_window to max_window
                    ratio = j / max(1, N - 2)  # 0.0 to 1.0 for first N-1 subs
                    w = int(min_window * (max_window / min_window) ** ratio)
                    w = min(w, max_window)
                    self.sub_window_sizes.append((w, 0))  # (left_window, right_window=0 for causal)
            print(f"[MST] Multi-scale sub windows: {self.sub_window_sizes}")
        else:
            self.sub_window_sizes = None

        # Token embedding (D-dim)
        self.wte = nn.Embedding(padded_vocab_size, D)

        # Axis 1: Input distribution
        self.input_layer = MSTInputLayer(
            D, d, N,
            mode=config.mst_input_mode,
            vocab_size=padded_vocab_size,
            rotated_slice_learned=config.mst_rotated_slice_learned,
            n_head=config.n_head,
        )

        # L transformer layers
        # Diversity penalty only at 1st, middle, and last layers to limit memory
        n = config.n_layer
        dw = config.mst_diversity_weight
        div_layers = {0, n // 2, n - 1} if dw > 0 else set()
        if config.mst_hybrid_dense:
            self._use_batched = False
            # Alternate: even layers = dense (full D), odd layers = MST (N×d)
            layers = []
            for i in range(n):
                if i % 2 == 0:
                    layers.append(DenseSubLayer(config, layer_idx=i))
                else:
                    layers.append(MSTLayer(config, layer_idx=i,
                                           diversity_weight=dw if i in div_layers else 0.0))
            self.layers = nn.ModuleList(layers)
        elif config.mst_progressive_merge:
            self._use_batched = False
            # N1: Progressive sub-merging pyramid.
            # Concat-doubles-d approach: merging groups of subs by concatenation.
            # Schedule for N=8, n=8: layers 0-3 = 8×d, layers 4-6 = 2×4d, layer 7 = 1×8d
            # Each merge entry: (after_layer, target_N)
            import math, copy
            # Build merge schedule: [(layer_idx, target_N), ...]
            # Strategy: first half at full N, then aggressive merge to 2, then to 1
            half = n // 2  # first half keeps full N
            merge_schedule = []
            # Merge to 2 subs after layer (half-1)
            if N > 2:
                merge_schedule.append((half - 1, 2))
            # Merge to 1 sub at second-to-last layer
            if N > 1:
                merge_schedule.append((n - 2, 1))
            self._merge_schedule = {layer: target for layer, target in merge_schedule}
            # Build layers with varying N and d
            layers = []
            current_n = N
            current_d = d
            for i in range(n):
                level_config = copy.copy(config)
                level_config.mst_n_subs = current_n
                level_config.mst_sub_dim = current_d
                level_config.n_embd = current_n * current_d
                layers.append(MSTLayer(level_config, layer_idx=i,
                                       diversity_weight=dw if i in div_layers else 0.0))
                if i in self._merge_schedule:
                    target_n = self._merge_schedule[i]
                    group_size = current_n // target_n
                    current_d = current_d * group_size
                    current_n = target_n
            self.layers = nn.ModuleList(layers)
        else:
            self._use_batched = _can_use_batched_layer(config)
            if self._use_batched:
                print0(f"[MST] Using compile-optimized BatchedMSTLayer (N={N}, d={d})")
                self.layers = nn.ModuleList([
                    BatchedMSTLayer(config, layer_idx=i)
                    for i in range(n)
                ])
            else:
                self.layers = nn.ModuleList([
                    MSTLayer(config, layer_idx=i, diversity_weight=dw if i in div_layers else 0.0)
                    for i in range(n)
                ])

        # Per-layer learnable residual scaling (matching base GPT)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Value embeddings (ResFormer-style): each sub-transformer gets its own VE
        # Alternating layers, last layer always included (same as base GPT)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, d)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        # Axis 2: Output routing (used by final head if aggregate_proj)
        # Axis 5: Final output
        final_topk = config.mst_final_topk if config.mst_final_topk >= 0 else config.mst_routing_topk
        self.final_head = MSTFinalHead(
            d, N, D, padded_vocab_size,
            mode=config.mst_final_mode,
            diversity_weight=config.mst_diversity_weight,
            routing_mode=config.mst_routing_mode,
            routing_topk=final_topk,
        )

        # Standard lm_head for aggregate_proj mode
        self.lm_head = Linear(D, padded_vocab_size, bias=False)

        # Global residual stream: D-dim broadcast channel all subs read/write
        self.use_global_residual = bool(config.mst_global_residual)
        if self.use_global_residual:
            self.global_read_projs = nn.ModuleList([
                nn.ModuleList([Linear(D, d, bias=False) for _ in range(N)])
                for _ in range(n)
            ])
            self.global_write_projs = nn.ModuleList([
                Linear(N * d, D, bias=False) for _ in range(n)
            ])
        # DR1: Delta residual — subs produce corrections to full-D residual stream.
        # Shared per-sub down_proj(D→d) and up_proj(d→D) reused at every layer.
        # Sharing across layers matches MoL's design and avoids n_layer× param overhead.
        self.delta_residual = bool(config.mst_delta_residual)
        if self.delta_residual:
            self.delta_down_projs = nn.ModuleList([
                Linear(D, d, bias=False) for _ in range(N)
            ])
            self.delta_up_projs = nn.ModuleList([
                Linear(d, D, bias=False) for _ in range(N)
            ])

        # H3: Per-sub auxiliary prediction heads (for specialization pressure)
        self.sub_aux_weight = config.mst_sub_aux_weight
        if self.sub_aux_weight > 0:
            # Each sub gets its own d → vocab LM head for auxiliary next-token prediction.
            # Forces each sub to independently predict, preventing collapse.
            self.sub_aux_heads = nn.ModuleList([
                Linear(d, padded_vocab_size, bias=False) for _ in range(N)
            ])
        else:
            self.sub_aux_heads = None

        # N1: Progressive sub-merging (pyramid)
        self.progressive_merge = bool(config.mst_progressive_merge)
        if self.progressive_merge:
            self.merge_layers = set(self._merge_schedule.keys())
            # Rotary: use max head_dim (final merged level has largest d)
            import math
            final_d = d * N  # worst case: all merged to 1 sub
            for layer_idx, target_n in sorted(self._merge_schedule.items()):
                pass  # just iterate to find final state
            max_head_dim = final_d // config.n_head if config.mst_head_dim <= 0 else config.mst_head_dim
        else:
            self.merge_layers = set()

        # Rotary embeddings (precomputed, same as GPT)
        # For progressive merge, use max head_dim (final level has largest d)
        if self.progressive_merge:
            head_dim = max_head_dim
        else:
            head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Diagnostics state (must be in __init__, not init_weights)
        self._diag_enabled = False
        self._diag_sub_states = {}  # layer_idx -> list of N tensors (detached)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """Compute per-layer window sizes for sliding window attention.

        Matches base GPT: pattern string (default 'SSSSL') is tiled across layers.
        Final layer always gets full context. S=short (256), L=long (full).
        """
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = 256
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights following the same conventions as GPT."""
        s = 1.0 / (self.config.n_embd ** 0.5)

        # Embeddings
        torch.nn.init.normal_(self.wte.weight, std=1.0)

        # lm_head
        torch.nn.init.normal_(self.lm_head.weight, std=0.001)

        # Input layer
        if hasattr(self.input_layer, 'projections'):
            for proj in self.input_layer.projections:
                torch.nn.init.uniform_(proj.weight, -s, s)
        if hasattr(self.input_layer, 'sub_embeddings'):
            for emb in self.input_layer.sub_embeddings:
                torch.nn.init.normal_(emb.weight, std=1.0)
        if hasattr(self.input_layer, 'stem_proj'):
            torch.nn.init.uniform_(self.input_layer.stem_proj.weight, -s, s)

        # Sub-transformer blocks / batched layers
        sub_s = 1.0 / (self.config.mst_sub_dim ** 0.5)
        for layer in self.layers:
            if isinstance(layer, BatchedMSTLayer):
                # BatchedMSTLayer has its own init_weights
                layer.init_weights()
            elif hasattr(layer, 'sub_blocks'):
                for block in layer.sub_blocks:
                    torch.nn.init.uniform_(block.attn.c_q.weight, -sub_s, sub_s)
                    torch.nn.init.uniform_(block.attn.c_k.weight, -sub_s, sub_s)
                    torch.nn.init.uniform_(block.attn.c_v.weight, -sub_s, sub_s)
                    torch.nn.init.zeros_(block.attn.c_proj.weight)
                    torch.nn.init.uniform_(block.ffn.c_fc.weight, -sub_s, sub_s)
                    if block.ffn.c_proj is not None:
                        torch.nn.init.zeros_(block.ffn.c_proj.weight)

        # Transition layers (legacy path only)
        for layer in self.layers:
            if isinstance(layer, BatchedMSTLayer):
                continue  # already handled above
            if hasattr(layer, 'transition'):
                if hasattr(layer.transition, 'distribute'):
                    for proj in layer.transition.distribute:
                        torch.nn.init.uniform_(proj.weight, -sub_s, sub_s)
                if hasattr(layer.transition, 'cross_weights'):
                    torch.nn.init.zeros_(layer.transition.cross_weights)
                if hasattr(layer.transition, 'router') and hasattr(layer.transition.router, 'router'):
                    torch.nn.init.zeros_(layer.transition.router.router.weight)

        # Router weights (legacy path only)
        for layer in self.layers:
            if isinstance(layer, BatchedMSTLayer):
                continue
            tr = getattr(layer, 'transition', None)
            if tr is not None and hasattr(tr, 'router') and hasattr(tr.router, 'router'):
                torch.nn.init.zeros_(tr.router.router.weight)

        # VE gates (legacy path only)
        for layer in self.layers:
            if isinstance(layer, BatchedMSTLayer):
                continue
            if hasattr(layer, 'sub_blocks'):
                for block in layer.sub_blocks:
                    if block.attn.ve_gate is not None:
                        torch.nn.init.uniform_(block.attn.ve_gate.weight, -sub_s, sub_s)

        # Final head
        if hasattr(self.final_head, 'proj'):
            torch.nn.init.uniform_(self.final_head.proj.weight, -sub_s, sub_s)
        if hasattr(self.final_head, 'agg_router'):
            torch.nn.init.zeros_(self.final_head.agg_router.router.weight)
        if hasattr(self.final_head, 'sub_heads'):
            for head in self.final_head.sub_heads:
                torch.nn.init.normal_(head.weight, std=0.001)

        # Per-layer scalars (matching base GPT init_weights)
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection

        # Value embeddings (init like c_v: uniform with same std as base GPT)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Rotary embeddings
        head_dim = self.config.mst_head_dim if self.config.mst_head_dim > 0 else self.config.mst_sub_dim // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to compute dtype
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    @property
    def transformer(self):
        """Compatibility shim for engine.py which accesses model.transformer.h.
        MST uses self.layers instead; return a minimal object so the engine's
        GQA/v_head_dim detection check passes gracefully (h is empty → check skipped)."""
        class _TransformerShim:
            h = []  # empty list — engine checks len() > 0 before accessing h[0]
        return _TransformerShim()

    @property
    def kv_cache_config(self):
        """Returns the correct KVCache constructor kwargs for this MST model.
        The engine checks for this property and uses it instead of computing from config.
        MST needs n_layer * N_subs cache slots (one per sub-transformer per layer)."""
        N = self.config.mst_n_subs
        head_dim = self.config.mst_head_dim if self.config.mst_head_dim > 0 else self.config.mst_sub_dim // self.config.n_head
        return {
            "num_heads": self.config.n_head,
            "head_dim": head_dim,
            "v_head_dim": head_dim,
            "num_layers": self.config.n_layer * N,
        }

    @property
    def max_seq_len(self):
        return self.config.sequence_len

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T

        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            head_dim = self.config.mst_head_dim if self.config.mst_head_dim > 0 else self.config.mst_sub_dim // self.config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        # Embed tokens
        x = self.wte(idx)  # (B, T, D)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Distribute to sub-transformers (Axis 1)
        N = self.config.mst_n_subs
        d = self.config.mst_sub_dim
        if self._use_batched:
            # ============ BATCHED TENSOR PATH ============
            # sub_states is a (B, T, N, d) tensor throughout — no lists, no stacking.
            if self.config.mst_input_mode == 'learned_proj':
                # Batched learned projection: (B,T,D) @ (N,d,D).T → (B,T,N,d)
                projs_w = torch.stack([p.weight for p in self.input_layer.projections], dim=0)  # (N, d, D)
                sub_states = torch.einsum('btD,ndD->btnd', x, projs_w.to(dtype=x.dtype))
            elif self.config.mst_input_mode == 'fixed_slice':
                sub_states = x.view(B, T, N, d)
            elif self.config.mst_input_mode == 'rotated_slice':
                rot = self.input_layer.rotation.to(x.dtype)
                x_rot = x @ rot.T
                sub_states = x_rot.view(B, T, N, d)
            else:
                # Fallback: use list-based input, then stack
                sub_list = self.input_layer(x, input_ids=idx)
                sub_states = torch.stack(sub_list, dim=2)

            # Save initial sub-states for x0 residual
            sub_x0 = sub_states.clone()

            # Process through L layers
            total_sub_layers = self.config.n_layer * N
            total_aux_loss = x.new_zeros(())

            for i, layer in enumerate(self.layers):
                # Per-layer residual scaling + x0 blend
                rl = self.resid_lambdas[i]
                x0l = self.x0_lambdas[i]
                sub_states = rl * sub_states + x0l * sub_x0

                # Value embedding: (B, T, d) shared across all subs
                ve = None
                if str(i) in self.value_embeds:
                    ve = self.value_embeds[str(i)](idx).to(sub_states.dtype)  # (B, T, d)

                # Per-sub window sizes (multi-scale or layer default)
                sub_ws = self.sub_window_sizes if self.sub_window_sizes is not None else None
                if sub_ws is None:
                    # Use the layer's window size for all subs
                    sub_ws = [self.window_sizes[i]] * N

                sub_states, aux_loss = layer(sub_states, cos_sin, ve=ve,
                                              window_sizes=sub_ws,
                                              kv_cache=kv_cache,
                                              total_sub_layers=total_sub_layers)
                total_aux_loss = total_aux_loss + aux_loss

                # Diagnostics (detached, no graph impact)
                if self._diag_enabled:
                    self._diag_sub_states[i] = [sub_states[:, :, j].detach() for j in range(N)]

            # Final output: concat_proj
            sub_states_normed = norm(sub_states)  # (B, T, N, d)
            concatenated = sub_states_normed.reshape(B, T, N * d)  # (B, T, D)
            h = self.final_head.proj(concatenated)  # (B, T, D)
            logits = self.lm_head(h)
            final_aux = x.new_zeros(())
            total_aux_loss = total_aux_loss + final_aux

        else:
            # ============ LEGACY LIST-BASED PATH ============
            if self.delta_residual:
                x_D = x
                sub_states = None
                sub_x0 = None
            else:
                if self.config.mst_input_mode == 'stem':
                    sub_states = self.input_layer.forward_with_cos_sin(x, cos_sin, input_ids=idx)
                else:
                    sub_states = self.input_layer(x, input_ids=idx)
                sub_x0 = [h.clone() for h in sub_states]
                x_D = None

            total_sub_layers = self.config.n_layer * N
            total_aux_loss = torch.tensor(0.0, device=x.device)
            global_stream = x if self.use_global_residual else None
            for i, layer in enumerate(self.layers):
                if self.delta_residual:
                    x_D_normed = norm(x_D)
                    sub_states = [self.delta_down_projs[j](x_D_normed) for j in range(N)]
                else:
                    rl = self.resid_lambdas[i]
                    x0l = self.x0_lambdas[i]
                    sub_states = [rl * h + x0l * h0 for h, h0 in zip(sub_states, sub_x0)]

                if global_stream is not None:
                    gs_normed = norm(global_stream)
                    for j in range(N):
                        sub_states[j] = sub_states[j] + self.global_read_projs[i][j](gs_normed)

                sub_ves = None
                current_sub_dim = sub_states[0].shape[-1]
                if str(i) in self.value_embeds and current_sub_dim == d:
                    ve_full = self.value_embeds[str(i)](idx).to(sub_states[0].dtype)
                    sub_ves = [ve_full] * len(sub_states)

                sub_ws = None
                if self.sub_window_sizes is not None:
                    n_current = len(sub_states)
                    sub_ws = self.sub_window_sizes[:n_current]

                sub_states, aux_loss = layer(sub_states, cos_sin, sub_ves=sub_ves,
                                              window_size=self.window_sizes[i],
                                              sub_window_sizes=sub_ws,
                                              kv_cache=kv_cache,
                                              total_sub_layers=total_sub_layers)
                total_aux_loss = total_aux_loss + aux_loss

                if self._diag_enabled:
                    self._diag_sub_states[i] = [h.detach() for h in sub_states]

                if self.delta_residual:
                    deltas = [self.delta_up_projs[j](sub_states[j]) for j in range(len(sub_states))]
                    x_D = x_D + torch.stack(deltas, dim=0).mean(dim=0)

                if i in self.merge_layers:
                    target_n = self._merge_schedule[i]
                    group_size = len(sub_states) // target_n
                    merged = []
                    for j in range(target_n):
                        group = sub_states[j * group_size : (j + 1) * group_size]
                        merged.append(torch.cat(group, dim=-1))
                    sub_states = merged
                    merged_x0 = []
                    for j in range(target_n):
                        group = sub_x0[j * group_size : (j + 1) * group_size]
                        merged_x0.append(torch.cat(group, dim=-1))
                    sub_x0 = merged_x0

                if global_stream is not None:
                    concatenated = torch.cat(sub_states, dim=-1)
                    global_stream = global_stream + self.global_write_projs[i](concatenated)

            # Final output
            if self.delta_residual:
                logits = self.lm_head(norm(x_D))
                final_aux = torch.tensor(0.0, device=logits.device)
            elif self.progressive_merge and len(sub_states) == 1:
                sub_states = [norm(sub_states[0])]
                logits = self.lm_head(sub_states[0])
                final_aux = torch.tensor(0.0, device=logits.device)
            else:
                sub_states = [norm(h) for h in sub_states]
                if self.config.mst_final_mode in ('aggregate_proj', 'concat_proj'):
                    logits, final_aux = self.final_head(sub_states, lm_head=self.lm_head)
                else:
                    logits, final_aux = self.final_head(sub_states)
            total_aux_loss = total_aux_loss + final_aux

        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        softcap = 20
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction
            )
            if loss_reduction == 'mean':
                loss = loss + total_aux_loss

                # H3: Per-sub auxiliary prediction loss
                if self.sub_aux_heads is not None and self.training:
                    sub_aux_loss = torch.tensor(0.0, device=logits.device)
                    for j, (head, sub_h) in enumerate(zip(self.sub_aux_heads, sub_states)):
                        sub_logits = head(sub_h)  # (B, T, vocab)
                        sub_logits = sub_logits[..., :self.config.vocab_size].float()
                        sub_logits = softcap * torch.tanh(sub_logits / softcap)
                        sub_loss = F.cross_entropy(
                            sub_logits.view(-1, sub_logits.size(-1)),
                            targets.view(-1), ignore_index=-1, reduction='mean'
                        )
                        sub_aux_loss = sub_aux_loss + sub_loss
                    loss = loss + self.sub_aux_weight * (sub_aux_loss / len(self.sub_aux_heads))

            return loss
        else:
            return logits

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        layers = sum(p.numel() for p in self.layers.parameters())
        input_layer = sum(p.numel() for p in self.input_layer.parameters())
        final_head = sum(p.numel() for p in self.final_head.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        transformer_matrices = layers + input_layer + final_head
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'wpe': 0, 'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': 0, 'scalars': scalars,
            'total': total,
        }

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, value embeds, and per-layer scalars
        # (matching dense GPT formula — these are lookups, not matrix multiplies)
        ve_numel = sum(p.numel() for n, p in self.named_parameters() if 'value_embed' in n)
        nparams_exclude = (self.wte.weight.numel() + ve_numel +
                           self.resid_lambdas.numel() + self.x0_lambdas.numel())
        N = self.config.mst_n_subs
        d = self.config.mst_sub_dim
        n_head = self.config.n_head
        head_dim = self.config.mst_head_dim if self.config.mst_head_dim > 0 else d // n_head
        t = self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window (matching dense GPT)
        attn_flops = 0
        for window_size in self.window_sizes:
            if self.sub_window_sizes is not None:
                # Multi-scale: each sub has its own window size (overrides layer window)
                for sub_ws in self.sub_window_sizes:
                    window = sub_ws[0]
                    effective_seq = t if window < 0 else min(window, t)
                    attn_flops += 12 * n_head * head_dim * effective_seq
            else:
                # All subs share the layer's window size
                window = window_size[0]  # (left, right) tuple, we use left
                effective_seq = t if window < 0 else min(window, t)
                attn_flops += N * 12 * n_head * head_dim * effective_seq
        total_flops = 6 * (nparams - nparams_exclude) + attn_flops

        # For topk_hard routing, only k/N sub-transformers execute per token.
        # Scale sub-transformer FLOPs and params by the active fraction.
        routing_mode = self.config.mst_routing_mode
        if routing_mode == 'topk_hard':
            k = self.config.mst_routing_topk
            active_fraction = k / N
        else:
            active_fraction = 1.0  # soft_weighted / sequence_path: all subs active

        # Params belonging to per-sub modules: scale by active_fraction.
        # Shared params (wte, lm_head, value_embeds, input_layer, final_head) are always active.
        sub_params = 0
        for layer in self.layers:
            if isinstance(layer, BatchedMSTLayer):
                # Batched layer: attention + FFN + VE gate weights are per-sub
                for name in ('c_q_w', 'c_k_w', 'c_v_w', 'c_proj_w', 'fc_w', 'fc_proj_w', 've_gate_w'):
                    p = getattr(layer, name, None)
                    if p is not None:
                        sub_params += p.numel()
            elif hasattr(layer, 'sub_blocks'):
                sub_params += sum(p.numel() for p in layer.sub_blocks.parameters())
        shared_params = nparams - sub_params
        active_params = int(shared_params + sub_params * active_fraction)

        # Scale only the sub-block portion of FLOPs; routing/transition/embedding overhead is always paid.
        sub_flops = 6 * sub_params + attn_flops
        shared_flops = total_flops - sub_flops
        active_flops = int(shared_flops + sub_flops * active_fraction)

        return total_flops, active_flops, active_params

    @torch.no_grad()
    def compute_diagnostics(self):
        """Compute diagnostic metrics from the last forward pass.

        Must be called after a forward with self._diag_enabled = True.
        Returns a flat dict of metrics suitable for JSONL logging.

        Metrics computed:
        - sub_sim_L{i}_mean/max: pairwise cosine similarity between sub outputs at layer i
        - sub_norm_L{i}_S{j}: activation L2 norm for sub j at layer i
        - route_entropy_L{i}: mean entropy of FFA routing weights at layer i
        - route_matrix_L{i}: mean N×N routing weight matrix at layer i (flattened)
        - grad_norm_S{j}: gradient norm for sub-block j (across all layers)
        """
        diag = {}
        N = self.config.mst_n_subs

        # --- Sub similarity and norms from stored sub_states ---
        for layer_idx, sub_states in self._diag_sub_states.items():
            # Each sub_state is (B, T, d). Flatten to (B*T, d) then average over batch.
            subs_flat = []
            for j, h in enumerate(sub_states):
                flat = h.float().reshape(-1, h.shape[-1])  # (B*T, d)
                # Per-sub activation norm (mean over tokens)
                diag[f'sub_norm_L{layer_idx}_S{j}'] = float(flat.norm(dim=-1).mean())
                subs_flat.append(F.normalize(flat.mean(dim=0), dim=0))  # (d,) normalized

            # Pairwise cosine similarity between subs
            subs_matrix = torch.stack(subs_flat, dim=0)  # (N, d)
            cos_sim = subs_matrix @ subs_matrix.T  # (N, N)
            # Upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=cos_sim.device), diagonal=1)
            off_diag = cos_sim[mask]
            if off_diag.numel() > 0:
                diag[f'sub_sim_L{layer_idx}_mean'] = float(off_diag.mean())
                diag[f'sub_sim_L{layer_idx}_max'] = float(off_diag.max())
                diag[f'sub_sim_L{layer_idx}_min'] = float(off_diag.min())

        # --- FFA routing entropy and weight distribution ---
        from nanochat.mst import MSTTransition
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchedMSTLayer):
                continue  # batched layers have inline routing, no separate transition
            trans = getattr(layer, 'transition', None)
            if trans is None:
                continue
            if trans.mode == 'free_for_all' and trans._last_route_weights is not None:
                rw = trans._last_route_weights  # (B, T, N_sender, N_target)
                # Mean routing matrix (averaged over batch and tokens)
                mean_matrix = rw.float().mean(dim=(0, 1))  # (N_sender, N_target)
                # Entropy per sender, averaged
                eps = 1e-8
                entropy = -(rw.float() * (rw.float() + eps).log()).sum(dim=-1).mean()
                diag[f'route_entropy_L{i}'] = float(entropy)
                # Store flattened mean routing matrix
                diag[f'route_matrix_L{i}'] = mean_matrix.cpu().tolist()

            elif trans.mode == 'aggregate_distribute':
                # For aggdist, check the router's last weights
                if hasattr(trans, 'router') and trans.router._last_entropy is not None:
                    diag[f'route_entropy_L{i}'] = float(trans.router._last_entropy)
                if hasattr(trans, 'router') and trans.router._last_balance is not None:
                    diag[f'route_balance_L{i}'] = float(trans.router._last_balance)

        # --- Per-sub gradient norms (cached by base_train.py before zero_grad) ---
        if hasattr(self, '_cached_grad_norms') and self._cached_grad_norms:
            diag.update(self._cached_grad_norms)
            self._cached_grad_norms = {}  # clear after reading

        # Clear stored data to free memory
        self._diag_sub_states.clear()

        return diag


    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2,
                        matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95),
                        scalar_lr=0.5, disable_mu_p=False, mu_p_scale_override=-1.0,
                        gate_lr_scale=0.3):
        from nanochat.optim import MuonAdamW, DistMuonAdamW
        ddp, rank, local_rank, world_size = get_dist_info()

        embed_params = list(self.wte.parameters())
        if hasattr(self.input_layer, 'sub_embeddings'):
            embed_params += list(self.input_layer.sub_embeddings.parameters())
        unembed_params = list(self.lm_head.parameters())
        if hasattr(self.final_head, 'sub_heads'):
            unembed_params += list(self.final_head.sub_heads.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # All remaining params: matrix (2D+) → Muon, scalar (1D) → AdamW
        # Note: BatchedMSTLayer stores weights as 2D (N*out, in) for Muon compatibility.
        covered_ids = {id(p) for p in embed_params + unembed_params + value_embeds_params + resid_params + x0_params}
        matrix_params = []
        scalar_params = []
        for p in self.parameters():
            if id(p) in covered_ids or not p.requires_grad:
                continue
            if p.ndim >= 2:
                matrix_params.append(p)
            else:
                scalar_params.append(p)

        # mu-P scaling
        model_dim = self.config.n_embd
        if mu_p_scale_override > 0.0:
            dmodel_lr_scale = mu_p_scale_override
            print0(f"μP LR scaling OVERRIDDEN to {dmodel_lr_scale:.6f}")
        elif disable_mu_p:
            dmodel_lr_scale = 1.0
            print0(f"μP LR scaling DISABLED (model_dim={model_dim})")
        else:
            dmodel_lr_scale = (model_dim / 768) ** -0.5
            print0(f"Scaling AdamW LRs ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups list matching the MuonAdamW interface (kind='muon'/'adamw')
        param_groups = [
            dict(kind='adamw', params=unembed_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embed_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=scalar_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Matrix params → Muon, grouped by shape (matching GPT convention)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if (ddp and world_size > 1) else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        return optimizer

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
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
            yield next_ids.item()
