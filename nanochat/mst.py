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
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
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
    """

    def __init__(self, d, mode='standard'):
        super().__init__()
        self.mode = mode
        self.c_fc = Linear(d, 4 * d, bias=False)
        if mode == 'standard':
            self.c_proj = Linear(4 * d, d, bias=False)
        else:
            self.c_proj = None  # no down-projection

    @property
    def out_dim(self):
        return self.c_fc.out_features if self.mode == 'no_downproj' else self.c_fc.in_features

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        if self.c_proj is not None:
            x = self.c_proj(x)
        return x


class SubTransformerBlock(nn.Module):
    """A single sub-transformer block: pre-norm attention + FFN with residuals."""

    def __init__(self, d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx=0, ffn_mode='standard'):
        super().__init__()
        self.attn = SubTransformerAttention(d, n_head, head_dim, layer_idx, n_layer, sub_layer_idx)
        self.ffn = SubTransformerFFN(d, mode=ffn_mode)
        # no_downproj: FFN outputs 4d — need a block-level projection back to d for the residual.
        # This decouples the expansion (d→4d + relu²) from compression (plain 4d→d linear),
        # which is the inductive bias difference being tested vs standard mode.
        self.ffn_out_proj = Linear(4 * d, d, bias=False) if ffn_mode == 'no_downproj' else None
        self.d = d

    def forward(self, x, cos_sin, ve=None, window_size=(-1, 0), kv_cache=None, total_sub_layers=1):
        """x: (B, T, d). Returns (B, T, d)."""
        x = x + self.attn(norm(x), cos_sin, ve=ve, window_size=window_size,
                          kv_cache=kv_cache, total_sub_layers=total_sub_layers)
        if self.ffn.mode == 'standard':
            x = x + self.ffn(norm(x))
        else:  # no_downproj: FFN gives 4d, block-level proj brings back to d
            x = x + self.ffn_out_proj(self.ffn(norm(x)))
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
        if self.training and self.diversity_weight > 0.0:
            normed = F.normalize(stacked, dim=-1)  # (B, T, N, d)
            # (B*T, N, N) pairwise cosine similarity
            flat = normed.flatten(0, 1)  # (B*T, N, d)
            sim = torch.bmm(flat, flat.transpose(-1, -2))  # (B*T, N, N)
            # Mean off-diagonal similarity
            mask = ~torch.eye(N, device=sim.device, dtype=torch.bool).unsqueeze(0)
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
    """

    def __init__(self, d, N, D, mode='parallel', diversity_weight=0.0):
        super().__init__()
        self.mode = mode
        self.d = d
        self.N = N

        if mode == 'aggregate_distribute':
            self.router = MSTRouter(d, N, D, mode='soft_weighted', diversity_weight=diversity_weight)
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
        else:
            raise ValueError(f"Unknown transition mode: {self.mode}")


class MSTFinalHead(nn.Module):
    """Axis 5: How the final layer produces vocabulary logits.

    Modes:
      'aggregate_proj':   Weighted sum of sub outputs -> single lm_head projection.
      'weighted_logits':  Each sub has independent lm_head; weighted sum of logits.
      'concat_proj':      Concatenate all N sub outputs, project to D, then lm_head.
    """

    def __init__(self, d, N, D, vocab_size, mode='aggregate_proj', diversity_weight=0.0):
        super().__init__()
        self.mode = mode
        self.d = d
        self.N = N
        self.D = D

        if mode == 'aggregate_proj':
            self.agg_router = MSTRouter(d, N, D, mode='soft_weighted', diversity_weight=diversity_weight)
            self.proj = Linear(d, D, bias=False)  # d -> D before lm_head
        elif mode == 'weighted_logits':
            self.sub_heads = nn.ModuleList([
                Linear(d, vocab_size, bias=False) for _ in range(N)
            ])
            self.head_weights = nn.Parameter(torch.ones(N) / N)
        elif mode == 'concat_proj':
            # Concatenate all N sub outputs (N*d), project directly to D.
            self.proj = Linear(N * d, D, bias=False)
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
            concatenated = torch.cat(sub_outputs, dim=-1)  # (B, T, N*d)
            h = self.proj(concatenated)  # (B, T, D)
            logits = lm_head(h)
            return logits, torch.tensor(0.0, device=concatenated.device)


class MSTLayer(nn.Module):
    """One full MST layer: N parallel sub-transformer blocks + transition."""

    def __init__(self, config, layer_idx):
        super().__init__()
        d = config.mst_sub_dim
        N = config.mst_n_subs
        D = config.n_embd
        n_head = config.n_head
        n_layer = config.n_layer

        self.N = N
        self.layer_idx = layer_idx
        head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // n_head
        self.sub_blocks = nn.ModuleList([
            SubTransformerBlock(d, n_head, head_dim, layer_idx, n_layer,
                                sub_layer_idx=layer_idx * N + j,
                                ffn_mode=config.mst_ffn_mode)
            for j in range(N)
        ])
        self.transition = MSTTransition(d, N, D, mode=config.mst_transition_mode,
                                         diversity_weight=config.mst_diversity_weight)

    def forward(self, sub_inputs, cos_sin, sub_ves=None, window_size=(-1, 0),
                kv_cache=None, total_sub_layers=1):
        """sub_inputs: list of N tensors (B, T, d).
        sub_ves: optional list of N tensors (B, T, d) value embeddings.
        Returns: list of N tensors (B, T, d), aux_loss."""
        if sub_ves is not None:
            sub_outputs = [
                block(h, cos_sin, ve=ve, window_size=window_size,
                      kv_cache=kv_cache, total_sub_layers=total_sub_layers)
                for block, h, ve in zip(self.sub_blocks, sub_inputs, sub_ves)
            ]
        else:
            sub_outputs = [
                block(h, cos_sin, window_size=window_size,
                      kv_cache=kv_cache, total_sub_layers=total_sub_layers)
                for block, h in zip(self.sub_blocks, sub_inputs)
            ]
        return self.transition(sub_outputs)


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
        self.layers = nn.ModuleList([
            MSTLayer(config, layer_idx=i) for i in range(config.n_layer)
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
        self.final_head = MSTFinalHead(
            d, N, D, padded_vocab_size,
            mode=config.mst_final_mode,
            diversity_weight=config.mst_diversity_weight,
        )

        # Standard lm_head for aggregate_proj mode
        self.lm_head = Linear(D, padded_vocab_size, bias=False)

        # Rotary embeddings (precomputed, same as GPT)
        head_dim = config.mst_head_dim if config.mst_head_dim > 0 else d // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

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

        # Sub-transformer blocks
        sub_s = 1.0 / (self.config.mst_sub_dim ** 0.5)
        for layer in self.layers:
            for block in layer.sub_blocks:
                torch.nn.init.uniform_(block.attn.c_q.weight, -sub_s, sub_s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -sub_s, sub_s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -sub_s, sub_s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                torch.nn.init.uniform_(block.ffn.c_fc.weight, -sub_s, sub_s)
                if block.ffn.c_proj is not None:
                    torch.nn.init.zeros_(block.ffn.c_proj.weight)

        # Transition layers
        for layer in self.layers:
            if hasattr(layer.transition, 'distribute'):
                for proj in layer.transition.distribute:
                    torch.nn.init.uniform_(proj.weight, -sub_s, sub_s)
            if hasattr(layer.transition, 'cross_weights'):
                torch.nn.init.zeros_(layer.transition.cross_weights)
            if hasattr(layer.transition, 'router'):
                torch.nn.init.zeros_(layer.transition.router.router.weight)

        # Final head
        if hasattr(self.final_head, 'proj'):
            torch.nn.init.uniform_(self.final_head.proj.weight, -sub_s, sub_s)
        if hasattr(self.final_head, 'agg_router'):
            torch.nn.init.zeros_(self.final_head.agg_router.router.weight)
        if hasattr(self.final_head, 'sub_heads'):
            for head in self.final_head.sub_heads:
                torch.nn.init.normal_(head.weight, std=0.001)

        # Router weights in all layers
        for layer in self.layers:
            tr = layer.transition
            if hasattr(tr, 'router') and hasattr(tr.router, 'router'):
                torch.nn.init.zeros_(tr.router.router.weight)

        # VE gates in sub-transformer attention (init like c_v projection)
        for layer in self.layers:
            for block in layer.sub_blocks:
                if block.attn.ve_gate is not None:
                    torch.nn.init.uniform_(block.attn.ve_gate.weight, -sub_s, sub_s)

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
        if self.config.mst_input_mode == 'stem':
            sub_states = self.input_layer.forward_with_cos_sin(x, cos_sin, input_ids=idx)
        else:
            sub_states = self.input_layer(x, input_ids=idx)

        # Save initial sub-states for x0 residual (matching GPT's x0 pattern)
        sub_x0 = [h.clone() for h in sub_states]

        # Process through L layers
        N = self.config.mst_n_subs
        d = self.config.mst_sub_dim
        total_sub_layers = self.config.n_layer * N
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers):
            # Per-layer residual scaling + x0 blend (matching base GPT)
            rl = self.resid_lambdas[i]
            x0l = self.x0_lambdas[i]
            sub_states = [rl * h + x0l * h0 for h, h0 in zip(sub_states, sub_x0)]

            # Value embeddings: per-sub VE from shared embedding, sliced like input
            sub_ves = None
            if str(i) in self.value_embeds:
                ve_full = self.value_embeds[str(i)](idx).to(sub_states[0].dtype)  # (B, T, d)
                # Each sub-transformer gets the same VE (at sub_dim d)
                sub_ves = [ve_full] * N

            sub_states, aux_loss = layer(sub_states, cos_sin, sub_ves=sub_ves,
                                         window_size=self.window_sizes[i],
                                         kv_cache=kv_cache,
                                         total_sub_layers=total_sub_layers)
            total_aux_loss = total_aux_loss + aux_loss

        # Normalize each sub output
        sub_states = [norm(h) for h in sub_states]

        # Final output (Axis 5)
        if self.config.mst_final_mode == 'aggregate_proj':
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
        nparams_exclude = self.wte.weight.numel()
        N = self.config.mst_n_subs
        d = self.config.mst_sub_dim
        n_head = self.config.n_head
        head_dim = d // n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * N * 12 * n_head * head_dim * t
        total_flops = 6 * (nparams - nparams_exclude) + attn_flops

        # For topk_hard routing, only k/N sub-transformers execute per token.
        # Scale sub-transformer FLOPs and params by the active fraction.
        routing_mode = self.config.mst_routing_mode
        if routing_mode == 'topk_hard':
            k = self.config.mst_routing_topk
            active_fraction = k / N
        else:
            active_fraction = 1.0  # soft_weighted / sequence_path: all subs active

        # Params belonging to per-sub modules (sub_blocks): scale by active_fraction.
        # Shared params (wte, lm_head, value_embeds, input_layer, final_head) are always active.
        sub_params = sum(p.numel() for layer in self.layers for p in layer.sub_blocks.parameters())
        shared_params = nparams - sub_params
        active_params = int(shared_params + sub_params * active_fraction)

        # Scale only the sub-block portion of FLOPs; routing/transition/embedding overhead is always paid.
        sub_flops = 6 * sub_params + attn_flops
        shared_flops = total_flops - sub_flops
        active_flops = int(shared_flops + sub_flops * active_fraction)

        return total_flops, active_flops, active_params


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
