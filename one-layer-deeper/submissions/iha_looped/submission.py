"""IHA-Looped: Interleaved Head Attention with weight-tied recurrence.

Theoretical motivation (from organizer-recommended papers):
  1. Strassen Attention (2501.19215): Standard attention PROVABLY cannot solve
     function composition — it only sees pairwise interactions. Modular squaring
     (x^(2^T) mod N) IS function composition: square, reduce, repeat.
  
  2. Interleaved Head Attention (2602.21371): H heads give H attention patterns.
     IHA constructs P pseudo-heads per head via cross-head mixing, yielding
     P² patterns per head. With H=4, P=4: 64 effective patterns vs 4.
     Overhead: only O(H²P) ≈ 64 extra parameters.

Architecture:
  - Weight-tied recurrent block (NUM_LOOPS iterations)
  - IHA attention: cross-head mixing of Q/K/V before standard SDPA
  - relu().square() FFN (quadratic activation for multiplicative structure)
  - Grokfast optimizer for grokking acceleration
  
Key insight: IHA gives the model enough relational capacity to simultaneously
parse N (3 digits), x (2 digits), T (1 digit), and route computation through
recurrent iterations — all with minimal parameter overhead.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────

D_MODEL = 128
NUM_HEADS = 4
NUM_PSEUDO = 4        # P pseudo-heads per head → P² = 16 patterns per head
NUM_LOOPS = 4         # weight-tied recurrence depth
FFN_MULT = 4
MAX_LOOPS = 32
CONTEXT_DIM = 32

# Grokking
WEIGHT_DECAY = 0.5
LR = 1e-3
GROKFAST_ALPHA = 0.98
GROKFAST_LAMB = 3.0


# ── Grokfast Optimizer ───────────────────────────────────────────────────────

class GrokfastAdamW(torch.optim.AdamW):
    """AdamW with Grokfast EMA gradient filtering."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1.0,
                 grokfast_alpha=0.98, grokfast_lamb=5.0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
        self.grokfast_alpha = grokfast_alpha
        self.grokfast_lamb = grokfast_lamb
        self._ema_grads: dict[int, Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        alpha = self.grokfast_alpha
        lamb = self.grokfast_lamb
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                if pid not in self._ema_grads:
                    self._ema_grads[pid] = torch.zeros_like(p.grad)
                ema = self._ema_grads[pid]
                ema.mul_(alpha).add_(p.grad, alpha=1 - alpha)
                p.grad.add_(ema, alpha=lamb)
        return super().step(closure)


# ── Utilities ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# ── Interleaved Head Attention ───────────────────────────────────────────────

class InterleavedHeadAttention(nn.Module):
    """IHA: Cross-head mixing of Q/K/V before standard scaled dot-product attention.
    
    Instead of each head operating in isolation (H patterns total), IHA constructs
    P pseudo-Q/K/V per head as learned linear combinations of ALL heads' Q/K/V.
    This yields up to P² attention patterns per head (H*P² total).
    
    For our config (H=4, P=4): 4 * 16 = 64 effective patterns vs 4 for standard MHA.
    Extra parameters: 3 * H * H * P = 3 * 4 * 4 * 4 = 192 (negligible).
    """

    def __init__(self, d_model: int, n_heads: int, n_pseudo: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_pseudo = n_pseudo
        self.head_dim = d_model // n_heads

        # Standard QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Cross-head mixing coefficients: α^Q, α^K, α^V ∈ R^{H × H × P}
        # Initialize as identity (pseudo-head p for head h = original head h)
        # so IHA starts as standard MHA and learns to deviate
        self.alpha_q = nn.Parameter(self._init_mixing(n_heads, n_pseudo))
        self.alpha_k = nn.Parameter(self._init_mixing(n_heads, n_pseudo))
        self.alpha_v = nn.Parameter(self._init_mixing(n_heads, n_pseudo))

        # Collapse matrix: R ∈ R^{H × (H*P)} — selects from pseudo-heads
        # Initialize to pick first pseudo-head from each head (= MHA behavior)
        collapse = torch.zeros(n_heads, n_heads * n_pseudo)
        for h in range(n_heads):
            collapse[h, h * n_pseudo] = 1.0
        self.collapse = nn.Parameter(collapse)

    @staticmethod
    def _init_mixing(n_heads: int, n_pseudo: int) -> Tensor:
        """Initialize mixing as identity: pseudo p of head h = original head h."""
        alpha = torch.zeros(n_heads, n_heads, n_pseudo)
        for h in range(n_heads):
            for p in range(n_pseudo):
                alpha[h, h, p] = 1.0  # start as identity routing
        # Add small noise to break symmetry across pseudo-heads
        alpha += torch.randn_like(alpha) * 0.01
        return alpha

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, T, D = x.shape
        H, P, hd = self.n_heads, self.n_pseudo, self.head_dim

        # Standard QKV projection → (B, T, H, hd) each
        qkv = self.qkv(x).view(B, T, 3, H, hd)
        q_all = qkv[:, :, 0]  # (B, T, H, hd)
        k_all = qkv[:, :, 1]
        v_all = qkv[:, :, 2]

        # Cross-head mixing: construct P pseudo-heads per head
        # α shape: (H_out, H_in, P)
        # q_all shape: (B, T, H_in, hd)
        # Result: (B, T, H_out, P, hd)
        q_pseudo = torch.einsum('bthd,ohp->btopd', q_all, self.alpha_q)
        k_pseudo = torch.einsum('bthd,ohp->btopd', k_all, self.alpha_k)
        v_pseudo = torch.einsum('bthd,ohp->btopd', v_all, self.alpha_v)

        # Merge pseudo-heads into sequence dimension for standard SDPA
        # (B, T, H, P, hd) → (B, H, T*P, hd)
        q_merged = q_pseudo.reshape(B, T * P, H, hd).transpose(1, 2)
        k_merged = k_pseudo.reshape(B, T * P, H, hd).transpose(1, 2)
        v_merged = v_pseudo.reshape(B, T * P, H, hd).transpose(1, 2)

        # Build attention mask for merged sequence
        attn_mask = None
        if mask is not None:
            # Expand mask: each original token's mask applies to all P pseudo-tokens
            # mask: (B, T) → (B, T*P)
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, P).reshape(B, T * P)
            attn_mask = expanded_mask[:, None, None, :].to(dtype=torch.bool, device=x.device)

        # Standard SDPA on merged sequence
        attn_out = F.scaled_dot_product_attention(
            q_merged, k_merged, v_merged, attn_mask=attn_mask
        )  # (B, H, T*P, hd)

        # Reshape to (B, T, H*P, hd) for collapse
        attn_out = attn_out.transpose(1, 2).reshape(B, T, P, H, hd)
        attn_out = attn_out.reshape(B, T, H * P, hd)

        # Collapse: (H, H*P) × (B, T, H*P, hd) → (B, T, H, hd)
        collapsed = torch.einsum('oh,bthd->btod', self.collapse, attn_out)

        # Concatenate heads and project
        output = collapsed.reshape(B, T, D)
        return self.out_proj(output)


# ── Recurrent Block ──────────────────────────────────────────────────────────

class RecurrentBlock(nn.Module):
    """Single weight-tied block with IHA + quadratic FFN."""

    def __init__(self, d: int, n_heads: int, n_pseudo: int) -> None:
        super().__init__()
        self.norm_a = RMSNorm(d)
        self.attn = InterleavedHeadAttention(d, n_heads, n_pseudo)
        self.norm_f = RMSNorm(d)
        self.up = nn.Linear(d, FFN_MULT * d)
        self.down = nn.Linear(FFN_MULT * d, d, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        # IHA attention
        x = x + self.attn(self.norm_a(x), mask)
        # Quadratic FFN (relu² captures multiplicative structure)
        h = self.up(self.norm_f(x))
        return x + self.down(F.relu(h).square())


# ── Model ────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, vs: int, ms: int) -> None:
        self.vocab_size = vs
        self.max_seq_len = ms


class IHALoopedModel(nn.Module):
    """IHA-Looped Transformer: weight-tied recurrence with Interleaved Head Attention."""
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)

        # Single weight-tied block with IHA
        self.block = RecurrentBlock(D_MODEL, NUM_HEADS, NUM_PSEUDO)

        # Iteration embedding for recurrence
        self.iter_emb = nn.Embedding(MAX_LOOPS, D_MODEL)
        nn.init.normal_(self.iter_emb.weight, std=0.02)

        # Residual blending per iteration
        self.resid_scale = nn.Parameter(torch.ones(MAX_LOOPS))
        self.x0_scale = nn.Parameter(torch.zeros(MAX_LOOPS) + 0.1)

        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        x0 = x

        for i in range(self.num_loops):
            # Residual blending with input
            x = self.resid_scale[i] * x + self.x0_scale[i] * x0
            # Add iteration embedding (broadcast over sequence)
            x = x + self.iter_emb(torch.tensor(i, device=ids.device))
            # Apply weight-tied block
            x = self.block(x, attention_mask)

        return self.head(self.norm(x)), None


# ── Submission ───────────────────────────────────────────────────────────────

def build_model(spec: ModelSpec) -> IHALoopedModel:
    model = IHALoopedModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        GrokfastAdamW(
            model.parameters(),
            lr=LR,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
            grokfast_alpha=GROKFAST_ALPHA,
            grokfast_lamb=GROKFAST_LAMB,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=32,
)
