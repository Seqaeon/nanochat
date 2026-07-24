"""IHA-Lite: Simplified cross-head mixing without sequence expansion.

The full IHA paper interleaves P pseudo-tokens into the sequence dimension,
creating a T*P length sequence. For seq_len=10 this is wasteful — we expand
to 40 tokens when the input only has 10.

IHA-Lite takes the core insight (cross-head mixing of Q/K/V) without the
pseudo-sequence expansion. Each head's Q/K/V is a learned linear combination
of ALL heads' Q/K/V. This gives H² effective attention patterns from H heads
with zero sequence-length overhead.

Combined with weight-tied recurrence and Grokfast for grokking.
"""

from __future__ import annotations

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
NUM_LOOPS = 4
FFN_MULT = 4
MAX_LOOPS = 32

WEIGHT_DECAY = 0.5
LR = 1e-3
GROKFAST_ALPHA = 0.98
GROKFAST_LAMB = 3.0


# ── Grokfast Optimizer ───────────────────────────────────────────────────────

class GrokfastAdamW(torch.optim.AdamW):
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


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# ── Cross-Head Mixed Attention (IHA-Lite) ────────────────────────────────────

class CrossHeadAttention(nn.Module):
    """Cross-head mixing of Q/K/V without sequence expansion.
    
    Core IHA idea: each head's Q/K/V is a learned linear combination of ALL
    heads' Q/K/V. This gives H² effective attention patterns from H heads.
    
    Unlike full IHA, we DON'T expand the sequence dimension with pseudo-tokens.
    For tiny seq_len=10, this is much more efficient.
    
    Implementation: after standard QKV projection, apply learned H×H mixing
    matrices to the head dimension of Q, K, V independently.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

        # Cross-head mixing: H×H matrices for Q, K, V
        # Initialize as identity + small noise (starts as standard MHA)
        self.mix_q = nn.Parameter(torch.eye(n_heads) + torch.randn(n_heads, n_heads) * 0.01)
        self.mix_k = nn.Parameter(torch.eye(n_heads) + torch.randn(n_heads, n_heads) * 0.01)
        self.mix_v = nn.Parameter(torch.eye(n_heads) + torch.randn(n_heads, n_heads) * 0.01)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        # Standard QKV projection
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, hd)  # (B, T, H, hd)
        k = k.view(B, T, H, hd)
        v = v.view(B, T, H, hd)

        # Cross-head mixing: each head becomes a combination of all heads
        # mix shape: (H_out, H_in), applied to head dimension
        q = torch.einsum('btid,oi->btod', q, self.mix_q)  # (B, T, H, hd)
        k = torch.einsum('btid,oi->btod', k, self.mix_k)
        v = torch.einsum('btid,oi->btod', v, self.mix_v)

        # Transpose to (B, H, T, hd) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention mask
        m = None
        if mask is not None:
            m = mask[:, None, None, :].to(dtype=torch.bool, device=x.device) if mask.ndim == 2 \
                else mask[:, None].to(dtype=torch.bool, device=x.device)

        # Standard SDPA
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, D))


# ── Recurrent Block ──────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.na = RMSNorm(d)
        self.attn = CrossHeadAttention(d, n_heads)
        self.nf = RMSNorm(d)
        self.up = nn.Linear(d, FFN_MULT * d)
        self.down = nn.Linear(FFN_MULT * d, d, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.attn(self.na(x), mask)
        h = self.up(self.nf(x))
        return x + self.down(F.relu(h).square())


# ── Model ────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, vs: int, ms: int) -> None:
        self.vocab_size = vs
        self.max_seq_len = ms


class IHALiteModel(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block(D_MODEL, NUM_HEADS)
        self.iter_emb = nn.Embedding(MAX_LOOPS, D_MODEL)
        nn.init.normal_(self.iter_emb.weight, std=0.02)
        self.resid_scale = nn.Parameter(torch.ones(MAX_LOOPS))
        self.x0_scale = nn.Parameter(torch.zeros(MAX_LOOPS) + 0.1)
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        x0 = x
        for i in range(self.num_loops):
            x = self.resid_scale[i] * x + self.x0_scale[i] * x0
            x = x + self.iter_emb(torch.tensor(i, device=ids.device))
            x = self.block(x, attention_mask)
        return self.head(self.norm(x)), None


def build_model(spec: ModelSpec) -> IHALiteModel:
    model = IHALiteModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        GrokfastAdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY, grokfast_alpha=GROKFAST_ALPHA,
            grokfast_lamb=GROKFAST_LAMB,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=32,
)
