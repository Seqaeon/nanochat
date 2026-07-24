"""Bilinear-Gated Transformer: SwiGLU-style FFN for modular arithmetic.

Key insight from the Strassen paper: the critical operation for function
composition is the Hadamard product (element-wise multiply) between value
vectors. This captures the MULTIPLICATIVE structure of the task.

For modular squaring (x^2 mod N), the core operation IS multiplication.
Standard relu().square() captures some multiplicative structure, but a
bilinear gate x ⊙ (Wx) is a more direct inductive bias for multiplication.

This submission uses a 2-block non-recurrent architecture (our fastest)
with a SwiGLU-style FFN: gate(x) ⊙ up(x) instead of relu(up(x)).square().
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

D_MODEL = 128
NUM_HEADS = 4
FFN_MULT = 4

WEIGHT_DECAY = 0.5
LR = 1e-3
GROKFAST_ALPHA = 0.98
GROKFAST_LAMB = 3.0


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


class Block(nn.Module):
    """Transformer block with SwiGLU FFN for multiplicative structure."""
    def __init__(self) -> None:
        super().__init__()
        self.na = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.nf = RMSNorm(D_MODEL)
        # SwiGLU: gate and up are separate projections, output = silu(gate) * up
        # This doubles the intermediate parameters but directly captures multiplication
        ffn_dim = FFN_MULT * D_MODEL
        self.gate = nn.Linear(D_MODEL, ffn_dim)
        self.up = nn.Linear(D_MODEL, ffn_dim)
        self.down = nn.Linear(ffn_dim, D_MODEL, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        r = x
        x = self.na(x)
        B, T, _ = x.shape
        hd = D_MODEL // NUM_HEADS

        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, NUM_HEADS, hd).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, hd).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, hd).transpose(1, 2)

        m = None
        if mask is not None:
            m = mask[:, None, None, :].to(dtype=torch.bool, device=x.device) if mask.ndim == 2 \
                else mask[:, None].to(dtype=torch.bool, device=x.device)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = r + self.out(x.transpose(1, 2).contiguous().view(B, T, -1))

        # SwiGLU FFN: silu(gate) ⊙ up — direct multiplicative gating
        h = self.nf(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Config:
    def __init__(self, vs: int, ms: int) -> None:
        self.vocab_size = vs
        self.max_seq_len = ms


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.head(self.norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
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
