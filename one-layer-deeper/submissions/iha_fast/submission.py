"""IHA-Fast: 2-block non-recurrent transformer with cross-head mixing.

Combines two findings:
  1. 2-block non-recurrent is our fastest model (61K steps in 600s, 5.2% mean)
  2. Cross-head mixing gives best OOD (7.0%) but was unstable with recurrence

This version: stacked 2-block (NOT weight-tied) + cross-head Q/K/V mixing.
The key difference from IHA-Lite: no recurrence, no iteration embedding.
Each block has its OWN cross-head mixing matrices, allowing specialization.
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
    """Transformer block with cross-head Q/K/V mixing."""
    def __init__(self) -> None:
        super().__init__()
        self.na = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        # Cross-head mixing matrices (H×H each, ~48 extra params total)
        self.mix_q = nn.Parameter(torch.eye(NUM_HEADS) + torch.randn(NUM_HEADS, NUM_HEADS) * 0.01)
        self.mix_k = nn.Parameter(torch.eye(NUM_HEADS) + torch.randn(NUM_HEADS, NUM_HEADS) * 0.01)
        self.mix_v = nn.Parameter(torch.eye(NUM_HEADS) + torch.randn(NUM_HEADS, NUM_HEADS) * 0.01)
        self.nf = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, FFN_MULT * D_MODEL)
        self.down = nn.Linear(FFN_MULT * D_MODEL, D_MODEL, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        r = x
        x = self.na(x)
        B, T, _ = x.shape
        hd = D_MODEL // NUM_HEADS

        # QKV projection
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, NUM_HEADS, hd)
        k = k.view(B, T, NUM_HEADS, hd)
        v = v.view(B, T, NUM_HEADS, hd)

        # Cross-head mixing
        q = torch.einsum('btid,oi->btod', q, self.mix_q)
        k = torch.einsum('btid,oi->btod', k, self.mix_k)
        v = torch.einsum('btid,oi->btod', v, self.mix_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        m = None
        if mask is not None:
            m = mask[:, None, None, :].to(dtype=torch.bool, device=x.device) if mask.ndim == 2 \
                else mask[:, None].to(dtype=torch.bool, device=x.device)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = r + self.out(x.transpose(1, 2).contiguous().view(B, T, -1))

        h = self.up(self.nf(x))
        return x + self.down(F.relu(h).square())


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
