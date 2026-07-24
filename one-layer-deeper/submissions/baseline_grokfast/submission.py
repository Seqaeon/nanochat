"""Baseline submission + Grokfast grokking optimizations.

Same vanilla Transformer as the baseline, but with:
  - GrokfastAdamW optimizer (EMA gradient filtering)
  - Very high weight decay (2.0)
  
Purpose: isolate whether the grokking treatment alone beats the baseline,
independent of the RecurrentRemix architecture.
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


WEIGHT_DECAY = 2.0
LR = 1e-3
GROKFAST_ALPHA = 0.98
GROKFAST_LAMB = 5.0


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


# ── Vanilla Transformer (from baseline_adamw) ──

class RMSNorm(nn.Module):
    def __init__(self, w: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(w))
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Attention(nn.Module):
    def __init__(self, d: int, nh: int) -> None:
        super().__init__()
        self.nh = nh
        self.hd = d // nh
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        m = None
        if mask is not None:
            m = mask[:, None, None, :].to(dtype=torch.bool, device=x.device) if mask.ndim == 2 else mask[:, None].to(dtype=torch.bool, device=x.device)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(x.transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, d: int, nh: int) -> None:
        super().__init__()
        self.na = RMSNorm(d)
        self.attn = Attention(d, nh)
        self.nf = RMSNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.attn(self.na(x), mask)
        h = F.relu(self.fc(self.nf(x))).square()
        return x + self.proj(h)


class Config:
    def __init__(self, vs: int, ms: int) -> None:
        self.vocab_size = vs
        self.max_seq_len = ms


class BaselineModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        d = 128
        nh = 4
        n_layers = 4
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, d)
        self.pos = nn.Embedding(spec.max_seq_len, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x, attention_mask)
        return self.head(self.norm(x)), None


def build_model(spec: ModelSpec) -> BaselineModel:
    model = BaselineModel(spec)
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
