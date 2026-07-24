"""2-Block + Fourier Init: Best baseline + Fourier-initialized embeddings.

Grokking research (Power et al. 2022) showed that the generalizing solution
for modular arithmetic uses Fourier/trigonometric representations of the
inputs. Standard random init forces the model to discover these representations
through gradient descent — which is the SLOW part of grokking.

By initializing digit embeddings with sin/cos at multiple frequencies, we
provide the right basis functions from day 1. The model still needs to learn
the correct COMBINATION of frequencies (specific to each modulus N), but
starts much closer to the solution.

Architecture: identical to recurrent_remix_hard (our 5.2% best), only change
is the embedding initialization.
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

D_MODEL = 128
NUM_HEADS = 4

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
    def __init__(self, w: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(w))
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def fourier_init_embedding(vocab_size: int, d_model: int) -> nn.Embedding:
    """Initialize digit token embeddings with multi-frequency Fourier features.
    
    Token layout: 0:PAD 1:BOS 2:N 3:X 4:T 5:ANS 6:EOS 7-16:digits(0-9)
    
    For digits: embed digit value d as [sin(ω₁d), cos(ω₁d), sin(ω₂d), cos(ω₂d), ...]
    with frequencies chosen to span useful periods for modular arithmetic.
    """
    emb = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(emb.weight, std=0.02)
    
    DIGIT_OFFSET = 7
    n_freqs = d_model // 2
    
    with torch.no_grad():
        for digit in range(min(10, vocab_size - DIGIT_OFFSET)):
            tok_id = DIGIT_OFFSET + digit
            if tok_id >= vocab_size:
                break
            vec = torch.zeros(d_model)
            for f in range(n_freqs):
                # Use diverse frequencies: some aligned to base-10, others to primes
                # This gives basis functions useful for any modulus N
                freq = 2.0 * math.pi * (f + 1) / 10.0
                vec[2 * f] = math.sin(freq * digit) * 0.5
                vec[2 * f + 1] = math.cos(freq * digit) * 0.5
            emb.weight.data[tok_id] = vec
    
    return emb


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.na = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.nf = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        r = x
        x = self.na(x)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, -1).transpose(1, 2)
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
        self.tok = fourier_init_embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying still

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
