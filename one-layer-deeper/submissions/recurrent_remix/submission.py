"""Recurrent Transformer with multi-template RemixedLinear + Grokking optimizations.

The core insight: modular arithmetic generalization is a GROKKING problem.
The model memorizes quickly but generalizes only after much longer training
with the right regularization. Key techniques:

1. VERY HIGH weight decay (1.0) — forces compact Fourier representations
2. Grokfast EMA gradient filtering — amplifies slow "generalization" gradients
3. Multi-template RemixedLinear — per-token soft routing over 8 templates
4. Recurrent depth — 4 weight-tied loops for serial computation

References:
  - Power et al. 2022: "Grokking: Generalization Beyond Overfitting"
  - Lee et al. 2024: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
"""

from __future__ import annotations

import math
from collections import OrderedDict

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
BASIS_SIZE = 64
N_TEMPLATES = 8
CONTEXT_DIM = 32
FFN_MULT = 4
MAX_LOOPS = 32

# Grokking-specific
WEIGHT_DECAY = 1.0       # HIGH weight decay — key for grokking
LR = 1e-3
GROKFAST_ALPHA = 0.98    # EMA momentum for gradient filter
GROKFAST_LAMB = 5.0      # Amplification factor for slow gradients


# ── Grokfast: EMA gradient filter ────────────────────────────────────────────
#
# From Lee et al. 2024: apply between loss.backward() and optimizer.step().
# Amplifies slow-varying gradient components that drive generalization.
# We implement this via a custom training_loss that wraps the standard CE
# and stores the EMA state in the model (since we can't modify the training loop).
#
# BUT: the competition runner calls loss.backward() and optimizer.step() for us.
# We CAN use a custom optimizer that applies Grokfast internally.

class GrokfastAdamW(torch.optim.AdamW):
    """AdamW with integrated Grokfast EMA gradient filtering.
    
    After each backward pass, before the Adam update, this optimizer:
    1. Computes EMA of gradients (slow-varying component)
    2. Adds amplified slow component back to the current gradient
    
    This accelerates the transition from memorization to generalization.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1.0,
                 grokfast_alpha=0.98, grokfast_lamb=5.0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
        self.grokfast_alpha = grokfast_alpha
        self.grokfast_lamb = grokfast_lamb
        self._ema_grads: dict[int, Tensor] = {}
    
    @torch.no_grad()
    def step(self, closure=None):
        # Apply Grokfast EMA filter to all gradients before the Adam update
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
                # Amplify slow component: grad += lamb * ema
                p.grad.add_(ema, alpha=lamb)
        
        return super().step(closure)


# ── Utility ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, w: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(w))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# ── Multi-Template RemixedLinear ─────────────────────────────────────────────

class MultiTemplateRemixedLinear(nn.Module):
    """K-template RemixedLinear with soft per-token routing and context gates."""

    def __init__(self, in_f: int, out_f: int, ctx_dim: int, basis_size: int, n_templates: int = 8) -> None:
        super().__init__()
        self.n_templates = n_templates
        self.basis = nn.Linear(in_f, basis_size, bias=False)
        self.ln_basis = nn.LayerNorm(basis_size)
        self.template_bank = nn.Parameter(torch.randn(n_templates, out_f, basis_size))
        self.template_route = nn.Parameter(torch.randn(in_f, n_templates) / (in_f ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_f))

        self.basis_gate = nn.Linear(ctx_dim, basis_size, bias=True)
        nn.init.zeros_(self.basis_gate.weight)
        nn.init.zeros_(self.basis_gate.bias)

        r = 4
        self.out_coeffs = nn.Linear(ctx_dim, r, bias=True)
        self.out_vecs = nn.Parameter(torch.zeros(r, out_f))
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)

    @torch.no_grad()
    def init_weights(self) -> None:
        for k in range(self.n_templates):
            nn.init.kaiming_normal_(self.template_bank.data[k])

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        h_basis = self.ln_basis(self.basis(x))
        gate = torch.sigmoid(self.basis_gate(ctx))
        h_gated = h_basis * gate

        route_logits = x.float() @ self.template_route.float()
        route_weights = F.softmax(route_logits, dim=-1).to(x.dtype)

        T_stack = self.template_bank.to(dtype=x.dtype)
        all_out = torch.einsum('bts,kos->btko', h_gated, T_stack)
        pre_output = (all_out * route_weights.unsqueeze(-1)).sum(dim=2) + self.bias

        out_gate = 1.0 + torch.tanh(self.out_scale * (self.out_coeffs(ctx) @ self.out_vecs))
        return pre_output * out_gate


# ── Bidirectional Attention ──────────────────────────────────────────────────

class BidirAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        m = None
        if mask is not None:
            if mask.ndim == 2:
                m = mask[:, None, None, :].to(dtype=torch.bool, device=x.device)
            else:
                m = mask[:, None].to(dtype=torch.bool, device=x.device)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(x.transpose(1, 2).contiguous().view(B, T, -1))


# ── Recurrent Block ─────────────────────────────────────────────────────────

class RecurrentBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ctx_dim: int, basis: int, n_templates: int) -> None:
        super().__init__()
        self.norm_a = RMSNorm(d)
        self.attn = BidirAttention(d, n_heads)
        self.norm_f = RMSNorm(d)
        self.ffn_up = MultiTemplateRemixedLinear(d, FFN_MULT * d, ctx_dim, basis, n_templates)
        self.ffn_down = nn.Linear(FFN_MULT * d, d, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None, ctx: Tensor) -> Tensor:
        x = x + self.attn(self.norm_a(x), mask)
        h = self.ffn_up(self.norm_f(x), ctx)
        h = F.relu(h).square()
        return x + self.ffn_down(h)


# ── Model ────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RecurrentRemixModel(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok_emb = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(spec.max_seq_len, D_MODEL)

        self.iter_emb = nn.Embedding(MAX_LOOPS, CONTEXT_DIM)
        self.ctx_proj = nn.Linear(D_MODEL, CONTEXT_DIM, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)

        self.block = RecurrentBlock(D_MODEL, NUM_HEADS, CONTEXT_DIM, BASIS_SIZE, N_TEMPLATES)

        self.resid_scale = nn.Parameter(torch.ones(MAX_LOOPS))
        self.x0_scale = nn.Parameter(torch.zeros(MAX_LOOPS) + 0.1)

        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x0 = x

        for i in range(self.num_loops):
            rl = self.resid_scale[i]
            x0l = self.x0_scale[i]
            x = rl * x + x0l * x0

            ie = self.iter_emb(torch.tensor(i, device=input_ids.device))
            cs = self.ctx_proj(x.mean(dim=1, keepdim=True))
            ctx = (ie.unsqueeze(0).unsqueeze(0) + cs).expand(-1, T, -1)

            x = self.block(x, attention_mask, ctx)

        return self.head(self.final_norm(x)), None


# ── Submission ───────────────────────────────────────────────────────────────

LABEL_SMOOTHING = 0.1

def training_loss(logits: Tensor, labels: Tensor, auxiliary) -> Tensor:
    """Label smoothing prevents overconfident memorization."""
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)


def build_model(spec: ModelSpec) -> RecurrentRemixModel:
    model = RecurrentRemixModel(spec)
    model.block.ffn_up.init_weights()
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
    training_loss=training_loss,
    batch_size=32,
)
