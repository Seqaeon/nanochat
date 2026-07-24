"""Recurrent Remix v3: Architectural innovations for algorithmic generalization.

Beyond hyperparameter tuning — this version attacks generalization structurally:

1. SCRATCHPAD TOKENS: Prepend learnable "thinking" tokens that act as working
   memory during recurrent passes. The model can accumulate intermediate state
   in these registers (like a scratch area for multi-step computation).

2. CONTEXT-AWARE TEMPLATE ROUTING: The previous version's routing was
   input-only (x @ route_matrix). Now routing depends on BOTH input AND
   iteration context. This lets the model use different templates at different
   recurrent steps — critical for iterative algorithms.

3. CUSTOM TRAINING LOSS: Label smoothing (0.1) prevents overconfident
   memorization, keeping the model in a "searching" state longer. Combined
   with an auxiliary loss that encourages orthogonality between templates
   (prevents collapse to a single effective weight matrix).

4. RECURRENT DROPOUT: During training, randomly zero portions of the hidden
   state between recurrent iterations. Forces redundant representations and
   prevents "fragile" memorization circuits.

5. TEMPLATE DIVERSITY LOSS: Auxiliary loss penalizing template similarity.
   Ensures the 8 templates learn DIFFERENT functions rather than converging
   to the same matrix (which wastes capacity).

References:
  - Nye et al. 2021: "Show Your Work: Scratchpads for Intermediate Computation"
  - Lee et al. 2024: "Grokfast"
  - Yang et al. 2024: "Looped Transformers as Programmable Computers"
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
NUM_LOOPS = 4
BASIS_SIZE = 64
N_TEMPLATES = 8
CONTEXT_DIM = 32
FFN_MULT = 4
MAX_LOOPS = 32
N_SCRATCH = 4           # number of scratchpad / memory tokens

# Grokking
WEIGHT_DECAY = 1.0
LR = 1e-3
GROKFAST_ALPHA = 0.98
GROKFAST_LAMB = 5.0

# Regularization
LABEL_SMOOTHING = 0.1    # prevent overconfident memorization
TEMPLATE_DIV_WEIGHT = 0.01  # auxiliary loss for template diversity
RECURRENT_DROPOUT = 0.1  # dropout between recurrent iterations


# ── Grokfast optimizer ───────────────────────────────────────────────────────

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


# ── Utility ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, w: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(w))
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# ── Context-Aware Multi-Template RemixedLinear ───────────────────────────────

class ContextAwareRemixedLinear(nn.Module):
    """Multi-template RemixedLinear with context-dependent routing.

    Key innovation over v2: template routing now depends on BOTH the input
    AND the iteration context vector. This means the model can learn to use
    different templates at different recurrent steps:
    - Step 1: "parse input digits" template
    - Step 2: "multiply" template
    - Step 3: "reduce mod N" template
    - Step 4: "format output" template

    Also includes template diversity regularization via orthogonality loss.
    """

    def __init__(self, in_f: int, out_f: int, ctx_dim: int, basis_size: int, n_templates: int = 8) -> None:
        super().__init__()
        self.n_templates = n_templates
        self.basis = nn.Linear(in_f, basis_size, bias=False)
        self.ln_basis = nn.LayerNorm(basis_size)
        self.template_bank = nn.Parameter(torch.randn(n_templates, out_f, basis_size))
        # Context-aware routing: projects BOTH input and context to routing logits
        self.template_route_x = nn.Linear(in_f, n_templates, bias=False)
        self.template_route_ctx = nn.Linear(ctx_dim, n_templates, bias=False)
        nn.init.zeros_(self.template_route_ctx.weight)  # start input-only
        self.route_scale = nn.Parameter(torch.ones(1) / (in_f ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_f))

        # Context-conditioned basis gate
        self.basis_gate = nn.Linear(ctx_dim, basis_size, bias=True)
        nn.init.zeros_(self.basis_gate.weight)
        nn.init.zeros_(self.basis_gate.bias)

        # Low-rank output gate
        r = 4
        self.out_coeffs = nn.Linear(ctx_dim, r, bias=True)
        self.out_vecs = nn.Parameter(torch.zeros(r, out_f))
        self.out_scale = nn.Parameter(torch.ones(1) * 0.1)

    @torch.no_grad()
    def init_weights(self) -> None:
        for k in range(self.n_templates):
            nn.init.kaiming_normal_(self.template_bank.data[k])
        # Initialize different templates with different scales to encourage diversity
        for k in range(self.n_templates):
            self.template_bank.data[k] *= (0.5 + k * 0.1)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        h_basis = self.ln_basis(self.basis(x))
        gate = torch.sigmoid(self.basis_gate(ctx))
        h_gated = h_basis * gate

        # Context-AWARE routing: combines input signal with iteration signal
        route_from_x = self.template_route_x(x)
        route_from_ctx = self.template_route_ctx(ctx)
        route_logits = (route_from_x + route_from_ctx) * self.route_scale
        route_weights = F.softmax(route_logits.float(), dim=-1).to(x.dtype)

        T_stack = self.template_bank.to(dtype=x.dtype)
        all_out = torch.einsum('bts,kos->btko', h_gated, T_stack)
        pre_output = (all_out * route_weights.unsqueeze(-1)).sum(dim=2) + self.bias

        out_gate = 1.0 + torch.tanh(self.out_scale * (self.out_coeffs(ctx) @ self.out_vecs))
        return pre_output * out_gate

    def template_diversity_loss(self) -> Tensor:
        """Penalize template similarity to encourage diverse specialization.

        Computes cosine similarity between all pairs of flattened template
        matrices and penalizes high similarity.
        """
        # Flatten templates: (n_templates, out_f * basis_size)
        flat = self.template_bank.view(self.n_templates, -1)
        flat_norm = F.normalize(flat.float(), dim=-1)
        # Cosine similarity matrix
        sim = flat_norm @ flat_norm.T  # (n_templates, n_templates)
        # Penalize off-diagonal similarity (exclude self-similarity)
        mask = ~torch.eye(self.n_templates, dtype=torch.bool, device=sim.device)
        return sim[mask].abs().mean()


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
        self.ffn_up = ContextAwareRemixedLinear(d, FFN_MULT * d, ctx_dim, basis, n_templates)
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


class RecurrentRemixV3(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok_emb = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(spec.max_seq_len + N_SCRATCH, D_MODEL)

        # Scratchpad: learnable tokens prepended to the sequence
        # These act as working memory / registers for intermediate computation
        self.scratch_tokens = nn.Parameter(torch.randn(N_SCRATCH, D_MODEL) * 0.02)

        # Iteration context
        self.iter_emb = nn.Embedding(MAX_LOOPS, CONTEXT_DIM)
        self.ctx_proj = nn.Linear(D_MODEL, CONTEXT_DIM, bias=True)
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)

        # Single weight-tied recurrent block
        self.block = RecurrentBlock(D_MODEL, NUM_HEADS, CONTEXT_DIM, BASIS_SIZE, N_TEMPLATES)

        # Recurrent residual blending
        self.resid_scale = nn.Parameter(torch.ones(MAX_LOOPS))
        self.x0_scale = nn.Parameter(torch.zeros(MAX_LOOPS) + 0.1)

        # Recurrent dropout
        self.recurrent_dropout = nn.Dropout(RECURRENT_DROPOUT)

        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, T = input_ids.shape

        # Embed input tokens
        tok = self.tok_emb(input_ids)

        # Prepend scratchpad tokens
        scratch = self.scratch_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([scratch, tok], dim=1)  # (B, N_SCRATCH + T, D)

        # Position embeddings for full sequence (scratch + input)
        pos = torch.arange(N_SCRATCH + T, device=input_ids.device)
        x = x + self.pos_emb(pos)

        # Extend attention mask to include scratchpad (always attended)
        if attention_mask is not None:
            scratch_mask = torch.ones(B, N_SCRATCH, device=attention_mask.device,
                                      dtype=attention_mask.dtype)
            full_mask = torch.cat([scratch_mask, attention_mask], dim=1)
        else:
            full_mask = None

        x0 = x

        for i in range(self.num_loops):
            rl = self.resid_scale[i]
            x0l = self.x0_scale[i]
            x = rl * x + x0l * x0

            # Apply recurrent dropout during training
            if self.training:
                x = self.recurrent_dropout(x)

            # Build iteration context
            ie = self.iter_emb(torch.tensor(i, device=input_ids.device))
            cs = self.ctx_proj(x.mean(dim=1, keepdim=True))
            ctx = (ie.unsqueeze(0).unsqueeze(0) + cs).expand(-1, N_SCRATCH + T, -1)

            x = self.block(x, full_mask, ctx)

        # Extract only the original token positions (skip scratchpad)
        x_out = x[:, N_SCRATCH:, :]
        logits = self.head(self.final_norm(x_out))

        # Auxiliary output: template diversity loss (used by custom training_loss)
        div_loss = self.block.ffn_up.template_diversity_loss()

        return logits, div_loss


# ── Custom Training Loss ─────────────────────────────────────────────────────

def training_loss(logits: Tensor, labels: Tensor, auxiliary: Tensor) -> Tensor:
    """Custom loss with label smoothing + template diversity regularization.

    Label smoothing (0.1) prevents overconfident memorization — keeps the
    model in a "searching" state where it maintains uncertainty, which is
    critical for the grokking transition.

    Template diversity loss encourages the 8 templates to learn different
    functions rather than collapsing to redundant copies.
    """
    ce = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)

    # auxiliary is the template diversity loss from the forward pass
    div_loss = auxiliary if torch.is_tensor(auxiliary) else torch.tensor(0.0, device=logits.device)

    return ce + TEMPLATE_DIV_WEIGHT * div_loss


# ── Submission ───────────────────────────────────────────────────────────────

def build_model(spec: ModelSpec) -> RecurrentRemixV3:
    model = RecurrentRemixV3(spec)
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
