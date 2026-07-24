"""Recurrent MST-style Multi-Sub-Transformer for One Layer Deeper.

Adapted from the MST architecture (nanochat/mst.py). N parallel sub-networks
process the input simultaneously, with aggregate-distribute cross-sub
communication at each recurrent step.

Design choices for this task:
  - 4 subs × 64 dim = 256 total dim, same parameter budget as Remix
  - Fewer loops (4) for throughput — each loop has N sub-attentions
  - Larger batch size for GPU utilization
  - Sub-specialization through distributed init + transition routing
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

N_SUBS = 4
SUB_DIM = 64
NUM_HEADS = 4
NUM_LOOPS = 4
FFN_MULT = 4
MAX_LOOPS = 32
D_EMBED = N_SUBS * SUB_DIM  # 256


# ── Utility ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, w: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(w))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# ── Batched Sub-Transformer Layer ────────────────────────────────────────────

class BatchedSubLayer(nn.Module):
    """N parallel sub-transformers via stacked weight tensors."""

    def __init__(self, N: int, d: int, nh: int) -> None:
        super().__init__()
        self.N, self.d, self.nh = N, d, nh
        self.hd = d // nh
        inner = FFN_MULT * d

        # Stacked weights: (N*out, in)
        self.c_q_w = nn.Parameter(torch.empty(N * d, d))
        self.c_k_w = nn.Parameter(torch.empty(N * d, d))
        self.c_v_w = nn.Parameter(torch.empty(N * d, d))
        self.c_proj_w = nn.Parameter(torch.empty(N * d, d))
        self.fc_w = nn.Parameter(torch.empty(N * inner, d))
        self.fc_proj_w = nn.Parameter(torch.empty(N * d, inner))

        self.norm_a = RMSNorm(d)
        self.norm_f = RMSNorm(d)

    def forward(self, sub_states: Tensor, mask: Tensor | None) -> Tensor:
        B, T, N, d = sub_states.shape
        nh, hd = self.nh, self.hd

        # ── Attention
        x = self.norm_a(sub_states)
        q = torch.einsum('btnd,nod->btno', x, self.c_q_w.view(N, -1, d))
        k = torch.einsum('btnd,nod->btno', x, self.c_k_w.view(N, -1, d))
        v = torch.einsum('btnd,nod->btno', x, self.c_v_w.view(N, -1, d))

        # (B,T,N,d) → (B*N, nh, T, hd)
        q = q.permute(0, 2, 1, 3).reshape(B * N, T, nh, hd).transpose(1, 2)
        k = k.permute(0, 2, 1, 3).reshape(B * N, T, nh, hd).transpose(1, 2)
        v = v.permute(0, 2, 1, 3).reshape(B * N, T, nh, hd).transpose(1, 2)

        m = None
        if mask is not None:
            m = mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, T)[:, None, None, :]
            m = m.to(dtype=torch.bool, device=sub_states.device)

        ao = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        ao = ao.transpose(1, 2).reshape(B * N, T, -1).view(B, N, T, -1).permute(0, 2, 1, 3)
        ao = torch.einsum('btno,ndo->btnd', ao, self.c_proj_w.view(N, d, -1))
        sub_states = sub_states + ao

        # ── FFN
        x = self.norm_f(sub_states)
        h = torch.einsum('btnd,nod->btno', x, self.fc_w.view(N, -1, d))
        h = F.relu(h).square()
        h = torch.einsum('btno,ndo->btnd', h, self.fc_proj_w.view(N, d, -1))
        return sub_states + h

    @torch.no_grad()
    def init_weights(self) -> None:
        s = 1.0 / (self.d ** 0.5)
        N, d = self.N, self.d
        inner = FFN_MULT * d
        for w in (self.c_q_w, self.c_k_w, self.c_v_w):
            for j in range(N):
                nn.init.uniform_(w.data[j * d : (j + 1) * d], -s, s)
        nn.init.zeros_(self.c_proj_w)
        for j in range(N):
            nn.init.uniform_(self.fc_w.data[j * inner : (j + 1) * inner], -s, s)
        nn.init.zeros_(self.fc_proj_w)


# ── Aggregate-Distribute Transition ─────────────────────────────────────────

class AggDistTransition(nn.Module):
    def __init__(self, N: int, d: int) -> None:
        super().__init__()
        self.N, self.d = N, d
        self.router_w = nn.Parameter(torch.zeros(N, d))
        self.distribute_w = nn.Parameter(torch.empty(N * d, d))

    def forward(self, sub_states: Tensor) -> Tensor:
        B, T, N, d = sub_states.shape
        ri = sub_states.mean(dim=2)  # (B, T, d)
        w = F.softmax(F.linear(ri, self.router_w), dim=-1)  # (B, T, N)
        agg = (w.unsqueeze(-1) * sub_states).sum(dim=2)  # (B, T, d)
        dist = torch.einsum('btd,nod->btno', agg, self.distribute_w.view(N, d, d))
        return sub_states + dist

    @torch.no_grad()
    def init_weights(self) -> None:
        s = 1.0 / (self.d ** 0.5)
        nn.init.zeros_(self.router_w)
        dist = self.distribute_w.view(self.N, self.d, self.d)
        for j in range(self.N):
            nn.init.uniform_(dist[j], -s, s)


# ── Model ────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RecurrentMSTModel(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        N, d = N_SUBS, SUB_DIM

        self.tok_emb = nn.Embedding(spec.vocab_size, D_EMBED)
        self.pos_emb = nn.Embedding(spec.max_seq_len, D_EMBED)

        # Fixed slice input: D_EMBED = N*d, just reshape
        self.sub_layer = BatchedSubLayer(N, d, NUM_HEADS)
        self.transition = AggDistTransition(N, d)

        # Iteration embedding added to each sub
        self.iter_emb = nn.Embedding(MAX_LOOPS, d)

        # x0 residual blending
        self.resid_scale = nn.Parameter(torch.ones(MAX_LOOPS))
        self.x0_scale = nn.Parameter(torch.zeros(MAX_LOOPS) + 0.1)

        # Output
        self.final_norm = RMSNorm(d)
        self.final_router_w = nn.Parameter(torch.zeros(N, d))
        self.head = nn.Linear(d, spec.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B, T, D_EMBED)

        # Distribute: fixed slice → (B, T, N, d)
        N, d = N_SUBS, SUB_DIM
        sub_states = x.view(B, T, N, d)
        sub_x0 = sub_states.clone()

        for i in range(self.num_loops):
            # x0 residual blending
            rl = self.resid_scale[i]
            x0l = self.x0_scale[i]
            sub_states = rl * sub_states + x0l * sub_x0

            # Iteration signal
            ie = self.iter_emb(torch.tensor(i, device=input_ids.device))
            sub_states = sub_states + ie * 0.1

            sub_states = self.sub_layer(sub_states, attention_mask)
            sub_states = self.transition(sub_states)

        # Aggregate subs → predict
        ri = sub_states.mean(dim=2)
        w = F.softmax(F.linear(ri, self.final_router_w), dim=-1)
        agg = (w.unsqueeze(-1) * sub_states).sum(dim=2)
        return self.head(self.final_norm(agg)), None


# ── Submission ───────────────────────────────────────────────────────────────

def build_model(spec: ModelSpec) -> RecurrentMSTModel:
    model = RecurrentMSTModel(spec)
    with torch.no_grad():
        model.sub_layer.init_weights()
        model.transition.init_weights()
        nn.init.normal_(model.tok_emb.weight, std=1.0)
        nn.init.normal_(model.head.weight, std=0.001)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        capturable=spec.device_type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100000, eta_min=1e-5
    )
    return OptimizerBundle(optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
)
