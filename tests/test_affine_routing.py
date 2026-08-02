#!/usr/bin/env python3
"""Phase 36: affine-hull routing (--p22-route-affine 1).

    softmax   alpha = softmax(z)              reachable ops = convex hull of the bank
    affine    alpha = 1/K + s*(z - mean z)    reachable ops = affine hull of the bank

Both sum to 1, so the operator scale story is unchanged. What changes is that
alpha may leave [0,1], and that is what removes the two mechanisms behind the
collapse measured on the trained d12 checkpoint (59/72 modules at K_eff < 1.05,
usage CV = sqrt(K-1) exactly, mean K_eff 1.52):

  1. Gradient starvation.  dL/dz_k = alpha_k*(g_k - sum_j alpha_j g_j) under a
     softmax. The alpha_k prefactor makes a template's gradient proportional to
     how much it is already used, so a losing template cannot recover. Affine
     has no prefactor: dL/dz_k is the full projection onto template k.
  2. Norm/concentration coupling.  A uniform mixture of templates with mean
     pairwise cosine c has relative Frobenius norm sqrt(1/K + (K-1)c/K), which
     at the measured cosines is 0.40 to 0.55. On a simplex the only way to grow
     the operator is to concentrate. With unconstrained coefficients the scale
     can grow without collapsing.

Usage:  python tests/test_affine_routing.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig, RemixedLinear

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 8
FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def build(affine, **over):
    kw = dict(use_basis_gate=False, use_output_gate=True, use_context=True,
              basis_gate_mode='centered', n_templates=K, template_topk=0,
              chunk_routing_size=32, template_routing_learned=True,
              use_quantile_route=0, route_affine=affine)
    cfg = GPTConfig(sequence_len=128, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=64, remix_context_dim=32, remix_basis_size=64,
                    use_remix_linear=True, cclblock_modulation='weight',
                    p28_chunk_routing_size=32, p22_route_affine=affine,
                    remixed_linear_kwargs=kw, **over)
    with torch.device('meta'):
        m = GPT(cfg)
    m.to_empty(device=DEV)
    m.init_weights(verify=False)
    return m


def alphas(m, seed=0):
    torch.manual_seed(seed)
    rl = m.transformer.h[0].attn.c_q
    x = torch.randn(4, 128, 64, device=DEV)
    return rl._template_weights(x.float(), torch.float32).detach()


# ── 1. an affine combination is still a combination ─────────────────────────
print("\n── affine combination ──────────────────────────────────────────")
soft, aff = build(0), build(1)
w_s, w_a = alphas(soft), alphas(aff)
check("affine coefficients sum to exactly 1",
      (w_a.sum(-1) - 1.0).abs().max().item() < 1e-5,
      f"max |sum - 1| = {(w_a.sum(-1) - 1.0).abs().max().item():.2e}")


def keff(w):
    p = w.abs() / w.abs().sum(-1, keepdim=True)
    return math.exp(-(p * p.clamp(min=1e-9).log()).sum(-1).mean().item())


check("no dead start: initial spread comparable to softmax",
      abs(keff(w_a) - keff(w_s)) < 1.5,
      f"K_eff affine {keff(w_a):.2f} vs softmax {keff(w_s):.2f} of K={K}")

# Negative coefficients are the added capacity: subtracting a template becomes
# reachable once the learned scale grows past 1/K over the coefficient spread.
rl = aff.transformer.h[0].attn.c_q
with torch.no_grad():
    rl.route_affine_scale.fill_(1.0)
w_big = alphas(aff)
check("negative coefficients become reachable as the scale grows",
      (w_big < 0).any().item() and abs(w_big.sum(-1).mean().item() - 1.0) < 1e-4,
      f"{100 * (w_big < 0).float().mean().item():.0f}% negative, still sums to "
      f"{w_big.sum(-1).mean().item():.4f}")
with torch.no_grad():
    rl.route_affine_scale.fill_(1.0 / K)


# ── 2. gradient starvation, the thing this exists to remove ─────────────────
print("\n── gradient starvation ─────────────────────────────────────────")


def starvation(affine, dominance=4.0):
    """Ratio of the gradient a heavily-used coefficient gets to a barely-used one."""
    torch.manual_seed(1)
    z0 = torch.randn(4096, K, device=DEV)
    z0[:, 0] += dominance                       # template 0 already winning
    z = z0.clone().requires_grad_(True)
    a = torch.softmax(z, -1) if not affine else (1.0 / K + (z - z.mean(-1, keepdim=True)) / K)
    (a * torch.randn_like(a)).sum().backward()
    av, g = a.detach().flatten(), z.grad.abs().flatten()
    lo = g[av < av.quantile(0.10)].mean().item()
    hi = g[av > av.quantile(0.90)].mean().item()
    return hi / max(lo, 1e-12)


r_soft, r_aff = starvation(False), starvation(True)
check("softmax starves unused templates", r_soft > 10,
      f"most-used gets {r_soft:.0f}x the gradient of least-used")
check("affine does not", r_aff < 1.5,
      f"ratio {r_aff:.2f}x, i.e. every coefficient gets its full projection")
# The starvation worsens as one template pulls further ahead. That is the
# rich-get-richer loop; it should be flat for affine at any dominance.
prog_s = [round(starvation(False, d), 1) for d in (1.0, 3.0, 6.0)]
prog_a = [round(starvation(True, d), 2) for d in (1.0, 3.0, 6.0)]
check("softmax starvation compounds with dominance", prog_s[-1] > prog_s[0] * 3,
      f"softmax {prog_s} vs affine {prog_a} at dominance 1/3/6")


# ── 3. norm can grow without concentrating ──────────────────────────────────
print("\n── norm / concentration coupling ───────────────────────────────")
torch.manual_seed(2)
T = torch.randn(K, 64, 64, device=DEV)
T = T / T.flatten(1).norm(dim=1).view(K, 1, 1)         # unit, near-orthogonal
single = 1.0
uniform = torch.einsum('k,kij->ij', torch.full((K,), 1.0 / K, device=DEV), T).norm().item()
onehot = torch.einsum('k,kij->ij', torch.eye(K, device=DEV)[0], T).norm().item()
print(f"      uniform mixture norm  = {uniform:.3f} of a single template")
print(f"      one-hot mixture norm  = {onehot:.3f}      predicted 1/sqrt(K) = {K ** -0.5:.3f}")
check("uniform mixing really does shrink the operator by ~1/sqrt(K)",
      abs(uniform - K ** -0.5) < 0.05,
      f"{uniform:.3f} vs {K ** -0.5:.3f}, so the simplex pays {onehot / uniform:.1f}x to concentrate")
# Affine can reach full norm without any concentration: scale a zero-mean
# direction, which the simplex cannot represent at all.
a_aff = 1.0 / K + 3.0 * (torch.randn(K, device=DEV) - 0.0)
a_aff = a_aff - a_aff.mean() + 1.0 / K
grown = torch.einsum('k,kij->ij', a_aff, T).norm().item()
check("affine reaches full operator norm without concentrating",
      grown > onehot and keff(a_aff.view(1, 1, K)) > 0.5 * K,
      f"norm {grown:.2f} at K_eff {keff(a_aff.view(1, 1, K)):.1f} of {K}")


# ── 4. it trains ────────────────────────────────────────────────────────────
print("\n── end to end ──────────────────────────────────────────────────")
m = build(1)
ids = torch.randint(0, 512, (2, 128), device=DEV)
opt = m.setup_optimizer(unembedding_lr=1e-3, embedding_lr=1e-3, matrix_lr=1e-2)
losses = []
for _ in range(3):
    opt.zero_grad()
    loss = m(ids, ids)
    loss.backward()
    opt.step()
    losses.append(loss.item())
check("forward/backward/step all finite", all(math.isfinite(l) for l in losses),
      f"losses {[round(l, 4) for l in losses]}")
rl = m.transformer.h[0].attn.c_q
check("route_affine_scale is a learned gate parameter",
      any(p is rl.route_affine_scale for p in rl.gate_parameters()))
check("the router receives gradient", rl.template_route.grad is not None
      and rl.template_route.grad.abs().max().item() > 0,
      f"|dL/droute|max = {rl.template_route.grad.abs().max().item():.3e}")
check("softmax path is unchanged when the flag is off",
      not build(0).transformer.h[0].attn.c_q.route_affine)

print("\n" + "=" * 64)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
