#!/usr/bin/env python3
"""Nonlinear router for ConditionedLinear (--cond-router-act gelu|relu2).

Every router in the p35 sweep was linear in x, and conditioning_headroom.py
--nonlinear-reach on the dense d22 checkpoint says that was the binding
constraint rather than the operator's capacity:

    projection      lin_r    mlp_r     gain      (held out, r=64, control -0.024)
    attn.c_proj     0.263    0.413   +0.150
    attn.c_k        0.339    0.480   +0.141
    attn.c_q        0.328    0.462   +0.135
    attn.c_v        0.459    0.579   +0.120
    mlp.c_fc        0.244    0.298   +0.054
    resolved        0.338    0.445   +0.107

lin_r and mlp_r are parameter-matched and FLOP-matched, so the whole gain is the
nonlinearity. A rank-64 nonlinear router reaches 0.445 against the FULL-rank
(1408) linear router's 0.512, at 1/22 of its cost.

This file checks the implementation is that thing and not something else: same
FLOPs, genuinely nonlinear, identity at init, and it trains.

Usage:  python tests/test_nonlinear_router.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RR = 32
FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def build(act, rr=RR, **over):
    cfg = GPTConfig(sequence_len=128, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2,
                    n_embd=64, use_remix_linear=True, cclblock_modulation='cond',
                    cond_rank=16, cond_router_rank=rr, cond_router_act=act,
                    cond_gate_source='router', **over)
    with torch.device('meta'):
        m = GPT(cfg)
    m.to_empty(device=DEV)
    m.init_weights(verify=False)
    return m


def cl(m):
    return m.transformer.h[0].attn.c_q


lin, nl = build('none'), build('gelu')
a, b = cl(lin), cl(nl)

# ── 1. matched cost, which is the whole claim ───────────────────────────────
print("\n── cost parity ─────────────────────────────────────────────────")
na = sum(p.numel() for p in a.parameters())
nb = sum(p.numel() for p in b.parameters())
check("parameters differ by exactly the hidden bias", nb - na == RR,
      f"{na} vs {nb}, diff {nb - na} = rr {RR}")
check("the two matmuls are the same shapes, so the FLOPs are identical",
      a.route_down.shape == b.route_down.shape and a.route_w.shape == b.route_w.shape,
      f"down {tuple(b.route_down.shape)}, up {tuple(b.route_w.shape)}")
check("route_b exists only with the nonlinearity",
      a.route_b is None and b.route_b is not None)

# ── 2. it is actually nonlinear ─────────────────────────────────────────────
print("\n── nonlinearity ────────────────────────────────────────────────")
torch.manual_seed(0)
x = torch.randn(64, 128, 64, device=DEV)


def logits(m, xx):
    # _coefficients returns the router LOGITS; _activate turns them into c.
    c = cl(m)
    return c._coefficients(xx.float(), None, None, c.rank, torch.float32)


# Additivity is not a usable probe here: the router's first op is an rmsnorm, so
# norm(2x) == norm(x) and every arm looks scale-invariant. Ask the question
# directly instead: is the map from the post-norm signal to the logits reachable
# by ANY linear map? Fit the best one by least squares and look at the residual.
from nanochat.gpt import norm

for m in (lin, nl):
    with torch.no_grad():
        cl(m).route_w.normal_(std=0.3)

H = norm(x.float()).reshape(-1, 64).double()
for m, name, want_linear in ((lin, 'none', True), (nl, 'gelu', False)):
    with torch.no_grad():
        Z = logits(m, x).reshape(-1, cl(m).rank).double()
    B = torch.linalg.lstsq(H, Z).solution
    resid = ((Z - H @ B).pow(2).sum() / Z.pow(2).sum()).item()
    check(f"router act='{name}' is {'linear' if want_linear else 'NONlinear'} in its signal",
          (resid < 1e-6) == want_linear,
          f"unexplained by the best linear map: {resid:.1%}")

with torch.no_grad():
    cl(nl).route_w.zero_()
check("identity at init: zero route_w gives zero logits through the nonlinearity",
      logits(nl, x).abs().max().item() < 1e-6,
      f"max |z| = {logits(nl, x).abs().max().item():.2e}")
check("route_b is initialized, not left as to_empty() garbage",
      torch.isfinite(b.route_b).all().item() and b.route_b.std().item() > 0.1,
      f"std {b.route_b.std().item():.3f}")
check("route_b is a gate parameter", any(p is b.route_b for p in b.gate_parameters()))

# ── 3. a nonlinearity with no hidden layer is a configuration error ─────────
print("\n── guards ──────────────────────────────────────────────────────")
try:
    build('gelu', rr=0)
    check("full-rank router + act is rejected", False, "no error raised")
except ValueError as e:
    check("full-rank router + act is rejected", 'cond_router_rank' in str(e),
          str(e).split('.')[0][:60])
try:
    build('swish')
    check("unknown act is rejected", False, "no error raised")
except ValueError:
    check("unknown act is rejected", True)

# ── 4. relu2 works too, and both train ──────────────────────────────────────
print("\n── end to end ──────────────────────────────────────────────────")
# The router's hidden layer sits behind TWO zero-inits: add_u is zero (so the
# branch starts as an exact identity) and route_w is zero (so c starts at 1).
# Gradient therefore unlocks in a chain, add_u first, then route_w, then
# route_down and route_b. That is a delay, not a trap like the output_gate bug,
# but it is only a delay if the chain actually completes, so check it does rather
# than assume. The linear router has the same chain one link shorter.
for act in ('gelu', 'relu2'):
    m = build(act)
    ids = torch.randint(0, 512, (2, 128), device=DEV)
    opt = m.setup_optimizer(unembedding_lr=1e-3, embedding_lr=1e-3, matrix_lr=1e-2)
    r, losses, first_grad = cl(m), [], {}
    for step in range(12):
        opt.zero_grad()
        loss = m(ids, ids)
        loss.backward()
        for nm, p in (('add_u', r.add_u), ('route_w', r.route_w),
                      ('route_down', r.route_down), ('route_b', r.route_b)):
            if nm not in first_grad and p.grad is not None and p.grad.abs().max().item() > 0:
                first_grad[nm] = step
        opt.step()
        losses.append(loss.item())
    check(f"act='{act}': forward/backward/step all finite",
          all(math.isfinite(l) for l in losses), f"{[round(l, 4) for l in losses[:4]]} ...")
    check(f"act='{act}': every router parameter unlocks within 12 steps",
          set(first_grad) == {'add_u', 'route_w', 'route_down', 'route_b'},
          f"first nonzero grad at step {first_grad}")
    check(f"act='{act}': the hidden bias ends with real gradient",
          r.route_b.grad is not None and r.route_b.grad.abs().max().item() > 0,
          f"|dL/db|max = {r.route_b.grad.abs().max().item():.3e}")

check("act='none' path is untouched", cl(build('none')).route_b is None)

print("\n" + "=" * 64)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
