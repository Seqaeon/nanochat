#!/usr/bin/env python3
"""Phase 35 ConditionedLinear: conditioning in activation space instead of weight space.

    additive        y = W0 x + U (c(x) * V^T x)                 c(x) in R^R
    multiplicative  z_i = z_{i-1} + c_i(x) u_i (v_i^T z_{i-1}),  y = W0 z_m

What these checks are actually defending:

  1. Identity at init, with a LIVE gradient.  Zero-initializing both factors of a
     product looks identity-preserving and is really a fixed point: if V=0 and
     c=0 then dL/dU, dL/dV and dL/dc are all zero and the branch never leaves
     zero for the whole run (this is what ResidualAdaptiveLinear does).  Here
     only U is zero and c=1, so the correction is exactly zero at step 0 and
     dL/dU is not.
  2. The compact multiplicative path equals the sequential definition.  The
     naive loop is m memory-bound passes over the activations; the compact form
     solves the coefficient recursion instead and touches them once.  They must
     agree or the fast path is a different model.
  3. Init survives GPT.init_weights.  Storage is garbage after to_empty(), and
     init_weights xavier-initializes any Linear it does not recognize, which is
     why every parameter here is a bare nn.Parameter with an explicit reset.
  4. Degrees of freedom per token are real: R independent coefficients that
     actually vary across tokens, not one broadcast number.

Usage:  python tests/test_conditioned_linear.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import ConditionedLinear

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BOUND = 3 ** 0.5 * 64 ** -0.5  # what init_weights passes as the dense uniform bound

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def make(in_f=64, out_f=96, **kw):
    layer = ConditionedLinear(in_f, out_f, **kw).to(DEV)
    layer.reset_parameters(BOUND)
    return layer


# ── 1. identity at init ──────────────────────────────────────────────────────
print("\n── identity-preserving init ─────────────────────────────────────")
for tag, kw in [("additive R=32",        dict(rank=32)),
                ("multiplicative m=8",   dict(rank=0, mult_steps=8)),
                ("both R=32 m=8",        dict(rank=32, mult_steps=8)),
                ("tied gate",            dict(rank=32, gate_source='tied')),
                ("chunked",              dict(rank=32, chunk_size=4)),
                ("factored router",      dict(rank=32, router_rank=8)),
                ("loop impl",            dict(rank=0, mult_steps=8, mult_impl='loop'))]:
    layer = make(**kw)
    x = torch.randn(2, 16, 64, device=DEV)
    y = layer(x)
    dense = F.linear(x, layer.base_w)
    err = (y - dense).abs().max().item()
    check(f"{tag}: y == W0 x at init", err == 0.0, f"max|y - W0x| = {err:.3e}")

layer = make(rank=32, zero_init_base=True)
check("zero_init_base gives an exactly zero output projection",
      layer.base_w.abs().max().item() == 0.0)
layer = make(rank=32, zero_init_base=False)
check("in-projection base matches the dense uniform init bound",
      abs(layer.base_w.abs().max().item() - BOUND) < BOUND * 0.05,
      f"max|W0| = {layer.base_w.abs().max().item():.4f} vs bound {BOUND:.4f}")


# ── 2. the gradient is live at init (the RAL failure mode) ───────────────────
print("\n── live gradient at init ────────────────────────────────────────")
for tag, kw in [("additive", dict(rank=32)), ("multiplicative", dict(rank=0, mult_steps=8))]:
    layer = make(**kw)
    x = torch.randn(4, 16, 64, device=DEV)
    layer(x).square().mean().backward()
    # The two branches put identity at init in different places, on purpose.
    #   additive:        U = 0, c = 1   -> dL/dU is live, the router is not
    #   multiplicative:  U random, c = 0 -> the router is live, dL/dU is not
    # Zeroing U for the composition instead would bound nothing and would leave
    # the router with no gradient, which is the worse end to zero.
    if kw.get('rank'):
        check(f"{tag}: dL/dU != 0 at init", layer.add_u.grad.abs().max().item() > 0,
              f"|dL/dU|max = {layer.add_u.grad.abs().max().item():.3e}")
        check(f"{tag}: dL/dV == 0 at init (expected, U=0)",
              layer.add_v.grad.abs().max().item() == 0.0)
    else:
        check(f"{tag}: dL/d(router) != 0 at init", layer.route_w.grad.abs().max().item() > 0,
              f"|dL/droute|max = {layer.route_w.grad.abs().max().item():.3e}")
        check(f"{tag}: dL/dU == 0 at init (expected, c=0)",
              layer.mul_u.grad.abs().max().item() == 0.0)

# One optimizer step must wake the whole branch up.
layer = make(rank=32)
opt = torch.optim.SGD(layer.parameters(), lr=0.1)
x = torch.randn(8, 16, 64, device=DEV)
for _ in range(2):
    opt.zero_grad()
    layer(x).square().mean().backward()
    opt.step()
check("after one step, dL/dV and dL/drouter are both live",
      layer.add_v.grad.abs().max().item() > 0 and layer.route_w.grad.abs().max().item() > 0,
      f"|dL/dV|max = {layer.add_v.grad.abs().max().item():.3e}, "
      f"|dL/dRoute|max = {layer.route_w.grad.abs().max().item():.3e}")
check("the branch has left zero", layer.add_u.abs().max().item() > 0)


# ── 3. compact multiplicative form == sequential definition ─────────────────
print("\n── compact (wy) vs sequential (loop) composition ────────────────")
for m in (1, 4, 16):
    torch.manual_seed(m)
    wy = make(rank=0, mult_steps=m, mult_impl='wy').double()
    loop = make(rank=0, mult_steps=m, mult_impl='loop').double()
    loop.load_state_dict(wy.state_dict())
    # Push both off the identity: at init U=0 makes S=0 and the two paths agree
    # trivially, so the test would prove nothing.
    with torch.no_grad():
        for lay in (wy, loop):
            lay.mul_u.normal_(std=0.3)
            lay.route_w.normal_(std=0.5)
        loop.load_state_dict(wy.state_dict())
    x = torch.randn(3, 12, 64, device=DEV, dtype=torch.float64)
    err = (wy(x) - loop(x)).abs().max().item()
    scale = loop(x).abs().max().item()
    check(f"m={m}: wy matches loop", err < 1e-10 * max(scale, 1.0),
          f"max abs diff = {err:.3e} (output scale {scale:.3e})")

# And the composition really is a product of rank-1 updates: check z_m directly.
torch.manual_seed(7)
lay = make(rank=0, mult_steps=6, mult_impl='wy').double()
with torch.no_grad():
    lay.mul_u.normal_(std=0.3)
    lay.route_w.normal_(std=0.5)
    lay.base_w.copy_(torch.eye(96, 64, device=DEV, dtype=torch.float64))  # W0 = selection
x = torch.randn(1, 5, 64, device=DEV, dtype=torch.float64)
with torch.no_grad():
    vhat = F.normalize(lay.mul_v, dim=0)
    uhat = F.normalize(lay.mul_u, dim=0)
    zc = lay._coefficients(x, None, x @ vhat, lay.rank + lay.mult_steps, x.dtype)
    c = lay._mult_coefficients(zc, x.dtype)
    z = x.clone()
    for i in range(lay.mult_steps):
        z = z + (c[..., i] * (z @ vhat[:, i])).unsqueeze(-1) * uhat[:, i]
    err = (lay(x) - F.linear(z, lay.base_w)).abs().max().item()
check("y == W0 z_m for the explicit rank-1 product", err < 1e-10, f"max abs diff = {err:.3e}")


# ── 4. degrees of freedom per token ─────────────────────────────────────────
print("\n── degrees of freedom ───────────────────────────────────────────")
layer = make(rank=64)
with torch.no_grad():
    layer.route_w.normal_(std=0.5)   # a trained router, not the zero-init one
# N must be >> R: the covariance PR is Wishart-biased low by 1/(1+R/N).
x = torch.randn(16, 128, 64, device=DEV)
p_add = x @ layer.add_v
c = layer._coefficients(x, p_add, None, 64, x.dtype)
per_coeff_std = c.reshape(-1, 64).std(dim=0)
check("all R coefficients vary independently across tokens",
      (per_coeff_std > 1e-3).all().item(),
      f"min per-coefficient std = {per_coeff_std.min().item():.4f}")


def dof_of(coeffs):
    """What the layer records: participation ratio of the covariance spectrum."""
    layer._gate_stats = {}
    layer._record_conditioning(coeffs, coeffs.shape[-1])
    return layer._gate_stats['cond_dof_pr'].item()


# A square Gaussian router has Marchenko-Pastur singular values, whose spectrum
# has participation ratio R/2, so ~32 is the correct answer here, not a
# shortfall. An orthogonal router is the isotropic case and should recover R.
check("Gaussian router: PR ~ R/2 (Marchenko-Pastur)", 0.35 * 64 < dof_of(c) < 0.8 * 64,
      f"PR = {dof_of(c):.1f} of R = 64, MP predicts {64 / 2:.0f} before the tanh spreads it")
with torch.no_grad():
    torch.nn.init.orthogonal_(layer.route_w)
    layer.route_w.mul_(0.5)   # keep tanh off its saturating tails
c_orth = layer._coefficients(x, x @ layer.add_v, None, 64, x.dtype)
check("orthogonal router recovers ~R degrees of freedom", dof_of(c_orth) > 0.85 * 64,
      f"PR = {dof_of(c_orth):.1f} of R = 64")

# The bank's ceiling, for contrast: broadcasting one coefficient across the rank
# dimension collapses the same tensor to a single degree of freedom. A marginal
# per-coefficient variance would score this a full 64, which is exactly the
# accounting error that makes an 8x-parameter template bank look expressive.
pr_bank = dof_of(c[..., :1].expand_as(c).contiguous())
check("broadcast coefficients give PR = 1 (the RemixedLinear ceiling)",
      abs(pr_bank - 1.0) < 0.05, f"PR = {pr_bank:.3f}")
marginal = c[..., :1].expand_as(c).reshape(-1, 64).var(dim=0)
print(f"      (marginal-variance PR of the same broadcast tensor: "
      f"{(marginal.sum() ** 2 / (marginal ** 2).sum()).item():.1f}, the metric that hides it)")

# A rank-4 coefficient field should score ~4 regardless of R.
mix = torch.randn(16, 128, 4, device=DEV) @ torch.randn(4, 64, device=DEV)
check("a rank-4 coefficient field scores PR ~ 4", abs(dof_of(mix) - 4.0) < 1.0,
      f"PR = {dof_of(mix):.2f}")

# The Wishart bias is a property of the estimator, not of the layer. Pin it so a
# future reader does not mistake a few percent shortfall for dead capacity.
iso = torch.randn(1, 192, 64, device=DEV)          # N = 3R, isotropic by construction
check("finite-sample bias follows R/(1+R/N)", abs(dof_of(iso) - 64 / (1 + 64 / 192)) < 4.0,
      f"PR = {dof_of(iso):.1f}, predicted {64 / (1 + 64 / 192):.1f} at N = 3R")

# Diagnostics buffer that base_train's collect_gate_stats reads.
layer.train()
layer(x)
stats = layer._gate_stats
check("_gate_stats exposes cond_dof_pr / cond_tok_std",
      {'cond_c_mean', 'cond_c_std', 'cond_tok_std', 'cond_dof_pr'} <= set(stats),
      f"dof_pr = {stats.get('cond_dof_pr', float('nan')):.1f}, "
      f"tok_std = {stats.get('cond_tok_std', float('nan')):.4f}")


# ── 5. chunked routing is causal and chunk-constant ─────────────────────────
print("\n── chunk routing ───────────────────────────────────────────────")
layer = make(rank=16, chunk_size=4)
with torch.no_grad():
    layer.route_w.normal_(std=0.5)
x = torch.randn(2, 12, 64, device=DEV)
c = layer._coefficients(x, x @ layer.add_v, None, 16, x.dtype)
chunks = c.reshape(2, 3, 4, 16)
check("coefficients are constant within a chunk",
      (chunks - chunks[:, :, :1, :]).abs().max().item() == 0.0)
# Perturbing a non-anchor token must not change any coefficient: the routing
# signal is the chunk's first token only, so nothing later can leak backwards.
x2 = x.clone()
x2[:, 6, :] += 10.0
c2 = layer._coefficients(x2, x2 @ layer.add_v, None, 16, x.dtype)
check("a non-anchor token cannot change the routing",
      (c - c2).abs().max().item() == 0.0)
x3 = x.clone()
x3[:, 4, :] += 10.0   # anchor of chunk 1
c3 = layer._coefficients(x3, x3 @ layer.add_v, None, 16, x.dtype)
check("chunk 0 is unaffected when a later anchor moves",
      (c[:, :4] - c3[:, :4]).abs().max().item() == 0.0 and (c[:, 4:] - c3[:, 4:]).abs().max().item() > 0)


# ── 6. parameter accounting ─────────────────────────────────────────────────
print("\n── parameter cost ──────────────────────────────────────────────")
d = 768
dense = d * d
for tag, kw in [("additive R=256",          dict(rank=256)),
                ("additive R=256 + rr=64",  dict(rank=256, router_rank=64)),
                ("additive R=256 tied",     dict(rank=256, gate_source='tied')),
                ("multiplicative m=16",     dict(rank=0, mult_steps=16)),
                ("both R=256 m=16",         dict(rank=256, mult_steps=16))]:
    lay = ConditionedLinear(d, d, **kw)
    n = sum(p.numel() for p in lay.parameters())
    dof = lay.rank + lay.mult_steps
    print(f"    {tag:26s}  {n / dense:5.2f}x dense   DOF/token = {dof:4d}")
lay = ConditionedLinear(d, d, rank=256)
expected = dense + 256 * (d + d) + d * 256
check("additive parameter count matches d² + R(d_in+d_out) + d_sig·R",
      sum(p.numel() for p in lay.parameters()) == expected)


# ── 7. torch.compile ────────────────────────────────────────────────────────
print("\n── torch.compile ───────────────────────────────────────────────")
try:
    layer = make(rank=32, mult_steps=8)
    x = torch.randn(2, 16, 64, device=DEV)
    eager = layer(x)
    compiled = torch.compile(layer)(x)
    check("compiled output matches eager",
          torch.allclose(eager, compiled, atol=1e-5), f"max diff = {(eager - compiled).abs().max().item():.3e}")
except Exception as e:  # inductor is unavailable in some environments
    print(f"SKIP  torch.compile: {type(e).__name__}: {e}")


# ── 8. dtype handling ───────────────────────────────────────────────────────
print("\n── dtype ───────────────────────────────────────────────────────")
layer = make(rank=32, mult_steps=8)
with torch.no_grad():
    layer.mul_u.normal_(std=0.1)
    layer.add_u.normal_(std=0.1)
for dt in (torch.float32, torch.bfloat16, torch.float16):
    y = layer(torch.randn(2, 16, 64, device=DEV, dtype=dt))
    check(f"{str(dt).split('.')[-1]} in -> {str(y.dtype).split('.')[-1]} out, finite",
          y.dtype == dt and torch.isfinite(y).all().item())


# ── 9. whole-model integration ──────────────────────────────────────────────
print("\n── model integration (--cclblock-modulation cond) ───────────────")
from nanochat.gpt import GPT, GPTConfig

BASE = dict(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2,
            n_embd=64, remix_context_dim=32, remix_basis_size=64)


def build(**kw):
    cfg = GPTConfig(**BASE, **kw)
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device=DEV)
    model.init_weights()
    return model


try:
    model = build(use_remix_linear=True, cclblock_modulation='cond', cond_rank=16, cond_mult_steps=4)
    layers = [m for m in model.modules() if isinstance(m, ConditionedLinear)]
    check("cond replaces every attention and FFN projection", len(layers) == 2 * 6,
          f"{len(layers)} ConditionedLinear layers over 2 blocks")
    check("init_weights left the additive U at exactly zero",
          all(l.add_u.abs().max().item() == 0 for l in layers),
          "the xavier fallback in init_weights did not reach them")
    check("the composition generators are NOT zero (identity comes from c=0 there)",
          all(l.mul_u.abs().max().item() > 0 for l in layers))
    check("init_weights left every router at exactly zero",
          all(l.route_w.abs().max().item() == 0 for l in layers))
    check("init_weights left no uninitialized storage",
          all(torch.isfinite(p).all().item() for p in model.parameters()))
    check("no context stream is built for cond",
          all(getattr(b, 'ctx_stream', None) is None for b in model.transformer.h),
          "a stream would only add parameters, and parameters set the token budget")

    ids = torch.randint(0, 512, (2, 64), device=DEV)
    loss = model(ids, ids)
    loss.backward()
    check("forward/backward produces a finite loss", torch.isfinite(loss).item(), f"loss = {loss.item():.4f}")
    check("every ConditionedLinear parameter receives gradient",
          all(p.grad is not None and torch.isfinite(p.grad).all().item()
              for l in layers for p in l.parameters()))
    # The conditioning path wakes up on a fixed schedule, and it is worth knowing
    # the schedule rather than discovering it as a mystery flat loss:
    #   step 0  routers have zero gradient (U = 0), and the in-projections have
    #           zero gradient too, since the dense baseline zero-inits attn.c_proj and
    #           mlp.c_proj, so nothing reaches q/k/v/c_fc at all on the first step
    #   step 1  U moves on every layer
    #   step 2  every router is live
    # Three steps, not two, and only because of the repo's dense init.
    opt = model.setup_optimizer(unembedding_lr=1e-3, embedding_lr=1e-3, matrix_lr=1e-2)
    moved = []
    for _ in range(3):
        opt.zero_grad()
        model(ids, ids).backward()
        opt.step()
        moved.append(sum(l.route_w.abs().max().item() > 0 for l in layers))
    check("every router is learning by step 3", moved[-1] == len(layers),
          f"routers live after each step: {moved} of {len(layers)}")
    check("output projections lead the in-projections by one step", moved[0] < moved[-1],
          f"{moved[0]} live after step 1 (the c_proj layers), {moved[-1]} after step 3")

    gate_ids = {id(p) for l in layers for p in l.gate_parameters()}
    struct_ids = {id(p) for l in layers for p in l.non_gate_parameters()}
    check("gate/structural split covers every parameter exactly once",
          gate_ids.isdisjoint(struct_ids) and
          len(gate_ids | struct_ids) == sum(len(list(l.parameters())) for l in layers))

    dense = build()
    n_cond = sum(p.numel() for p in model.transformer.h.parameters())
    n_dense = sum(p.numel() for p in dense.transformer.h.parameters())
    print(f"      transformer params: cond {n_cond:,} vs dense {n_dense:,} "
          f"= {n_cond / n_dense:.2f}x  (this ratio sets the token budget at a fixed data:param ratio)")
    check("parameter inflation stays well under the K=8 bank's 8x", n_cond / n_dense < 3.0)
except Exception as e:
    import traceback
    traceback.print_exc()
    check("model integration", False, f"{type(e).__name__}: {e}")


# ── 10. the composition is bounded for any m ────────────────────────────────
print("\n── composition stability ───────────────────────────────────────")
worst = 0.0
for m in (4, 16, 64):
    for scale in (1.0, 64.0, 4096.0):
        lay = make(in_f=128, out_f=128, rank=0, mult_steps=m)
        with torch.no_grad():
            lay.mul_u.normal_(std=scale / 128 ** 0.5)
            lay.route_w.normal_(std=5.0)          # saturate the router
        y = lay(torch.randn(4, 32, 128, device=DEV))
        worst = max(worst, y.abs().max().item())
check("output stays bounded across m and generator magnitude", worst < 50,
      f"worst max|y| = {worst:.2f} over m in 4/16/64 and ||u|| spanning 4096x")
lay = make(in_f=128, out_f=128, rank=0, mult_steps=16)
check("|c| is bounded by 1/m by construction",
      abs(lay.mult_scale.item() - 1.0 / 16) < 1e-9,
      f"mult_scale = {lay.mult_scale.item():.6f} = 1/16")

print("\n" + "=" * 64)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
