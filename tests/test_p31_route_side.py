#!/usr/bin/env python3
"""Phase 31 ablations: route_side ('narrow'/'basis') and drop_basis_proj.

Both change WHICH matrix carries the routing, without reducing any rank:

  route_side='output' (legacy)  W_b (basis,in) shared  ->  bank (K,out,basis)
  route_side='basis'            bank (K,basis,in)      ->  W_m (out,basis) shared
  drop_basis_proj               (no W_b)               ->  bank (K,out,in)

The per-chunk materialized matrix is always the *routed* factor, so for an
expanding projection (c_fc: in < out) routing the basis side shrinks it by
out/basis, while drop_basis_proj enlarges it for a contracting projection
(c_proj: in > out) even though it removes a matmul.

Checks correctness/shape/grad-flow, then reports the cost model and measured
time so the FLOPs-vs-bandwidth tradeoff is visible per projection.

Usage:  python tests/test_p31_route_side.py [--depth 8]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import RemixedLinear

BASE_KW = dict(
    n_templates=8,
    chunk_routing_size=64,
    template_routing_learned=True,
    use_basis_gate=False,
    use_output_gate=True,
    use_context=True,
    output_gate_rank=8,
    basis_gate_mode='linear',
    use_quantile_route=0,
    template_topk=0,
)

MODES = [
    ('output (legacy)', dict()),
    ('narrow',          dict(route_side='narrow')),
    ('basis',           dict(route_side='basis')),
    ('drop_W_b',        dict(drop_basis_proj=True)),
]


def make(in_f, out_f, basis, extra, device, dtype, chunk=64, topk=0):
    kw = dict(BASE_KW, chunk_routing_size=chunk, template_topk=topk, **extra)
    torch.manual_seed(0)
    return RemixedLinear(in_f, out_f, 128, basis_size=basis,
                         remixed_linear_kwargs=kw, scale_basis=False).to(device, dtype)


# ── correctness ──────────────────────────────────────────────────────────────

def test_resolution():
    """'narrow' must resolve to 'basis' iff in < out, and never for the modes
    that have no W_b/W_m pair to swap."""
    print("\n  ── route_side resolution ──")
    # want_basis = "a separate shared W_b still exists".  route_side='basis' means
    # the bank *is* the input map, so W_b must be gone in that case.  drop_basis_proj
    # also removes W_b, but only where in_features <= basis_size.
    cases = [
        ("c_fc   512->2048", 512, 2048, 512, dict(route_side='narrow'), 'basis', False),
        ("c_proj 2048->512", 2048, 512, 512, dict(route_side='narrow'), 'output', True),
        ("attn   512->512",  512, 512, 512, dict(route_side='narrow'), 'output', True),
        ("explicit basis",   2048, 512, 512, dict(route_side='basis'), 'basis', False),
        # drop_basis_proj: applies at in<=basis, gated out at in>basis
        ("drop c_fc  (in=b)", 512, 2048, 512, dict(drop_basis_proj=True), 'output', False),
        ("drop attn  (in=b)", 512, 512, 512, dict(drop_basis_proj=True), 'output', False),
        ("drop c_proj GATED", 2048, 512, 512, dict(drop_basis_proj=True), 'output', True),
        # a gated-out drop must fall through to route_side rather than being stuck
        ("drop+narrow c_fc",  512, 2048, 512, dict(drop_basis_proj=True, route_side='narrow'), 'output', False),
        ("drop+basis c_proj", 2048, 512, 512, dict(drop_basis_proj=True, route_side='basis'), 'basis', False),
        ("delta wins",       512, 2048, 512, dict(route_side='narrow', template_delta_rank=8), 'output', True),
        ("K=1 no bank",      512, 2048, 512, dict(route_side='narrow', n_templates=1), 'output', True),
    ]
    ok_all = True
    for name, i, o, b, extra, want_side, want_basis in cases:
        m = make(i, o, b, extra, 'cpu', torch.float32)
        got_side, got_basis = m.route_side, m.basis is not None
        # drop_basis_proj must self-report whether it actually took effect
        want_drop = extra.get('drop_basis_proj', False) and i <= b and want_side == 'output'
        ok = (got_side == want_side and got_basis == want_basis
              and m.drop_basis_proj == want_drop)
        # when the drop fires, the bank contracts over in_features
        if m.drop_basis_proj:
            ok &= (m.basis_size == i and m.template_bank.shape[2] == i)
        ok_all &= ok
        print(f"    {name:20s} side={got_side:7s} W_b={'yes' if got_basis else 'no ':3s} "
              f"drop={str(m.drop_basis_proj):5s} (want {want_side}/"
              f"{'yes' if want_basis else 'no'}/{want_drop})  {'✓' if ok else '✗'}")
    return ok_all


def test_shapes_and_grads(device, dtype):
    print(f"\n  ── shapes + grad flow ({device}, {dtype}) ──")
    ok_all = True
    for label, i, o, b in [("c_fc", 512, 2048, 512), ("c_proj", 2048, 512, 512),
                           ("attn", 512, 512, 512)]:
        for mname, extra in MODES:
            for chunk, topk in [(64, 0), (64, 1), (0, 0)]:
                m = make(i, o, b, extra, device, dtype, chunk=chunk, topk=topk)
                m.train()
                torch.manual_seed(7)
                x = torch.randn(2, 200, i, device=device, dtype=dtype, requires_grad=True)
                ctx = torch.randn(2, 200, 128, device=device, dtype=dtype)
                y = m(x, ctx)
                shape_ok = y.shape == (2, 200, o)
                y.square().mean().backward()
                # every parameter that requires grad must receive a grad tensor,
                # or DDP reports an unused parameter
                missing = [n for n, p in m.named_parameters()
                           if p.requires_grad and p.grad is None]
                ok = shape_ok and not missing and torch.isfinite(y).all().item()
                ok_all &= ok
                if not ok:
                    print(f"    {label:6s} {mname:16s} chunk={chunk} topk={topk} "
                          f"shape={tuple(y.shape)} missing={missing}  ✗")
        print(f"    {label:6s} all 4 modes x {{chunk64,top1,per-token}}  "
              f"{'✓' if ok_all else '✗'}")
    return ok_all


def test_grouped_matches_compose(device, dtype):
    """route_side='basis' must also honour the grouped top-1 fast path.

    Unlike the output side, this is not bit-exact: the routed result here feeds
    ln_basis, and grouped/compose reduce in different orders (permuted grouped
    GEMM vs batched einsum), so a 1-ULP difference in h_pre propagates through
    LN and the shared W_m.  Judge by relative error.
    """
    print(f"\n  ── basis-side: grouped vs compose at topk=1 ({device}, {dtype}) ──")
    import copy
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    ok_all = True
    for label, i, o in [("c_fc", 512, 2048), ("attn", 512, 512)]:
        a = make(i, o, 512, dict(route_side='basis'), device, dtype, topk=1)
        bb = copy.deepcopy(a)
        bb.chunk_route_impl = 'grouped'
        torch.manual_seed(7)
        x = torch.randn(2, 256, i, device=device, dtype=dtype)
        ctx = torch.randn(2, 256, 128, device=device, dtype=dtype)
        a.eval(); bb.eval()
        with torch.no_grad():
            ya, yb = a(x, ctx), bb(x, ctx)
        err = ((ya - yb).float().norm() / ya.float().norm()).item()
        ok = err < tol
        ok_all &= ok
        print(f"    {label:6s} rel_fro={err:.2e} (tol {tol:.0e})  {'✓' if ok else '✗'}")
    return ok_all


# ── cost model + timing ──────────────────────────────────────────────────────

def cost_model(i, o, b, side, drop, K, chunk, T):
    """Returns (params, macs_per_token, materialized_elems_per_chunk)."""
    if drop:
        bank = K * o * i
        shared = 0
        macs = o * i
        mat = o * i
    elif side == 'basis':
        bank = K * b * i
        shared = o * b
        macs = b * i + o * b
        mat = b * i
    else:
        bank = K * o * b
        shared = b * i
        macs = b * i + o * b
        mat = o * b
    compose = K * mat / chunk if chunk > 0 else K * mat
    return bank + shared, macs + compose, mat


def report(depth, device, B, T, iters):
    C = ((depth * 64 + 127) // 128) * 128
    K, chunk = 8, 64
    projs = [("c_fc", C, 4 * C), ("c_proj", 4 * C, C), ("attn q/k/v/o", C, C)]
    dtype = torch.bfloat16
    print(f"\n{'='*94}")
    print(f" d{depth}: model_dim={C}  basis={C} (full rank)  K={K}  chunk={chunk}  "
          f"B={B} T={T} bf16")
    print(f"{'='*94}")

    block_mat = {m: 0.0 for m, _ in MODES}
    block_p = {m: 0.0 for m, _ in MODES}
    block_t = {m: [0.0, 0.0] for m, _ in MODES}
    for label, i, o in projs:
        mult = 4 if 'attn' in label else 1
        print(f"\n  {label}  ({i} -> {o})" + (f"  x{mult} per block" if mult > 1 else ""))
        print(f"    {'mode':16s} {'params':>9s} {'MAC/tok':>9s} {'W_eff/chunk':>12s} "
              f"{'fwd ms':>8s} {'f+b ms':>8s} {'peak MB':>8s}")
        for mname, extra in MODES:
            # mirror RemixedLinear.__init__: the drop is gated on in <= basis, and a
            # gated-out drop falls through to route_side
            drop = extra.get('drop_basis_proj', False) and i <= C
            side = ('output' if drop else
                    'basis' if (extra.get('route_side') == 'basis' or
                                (extra.get('route_side') == 'narrow' and i < o))
                    else 'output')
            p, macs, mat = cost_model(i, o, C, side, drop, K, chunk, T)
            block_mat[mname] += mult * mat
            block_p[mname] += mult * p
            try:
                m = make(i, o, C, extra, device, dtype, chunk=chunk)
                x = torch.randn(B, T, i, device=device, dtype=dtype)
                ctx = torch.randn(B, T, 128, device=device, dtype=dtype)

                def run(bw):
                    def one():
                        if bw:
                            m.zero_grad(set_to_none=True)
                            m(x, ctx).square().mean().backward()
                        else:
                            with torch.no_grad():
                                m(x, ctx)
                    for _ in range(3):
                        one()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        one()
                    torch.cuda.synchronize()
                    return ((time.perf_counter() - t0) / iters,
                            torch.cuda.max_memory_allocated() / 1e6)
                m.train()
                f, mf = run(False)
                bwd, mb = run(True)
                del m, x, ctx
                torch.cuda.empty_cache()
                block_t[mname][0] += mult * f * 1e3
                block_t[mname][1] += mult * bwd * 1e3
                tstr = f"{f*1e3:8.2f} {bwd*1e3:8.2f} {max(mf,mb):8.0f}"
            except Exception as e:
                tstr = f"  {type(e).__name__}"
            print(f"    {mname:16s} {p/1e6:8.2f}M {macs/1e6:8.2f}M {mat/1e6:11.2f}M "
                  f"{tstr}")

    print(f"\n  Per-block totals (4x attn + c_fc + c_proj):")
    print(f"    {'mode':16s} {'params':>9s} {'W_eff/chunk':>12s} {'fwd ms':>9s} "
          f"{'vs legacy':>10s} {'f+b ms':>9s} {'vs legacy':>10s}")
    ref_m, ref_f, ref_b = (block_mat['output (legacy)'], block_t['output (legacy)'][0],
                           block_t['output (legacy)'][1])
    for mname, _ in MODES:
        v, p = block_mat[mname], block_p[mname]
        f, b = block_t[mname]
        print(f"    {mname:16s} {p/1e6:8.2f}M {v/1e6:11.2f}M {f:9.2f} "
              f"{ref_f/f:9.2f}x {b:9.2f} {ref_b/b:9.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--T', type=int, default=2048)
    ap.add_argument('--iters', type=int, default=10)
    a = ap.parse_args()

    print("=" * 94)
    print(" Phase 31 — route_side / drop_basis_proj ablation")
    print("=" * 94)
    ok = test_resolution()
    ok &= test_shapes_and_grads('cpu', torch.float32)
    ok &= test_grouped_matches_compose('cpu', torch.float32)
    if torch.cuda.is_available():
        ok &= test_shapes_and_grads('cuda', torch.bfloat16)
        ok &= test_grouped_matches_compose('cuda', torch.float32)
        ok &= test_grouped_matches_compose('cuda', torch.bfloat16)
        report(a.depth, 'cuda', a.B, a.T, a.iters)
    else:
        print("\n[SKIP] no CUDA — correctness only, no timing")
    print("\n" + "=" * 94)
    print(" ALL PASSED ✓" if ok else " FAILURES ✗")
    print("=" * 94)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
