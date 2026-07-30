#!/usr/bin/env python3
"""Phase 31: grouped top-1 chunk routing must be numerically equivalent to the
legacy compose path, and must be faster.

The compose path builds W_eff = sum_k a_k T_k for every chunk and contracts it.
With topk=1, a is one-hot, so W_eff == T_{argmax} and the K-way contraction is
pure waste.  The grouped path permutes chunks into template-contiguous order and
runs one dense GEMM per template instead.

Checks:
  1. forward equivalence, compose vs grouped, across shapes/dtypes/devices
  2. gradient equivalence for every parameter
  3. top1_gate='ones' gives the router zero gradient; 'switch' does not
  4. shapes that need chunk padding, and chunks that all pick one template
  5. throughput + peak memory on GPU

Usage:  python tests/test_chunk_route_grouped.py
Exit 0 = all passed.
"""

import copy
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import RemixedLinear

BASE_KW = dict(
    n_templates=8,
    chunk_routing_size=64,
    template_topk=1,
    template_routing_learned=True,
    use_basis_gate=False,
    use_output_gate=True,
    use_context=True,
    output_gate_rank=8,
    basis_gate_mode='linear',
    use_quantile_route=0,
)


def _pair(in_f, out_f, ctx_dim, basis, device, dtype, extra=None, gate='ones'):
    """Two RemixedLinears with identical weights, one per impl."""
    kw = dict(BASE_KW, top1_gate=gate, **(extra or {}))
    torch.manual_seed(0)
    a = RemixedLinear(in_f, out_f, ctx_dim, basis_size=basis,
                      remixed_linear_kwargs=dict(kw, chunk_route_impl='compose'),
                      scale_basis=False).to(device=device, dtype=dtype)
    b = copy.deepcopy(a)
    b.chunk_route_impl = 'grouped'
    return a, b


def test_forward(device, dtype):
    print(f"\n  ── forward equivalence ({device}, {dtype}) ──")
    tol = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=6e-3, rtol=6e-3)
    cases = [
        ("square  B2 T128",   128, 128, 32, 32, 2, 128, 64),
        ("d->4d   B2 T128",   128, 512, 32, 32, 2, 128, 64),
        ("4d->d   B2 T128",   512, 128, 32, 32, 2, 128, 64),
        ("pad     B3 T200",   128, 128, 32, 32, 3, 200, 64),
        ("T<chunk B1 T40",    128, 128, 32, 32, 1, 40, 64),
        ("T==1    B1 T1",     128, 128, 32, 32, 1, 1, 64),
        ("chunk256 B2 T512",  128, 128, 32, 32, 2, 512, 256),
        ("K=16    B2 T256",   128, 256, 32, 64, 2, 256, 64),
    ]
    ok_all = True
    for name, inf, outf, cd, bs, B, T, ch in cases:
        extra = dict(chunk_routing_size=ch)
        if 'K=16' in name:
            extra['n_templates'] = 16
        a, b = _pair(inf, outf, cd, bs, device, dtype, extra)
        torch.manual_seed(7)
        x = torch.randn(B, T, inf, device=device, dtype=dtype)
        ctx = torch.randn(B, T, cd, device=device, dtype=dtype)
        a.eval(); b.eval()
        with torch.no_grad():
            ya, yb = a(x, ctx), b(x, ctx)
        err = (ya - yb).abs().max().item()
        ok = torch.allclose(ya, yb, **tol) and ya.shape == yb.shape
        ok_all &= ok
        print(f"    {name:18s} err={err:.2e}  {'✓' if ok else '✗'}")
    return ok_all


def test_grads(device, dtype):
    """Equivalence needs the same math on both sides, so gate='ones' here.
    The 'switch' gate is a deliberate change of function and is covered by
    test_router_gradient."""
    print(f"\n  ── gradient equivalence ({device}, {dtype}) ──")
    tol = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    a, b = _pair(128, 512, 32, 32, device, dtype, gate='ones')
    a.train(); b.train()
    grads = []
    for m in (a, b):
        torch.manual_seed(7)
        x = torch.randn(2, 128, 128, device=device, dtype=dtype, requires_grad=True)
        ctx = torch.randn(2, 128, 32, device=device, dtype=dtype)
        m(x, ctx).square().mean().backward()
        grads.append((x.grad, {n: p.grad for n, p in m.named_parameters()}))
    ok_all = torch.allclose(grads[0][0], grads[1][0], **tol)
    print(f"    input grad          {'✓' if ok_all else '✗'}")
    for n, ga in grads[0][1].items():
        gb = grads[1][1].get(n)
        # Every param that gets a grad on one impl must get one on the other:
        # a None where the other has a tensor is a DDP unused-parameter hazard.
        if ga is None and gb is None:
            continue
        ok = ga is not None and gb is not None and torch.allclose(ga, gb, **tol)
        ok_all &= ok
        print(f"    grad {n:24s} err={(ga-gb).abs().max().item():.2e}  {'✓' if ok else '✗'}")
    return ok_all


def test_router_gradient(device, dtype):
    """The point of top1_gate='switch': 'ones' leaves the router untrained."""
    print(f"\n  ── router gradient ({device}, {dtype}) ──")
    ok_all = True
    for impl in ('compose', 'grouped'):
        for gate in ('ones', 'switch'):
            kw = dict(BASE_KW, chunk_route_impl=impl, top1_gate=gate)
            torch.manual_seed(0)
            m = RemixedLinear(128, 256, 32, basis_size=32, remixed_linear_kwargs=kw,
                              scale_basis=False).to(device=device, dtype=dtype)
            m.train()
            torch.manual_seed(7)
            x = torch.randn(2, 128, 128, device=device, dtype=dtype)
            ctx = torch.randn(2, 128, 32, device=device, dtype=dtype)
            m(x, ctx).square().mean().backward()
            g = m.template_route.grad
            gn = 0.0 if g is None else g.abs().max().item()
            # 'ones' -> exactly zero (compose, and grouped which skips the coef);
            # 'switch' -> nonzero, but only on the grouped path (compose ignores it).
            want_nonzero = (gate == 'switch' and impl == 'grouped')
            ok = (gn > 0) == want_nonzero
            ok_all &= ok
            print(f"    {impl:8s} gate={gate:7s} |grad route|={gn:.3e}  "
                  f"expect {'nonzero' if want_nonzero else 'zero':8s} {'✓' if ok else '✗'}")
    return ok_all


def test_degenerate_routing(device, dtype):
    """All chunks selecting the same template must still work (K-1 empty groups)."""
    print(f"\n  ── degenerate routing ({device}, {dtype}) ──")
    a, b = _pair(128, 256, 32, 32, device, dtype)
    with torch.no_grad():  # force every chunk onto template 3
        for m in (a, b):
            m.template_route.fill_(-1.0)
            m.template_route[:, 3] = 1.0
    torch.manual_seed(7)
    # positive anchors, so sign(dot(anchor, route[:,k])) is unambiguous and no
    # two logits tie (a tie would let topk and argmax disagree)
    x = torch.randn(2, 256, 128, device=device, dtype=dtype).abs() + 0.1
    ctx = torch.randn(2, 256, 32, device=device, dtype=dtype)
    a.eval(); b.eval()
    with torch.no_grad():
        ya, yb = a(x, ctx), b(x, ctx)
    tol = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=6e-3, rtol=6e-3)
    ok = torch.allclose(ya, yb, **tol)
    print(f"    all chunks -> template 3   err={(ya-yb).abs().max().item():.2e}  "
          f"{'✓' if ok else '✗'}")
    return ok


def test_delta_equivalence(device, dtype):
    """A rank-r delta bank is a *constrained* full bank, so it must reproduce the
    compose path exactly when the full bank is built as T_k = T_0 + U_k V_k^T."""
    print(f"\n  ── delta bank vs equivalent full bank ({device}, {dtype}) ──")
    # The delta form does two GEMMs where the full bank does one, so identical math
    # still rounds differently.  In bf16 that is ~1 ULP (0.25 at output magnitude 32),
    # which elementwise allclose flags wherever the reference is near zero — so judge
    # bf16 by relative Frobenius error instead.
    is_f32 = dtype == torch.float32
    ok_all = True
    for topk, chunk, scope in [(0, 64, 'per_sequence'), (0, 256, 'per_sequence'),
                               (1, 64, 'per_sequence'), (0, 0, 'per_token'),
                               (0, 0, 'per_sequence')]:
        in_f, out_f, cd, bs, r, K = 128, 256, 32, 64, 8, 4
        kw = dict(BASE_KW, n_templates=K, template_topk=topk,
                  chunk_routing_size=chunk)
        # Build and equate in float32, then cast: constructing T_0 + V_k U_k in a
        # low-precision dtype would round the bank and make the two models differ
        # for reasons unrelated to the code under test.
        torch.manual_seed(0)
        delta = RemixedLinear(in_f, out_f, cd, basis_size=bs,
                              remixed_linear_kwargs=dict(kw, template_delta_rank=r),
                              scale_basis=False, routing_scope=scope).to(device, torch.float32)
        full = RemixedLinear(in_f, out_f, cd, basis_size=bs,
                            remixed_linear_kwargs=kw,
                            scale_basis=False, routing_scope=scope).to(device, torch.float32)
        with torch.no_grad():
            full_p = dict(full.named_parameters())
            for n, p in delta.named_parameters():
                if n in full_p and 'template_' not in n:
                    full_p[n].copy_(p)
            full.ln_basis.load_state_dict(delta.ln_basis.state_dict())
            if delta.template_route is not None:
                full.template_route.copy_(delta.template_route)
            # non-trivial deltas (U is zero-init, which would make the test vacuous)
            delta.template_delta_u.normal_(0, 0.1)
            V = delta.template_delta_v.reshape(bs, K, r).permute(1, 0, 2)    # (K,bs,r)
            U = delta.template_delta_u.reshape(K, r, out_f)                  # (K,r,out)
            for k in range(K):
                full.template_bank[k].copy_(delta.template_base + (V[k] @ U[k]).t())
        delta = delta.to(dtype); full = full.to(dtype)

        torch.manual_seed(7)
        x = torch.randn(2, 256, in_f, device=device, dtype=dtype)
        ctx = torch.randn(2, 256, cd, device=device, dtype=dtype)
        delta.eval(); full.eval()
        with torch.no_grad():
            yd, yf = delta(x, ctx), full(x, ctx)
        if is_f32:
            err = (yd - yf).abs().max().item()
            ok, unit = torch.allclose(yd, yf, atol=2e-5, rtol=1e-4), 'max_abs'
        else:
            err = ((yd - yf).float().norm() / yf.float().norm()).item()
            ok, unit = err < 2e-2, 'rel_fro'
        ok_all &= ok
        name = f"topk={topk} chunk={chunk or 'per-token'} {scope}"
        print(f"    {name:36s} {unit}={err:.2e}  {'✓' if ok else '✗'}")
    return ok_all


def test_delta_trains(device, dtype):
    """Deltas start at exactly zero (identity-preserving init) and must receive
    gradient; every delta parameter must get one, for DDP."""
    print(f"\n  ── delta bank trainability ({device}, {dtype}) ──")
    kw = dict(BASE_KW, template_topk=0, template_delta_rank=8, chunk_routing_size=64)
    torch.manual_seed(0)
    m = RemixedLinear(128, 256, 32, basis_size=64, remixed_linear_kwargs=kw,
                      scale_basis=False).to(device, dtype)
    torch.nn.init.zeros_(m.template_delta_u)
    m.train()
    torch.manual_seed(7)
    x = torch.randn(2, 128, 128, device=device, dtype=dtype)
    ctx = torch.randn(2, 128, 32, device=device, dtype=dtype)
    m(x, ctx).square().mean().backward()
    # At U=0 the delta term is identically zero, so dL/dV and dL/dalpha are both
    # zero at step 0 — mathematically correct, and it self-starts because
    # dL/dU = z^T g is nonzero.  What matters here is that every parameter gets a
    # grad *tensor* (not None), or DDP reports an unused parameter.
    ok_all = True
    for n, expect_nonzero in (('template_base', True), ('template_delta_v', False),
                              ('template_delta_u', True), ('template_route', False)):
        g = getattr(m, n).grad
        ok = g is not None and ((g.abs().max().item() > 0) == expect_nonzero)
        ok_all &= ok
        gn = 'None' if g is None else f"{g.abs().max().item():.3e}"
        print(f"    grad {n:20s} = {gn:12s} expect "
              f"{'nonzero' if expect_nonzero else '0 at U=0':10s} {'✓' if ok else '✗'}")
    npar = sum(p.numel() for p in m.non_gate_parameters())
    print(f"    non_gate_parameters covers {npar} elements")
    return ok_all


def test_timing(device, in_f, out_f, basis, B, T, chunk, label, iters=15):
    print(f"\n  ── timing: {label} (B={B} T={T} chunk={chunk}) ──")
    dtype = torch.bfloat16
    a, b = _pair(in_f, out_f, 128, basis, device, dtype, dict(chunk_routing_size=chunk))
    x = torch.randn(B, T, in_f, device=device, dtype=dtype)
    ctx = torch.randn(B, T, 128, device=device, dtype=dtype)

    def run(m, backward):
        def one():
            if backward:
                m.zero_grad(set_to_none=True)
                m(x, ctx).square().mean().backward()
            else:
                with torch.no_grad():
                    m(x, ctx)
        for _ in range(4):
            one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated() / 1e6

    a.train(); b.train()
    fa, ma = run(a, False)
    fb, mb = run(b, False)
    ba, pa = run(a, True)
    bb, pb = run(b, True)
    print(f"    fwd   compose={fa*1e3:8.2f} ms   grouped={fb*1e3:8.2f} ms   {fa/fb:5.2f}x")
    print(f"    f+b   compose={ba*1e3:8.2f} ms   grouped={bb*1e3:8.2f} ms   {ba/bb:5.2f}x")
    print(f"    peak  compose={max(ma,pa):8.0f} MB  grouped={max(mb,pb):8.0f} MB  "
          f"{max(ma,pa)/max(mb,pb):5.2f}x less")
    return fb < fa


def test_timing_soft(device, in_f, out_f, basis, B, T, chunk, label, r=16, iters=12):
    """The 29C config: soft mixing over all K.  compose vs low-rank deltas."""
    print(f"\n  ── timing (SOFT, topk=0): {label} (B={B} T={T} chunk={chunk} r={r}) ──")
    dtype = torch.bfloat16
    kw = dict(BASE_KW, template_topk=0, chunk_routing_size=chunk)
    torch.manual_seed(0)
    mk = lambda extra: RemixedLinear(in_f, out_f, 128, basis_size=basis,
                                     remixed_linear_kwargs=dict(kw, **extra),
                                     scale_basis=False).to(device, dtype)
    compose = mk({})
    delta = mk(dict(template_delta_rank=r))
    x = torch.randn(B, T, in_f, device=device, dtype=dtype)
    ctx = torch.randn(B, T, 128, device=device, dtype=dtype)

    def run(m, backward):
        def one():
            if backward:
                m.zero_grad(set_to_none=True)
                m(x, ctx).square().mean().backward()
            else:
                with torch.no_grad():
                    m(x, ctx)
        for _ in range(4):
            one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated() / 1e6

    compose.train(); delta.train()
    fc, mc = run(compose, False)
    fd, md = run(delta, False)
    bc, pc = run(compose, True)
    bd, pd = run(delta, True)
    npc = sum(p.numel() for p in compose.parameters()) / 1e6
    npd = sum(p.numel() for p in delta.parameters()) / 1e6
    print(f"    fwd   compose={fc*1e3:8.2f} ms   delta={fd*1e3:8.2f} ms   {fc/fd:5.2f}x")
    print(f"    f+b   compose={bc*1e3:8.2f} ms   delta={bd*1e3:8.2f} ms   {bc/bd:5.2f}x")
    print(f"    peak  compose={max(mc,pc):8.0f} MB  delta={max(md,pd):8.0f} MB  "
          f"{max(mc,pc)/max(md,pd):5.2f}x less")
    print(f"    params compose={npc:.2f}M  delta={npd:.2f}M")
    return fd < fc


def main():
    print("=" * 74)
    print(" Phase 31 — chunk-routing throughput (grouped top-1 + low-rank deltas)")
    print("=" * 74)
    ok = True
    for fn in (test_forward, test_grads, test_router_gradient, test_degenerate_routing,
               test_delta_equivalence, test_delta_trains):
        ok &= fn('cpu', torch.float32)

    if torch.cuda.is_available():
        print(f"\n{'─'*74}\n GPU: {torch.cuda.get_device_name(0)}\n{'─'*74}")
        for dt in (torch.float32, torch.bfloat16):
            for fn in (test_forward, test_router_gradient, test_degenerate_routing,
                       test_delta_equivalence):
                ok &= fn('cuda', dt)
        ok &= test_grads('cuda', torch.float32)
        ok &= test_delta_trains('cuda', torch.float32)
        test_timing('cuda', 768, 3072, 768, 4, 2048, 64, "d12 c_fc")
        test_timing('cuda', 768, 768, 768, 4, 2048, 64, "d12 attn q/k/v/o")
        test_timing_soft('cuda', 768, 3072, 768, 4, 2048, 64, "d12 c_fc")
        test_timing_soft('cuda', 768, 768, 768, 4, 2048, 64, "d12 attn q/k/v/o")
        test_timing_soft('cuda', 768, 3072, 768, 4, 2048, 0, "d12 c_fc PER-TOKEN")
    else:
        print("\n[SKIP] no CUDA device")

    print("\n" + "=" * 74)
    print(" ALL PASSED ✓" if ok else " FAILURES ✗")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
