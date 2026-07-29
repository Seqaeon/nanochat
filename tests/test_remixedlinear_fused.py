#!/usr/bin/env python3
"""Comprehensive numerical equivalence test: RemixedLinear vs RemixedLinearFused.

Tests across multiple:
  - Tensor shapes (B, T — including non-chunk-aligned, edge cases)
  - Dtypes (float32, bfloat16)
  - Configs (with/without basis gate, different n_templates, chunk sizes)
  - Devices (CPU, GPU if available)

Also tests:
  - Determinism (same input → bit-identical output across repeated calls)
  - Gradient accumulation (multiple backward passes)
  - Multi-step training loop (loss and weight divergence after N steps)
  - Timing comparison (Original vs Fused vs Fused+Compiled)

Usage:
    python tests/test_remixedlinear_fused.py

Exit code 0 = all tests passed, 1 = at least one test failed.
"""

import copy
import os
import sys
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import RemixedLinear, RemixedLinearFused


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_orig(in_f, out_f, ctx_dim, basis, kwargs, device, dtype):
    """Create a RemixedLinear with given kwargs."""
    return RemixedLinear(
        in_f, out_f, ctx_dim,
        basis_size=basis,
        remixed_linear_kwargs=kwargs,
        scale_basis=False,
        routing_scope='per_sequence',
    ).to(device=device, dtype=dtype)


def _make_pair(in_f, out_f, ctx_dim, basis, kwargs, device, dtype):
    """Create matched pair with identical weights."""
    torch.manual_seed(0)
    orig = _make_orig(in_f, out_f, ctx_dim, basis, kwargs, device, dtype)
    fused = RemixedLinearFused.from_remixed_linear(orig)
    fused = fused.to(device=device, dtype=dtype)
    return orig, fused


def _check_close(a, b, atol, rtol, label):
    """Check allclose and report."""
    max_abs = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = 'PASS ✓' if ok else 'FAIL ✗'
    print(f"    {label}: max_abs={max_abs:.2e}  {status}")
    return ok


def _time_one(model, x, ctx, n_iters, is_cuda, backward=False):
    """Time forward (and optionally backward) for a single model."""
    if backward:
        model.train()
        for _ in range(5):
            x_ = torch.randn_like(x, requires_grad=True)
            model(x_, ctx).sum().backward()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x_ = torch.randn_like(x, requires_grad=True)
            model(x_, ctx).sum().backward()
        if is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters
    else:
        model.eval()
        for _ in range(10):
            with torch.no_grad():
                model(x, ctx)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                model(x, ctx)
        if is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters


# ── Default 29C config ───────────────────────────────────────────────────────

DEFAULT_KW = dict(
    n_templates=8,
    chunk_routing_size=64,
    template_routing_learned=False,
    use_basis_gate=False,
    use_output_gate=True,
    use_context=True,
    output_gate_rank=8,
    basis_gate_mode='linear',
    template_topk=0,
)


# ── Core tests ───────────────────────────────────────────────────────────────

def test_shapes(device='cpu', dtype=torch.float32):
    """Test many (B, T) shapes including edge cases and non-chunk-aligned."""
    print(f"\n  ── Shape sweep ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    chunk = 64
    kw = dict(DEFAULT_KW, chunk_routing_size=chunk)

    shapes = [
        (1, 64),     # B=1, T exactly = chunk
        (1, 65),     # T = chunk + 1  (padding needed)
        (2, 128),    # standard
        (2, 127),    # T = 2*chunk - 1 (padding needed)
        (1, 1),      # T = 1 (extreme edge case — single token)
        (4, 256),    # larger batch
        (1, 63),     # T < chunk (needs 1 padded token)
        (2, 192),    # T = 3 * chunk (exact multiple)
        (3, 200),    # odd batch, non-aligned T
    ]

    orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    all_ok = True

    for B, T in shapes:
        torch.manual_seed(42 + B * 1000 + T)
        x = torch.randn(B, T, in_f, device=device, dtype=dtype)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        orig.eval(); fused.eval()
        with torch.no_grad():
            y_o = orig(x, ctx)
            y_f = fused(x, ctx)
        max_abs = (y_o - y_f).abs().max().item()
        ok = torch.allclose(y_o, y_f, atol=1e-5, rtol=1e-4)
        if not ok:
            all_ok = False
        status = '✓' if ok else '✗'
        print(f"    B={B:2d} T={T:4d}  err={max_abs:.2e}  {status}")

    return all_ok


def test_configs(device='cpu', dtype=torch.float32):
    """Test different configs: basis gate on/off, various n_templates, chunk sizes."""
    print(f"\n  ── Config sweep ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 128

    configs = [
        ("no_basis_gate",  dict(DEFAULT_KW, use_basis_gate=False)),
        ("basis_gate_lin", dict(DEFAULT_KW, use_basis_gate=True, basis_gate_mode='linear')),
        ("basis_gate_mlp", dict(DEFAULT_KW, use_basis_gate=True, basis_gate_mode='mlp')),
        ("K=4 chunk=32",   dict(DEFAULT_KW, n_templates=4, chunk_routing_size=32)),
        ("K=16 chunk=64",  dict(DEFAULT_KW, n_templates=16, chunk_routing_size=64)),
        ("K=2 chunk=128",  dict(DEFAULT_KW, n_templates=2, chunk_routing_size=128)),
        ("no_output_gate", dict(DEFAULT_KW, use_output_gate=False)),
        ("learned_route",  dict(DEFAULT_KW, template_routing_learned=True)),
    ]

    all_ok = True
    for name, kw in configs:
        torch.manual_seed(0)
        orig = _make_orig(in_f, out_f, ctx_dim, basis, kw, device, dtype)
        fused = RemixedLinearFused.from_remixed_linear(orig).to(device, dtype)

        torch.manual_seed(42)
        x = torch.randn(B, T, in_f, device=device, dtype=dtype)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        orig.eval(); fused.eval()
        with torch.no_grad():
            y_o = orig(x, ctx)
            y_f = fused(x, ctx)
        max_abs = (y_o - y_f).abs().max().item()
        ok = torch.allclose(y_o, y_f, atol=1e-5, rtol=1e-4)
        if not ok:
            all_ok = False
        print(f"    {name:20s}  err={max_abs:.2e}  {'✓' if ok else '✗'}")

    return all_ok


def test_asymmetric_dims(device='cpu', dtype=torch.float32):
    """Test non-square projections like FFN: d→4d and 4d→d."""
    print(f"\n  ── Asymmetric dims ({device}, {dtype}) ──")
    B, T = 2, 128
    all_ok = True

    for in_f, out_f, label in [(128, 512, "d→4d (c_fc)"), (512, 128, "4d→d (c_proj)")]:
        ctx_dim, basis = 32, 32
        kw = dict(DEFAULT_KW)
        orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
        torch.manual_seed(42)
        x = torch.randn(B, T, in_f, device=device, dtype=dtype)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        orig.eval(); fused.eval()
        with torch.no_grad():
            y_o = orig(x, ctx)
            y_f = fused(x, ctx)
        ok = _check_close(y_o, y_f, 1e-5, 1e-4, label)
        if not ok:
            all_ok = False

    return all_ok


def test_backward_grads(device='cpu', dtype=torch.float32):
    """Full backward pass: check input grad + all param grads."""
    print(f"\n  ── Backward grads ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 128
    kw = dict(DEFAULT_KW)
    orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    orig.train(); fused.train()

    torch.manual_seed(42)
    x1 = torch.randn(B, T, in_f, device=device, dtype=dtype, requires_grad=True)
    ctx1 = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
    orig(x1, ctx1).sum().backward()

    torch.manual_seed(42)
    x2 = torch.randn(B, T, in_f, device=device, dtype=dtype, requires_grad=True)
    ctx2 = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
    fused(x2, ctx2).sum().backward()

    all_ok = _check_close(x1.grad, x2.grad, 1e-5, 1e-4, "input grad")

    orig_p = dict(orig.named_parameters())
    fused_p = dict(fused.named_parameters())
    n_checked = 0
    for name, pf in fused_p.items():
        if pf.grad is None or name not in orig_p or orig_p[name].grad is None:
            continue
        n_checked += 1
        if not _check_close(pf.grad, orig_p[name].grad, 1e-5, 1e-4, f"grad({name})"):
            all_ok = False

    print(f"    ({n_checked} param grads checked)")
    return all_ok


def test_determinism(device='cpu', dtype=torch.float32):
    """Same input → bit-identical output across 5 repeated forward calls."""
    print(f"\n  ── Determinism ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 128
    kw = dict(DEFAULT_KW)
    _, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    fused.eval()

    torch.manual_seed(42)
    x = torch.randn(B, T, in_f, device=device, dtype=dtype)
    ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)

    with torch.no_grad():
        ref = fused(x, ctx).clone()
        all_ok = True
        for i in range(5):
            y = fused(x, ctx)
            err = (y - ref).abs().max().item()
            ok = err == 0.0
            if not ok:
                all_ok = False
            print(f"    Run {i+1}: max_diff={err:.2e}  {'✓' if ok else '✗'}")

    return all_ok


def test_grad_accumulation(device='cpu', dtype=torch.float32):
    """Multiple backward passes (gradient accumulation) should match."""
    print(f"\n  ── Gradient accumulation ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 64
    kw = dict(DEFAULT_KW)
    orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    orig.train(); fused.train()
    orig.zero_grad(); fused.zero_grad()

    n_accum = 4
    for step in range(n_accum):
        torch.manual_seed(100 + step)
        x1 = torch.randn(B, T, in_f, device=device, dtype=dtype, requires_grad=True)
        ctx1 = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        orig(x1, ctx1).sum().backward()

        torch.manual_seed(100 + step)
        x2 = torch.randn(B, T, in_f, device=device, dtype=dtype, requires_grad=True)
        ctx2 = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        fused(x2, ctx2).sum().backward()

    all_ok = True
    orig_p = dict(orig.named_parameters())
    fused_p = dict(fused.named_parameters())
    worst_name, worst_err = '', 0.0

    for name, pf in fused_p.items():
        if pf.grad is None or name not in orig_p or orig_p[name].grad is None:
            continue
        err = (pf.grad - orig_p[name].grad).abs().max().item()
        if err > worst_err:
            worst_name, worst_err = name, err
        if not torch.allclose(pf.grad, orig_p[name].grad, atol=1e-5, rtol=1e-4):
            all_ok = False

    print(f"    {n_accum} accum steps  worst={worst_name} {worst_err:.2e}  "
          f"{'PASS ✓' if all_ok else 'FAIL ✗'}")
    return all_ok


def test_training_loop(device='cpu', dtype=torch.float32, n_steps=20):
    """Multi-step training loop: check loss and weight divergence after N steps."""
    print(f"\n  ── Training loop ({n_steps} steps, {device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 128
    kw = dict(DEFAULT_KW)
    orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    orig.train(); fused.train()

    lr = 1e-3
    opt_o = torch.optim.AdamW(orig.parameters(), lr=lr)
    opt_f = torch.optim.AdamW(fused.parameters(), lr=lr)

    losses_o, losses_f = [], []

    for step in range(n_steps):
        torch.manual_seed(step * 7 + 13)
        x = torch.randn(B, T, in_f, device=device, dtype=dtype)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
        target = torch.randn(B, T, out_f, device=device, dtype=dtype)

        # Original
        opt_o.zero_grad()
        y_o = orig(x, ctx)
        loss_o = F.mse_loss(y_o, target)
        loss_o.backward()
        opt_o.step()
        losses_o.append(loss_o.item())

        # Fused
        opt_f.zero_grad()
        y_f = fused(x, ctx)
        loss_f = F.mse_loss(y_f, target)
        loss_f.backward()
        opt_f.step()
        losses_f.append(loss_f.item())

    # Check loss trajectory divergence
    loss_diff = [abs(a - b) for a, b in zip(losses_o, losses_f)]
    max_loss_diff = max(loss_diff)
    final_loss_diff = loss_diff[-1]

    # Check final weight divergence
    orig_sd = dict(orig.named_parameters())
    fused_sd = dict(fused.named_parameters())
    max_w_err = 0.0
    for name, pf in fused_sd.items():
        if name in orig_sd:
            err = (pf.data - orig_sd[name].data).abs().max().item()
            max_w_err = max(max_w_err, err)

    ok_loss = max_loss_diff < 1e-4
    ok_weight = max_w_err < 1e-4

    print(f"    Loss diff:   max={max_loss_diff:.2e}  final={final_loss_diff:.2e}  "
          f"{'✓' if ok_loss else '✗'}")
    print(f"    Weight diff: max={max_w_err:.2e}  {'✓' if ok_weight else '✗'}")
    print(f"    Loss[0] o={losses_o[0]:.6f} f={losses_f[0]:.6f}")
    print(f"    Loss[{n_steps-1}] o={losses_o[-1]:.6f} f={losses_f[-1]:.6f}")

    return ok_loss and ok_weight


def test_quantile_routing(device='cpu', dtype=torch.float32):
    """Test EMA quantile-balanced routing and cross-attention routing."""
    print(f"\n  ── Quantile routing ({device}, {dtype}) ──")
    in_f, out_f, ctx_dim, basis = 128, 128, 32, 32
    B, T = 2, 128
    all_ok = True

    for qr_mode, name in [(1, "QuantileBalanced"), (2, "QuantileCrossAttn")]:
        kw = dict(DEFAULT_KW,
                  use_quantile_route=qr_mode,
                  template_topk=4,   # required for quantile routing
                  template_routing_learned=True)

        torch.manual_seed(0)
        fused = RemixedLinearFused(
            in_f, out_f, ctx_dim, basis_size=basis,
            remixed_linear_kwargs=kw, scale_basis=False,
        ).to(device=device, dtype=dtype)

        # ── Forward (train + eval) ──
        torch.manual_seed(42)
        x = torch.randn(B, T, in_f, device=device, dtype=dtype)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)

        fused.train()
        y_train = fused(x, ctx)
        ok_fwd = y_train.shape == (B, T, out_f)
        print(f"    {name:22s} fwd shape: {ok_fwd}  {'✓' if ok_fwd else '✗'}")
        if not ok_fwd:
            all_ok = False
            continue

        # ── Backward ──
        loss = y_train.sum()
        loss.backward()
        n_grads = sum(1 for p in fused.parameters() if p.grad is not None)
        ok_bwd = n_grads > 0
        print(f"    {name:22s} bwd grads: {n_grads} params  {'✓' if ok_bwd else '✗'}")
        if not ok_bwd:
            all_ok = False

        # ── Eval determinism ──
        fused.eval()
        fused.zero_grad()
        with torch.no_grad():
            y1 = fused(x, ctx)
            y2 = fused(x, ctx)
        det_err = (y1 - y2).abs().max().item()
        ok_det = det_err == 0.0
        print(f"    {name:22s} eval det:  {det_err:.2e}  {'✓' if ok_det else '✗'}")
        if not ok_det:
            all_ok = False

        # ── Mini training loop (5 steps) ──
        fused.train()
        opt = torch.optim.AdamW(fused.parameters(), lr=1e-3)
        losses = []
        for step in range(5):
            torch.manual_seed(step * 7)
            x_s = torch.randn(B, T, in_f, device=device, dtype=dtype)
            ctx_s = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
            tgt = torch.randn(B, T, out_f, device=device, dtype=dtype)
            opt.zero_grad()
            y = fused(x_s, ctx_s)
            l = F.mse_loss(y, tgt)
            l.backward()
            opt.step()
            losses.append(l.item())
        ok_train = losses[-1] < losses[0] * 1.5  # sanity: not diverging
        print(f"    {name:22s} train:     L0={losses[0]:.4f} L4={losses[-1]:.4f}  "
              f"{'✓' if ok_train else '✗'}")
        if not ok_train:
            all_ok = False

    return all_ok


def test_timing(device, dtype, in_f, out_f, ctx_dim, basis, B, T, label,
                n_iters=200, include_compiled=False):
    """Timing: Original vs Fused vs Fused+Compiled."""
    print(f"\n  ── Timing: {label} ({device}, {dtype.__name__ if hasattr(dtype,'__name__') else dtype}) ──")
    kw = dict(DEFAULT_KW)
    orig, fused = _make_pair(in_f, out_f, ctx_dim, basis, kw, device, dtype)
    x = torch.randn(B, T, in_f, device=device, dtype=dtype)
    ctx = torch.randn(B, T, ctx_dim, device=device, dtype=dtype)
    is_cuda = (device != 'cpu')

    t_orig = _time_one(orig, x, ctx, n_iters, is_cuda, backward=False)
    t_fused = _time_one(fused, x, ctx, n_iters, is_cuda, backward=False)
    sp = t_orig / max(t_fused, 1e-12)
    print(f"    Fwd:  orig={t_orig*1000:.3f}ms  fused={t_fused*1000:.3f}ms  ({sp:.2f}x)")

    t_orig_fb = _time_one(orig, x, ctx, n_iters, is_cuda, backward=True)
    t_fused_fb = _time_one(fused, x, ctx, n_iters, is_cuda, backward=True)
    sp_fb = t_orig_fb / max(t_fused_fb, 1e-12)
    print(f"    F+B:  orig={t_orig_fb*1000:.3f}ms  fused={t_fused_fb*1000:.3f}ms  ({sp_fb:.2f}x)")

    if include_compiled and is_cuda:
        fused_c = copy.deepcopy(fused).compile(mode='default', fullgraph=True)
        # Warmup
        print(f"    [compiling...]", end='', flush=True)
        fused_c.eval()
        for _ in range(5):
            with torch.no_grad():
                fused_c(x, ctx)
        torch.cuda.synchronize()
        t_comp = _time_one(fused_c, x, ctx, n_iters, is_cuda, backward=False)
        sp_c = t_orig / max(t_comp, 1e-12)
        print(f"\r    Fwd compiled: {t_comp*1000:.3f}ms  ({sp_c:.2f}x)")

        fused_c_t = copy.deepcopy(fused).compile(mode='default', fullgraph=True)
        fused_c_t.train()
        for _ in range(5):
            x_ = torch.randn_like(x, requires_grad=True)
            fused_c_t(x_, ctx).sum().backward()
        torch.cuda.synchronize()
        t_comp_fb = _time_one(fused_c_t, x, ctx, n_iters, is_cuda, backward=True)
        sp_c_fb = t_orig_fb / max(t_comp_fb, 1e-12)
        print(f"    F+B compiled: {t_comp_fb*1000:.3f}ms  ({sp_c_fb:.2f}x)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 70
    print(SEP)
    print(" RemixedLinearFused — Comprehensive Numerical Equivalence Test")
    print(SEP)

    all_passed = True

    # ── CPU tests (float32 — exact numerics) ──
    print("\n" + "─" * 70)
    print(" CPU · float32")
    print("─" * 70)

    for test_fn in [test_shapes, test_configs, test_asymmetric_dims,
                    test_backward_grads, test_determinism,
                    test_grad_accumulation, test_training_loop,
                    test_quantile_routing]:
        if not test_fn('cpu', torch.float32):
            all_passed = False

    test_timing('cpu', torch.float32, 128, 128, 32, 32, B=2, T=128,
                label="d4-like", n_iters=50)

    # ── GPU tests ──
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        for dtype, dname in [(torch.float32, "float32"), (torch.bfloat16, "bfloat16")]:
            print(f"\n{'─'*70}")
            print(f" GPU ({dev}) · {dname}")
            print("─" * 70)

            for test_fn in [test_shapes, test_configs, test_asymmetric_dims,
                            test_backward_grads, test_determinism,
                            test_grad_accumulation,
                            test_quantile_routing]:
                if not test_fn('cuda', dtype):
                    all_passed = False

            if dtype == torch.float32:
                if not test_training_loop('cuda', dtype, n_steps=20):
                    all_passed = False

        # Timing benchmarks (bf16, with compiled)
        test_timing('cuda', torch.bfloat16, 128, 128, 32, 32, B=4, T=256,
                    label="d4-like", n_iters=200, include_compiled=True)
        test_timing('cuda', torch.bfloat16, 768, 768, 128, 192, B=2, T=256,
                    label="d12-like", n_iters=200, include_compiled=True)
        test_timing('cuda', torch.bfloat16, 256, 1024, 64, 64, B=2, T=256,
                    label="d→4d (c_fc)", n_iters=200, include_compiled=True)
    else:
        print("\n[SKIP] No CUDA device — GPU tests skipped")

    # ── Summary ──
    print(f"\n{SEP}")
    if all_passed:
        print(" ALL TESTS PASSED ✓")
    else:
        print(" SOME TESTS FAILED ✗")
    print(SEP)
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
