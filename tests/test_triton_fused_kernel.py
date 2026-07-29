#!/usr/bin/env python3
"""Approach 3: FlashAttention-style fused Triton kernel for RemixedLinear.

Key insight (FlashAttention analogy):
  Standard path materializes W_eff (BN, O, B_sz) in HBM between two kernels:
    1. W_eff = weights @ templates       (template assembly)
    2. out   = h_chunked @ W_eff.T       (chunk mixing)

  Fused kernel: compute W_eff tiles in SRAM, immediately multiply with h,
  never write W_eff to HBM. Saves one full round-trip of the W_eff tensor.

  Memory saved per layer: BN * O * B_sz * dtype_size
    d4:  16 * 256 * 64 * 2 = 0.5 MB
    d12:  8 * 768 * 192 * 2 = 2.4 MB

Benchmarks Triton fused vs matmul+bmm vs torch.compile(matmul+bmm).
"""

import os
import sys
import time
import copy

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import RemixedLinear, RemixedLinearFused


# ══════════════════════════════════════════════════════════════════════════════
# Triton kernel: fused template assembly + chunk mixing
# ══════════════════════════════════════════════════════════════════════════════
#
# Computes: out[bn, s, o] = sum_b h[bn, s, b] * sum_k w[bn, k] * T[k, o, b]
#
# Grid: (BN, ceil(O / BLOCK_O))
# Each program handles all S (chunk) positions for one bn and one O-tile.
# The K loop and B_sz reduction happen in registers/SRAM.


@triton.jit
def _fused_template_mix_fwd(
    # Pointers
    H, W, T_bank, Out,
    # Dims
    BN, S, B_sz, O,
    K: tl.constexpr,
    # Strides for H: (BN, S, B_sz)
    h_s0, h_s1, h_s2,
    # Strides for W: (BN, K)
    w_s0, w_s1,
    # Strides for T_bank: (K, O, B_sz)
    t_s0, t_s1, t_s2,
    # Strides for Out: (BN, S, O)
    o_s0, o_s1, o_s2,
    # Tile sizes
    BLOCK_O: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_o = tl.program_id(1)

    # O-tile offsets
    o_off = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)  # (BLOCK_O,)
    o_mask = o_off < O

    # Base pointer for routing weights of this bn
    w_base = W + pid_bn * w_s0

    # For each chunk position s:
    for s in range(S):
        # Accumulate: out[bn, s, o_tile] = sum_b h[bn,s,b] * sum_k w[k]*T[k,o,b]
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)

        # Tile over B_sz dimension
        for b_start in range(0, B_sz, BLOCK_B):
            b_off = b_start + tl.arange(0, BLOCK_B)  # (BLOCK_B,)
            b_mask = b_off < B_sz

            # Load h[bn, s, b_tile]: (BLOCK_B,)
            h_ptrs = H + pid_bn * h_s0 + s * h_s1 + b_off * h_s2
            h_tile = tl.load(h_ptrs, mask=b_mask, other=0.0)  # (BLOCK_B,)

            # Build W_eff tile: (BLOCK_O, BLOCK_B) = sum_k w[k] * T[k, o_tile, b_tile]
            weff = tl.zeros((BLOCK_O, BLOCK_B), dtype=tl.float32)
            for k in range(K):
                # Load scalar routing weight w[bn, k]
                wk = tl.load(w_base + k * w_s1)

                # T[k, o_tile, b_tile]: load (BLOCK_O, BLOCK_B) tile
                t_ptrs = (T_bank
                          + k * t_s0
                          + o_off[:, None] * t_s1
                          + b_off[None, :] * t_s2)
                t_tile = tl.load(t_ptrs,
                                 mask=o_mask[:, None] & b_mask[None, :],
                                 other=0.0)  # (BLOCK_O, BLOCK_B)
                weff += wk * t_tile

            # acc += weff @ h_tile  → dot product along B dim → (BLOCK_O,)
            acc += tl.sum(weff * h_tile[None, :], axis=1)

        # Store out[bn, s, o_tile]
        out_ptrs = Out + pid_bn * o_s0 + s * o_s1 + o_off * o_s2
        tl.store(out_ptrs, acc, mask=o_mask)


def fused_template_mix(h_chunked, weights, templates):
    """Fused template assembly + chunk mixing via Triton.

    Args:
        h_chunked: (BN, S, B_sz)  — basis-projected + LN'd input chunks
        weights:   (BN, K)        — softmax routing weights
        templates: (K, O, B_sz)   — template bank

    Returns:
        out: (BN, S, O)
    """
    BN, S, B_sz = h_chunked.shape
    K, O, B_sz2 = templates.shape
    assert B_sz == B_sz2

    out = torch.empty(BN, S, O, device=h_chunked.device, dtype=h_chunked.dtype)

    # Tile sizes — tuned for SM86 (RTX 3050 Ti, 48KB shared memory)
    BLOCK_O = min(64, triton.next_power_of_2(O))
    BLOCK_B = min(64, triton.next_power_of_2(B_sz))

    grid = (BN, triton.cdiv(O, BLOCK_O))

    _fused_template_mix_fwd[grid](
        h_chunked, weights, templates, out,
        BN, S, B_sz, O, K,
        # Strides
        h_chunked.stride(0), h_chunked.stride(1), h_chunked.stride(2),
        weights.stride(0), weights.stride(1),
        templates.stride(0), templates.stride(1), templates.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_O=BLOCK_O, BLOCK_B=BLOCK_B,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Reference implementations for comparison
# ══════════════════════════════════════════════════════════════════════════════

def ref_matmul_bmm(h_chunked, weights, templates):
    """Current RemixedLinearFused approach: matmul + bmm."""
    BN, S, B_sz = h_chunked.shape
    K, O, _ = templates.shape
    W_eff = weights.reshape(BN, K) @ templates.reshape(K, O * B_sz)
    W_eff_t = W_eff.reshape(BN, O, B_sz).transpose(1, 2)
    return torch.bmm(h_chunked, W_eff_t)


def ref_einsum(h_chunked, weights, templates):
    """Original RemixedLinear approach: two einsums."""
    # weights: (BN, K), templates: (K, O, B_sz)
    BN = weights.shape[0]
    W_eff = torch.einsum('nk,kob->nob', weights, templates)
    return torch.einsum('nsc,noc->nso', h_chunked, W_eff)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_kernel(name, fn, h, w, t, n_iters=500, warmup=50):
    """Time a kernel function."""
    for _ in range(warmup):
        fn(h, w, t)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(h, w, t)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters

    return elapsed


def run_benchmark(label, BN, S, B_sz, O, K, dtype=torch.bfloat16):
    """Run all three approaches and compare."""
    print(f"\n  ── {label} (BN={BN}, S={S}, B_sz={B_sz}, O={O}, K={K}) ──")

    h = torch.randn(BN, S, B_sz, device='cuda', dtype=dtype)
    w = F.softmax(torch.randn(BN, K, device='cuda', dtype=dtype), dim=-1)
    t = torch.randn(K, O, B_sz, device='cuda', dtype=dtype)

    # Correctness check
    ref = ref_matmul_bmm(h, w, t)
    tri = fused_template_mix(h, w, t)
    err = (ref - tri).abs().max().item()
    print(f"    Correctness vs matmul+bmm: max_abs={err:.2e}  "
          f"{'✓' if err < 1e-2 else '✗'}")

    # Timing
    t_einsum = benchmark_kernel("einsum", ref_einsum, h, w, t)
    t_matmul = benchmark_kernel("matmul+bmm", ref_matmul_bmm, h, w, t)
    t_triton = benchmark_kernel("triton_fused", fused_template_mix, h, w, t)

    print(f"    Einsum:       {t_einsum*1000:.3f} ms  (baseline)")
    print(f"    Matmul+bmm:   {t_matmul*1000:.3f} ms  ({t_einsum/t_matmul:.2f}x vs einsum)")
    print(f"    Triton fused: {t_triton*1000:.3f} ms  ({t_einsum/t_triton:.2f}x vs einsum, "
          f"{t_matmul/t_triton:.2f}x vs matmul+bmm)")


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: plug Triton kernel into RemixedLinearFused forward
# ══════════════════════════════════════════════════════════════════════════════

def test_e2e_replacement(in_f=256, out_f=256, ctx_dim=64, basis=64,
                         K=8, chunk=64, B=4, T=256, dtype=torch.bfloat16):
    """Replace the matmul+bmm in RemixedLinearFused with the Triton kernel
    and measure end-to-end layer performance."""
    print(f"\n  ── End-to-end layer comparison (B={B}, T={T}, {in_f}→{out_f}) ──")

    from nanochat.gpt import RemixedLinearFused
    kw = dict(
        n_templates=K, chunk_routing_size=chunk,
        template_routing_learned=False,
        use_basis_gate=False, use_output_gate=True,
        use_context=True, output_gate_rank=8,
        template_topk=0,
    )
    torch.manual_seed(0)
    fused = RemixedLinearFused(
        in_f, out_f, ctx_dim, basis_size=basis,
        remixed_linear_kwargs=kw, scale_basis=False,
    ).cuda().to(dtype)

    x = torch.randn(B, T, in_f, device='cuda', dtype=dtype)
    ctx = torch.randn(B, T, ctx_dim, device='cuda', dtype=dtype)

    # Monkey-patch forward to use Triton kernel for the hot path
    _orig_forward = fused.forward

    def _triton_forward(x, context_state=None, route_weights=None, **kwargs):
        """Forward with Triton kernel replacing the matmul+bmm hot path."""
        dt = x.dtype
        B_, T_len, C_ = x.shape
        K_ = fused.n_templates
        O_ = fused.out_features
        B_sz_ = fused.basis_size
        ch = fused.chunk_routing_size

        _ln_dtype = fused.ln_basis.weight.dtype
        h = fused.ln_basis(fused.basis(x).to(dtype=_ln_dtype)).to(dtype=dt)

        gate_out = None
        if context_state is not None:
            ctx_ = context_state.to(dtype=dt)
            if fused.use_basis_gate and fused.basis_modulator is not None:
                gate_logits = fused.basis_modulator(ctx_)
                h = h * torch.sigmoid(gate_logits / fused.gate_temperature).to(dtype=dt)
            if fused.use_output_gate:
                coeffs = fused.output_gate_coeffs(ctx_)
                gl = torch.matmul(coeffs, fused.output_gate_basis.to(dtype=dt))
                gate_out = 1.0 + torch.tanh(fused.output_gate_scale.to(dtype=dt) * gl)

        T_stack = fused.template_bank.to(dtype=dt)
        n_chunks = (T_len + ch - 1) // ch
        pad = n_chunks * ch - T_len

        x_p = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        h_p = F.pad(h, (0, 0, 0, pad)) if pad > 0 else h

        x_anchors = x_p.reshape(B_, n_chunks, ch, C_)[:, :, 0, :].float()
        logits_all = x_anchors @ fused.template_route.float()
        weights_all = F.softmax(logits_all, dim=-1).to(dt)

        # ── TRITON KERNEL replaces matmul+bmm ──
        BN_ = B_ * n_chunks
        h_chunked = h_p.reshape(BN_, ch, B_sz_)
        pre_output = fused_template_mix(h_chunked, weights_all.reshape(BN_, K_), T_stack)
        pre_output = pre_output.reshape(B_, n_chunks * ch, O_)[:, :T_len, :]

        if gate_out is not None:
            pre_output = pre_output * gate_out
        return (pre_output + fused.bias.to(dtype=dt)).to(dtype=dt)

    # Warmup
    n_iters = 300
    fused.eval()
    for _ in range(20):
        with torch.no_grad():
            _orig_forward(x, ctx)
            _triton_forward(x, ctx)
    torch.cuda.synchronize()

    # Time: standard fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _orig_forward(x, ctx)
    torch.cuda.synchronize()
    t_std = (time.perf_counter() - t0) / n_iters

    # Time: triton fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _triton_forward(x, ctx)
    torch.cuda.synchronize()
    t_tri = (time.perf_counter() - t0) / n_iters

    # Time: compiled fused
    fused_c = copy.deepcopy(fused).compile(mode='default', fullgraph=True)
    fused_c.eval()
    for _ in range(5):
        with torch.no_grad():
            fused_c(x, ctx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            fused_c(x, ctx)
    torch.cuda.synchronize()
    t_comp = (time.perf_counter() - t0) / n_iters

    # Correctness
    with torch.no_grad():
        y_std = _orig_forward(x, ctx)
        y_tri = _triton_forward(x, ctx)
    err = (y_std - y_tri).abs().max().item()

    print(f"    Correctness: max_abs={err:.2e}  {'✓' if err < 1e-2 else '✗'}")
    print(f"    Fused (matmul+bmm):  {t_std*1000:.3f} ms")
    print(f"    Fused+Compiled:      {t_comp*1000:.3f} ms  ({t_std/t_comp:.2f}x vs fused)")
    print(f"    Triton kernel:       {t_tri*1000:.3f} ms  ({t_std/t_tri:.2f}x vs fused, "
          f"{t_comp/t_tri:.2f}x vs compiled)")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(" Approach 3: Triton Fused Template Assembly + Chunk Mixing")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for Triton benchmarks")
        return 1

    dev = torch.cuda.get_device_name(0)
    print(f"  Device: {dev}")
    print(f"  Triton: {triton.__version__}")

    # ── Kernel-only benchmarks ──
    print("\n" + "─" * 70)
    print(" Kernel-level: template assembly + chunk mixing only")
    print("─" * 70)

    # d4-like: BN=B*n_chunks, S=chunk, B_sz=64, O=256, K=8
    run_benchmark("d4-like  (B=4, T=256, chunk=64)", BN=16, S=64, B_sz=64, O=256, K=8)
    run_benchmark("d8-like  (B=2, T=256, chunk=64)", BN=8,  S=64, B_sz=128, O=512, K=8)
    run_benchmark("d12-like (B=2, T=256, chunk=64)", BN=8,  S=64, B_sz=192, O=768, K=8)
    run_benchmark("d12 c_fc (B=2, T=256, chunk=64)", BN=8,  S=64, B_sz=192, O=3072, K=8)

    # ── End-to-end layer benchmarks ──
    print("\n" + "─" * 70)
    print(" End-to-end: full RemixedLinearFused layer")
    print("─" * 70)

    test_e2e_replacement(in_f=256, out_f=256, ctx_dim=64, basis=64,
                         K=8, chunk=64, B=4, T=256)
    test_e2e_replacement(in_f=768, out_f=768, ctx_dim=128, basis=192,
                         K=8, chunk=64, B=2, T=256)

    print("\n" + "=" * 70)
    print(" Benchmark complete")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
