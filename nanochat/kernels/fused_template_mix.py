"""
Fused RemixedLinear template-mixed GEMM kernel in Triton.

Eliminates the intermediate W_eff = Σ α_k T_k materialization by computing
y = Σ_k α_k (T_k @ h) directly in a single kernel, where h = LN(W_b @ x).

This avoids:
  - Materializing W_eff (d_out × B elements per chunk)
  - K separate matmuls or a weighted sum followed by a matmul
  - Multiple kernel launches per RemixedLinear module

The kernel processes ALL chunks in parallel via the grid's first dimension,
handling the per-chunk varying routing weights α naturally.

Expected speedup: 1.5-2× for RemixedLinear forward pass by:
  1. Eliminating W_eff write/read (saves d_out × B × 2 bytes per chunk)
  2. Loading h once per B-tile instead of K times
  3. Reducing kernel launches from ~6 to 1 per RemixedLinear module
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_template_mix_fwd_kernel(
        # Pointers
        H_ptr,       # (num_chunks, chunk_size, B) — LN'd basis-projected input
        T_ptr,       # (K, d_out, B) — template bank (contiguous)
        Alpha_ptr,   # (num_chunks, K) — routing weights per chunk
        Y_ptr,       # (num_chunks, chunk_size, d_out) — output
        # Dimensions
        num_chunks,
        chunk_size,
        d_out,
        B: tl.constexpr,
        K: tl.constexpr,
        # Strides for H (num_chunks, chunk_size, B)
        stride_hc, stride_ht, stride_hb,
        # Strides for T (K, d_out, B)
        stride_tk, stride_tn, stride_tb,
        # Strides for Alpha (num_chunks, K)
        stride_ac, stride_ak,
        # Strides for Y (num_chunks, chunk_size, d_out)
        stride_yc, stride_yt, stride_yn,
        # Block sizes
        BLOCK_M: tl.constexpr,   # tile over chunk_size (tokens)
        BLOCK_N: tl.constexpr,   # tile over d_out
        BLOCK_K: tl.constexpr,   # tile over B (reduction dim)
    ):
        """
        Computes y[c, t, n] = Σ_k α[c,k] * Σ_b T[k,n,b] * h[c,t,b]
        for all chunks c, tokens t, output dims n.

        Grid: (num_chunks, cdiv(chunk_size, BLOCK_M), cdiv(d_out, BLOCK_N))
        """
        pid_c = tl.program_id(0)   # chunk index
        pid_m = tl.program_id(1)   # token tile index
        pid_n = tl.program_id(2)   # output dim tile index

        # Token and output dim ranges for this tile
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm < chunk_size
        mask_n = rn < d_out

        # Accumulator in float32 for numerical stability
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Base pointers for this chunk
        h_base = H_ptr + pid_c * stride_hc
        a_base = Alpha_ptr + pid_c * stride_ac

        # Outer loop over B (reduction dim) — h is loaded ONCE per B-tile
        for b_start in range(0, B, BLOCK_K):
            rb = b_start + tl.arange(0, BLOCK_K)
            mask_b = rb < B

            # Load h[c, rm, rb] — shape (BLOCK_M, BLOCK_K)
            h_tile = tl.load(
                h_base + rm[:, None] * stride_ht + rb[None, :] * stride_hb,
                mask=mask_m[:, None] & mask_b[None, :],
                other=0.0,
            )

            # Inner loop over K templates — h_tile stays in registers
            for k in range(K):
                alpha_k = tl.load(a_base + k * stride_ak)

                # Load T[k, rn, rb] — shape (BLOCK_N, BLOCK_K)
                t_tile = tl.load(
                    T_ptr + k * stride_tk + rn[:, None] * stride_tn + rb[None, :] * stride_tb,
                    mask=mask_n[:, None] & mask_b[None, :],
                    other=0.0,
                )

                # acc += α_k * h_tile @ t_tile^T
                # h_tile: (BLOCK_M, BLOCK_K), t_tile: (BLOCK_N, BLOCK_K)
                # result: (BLOCK_M, BLOCK_N)
                acc += alpha_k * tl.dot(h_tile, tl.trans(t_tile))

        # Store y[c, rm, rn]
        tl.store(
            Y_ptr + pid_c * stride_yc + rm[:, None] * stride_yt + rn[None, :] * stride_yn,
            acc.to(Y_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


    @triton.jit
    def _fused_template_mix_bwd_dh_kernel(
        # Backward kernel: compute grad_h = Σ_k α_k * grad_y @ T_k
        # (transpose of forward: instead of T_k @ h, we do grad_y @ T_k in the other direction)
        GradY_ptr,   # (num_chunks, chunk_size, d_out)
        T_ptr,       # (K, d_out, B)
        Alpha_ptr,   # (num_chunks, K)
        GradH_ptr,   # (num_chunks, chunk_size, B) — output
        # Dimensions
        num_chunks,
        chunk_size,
        d_out,
        B: tl.constexpr,
        K: tl.constexpr,
        # Strides for GradY
        stride_gyc, stride_gyt, stride_gyn,
        # Strides for T
        stride_tk, stride_tn, stride_tb,
        # Strides for Alpha
        stride_ac, stride_ak,
        # Strides for GradH
        stride_ghc, stride_ght, stride_ghb,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        grad_h[c, t, b] = Σ_k α[c,k] * Σ_n grad_y[c,t,n] * T[k,n,b]

        Grid: (num_chunks, cdiv(chunk_size, BLOCK_M), cdiv(B, BLOCK_N))
        """
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)   # token tile
        pid_n = tl.program_id(2)   # B tile (output of this kernel)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # indices into B
        mask_m = rm < chunk_size
        mask_n = rn < B

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        gy_base = GradY_ptr + pid_c * stride_gyc
        a_base = Alpha_ptr + pid_c * stride_ac

        # Tile over d_out (reduction dim for backward)
        for n_start in range(0, d_out, BLOCK_K):
            rk = n_start + tl.arange(0, BLOCK_K)
            mask_k = rk < d_out

            # Load grad_y[c, rm, rk]
            gy_tile = tl.load(
                gy_base + rm[:, None] * stride_gyt + rk[None, :] * stride_gyn,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            for k in range(K):
                alpha_k = tl.load(a_base + k * stride_ak)
                # Load T[k, rk, rn] — note: T is (K, d_out, B)
                t_tile = tl.load(
                    T_ptr + k * stride_tk + rk[:, None] * stride_tn + rn[None, :] * stride_tb,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                # acc += α_k * gy_tile @ t_tile
                acc += alpha_k * tl.dot(gy_tile, t_tile)

        tl.store(
            GradH_ptr + pid_c * stride_ghc + rm[:, None] * stride_ght + rn[None, :] * stride_ghb,
            acc.to(GradH_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


# ─── PyTorch wrappers ────────────────────────────────────────────────────────


def _naive_template_mix(h: torch.Tensor, T_bank: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference implementation: y = Σ_k α_k (T_k @ h^T) for each chunk.

    Args:
        h:      (num_chunks, chunk_size, B) — LN'd basis-projected input
        T_bank: (K, d_out, B) — template bank
        alpha:  (num_chunks, K) — routing weights
    Returns:
        y:      (num_chunks, chunk_size, d_out)
    """
    # W_eff[c] = Σ_k α[c,k] * T_k → (num_chunks, d_out, B)
    W_eff = torch.einsum('ck, knb -> cnb', alpha, T_bank)
    # y[c] = h[c] @ W_eff[c]^T → (num_chunks, chunk_size, d_out)
    y = torch.bmm(h, W_eff.transpose(1, 2))
    return y


class FusedTemplateMix(torch.autograd.Function):
    """Autograd function for the fused template-mixed GEMM.

    Forward:  y = Σ_k α_k (T_k @ h^T)  — fused, no W_eff materialization
    Backward: grad_h via fused kernel, grad_alpha and grad_T via PyTorch
    """

    @staticmethod
    def forward(ctx, h, T_bank, alpha):
        """
        Args:
            h:      (num_chunks, chunk_size, B)
            T_bank: (K, d_out, B) — contiguous
            alpha:  (num_chunks, K)
        Returns:
            y:      (num_chunks, chunk_size, d_out)
        """
        assert HAS_TRITON, "Triton not available"
        num_chunks, chunk_size, B_dim = h.shape
        K, d_out, B2 = T_bank.shape
        assert B_dim == B2

        # Ensure contiguous
        h = h.contiguous()
        T_bank = T_bank.contiguous()
        alpha = alpha.contiguous()

        # Output
        y = torch.empty(num_chunks, chunk_size, d_out, device=h.device, dtype=h.dtype)

        # Block sizes (tune these for your GPU)
        BLOCK_M = min(64, chunk_size)
        BLOCK_N = min(64, d_out)
        BLOCK_K = min(32, B_dim)

        grid = (
            num_chunks,
            triton.cdiv(chunk_size, BLOCK_M),
            triton.cdiv(d_out, BLOCK_N),
        )

        _fused_template_mix_fwd_kernel[grid](
            h, T_bank, alpha, y,
            num_chunks, chunk_size, d_out, B_dim, K,
            h.stride(0), h.stride(1), h.stride(2),
            T_bank.stride(0), T_bank.stride(1), T_bank.stride(2),
            alpha.stride(0), alpha.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(h, T_bank, alpha)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        h, T_bank, alpha = ctx.saved_tensors
        num_chunks, chunk_size, B_dim = h.shape
        K, d_out, _ = T_bank.shape

        grad_y = grad_y.contiguous()

        # 1. grad_h via fused kernel: grad_h = Σ_k α_k * grad_y @ T_k
        grad_h = torch.empty_like(h)
        BLOCK_M = min(64, chunk_size)
        BLOCK_N = min(64, B_dim)
        BLOCK_K = min(32, d_out)

        grid = (
            num_chunks,
            triton.cdiv(chunk_size, BLOCK_M),
            triton.cdiv(B_dim, BLOCK_N),
        )
        _fused_template_mix_bwd_dh_kernel[grid](
            grad_y, T_bank, alpha, grad_h,
            num_chunks, chunk_size, d_out, B_dim, K,
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
            T_bank.stride(0), T_bank.stride(1), T_bank.stride(2),
            alpha.stride(0), alpha.stride(1),
            grad_h.stride(0), grad_h.stride(1), grad_h.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2. grad_T_k = α_k * grad_y^T @ h (per chunk, summed across chunks)
        # grad_T[k, n, b] = Σ_c α[c,k] * Σ_t grad_y[c,t,n] * h[c,t,b]
        # = Σ_c α[c,k] * (grad_y[c]^T @ h[c])[n, b]
        # Use einsum for clarity; this is not the bottleneck
        grad_T = torch.einsum('ck, ctn, ctb -> knb', alpha, grad_y.float(), h.float()).to(T_bank.dtype)

        # 3. grad_alpha[c, k] = Σ_t Σ_n grad_y[c,t,n] * (T_k @ h[c,t,:])_n
        # = Σ_t (grad_y[c,t,:] · (T_k @ h[c,t,:]))
        # = trace(grad_y[c]^T @ h[c] @ T_k^T)  per chunk
        # Compute T_k @ h^T for each k and chunk, then dot with grad_y
        # h @ T_k^T → (num_chunks, chunk_size, d_out), then elementwise with grad_y
        grad_alpha = torch.einsum('ctn, knb, ctb -> ck', grad_y.float(), T_bank.float(), h.float()).to(alpha.dtype)

        return grad_h, grad_T, grad_alpha


def fused_template_mix(h: torch.Tensor, T_bank: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for naive template mixing.

    Args:
        h:      (num_chunks, chunk_size, B) — LN'd basis-projected input
        T_bank: (K, d_out, B) — template bank
        alpha:  (num_chunks, K) — routing weights per chunk
    Returns:
        y:      (num_chunks, chunk_size, d_out)
    """
    if HAS_TRITON and h.is_cuda:
        return FusedTemplateMix.apply(h, T_bank, alpha)
    else:
        return _naive_template_mix(h, T_bank, alpha)


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark():
    """Compare fused vs naive template mixing."""
    import time

    device = 'cuda'
    dtype = torch.bfloat16

    configs = [
        # (num_chunks, chunk_size, B, d_out, K)
        ("d4",  512, 64, 256, 256, 8),
        ("d8",  256, 64, 512, 512, 8),
        ("d12", 128, 64, 768, 768, 8),
    ]

    for name, nc, cs, B, d_out, K in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name} — chunks={nc}, chunk_size={cs}, B={B}, d_out={d_out}, K={K}")
        print(f"{'='*60}")

        h = torch.randn(nc, cs, B, device=device, dtype=dtype)
        T_bank = torch.randn(K, d_out, B, device=device, dtype=dtype)
        alpha = torch.softmax(torch.randn(nc, K, device=device, dtype=torch.float32), dim=-1).to(dtype)

        # Correctness check
        y_naive = _naive_template_mix(h, T_bank, alpha)
        if HAS_TRITON:
            y_fused = FusedTemplateMix.apply(h, T_bank, alpha)
            max_diff = (y_naive - y_fused).abs().max().item()
            rel_err = max_diff / (y_naive.abs().mean().item() + 1e-8)
            print(f"  Max abs diff:  {max_diff:.6f}")
            print(f"  Rel error:     {rel_err:.6f}")
        else:
            print("  Triton not available, skipping fused kernel")

        # Timing
        warmup, measure = 10, 50

        # Naive
        for _ in range(warmup):
            _ = _naive_template_mix(h, T_bank, alpha)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            _ = _naive_template_mix(h, T_bank, alpha)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) / measure * 1000
        print(f"  Naive:  {naive_ms:.3f} ms")

        if HAS_TRITON:
            for _ in range(warmup):
                _ = FusedTemplateMix.apply(h, T_bank, alpha)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(measure):
                _ = FusedTemplateMix.apply(h, T_bank, alpha)
            torch.cuda.synchronize()
            fused_ms = (time.perf_counter() - t0) / measure * 1000
            print(f"  Fused:  {fused_ms:.3f} ms")
            print(f"  Speedup: {naive_ms / fused_ms:.2f}x")

        # Backward correctness (if Triton available)
        if HAS_TRITON:
            h_n = h.clone().requires_grad_(True)
            T_n = T_bank.clone().requires_grad_(True)
            a_n = alpha.clone().requires_grad_(True)
            y_n = _naive_template_mix(h_n, T_n, a_n)
            y_n.sum().backward()

            h_f = h.clone().requires_grad_(True)
            T_f = T_bank.clone().requires_grad_(True)
            a_f = alpha.clone().requires_grad_(True)
            y_f = FusedTemplateMix.apply(h_f, T_f, a_f)
            y_f.sum().backward()

            print(f"  Backward grad_h diff:     {(h_n.grad - h_f.grad).abs().max().item():.6f}")
            print(f"  Backward grad_T diff:     {(T_n.grad - T_f.grad).abs().max().item():.6f}")
            print(f"  Backward grad_alpha diff: {(a_n.grad - a_f.grad).abs().max().item():.6f}")


if __name__ == "__main__":
    benchmark()
