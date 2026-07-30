import time
import torch
import torch.nn as nn
from nanochat.gpt import RemixedLinear, RemixedLinearFused

def benchmark_d12():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping benchmark.")
        return

    print("=" * 80)
    print(" Diagnostic Benchmark: RemixedLinear vs RemixedLinearFused at Depth 12 (d12)")
    print("=" * 80)

    # d12 parameters (aspect_ratio = 64 => model_dim = 768, ffn_dim = 3072)
    in_f = 768
    out_f = 3072  # FFN c_fc
    ctx_dim = 768
    basis_sz = 768
    K = 8
    chunk_sz = 64

    kw = dict(
        n_templates=K,
        chunk_routing_size=chunk_sz,
        use_basis_gate=False,
        use_output_gate=True,
        use_context=True,
        output_gate_rank=16,
        basis_gate_mode='centered',
        template_routing_learned=True,
        template_topk=0,
    )

    print(f"Layer config: in={in_f}, out={out_f}, basis={basis_sz}, K={K}, chunk={chunk_sz}")
    print(f"W_eff per-chunk matrix size: {out_f} x {basis_sz} = {out_f * basis_sz:,} elements")
    print("-" * 80)

    for B in [1, 2, 4, 8]:
        T = 2048
        n_chunks = T // chunk_sz
        BN = B * n_chunks
        w_eff_mb = (BN * out_f * basis_sz * 2) / (1024 * 1024)

        print(f"\n--- Batch Size B = {B} (BN = {BN} chunks, seq_len = {T}) ---")
        print(f"  Explicit W_eff_flat size in RAM: {w_eff_mb:.1f} MB per FFN layer")

        x = torch.randn(B, T, in_f, device=device, dtype=torch.bfloat16)
        ctx = torch.randn(B, T, ctx_dim, device=device, dtype=torch.bfloat16)

        # 1. RemixedLinear (Original)
        orig = RemixedLinear(in_f, out_f, context_dim=ctx_dim, basis_size=basis_sz,
                              remixed_linear_kwargs=kw, scale_basis=False).to(device, torch.bfloat16)
        
        # Test Orig Eager
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            orig.train()
            for _ in range(3):
                y = orig(x, ctx)
                y.sum().backward()
                orig.zero_grad()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(10):
                y = orig(x, ctx)
                y.sum().backward()
                orig.zero_grad()
            torch.cuda.synchronize()
            t_orig = (time.perf_counter() - t0) / 10 * 1000
            mem_orig = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  [1. Orig Eager]        F+B: {t_orig:6.2f} ms | Peak VRAM: {mem_orig:6.1f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  [1. Orig Eager]        F+B:   OOM    | Peak VRAM:   OOM")
            t_orig = 0
            torch.cuda.empty_cache()

        # 2. RemixedLinearFused Eager
        fused = RemixedLinearFused.from_remixed_linear(orig).to(device, torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            fused.train()
            for _ in range(3):
                y = fused(x, ctx)
                y.sum().backward()
                fused.zero_grad()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(10):
                y = fused(x, ctx)
                y.sum().backward()
                fused.zero_grad()
            torch.cuda.synchronize()
            t_fused = (time.perf_counter() - t0) / 10 * 1000
            mem_fused = torch.cuda.max_memory_allocated() / (1024 * 1024)
            sp_fused = t_orig / t_fused if t_orig > 0 else 0
            print(f"  [2. Fused Eager]       F+B: {t_fused:6.2f} ms | Peak VRAM: {mem_fused:6.1f} MB | Speedup: {sp_fused:.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"  [2. Fused Eager]       F+B:   OOM    | Peak VRAM:   OOM")
            t_fused = 0
            torch.cuda.empty_cache()

        # 3. RemixedLinear Compiled
        orig_c = torch.compile(orig, mode='default')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            orig_c.train()
            for _ in range(3):
                y = orig_c(x, ctx)
                y.sum().backward()
                orig_c.zero_grad()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(10):
                y = orig_c(x, ctx)
                y.sum().backward()
                orig_c.zero_grad()
            torch.cuda.synchronize()
            t_orig_c = (time.perf_counter() - t0) / 10 * 1000
            mem_orig_c = torch.cuda.max_memory_allocated() / (1024 * 1024)
            sp_orig_c = t_orig / t_orig_c if t_orig > 0 else 0
            print(f"  [3. Orig Compiled]     F+B: {t_orig_c:6.2f} ms | Peak VRAM: {mem_orig_c:6.1f} MB | Speedup: {sp_orig_c:.2f}x")
        except Exception as e:
            print(f"  [3. Orig Compiled]     Failed: {e}")
            t_orig_c = 0
            torch.cuda.empty_cache()

        # 4. RemixedLinearFused Compiled
        fused_c = torch.compile(fused, mode='default')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            fused_c.train()
            for _ in range(3):
                y = fused_c(x, ctx)
                y.sum().backward()
                fused_c.zero_grad()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(10):
                y = fused_c(x, ctx)
                y.sum().backward()
                fused_c.zero_grad()
            torch.cuda.synchronize()
            t_fused_c = (time.perf_counter() - t0) / 10 * 1000
            mem_fused_c = torch.cuda.max_memory_allocated() / (1024 * 1024)
            sp_fused_c = t_orig / t_fused_c if t_orig > 0 else 0
            print(f"  [4. Fused Compiled]    F+B: {t_fused_c:6.2f} ms | Peak VRAM: {mem_fused_c:6.1f} MB | Speedup: {sp_fused_c:.2f}x")
        except Exception as e:
            print(f"  [4. Fused Compiled]    Failed: {e}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    benchmark_d12()
