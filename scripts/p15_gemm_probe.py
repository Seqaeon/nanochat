#!/usr/bin/env python3
"""P15: what is cuBLAS actually doing on MST's block-diagonal GEMMs, and can anything beat it?

WHY THIS EXISTS
    MST reaches ~0.46 of dense's utilization on an H200. Every explanation offered so far
    has been inference from shapes rather than measurement of kernels:

      - "it is bandwidth"      -> ruled out: MST sits at 302 FLOP/byte, the H200 ridge is
                                  206, so both arms are compute-bound.
      - "raise the batch"      -> ruled out: asymptotic intensity is K*N/(K+N), fixed at a
                                  0.25 ratio by the shapes alone, independent of M.
      - "wave quantization"    -> untested. MST emits the same tile COUNT as dense.
      - "short K starves the
         software pipeline"    -> untested, and the leading hypothesis. A cuBLAS kernel
                                  with 4-6 pipeline stages needs that many K-steps to fill.
                                  MST's K=384 with BLOCK_K=128 gives THREE.

    Guessing stops here. This script names the kernel cuBLAS selects, parses its tile shape
    out of the name, computes wave occupancy from the real SM count, and races four
    alternatives against it on the exact shapes MST uses.

WHAT IT TESTS, SEPARATELY
    bmm-transpose   torch.bmm on a runtime-transposed weight. What _batched_linear does now.
    bmm-pretransp   the same with the weight already stored K-major, so no transpose view.
                    Isolates whether the transpose changes cuBLAS's kernel choice.
    mm-loop         a Python loop of torch.mm, one per stream.
    grouped_mm      torch._grouped_mm where available.
    compiled        torch.compile default, i.e. whatever Inductor lowers bmm to.
    max-autotune    torch.compile(mode="max-autotune-no-cudagraphs"), which autotunes Triton
                    GEMM templates over BLOCK_K and num_stages. This is the direct test of
                    the short-K-pipeline hypothesis: if a shallower, finer-K config wins,
                    the pipeline is the problem and a custom kernel is worth writing.
    triton-sweep    an explicit Triton batched matmul swept over BLOCK_M/N/K, stages and
                    warps, reporting the BEST config found. Tells you the headroom a
                    hand-written kernel (CUTLASS Stream-K or otherwise) would be chasing,
                    without building one.

INTERPRETING IT
    If max-autotune or triton-sweep beats cuBLAS by a wide margin, the gap is a kernel
    selection problem and worth engineering. If nothing beats cuBLAS, the gap is intrinsic
    to the shape and the paper should say so rather than promise future kernels.

Usage:
    python -m scripts.p15_gemm_probe                       # L=24, N=4, the paper's config
    python -m scripts.p15_gemm_probe --depth 32 --batch 8
    python -m scripts.p15_gemm_probe --no-triton-sweep     # skip the slow part
"""

import argparse
import re

import torch

ASPECT, HEAD_DIM = 64, 128


def model_dim(depth):
    return ((depth * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def peak_tflops(name):
    for key, tf in (("B200", 2250.0), ("H200", 989.0), ("H100", 989.0),
                    ("A100", 312.0), ("L40", 181.0), ("4090", 165.2), ("3050", 45.0)):
        if key in name.upper():
            return tf
    return None


def bench(fn, warm=8, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def kernel_name(fn):
    """The CUDA kernel that actually ran, by self device time. This is how you see
    cuBLAS's heuristic choice: names like nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT
    encode the tile shape (256x128) and the pipeline configuration (64x4)."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    if not evs:
        return "?"
    return max(evs, key=lambda e: e.self_device_time_total).key


def parse_tile(name):
    """Pull an MxN tile out of a cuBLAS/CUTLASS kernel name, if it has one."""
    m = re.search(r"(\d+)x(\d+)_(\d+)x(\d+)", name)
    if m:
        return f"{m.group(1)}x{m.group(2)}", f"{m.group(3)}x{m.group(4)}"
    m = re.search(r"(\d+)x(\d+)", name)
    return (f"{m.group(1)}x{m.group(2)}", "-") if m else ("-", "-")


# --------------------------------------------------------------------- Triton
TRITON_OK = True
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _bmm_kernel(A, B, C, G, M, N, K,
                    sam, sak, sbk, sbn, scm, scn, sa_g, sb_g, sc_g,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        g = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptr = A + g * sa_g + offs_m[:, None] * sam + offs_k[None, :] * sak
        b_ptr = B + g * sb_g + offs_k[:, None] * sbk + offs_n[None, :] * sbn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptr, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptr, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptr += BLOCK_K * sak
            b_ptr += BLOCK_K * sbk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptr = C + g * sc_g + offs_cm[:, None] * scm + offs_cn[None, :] * scn
        tl.store(c_ptr, acc.to(C.dtype.element_ty),
                 mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    def triton_bmm(a, b, c, BM, BN, BK, GM, stages, warps):
        G, M, K = a.shape
        N = b.shape[2]
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), G)
        _bmm_kernel[grid](a, b, c, G, M, N, K,
                          a.stride(1), a.stride(2), b.stride(1), b.stride(2),
                          c.stride(1), c.stride(2), a.stride(0), b.stride(0), c.stride(0),
                          BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                          num_stages=stages, num_warps=warps)
except Exception:
    TRITON_OK = False


def sweep_triton(a, b, peak, flops, verbose=False):
    """Best Triton config found, and whether a short-K-friendly one wins.

    The point is not to ship this kernel. It is to bound how much a hand-written
    kernel could recover, so the decision to write one is made on evidence.
    """
    G, M, K = a.shape
    N = b.shape[2]
    c = torch.empty((G, M, N), device=a.device, dtype=a.dtype)
    ref = torch.bmm(a, b)
    best = None
    cfgs = [(bm, bn, bk, 8, st, wp)
            for bm in (64, 128)
            for bn in (64, 128, 256)
            for bk in (32, 64, 128)
            for st in (2, 3, 4, 5)
            for wp in (4, 8)]
    for (bm, bn, bk, gm, st, wp) in cfgs:
        if bk > K:
            continue
        try:
            triton_bmm(a, b, c, bm, bn, bk, gm, st, wp)
            torch.cuda.synchronize()
            if not torch.allclose(c, ref, atol=2e-2, rtol=2e-2):
                continue
            ms = bench(lambda: triton_bmm(a, b, c, bm, bn, bk, gm, st, wp), warm=3, iters=10)
            tf = flops / (ms * 1e-3) / 1e12
            if best is None or tf > best[0]:
                best = (tf, ms, (bm, bn, bk, st, wp))
        except Exception:
            continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--n-subs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="device micro-batch")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--triton-sweep", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    pk = peak_tflops(props.name)
    SMs = props.multi_processor_count
    M = args.batch * args.seq
    D = model_dim(args.depth)
    d = D // args.n_subs
    print(f"device: {props.name}  {SMs} SMs" + (f"  peak bf16 {pk} TFLOP/s" if pk else ""))
    print(f"L={args.depth}  d_model={D}  n_subs={args.n_subs}  sub_dim={d}  M={M:,} "
          f"(batch {args.batch} x seq {args.seq})\n")

    # The two GEMMs that dominate: FFN up (d -> mult*d) and FFN down (mult*d -> d).
    inner = int(args.ffn_mult * d)
    dense_inner = int(args.ffn_mult * D)
    shapes = [
        ("FFN up   dense", 1, M, D, dense_inner),
        ("FFN up   MST", args.n_subs, M, d, inner),
        ("FFN down dense", 1, M, dense_inner, D),
        ("FFN down MST", args.n_subs, M, inner, d),
    ]
    for label, G, Mm, K, N in shapes:
        flops = 2 * G * Mm * K * N
        a = torch.randn(G, Mm, K, device=dev, dtype=torch.bfloat16)
        w_kmajor = torch.randn(G, K, N, device=dev, dtype=torch.bfloat16)     # pre-transposed
        w_nmajor = w_kmajor.transpose(1, 2).contiguous()                      # as stored today
        tiles_m, tiles_n = -(-Mm // 128), -(-N // 128)
        tiles = G * tiles_m * tiles_n
        waves = -(-tiles // SMs)
        print(f"── {label}   G={G} M={Mm} K={K} N={N}   {flops/1e12:.2f} TFLOP")
        print(f"   tiles(128x128) {tiles:,}  waves {waves}  "
              f"wave efficiency {tiles/(waves*SMs)*100:.0f}%  "
              f"K-steps per tile at BLOCK_K=128: {-(-K//128)}")
        variants = {
            "bmm-transpose": lambda: torch.bmm(a, w_nmajor.transpose(1, 2)),
            "bmm-pretransp": lambda: torch.bmm(a, w_kmajor),
            "mm-loop": lambda: [torch.mm(a[i], w_kmajor[i]) for i in range(G)],
        }
        if hasattr(torch, "_grouped_mm"):
            try:
                torch._grouped_mm(a, w_kmajor)
                variants["grouped_mm"] = lambda: torch._grouped_mm(a, w_kmajor)
            except Exception:
                pass
        base = None
        for vname, fn in variants.items():
            try:
                ms = bench(fn)
            except Exception as ex:
                print(f"   {vname:<16} FAILED {type(ex).__name__}")
                continue
            tf = flops / (ms * 1e-3) / 1e12
            base = base or tf
            kn = kernel_name(fn)
            tile, stage = parse_tile(kn)
            print(f"   {vname:<16}{ms:8.3f} ms{tf:8.1f} TF/s"
                  + (f"{100*tf/pk:6.1f}%" if pk else "      ")
                  + f"{tf/base:7.3f}x   tile {tile:<9} cfg {stage:<6} {kn[:44]}")
        for mode in ("default", "max-autotune-no-cudagraphs"):
            try:
                torch._dynamo.reset()
                f = torch.compile(lambda x, y: torch.bmm(x, y), mode=mode)
                ms = bench(lambda: f(a, w_kmajor), warm=10)
                tf = flops / (ms * 1e-3) / 1e12
                print(f"   {('compiled-' + mode.split('-')[0]):<16}{ms:8.3f} ms{tf:8.1f} TF/s"
                      + (f"{100*tf/pk:6.1f}%" if pk else "      ")
                      + f"{tf/base:7.3f}x")
            except Exception as ex:
                print(f"   compiled-{mode:<20} FAILED {type(ex).__name__}")
        if args.triton_sweep and TRITON_OK:
            bestt = sweep_triton(a, w_kmajor, pk, flops)
            if bestt:
                tf, ms, cfg = bestt
                print(f"   {'triton-best':<16}{ms:8.3f} ms{tf:8.1f} TF/s"
                      + (f"{100*tf/pk:6.1f}%" if pk else "      ")
                      + f"{tf/base:7.3f}x   BLOCK_M/N/K={cfg[0]}/{cfg[1]}/{cfg[2]} "
                        f"stages={cfg[3]} warps={cfg[4]}")
            else:
                print(f"   {'triton-best':<16} no config passed the correctness check")
        elif args.triton_sweep:
            print(f"   {'triton-best':<16} triton unavailable")
        print()
        del a, w_kmajor, w_nmajor
        torch.cuda.empty_cache()

    print("HOW TO READ THIS")
    print("  * Compare 'FFN up MST' against 'FFN up dense' on TF/s: that ratio, not the")
    print("    model-level MFU, is the pure GEMM penalty.")
    print("  * bmm-pretransp vs bmm-transpose isolates the weight-layout question.")
    print("  * If max-autotune or triton-best beats bmm by a wide margin, the gap is kernel")
    print("    selection and a custom kernel is worth writing. If not, it is intrinsic to")
    print("    the shape and no amount of engineering will close it.")
    print("  * A winning triton config with small BLOCK_K and few stages confirms the")
    print("    short-K-pipeline hypothesis specifically.")


if __name__ == "__main__":
    main()
