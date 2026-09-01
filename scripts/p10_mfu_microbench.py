"""Isolate where MST loses wall clock to the dense baseline.

Pure GEMM shapes. No flash_attn, no model, no data. Runs anywhere with a CUDA
device, so it can be pointed at the training GPU without a checkout of the run
environment.

Motivation. The A1-A5 harness showed MST at 0.22x dense throughput in training
but only 1.16x slower than dense in forward-only prefill at large T. Forward-only
parity plus training non-parity localises the loss to the backward pass, and the
specific suspect is the weight-gradient GEMM: for a block-diagonal layer it has a
d x d output per stream, which is a handful of tiles on a 132-SM device, while the
reduction dimension M is enormous and wants split-K that cuBLAS is reluctant to
apply inside a batched call.

Four questions, in the order that decides what to do next.

  A  Is MST's forward GEMM throughput simply "dense at width d"? If yes, no
     kernel engineering on the forward path can help and the only lever is d.

  B  Is the weight-gradient GEMM the hole? Compares one batched call against a
     loop of unbatched calls (which lets cuBLAS pick split-K per GEMM) and
     against torch._grouped_mm where available.

  C  What does the (B, T, N, d) layout cost? _batched_linear permutes to
     (N, B*T, d), which is non-contiguous, so the reshape materialises a full
     copy of the activations, and the result is handed back non-contiguous so the
     next op copies again. This prices carrying the stream axis leading instead.

  D  THE DECISIVE ONE. Would running the block-diagonal layer as a masked DENSE
     GEMM at width D be faster in wall clock than the batched block-diagonal one,
     despite doing N times the FLOPs? Block-diagonal only pays when

         throughput(d) / throughput(D)  >  1/N

     If D reports a ratio below 1, the structure is a net wall-clock loss at this
     size and the paper's speed claim has to be about the crossover, not about a
     measured win.

Usage:
    python -m scripts.p10_mfu_microbench                  # D=768 N=4 M=16384
    python -m scripts.p10_mfu_microbench --model-dim 2048 --tokens 32768
    python -m scripts.p10_mfu_microbench --sweep          # the depth ladder
"""

import argparse
import torch


# Per-layer GEMM shapes of an MST stream, as (name, in_mult, out_mult) in units of
# the stream width d. Mirrors BatchedMSTLayer: three qkv projections at d -> qkv_dim
# (qkv_dim == d whenever mst_sub_head_dim divides d evenly, which G1 arranges), then
# the FFN pair. The attention output projection is deliberately absent: the headline
# arm runs --mst-wo-mode dense, so that one is already a full D x D dense GEMM and is
# not part of the block-diagonal cost.
LAYER_SHAPES = [
    ("q", 1, 1),
    ("k", 1, 1),
    ("v", 1, 1),
    ("fc", 1, 4),
    ("fc_proj", 4, 1),
]


def _time(fn, iters=50, warmup=20):
    """Median-of-iters wall clock in ms, via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


# ─────────────────────────────────────────────────────────────── A: forward width

def bench_forward(D, N, M, dtype, dev):
    """Block-diagonal forward against the dense GEMM it replaces."""
    d = D // N
    rows = []
    for name, im, om in LAYER_SHAPES:
        d_in, d_out = im * d, om * d
        D_in, D_out = im * D, om * D

        xb = torch.randn(N, M, d_in, device=dev, dtype=dtype)
        wb = torch.randn(N, d_in, d_out, device=dev, dtype=dtype)
        f_blk = 2 * N * M * d_in * d_out
        t_blk = _time(lambda: torch.bmm(xb, wb))

        xd = torch.randn(M, D_in, device=dev, dtype=dtype)
        wd = torch.randn(D_in, D_out, device=dev, dtype=dtype)
        f_dns = 2 * M * D_in * D_out
        t_dns = _time(lambda: torch.mm(xd, wd))

        rows.append((name, d_in, d_out, t_blk, _tflops(f_blk, t_blk),
                     D_in, D_out, t_dns, _tflops(f_dns, t_dns)))
    return rows


# ──────────────────────────────────────────────────── B: weight-gradient GEMM

def bench_wgrad(D, N, M, dtype, dev):
    """dW = x^T @ dy. Tiny output, enormous K. The suspect."""
    d = D // N
    rows = []
    for name, im, om in LAYER_SHAPES:
        d_in, d_out = im * d, om * d
        flops = 2 * N * M * d_in * d_out

        # Batched: N GEMMs of (d_in x M) @ (M x d_out) in one call.
        xb = torch.randn(N, d_in, M, device=dev, dtype=dtype)
        gb = torch.randn(N, M, d_out, device=dev, dtype=dtype)
        t_bmm = _time(lambda: torch.bmm(xb, gb))

        # Loop of unbatched GEMMs. Slower on paper (N launches) but each one is a
        # standalone cuBLAS call free to pick a split-K algorithm, which the
        # batched path will not do.
        xs = [xb[i].contiguous() for i in range(N)]
        gs = [gb[i].contiguous() for i in range(N)]
        t_loop = _time(lambda: [torch.mm(xs[i], gs[i]) for i in range(N)])

        # Grouped GEMM, if this build has it.
        t_grp = float("nan")
        try:
            xg = torch.randn(N * d_in, M, device=dev, dtype=dtype)
            offs = torch.arange(1, N + 1, device=dev, dtype=torch.int32) * d_in
            t_grp = _time(lambda: torch._grouped_mm(xg, gb, offs=offs))
        except Exception:
            pass

        # The dense layer's own dW, for reference.
        xd = torch.randn(D * im, M, device=dev, dtype=dtype)
        gd = torch.randn(M, D * om, device=dev, dtype=dtype)
        f_dns = 2 * M * (D * im) * (D * om)
        t_dns = _time(lambda: torch.mm(xd, gd))

        rows.append((name, d_in, d_out, t_bmm, _tflops(flops, t_bmm),
                     t_loop, _tflops(flops, t_loop),
                     t_grp, _tflops(flops, t_grp) if t_grp == t_grp else float("nan"),
                     t_dns, _tflops(f_dns, t_dns)))
    return rows


# ───────────────────────────────────────────────────────────── C: layout cost

def bench_layout(D, N, M, dtype, dev):
    """_batched_linear's permute-reshape against a contiguous stream-major bmm."""
    d = D // N
    w = torch.randn(N, d, d, device=dev, dtype=dtype)
    wt = w.transpose(1, 2).contiguous()
    flops = 2 * N * M * d * d

    # As written in mst.py: (B, T, N, d) in, permute to (N, B*T, d), bmm, permute back.
    x_btnd = torch.randn(1, M, N, d, device=dev, dtype=dtype)

    def current():
        B, T, n, dd = x_btnd.shape
        xr = x_btnd.permute(2, 0, 1, 3).reshape(n, B * T, dd)
        y = torch.bmm(xr, w)
        return y.view(n, B, T, -1).permute(1, 2, 0, 3)

    # Stream-major: (N, B*T, d) already contiguous, contiguous result.
    x_nmd = torch.randn(N, M, d, device=dev, dtype=dtype)

    def stream_major():
        return torch.bmm(x_nmd, wt)

    t_cur = _time(current)
    t_sm = _time(stream_major)
    return t_cur, _tflops(flops, t_cur), t_sm, _tflops(flops, t_sm)


# ───────────────────────────────────────────── D: block-diagonal vs masked dense

def bench_crossover(D, N, M, dtype, dev):
    """Full layer, fwd + dgrad + wgrad, block-diagonal against masked dense.

    The masked-dense arm is the same mathematics executed as a full-width GEMM with
    the off-diagonal blocks zeroed: N times the FLOPs, but at dense throughput and
    dense tile efficiency. If it wins, the structure costs more than it saves.
    """
    d = D // N
    t_blk = 0.0
    t_dns = 0.0
    f_blk = 0
    f_dns = 0
    for _name, im, om in LAYER_SHAPES:
        d_in, d_out = im * d, om * d
        D_in, D_out = im * D, om * D

        xb = torch.randn(N, M, d_in, device=dev, dtype=dtype)
        wb = torch.randn(N, d_in, d_out, device=dev, dtype=dtype)
        gb = torch.randn(N, M, d_out, device=dev, dtype=dtype)
        xbt = xb.transpose(1, 2).contiguous()
        wbt = wb.transpose(1, 2).contiguous()
        t_blk += _time(lambda: torch.bmm(xb, wb))       # forward
        t_blk += _time(lambda: torch.bmm(gb, wbt))      # dgrad
        t_blk += _time(lambda: torch.bmm(xbt, gb))      # wgrad
        f_blk += 3 * 2 * N * M * d_in * d_out

        xd = torch.randn(M, D_in, device=dev, dtype=dtype)
        wd = torch.randn(D_in, D_out, device=dev, dtype=dtype)
        gd = torch.randn(M, D_out, device=dev, dtype=dtype)
        xdt = xd.t().contiguous()
        wdt = wd.t().contiguous()
        t_dns += _time(lambda: torch.mm(xd, wd))
        t_dns += _time(lambda: torch.mm(gd, wdt))
        t_dns += _time(lambda: torch.mm(xdt, gd))
        f_dns += 3 * 2 * M * D_in * D_out

    return t_blk, f_blk, t_dns, f_dns


# ────────────────────────────────────────────────────────────────────── report

def run(D, N, M, dtype, dev):
    d = D // N
    print(f"\n{'='*78}\n  D={D}  N={N}  d={d}  M(=B*T)={M}  {dtype}\n{'='*78}")

    print("\n[A] forward GEMM: block-diagonal at d  vs  dense at D")
    print(f"  {'shape':<9} {'blk in>out':<12} {'ms':>8} {'TFLOP/s':>9}   "
          f"{'dns in>out':<13} {'ms':>8} {'TFLOP/s':>9}   {'ratio':>6}")
    for n, di, do, tb, fb, Di, Do, td, fd in bench_forward(D, N, M, dtype, dev):
        print(f"  {n:<9} {f'{di}>{do}':<12} {tb:8.3f} {fb:9.1f}   "
              f"{f'{Di}>{Do}':<13} {td:8.3f} {fd:9.1f}   {fb/fd:6.3f}")

    print("\n[B] weight-gradient GEMM (out is d x d, K = M). Suspect for the")
    print("    training-only gap: forward-only prefill nearly reaches parity.")
    print(f"  {'shape':<9} {'bmm ms':>8} {'TF/s':>7} {'loop ms':>8} {'TF/s':>7} "
          f"{'grpd ms':>8} {'TF/s':>7} {'dense ms':>9} {'TF/s':>7}")
    for (n, di, do, tb, fb, tl, fl, tg, fg, td, fd) in bench_wgrad(D, N, M, dtype, dev):
        gs = f"{tg:8.3f} {fg:7.1f}" if tg == tg else f"{'n/a':>8} {'':>7}"
        print(f"  {n:<9} {tb:8.3f} {fb:7.1f} {tl:8.3f} {fl:7.1f} {gs} {td:9.3f} {fd:7.1f}")

    tc, fc, ts, fs = bench_layout(D, N, M, dtype, dev)
    print(f"\n[C] activation layout, d>d projection")
    print(f"  _batched_linear (permute+reshape) {tc:7.3f} ms  {fc:7.1f} TFLOP/s")
    print(f"  stream-major (N, M, d) contiguous {ts:7.3f} ms  {fs:7.1f} TFLOP/s"
          f"   speedup {tc/ts:.2f}x")

    tb, fb, td, fd = bench_crossover(D, N, M, dtype, dev)
    print(f"\n[D] whole layer, fwd + dgrad + wgrad")
    print(f"  block-diagonal  {tb:8.3f} ms   {fb/1e9:9.1f} GFLOP   {_tflops(fb, tb):7.1f} TFLOP/s")
    print(f"  masked dense    {td:8.3f} ms   {fd/1e9:9.1f} GFLOP   {_tflops(fd, td):7.1f} TFLOP/s")
    print(f"  FLOP ratio dense/blk {fd/fb:5.2f}x     wall-clock ratio dense/blk {td/tb:5.2f}x")
    ratio = _tflops(fb, tb) / _tflops(fd, td)
    print(f"  throughput(d)/throughput(D) = {ratio:5.3f}   crossover threshold 1/N = {1/N:5.3f}")
    if td < tb:
        print(f"  VERDICT: block-diagonal LOSES. Executing this layer as a masked dense")
        print(f"           GEMM is {tb/td:.2f}x faster despite {fd/fb:.1f}x the FLOPs.")
    else:
        print(f"  VERDICT: block-diagonal wins by {td/tb:.2f}x in wall clock.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=768)
    ap.add_argument("--n-subs", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16384, help="M = B*T per step")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--sweep", action="store_true",
                    help="run the p08 depth ladder (D = 128*ceil(depth*aspect/128))")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA device"
    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    dims = [256, 512, 768, 1024] if args.sweep else [args.model_dim]
    for D in dims:
        assert D % args.n_subs == 0, f"D={D} not divisible by N={args.n_subs}"
        run(D, args.n_subs, args.tokens, dtype, dev)


if __name__ == "__main__":
    main()
