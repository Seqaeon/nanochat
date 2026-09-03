#!/usr/bin/env python3
"""P14: does MST's hardware utilization catch up to dense, and where?

WHY THIS EXISTS
    MST reaches roughly a third of dense's MFU at L=16 and closer to half at L=32.
    Whether that gap ever closes decides how the paper frames wall clock: a gap that
    narrows with scale is a different claim from one that is constant. Nothing in the
    repo measured the curve directly. The numbers we had came from two sources that
    could not be compared:

      - paper_bench A3 times forward+backward ONLY, at batch 4 (8192 tokens), with
        warmup 5 and 8 iterations. It omits the optimizer, underfeeds the GPU, and is
        unstable: dense L=16 measured 36.90 ms in one run and 58.98 ms in another with
        byte-identical peak memory.
      - base_train's dt is the real thing but only exists for depths someone happened
        to train, and mixes hardware across runs.

    This script measures what base_train's dt measures -- a full optimizer step at the
    real total batch, including gradient accumulation and the Muon/AdamW step -- but
    sweeps depth cheaply on random weights, and reports the MST/dense utilization ratio
    with a fit and an honest extrapolation.

WHAT IT GETS RIGHT THAT A3 DOES NOT
    1. The optimizer is included. Muon's Newton-Schulz runs on every matrix parameter
       each step, and MST has N narrow matrices per layer where dense has one wide one,
       so the cost does not scale identically between the arms. Excluding it is not
       neutral.
    2. Gradient accumulation to a real total batch. At batch 4 both arms are starved and
       MST more so, since its GEMMs are already narrower.
    3. Enough warmup and iterations to be reproducible, defaulting to 10 and 30.

GEMM ALIGNMENT
    MST's per-stream matrices are sub_dim = d_model / n_subs wide. When sub_dim is not a
    multiple of 128 the GEMMs miss tensor-core alignment and MFU drops off-trend: L=28
    (sub_dim 448) measured 18.7% against 21.7% at the smaller L=24 and 22.4% at L=32.
    Misaligned depths are flagged and excluded from the fit by default.

Usage:
    python -m scripts.p14_mfu_curve                          # default sweep
    python -m scripts.p14_mfu_curve --depths 16 24 32 40
    python -m scripts.p14_mfu_curve --device-batch 4 --total-batch 131072
    python -m scripts.p14_mfu_curve --include-misaligned --json out/p14.json
"""

import argparse
import json
import math
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST

ASPECT, HEAD_DIM, N_SUBS, SUB_HEAD_DIM, SEQ, VOCAB = 64, 128, 4, 64, 2048, 32768


def model_dim(depth):
    return ((depth * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM


def aligned(depth):
    """True when MST's per-stream GEMMs hit tensor-core alignment."""
    return (model_dim(depth) // N_SUBS) % 128 == 0


def valid_mst(depth):
    return (model_dim(depth) // N_SUBS) % SUB_HEAD_DIM == 0


def peak_tflops():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0).upper()
    for key, tf in (("B200", 2250.0), ("H200", 989.0), ("H100", 989.0),
                    ("A100", 312.0), ("L40", 181.0), ("4090", 165.2)):
        if key in name:
            return tf
    return None


def build(depth, mst, device):
    D = model_dim(depth)
    common = dict(sequence_len=SEQ, n_layer=depth, n_embd=D, n_head=D // HEAD_DIM,
                  n_kv_head=D // HEAD_DIM, vocab_size=VOCAB, window_pattern="SSSL")
    if not mst:
        cfg = GPTConfig(**common)
        cls = GPT
    else:
        cfg = GPTConfig(**common, use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=D // N_SUBS,
                        mst_head_dim=0, mst_input_mode="learned_proj",
                        mst_routing_mode="soft_weighted", mst_routing_topk=0,
                        mst_ffn_mode="standard", mst_transition_mode="aggregate_distribute",
                        mst_final_mode="concat_proj", mst_final_topk=0,
                        mst_routing_aux_weight=0.01, mst_grad_equalize=1,
                        mst_block_diagonal_muon=1, mst_transition_width_mult=float(N_SUBS),
                        mst_sub_lr_scale=2.0, mst_multi_scale_windows=1,
                        mst_sub_head_dim=SUB_HEAD_DIM, mst_compose_windows=1,
                        mst_wo_mode="dense", mst_per_stream_ve=1,
                        mst_stream_topk=1, mst_stream_router_noise=1.0)
        cls = MST
    with torch.device("meta"):
        model = cls(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model, cfg


def time_step(model, device, device_batch, total_batch, warmup, iters, compile_model):
    """One full optimizer step: grad accumulation, backward, optimizer.step().

    This is the quantity base_train reports as dt, not a bare forward+backward.
    """
    accum = max(1, total_batch // (device_batch * SEQ))
    opt = model.setup_optimizer()
    optimizers = opt if isinstance(opt, (list, tuple)) else [opt]
    fwd = torch.compile(model) if compile_model else model
    x = torch.randint(0, VOCAB, (device_batch, SEQ), device=device)
    y = torch.randint(0, VOCAB, (device_batch, SEQ), device=device)

    def step():
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        for _ in range(accum):
            loss = fwd(x, y)
            loss = loss if torch.is_tensor(loss) else loss[0]
            (loss / accum).backward()
        for o in optimizers:
            o.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    # Median, not minimum: a full optimizer step is long enough that the minimum is a
    # lucky sample rather than the achievable rate, and the paper reports sustained cost.
    return ts[len(ts) // 2], ts[0], accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    ap.add_argument("--arms", type=str, default="mst,dense")
    ap.add_argument("--device-batch", type=int, default=8,
                    help="micro-batch per step; raise until it OOMs for best utilization")
    ap.add_argument("--total-batch", type=int, default=262144,
                    help="tokens per optimizer step, accumulated. 262144 is what "
                         "base_train's auto rule chose for the isoFLOP sweep")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-misaligned", action="store_true",
                    help="keep depths whose sub_dim is not a multiple of 128 in the fit")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    dev = "cuda"
    pk = peak_tflops()
    print(f"device: {torch.cuda.get_device_name(0)}"
          + (f"  peak bf16: {pk} TFLOP/s" if pk else "  (peak unknown, MFU omitted)"))
    print(f"total batch {args.total_batch:,} tokens, device batch {args.device_batch}, "
          f"seq {SEQ}, warmup {args.warmup}, iters {args.iters}, compile {args.compile}")
    print("step = grad accumulation + backward + optimizer.step(), i.e. base_train's dt\n")

    arms = [a for a in ("mst", "dense") if a in args.arms]
    res = {}
    hdr = f"{'arm':>6}{'L':>4}{'d_model':>8}{'sub_dim':>9}{'algn':>6}{'accum':>7}{'dt ms':>10}{'TFLOP/s':>10}"
    print(hdr + (f"{'MFU':>7}" if pk else "") + f"{'peak GB':>9}")
    for depth in args.depths:
        for arm in arms:
            if arm == "mst" and not valid_mst(depth):
                print(f"{'mst':>6}{depth:>4}{model_dim(depth):>8}{model_dim(depth)//N_SUBS:>9}"
                      f"     --   sub_dim not divisible by {SUB_HEAD_DIM}, no MST arm")
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model, _ = build(depth, arm == "mst", dev)
                _, active_fpt, _ = model.estimate_flops()
                med, lo, accum = time_step(model, dev, args.device_batch, args.total_batch,
                                           args.warmup, args.iters, args.compile)
                tf = active_fpt * args.total_batch / (med * 1e-3) / 1e12
                gb = torch.cuda.max_memory_allocated() / 2**30
                al = "yes" if (arm == "dense" or aligned(depth)) else "NO"
                res.setdefault(arm, {})[depth] = dict(
                    dt_ms=med, dt_min_ms=lo, tflops=tf, mfu=(100 * tf / pk) if pk else None,
                    peak_gb=gb, accum=accum, active_flops_per_token=active_fpt,
                    aligned=(al == "yes"))
                sub = f"{model_dim(depth)//N_SUBS}" if arm == "mst" else "-"
                line = (f"{arm:>6}{depth:>4}{model_dim(depth):>8}{sub:>9}"
                        f"{al:>6}{accum:>7}{med:>10.2f}{tf:>10.1f}")
                print(line + (f"{100*tf/pk:>6.1f}%" if pk else "") + f"{gb:>9.2f}")
                del model
            except torch.cuda.OutOfMemoryError:
                print(f"{arm:>6}{depth:>4}  OOM at device batch {args.device_batch}")
            torch.cuda.empty_cache()

    # ---- the question the script exists to answer ----
    shared = sorted(set(res.get("mst", {})) & set(res.get("dense", {})))
    if len(shared) < 2:
        print("\nneed at least two depths measured on BOTH arms to fit a curve")
        return
    print(f"\n{'L':>4}{'MST TF/s':>10}{'dense TF/s':>12}{'ratio':>8}{'aligned':>9}")
    xs, ys = [], []
    for d in shared:
        m, n = res["mst"][d]["tflops"], res["dense"][d]["tflops"]
        ok = res["mst"][d]["aligned"]
        print(f"{d:>4}{m:>10.1f}{n:>12.1f}{m/n:>8.3f}{'yes' if ok else 'NO':>9}")
        if ok or args.include_misaligned:
            xs.append(math.log(d))
            ys.append(math.log(m / n))
    if len(xs) < 2:
        print("\nnot enough aligned depths to fit; rerun with --include-misaligned "
              "or add depths where sub_dim is a multiple of 128")
        return
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = sxy / sxx
    inter = my - slope * mx
    print(f"\nfit on {n} aligned depth(s): ratio = exp({inter:.3f}) * L^{slope:.3f}")
    if slope <= 0:
        print("  ratio is flat or shrinking with depth: MST does not catch up.")
    else:
        Lx = math.exp((0.0 - inter) / slope)
        print(f"  ratio reaches 1.0 (parity) at L = {Lx:,.0f}, d_model = {model_dim(round(Lx)):,}")
        print(f"  CAUTION: that is an extrapolation from L in [{min(shared)}, {max(shared)}]. "
              "A power law fitted over a 2x depth range says little about a 10x one, and "
              "block-diagonal GEMM throughput has a hardware ceiling below 1.0 regardless "
              "of depth. Treat it as a direction, not a prediction.")
        for L in (40, 48, 64, 96):
            print(f"    projected ratio at L={L}: {math.exp(inter) * L**slope:.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=torch.cuda.get_device_name(0), peak_tflops=pk,
                           total_batch=args.total_batch, device_batch=args.device_batch,
                           seq=SEQ, results=res, fit=dict(slope=slope, intercept=inter)), f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
