"""Group A of the MST paper experiments: everything that needs a GPU but no
trained weights and no checkpoints.

Covers A1-A6 in one pass, because they all build the same two models and differ
only in what they time:

  A1  fused-attention throughput at head_dim in {32,64,128}   -> Fig. 2(b)
  A2  profiler breakdown of MST step time by kernel class      -> Fig. 2(a)
  A3  train step time (fwd+bwd), MST vs dense                  -> Table 5
  A4  prefill throughput vs sequence length                    -> Table 5 / Fig
  A5  decode latency vs KV-cache length                        -> Table 5
  A6  peak memory and achieved MFU                             -> Table 5

Random initialisation is deliberate and sufficient: these are architecture-level
measurements that do not depend on the values of the weights. MoL reports its
speed tables the same way.

    python -m scripts.paper_bench --depths 24 --out scratch/paper_bench.json
    python -m scripts.paper_bench --only a1        # just the head_dim sweep
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

# Pin the Inductor cache before importing torch. Without a stable cache dir,
# every invocation re-runs matmul autotuning and can land on a different kernel
# config, which shows up as large run-to-run swings that look like noise but are
# not. Set TORCHINDUCTOR_CACHE_DIR yourself to override.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.getcwd(), "scratch", ".inductor_cache"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST
from scripts.paper_lib import gpu_peak_tflops

ASPECT, HEAD_DIM, N_SUBS, SEQ, VOCAB = 64, 128, 4, 2048, 32768
WINDOW_PATTERN = "SSSL"


def dims(depth):
    D = ((depth * ASPECT + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    return D, D // HEAD_DIM, D // N_SUBS


def make_config(depth, mst, seq_len=SEQ):
    D, nh, d = dims(depth)
    common = dict(sequence_len=seq_len, vocab_size=VOCAB, n_layer=depth,
                  n_head=nh, n_kv_head=nh, n_embd=D, window_pattern=WINDOW_PATTERN)
    if not mst:
        return GPTConfig(**common)
    return GPTConfig(**common, use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=d,
                     mst_head_dim=0, mst_input_mode='learned_proj',
                     mst_routing_mode='soft_weighted', mst_routing_topk=0,
                     mst_ffn_mode='standard',
                     mst_transition_mode='aggregate_distribute',
                     mst_final_mode='concat_proj', mst_final_topk=0,
                     mst_routing_aux_weight=0.01, mst_diversity_weight=0.0,
                     mst_grad_equalize=1, mst_block_diagonal_muon=1,
                     mst_transition_width_mult=float(N_SUBS),
                     mst_sub_lr_scale=2.0,
                     mst_multi_scale_windows=int(MULTI_SCALE))


COMPILE = True   # set by --no-compile
ITERS = 0        # set by --iters; 0 means use each experiment's default
ARMS = ("mst", "dense")   # set by --arms
MULTI_SCALE = True        # set by --no-multi-scale


def _arms():
    """The (label, is_mst) pairs selected by --arms, in a stable order."""
    return [(n, m) for n, m in (("mst", True), ("dense", False)) if n in ARMS]


def build(depth, mst, device, seq_len=SEQ, compile_model=None):
    """Build and, by default, torch.compile the model.

    Compilation is not optional for a fair comparison. base_train.py compiles by
    default, so every wall-clock number in the paper comes from a compiled
    model, and BatchedMSTLayer exists specifically so Inductor can fuse the
    per-stream ops. Benchmarking MST uncompiled measures a model nobody trains:
    it leaves the batched einsums and the N-way RMSNorm as separate elementwise
    kernels and understates MST by roughly 4x.
    """
    cfg = make_config(depth, mst, seq_len)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = (MST if mst else GPT)(cfg)
    model.to_empty(device=device)
    model.init_weights()
    do_compile = COMPILE if compile_model is None else compile_model
    if do_compile:
        model = torch.compile(model)
    return model, cfg


def cuda_time(fn, warmup=None, iters=10, stat="min"):
    """Wall time in ms of fn(), measured with CUDA events.

    Reports the MINIMUM by default. On a GPU driven near its power limit the
    clocks drop under sustained load, so the median drifts upward with the
    iteration count and the arm working the card hardest is penalised most.
    The minimum is the standard estimator for achievable kernel time and is
    stable against both throttling and scheduler noise. The median is returned
    alongside it so the gap is visible.
    """
    # A compiled model spends its first call(s) compiling; absorb them.
    warmup = (5 if COMPILE else 3) if warmup is None else warmup
    if ITERS > 0:
        iters = ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    lo, med = ts[0], ts[len(ts) // 2]
    return lo if stat == "min" else med, med


# ---------------------------------------------------------------- A1
def a1_head_dim_sweep(device, seq=2048, batch=4, heads_total=16):
    """Isolated fused-attention throughput at several head dimensions.

    Total attention width is held constant (heads_total * head_dim fixed by
    varying the head count), so the only thing changing is the kernel's tile
    shape. This is what separates 'MST is slow' from 'head_dim 32 is slow'.
    """
    from nanochat.flash_attention import flash_attn
    out = []
    width = heads_total * 128          # keep n_head*head_dim constant
    for hd in (32, 64, 128):
        nh = width // hd
        q, k, v = (torch.randn(batch, seq, nh, hd, device=device,
                               dtype=torch.bfloat16) for _ in range(3))
        fn = lambda: flash_attn.flash_attn_func(q, k, v, causal=True,
                                                window_size=(-1, 0))
        ms, _ = cuda_time(fn, iters=20)
        # causal attention: ~2 * 2 * B * H * T^2 * hd FLOPs (QK^T and AV)
        flops = 4 * batch * nh * seq * seq * hd * 0.5
        out.append(dict(head_dim=hd, n_head=nh, ms=ms, tflops=flops / (ms * 1e-3) / 1e12))
        print(f"  head_dim={hd:3d} n_head={nh:2d}  {ms:7.3f} ms  "
              f"{out[-1]['tflops']:6.1f} TFLOP/s")
    return out


# ---------------------------------------------------------------- A2
def a2_kernel_breakdown(depth, device, batch=4, top=12):
    from torch.profiler import profile, ProfilerActivity
    model, cfg = build(depth, mst=True, device=device)
    x = torch.randint(0, VOCAB, (batch, SEQ), device=device)
    y = torch.randint(0, VOCAB, (batch, SEQ), device=device)
    step = lambda: model(x, y).backward()
    for _ in range(3):
        step(); model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
    torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            rows.append((evt.key, evt.self_device_time_total / 1e3))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    print(f"  total device time {total:.1f} ms over {len(rows)} kernels")
    for k, ms in rows[:top]:
        print(f"    {ms:8.2f} ms  {100*ms/total:5.1f}%  {k[:70]}")
    del model
    torch.cuda.empty_cache()
    return dict(total_ms=total, kernels=[dict(name=k, ms=ms) for k, ms in rows[:top]])


# ---------------------------------------------------------------- A3 / A6
def a3_train_step(depth, device, batch=4):
    res = {}
    peak_tf = gpu_peak_tflops()
    for name, mst in _arms():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, cfg = build(depth, mst, device)
        fpt, _, _ = model.estimate_flops()
        x = torch.randint(0, VOCAB, (batch, SEQ), device=device)
        y = torch.randint(0, VOCAB, (batch, SEQ), device=device)
        def step():
            model.zero_grad(set_to_none=True)
            model(x, y).backward()
        ms, _ = cuda_time(step, iters=8)
        toks = batch * SEQ
        achieved = fpt * toks / (ms * 1e-3) / 1e12
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        res[name] = dict(ms=ms, tok_per_s=toks / (ms * 1e-3),
                         flops_per_token=fpt, achieved_tflops=achieved,
                         mfu=(100 * achieved / peak_tf) if peak_tf else None,
                         peak_gb=peak_gb)
        print(f"  {name:5s} {ms:8.2f} ms/step  {achieved:6.1f} TFLOP/s"
              + (f"  MFU {res[name]['mfu']:.1f}%" if peak_tf else "")
              + f"  peak {peak_gb:.2f} GB")
        del model
        torch.cuda.empty_cache()
    if "mst" in res and "dense" in res:
        print(f"  speedup (dense/mst): {res['dense']['ms']/res['mst']['ms']:.2f}x")
    return res


# ---------------------------------------------------------------- A4
def a4_prefill(depth, device, lengths=(2048, 4096, 8192, 16384, 32768), batch=1):
    """Prefill throughput vs sequence length.

    One model per arm for the whole sweep, built at the longest length. Building
    per length meant five separate compilations per arm, and each one re-ran
    Inductor's autotuner independently, which was the dominant source of
    run-to-run variance. Window semantics are unaffected for T <= max: the long
    window is min(sequence_len, T), so a model built at 32768 and prefilled at
    2048 attends exactly as one built at 2048 would.
    """
    res = {}
    for name, mst in _arms():
        res[name] = []
        torch.cuda.empty_cache()
        model, _ = build(depth, mst, device, seq_len=max(lengths))
        for T in lengths:
            try:
                x = torch.randint(0, VOCAB, (batch, T), device=device)
                with torch.inference_mode():
                    ms, med = cuda_time(lambda: model(x), warmup=8, iters=20)
                res[name].append(dict(T=T, ms=ms, ms_median=med,
                                      tok_per_s=batch * T / (ms * 1e-3)))
                drift = 100 * (med - ms) / ms
                flag = "  <-- unstable" if drift > 25 else ""
                print(f"  {name:5s} T={T:6d}  {ms:9.2f} ms  "
                      f"{res[name][-1]['tok_per_s']:10.0f} tok/s"
                      f"   (median {med:.2f}, +{drift:.0f}%){flag}")
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:5s} T={T:6d}  OOM")
                res[name].append(dict(T=T, ms=None, oom=True))
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- A5
def a5_decode(depth, device, contexts=(256, 1024, 4096, 8192, 16384)):
    """Per-token decode latency with a KV cache, batch 1.

    Deliberately NOT compiled. FA3's flash_attn_with_kvcache is not traceable by
    Dynamo (it raises "tracing with num_splits <= 0 not supported"), and
    KVCache.get_pos() calls .item(), which breaks the graph anyway. Both arms hit
    this identically, so it is a kernel limitation rather than an MST one, and
    nanochat's own inference path does not compile for generation either.
    Uncompiled decode is therefore the deployment configuration, not a fallback.
    """
    from nanochat.engine import KVCache
    res = {}
    for name, mst in _arms():
        res[name] = []
        model, cfg = build(depth, mst, device, seq_len=max(contexts) + 8,
                           compile_model=False)
        kv_kwargs = getattr(model, "kv_cache_config", None)
        for ctx in contexts:
            try:
                torch.cuda.empty_cache()
                kw = dict(kv_kwargs) if kv_kwargs else dict(
                    num_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
                    v_head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer)
                cache = KVCache(batch_size=1, seq_len=ctx + 8,
                                device=device, dtype=torch.bfloat16, **kw)
                with torch.inference_mode():
                    prime = torch.randint(0, VOCAB, (1, ctx), device=device)
                    model(prime, kv_cache=cache)
                    nxt = torch.randint(0, VOCAB, (1, 1), device=device)
                    ms, _ = cuda_time(lambda: model(nxt, kv_cache=cache),
                                      warmup=3, iters=20)
                res[name].append(dict(ctx=ctx, ms_per_token=ms))
                print(f"  {name:5s} ctx={ctx:6d}  {ms:7.3f} ms/token")
                del cache
            except Exception as e:
                print(f"  {name:5s} ctx={ctx:6d}  FAILED: {type(e).__name__}: {e}")
                res[name].append(dict(ctx=ctx, ms_per_token=None, error=str(e)))
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[24])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--multi-scale", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="per-stream attention windows. Set --no-multi-scale to "
                         "match a checkpoint trained without them, e.g. the "
                         "plain S7_COMBO_A runs.")
    ap.add_argument("--arms", type=str, default="mst,dense",
                    help="which arms to measure, e.g. --arms dense. Note that "
                         "timings from separate invocations share no thermal "
                         "state; min-based timing makes this largely safe, but "
                         "prefer one invocation when both arms go in one table.")
    ap.add_argument("--iters", type=int, default=0,
                    help="timing iterations per measurement (0 = per-experiment "
                         "default). Raise it if a timing looks non-monotonic.")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of a1,a2,a3,a4,a5 "
                         "(default: all of them)")
    ap.add_argument("--skip", type=str, default=None,
                    help="comma-separated subset to skip, e.g. --skip a1")
    ap.add_argument("--out", type=str, default="scratch/paper_bench.json")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile both arms (default on, matches base_train.py). "
                         "--no-compile is for quick iteration only; it is not a "
                         "valid configuration to report.")
    args = ap.parse_args()

    global COMPILE, ITERS, ARMS, MULTI_SCALE
    COMPILE = args.compile
    MULTI_SCALE = args.multi_scale
    ITERS = args.iters
    ARMS = tuple(a.strip() for a in args.arms.split(","))
    bad = [a for a in ARMS if a not in ("mst", "dense")]
    if bad:
        print(f"unknown arm(s): {bad}; valid are mst, dense"); sys.exit(1)

    if not torch.cuda.is_available():
        print("paper_bench needs a GPU."); sys.exit(1)
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"peak bf16: {gpu_peak_tflops()} TFLOP/s")
    print(f"torch.compile: {'ON' if COMPILE else 'OFF'}"
          + ("" if COMPILE else "   <-- NOT a reportable configuration"))

    out = {"gpu": torch.cuda.get_device_name(0), "compiled": COMPILE,
           "arms": list(ARMS), "iters": ITERS, "multi_scale": MULTI_SCALE,
           "peak_tflops": gpu_peak_tflops(), "runs": {}}
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    want = lambda k: (only is None or k in only) and k not in skip
    planned = [k for k in ("a1", "a2", "a3", "a4", "a5") if want(k)]
    print(f"running: {', '.join(planned) if planned else '(nothing)'}"
          f"   arms: {', '.join(ARMS)}")

    if want("a1"):
        print("\n[A1] fused-attention throughput vs head_dim")
        out["a1_head_dim"] = a1_head_dim_sweep(dev)

    for depth in args.depths:
        D, nh, d = dims(depth)
        print(f"\n=== depth {depth}  D={D} d={d} ===")
        r = out["runs"].setdefault(str(depth), {})
        if want("a2"):
            print("[A2] MST kernel breakdown"); r["a2"] = a2_kernel_breakdown(depth, dev, args.batch)
        if want("a3"):
            print("[A3/A6] train step, memory, MFU"); r["a3"] = a3_train_step(depth, dev, args.batch)
        if want("a4"):
            print("[A4] prefill"); r["a4"] = a4_prefill(depth, dev)
        if want("a5"):
            print("[A5] decode"); r["a5"] = a5_decode(depth, dev)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
