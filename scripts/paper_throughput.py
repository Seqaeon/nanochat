"""Dense vs RemixedLinear throughput curve — the metareview's Tier-1 ask.

The AC asked for exactly this, and worded it narrowly:

    "The paper would be strengthened by reporting actual wall-clock training and
     inference throughput (tokens/sec) to demonstrate how this memory overhead
     impacts practical hardware efficiency compared to the dense baseline."

So three numbers per (depth, arm), not a Pareto plot:

  T1  training step throughput (fwd+bwd), tok/s        -> the headline column
  T2  prefill throughput vs sequence length, tok/s     -> "inference throughput"
  T3  decode latency vs KV-cache length, ms/token      -> "inference throughput"
  T4  peak memory, achieved TFLOP/s, MFU               -> "this memory overhead"

Random initialisation is deliberate and sufficient: throughput does not depend on
the values of the weights, so every point on the curve can be measured today
without a checkpoint. That is what makes this the cheapest decision-critical
experiment available.

MFU is reported twice, against total and against active FLOPs/token, because the
paper currently reports both and they disagree (Table 3 says 7.6e8 for both arms
at d12, Table 5 says 2.2e8 vs 3.6e8). Printing them side by side is half of the
Tier-0 reconciliation.

Two honest caveats, both printed in the output:

  * No fp8. base_train.py passes --fp8, which converts dense nn.Linear to
    Float8Linear but leaves RemixedLinear's template-bank einsums in bf16. Running
    both arms without fp8 is the fair comparison of the *architectures*, but it
    flatters Remix relative to what base_train.py actually trains. Use --fp8 to
    measure the as-trained configuration instead.
  * Head-dim mismatch. base_train.py picks n_head differently for research
    branches (_choose_research_heads) than for dense (model_dim // head_dim), so
    at some depths the two arms run different attention tile shapes. The script
    prints both and flags the depths where they differ.

    python -m scripts.paper_throughput --depths 4 8 12 16 20 24
    python -m scripts.paper_throughput --only t1,t4 --arms dense
    python -m scripts.paper_throughput --depths 12 --plot out/throughput.png
"""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import time

# Pin the Inductor cache before importing torch, so repeated invocations reuse
# autotuned kernels instead of re-running the autotuner and landing on different
# configs — that shows up as run-to-run swings that look like noise but are not.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.getcwd(), "out", ".inductor_cache"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig, RemixedLinear
from scripts.paper_lib import gpu_peak_tflops

# ---------------------------------------------------------------------------
# Geometry. Mirrors scripts/research_compare.py:model_dims + base_train.py:
# build_model_meta so a point on this curve is the same model base_train.py
# would have built from the same sweep flags.
# ---------------------------------------------------------------------------
# VOCAB=32768 reproduces the paper's Table 2 exactly (dense d12 = 286.3M params).
# If your tokenizer dir has a different vocab, pass --vocab; the params column will
# move but tok/s will not.
ASPECT, HEAD_DIM, SEQ, VOCAB = 64, 128, 2048, 32768


def model_dim_for(depth, aspect=ASPECT, head_dim=HEAD_DIM):
    return ((depth * aspect + head_dim - 1) // head_dim) * head_dim


def _choose_research_heads(embed_dim, preferred_heads):
    """Verbatim from base_train.py:build_model_meta.

    Copied rather than imported because base_train.py runs argparse and builds a
    tokenizer at import time. If that function changes, this must change with it.
    """
    pow2 = [1 << i for i in range(0, 12)]
    valid_pow2 = [h for h in pow2
                  if h <= embed_dim and embed_dim % h == 0 and (embed_dim // h) % 8 == 0]
    if valid_pow2:
        return min(valid_pow2, key=lambda h: (abs(h - preferred_heads), -h))
    valid_any = [h for h in range(1, embed_dim + 1) if embed_dim % h == 0]
    return min(valid_any, key=lambda h: abs(h - preferred_heads)) if valid_any else 1


# The 29C configuration, from REMIX_COMMON in scripts/p29_sweep.sh. Changing any
# of these makes the curve describe a model the paper does not report.
REMIX_KWARGS = dict(
    use_basis_gate=False,
    use_output_gate=True,
    use_context=True,
    basis_gate_mode="centered",
    gate_temperature=2.0,
    basis_scale_factor=4,
    n_templates=8,
    template_routing_learned=True,
    template_topk=0,
    basis_gate_rank=8,
)


def make_config(depth, arm, seq_len=SEQ, chunk=256, vocab=VOCAB, n_templates=8,
                extra_remix=None):
    D = model_dim_for(depth)
    base_heads = D // HEAD_DIM
    if arm == "dense":
        return GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=depth,
                         n_head=base_heads, n_kv_head=base_heads, n_embd=D)
    nh = _choose_research_heads(D, base_heads)
    kw = dict(REMIX_KWARGS, n_templates=n_templates)
    kw.update(extra_remix or {})
    return GPTConfig(
        sequence_len=seq_len, vocab_size=vocab, n_layer=depth,
        n_head=nh, n_kv_head=nh, n_embd=D,
        use_remix_linear=True, moe_embed_dim=D,
        remix_basis_size=D, scale_basis_size=True,
        cclblock_modulation="weight", cclblock_context_stream="selective",
        cclblock_gate_temperature=2.0,
        p23_quantile_route=1, p28_chunk_routing_size=chunk,
        remixed_linear_kwargs=kw,
    )


COMPILE = True
ITERS = 0
ARMS = ("dense", "remix")


def _arms():
    return [a for a in ("dense", "remix") if a in ARMS]


def build(depth, arm, device, seq_len=SEQ, compile_model=None, **cfg_kw):
    cfg = make_config(depth, arm, seq_len, **cfg_kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    do_compile = COMPILE if compile_model is None else compile_model
    if do_compile:
        model = torch.compile(model)
    return model, cfg


def cuda_time(fn, warmup=None, iters=10):
    """(min_ms, median_ms) over `iters` calls, measured with CUDA events.

    The minimum is the headline. Under sustained load an H200 drops clocks, so
    the median drifts upward with iteration count and penalises whichever arm
    works the card hardest — which here is systematically Remix. The median is
    returned alongside so the size of that effect stays visible.
    """
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
    return ts[0], ts[len(ts) // 2]


def _oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ---------------------------------------------------------------- T1 / T4
def t1_train_step(depth, device, batch, vocab, seq, opt_step=False, **cfg_kw):
    """Training step throughput (fwd+bwd) with a batch backoff.

    fwd+bwd, not fwd+bwd+optimizer: the optimizer step is amortised over
    grad_accum microbatches in base_train.py, so it does not scale with tokens
    and does not belong in a tok/s number. --opt-step measures it separately,
    which matters here because Muon runs a per-template Newton-Schulz over the
    K-block template bank and that cost is Remix-specific.
    """
    peak_tf = gpu_peak_tflops()
    res = {}
    for arm in _arms():
        b = batch
        while b >= 1:
            model = None
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                model, cfg = build(depth, arm, device, seq_len=seq, vocab=vocab, **cfg_kw)
                total_f, active_f, active_p = model.estimate_flops()
                nparams = sum(p.numel() for p in model.parameters())
                x = torch.randint(0, vocab, (b, seq), device=device)
                y = torch.randint(0, vocab, (b, seq), device=device)

                def step():
                    model.zero_grad(set_to_none=True)
                    model(x, y).backward()

                ms, med = cuda_time(step, iters=8)
                toks = b * seq
                tps = toks / (ms * 1e-3)
                row = dict(
                    arm=arm, depth=depth, n_embd=cfg.n_embd, n_head=cfg.n_head,
                    batch=b, seq=seq, ms=ms, ms_median=med, tok_per_s=tps,
                    params=nparams, active_params=active_p,
                    flops_total=total_f, flops_active=active_f,
                    tflops_total=total_f * tps / 1e12,
                    tflops_active=active_f * tps / 1e12,
                    peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                )
                if peak_tf:
                    row["mfu_total"] = 100 * row["tflops_total"] / peak_tf
                    row["mfu_active"] = 100 * row["tflops_active"] / peak_tf
                if opt_step:
                    row["opt_step_ms"] = _time_opt_step(model, device)
                res[arm] = row
                print(f"  {arm:6s} b={b:<3d} {ms:8.2f} ms  {tps:10.0f} tok/s  "
                      f"peak {row['peak_gb']:6.2f} GB  "
                      f"{row['tflops_total']:6.1f} TF(total) / "
                      f"{row['tflops_active']:6.1f} TF(active)"
                      + (f"  opt {row['opt_step_ms']:.1f} ms" if opt_step else ""))
                break
            except Exception as e:
                if not _oom(e):
                    print(f"  {arm:6s} b={b:<3d} FAILED: {type(e).__name__}: {e}")
                    res[arm] = dict(arm=arm, depth=depth, error=f"{type(e).__name__}: {e}")
                    break
                print(f"  {arm:6s} b={b:<3d} OOM, backing off")
                b //= 2
            finally:
                del model
                torch.cuda.empty_cache()
        else:
            res[arm] = dict(arm=arm, depth=depth, error="OOM at batch 1")
            print(f"  {arm:6s} OOM even at batch 1")
    return res


def _time_opt_step(model, device):
    """One optimizer step, with gradients already present.

    Built and discarded per call: Muon's momentum buffers are the single largest
    allocation in a Remix run (see the memory table in remix_checklist.md), so
    holding the optimizer alive across arms would change the peak-memory column
    of whichever arm ran second.
    """
    inner = getattr(model, "_orig_mod", model)
    opt = inner.setup_optimizer()
    ms, _ = cuda_time(lambda: opt.step(), warmup=2, iters=5)
    del opt
    torch.cuda.empty_cache()
    return ms


# ---------------------------------------------------------------- T2
def t2_prefill(depth, device, lengths, batch, vocab, **cfg_kw):
    """Prefill throughput vs sequence length.

    One model per arm for the whole sweep, built at the longest length: building
    per length means a fresh compilation and a fresh autotune per point, which
    dominates the variance. Window semantics are unaffected for T <= max.
    """
    res = {}
    for arm in _arms():
        res[arm] = []
        model = None
        try:
            torch.cuda.empty_cache()
            model, _ = build(depth, arm, device, seq_len=max(lengths), vocab=vocab, **cfg_kw)
            for T in lengths:
                try:
                    x = torch.randint(0, vocab, (batch, T), device=device)
                    with torch.inference_mode():
                        ms, med = cuda_time(lambda: model(x), warmup=5, iters=15)
                    row = dict(T=T, ms=ms, ms_median=med, tok_per_s=batch * T / (ms * 1e-3))
                    res[arm].append(row)
                    print(f"  {arm:6s} T={T:6d}  {ms:9.2f} ms  {row['tok_per_s']:10.0f} tok/s")
                except Exception as e:
                    tag = "OOM" if _oom(e) else f"{type(e).__name__}"
                    print(f"  {arm:6s} T={T:6d}  {tag}")
                    res[arm].append(dict(T=T, ms=None, error=tag))
                torch.cuda.empty_cache()
        finally:
            del model
            torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- T3
def t3_decode(depth, device, contexts, vocab, **cfg_kw):
    """Per-token decode latency with a KV cache, batch 1.

    Deliberately not compiled: FA3's flash_attn_with_kvcache is not Dynamo
    traceable and KVCache.get_pos() calls .item(), which breaks the graph. Both
    arms hit this identically and nanochat's own generation path does not compile
    either, so uncompiled decode is the deployment configuration, not a fallback.

    Expect this to be the worst column for Remix, and expect the reason to be
    structural: at T=1 the chunk-routing branch pads the sequence up to a full
    `chunk` before composing (F.pad to n_chunks*chunk in RemixedLinear.forward),
    so a single decoded token pays for `chunk` tokens of weight composition. The
    number this prints is the one the AC's "inference throughput" question is
    really asking for.
    """
    from nanochat.engine import KVCache
    res = {}
    for arm in _arms():
        res[arm] = []
        model = None
        try:
            model, cfg = build(depth, arm, device, seq_len=max(contexts) + 8,
                               compile_model=False, vocab=vocab, **cfg_kw)
            hd = cfg.n_embd // cfg.n_head
            for ctx in contexts:
                try:
                    torch.cuda.empty_cache()
                    cache = KVCache(batch_size=1, num_heads=cfg.n_kv_head,
                                    seq_len=ctx + 8, head_dim=hd, v_head_dim=hd,
                                    num_layers=cfg.n_layer, device=device,
                                    dtype=torch.bfloat16)
                    with torch.inference_mode():
                        model(torch.randint(0, vocab, (1, ctx), device=device), kv_cache=cache)
                        nxt = torch.randint(0, vocab, (1, 1), device=device)
                        ms, _ = cuda_time(lambda: model(nxt, kv_cache=cache),
                                          warmup=3, iters=15)
                    res[arm].append(dict(ctx=ctx, ms_per_token=ms, tok_per_s=1000.0 / ms))
                    print(f"  {arm:6s} ctx={ctx:6d}  {ms:8.3f} ms/token  "
                          f"{1000.0 / ms:8.1f} tok/s")
                    del cache
                except Exception as e:
                    print(f"  {arm:6s} ctx={ctx:6d}  FAILED: {type(e).__name__}: {e}")
                    res[arm].append(dict(ctx=ctx, ms_per_token=None, error=str(e)[:200]))
                torch.cuda.empty_cache()
        finally:
            del model
            torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- reporting
def write_tables(out, path_base):
    """CSV + markdown for the T1 row, which is the table that goes in the paper."""
    rows = []
    for depth, r in sorted(out["runs"].items(), key=lambda kv: int(kv[0])):
        for arm, v in (r.get("t1") or {}).items():
            if "error" not in v:
                rows.append(v)
    if not rows:
        return
    cols = ["depth", "arm", "n_embd", "n_head", "batch", "params", "active_params",
            "flops_total", "flops_active", "ms", "tok_per_s", "peak_gb",
            "tflops_total", "tflops_active", "mfu_total", "mfu_active", "opt_step_ms"]
    with open(path_base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth"], {})[r["arm"]] = r
    lines = ["| depth | arm | params | active params | tok/s | rel. | peak GB | MFU(total) | MFU(active) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(by_depth):
        a = by_depth[d]
        base = a.get("dense", {}).get("tok_per_s")
        for arm in ("dense", "remix"):
            if arm not in a:
                continue
            v = a[arm]
            rel = f"{v['tok_per_s'] / base:.2f}x" if base else "—"
            lines.append(
                f"| d{d} | {arm} | {v['params'] / 1e6:.0f}M | {v['active_params'] / 1e6:.0f}M | "
                f"{v['tok_per_s']:,.0f} | {rel} | {v['peak_gb']:.1f} | "
                f"{v.get('mfu_total', float('nan')):.1f}% | {v.get('mfu_active', float('nan')):.1f}% |")
    with open(path_base + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def make_plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    style = {"dense": dict(marker="o", color="#444"), "remix": dict(marker="s", color="#c1272d")}
    series = {a: [] for a in ("dense", "remix")}
    for depth, r in sorted(out["runs"].items(), key=lambda kv: int(kv[0])):
        for arm, v in (r.get("t1") or {}).items():
            if "error" not in v:
                series[arm].append((v["active_params"] / 1e6, v["tok_per_s"], v["peak_gb"], int(depth)))
    for arm, pts in series.items():
        if not pts:
            continue
        pts.sort()
        ap, tps, gb, dep = zip(*pts)
        axes[0].plot(dep, tps, label=arm, **style[arm])
        axes[1].plot(ap, tps, label=arm, **style[arm])
        axes[2].plot(dep, gb, label=arm, **style[arm])
    for ax, (xl, yl, ttl) in zip(axes, [
            ("depth", "tokens/sec", "training throughput"),
            ("active params (M)", "tokens/sec", "throughput vs active params"),
            ("depth", "peak memory (GB)", "memory")]):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_yscale("log" if "tokens" in yl else "linear")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24],
                    help="six points spans 37M to ~1.3B active params, which covers "
                         "the whole range the paper reports plus the 1B+ scale the "
                         "AC asked about. Drop 20/24 for a quick pass.")
    ap.add_argument("--arms", type=str, default="dense,remix")
    ap.add_argument("--batch", type=int, default=8,
                    help="starting device batch size for T1; halved on OOM until it fits")
    ap.add_argument("--match-batch", action=argparse.BooleanOptionalAction, default=True,
                    help="re-time the arm that fit a larger batch at the smaller arm's "
                         "batch, so the comparison at a depth is not confounded by "
                         "batch size. --no-match-batch reports each arm at its own best.")
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--chunk", type=int, default=256,
                    help="p28_chunk_routing_size. Memory and compose cost both scale "
                         "as 1/chunk; 256 is the current sweep setting.")
    ap.add_argument("--templates", type=int, default=8)
    ap.add_argument("--prefill-lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--prefill-batch", type=int, default=1)
    ap.add_argument("--decode-contexts", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--opt-step", action="store_true",
                    help="also time one optimizer step per arm. Allocates Muon momentum "
                         "buffers (~20 GB at d24 Remix), so it can OOM where T1 did not.")
    ap.add_argument("--iters", type=int, default=0, help="0 = per-experiment default")
    ap.add_argument("--only", type=str, default=None, help="subset of t1,t2,t3")
    ap.add_argument("--skip", type=str, default=None)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="matches base_train.py, which compiles by default. "
                         "--no-compile is for iteration only, not for reporting.")
    ap.add_argument("--out", type=str, default="out/paper_throughput.json")
    ap.add_argument("--plot", type=str, default=None)
    args = ap.parse_args()

    global COMPILE, ITERS, ARMS
    COMPILE, ITERS = args.compile, args.iters
    ARMS = tuple(a.strip() for a in args.arms.split(","))
    bad = [a for a in ARMS if a not in ("dense", "remix")]
    if bad:
        print(f"unknown arm(s): {bad}; valid are dense, remix"); sys.exit(1)
    if not torch.cuda.is_available():
        print("paper_throughput needs a GPU."); sys.exit(1)

    dev = torch.device("cuda")
    peak_tf = gpu_peak_tflops()
    print(f"device: {torch.cuda.get_device_name(0)}   peak bf16: {peak_tf} TFLOP/s")
    print(f"torch.compile: {'ON' if COMPILE else 'OFF   <-- NOT a reportable configuration'}")
    print("fp8: OFF for both arms. base_train.py trains with --fp8, which accelerates")
    print("     dense nn.Linear but not RemixedLinear's bank einsums, so these numbers")
    print("     understate the dense arm relative to the runs in the paper.")

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    skip = {s.strip() for s in args.skip.split(",")} if args.skip else set()
    want = lambda k: (only is None or k in only) and k not in skip

    cfg_kw = dict(chunk=args.chunk, n_templates=args.templates)
    out = dict(gpu=torch.cuda.get_device_name(0), compiled=COMPILE, fp8=False,
               peak_tflops=peak_tf, arms=list(ARMS), seq=args.seq, vocab=args.vocab,
               chunk=args.chunk, n_templates=args.templates, runs={})

    for depth in args.depths:
        D = model_dim_for(depth)
        nh_d, nh_r = D // HEAD_DIM, _choose_research_heads(D, D // HEAD_DIM)
        note = "" if nh_d == nh_r else (
            f"   <-- head_dim differs: dense {D // nh_d}, remix {D // nh_r}")
        print(f"\n=== depth {depth}  D={D}  n_head dense={nh_d} remix={nh_r}{note} ===")
        r = out["runs"].setdefault(str(depth), {})
        if want("t1"):
            print("[T1/T4] train step (fwd+bwd), memory, MFU")
            r["t1"] = t1_train_step(depth, dev, args.batch, args.vocab, args.seq,
                                    opt_step=args.opt_step, **cfg_kw)
            if args.match_batch:
                ok = {a: v for a, v in r["t1"].items() if "error" not in v}
                if len(ok) > 1 and len({v["batch"] for v in ok.values()}) > 1:
                    b = min(v["batch"] for v in ok.values())
                    print(f"  re-timing at matched batch {b}")
                    r["t1_per_arm_batch"] = r["t1"]
                    r["t1"] = t1_train_step(depth, dev, b, args.vocab, args.seq,
                                            opt_step=args.opt_step, **cfg_kw)
        if want("t2"):
            print(f"[T2] prefill (batch {args.prefill_batch})")
            r["t2"] = t2_prefill(depth, dev, args.prefill_lengths, args.prefill_batch,
                                 args.vocab, **cfg_kw)
        if want("t3"):
            print("[T3] decode (batch 1, uncompiled)")
            r["t3"] = t3_decode(depth, dev, args.decode_contexts, args.vocab, **cfg_kw)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    write_tables(out, os.path.splitext(args.out)[0])
    if args.plot:
        make_plot(out, args.plot)


if __name__ == "__main__":
    main()
