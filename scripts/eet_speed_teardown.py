#!/usr/bin/env python3
"""Where did EET's speedup go? Teardown of the d8 step time, one factor at a time.

The sweep measured EET at 227.6ms against dense's 203.4ms, which cannot be right as a
statement about the architecture: at target_active_frac=0.10 the bell schedule leaves an
average active fraction of 0.606, and blocks plus attention are 64.8% of active FLOPs, so
the compute floor is 0.722x dense. Something outside the routed compute is eating it.

This runs the same model shape at several configurations and reports where the time goes,
so the answer is a measurement rather than an argument. Every arm shares the shape, the
batch and the compile setting; only one thing changes at a time.

    dense                     reference
    eet_p01                   exactly the configuration the sweep ran
    eet_no_ema                minus --eet-depth-weight-type ema, whose per-exit Python
                              loop writes scalars into a buffer inside the forward
    eet_no_aux                minus every auxiliary loss term
    eet_bare                  routing only: no aux, no router task-grad, no gumbel
    eet_bare_nocompile        the same, uncompiled, to price compilation itself

Report includes the analytic FLOP ratio for the schedule, so each arm can be read against
what it should cost.

Usage:
    python -m scripts.eet_speed_teardown --depth 8 --batch 2 --steps 12
"""
import argparse
import time

import torch

from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT
from nanochat.gpt import GPT
from scripts.eet_context_oracle import bell_capacities


def flop_ratio(depth, d, V, seq_len, window_pattern, target_frac, min_exit):
    blocks = 6 * 12 * d * d * depth
    head = 6 * d * V
    win = [(seq_len // 8 if c.upper() == 'S' else seq_len) for c in window_pattern]
    attn = [12 * d * min(win[i % len(win)], seq_len) for i in range(depth)]
    _, a = bell_capacities(depth, min_exit, target_frac, 'bell')
    tot = blocks + head + sum(attn)
    eet = blocks * (sum(a) / depth) + sum(ai * ai * al for ai, al in zip(a, attn)) + head
    return eet / tot, blocks / tot, head / tot, sum(attn) / tot, sum(a) / depth


def base_kwargs(depth, d, V, seq_len, window_pattern):
    return dict(sequence_len=seq_len, vocab_size=V, n_layer=depth,
                n_head=max(1, d // 128), n_kv_head=max(1, d // 128), n_embd=d,
                window_pattern=window_pattern)


def eet_kwargs(**over):
    kw = dict(
        use_eet=True, eet_global_router=True, eet_compute_skip=True,
        eet_min_exit_layer=1, eet_capacity_schedule='bell',
        eet_target_active_frac=0.10, eet_loss_variant='ce_guided',
        eet_warmup_frac=0.0, eet_explore_frac=0.0,
        eet_router_type='mlp1', eet_router_task_grad=True,
        eet_gumbel_temp_start=1.0, eet_gumbel_temp_end=0.1, eet_gumbel_hard=1,
        eet_ce_guided_lambda=1.0, eet_surprise_lambda=0.1,
        eet_depth_weight_type='ema', eet_capacity_alignment_lambda=1.0,
    )
    kw.update(over)
    return kw


def time_arm(name, model, x, y, steps, compile_it, fwd_kwargs):
    dev = x.device
    if compile_it:
        model = torch.compile(model, dynamic=False)
    for _ in range(3):                                  # warmup / compile
        try:
            loss = model(x, y, **fwd_kwargs)
            loss.backward()
            model.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"  {name:<22} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
            return None
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = model(x, y, **fwd_kwargs)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--model-dim", type=int, default=0, help="0 = depth*64")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = args.model_dim or args.depth * 64
    compile_it = not args.no_compile
    r, bs, hs, ats, mean_a = flop_ratio(args.depth, d, args.vocab, args.seq_len,
                                        args.window_pattern, 0.10, 1)
    print("=" * 76)
    print(f"  EET speed teardown   depth={args.depth} d_model={d} V={args.vocab} "
          f"T={args.seq_len} B={args.batch} device={dev.type}")
    print(f"  FLOP split: blocks {bs:.1%}  head {hs:.1%} (never routed)  attn {ats:.1%}")
    print(f"  mean active fraction {mean_a:.3f}  ->  compute floor {r:.3f}x dense")
    print("=" * 76)

    torch.manual_seed(0)
    g = torch.Generator(device='cpu').manual_seed(0)
    x = torch.randint(0, args.vocab, (args.batch, args.seq_len), generator=g).to(dev)
    y = torch.randint(0, args.vocab, (args.batch, args.seq_len), generator=g).to(dev)

    bk = base_kwargs(args.depth, d, args.vocab, args.seq_len, args.window_pattern)
    arms = [
        ("dense", GPT, dict(bk), {}, compile_it),
        ("eet_p01", EarlyExitGPT, {**bk, **eet_kwargs()},
         dict(eet_do_route=True, eet_phase=3), compile_it),
        ("eet_no_ema", EarlyExitGPT, {**bk, **eet_kwargs(eet_depth_weight_type='none')},
         dict(eet_do_route=True, eet_phase=3), compile_it),
        ("eet_no_aux", EarlyExitGPT,
         {**bk, **eet_kwargs(eet_depth_weight_type='none', eet_capacity_alignment_lambda=0.0,
                             eet_surprise_lambda=0.0, eet_ce_guided_lambda=0.0)},
         dict(eet_do_route=True, eet_phase=3), compile_it),
        ("eet_bare", EarlyExitGPT,
         {**bk, **eet_kwargs(eet_depth_weight_type='none', eet_capacity_alignment_lambda=0.0,
                             eet_surprise_lambda=0.0, eet_ce_guided_lambda=0.0,
                             eet_router_task_grad=False, eet_gumbel_temp_start=0.0)},
         dict(eet_do_route=True, eet_phase=3), compile_it),
        ("eet_bare_nocompile", EarlyExitGPT,
         {**bk, **eet_kwargs(eet_depth_weight_type='none', eet_capacity_alignment_lambda=0.0,
                             eet_surprise_lambda=0.0, eet_ce_guided_lambda=0.0,
                             eet_router_task_grad=False, eet_gumbel_temp_start=0.0)},
         dict(eet_do_route=True, eet_phase=3), False),
    ]

    results = {}
    for name, cls, kw, fwd, comp in arms:
        torch._dynamo.reset()
        torch.manual_seed(0)
        with torch.device("meta"):
            m = cls(GPTConfig(**kw))
        m.to_empty(device=dev)
        m.init_weights()
        m.train()
        ms = time_arm(name, m, x, y, args.steps, comp, fwd)
        results[name] = ms
        del m
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        if ms is not None:
            base = results.get("dense")
            rel = f"{ms/base:.3f}x dense" if base else ""
            print(f"  {name:<22}{ms:>9.1f} ms   {rel}")

    base = results.get("dense")
    if base and results.get("eet_bare"):
        print("\n  teardown")
        order = ["eet_p01", "eet_no_ema", "eet_no_aux", "eet_bare"]
        prev = None
        for n in order:
            if results.get(n) is None:
                continue
            if prev is not None:
                print(f"    removing {n.replace('eet_',''):<12} saved {prev - results[n]:>7.1f} ms")
            prev = results[n]
        print(f"\n  routing-only arm: {results['eet_bare']/base:.3f}x dense "
              f"against a compute floor of {r:.3f}x")
        excess = results['eet_bare'] / base - r
        print(f"  gather/scatter and launch overhead: {excess:+.3f}x of dense step time")
        if results['eet_bare'] < base:
            print("  -> the architecture IS faster once the aux machinery is out of the way.")
        else:
            print("  -> even bare routing is slower here: overhead exceeds the FLOP saving")
            print("     at this shape. Re-run at a larger --batch and --depth, where the")
            print("     saving grows and the per-launch overhead amortises.")


if __name__ == "__main__":
    main()
