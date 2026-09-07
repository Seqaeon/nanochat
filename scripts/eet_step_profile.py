#!/usr/bin/env python3
"""Attribute EET's step time on the machine that actually trains it.

Four theories for EET's slowdown have now been tested and all four were wrong:
the auxiliary losses (free once compiled), --eet-router-task-grad (a torch.compile
dtype bug, fixed), DDP itself (6-9%, and EET still faster), and
find_unused_parameters (identical either way). On an RTX 3050 Ti, EET under exactly
the wrapping base_train builds runs at 0.787x dense. On an H200 inside the training
loop the same configuration runs at ~1.16x.

Everything those benchmarks omit is in this script: the real optimizer (Muon plus
AdamW over the real parameter groups), the real batch size, and a separate timer per
phase. It prints forward, backward and optimizer time for dense and EET side by side,
so the answer comes off your own hardware instead of an extrapolation from mine.

    torchrun --standalone --nproc_per_node=1 -m scripts.eet_step_profile \
        --depth 8 --device-batch-size 128

Read it as: whichever phase differs by more than the FLOP ratio explains the gap.
Forward and backward should come in near 0.72x. If they do and the total does not,
the optimizer is the answer, and the router's parameter tensors are the thing to look
at -- Muon orthogonalises per tensor, so a handful of small tensors can cost more than
their size suggests.
"""
import argparse
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from nanochat.common import compute_init, print0
from nanochat.gpt import GPT, GPTConfig
from nanochat.eet import EarlyExitGPT


def build(cls, kw, device, ddp, fu):
    torch._dynamo.reset()
    torch.manual_seed(0)
    with torch.device("meta"):
        m = cls(GPTConfig(**kw))
    m.to_empty(device=device)
    m.init_weights()
    m.train()
    wrapped = m
    if ddp:
        wrapped = nn.parallel.DistributedDataParallel(
            m, device_ids=[device.index or 0], find_unused_parameters=fu)
    return m, torch.compile(wrapped, dynamic=False)


def phase_times(m, fn, x, y, fwd_kwargs, opt, steps):
    """Median forward, backward and optimizer time in ms, timed separately."""
    f, b, o = [], [], []
    for _ in range(steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss = fn(x, y, **fwd_kwargs)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if opt is not None:
            for o_ in opt:
                o_.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        m.zero_grad(set_to_none=True)
        f.append((t1 - t0) * 1e3); b.append((t2 - t1) * 1e3); o.append((t3 - t2) * 1e3)
    n = max(1, len(f) // 4)          # drop the warmest quarter, clocks drift on laptops
    med = lambda v: statistics.median(v[n:])
    return med(f), med(b), med(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--model-dim", type=int, default=0, help="0 = depth*64")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--window-pattern", default="SSSL")
    ap.add_argument("--target-active-frac", type=float, default=0.10)
    ap.add_argument("--no-optimizer", action="store_true", help="skip the optimizer phase")
    ap.add_argument("--no-ddp", action="store_true", help="do not wrap in DDP")
    args = ap.parse_args()

    ddp, rank, local_rank, world, device = compute_init("cuda")
    d = args.model_dim or args.depth * 64
    B, T = args.device_batch_size, args.seq_len
    base = dict(sequence_len=T, vocab_size=args.vocab, n_layer=args.depth,
                n_head=max(1, d // 128), n_kv_head=max(1, d // 128), n_embd=d,
                window_pattern=args.window_pattern)
    eet = dict(base, use_eet=True, eet_global_router=True, eet_compute_skip=True,
               eet_min_exit_layer=1, eet_capacity_schedule='bell',
               eet_target_active_frac=args.target_active_frac,
               eet_warmup_frac=0.0, eet_explore_frac=0.0, eet_router_type='mlp1',
               eet_gumbel_temp_start=1.0, eet_gumbel_temp_end=0.1, eet_gumbel_hard=1,
               eet_router_task_grad=False, eet_loss_variant='none',
               eet_depth_weight_type='none', eet_capacity_alignment_lambda=0.0)

    print0("=" * 74)
    print0(f"  EET step profile   depth={args.depth} d={d} V={args.vocab} B={B} T={T}")
    print0(f"  ddp={ddp and not args.no_ddp}  world={world}  optimizer={not args.no_optimizer}")
    print0("=" * 74)

    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, args.vocab, (B, T), generator=g).to(device)
    y = torch.randint(0, args.vocab, (B, T), generator=g).to(device)

    rows = {}
    for name, cls, kw, fwd, fu in (
        ("dense", GPT, base, {}, False),
        ("eet", EarlyExitGPT, eet, dict(eet_do_route=True, eet_phase=3), True),
    ):
        m, fn = build(cls, kw, device, ddp and not args.no_ddp, fu)
        opt = None
        if not args.no_optimizer:
            # The real thing: base_train's own optimizer groups, Muon included.
            opt = m.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2,
                                    matrix_lr=0.02, weight_decay=0.0)
            opt = opt if isinstance(opt, (list, tuple)) else [opt]
        for _ in range(3):
            loss = fn(x, y, **fwd); loss.backward(); m.zero_grad(set_to_none=True)
        f, b, o = phase_times(m, fn, x, y, fwd, opt, args.steps)
        rows[name] = (f, b, o, f + b + o)
        print0(f"  {name:<7} fwd {f:7.1f}   bwd {b:7.1f}   opt {o:7.1f}   total {f+b+o:7.1f} ms")
        del m, fn, opt
        torch.cuda.empty_cache()

    if len(rows) == 2:
        dn, ee = rows["dense"], rows["eet"]
        names = ("forward", "backward", "optimizer", "TOTAL")
        print0("")
        for i, nm in enumerate(names):
            r = ee[i] / dn[i] if dn[i] > 1e-9 else float('nan')
            flag = ""
            if nm != "TOTAL" and r > 0.9:
                flag = "  <-- not shrinking with the routed compute"
            print0(f"  {nm:<10}{ee[i]:8.1f} / {dn[i]:7.1f} = {r:5.3f}x dense{flag}")
        print0("")
        print0("  The routed FLOP ratio is ~0.72x. Any phase materially above that is where")
        print0("  the speedup is going; a phase near 0.72x is behaving as designed.")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
