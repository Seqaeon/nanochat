"""
Profile RemixedLinear forward pass to precisely identify bottlenecks.

Uses torch.profiler to get exact operation-level timing, memory traffic,
and GPU utilization breakdown.

Usage:
    python scripts/profile_forward.py \
        --checkpoint-dir /path/to/ckpt_remixed-linear/remixed-linear \
        --batch-size 64

    # Compare dense vs remix side by side:
    python scripts/profile_forward.py \
        --checkpoint-dir /path/to/dense --batch-size 64 --output dense_profile.txt
    python scripts/profile_forward.py \
        --checkpoint-dir /path/to/remix --batch-size 64 --output remix_profile.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.dirname(_SCRIPT_DIR), "/root/nanochat", _SCRIPT_DIR]:
    if os.path.isdir(os.path.join(_candidate, "nanochat")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

import torch
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import autodetect_device_type


@torch.no_grad()
def profile_forward(model, device, batch_size=64, seq_len=2048,
                    warmup=3, profile_steps=5, output_path=None):
    """Profile the model forward pass and print detailed breakdown."""
    model.eval()
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print(f"  Warming up ({warmup} steps)...")
    for _ in range(warmup):
        _ = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # ── 1. Manual per-op timing with CUDA events ─────────────────────────
    print(f"\n  Profiling with CUDA events ({profile_steps} steps)...")
    event_times = {}

    # Hook into key operations
    hooks = []
    op_starts = {}
    op_ends = {}

    def make_pre_hook(name):
        def hook(module, input):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                op_starts[name] = start
        return hook

    def make_post_hook(name):
        def hook(module, input, output):
            if device.type == 'cuda':
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                op_ends[name] = end
                if name not in event_times:
                    event_times[name] = []
        return hook

    # Register hooks on all named modules
    for name, mod in model.named_modules():
        # Skip the top-level model
        depth = name.count('.')
        if depth > 3:
            continue  # Don't go too deep
        if name == '':
            continue
        hooks.append(mod.register_forward_pre_hook(make_pre_hook(name)))
        hooks.append(mod.register_forward_hook(make_post_hook(name)))

    for step in range(profile_steps):
        op_starts.clear()
        op_ends.clear()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        # Collect timings
        for name in op_ends:
            if name in op_starts:
                ms = op_starts[name].elapsed_time(op_ends[name])
                if name not in event_times:
                    event_times[name] = []
                event_times[name].append(ms)

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── 2. torch.profiler for GPU kernel breakdown ───────────────────────
    print(f"  Running torch.profiler ({profile_steps} steps)...")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(profile_steps):
            _ = model(input_ids)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # ── 3. Report ─────────────────────────────────────────────────────────
    lines = []
    def p(msg=""):
        print(msg)
        lines.append(msg)

    tokens_per_step = batch_size * seq_len

    p(f"\n{'='*80}")
    p(f"Forward Pass Profile")
    p(f"{'='*80}")
    p(f"  Batch: {batch_size}, Seq: {seq_len}, Tokens/step: {tokens_per_step:,}")
    p(f"  Device: {device}")
    if device.type == 'cuda':
        p(f"  GPU: {torch.cuda.get_device_name(0)}")
    p()

    # Module-level timing
    p(f"{'='*80}")
    p(f"Module-Level Timing (avg over {profile_steps} steps)")
    p(f"{'='*80}")
    p(f"  {'Module':<55} {'Avg ms':>8} {'% total':>8}")
    p(f"  {'-'*55} {'-'*8} {'-'*8}")

    # Sort by average time
    avg_times = {}
    for name, times in event_times.items():
        avg_times[name] = sum(times) / len(times)

    # Find total forward time (model-level)
    total_ms = max(avg_times.values()) if avg_times else 1.0

    for name, avg_ms in sorted(avg_times.items(), key=lambda x: -x[1])[:30]:
        pct = 100 * avg_ms / total_ms
        p(f"  {name:<55} {avg_ms:>8.2f} {pct:>7.1f}%")

    p()

    # CUDA kernel breakdown from torch.profiler
    p(f"{'='*80}")
    p(f"Top CUDA Kernels (by total GPU time)")
    p(f"{'='*80}")

    key_averages = prof.key_averages()

    # Sort by total CUDA time
    cuda_events = []
    for evt in key_averages:
        cuda_time = evt.cuda_time_total if hasattr(evt, 'cuda_time_total') else 0
        if cuda_time > 0:
            cuda_events.append(evt)

    cuda_events.sort(key=lambda e: e.cuda_time_total, reverse=True)
    total_cuda_us = sum(e.cuda_time_total for e in cuda_events) or 1

    p(f"  {'Kernel':<55} {'CUDA ms':>8} {'%':>6} {'Calls':>6}")
    p(f"  {'-'*55} {'-'*8} {'-'*6} {'-'*6}")
    for evt in cuda_events[:25]:
        name = evt.key[:55]
        ms = evt.cuda_time_total / 1000
        pct = 100 * evt.cuda_time_total / total_cuda_us
        count = evt.count
        p(f"  {name:<55} {ms:>8.2f} {pct:>5.1f}% {count:>6}")

    p()

    # Memory summary
    if device.type == 'cuda':
        p(f"{'='*80}")
        p(f"Memory Traffic (from profiler)")
        p(f"{'='*80}")

        total_alloc = 0
        total_free = 0
        for evt in key_averages:
            alloc = getattr(evt, 'cuda_memory_usage', 0)
            if alloc > 0:
                total_alloc += alloc
            else:
                total_free += abs(alloc)

        p(f"  Allocated: {total_alloc / 1e9:.2f} GB")
        p(f"  Freed:     {total_free / 1e9:.2f} GB")
        p(f"  Net:       {(total_alloc - total_free) / 1e9:.2f} GB")

        # Peak memory
        peak = torch.cuda.max_memory_allocated() / 1e9
        p(f"  Peak allocated: {peak:.2f} GB")
        p()

    # Compute utilization
    if device.type == 'cuda':
        p(f"{'='*80}")
        p(f"Compute Utilization")
        p(f"{'='*80}")

        # Total wall time for profile_steps
        total_wall_s = sum(avg_times.get(name, 0) for name in avg_times
                          if avg_times[name] == max(avg_times.values())) * profile_steps / 1000

        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(input_ids)
        total_flops = flop_counter.get_total_flops()

        flops_per_step = total_flops
        measured_tflops = (flops_per_step * profile_steps) / (total_wall_s * 1e12) if total_wall_s > 0 else 0

        # H200 peak: ~989 TFLOPS bf16
        peak_tflops = 989.0
        utilization = 100 * measured_tflops / peak_tflops

        p(f"  FLOPs/step: {flops_per_step:.2e}")
        p(f"  Wall time/step: {total_wall_s*1000/profile_steps:.1f} ms")
        p(f"  Measured TFLOPS: {measured_tflops:.1f}")
        p(f"  H200 peak TFLOPS (bf16): {peak_tflops:.0f}")
        p(f"  Utilization: {utilization:.1f}%")
        p()

    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nProfile saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile forward pass")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    device = torch.device(device_type)

    step = args.step or find_last_step(args.checkpoint_dir)
    print(f"Loading step {step} from {args.checkpoint_dir}")

    model, tokenizer, meta_data = build_model(
        args.checkpoint_dir, step, device, phase="eval",
        tokenizer_dir=args.tokenizer_dir,
    )

    seq_len = args.seq_len or model.config.sequence_len
    is_remix = model.config.use_remix_linear
    print(f"Model: {'remix' if is_remix else 'dense'}, n_layer={model.config.n_layer}")

    profile_forward(
        model, device,
        batch_size=args.batch_size,
        seq_len=seq_len,
        warmup=args.warmup,
        profile_steps=args.profile_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
