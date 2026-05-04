"""
Inference throughput benchmark for Dense vs RemixedLinear models.

Measures tokens/sec, peak memory, and actual hardware FLOPs.
Designed to be run on a single GPU for fair comparison.

Usage:
    # Dense d12 baseline
    python scripts/inference_benchmark.py \
        --checkpoint-dir /path/to/ckpt_base/base --batch-size 8

    # RemixedLinear d12
    python scripts/inference_benchmark.py \
        --checkpoint-dir /path/to/ckpt_remixed-linear/remixed-linear --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure repo root is on path when run from Modal volumes or other non-installed environments.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.dirname(_SCRIPT_DIR),   # scripts/../  (standard layout)
    "/root/nanochat",               # Modal default mount point
    _SCRIPT_DIR,                    # fallback: script itself at repo root
]:
    if os.path.isdir(os.path.join(_candidate, "nanochat")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import autodetect_device_type


@torch.no_grad()
def count_hardware_flops(model, device, batch_size: int = 8,
                         seq_len: int = 2048) -> dict:
    """Count actual hardware FLOPs using torch.utils.flop_counter."""
    model.eval()
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    from torch.utils.flop_counter import FlopCounterMode

    # Run once to warm up
    _ = model(input_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Count FLOPs
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        _ = model(input_ids)

    total_flops = flop_counter.get_total_flops()
    tokens = batch_size * seq_len
    flops_per_token = total_flops / tokens

    # Get per-module breakdown (top-level only)
    flops_by_module = {}
    try:
        for name, mod_flops in flop_counter.get_flop_counts().items():
            total_mod = sum(mod_flops.values())
            if total_mod > 0:
                flops_by_module[str(name)] = total_mod
    except Exception:
        pass  # older PyTorch versions may not support this

    return {
        'hw_total_flops': total_flops,
        'hw_flops_per_token': flops_per_token,
        'hw_flops_by_module': flops_by_module,
    }


@torch.no_grad()
def benchmark_throughput(model, device, batch_size: int = 8,
                         seq_len: int = 2048, warmup_steps: int = 5,
                         measure_steps: int = 20) -> dict:
    """Measure inference throughput in tokens/sec."""
    model.eval()
    vocab_size = model.config.vocab_size

    # Generate random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print(f"  Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        _ = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Reset memory stats after warmup
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Measure
    print(f"  Measuring ({measure_steps} steps)...")
    times = []
    for _ in range(measure_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    tokens_per_step = batch_size * seq_len
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    peak_mem_gb = 0.0
    if device.type == 'cuda':
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    results = {
        'tokens_per_sec': tokens_per_step / avg_time,
        'avg_latency_ms': avg_time * 1000,
        'std_latency_ms': std_time * 1000,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'tokens_per_step': tokens_per_step,
        'peak_memory_gb': round(peak_mem_gb, 2),
        'measure_steps': measure_steps,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference throughput benchmark")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to checkpoint directory containing model_XXXXXX.pt and meta_XXXXXX.json")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for benchmark")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: from model config)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--measure-steps", type=int, default=20, help="Measurement iterations")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--no-flop-count", action="store_true", help="Skip hardware FLOP counting")
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    device = torch.device(device_type)

    # Find latest step if not specified
    step = args.step
    if step is None:
        step = find_last_step(args.checkpoint_dir)
    print(f"Loading step {step} from {args.checkpoint_dir}")

    # Build model from checkpoint (reads model_config from meta JSON)
    model, tokenizer, meta_data = build_model(
        args.checkpoint_dir, step, device, phase="eval",
        tokenizer_dir=args.tokenizer_dir,
    )

    # Override seq_len if requested
    seq_len = args.seq_len or model.config.sequence_len

    # Model info
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    try:
        est_total_flops, est_active_flops, active_params = model.estimate_flops()
    except Exception:
        est_total_flops = est_active_flops = active_params = 0
    is_remix = config.use_remix_linear
    model_tag = "remix" if is_remix else "dense"

    print(f"\n{'='*60}")
    print(f"Inference Throughput Benchmark")
    print(f"{'='*60}")
    print(f"  Device:        {device_type}")
    if device_type == 'cuda':
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Step:          {step}")
    print(f"  Model:         {model_tag}")
    print(f"  n_layer:       {config.n_layer}")
    print(f"  n_embd:        {config.n_embd}")
    print(f"  Total params:  {total_params:,}")
    print(f"  Active params: {active_params:,}")
    print(f"  Est. Total FLOPs/tok: {est_total_flops:.2e}")
    print(f"  Est. Active FLOPs/tok: {est_active_flops:.2e}")
    print(f"  Batch:         {args.batch_size}")
    print(f"  Seq len:       {seq_len}")
    print(f"  Compile:       {args.compile}")
    if is_remix:
        rl_kw = config.remixed_linear_kwargs or {}
        print(f"  K templates:   {rl_kw.get('n_templates', '?')}")
        print(f"  Chunk size:    {rl_kw.get('chunk_routing_size', '?')}")
    print()

    # Hardware FLOP counting (before compile, on uncompiled model)
    hw_flops_results = {}
    if not args.no_flop_count:
        print("Counting actual hardware FLOPs...")
        try:
            hw_flops_results = count_hardware_flops(model, device,
                                                     batch_size=args.batch_size,
                                                     seq_len=seq_len)
            hw_fpt = hw_flops_results['hw_flops_per_token']
            print(f"  Hardware FLOPs/token: {hw_fpt:.2e}")
            print(f"  Hardware total FLOPs:  {hw_flops_results['hw_total_flops']:.2e}")
            if est_active_flops > 0:
                print(f"  HW/Est. Active ratio:  {hw_fpt / est_active_flops:.2f}x")
            print()
        except Exception as e:
            print(f"  FLOP counting failed: {e}")
            print(f"  (requires PyTorch >= 2.1 with torch.utils.flop_counter)")
            print()

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Run benchmark
    print("Running benchmark...")
    results = benchmark_throughput(
        model, device,
        batch_size=args.batch_size,
        seq_len=seq_len,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    # Add model info to results
    results['model_tag'] = model_tag
    results['n_layer'] = config.n_layer
    results['n_embd'] = config.n_embd
    results['total_params'] = total_params
    results['active_params'] = active_params
    results['est_total_flops_per_token'] = est_total_flops
    results['est_active_flops_per_token'] = est_active_flops
    results.update(hw_flops_results)
    results['device'] = device_type
    results['step'] = step
    if device_type == 'cuda':
        results['gpu_name'] = torch.cuda.get_device_name(0)
    results['compiled'] = args.compile
    if is_remix:
        rl_kw = config.remixed_linear_kwargs or {}
        results['n_templates'] = rl_kw.get('n_templates', None)
        results['chunk_routing_size'] = rl_kw.get('chunk_routing_size', None)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({model_tag})")
    print(f"{'='*60}")
    print(f"  Throughput:      {results['tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Avg latency:     {results['avg_latency_ms']:.1f} ms ± {results['std_latency_ms']:.1f} ms")
    print(f"  Peak memory:     {results['peak_memory_gb']:.2f} GB")
    print(f"  Tokens/step:     {results['tokens_per_step']:,}")
    if 'hw_flops_per_token' in results:
        print(f"  HW FLOPs/token:  {results['hw_flops_per_token']:.2e}")
        throughput = results['tokens_per_sec']
        hw_tflops = results['hw_flops_per_token'] * throughput / 1e12
        print(f"  HW TFLOPS:       {hw_tflops:.1f}")
    print()

    # Save results
    out_path = args.output or f"inference_bench_{model_tag}_d{config.n_layer}.json"
    # Remove non-serializable items
    save_results = {k: v for k, v in results.items() if k != 'hw_flops_by_module'}
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
