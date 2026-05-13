#!/usr/bin/env python3
"""Compute router diagnostics (entropy, load balance) from a saved MST checkpoint.

Usage:
    python scripts/mst_router_diagnostics.py --ckpt-dir /path/to/ckpt/base --step 1680

Loads the model, runs a forward pass on validation data, and prints router stats.
"""
import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPTConfig
from nanochat.mst import MST, MSTRouter
from nanochat.checkpoint_manager import load_checkpoint


def compute_router_diagnostics(model, data_tokens, n_batches=4, batch_size=16, seq_len=2048):
    """Run forward passes and collect router diagnostics."""
    device = next(model.parameters()).device
    N = model.config.mst_n_subs
    max_entropy = math.log2(N)

    from nanochat.mst import MSTTransition

    # Collect tracked modules: MSTRouter instances + MSTTransition (for free_for_all)
    tracked = {}  # name -> {'entropy': [], 'balance': []}

    # Track MSTRouter instances (aggregate_distribute, aggregate_proj)
    for name, module in model.named_modules():
        if isinstance(module, MSTRouter):
            tracked[name] = {'entropy': [], 'balance': [], 'type': 'MSTRouter'}

    # Track MSTTransition with free_for_all mode
    for name, module in model.named_modules():
        if isinstance(module, MSTTransition) and module.mode == 'free_for_all':
            tracked[name] = {'entropy': [], 'balance': [], 'type': 'FFA'}

    if not tracked:
        print("No routable modules found (no MSTRouter or free_for_all transitions).")
        return {}

    def make_mst_router_hook(key):
        def hook_fn(module, input, output):
            if hasattr(module, '_last_entropy') and module._last_entropy is not None:
                tracked[key]['entropy'].append(module._last_entropy.item())
            if hasattr(module, '_last_balance') and module._last_balance is not None:
                tracked[key]['balance'].append(module._last_balance.item())
        return hook_fn

    def make_ffa_hook(key):
        """Hook on MSTTransition.forward to compute entropy/balance from sub_routers."""
        def hook_fn(module, input, output):
            sub_outputs = input[0]  # list of N tensors (B, T, d)
            # Recompute routing weights from sub_routers
            route_logits = torch.stack([
                router(sub_out) for router, sub_out in zip(module.sub_routers, sub_outputs)
            ], dim=2)  # (B, T, N_sender, N_target)
            route_weights = F.softmax(route_logits, dim=-1)  # (B, T, N_sender, N_target)
            # Per-sender entropy (how concentrated is each sender's routing?)
            ent = -(route_weights * torch.log(route_weights + 1e-8)).sum(dim=-1)  # (B, T, N_sender)
            ent_bits = ent / math.log(2)
            tracked[key]['entropy'].append(ent_bits.mean().item())
            # Load balance: avg fraction each target receives
            target_load = route_weights.mean(dim=(0, 1, 2))  # (N_target,)
            balance = target_load.min() / (target_load.max() + 1e-8)
            tracked[key]['balance'].append(balance.item())
        return hook_fn

    handles = []
    for name, info in tracked.items():
        module = dict(model.named_modules())[name]
        if info['type'] == 'MSTRouter':
            handles.append(module.register_forward_hook(make_mst_router_hook(name)))
        elif info['type'] == 'FFA':
            handles.append(module.register_forward_hook(make_ffa_hook(name)))

    # Run forward passes
    n_tokens = data_tokens.shape[0]
    with torch.no_grad():
        for batch_i in range(n_batches):
            start = batch_i * batch_size * seq_len
            end = start + batch_size * seq_len
            if end > n_tokens:
                break
            x = data_tokens[start:end].reshape(batch_size, seq_len).to(device)
            targets = x.clone()
            model.train()
            _ = model(x, targets=targets)
            model.eval()

    for h in handles:
        h.remove()

    # Aggregate stats
    print(f"\n{'='*70}")
    print(f"Router Diagnostics (N={N}, max_entropy={max_entropy:.4f})")
    print(f"{'='*70}")

    all_entropies = []
    all_balances = []

    for name, info in tracked.items():
        ents = info['entropy']
        bals = info['balance']
        tag = f"[{info['type']}]"
        if ents:
            mean_ent = sum(ents) / len(ents)
            min_ent = min(ents)
            all_entropies.extend(ents)
            bal_str = f"{sum(bals)/len(bals):.4f}" if bals else "N/A"
            print(f"  {name:45s} {tag:12s} H_mean={mean_ent:.4f} H_min={min_ent:.4f} H/Hmax={mean_ent/max_entropy:.1%} balance={bal_str}")
            if bals:
                all_balances.extend(bals)

    if all_entropies:
        overall_mean = sum(all_entropies) / len(all_entropies)
        overall_min = min(all_entropies)
        overall_balance = sum(all_balances) / len(all_balances) if all_balances else 0

        print(f"\n  {'OVERALL':45s} {'':12s} H_mean={overall_mean:.4f} H_min={overall_min:.4f} H/Hmax={overall_mean/max_entropy:.1%} balance={overall_balance:.4f}")

        return {
            'router_entropy_mean': f'{overall_mean:.4f}',
            'router_entropy_min': f'{overall_min:.4f}',
            'load_balance_score': f'{overall_balance:.4f}',
        }

    return {}


def main():
    parser = argparse.ArgumentParser(description='Compute MST router diagnostics from checkpoint')
    parser.add_argument('--ckpt-dir', required=True, help='Path to checkpoint directory (containing model_*.pt)')
    parser.add_argument('--step', type=int, default=-1, help='Checkpoint step (-1 = auto-detect last)')
    parser.add_argument('--data-dir', default='data', help='Data directory for validation tokens')
    parser.add_argument('--tokenizer-dir', default='tokenizer', help='Tokenizer directory')
    parser.add_argument('--n-batches', type=int, default=4, help='Number of batches for diagnostics')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for diagnostics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Auto-detect step
    if args.step < 0:
        import glob
        model_files = glob.glob(os.path.join(args.ckpt_dir, 'model_*.pt'))
        if not model_files:
            print(f"No model files found in {args.ckpt_dir}")
            sys.exit(1)
        steps = [int(f.split('_')[-1].replace('.pt', '')) for f in model_files]
        args.step = max(steps)
        print(f"Auto-detected step: {args.step}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_dir} step {args.step}...")
    model_data, _, meta_data = load_checkpoint(args.ckpt_dir, args.step, 'cpu', load_optimizer=False)

    # Reconstruct config
    model_config = meta_data.get('model_config', {})
    config = GPTConfig(**model_config)

    print(f"Model config: n_layer={config.n_layer}, mst_n_subs={config.mst_n_subs}, "
          f"transition={config.mst_transition_mode}, final={config.mst_final_mode}")

    # Build model on CPU and load weights (45M params fits easily)
    model = MST(config)
    # Convert bf16 state dict to float32 for CPU compatibility
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model.load_state_dict(model_data, strict=False)
    if args.device != 'cpu':
        model = model.to(args.device)

    # Load some data for forward passes
    print(f"Loading validation data from {args.data_dir}...")
    import glob
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, 'fineweb*val*.bin')))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(args.data_dir, '*val*.bin')))
    if not shard_files:
        # Fallback: use random tokens
        print("  No validation data found, using random tokens")
        data_tokens = torch.randint(0, config.vocab_size, (args.n_batches * args.batch_size * config.sequence_len,))
    else:
        import numpy as np
        print(f"  Using {shard_files[0]}")
        data = np.memmap(shard_files[0], dtype=np.uint16, mode='r')
        data_tokens = torch.from_numpy(data.astype(np.int64))

    # Compute diagnostics
    results = compute_router_diagnostics(
        model, data_tokens,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seq_len=config.sequence_len,
    )

    if results:
        print(f"\n{'='*70}")
        print("Values for mst_results.csv:")
        print(f"  router_entropy_mean = {results['router_entropy_mean']}")
        print(f"  router_entropy_min  = {results['router_entropy_min']}")
        print(f"  load_balance_score  = {results['load_balance_score']}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
