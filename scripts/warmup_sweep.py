"""
Warmup-ratio sweep for CCL research models.

For each depth, runs N training jobs — one per (model_type, warmup_frac) pair —
sequentially under DDP via torchrun. Learning rates are fixed to the LR-sweep winners.
After all runs finish, reads per-step val_bpb and produces:

  • warmup_sweep_depth_{D}.png  — loss-curve overlay, spiky runs marked ⚠
  • warmup_sweep_depth_{D}.tsv  — per-run metrics table
  • Console rankings using 3-priority ordering:
      1. Stability   — runs with loss spike after 5M tokens are deprioritised
      2. Speed       — warmup that crosses BPB_THRESHOLD first wins
      3. Final BPB   — tiebreaker

Warmup fracs (e.g. 0.01, 0.05) are expressed as fractions of --target-tokens (the
FULL training budget, e.g. 20B). Each candidate run is capped to --early-stop-tokens
(or legacy --run-tokens), e.g. 100M. Warmup step counts are thus consistent across
all candidate runs.

Usage:
    python -m scripts.warmup_sweep \\
        --depth 8 --run-dir out/warmup_sweep \\
        --target-tokens 20000000000 --run-tokens 100000000 \\
        --max-shards 170 --fp8
    # or via warmup_sweep.sh which handles env setup.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from scripts._sweep_utils import resolve_runner, estimate_tokens_from_base, model_dims, check_and_prepare_env


RUNNER = resolve_runner()


# ---------------------------------------------------------------------------
# Per-model configuration — EDIT THESE WHEN LR SWEEP PRODUCES NEW WINNERS
# ---------------------------------------------------------------------------

# Winner LRs from the LR sweep.  Applied uniformly across all LR groups.
# Values synced from research_compare.py BEST_LRS (source of truth).
FIXED_LRS: dict[str, dict[str, float]] = {
    "base": {
        "embedding_lr":   0.3,
        "unembedding_lr": 0.004,
        "matrix_lr":      0.02,
        "scalar_lr":      0.5,
    },
    "moe_perm": {
        "embedding_lr":   0.5,
        "unembedding_lr": 0.5,
        "matrix_lr":      0.5,
        "scalar_lr":      2.5,
    },
    "moe_no_perm": {
        "embedding_lr":   0.1,
        "unembedding_lr": 0.3,
        "matrix_lr":      1.0,
        "scalar_lr":      0.1,
    },
    "remixed-linear": {
        "embedding_lr":   0.5,
        "unembedding_lr": 0.5,
        "matrix_lr":      0.5,
        "scalar_lr":      0.5,
    },
}

# Architecture flags — depth-dependent args (moe_embed_dim, moe_router_dim, …) are
# appended dynamically in run_warmup_sweep().
MODEL_ARCH_FLAGS: dict[str, list[str]] = {
    "base": [],
    "moe_perm": [
        "--use-moe",
        "--use-perm",
        "--moe-num-experts", "8",
    ],
    "moe_no_perm": [
        "--use-moe",
        "--moe-num-experts", "8",
    ],
    "remixed-linear": [
        "--use-remix-linear",
    ],
}


# ---------------------------------------------------------------------------
# Loss-curve reader (same as lr_sweep.py)
# ---------------------------------------------------------------------------

def read_loss_curve(checkpoint_dir: Path) -> tuple[list[int], list[float]]:
    """Read all meta_*.json files and return (training_token_counts, val_bpbs)."""
    meta_files = sorted(glob.glob(str(checkpoint_dir / "meta_*.json")))
    tokens_list: list[int] = []
    bpbs_list: list[float] = []

    for mf in meta_files:
        try:
            with open(mf) as f:
                data = json.load(f)
            bpb = data.get("val_bpb")
            if bpb is None:
                continue
            step = data.get("step", 0)
            tbs = data.get("total_batch_size", 524288)
            tokens_list.append(step * tbs)
            bpbs_list.append(float(bpb))
        except Exception:
            continue

    return tokens_list, bpbs_list


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyse_run(
    tokens: list[int],
    losses: list[float],
    spike_window_tokens: int = 5_000_000,
    spike_tolerance: float = 0.05,
) -> dict:
    """
    Compute per-run metrics for train_loss ranking:
      - Applies an Exponential Moving Average (alpha=0.1) to smooth the jagged training loss.
      - has_spike: True if the smoothed train loss spikes up significantly after the window.
      - final_train_loss: last smoothed loss.
      - min_train_loss: minimum smoothed loss.
    """
    if not tokens or not losses:
        return {
            "has_spike": False,
            "final_train_loss": float("inf"),
            "min_train_loss": float("inf"),
            "smoothed_losses": [],
        }

    alpha = 0.1 # Weight of the new value
    smoothed_losses: list[float] = []
    current_ema = losses[0]

    for x in losses:
        current_ema = alpha * x + (1.0 - alpha) * current_ema
        smoothed_losses.append(current_ema)

    has_spike = False
    for i in range(1, len(smoothed_losses)):
        if tokens[i] < spike_window_tokens:
            continue
        if smoothed_losses[i] > smoothed_losses[i - 1] + spike_tolerance:
            has_spike = True
            break

    return {
        "has_spike": has_spike,
        "final_train_loss": smoothed_losses[-1],
        "min_train_loss": min(smoothed_losses),
        "smoothed_losses": smoothed_losses,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_warmup_sweep(args: argparse.Namespace) -> None:
    check_and_prepare_env(args)

    depth = args.depth
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    aspect_ratio, head_dim, model_dim, target_dim = model_dims(depth)
    device_batch_size = {4: 8, 8: 32, 16: 16, 24: 8}.get(depth, 16)
    max_seq_len = 2048
    total_batch_size = 524288

    # Determine the full budget
    target_tokens = getattr(args, "target_tokens", 0)
    if target_tokens <= 0:
        target_tokens = estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)

    # Steps in the full training budget — used to convert warmup fracs to absolute steps
    full_budget_steps = target_tokens // total_batch_size

    # Global early stop based on explicit early-stop arg (preferred) or legacy run-tokens.
    # Default remains full training budget.
    if getattr(args, "early_stop_tokens", -1) > 0:
        global_early_stop_tokens = args.early_stop_tokens
    elif getattr(args, "run_tokens", 0) > 0:
        global_early_stop_tokens = args.run_tokens
    else:
        global_early_stop_tokens = target_tokens
    global_early_stop_steps = global_early_stop_tokens // total_batch_size

    # Filter requested models against what we have config for
    valid_models = [m for m in args.models if m in FIXED_LRS]
    skipped = set(args.models) - set(valid_models)
    if skipped:
        print(f"[warmup_sweep] WARNING: skipping unknown models: {skipped}")

    print("=" * 64)
    print(f"Warmup Sweep | Depth {depth}")
    print(f"Full budget (warmup basis): {target_tokens:,} ({full_budget_steps:,} steps)")
    print(f"Warmup fracs: {args.warmup_fracs}")
    print(f"Global early stop: {global_early_stop_tokens/1e6:.1f}M tokens ({global_early_stop_steps:,} steps)")
    print(f"Models: {valid_models}")
    print(f"model_dim={model_dim}  target_dim={target_dim}  eval_every=0 (disabled)")
    print("=" * 64)

    # Args shared by every run
    # --target-tokens sets the full LR schedule (20B)
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--max-seq-len", str(max_seq_len),
        "--device-batch-size", str(device_batch_size),
        "--total-batch-size", str(total_batch_size),
        "--target-tokens", str(target_tokens),        # full LR schedule
        "--log-every", str(args.log_every),
        "--eval-every", "0",                               # NO evals (purely train loss)
        "--core-metric-every", "0",
        "--sample-every", "-1",
        "--adam-beta2", "0.99",
        "--use-onecycle", str(args.use_onecycle),
    ]
    if args.compile:
        common_args.append("--compile")
    else:
        common_args.append("--no-compile")
    if args.fp8:
        common_args.append("--fp8")
    if args.tokenizer_dir:
        common_args.extend(["--tokenizer-dir", args.tokenizer_dir])
    if args.data_dir:
        common_args.extend(["--data-dir", args.data_dir])
    if args.max_shards > 0:
        common_args.extend(["--max-shards", str(args.max_shards)])
    if getattr(args, "research_dim", 0) > 0:
        common_args.extend(["--research-dim", str(args.research_dim)])
    common_args.extend([
        "--cclblock-modulation",     getattr(args, 'cclblock_modulation', 'weight'),
        "--cclblock-context-stream", getattr(args, 'cclblock_context_stream', 'local'),
        "--cclblock-ema-factor", str(getattr(args, 'cclblock_ema_factor', 0.99)),
        "--cclblock-stale-ctx-lag",  str(getattr(args, 'cclblock_stale_ctx_lag', 0)),
        # Novel ablation designs
        "--cclblock-sparse-gate-k",    str(getattr(args, 'cclblock_sparse_gate_k', 0)),
        "--cclblock-gate-temperature", str(getattr(args, 'cclblock_gate_temperature', 1.0)),
        "--cclblock-context-bank-size",str(getattr(args, 'cclblock_context_bank_size', 0)),
        "--cclblock-per-head-ctx",     str(getattr(args, 'cclblock_per_head_ctx', 0)),
    ])

    env = os.environ.copy()

    # results[model_name][warmup_frac] = {tokens, bpbs, has_spike, first_below, final_bpb, min_bpb}
    results: dict[str, dict[float, dict]] = {m: {} for m in valid_models}

    for model_name in valid_models:
        lrs = FIXED_LRS[model_name]
        arch_flags = list(MODEL_ARCH_FLAGS[model_name])

        # Append dynamic dimension args to ensure research parameters scale with depth
        if model_name != "base":
            arch_flags += [
                "--moe-embed-dim",  str(target_dim),
                "--moe-router-dim", str(target_dim),
                "--moe-num-experts", str(8),
            ]
            if model_name == "remixed-linear":
                arch_flags += [
                    "--remix-context-dim", str(target_dim),
                    "--remix-basis-size",  str(target_dim),
                ]

        lr_args = [
            "--embedding-lr",   f"{lrs['embedding_lr']:.6g}",
            "--unembedding-lr", f"{lrs['unembedding_lr']:.6g}",
            "--matrix-lr",      f"{lrs['matrix_lr']:.6g}",
            "--scalar-lr",      f"{lrs['scalar_lr']:.6g}",
        ]

        print(f"\n{'='*64}")
        print(f"Model: {model_name}")
        print(
            f"Fixed LRs: emb={lrs['embedding_lr']}  unemb={lrs['unembedding_lr']}  "
            f"mat={lrs['matrix_lr']}  scl={lrs['scalar_lr']}"
        )
        print(f"{'='*64}")

        for warmup_frac in args.warmup_fracs:
            # Absolute warmup steps = frac × full_budget_steps
            # warmup_ratio = warmup_steps / full_budget_steps (expressed relative to the 20B schedule)
            warmup_steps = int(warmup_frac * full_budget_steps)
            warmup_ratio = warmup_frac  # already a fraction of the full schedule
            early_stop_tokens = global_early_stop_tokens

            warmup_pct = warmup_frac * 100
            run_name = f"{model_name}_wu{warmup_pct:.1f}pct"

            ckpts_root = run_dir / f"ckpt_{run_name}"
            checkpoint_dir = ckpts_root / run_name
            step_loss_file = run_dir / f"{run_name}_step_loss.jsonl"

            run_args = (
                common_args
                + arch_flags
                + lr_args
                + [
                    "--warmup-ratio", f"{warmup_ratio:.6g}",
                    "--early-stop-tokens", str(early_stop_tokens),
                    "--step-loss-file", str(step_loss_file),
                    "--checkpoints-dir", str(ckpts_root),
                    "--model-tag", run_name,
                ]
            )

            cmd = RUNNER + ["-m", "scripts.base_train"] + run_args

            print(f"\n--- {run_name} ---")
            print(
                f"  warmup_frac={warmup_frac:.1%}  "
                f"→ peak at: {warmup_steps:,} steps  "
                f"→ stops at global peak: {global_early_stop_steps:,} steps ({early_stop_tokens/1e6:.1f}M tokens)"
            )
            print(f"  cmd: {' '.join(cmd)}")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                tokens_list: list[int] = []
                bpbs_list: list[float] = []

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        print(line, end="", flush=True)
                        if " | loss: " in line and "step " in line:
                            # Robust parse even with logger/timestamp prefixes.
                            m = re.search(r"step\s+(\d+)\s*/\s*\d+.*?\|\s*loss:\s*([0-9eE+\-\.]+)", line)
                            if m:
                                step = int(m.group(1))
                                bpb = float(m.group(2))
                                tokens_list.append(step * total_batch_size)
                                bpbs_list.append(bpb)
                process.communicate()

                if process.returncode != 0:
                    print(f"[warmup_sweep] {run_name} failed (exit {process.returncode}), skipping.")
                    continue
                # Prefer per-step loss file if available (captures every optimizer step).
                if step_loss_file.exists():
                    try:
                        step_tokens: list[int] = []
                        step_losses: list[float] = []
                        with open(step_loss_file, "r", encoding="utf-8") as f:
                            for raw in f:
                                rec = json.loads(raw)
                                step_tokens.append(int(rec["tokens"]))
                                step_losses.append(float(rec["loss"]))
                        if step_losses:
                            tokens_list = step_tokens
                            bpbs_list = step_losses
                    except Exception as exc:
                        print(f"[warmup_sweep] {run_name}: failed to read step loss file ({exc}); using stdout parser fallback.")

                if bpbs_list:
                    metrics = analyse_run(
                        tokens_list, bpbs_list,
                        spike_window_tokens=5_000_000,
                        spike_tolerance=0.05,
                    )
                    results[model_name][warmup_frac] = {
                        "tokens":    tokens_list,
                        "bpbs":      bpbs_list,
                        "warmup_steps": warmup_steps,
                        "run_name":  run_name,
                        **metrics,
                    }
                    spike_flag = " ⚠ SPIKE" if metrics["has_spike"] else ""
                    print(
                        f"[warmup_sweep] {run_name}: "
                        f"final_train_loss={metrics['final_train_loss']:.4f}  "
                        f"min_train_loss={metrics['min_train_loss']:.4f}"
                        f"{spike_flag}"
                    )
                else:
                    print(f"[warmup_sweep] {run_name}: no train loss values found in stdout")

            except Exception as exc:
                print(f"[warmup_sweep] Exception during {run_name}: {exc}")

        if results[model_name]:
            _plot_and_rank_model(
                model_name, results[model_name], depth,
                target_tokens, run_dir,
            )
        else:
            print(f"[warmup_sweep] {model_name}: no successful runs — skipping per-model plot.")

    # -----------------------------------------------------------------------
    # Aggregate report: combined plot + TSV across all models
    # -----------------------------------------------------------------------
    _generate_aggregate_report(results, depth, target_tokens, run_dir)


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

def _rank_key(data: dict) -> tuple:
    """
    Two-priority ranking key (lower is better):
      1. Stability — spiky runs last
      2. Peak Loss — lowest smoothed training loss at peak LR wins
    """
    spike_penalty = 1 if data["has_spike"] else 0
    return (spike_penalty, data["final_train_loss"])


# ---------------------------------------------------------------------------
# Per-model plot + ranking (called as soon as a model's fracs are done)
# ---------------------------------------------------------------------------

def _plot_and_rank_model(
    model_name: str,
    model_results: dict[float, dict],
    depth: int,
    full_budget_tokens: int,
    run_dir: Path,
) -> None:
    print(f"\n{'='*64}")
    print(f"Results for {model_name} — Depth {depth}")
    print(f"{'='*64}")

    sns.set_theme(style="whitegrid")
    n_fracs = len(model_results)
    palette = sns.color_palette("husl", max(n_fracs, 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_items = sorted(model_results.items())
    for i, (warmup_frac, data) in enumerate(sorted_items):
        x = [t / 1e9 for t in data["tokens"]] # tokens in billions
        y = data["smoothed_losses"] # using smoothed train loss
        spike_tag = " ⚠" if data["has_spike"] else ""
        label = (
            f"{warmup_frac:.1%}  ({data['warmup_steps']:,} steps)  "
            f"peak_loss={data['final_train_loss']:.4f}{spike_tag}"
        )
        color = palette[i % len(palette)]
        lw = 1.5 if data["has_spike"] else 2.5
        ls = "--" if data["has_spike"] else "-"
        # Plot the EMA lines
        ax.plot(x, y, color=color, label=label, linewidth=lw, linestyle=ls, marker="", markersize=0)
        # Plot raw loss faintly
        ax.plot(x, data["bpbs"], color=color, alpha=0.15, linewidth=0.8)

    ax.set_title(
        f"{model_name} — Depth {depth}  "
        f"(warmup basis: {full_budget_tokens/1e9:.0f}B)",
        fontsize=13,
    )
    ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
    ax.set_ylabel("Train Loss (EMA α=0.1) ↓", fontsize=12)
    ax.legend(title="Warmup (% of full budget)", fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.replace("-", "_")
    plot_path = run_dir / f"warmup_{safe_name}_depth_{depth}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved model plot: {plot_path}")

    # Console ranking for this model
    sorted_runs = sorted(model_results.items(), key=lambda kv: _rank_key(kv[1]))
    print(f"\n  Rankings for {model_name} (priority: stability > lowest peak train loss):")
    for rank, (warmup_frac, data) in enumerate(sorted_runs, 1):
        winner = " ← WINNER" if rank == 1 else ""
        spike_flag = " ⚠ SPIKE" if data["has_spike"] else ""
        print(
            f"    #{rank}: {warmup_frac:.1%}  ({data['warmup_steps']:,} steps)  "
            f"peak_loss={data['final_train_loss']:.4f}  "
            f"min_loss={data['min_train_loss']:.4f}"
            f"{spike_flag}{winner}"
        )


# ---------------------------------------------------------------------------
# Aggregate plot + TSV across all models (called at the very end)
# ---------------------------------------------------------------------------

def _generate_aggregate_report(
    results: dict[str, dict[float, dict]],
    depth: int,
    full_budget_tokens: int,
    run_dir: Path,
) -> None:
    models_with_data = [m for m in results if results[m]]
    if not models_with_data:
        print("[warmup_sweep] No results to aggregate.")
        return

    print("\n--- Generating Aggregate Report ---")
    sns.set_theme(style="whitegrid")

    n_models = len(models_with_data)
    fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 6), squeeze=False)
    palette = sns.color_palette("husl", 5)

    for col, model_name in enumerate(models_with_data):
        ax = axes[0][col]
        model_results = results[model_name]

        sorted_items = sorted(model_results.items())
        for i, (warmup_frac, data) in enumerate(sorted_items):
            x = [t / 1e9 for t in data["tokens"]] # tokens in billions
            y = data["smoothed_losses"] # using smoothed train loss
            spike_tag = " ⚠" if data["has_spike"] else ""
            label = (
                f"{warmup_frac:.1%}  ({data['warmup_steps']:,} steps)  "
                f"peak_loss={data['final_train_loss']:.4f}{spike_tag}"
            )
            color = palette[i % len(palette)]
            lw = 1.5 if data["has_spike"] else 2.5
            ls = "--" if data["has_spike"] else "-"
            # Plot the EMA lines
            ax.plot(x, y, color=color, label=label, linewidth=lw, linestyle=ls, marker="", markersize=0)
            # Plot raw loss faintly
            ax.plot(x, data["bpbs"], color=color, alpha=0.15, linewidth=0.8)

        ax.set_title(f"{model_name} — Depth {depth}", fontsize=14)
        ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
        ax.set_ylabel("Train Loss (EMA α=0.1) ↓", fontsize=12)
        ax.legend(title="Warmup (% of full budget)", fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Warmup Sweep — Depth {depth}  "
        f"(warmup basis: {full_budget_tokens/1e9:.0f}B)",
        fontsize=15,
    )
    plt.tight_layout()

    plot_path = run_dir / f"warmup_sweep_depth_{depth}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved aggregate plot: {plot_path}")

    # TSV summary
    tsv_path = run_dir / f"warmup_sweep_depth_{depth}.tsv"
    with open(tsv_path, "w") as f:
        f.write(
            "model\twarmup_frac\twarmup_steps"
            "\thas_spike\tfinal_train_loss\tmin_train_loss\n"
        )
        for model_name in models_with_data:
            for warmup_frac, data in sorted(results[model_name].items()):
                f.write(
                    f"{model_name}\t{warmup_frac:.4f}\t{data['warmup_steps']}"
                    f"\t{int(data['has_spike'])}"
                    f"\t{data['final_train_loss']:.6f}\t{data['min_train_loss']:.6f}\n"
                )
    print(f"Saved aggregate TSV:  {tsv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Warmup ratio sweep for CCL research models at a given depth"
    )
    parser.add_argument("--depth", type=int, required=True, help="model depth")
    parser.add_argument("--run-dir", type=str, required=True, help="output root directory")
    parser.add_argument(
        "--target-tokens", type=int, default=0,
        help="full training budget used to calculate warmup step counts (default: 0 = dynamic calculation from Depth)",
    )
    parser.add_argument(
        "--run-tokens", type=int, default=0,
        help="legacy alias for per-run early stop tokens (default: 0 = ignored unless --early-stop-tokens is unset)",
    )
    parser.add_argument(
        "--early-stop-tokens", type=int, default=-1,
        help="per-run early stop length in tokens (default: -1 = full --target-tokens budget)",
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="print base_train logs to console every N steps (default: 1)",
    )
    parser.add_argument(
        "--warmup-fracs", type=float, nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
        help="warmup fractions of target-tokens full budget (default: 0.5%% 1%% 2%% 5%% 10%%)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["moe_no_perm", "moe_perm", "remixed-linear"],
        choices=["base" ,"moe_no_perm", "moe_perm", "remixed-linear"],
        help="model types to sweep",
    )
    parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=-1)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="enable/disable torch.compile (default: enabled)",
    )
    parser.add_argument(
        "--use-onecycle", type=int, default=1, choices=[0, 1],
        help="research branches: 1=OneCycle, 0=use base schedule",
    )
    # CCL block modulation
    parser.add_argument("--cclblock-modulation", type=str, default="weight",
                        choices=["weight", "normalization"],
                        help="CCL block strategy passed to remixed-linear runs")
    parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                        choices=["local", "shifted", "ema", "selective", "multiscale"],
                        help="Context stream type")
    parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                        help="EMA factor for the legacy EMAContextStream")
    parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                        help="Design C stale context lag (0=disabled)")
    # Novel ablation designs
    parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                        help="Design 3: sparse top-k basis gate (0=off, N=top-N)")
    parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                        help="Design 6: gate temperature (<1=sharper, >1=softer)")
    parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                        help="Design 4: context prototype bank size (0=off, e.g. 16)")
    parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                        help="Design 7: separate attn/ffn context projections (0=off, 1=on)")
    # Research dimension override
    parser.add_argument("--research-dim", type=int, default=0, help="override default 1/8th model_dim for research branches")

    args = parser.parse_args()
    run_warmup_sweep(args)
