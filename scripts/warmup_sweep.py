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
FULL training budget, e.g. 20B). Each candidate run is capped to --run-tokens (e.g.
100M). Warmup step counts are thus consistent across all candidate runs.

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


# ---------------------------------------------------------------------------
# DDP runner (identical to lr_sweep.py)
# ---------------------------------------------------------------------------

def _resolve_runner() -> list[str]:
    """Return torchrun command prefix, capped to available GPUs."""
    nproc_requested = int(os.environ.get("NANOCHAT_NPROC", 8))
    gpu_count = max(torch.cuda.device_count(), 1)
    nproc = min(nproc_requested, gpu_count)
    if nproc < nproc_requested:
        print(
            f"[warmup_sweep] Requested {nproc_requested} DDP workers but only "
            f"{gpu_count} GPU(s) available — using {nproc}."
        )
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun, "--standalone", f"--nproc_per_node={nproc}"]
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
    ]


RUNNER = _resolve_runner()


# ---------------------------------------------------------------------------
# Model sizing helper (same as lr_sweep.py)
# ---------------------------------------------------------------------------

def estimate_tokens_from_base(
    depth: int,
    target_ratio: float = 10.5,
    tokenizer_dir: str | None = None,
) -> int:
    from nanochat.gpt import GPT, GPTConfig

    vocab_size = 32768
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
    except Exception:
        pass

    aspect_ratio = 64
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )
    with torch.device("meta"):
        model = GPT(config)
    params_counts = model.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return int(scaling_params * target_ratio)


# ---------------------------------------------------------------------------
# Per-model configuration — EDIT THESE WHEN LR SWEEP PRODUCES NEW WINNERS
# ---------------------------------------------------------------------------

# Winner LRs from the LR sweep.  Applied uniformly across all LR groups.
FIXED_LRS: dict[str, dict[str, float]] = {
    "moe_perm": {
        "embedding_lr":   0.1,
        "unembedding_lr": 0.1,
        "matrix_lr":      0.1,
        "scalar_lr":      0.1,
    },
    "moe_no_perm": {
        "embedding_lr":   0.1,
        "unembedding_lr": 0.1,
        "matrix_lr":      0.1,
        "scalar_lr":      0.1,
    },
    "remixed-linear": {
        "embedding_lr":   0.05,
        "unembedding_lr": 0.05,
        "matrix_lr":      0.05,
        "scalar_lr":      0.05,
    },
}

# Architecture flags — depth-dependent args (target_dim, router_dim, …) are
# appended dynamically in run_warmup_sweep().
MODEL_ARCH_FLAGS: dict[str, list[str]] = {
    "moe_perm": [
        "--use-moe",
        "--use-perm",
        "--num-experts", "8",
        "--selection-mode", "soft",
    ],
    "moe_no_perm": [
        "--use-moe",
        "--num-experts", "8",
    ],
    "remixed-linear": [
        "--use-remixed-linear",
    ],
}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def check_and_prepare_env(args: argparse.Namespace) -> None:
    from nanochat.common import get_base_dir
    from nanochat.dataset import resolve_data_dir, list_parquet_files

    data_dir = resolve_data_dir()
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    shards = list_parquet_files(data_dir=data_dir)
    if not shards:
        num_files = args.max_shards if args.max_shards > 0 else 2
        cmd = [sys.executable, "-m", "nanochat.dataset", "-n", str(num_files), "--data-dir", data_dir]
        print(f"[warmup_sweep] Downloading data: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_pkl):
        cmd = [
            sys.executable, "-m", "scripts.tok_train",
            "--max-chars", "10000000",
            "--data-dir", data_dir,
            "--tokenizer-dir", tokenizer_dir,
        ]
        print(f"[warmup_sweep] Training tokenizer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


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

    # Architecture sizing
    aspect_ratio = 64
    head_dim = 128
    max_seq_len = 2048
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    # Device batch size scales down with depth to avoid OOM on deeper models.
    # Tune these values if you hit VRAM limits or want to squeeze more MFU.
    DEPTH_TO_DEVICE_BS: dict[int, int] = {
        8:  32,
        16: 16,
        24: 8,
    }
    device_batch_size = DEPTH_TO_DEVICE_BS.get(depth, 16)

    total_batch_size = 524288

    raw_target_dim = max(model_dim // 8, 1)
    target_dim = ((raw_target_dim + head_dim - 1) // head_dim) * head_dim
    target_dim = min(target_dim, model_dim)
    # Determine the full budget
    target_tokens = getattr(args, "target_tokens", 0)
    if target_tokens <= 0:
        target_tokens = estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)

    # Steps in the full training budget — used to convert warmup fracs to absolute steps
    full_budget_steps = target_tokens // total_batch_size

    # Filter requested models against what we have config for
    valid_models = [m for m in args.models if m in FIXED_LRS]
    skipped = set(args.models) - set(valid_models)
    if skipped:
        print(f"[warmup_sweep] WARNING: skipping unknown models: {skipped}")

    print("=" * 64)
    print(f"Warmup Sweep | Depth {depth}")
    print(f"Full budget (warmup basis): {target_tokens:,} ({full_budget_steps:,} steps)")
    print(f"Warmup fracs (early stop points): {args.warmup_fracs}")
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
        "--eval-every", "0",                               # NO evals (purely train loss)
        "--core-metric-every", "0",
        "--sample-every", "-1",
        "--adam-beta2", "0.99",
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

    env = os.environ.copy()

    # results[model_name][warmup_frac] = {tokens, bpbs, has_spike, first_below, final_bpb, min_bpb}
    results: dict[str, dict[float, dict]] = {m: {} for m in valid_models}

    for model_name in valid_models:
        lrs = FIXED_LRS[model_name]
        arch_flags = list(MODEL_ARCH_FLAGS[model_name])

        # Append dynamic dimension args
        arch_flags += ["--target-dim", str(target_dim), "--router-dim", str(target_dim)]
        if model_name == "remixed-linear":
            arch_flags += ["--context-dim", str(target_dim), "--linear-basis-size", str(target_dim)]

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

            warmup_pct = warmup_frac * 100
            run_name = f"{model_name}_wu{warmup_pct:.1f}pct"

            ckpts_root = run_dir / f"ckpt_{run_name}"
            checkpoint_dir = ckpts_root / run_name

            run_args = (
                common_args
                + arch_flags
                + lr_args
                + [
                    "--warmup-ratio", f"{warmup_ratio:.6g}",
                    "--early-stop-tokens", str(early_stop_tokens),
                    "--checkpoints-dir", str(ckpts_root),
                    "--model-tag", run_name,
                ]
            )

            cmd = RUNNER + ["-m", "scripts.base_train"] + run_args

            print(f"\n--- {run_name} ---")
            print(
                f"  warmup_frac={warmup_frac:.1%} of budget  "
                f"→ stops exactly at peak LR: {warmup_steps:,} steps ({early_stop_tokens/1e6:.1f}M tokens)"
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
                        if " | loss: " in line:
                            try:
                                parts = line.strip().split(" | loss: ")
                                step_str = parts[0].split(" ")[1].split("/")[0] # "00062"
                                step = int(step_str)
                                loss_str = parts[1].split(" | ")[0]
                                bpb = float(loss_str)
                                tokens_list.append(step * total_batch_size)
                                bpbs_list.append(bpb)
                            except Exception:
                                pass
                process.communicate()

                if process.returncode != 0:
                    print(f"[warmup_sweep] {run_name} failed (exit {process.returncode}), skipping.")
                    continue
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
        "--warmup-fracs", type=float, nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
        help="warmup fractions of target-tokens full budget (default: 0.5%% 1%% 2%% 5%% 10%%)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["moe_no_perm", "moe_perm", "remixed-linear"],
        choices=["moe_no_perm", "moe_perm", "remixed-linear"],
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

    args = parser.parse_args()
    run_warmup_sweep(args)
