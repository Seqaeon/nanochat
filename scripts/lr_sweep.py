"""
Learning-rate sweep for base and CCL (MoE perm) models.

For each depth, runs N training jobs — one per (model_type, lr_scale_factor) pair —
sequentially under DDP via torchrun. After all runs finish, reads the per-step
val_bpb saved in each run's checkpoint meta_*.json files and produces:

  • lr_sweep_depth_{D}.png — loss-curve overlay for all candidates
  • lr_sweep_depth_{D}.tsv — final/min val_bpb table

Usage:
    python -m scripts.lr_sweep --depth 8 --run-dir out/lr_sweep --fp8 --max-shards 170
    # or from lr_sweep.sh which handles env setup and calls this per depth.

Equivalent to running research_compare.py but sweeping LR instead of model type.
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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


# ---------------------------------------------------------------------------
# DDP runner (identical logic to research_compare.py)
# ---------------------------------------------------------------------------

def _resolve_runner() -> list[str]:
    """Return torchrun command prefix, capped to available GPUs."""
    nproc_requested = int(os.environ.get("NANOCHAT_NPROC", 8))
    gpu_count = max(torch.cuda.device_count(), 1)
    nproc = min(nproc_requested, gpu_count)
    if nproc < nproc_requested:
        print(
            f"[lr_sweep] Requested {nproc_requested} DDP workers but only "
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
# Model sizing helper (same as research_compare.py)
# ---------------------------------------------------------------------------

def estimate_tokens_from_base(depth: int, target_ratio: float = 10.5, tokenizer_dir: str | None = None) -> int:
    """Estimate Chinchilla-optimal token budget for a given depth."""
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
# Per-model-type configuration
# ---------------------------------------------------------------------------

# Base LRs for each model type.  These are the ×1 reference values.
# Changing these here changes what every scale factor multiplies against.
BASE_LRS: dict[str, dict[str, float]] = {
#    "base": {
#        "embedding_lr":   0.3,
#        "unembedding_lr": 0.004,
#        "matrix_lr":      0.02,
#        "scalar_lr":      0.5,
#    },
#    "moe_perm": {
#        # CCL reference: all LR groups set uniformly (matches research_compare.py)
#        "embedding_lr":   0.05,
#        "unembedding_lr": 0.05,
#        "matrix_lr":      0.05,
#        "scalar_lr":      0.05,
#    },
    "moe_no_perm": {
        "embedding_lr":   0.05,
        "unembedding_lr": 0.05,
        "matrix_lr":      0.05,
        "scalar_lr":      0.05,
    },
    "remixed-linear": {
        "embedding_lr":   0.05,
        "unembedding_lr": 0.05,
        "matrix_lr":      0.05,
        "scalar_lr":      0.05,
    },
}

# Architecture flags per model type (depth-dependent args added dynamically)
MODEL_ARCH_FLAGS: dict[str, list[str]] = {
#    "base": [],
#    "moe_perm": [
#        "--use-moe",
#        "--use-perm",
#        "--num-experts", "8",
#        "--selection-mode", "soft",
#        # --target-dim and --router-dim appended dynamically in run_lr_sweep()
#    ],
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
        print(f"[lr_sweep] Downloading data: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_pkl):
        cmd = [
            sys.executable, "-m", "scripts.tok_train",
            "--max-chars", "10000000",
            "--data-dir", data_dir,
            "--tokenizer-dir", tokenizer_dir,
        ]
        print(f"[lr_sweep] Training tokenizer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Loss-curve reader
# ---------------------------------------------------------------------------

def read_loss_curve(checkpoint_dir: Path) -> tuple[list[int], list[float]]:
    """
    Read all meta_*.json files saved by base_train's save_checkpoint() and
    return parallel lists of (training_token_count, val_bpb).

    base_train saves meta_*.json directly into checkpoint_dir (which is
    checkpoints_root / model_tag), with keys: step, val_bpb, total_batch_size.
    """
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
# Main sweep
# ---------------------------------------------------------------------------

def run_lr_sweep(args: argparse.Namespace) -> None:
    check_and_prepare_env(args)

    depth = args.depth
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Architecture sizing (mirrors research_compare.py exactly)
    aspect_ratio = 64
    head_dim = 128
    max_seq_len = 2048
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    device_batch_size = 16
    total_batch_size = 524288

    # target_dim for research branches: ~1/8th of model_dim, rounded to head_dim
    raw_target_dim = max(model_dim // 8, 1)
    target_dim = ((raw_target_dim + head_dim - 1) // head_dim) * head_dim
    target_dim = min(target_dim, model_dim)

    print("=" * 64)
    print(f"LR Sweep | Depth {depth} | Target tokens: {args.target_tokens:,}")
    print(f"Scale factors: {args.lr_scale_factors}")
    print(f"Models: {args.models}")
    print(f"model_dim={model_dim}  target_dim={target_dim}  eval_every={args.eval_every}")
    print("=" * 64)

    # Args shared by every run
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--max-seq-len", str(max_seq_len),
        "--device-batch-size", str(device_batch_size),
        "--total-batch-size", str(total_batch_size),
        "--target-tokens", str(args.target_tokens),
        "--eval-every", str(args.eval_every),
        "--core-metric-every", "0",
        "--sample-every", "-1",
        "--warmup-ratio", "0.3",
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

    # results[model_name][scale] = {tokens, bpbs, final_bpb, min_bpb, run_name}
    results: dict[str, dict[float, dict]] = {m: {} for m in args.models}

    for model_name in args.models:
        base_lrs = BASE_LRS[model_name]
        arch_flags = list(MODEL_ARCH_FLAGS[model_name])

        # Append dynamic dimension args for research branches
        if model_name != "base":
            arch_flags += [
                "--target-dim", str(target_dim),
                "--router-dim", str(target_dim),
            ]
            if model_name == "remixed-linear":
                arch_flags += [
                    "--context-dim", str(target_dim),
                    "--linear-basis-size", str(target_dim),
                ]

        print(f"\n{'='*64}")
        print(f"Model: {model_name}")
        print(f"Base LRs: emb={base_lrs['embedding_lr']}, unemb={base_lrs['unembedding_lr']}, "
              f"mat={base_lrs['matrix_lr']}, scl={base_lrs['scalar_lr']}")
        print(f"{'='*64}")

        for scale in args.lr_scale_factors:
            run_name = f"{model_name}_lr{scale:.1f}x"

            # base_train saves to: checkpoints_root / model_tag
            # So checkpoint files sit at:  run_dir / ckpt_{run_name} / {run_name} / meta_*.json
            ckpts_root = run_dir / f"ckpt_{run_name}"
            checkpoint_dir = ckpts_root / run_name  # where meta_*.json will live

            scaled_lr_args = [
                "--embedding-lr",   f"{base_lrs['embedding_lr']   * scale:.6g}",
                "--unembedding-lr", f"{base_lrs['unembedding_lr'] * scale:.6g}",
                "--matrix-lr",      f"{base_lrs['matrix_lr']      * scale:.6g}",
                "--scalar-lr",      f"{base_lrs['scalar_lr']      * scale:.6g}",
            ]

            run_args = (
                common_args
                + arch_flags
                + scaled_lr_args
                + [
                    "--checkpoints-dir", str(ckpts_root),
                    "--model-tag", run_name,
                ]
            )

            cmd = RUNNER + ["-m", "scripts.base_train"] + run_args

            print(f"\n--- {run_name} (×{scale:.1f}) ---")
            print(
                f"  emb={base_lrs['embedding_lr']*scale:.4g}  "
                f"unemb={base_lrs['unembedding_lr']*scale:.4g}  "
                f"mat={base_lrs['matrix_lr']*scale:.4g}  "
                f"scl={base_lrs['scalar_lr']*scale:.4g}"
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
                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        print(line, end="", flush=True)
                process.communicate()

                if process.returncode != 0:
                    print(f"[lr_sweep] {run_name} failed (exit {process.returncode}), skipping.")
                    continue

                tokens_list, bpbs_list = read_loss_curve(checkpoint_dir)
                if bpbs_list:
                    results[model_name][scale] = {
                        "tokens":    tokens_list,
                        "bpbs":      bpbs_list,
                        "final_bpb": bpbs_list[-1],
                        "min_bpb":   min(bpbs_list),
                        "run_name":  run_name,
                    }
                    print(
                        f"[lr_sweep] {run_name}: "
                        f"final_bpb={bpbs_list[-1]:.4f}  min_bpb={min(bpbs_list):.4f}  "
                        f"({len(bpbs_list)} eval points)"
                    )
                else:
                    print(f"[lr_sweep] {run_name}: no val_bpb values found in {checkpoint_dir}")

            except Exception as exc:
                print(f"[lr_sweep] Exception during {run_name}: {exc}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    _generate_report(results, depth, args.target_tokens, run_dir)


# ---------------------------------------------------------------------------
# Plotting and TSV
# ---------------------------------------------------------------------------

def _generate_report(
    results: dict[str, dict[float, dict]],
    depth: int,
    target_tokens: int,
    run_dir: Path,
) -> None:
    models_with_data = [m for m in results if results[m]]
    if not models_with_data:
        print("[lr_sweep] No results to plot.")
        return

    print("\n--- Generating Report ---")
    sns.set_theme(style="whitegrid")

    n_models = len(models_with_data)
    fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 6), squeeze=False)
    palette = sns.color_palette("husl", max(len(results[m]) for m in models_with_data))

    for col, model_name in enumerate(models_with_data):
        ax = axes[0][col]
        model_results = results[model_name]
        base_lrs = BASE_LRS[model_name]

        sorted_items = sorted(model_results.items())  # sort by scale factor
        for i, (scale, data) in enumerate(sorted_items):
            x = [t / 1e9 for t in data["tokens"]]          # billions of tokens
            y = data["bpbs"]
            mat_lr = base_lrs["matrix_lr"] * scale
            label = f"×{scale:.1f}  mat_lr={mat_lr:.4g}  final={data['final_bpb']:.4f}"
            ax.plot(x, y, color=palette[i], label=label, linewidth=2, marker=".", markersize=4)

        ax.set_title(f"{model_name} — Depth {depth}", fontsize=14)
        ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
        ax.set_ylabel("Validation BPB ↓", fontsize=12)
        ax.legend(title="LR Scale Factor", fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"LR Sweep — Depth {depth}  ({target_tokens/1e9:.1f}B tokens each)",
        fontsize=15,
    )
    plt.tight_layout()

    plot_path = run_dir / f"lr_sweep_depth_{depth}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # TSV summary
    tsv_path = run_dir / f"lr_sweep_depth_{depth}.tsv"
    with open(tsv_path, "w") as f:
        f.write("model\tscale_factor\tembedding_lr\tunembedding_lr\tmatrix_lr\tscalar_lr\tfinal_val_bpb\tmin_val_bpb\n")
        for model_name in models_with_data:
            base_lrs = BASE_LRS[model_name]
            for scale, data in sorted(results[model_name].items()):
                f.write(
                    f"{model_name}\t{scale:.1f}"
                    f"\t{base_lrs['embedding_lr']*scale:.6g}"
                    f"\t{base_lrs['unembedding_lr']*scale:.6g}"
                    f"\t{base_lrs['matrix_lr']*scale:.6g}"
                    f"\t{base_lrs['scalar_lr']*scale:.6g}"
                    f"\t{data['final_bpb']:.6f}"
                    f"\t{data['min_bpb']:.6f}\n"
                )
    print(f"Saved TSV:  {tsv_path}")

    # Console ranking
    print("\n--- Rankings by final val_bpb ---")
    for model_name in models_with_data:
        base_lrs = BASE_LRS[model_name]
        sorted_runs = sorted(results[model_name].items(), key=lambda kv: kv[1]["final_bpb"])
        print(f"\n  {model_name}:")
        for rank, (scale, data) in enumerate(sorted_runs, 1):
            winner = " ← WINNER" if rank == 1 else ""
            print(
                f"    #{rank}: ×{scale:.1f}  "
                f"mat_lr={base_lrs['matrix_lr']*scale:.4g}  "
                f"final_bpb={data['final_bpb']:.4f}  "
                f"min_bpb={data['min_bpb']:.4f}{winner}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LR sweep for base and CCL models at a given depth"
    )
    parser.add_argument("--depth", type=int, required=True, help="model depth")
    parser.add_argument("--run-dir", type=str, required=True, help="output root directory")
    parser.add_argument(
        "--target-tokens", type=int, default=1_000_000_000,
        help="tokens per LR candidate run (default: 1B)",
    )
    parser.add_argument(
        "--lr-scale-factors", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.0, 10.0],
        help="scale factors applied to each model's base LRs (default: 1 3 5 7 10)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["base", "moe_no_perm", "moe_perm", "remixed-linear"],
        choices=["base", "moe_no_perm", "moe_perm", "remixed-linear"],
        help="model types to sweep (default: all research branches)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=500,
        help="val bpb evaluation cadence in steps (default: 500)",
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
    run_lr_sweep(args)
