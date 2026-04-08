"""
Phased coordinate-descent LR sweep for nanochat research model variants.

Now operates on **absolute LR values** directly. All research runs are trained
with --disable-mu-p so gpt.py does NOT apply the (model_dim/768)^-0.5 AdamW
scaling — the flags you pass are the exact LRs used in the optimizer.

Three phases:
  Phase 1 — Grid sweep.
    Train each model with all 4 LR groups set to the same value, sweeping over
    a range of absolute LR candidates. Identifies the best overall LR.

  Phase 2 — Coordinate descent around the phase-1 winner.
    Hold all groups at the phase-1 best LR, perturb one group at a time by
    multiplying it by a set of per-group multipliers.
    Answers: "does group X benefit from being higher/lower than the uniform winner?"

  Phase 3 — Optional joint refinement.
    Random log-space search around the phase-2 winner configuration.

Usage:
    python -m scripts.actual_lr_research_sweep \
        --depth 8 --run-dir out/actual_lr_sweep \
        --early-stop-tokens 100000000 --target-tokens 20000000000 \
        --models moe_perm moe_no_perm remixed-linear \
        --phase1-lrs 0.003 0.01 0.03 0.1 0.3 1.0 --fp8
    # or via actual_lr_research_sweep.sh which handles env setup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from scripts._sweep_utils import resolve_runner, estimate_tokens_from_base, model_dims


# ---------------------------------------------------------------------------
# LR group names
# ---------------------------------------------------------------------------

LR_GROUPS = ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"]

# Default starting-point LRs (used as the center for phase 2/3 if no --lr-star is given).
# These are absolute values — μP scaling is disabled for research runs.
DEFAULT_STARTING_LRS: dict[str, dict[str, float]] = {
    "base":          {"embedding_lr": 0.003, "unembedding_lr": 0.003, "matrix_lr": 0.02, "scalar_lr": 0.003},
    "moe_no_perm":   {"embedding_lr": 0.003, "unembedding_lr": 0.003, "matrix_lr": 0.02, "scalar_lr": 0.003},
    "moe_perm":      {"embedding_lr": 0.003, "unembedding_lr": 0.003, "matrix_lr": 0.02, "scalar_lr": 0.003},
    "remixed-linear":{"embedding_lr": 0.003, "unembedding_lr": 0.003, "matrix_lr": 0.02, "scalar_lr": 0.003},
}

MODEL_ARCH_FLAGS: dict[str, list[str]] = {
    "base":          [],
    "moe_no_perm":   ["--use-moe"],
    "moe_perm":      ["--use-moe", "--use-perm"],
    "remixed-linear": ["--use-remix-linear"],
}


# ---------------------------------------------------------------------------
# DDP runner
# ---------------------------------------------------------------------------

def _resolve_runner() -> list[str]:
    return resolve_runner()


RUNNER = _resolve_runner()


# ---------------------------------------------------------------------------
# Phase config generation (absolute LRs, no scale indirection)
# ---------------------------------------------------------------------------

@dataclass
class SweepRun:
    phase: int
    model: str
    run_name: str
    lrs: dict[str, float]
    meta: dict


def make_phase_runs(
    models: list[str],
    phase1_lrs: list[float],
    phase2_multipliers: list[float],
    phase3_samples: int,
    phase3_log_radius: float,
    lr_star: dict[str, dict[str, float]],  # model → {group → absolute lr}
    seed: int,
) -> list[SweepRun]:
    """Generate all sweep runs.

    Phase 1: sweep absolute uniform LR values (all groups the same).
    Phase 2: coordinate descent — all groups at lr_star, perturb one at a time.
    Phase 3: random log-space refinement around the phase-2 center.
    """
    rng = random.Random(seed)
    runs: list[SweepRun] = []

    # Phase 1: uniform absolute LR sweep — all LR groups set to the same value.
    for model in models:
        for lr in phase1_lrs:
            lrs = {k: lr for k in LR_GROUPS}
            runs.append(SweepRun(
                phase=1,
                model=model,
                run_name=f"{model}_p1_lr{lr:g}",
                lrs=lrs,
                meta={"lr": lr},
            ))

    # Phase 2: coordinate descent around lr_star.
    for model in models:
        center = lr_star.get(model, {k: phase1_lrs[len(phase1_lrs)//2] for k in LR_GROUPS})
        for group in LR_GROUPS:
            for mult in phase2_multipliers:
                lrs = dict(center)
                lrs[group] = center[group] * mult
                runs.append(SweepRun(
                    phase=2,
                    model=model,
                    run_name=f"{model}_p2_{group}_x{mult:g}",
                    lrs=lrs,
                    meta={"group": group, "mult": mult, "center": dict(center)},
                ))

    # Phase 3: random log-space refinement around the phase-2 center.
    for model in models:
        center = lr_star.get(model, {k: phase1_lrs[len(phase1_lrs)//2] for k in LR_GROUPS})
        for i in range(phase3_samples):
            lrs = {g: v * (10 ** rng.uniform(-phase3_log_radius, phase3_log_radius))
                   for g, v in center.items()}
            runs.append(SweepRun(
                phase=3,
                model=model,
                run_name=f"{model}_p3_rand{i:02d}",
                lrs=lrs,
                meta={"idx": i, "log_radius": phase3_log_radius, "seed": seed},
            ))

    return runs


def dump_run_configs(runs: list[SweepRun], config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for idx, run in enumerate(runs):
        obj = {"run_index": idx, **asdict(run)}
        with open(config_dir / f"run_{idx:03d}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Loss parsing — from step_loss.jsonl (mean over last 10% of steps)
# ---------------------------------------------------------------------------

def _read_step_losses(step_loss_file: Path) -> tuple[list[int], list[float]]:
    """Return (tokens_list, loss_list) from a step_loss.jsonl file."""
    if not step_loss_file.exists():
        return [], []
    tokens_list, losses = [], []
    with open(step_loss_file, "r", encoding="utf-8") as f:
        for raw in f:
            try:
                rec = json.loads(raw)
                tokens_list.append(int(rec["tokens"]))
                losses.append(float(rec["loss"]))
            except Exception:
                continue
    return tokens_list, losses


def _mean_last_10pct(losses: list[float]) -> float | None:
    if not losses:
        return None
    tail_n = max(1, math.ceil(0.1 * len(losses)))
    tail = losses[-tail_n:]
    return sum(tail) / len(tail)


# ---------------------------------------------------------------------------
# Single-run executor — streams stdout live (critical for debugging)
# ---------------------------------------------------------------------------

def _run_one(
    run: SweepRun,
    run_root: Path,
    common_train_args: list[str],
    target_dim: int,
    num_experts: int,
    disable_base_mu_p: bool = False,
) -> dict:
    out_dir = run_root / run.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    step_loss_file = out_dir / "step_loss.jsonl"

    arch_flags = list(MODEL_ARCH_FLAGS[run.model])
    if run.model != "base":
        arch_flags += [
            "--moe-embed-dim",  str(target_dim),
            "--moe-router-dim", str(target_dim),
            "--moe-num-experts", str(num_experts),
        ]
        if run.model == "remixed-linear":
            arch_flags += [
                "--remix-context-dim", str(target_dim),
                "--remix-basis-size",  str(target_dim),
            ]
        # Always disable μP for research models so the swept LRs are exact.
        arch_flags.append("--disable-mu-p")
    elif disable_base_mu_p:
        arch_flags.append("--disable-mu-p")

    lr_flags = []
    for k in LR_GROUPS:
        lr_flags += [f"--{k.replace('_', '-')}", f"{run.lrs[k]:.8g}"]

    cmd = RUNNER + ["-m", "scripts.base_train"] + common_train_args + arch_flags + lr_flags + [
        "--model-tag",       run.run_name,
        "--checkpoints-dir", str(run_root / "checkpoints" / run.run_name),
        "--step-loss-file",  str(step_loss_file),
    ]

    print(f"  cmd: {' '.join(cmd)}", flush=True)

    log_path = out_dir / "stdout.log"
    tokens_live: list[int] = []
    losses_live: list[float] = []

    # Stream stdout live — same pattern as warmup_sweep.py
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),  # inherit full env (LD_LIBRARY_PATH etc.)
        )
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                print(line, end="", flush=True)
                log_f.write(line)
                # Parse live loss for immediate feedback
                m = re.search(r"step\s+(\d+)\s*/\s*\d+.*?\|\s*loss:\s*([0-9eE+\-\.]+)", line)
                if m:
                    step_n = int(m.group(1))
                    loss_v = float(m.group(2))
                    tokens_live.append(step_n * 524288)  # rough, overridden by file below
                    losses_live.append(loss_v)
        proc.communicate()

    # Prefer the precise step_loss.jsonl over the stdout-parsed values
    tokens_list, losses = _read_step_losses(step_loss_file)
    if not losses:
        tokens_list, losses = tokens_live, losses_live

    score = _mean_last_10pct(losses)
    return {
        "run_name":              run.run_name,
        "phase":                 run.phase,
        "model":                 run.model,
        "returncode":            proc.returncode,
        "mean_last_10pct_loss":  score,
        "tokens":                tokens_list,
        "losses":                losses,
        "step_loss_file":        str(step_loss_file),
        "stdout_log":            str(log_path),
        "lrs":                   run.lrs,
        "meta":                  run.meta,
    }


# ---------------------------------------------------------------------------
# Plotting — one plot per (phase, model), then an aggregate TSV
# ---------------------------------------------------------------------------

def _plot_phase_model(
    phase: int,
    model: str,
    run_results: list[dict],
    depth: int,
    run_dir: Path,
) -> None:
    """Loss curves for all runs belonging to (phase, model)."""
    plot_data = []
    for r in run_results:
        losses = r.get("losses")
        tokens = r.get("tokens")
        
        # If missing (due to resumption from slim results.json), try to reload from individual record
        if (not losses or not tokens) and r.get("step_loss_file"):
            t_list, l_list = _read_step_losses(Path(r["step_loss_file"]))
            if l_list:
                tokens, losses = t_list, l_list

        if losses and tokens:
            plot_data.append({
                "tokens": tokens,
                "losses": losses,
                "run_name": r["run_name"],
                "score": r.get("mean_last_10pct_loss")
            })

    if not plot_data:
        return

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", max(len(plot_data), 1))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, data in enumerate(plot_data):
        x = [t / 1e9 for t in data["tokens"]]
        y = data["losses"]
        score = data["score"]
        label = f"{data['run_name']}  loss={score:.4f}" if score is not None else data["run_name"]
        ax.plot(x, y, color=palette[i % len(palette)], label=label, linewidth=2)

    phase_labels = {1: "Phase 1 — Uniform Scale", 2: "Phase 2 — Coord Descent", 3: "Phase 3 — Refinement"}
    ax.set_title(f"{phase_labels.get(phase, f'Phase {phase}')} | {model} | Depth {depth}", fontsize=13)
    ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
    ax.set_ylabel("Train Loss (EMA) ↓", fontsize=12)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe = model.replace("-", "_")
    path = run_dir / f"phase{phase}_{safe}_depth{depth}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[actual_lr_sweep] Saved plot: {path}")


def _write_tsv(results: list[dict], run_dir: Path) -> None:
    tsv_path = run_dir / "sweep_results.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("phase\tmodel\trun_name\treturncode\tmean_last_10pct_loss\t"
                "embedding_lr\tunembedding_lr\tmatrix_lr\tscalar_lr\n")
        for r in results:
            lrs = r["lrs"]
            score = r["mean_last_10pct_loss"]
            f.write(
                f"{r['phase']}\t{r['model']}\t{r['run_name']}\t{r['returncode']}\t"
                f"{score if score is not None else 'NA'}\t"
                f"{lrs['embedding_lr']:.6g}\t{lrs['unembedding_lr']:.6g}\t"
                f"{lrs['matrix_lr']:.6g}\t{lrs['scalar_lr']:.6g}\n"
            )
    print(f"[actual_lr_sweep] Saved TSV: {tsv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phased coordinate-descent LR sweep for nanochat research models")
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+",
                   default=["moe_no_perm", "moe_perm", "remixed-linear"],
                   choices=list(MODEL_ARCH_FLAGS.keys()))
    p.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4], default=0,
                   help="0=all phases (default)")
    p.add_argument("--generate-only", action="store_true",
                   help="only emit JSON run configs — no training")
    p.add_argument("--disable-base-mu-p", action="store_true",
                   help="also disable μP scaling for the base model (research models always disable it)")

    # Phase 1: uniform absolute LR grid
    p.add_argument("--phase1-lrs", type=float, nargs="+",
                   default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                   help="absolute LR grid for phase 1 (all groups same value)")

    # Phase 2: coordinate descent multipliers around the phase-1 winner
    p.add_argument("--phase2-multipliers", type=float, nargs="+",
                   default=[0.3, 1.0, 3.0],
                   help="per-group multipliers for phase 2 coord descent")

    # lr_star: absolute LR to use as center for phase 2/3.
    # Can be a single value (used for all models+groups) or a JSON dict.
    p.add_argument("--lr-star", type=float, default=None,
                   help="absolute LR winner from phase 1 — center for phase 2/3 "
                        "(if not set, mid-point of --phase1-lrs is used)")
    p.add_argument("--lr-star-json", type=str, default=None,
                   help="path to JSON file with per-model lr_star, e.g. results from phase 1")

    # Phase 3
    p.add_argument("--phase3-samples", type=int, default=10)
    p.add_argument("--phase3-log-radius", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1337)

    # Phase 4: Bayesian Optimization via Optuna
    p.add_argument("--phase4-trials", type=int, default=50,
                   help="number of Optuna trials to run")
    p.add_argument("--phase4-lr-min", type=float, default=1e-4,
                   help="lower bound of the Phase 4 LR search space")
    p.add_argument("--phase4-lr-max", type=float, default=0.5,
                   help="upper bound of the Phase 4 LR search space")
    p.add_argument("--phase4-warm-start", action=argparse.BooleanOptionalAction, default=True,
                   help="seed Optuna with completed Phase 1-3 results before running new trials")

    # Training flags (mirror warmup_sweep.py)
    p.add_argument("--target-tokens", type=int, default=-1,
                   help="full LR-schedule token budget (-1 = auto from Chinchilla scaling, same as base_train)")
    p.add_argument("--early-stop-tokens", type=int, default=100_000_000)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--research-warmup-ratio", type=float, default=0.0)
    p.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1])
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--total-batch-size", type=int, default=524_288)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--moe-num-experts", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--core-metric-every", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=-1)
    
    # Remixed-linear components
    p.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1])
    p.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1])
    p.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1])

    # CCL block modulation
    p.add_argument("--cclblock-modulation", type=str, default="weight",
                   choices=["weight", "normalization"],
                   help="CCL block strategy passed to remixed-linear runs")
    p.add_argument("--cclblock-context-stream", type=str, default="local", 
                   choices=["local", "shifted", "ema", "selective", "multiscale", "boundary"],
                   help="Context stream type")
    p.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                   help="EMA factor for the legacy EMAContextStream")
    p.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                   help="Design C stale context lag (0=disabled)")
    # Novel ablation designs
    p.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                   help="Design 3: sparse top-k basis gate (0=off, N=top-N)")
    p.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                   help="Design 6: gate temperature (<1=sharper, >1=softer)")
    p.add_argument("--cclblock-context-bank-size", type=int, default=0,
                   help="Design 4: context prototype bank size (0=off, e.g. 16)")
    p.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                   help="Design 7: separate attn/ffn context projections (0=off, 1=on)")
    p.add_argument("--cclblock-context-source", type=str, default="norm_x",
                   choices=["norm_x", "attn_heads"])
    # Phase 8
    p.add_argument("--cclblock-chunk-size", type=int, default=0)
    p.add_argument("--cclblock-aux-objective", type=str, default="none", choices=["none", "boundary", "entropy"])
    p.add_argument("--cclblock-aux-lambda", type=float, default=0.1)
    p.add_argument("--cclblock-boundary-token-id", type=int, default=198)
    # Research dimension override
    p.add_argument("--research-dim", type=int, default=0, help="override default 1/8th model_dim for research branches")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-shards", type=int, default=-1)

    # Resumption and Indexing
    p.add_argument("--start-index", type=int, default=0, help="start execution from this config index")
    p.add_argument("--end-index", type=int, default=-1, help="stop execution after this config index (-1=end)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="if a run's results already exist on disk, skip training and load them")

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / "configs"

    # Build the lr_star dict — center for phase 2/3.
    # Priority: --lr-star-json > --lr-star > midpoint of phase1_lrs.
    if args.lr_star_json:
        with open(args.lr_star_json, "r", encoding="utf-8") as f:
            lr_star_raw = json.load(f)
        # Handle both raw {model: {group: float}} and best_config.json {model: {lrs: {group: float}}}
        lr_star = {}
        for m, val in lr_star_raw.items():
            if isinstance(val, dict) and "lrs" in val:
                lr_star[m] = val["lrs"]
            else:
                lr_star[m] = val
    elif args.lr_star is not None:
        lr_star = {m: {k: args.lr_star for k in LR_GROUPS} for m in args.models}
    else:
        mid = args.phase1_lrs[len(args.phase1_lrs) // 2]
        lr_star = {m: {k: mid for k in LR_GROUPS} for m in args.models}

    # Load or build full run list
    if config_dir.exists() and any(config_dir.glob("run_*.json")):
        print(f"Loading existing configs from {config_dir}...")
        runs = []
        config_files = sorted(config_dir.glob("run_*.json"))
        for cf in config_files:
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs.append(SweepRun(
                    phase=data["phase"],
                    model=data["model"],
                    run_name=data["run_name"],
                    lrs=data["lrs"],
                    meta=data["meta"]
                ))
    else:
        runs = make_phase_runs(
            models=args.models,
            phase1_lrs=args.phase1_lrs,
            phase2_multipliers=args.phase2_multipliers,
            phase3_samples=args.phase3_samples,
            phase3_log_radius=args.phase3_log_radius,
            lr_star=lr_star,
            seed=args.seed,
        )
        dump_run_configs(runs, config_dir)
        print(f"Wrote {len(runs)} configs to {config_dir}")

    if args.phase in (1, 2, 3):
        # We index the full list before filtering so indices stay stable
        runs = [r for r in runs if r.phase == args.phase]
    elif args.phase == 4:
        # Phase 4 is handled separately via Optuna — nothing to do in the static config loop
        runs = []

    print(f"Total configs matching phase filter: {len(runs)}")
    print(f"μP LR scaling: DISABLED for all research models (--disable-mu-p always passed)")
    if hasattr(args, 'disable_base_mu_p') and args.disable_base_mu_p:
        print(f"μP LR scaling: also DISABLED for base model (--disable-base-mu-p set)")

    if args.generate_only:
        return

    # Architecture sizing
    aspect_ratio, head_dim, _, target_dim = model_dims(args.depth)

    # Resolve target tokens — auto-compute Chinchilla optimum when not supplied
    target_tokens = args.target_tokens
    if target_tokens <= 0:
        target_tokens = estimate_tokens_from_base(args.depth, tokenizer_dir=args.tokenizer_dir)
        print(f"[actual_lr_sweep] Auto-computed target tokens (Chinchilla): {target_tokens:,}")

    # Args shared by every training run
    common_train_args = [
        "--depth",               str(args.depth),
        "--aspect-ratio",        str(aspect_ratio),
        "--head-dim",            str(head_dim),
        "--max-seq-len",         str(args.max_seq_len),
        "--device-batch-size",   str(args.device_batch_size),
        "--total-batch-size",    str(args.total_batch_size),
        "--target-tokens",       str(target_tokens),
        "--early-stop-tokens",   str(args.early_stop_tokens),
        "--warmup-ratio",        str(args.warmup_ratio),
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle",        str(args.use_onecycle),
        "--log-every",           str(args.log_every),
        "--eval-every",          str(args.eval_every),
        "--core-metric-every",   str(args.core_metric_every),
        "--sample-every",        str(args.sample_every),
        "--router-context-window", str(getattr(args, 'router_context_window', -1)),
        "--remix-use-basis-gate",  str(getattr(args, 'remix_use_basis_gate', 1)),
        "--remix-use-output-gate", str(getattr(args, 'remix_use_output_gate', 1)),
        "--remix-use-context",     str(getattr(args, 'remix_use_context', 1)),
        "--moe-use-abs-pos-embed", "0",
        "--cclblock-modulation", getattr(args, 'cclblock_modulation', 'weight'),
        "--cclblock-context-stream", getattr(args, 'cclblock_context_stream', 'local'),
        "--cclblock-ema-factor", str(getattr(args, 'cclblock_ema_factor', 0.99)),
        "--cclblock-stale-ctx-lag", str(getattr(args, 'cclblock_stale_ctx_lag', 0)),
        # Novel ablation designs
        "--cclblock-sparse-gate-k",    str(getattr(args, 'cclblock_sparse_gate_k', 0)),
        "--cclblock-gate-temperature", str(getattr(args, 'cclblock_gate_temperature', 1.0)),
        "--cclblock-context-bank-size",str(getattr(args, 'cclblock_context_bank_size', 0)),
        "--cclblock-per-head-ctx",     str(getattr(args, 'cclblock_per_head_ctx', 0)),
        "--cclblock-context-source",   str(getattr(args, 'cclblock_context_source', 'norm_x')),
        # Phase 8
        "--cclblock-chunk-size",        str(getattr(args, 'cclblock_chunk_size', 0)),
        "--cclblock-aux-objective",     str(getattr(args, 'cclblock_aux_objective', 'none')),
        "--cclblock-aux-lambda",        str(getattr(args, 'cclblock_aux_lambda', 0.1)),
        "--cclblock-boundary-token-id", str(getattr(args, 'cclblock_boundary_token_id', 198)),
        "--research-dim", str(getattr(args, 'research_dim', 0)),
    ]
    if args.compile:
        common_train_args.append("--compile")
    else:
        common_train_args.append("--no-compile")
    if args.fp8:
        common_train_args.append("--fp8")
    if args.tokenizer_dir:
        common_train_args += ["--tokenizer-dir", args.tokenizer_dir]
    if args.data_dir:
        common_train_args += ["--data-dir", args.data_dir]
    if args.max_shards > 0:
        common_train_args += ["--max-shards", str(args.max_shards)]

    results: list[dict] = []
    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {results_path}")
        except Exception as e:
            print(f"Warning: Could not load existing results.json: {e}")

    # Set of run names already in results to avoid duplicates
    finished_names = {r["run_name"] for r in results}

    # Handle end-index
    end_idx = args.end_index if args.end_index != -1 else len(runs) - 1
    
    for i, run in enumerate(runs):
        if i < args.start_index or i > end_idx:
            continue
            
        if run.run_name in finished_names:
            print(f"\n[{i+1}/{len(runs)}] run={run.run_name} already in results.json, skipping.")
            continue

        print(f"\n[{i+1}/{len(runs)}] phase={run.phase} model={run.model} run={run.run_name}", flush=True)
        
        # Check for resume
        out_dir = run_dir / run.run_name
        step_loss_file = out_dir / "step_loss.jsonl"
        log_path = out_dir / "stdout.log"
        
        if args.resume and step_loss_file.exists() and step_loss_file.stat().st_size > 0:
            print(f"  ✓ found existing results, skipping training.")
            tokens_list, losses = _read_step_losses(step_loss_file)
            score = _mean_last_10pct(losses)
            res = {
                "run_name":              run.run_name,
                "phase":                 run.phase,
                "model":                 run.model,
                "returncode":            0,
                "mean_last_10pct_loss":  score,
                "tokens":                tokens_list,
                "losses":                losses,
                "step_loss_file":        str(step_loss_file),
                "stdout_log":            str(log_path),
                "lrs":                   run.lrs,
                "meta":                  run.meta,
            }
        else:
            res = _run_one(
                run, run_dir, common_train_args, target_dim,
                args.moe_num_experts,
                disable_base_mu_p=getattr(args, 'disable_base_mu_p', False),
            )
            if res["returncode"] != 0:
                print(f"  ✗ failed (exit code {res['returncode']})  — see {res['stdout_log']}")
            else:
                print(f"  ✓ mean_last_10pct_loss = {res['mean_last_10pct_loss']}")

        results.append(res)

        # Incremental Save - JSON
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            slim = [{k: v for k, v in r.items() if k not in ("tokens", "losses")}
                    for r in results]
            json.dump(slim, f, indent=2)

        # Incremental Save - TSV
        _write_tsv(results, run_dir)

        # Update per-(phase, model) plot
        phase_model_results = [r for r in results
                                if r["phase"] == run.phase and r["model"] == run.model]
        _plot_phase_model(run.phase, run.model, phase_model_results, args.depth, run_dir)
        
        # Update per-model bests
        _update_best_configs(results, run_dir, args.models)

    print(f"\n[actual_lr_sweep] Finished processing range {args.start_index} to {end_idx}.")

    # Phase 4: Bayesian Optimisation
    if args.phase in (0, 4):
        for model in args.models:
            _run_phase4_optuna(
                model=model,
                run_dir=run_dir,
                common_train_args=common_train_args,
                target_dim=target_dim,
                num_experts=args.moe_num_experts,
                disable_base_mu_p=getattr(args, "disable_base_mu_p", False),
                n_trials=args.phase4_trials,
                lr_min=args.phase4_lr_min,
                lr_max=args.phase4_lr_max,
                warm_start=args.phase4_warm_start,
                depth=args.depth,
                results=results,
            )
            _update_best_configs(results, run_dir, args.models)


def _run_phase4_optuna(
    model: str,
    run_dir: Path,
    common_train_args: list[str],
    target_dim: int,
    num_experts: int,
    disable_base_mu_p: bool,
    n_trials: int,
    lr_min: float,
    lr_max: float,
    warm_start: bool,
    depth: int,
    results: list[dict],
) -> None:
    """Run Bayesian Optimisation (Optuna/TPE) for one model as Phase 4."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[phase4] optuna is not installed. Run: pip install optuna")
        return

    storage_path = run_dir / f"optuna_{model.replace('-', '_')}.db"
    study_name = f"lr_sweep_phase4_{model}"

    print(f"\n[phase4] Starting Bayesian Optimisation for {model} ({n_trials} trials)")
    print(f"[phase4] Optuna storage: {storage_path}")

    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,  # resume if interrupted
        sampler=optuna.samplers.CmaEsSampler(
            seed=42,
            n_startup_trials=10,  # random init before CMA-ES kicks in (already past this with warm-start)
        ),
    )

    # Warm-start: seed with Phase 1/2/3 results already on disk
    if warm_start:
        prior = [
            r for r in results
            if r["model"] == model
            and r["returncode"] == 0
            and r["mean_last_10pct_loss"] is not None
        ]
        already_seeded = len(study.trials)
        if prior and already_seeded == 0:
            print(f"[phase4] Warm-starting with {len(prior)} prior results...")
            for r in prior:
                lrs = r["lrs"]
                # Clamp to search bounds — prior results may be outside phase4 range
                params = {g: max(lr_min, min(lr_max, lrs[g])) for g in LR_GROUPS}
                dist = {g: optuna.distributions.FloatDistribution(lr_min, lr_max, log=True)
                        for g in LR_GROUPS}
                trial = optuna.trial.create_trial(
                    params=params,
                    distributions=dist,
                    value=r["mean_last_10pct_loss"],
                )
                study.add_trial(trial)
            print(f"[phase4] Seeded {len(prior)} prior trials.")
        elif already_seeded > 0:
            print(f"[phase4] Resuming study with {already_seeded} existing trials, skipping warm-start.")

    trial_counter = [len(study.trials)]

    def objective(trial: "optuna.Trial") -> float:
        lrs = {
            g: trial.suggest_float(g, lr_min, lr_max, log=True)
            for g in LR_GROUPS
        }
        trial_idx = trial_counter[0]
        trial_counter[0] += 1
        run_name = f"{model}_p4_trial{trial_idx:03d}"

        sweep_run = SweepRun(
            phase=4,
            model=model,
            run_name=run_name,
            lrs=lrs,
            meta={"trial": trial_idx, "optuna_trial": trial.number},
        )
        res = _run_one(
            sweep_run, run_dir, common_train_args, target_dim,
            num_experts, disable_base_mu_p=disable_base_mu_p,
        )
        score = res["mean_last_10pct_loss"]
        results.append(res)

        # Incremental save after every trial
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            slim = [{k: v for k, v in r.items() if k not in ("tokens", "losses")}
                    for r in results]
            json.dump(slim, f, indent=2)
        _write_tsv(results, run_dir)

        # Update plot for phase 4 after each trial
        p4_results = [r for r in results if r["phase"] == 4 and r["model"] == model]
        _plot_phase_model(4, model, p4_results, depth, run_dir)

        if res["returncode"] != 0:
            print(f"  [phase4] trial {trial_idx} failed — pruning.")
            raise optuna.exceptions.TrialPruned()

        print(f"  [phase4] trial {trial_idx} | loss={score:.6f} | lrs={lrs}")
        return score if score is not None else float("inf")

    study.optimize(objective, n_trials=n_trials)

    print(f"\n[phase4] Best trial for {model}:")
    best = study.best_trial
    print(f"  loss: {best.value:.6f}")
    for g in LR_GROUPS:
        print(f"  {g}: {best.params[g]:.8g}")


def _update_best_configs(results: list[dict], run_dir: Path, models_to_watch: list[str]) -> None:
    """Updates best_config.json and best_run_config.json incrementally."""
    successful = [r for r in results
                  if r["returncode"] == 0 and r["mean_last_10pct_loss"] is not None]
    
    if not successful:
        return
        
    models_seen = sorted({r["model"] for r in successful})
    
    # 1. Classical "Single Best Run" winner
    per_model_best_run: dict = {}
    for model in models_seen:
        model_runs = [r for r in successful if r["model"] == model]
        if not model_runs: continue
        best = min(model_runs, key=lambda r: r["mean_last_10pct_loss"])
        per_model_best_run[model] = {k: v for k, v in best.items() if k not in ("tokens", "losses")}
    
    with open(run_dir / "best_run_config.json", "w", encoding="utf-8") as f:
        json.dump(per_model_best_run, f, indent=2)

    # 2. Synthesized "Best Per Group" winner
    per_group_best = _compute_per_group_bests(results, models_seen)
    with open(run_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(per_group_best, f, indent=2)


def _compute_per_group_bests(results: list[dict], models: list[str]) -> dict:
    """
    Analyzes Phase 2 results to find the best LR for each group independently.
    Returns a dict: { model: { "lrs": {...}, "group_details": {...} } }
    """
    summary = {}
    for model in models:
        model_results = [r for r in results if r["model"] == model and r["returncode"] == 0]
        if not model_results:
            continue

        # 1. Identify the Phase 1 winner (the center for Phase 2)
        p1_runs = [r for r in model_results if r["phase"] == 1]
        if not p1_runs:
            # If no P1 (maybe user ran phase 2 only), just look for any baseline
            baselines = [r for r in model_results if r["phase"] == 2 and r["meta"].get("mult") == 1.0]
            if not baselines:
                continue
            p1_winner = min(baselines, key=lambda r: r["mean_last_10pct_loss"])
        else:
            p1_winner = min(p1_runs, key=lambda r: r["mean_last_10pct_loss"])
        
        # 2. For each group, find the best value from Phase 2
        best_lrs = dict(p1_winner["lrs"])
        details = {}

        for group in ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"]:
            # Gather all runs for this group in Phase 2
            group_runs = [r for r in model_results if r["phase"] == 2 and r.get("meta", {}).get("group") == group]
            # Include the baseline (P1 winner)
            relevant_runs = group_runs + [p1_winner]
            
            best_run = min(relevant_runs, key=lambda r: r["mean_last_10pct_loss"])
            best_lrs[group] = best_run["lrs"][group]
            
            details[group] = {
                "value": best_run["lrs"][group],
                "loss": best_run["mean_last_10pct_loss"],
                "run": best_run["run_name"]
            }

        summary[model] = {
            "lrs": best_lrs,
            "p1_center_loss": p1_winner["mean_last_10pct_loss"],
            "group_details": details
        }
    return summary


if __name__ == "__main__":
    main()
