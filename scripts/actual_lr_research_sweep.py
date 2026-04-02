"""
Phased coordinate-descent LR sweep for nanochat research model variants.

Follows the same patterns as warmup_sweep.py:
  - Streams base_train stdout live (so you always see the real error)
  - Passes full os.environ to subprocesses (torchrun inherits LD_LIBRARY_PATH etc.)
  - Produces per-phase loss-curve PNGs and a TSV summary

Three phases:
  Phase 1 — Uniform scale sweep.
    Train each model with all 4 LR groups scaled together by a shared factor s.
    Identifies the best overall scale s*.

  Phase 2 — Coordinate descent around s*.
    Hold all groups at base_lr * s*, perturb one group at a time.
    Cheaply answers "does group X benefit from being higher/lower than uniform?"

  Phase 3 — Optional joint refinement.
    Random log-space search around the phase-2 winner.
    Only needed if phase 2 shows groups prefer non-unity multipliers.

Usage:
    python -m scripts.actual_lr_research_sweep \\
        --depth 8 --run-dir out/actual_lr_sweep \\
        --early-stop-tokens 100000000 --target-tokens 20000000000 \\
        --models moe_perm moe_no_perm remixed-linear --fp8
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


# ---------------------------------------------------------------------------
# Base LR profiles — the "unit" configuration for each model type
# ---------------------------------------------------------------------------

LR_GROUPS = ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"]

BASE_LRS: dict[str, dict[str, float]] = {
    "base": {
        "embedding_lr":   0.3,
        "unembedding_lr": 0.004,
        "matrix_lr":      0.02,
        "scalar_lr":      0.5,
    },
    "moe_perm": {
        "embedding_lr":   0.05,
        "unembedding_lr": 0.05,
        "matrix_lr":      0.05,
        "scalar_lr":      0.05,
    },
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

MODEL_ARCH_FLAGS: dict[str, list[str]] = {
    "base":          [],
    "moe_no_perm":   ["--use-moe"],
    "moe_perm":      ["--use-moe", "--use-perm"],
    "remixed-linear": ["--use-remixed-linear"],
}


# ---------------------------------------------------------------------------
# DDP runner — mirrors warmup_sweep.py exactly
# ---------------------------------------------------------------------------

def _resolve_runner() -> list[str]:
    """Return torchrun command prefix, capped to available GPUs."""
    nproc_requested = int(os.environ.get("NPROC_PER_NODE", os.environ.get("NANOCHAT_NPROC", 8)))
    gpu_count = max(torch.cuda.device_count(), 1)
    nproc = min(nproc_requested, gpu_count)
    if nproc < nproc_requested:
        print(
            f"[actual_lr_sweep] Requested {nproc_requested} DDP workers but only "
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
# Model sizing — same helper as in warmup_sweep.py / lr_sweep.py
# ---------------------------------------------------------------------------

def _model_dims(depth: int) -> tuple[int, int, int, int]:
    aspect_ratio = 64
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    raw_target_dim = max(model_dim // 8, 1)
    target_dim = ((raw_target_dim + head_dim - 1) // head_dim) * head_dim
    target_dim = min(target_dim, model_dim)
    return aspect_ratio, head_dim, model_dim, target_dim


# ---------------------------------------------------------------------------
# Phase config generation
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
    phase1_scales: list[float],
    phase2_multipliers: list[float],
    phase3_samples: int,
    phase3_log_radius: float,
    s_star: float,
    seed: int,
) -> list[SweepRun]:
    rng = random.Random(seed)
    runs: list[SweepRun] = []

    # Phase 1: uniform scale sweep — all LR groups scaled together.
    for model in models:
        base = BASE_LRS[model]
        for s in phase1_scales:
            lrs = {k: v * s for k, v in base.items()}
            runs.append(SweepRun(
                phase=1,
                model=model,
                run_name=f"{model}_p1_s{s:g}",
                lrs=lrs,
                meta={"scale": s},
            ))

    # Phase 2: coordinate descent — perturb one group at a time around s*.
    for model in models:
        base = BASE_LRS[model]
        for group in LR_GROUPS:
            for mult in phase2_multipliers:
                lrs = {g: base[g] * s_star for g in LR_GROUPS}
                lrs[group] = base[group] * s_star * mult
                runs.append(SweepRun(
                    phase=2,
                    model=model,
                    run_name=f"{model}_p2_{group}_x{mult:g}",
                    lrs=lrs,
                    meta={"group": group, "mult": mult, "s_star": s_star},
                ))

    # Phase 3: random log-space refinement around the phase-2 prior center.
    for model in models:
        base = BASE_LRS[model]
        center = {g: base[g] * s_star for g in LR_GROUPS}
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
) -> dict:
    out_dir = run_root / run.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    step_loss_file = out_dir / "step_loss.jsonl"

    arch_flags = list(MODEL_ARCH_FLAGS[run.model])
    if run.model != "base":
        arch_flags += [
            "--target-dim",    str(target_dim),
            "--router-dim",    str(target_dim),
            "--num-experts",   str(num_experts),
        ]
        if run.model == "remixed-linear":
            arch_flags += [
                "--context-dim",      str(target_dim),
                "--linear-basis-size", str(target_dim),
            ]

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
    successful = [r for r in run_results if r["losses"]]
    if not successful:
        return

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", max(len(successful), 1))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, res in enumerate(successful):
        x = [t / 1e9 for t in res["tokens"]]
        y = res["losses"]
        score = res["mean_last_10pct_loss"]
        label = f"{res['run_name']}  loss={score:.4f}" if score is not None else res["run_name"]
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
                   default=["base", "moe_no_perm", "moe_perm", "remixed-linear"],
                   choices=list(BASE_LRS.keys()))
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 0], default=0,
                   help="0=all phases (default)")
    p.add_argument("--generate-only", action="store_true",
                   help="only emit JSON run configs — no training")

    # Phase tuning
    p.add_argument("--phase1-scales", type=float, nargs="+",
                   default=[0.03, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--s-star", type=float, default=0.1,
                   help="phase-2/3 center scale (best scale from phase 1)")
    p.add_argument("--phase2-multipliers", type=float, nargs="+",
                   default=[0.3, 1.0, 3.0])
    p.add_argument("--phase3-samples", type=int, default=10)
    p.add_argument("--phase3-log-radius", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1337)

    # Training flags (mirror warmup_sweep.py)
    p.add_argument("--target-tokens", type=int, default=20_000_000_000)
    p.add_argument("--early-stop-tokens", type=int, default=100_000_000)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--research-warmup-ratio", type=float, default=0.0)
    p.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1])
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--total-batch-size", type=int, default=524_288)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--core-metric-every", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=-1)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-shards", type=int, default=-1)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / "configs"

    # Build full run list, then filter by phase if requested
    runs = make_phase_runs(
        models=args.models,
        phase1_scales=args.phase1_scales,
        phase2_multipliers=args.phase2_multipliers,
        phase3_samples=args.phase3_samples,
        phase3_log_radius=args.phase3_log_radius,
        s_star=args.s_star,
        seed=args.seed,
    )
    if args.phase in (1, 2, 3):
        runs = [r for r in runs if r.phase == args.phase]

    dump_run_configs(runs, config_dir)
    print(f"Wrote {len(runs)} configs to {config_dir}")

    if args.generate_only:
        return

    # Architecture sizing
    aspect_ratio, head_dim, _, target_dim = _model_dims(args.depth)

    # Args shared by every training run
    common_train_args = [
        "--depth",               str(args.depth),
        "--aspect-ratio",        str(aspect_ratio),
        "--head-dim",            str(head_dim),
        "--max-seq-len",         str(args.max_seq_len),
        "--device-batch-size",   str(args.device_batch_size),
        "--total-batch-size",    str(args.total_batch_size),
        "--target-tokens",       str(args.target_tokens),
        "--early-stop-tokens",   str(args.early_stop_tokens),
        "--warmup-ratio",        str(args.warmup_ratio),
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle",        str(args.use_onecycle),
        "--log-every",           str(args.log_every),
        "--eval-every",          str(args.eval_every),
        "--core-metric-every",   str(args.core_metric_every),
        "--sample-every",        str(args.sample_every),
        "--moe-use-abs-pos-embed", "0",
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

    for i, run in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] phase={run.phase} model={run.model} run={run.run_name}", flush=True)
        res = _run_one(run, run_dir, common_train_args, target_dim, args.num_experts)
        results.append(res)
        if res["returncode"] != 0:
            print(f"  ✗ failed (exit code {res['returncode']})  — see {res['stdout_log']}")
        else:
            print(f"  ✓ mean_last_10pct_loss = {res['mean_last_10pct_loss']}")

        # Per-(phase, model) plot — regenerate after each run so progress is visible
        phase_model_results = [r for r in results
                                if r["phase"] == run.phase and r["model"] == run.model]
        _plot_phase_model(run.phase, run.model, phase_model_results, args.depth, run_dir)

    # Write results JSON
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        # Don't serialise potentially huge tokens/losses lists into results.json
        slim = [{k: v for k, v in r.items() if k not in ("tokens", "losses")}
                for r in results]
        json.dump(slim, f, indent=2)

    _write_tsv(results, run_dir)

    # Overall best
    successful = [r for r in results
                  if r["returncode"] == 0 and r["mean_last_10pct_loss"] is not None]
    if successful:
        best = min(successful, key=lambda r: r["mean_last_10pct_loss"])
        best_slim = {k: v for k, v in best.items() if k not in ("tokens", "losses")}
        with open(run_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(best_slim, f, indent=2)
        print(f"\n[actual_lr_sweep] Best: {best['run_name']}  loss={best['mean_last_10pct_loss']:.6f}")
    else:
        print("\n[actual_lr_sweep] No successful runs.")


if __name__ == "__main__":
    main()
