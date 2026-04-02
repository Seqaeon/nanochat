#!/usr/bin/env python3
"""Actual-LR research sweep with phased coordinate-descent refinement.

Designed for nanochat's existing training entrypoints. It can:
1) Generate phase configs (phase1/phase2/phase3) as JSON files.
2) Optionally execute selected phases locally (sequentially) using scripts.base_train.
3) Score runs via mean training loss over the last 10% of recorded steps.

This keeps the workflow repo-native (no Slurm requirement), while still supporting
phase-by-phase progression and reproducible config artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


LR_GROUPS = ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"]

BASE_LRS: dict[str, dict[str, float]] = {
    "base": {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
    },
    "moe_perm": {
        "embedding_lr": 0.05,
        "unembedding_lr": 0.05,
        "matrix_lr": 0.05,
        "scalar_lr": 0.05,
    },
    "moe_no_perm": {
        "embedding_lr": 0.05,
        "unembedding_lr": 0.05,
        "matrix_lr": 0.05,
        "scalar_lr": 0.05,
    },
    "remixed-linear": {
        "embedding_lr": 0.05,
        "unembedding_lr": 0.05,
        "matrix_lr": 0.05,
        "scalar_lr": 0.05,
    },
}

MODEL_ARCH_FLAGS: dict[str, list[str]] = {
    "base": [],
    "moe_no_perm": ["--use-moe"],
    "moe_perm": ["--use-moe", "--use-perm"],
    "remixed-linear": ["--use-remixed-linear"],
}


@dataclass
class SweepRun:
    phase: int
    model: str
    run_name: str
    lrs: dict[str, float]
    meta: dict


def _resolve_runner() -> list[str]:
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun, "--standalone", "--nproc_per_node=1"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]


def _model_dims(depth: int) -> tuple[int, int, int, int]:
    aspect_ratio = 64
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    raw_target_dim = max(model_dim // 8, 1)
    target_dim = ((raw_target_dim + head_dim - 1) // head_dim) * head_dim
    target_dim = min(target_dim, model_dim)
    return aspect_ratio, head_dim, model_dim, target_dim


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

    # Phase 1: uniform scale sweep around each model's base LR profile.
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

    # Phase 2: coordinate descent around s*.
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

    # Phase 3: random log-space refinement around phase2 prior center.
    for model in models:
        base = BASE_LRS[model]
        center = {g: base[g] * s_star for g in LR_GROUPS}
        for i in range(phase3_samples):
            lrs = {}
            for g, v in center.items():
                lrs[g] = v * (10 ** rng.uniform(-phase3_log_radius, phase3_log_radius))
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
        obj = {
            "run_index": idx,
            **asdict(run),
        }
        with open(config_dir / f"run_{idx:03d}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


def _read_step_loss_mean_last10pct(step_loss_file: Path) -> float | None:
    if not step_loss_file.exists():
        return None
    losses: list[float] = []
    with open(step_loss_file, "r", encoding="utf-8") as f:
        for raw in f:
            rec = json.loads(raw)
            losses.append(float(rec["loss"]))
    if not losses:
        return None
    tail_n = max(1, math.ceil(0.1 * len(losses)))
    tail = losses[-tail_n:]
    return sum(tail) / len(tail)


def _run_one(
    run: SweepRun,
    args: argparse.Namespace,
    runner: list[str],
    run_root: Path,
    common_train_args: list[str],
    target_dim: int,
) -> dict:
    out_dir = run_root / run.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    step_loss_file = out_dir / "step_loss.jsonl"

    arch_flags = list(MODEL_ARCH_FLAGS[run.model])
    if run.model != "base":
        arch_flags += [
            "--target-dim", str(target_dim),
            "--router-dim", str(target_dim),
            "--num-experts", str(args.num_experts),
        ]
        if run.model == "remixed-linear":
            arch_flags += [
                "--context-dim", str(target_dim),
                "--linear-basis-size", str(target_dim),
            ]

    lr_flags = []
    for k in LR_GROUPS:
        lr_flags += [f"--{k.replace('_', '-')}", f"{run.lrs[k]:.8g}"]

    cmd = runner + ["-m", "scripts.base_train"] + common_train_args + arch_flags + lr_flags + [
        "--run", f"actual-lr-{run.run_name}",
        "--model-tag", run.run_name,
        "--checkpoints-dir", str(run_root / "checkpoints"),
        "--step-loss-file", str(step_loss_file),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    score = _read_step_loss_mean_last10pct(step_loss_file)
    return {
        "run_name": run.run_name,
        "phase": run.phase,
        "model": run.model,
        "returncode": proc.returncode,
        "mean_last_10pct_loss": score,
        "step_loss_file": str(step_loss_file),
        "stdout_log": str(out_dir / "stdout.log"),
        "lrs": run.lrs,
        "meta": run.meta,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Actual-LR phased research sweep for nanochat")
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", default=["base", "moe_no_perm", "moe_perm", "remixed-linear"], choices=list(BASE_LRS.keys()))
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 0], default=0, help="0=all phases")
    p.add_argument("--generate-only", action="store_true", help="only emit JSON run configs")

    p.add_argument("--phase1-scales", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--s-star", type=float, default=0.1, help="phase2/3 center scale factor")
    p.add_argument("--phase2-multipliers", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    p.add_argument("--phase3-samples", type=int, default=10)
    p.add_argument("--phase3-log-radius", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1337)

    # Training flags (mirrors other sweep scripts)
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

    aspect_ratio, head_dim, _, target_dim = _model_dims(args.depth)
    runner = _resolve_runner()

    common_train_args = [
        "--depth", str(args.depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--max-seq-len", str(args.max_seq_len),
        "--device-batch-size", str(args.device_batch_size),
        "--total-batch-size", str(args.total_batch_size),
        "--target-tokens", str(args.target_tokens),
        "--early-stop-tokens", str(args.early_stop_tokens),
        "--warmup-ratio", str(args.warmup_ratio),
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle", str(args.use_onecycle),
        "--log-every", str(args.log_every),
        "--eval-every", str(args.eval_every),
        "--core-metric-every", str(args.core_metric_every),
        "--sample-every", str(args.sample_every),
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
        print(f"[{i+1}/{len(runs)}] phase={run.phase} model={run.model} run={run.run_name}")
        res = _run_one(run, args, runner, run_dir, common_train_args, target_dim)
        results.append(res)
        if res["returncode"] != 0:
            print(f"  failed (code={res['returncode']})")
        else:
            print(f"  mean_last_10pct_loss={res['mean_last_10pct_loss']}")

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    successful = [r for r in results if r["returncode"] == 0 and r["mean_last_10pct_loss"] is not None]
    if successful:
        best = min(successful, key=lambda r: r["mean_last_10pct_loss"])
        with open(run_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print("Best:", best["run_name"], "loss=", best["mean_last_10pct_loss"])


if __name__ == "__main__":
    main()
