#!/usr/bin/env python3
"""5-minute research-run wrapper around `scripts.base_train`.

`train.py` is intentionally thin: it launches nanochat pretraining with sane
short-run defaults and a wall-clock budget, while forwarding research flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_base_train_cmd(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.base_train"]

    # Short-run defaults suitable for autoresearch iterations.
    defaults = {
        "--run": args.run,
        "--model-tag": args.model_tag,
        "--core-metric-every": str(args.core_metric_every),
        "--eval-every": str(args.eval_every),
        "--save-every": str(args.save_every),
    }
    for k, v in defaults.items():
        if v is not None and v != "":
            cmd += [k, v]

    if args.num_iterations > 0:
        cmd += ["--num-iterations", str(args.num_iterations)]

    # Research branch toggles (these map to your modified base_train/gpt paths).
    if args.use_moe:
        cmd.append("--use-moe")
    if args.use_perm:
        cmd.append("--use-perm")
    if args.use_remixed_linear:
        cmd.append("--use-remixed-linear")
    if args.allow_replacement:
        cmd.append("--allow-replacement")

    cmd += ["--num-experts", str(args.num_experts)]
    cmd += ["--router-dim", str(args.router_dim)]
    cmd += ["--target-dim", str(args.target_dim)]
    cmd += ["--selection-mode", args.selection_mode]
    cmd += ["--context-dim", str(args.context_dim)]
    cmd += ["--linear-basis-size", str(args.linear_basis_size)]
    cmd += ["--moe-use-abs-pos-embed", str(int(bool(args.moe_use_abs_pos_embed)))]

    # OneCycle controls still come from base_train schedule knobs.
    cmd += ["--warmup-ratio", str(args.warmup_ratio)]
    cmd += ["--warmdown-ratio", str(args.warmdown_ratio)]
    cmd += ["--final-lr-frac", str(args.final_lr_frac)]
    cmd += ["--use-onecycle", str(int(args.use_onecycle))]
    cmd += ["--research-warmup-ratio", str(args.research_warmup_ratio)]

    # Forward any additional base_train flags untouched.
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch training wrapper for nanochat")
    parser.add_argument("--time-budget-s", type=int, default=300, help="wall-clock budget in seconds")
    parser.add_argument("--num-iterations", type=int, default=-1, help="if >0, pass explicit iteration count")

    # Logging/checkpoint defaults
    parser.add_argument("--run", type=str, default="autoresearch")
    parser.add_argument("--model-tag", type=str, default="autoresearch")
    parser.add_argument("--core-metric-every", type=int, default=-1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=-1)

    # Research branch args
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--use-perm", action="store_true")
    parser.add_argument("--use-remixed-linear", action="store_true")
    parser.add_argument("--allow-replacement", action="store_true")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--router-dim", type=int, default=64)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--selection-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--linear-basis-size", type=int, default=64)
    parser.add_argument("--moe-use-abs-pos-embed", type=int, default=0, choices=[0, 1])

    # Scheduler knobs (research branches use OneCycle in your modified base_train)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmdown-ratio", type=float, default=0.9)
    parser.add_argument("--final-lr-frac", type=float, default=0.1)
    parser.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1])
    parser.add_argument("--research-warmup-ratio", type=float, default=0.0)

    args, passthrough = parser.parse_known_args()

    cmd = build_base_train_cmd(args, passthrough)
    print("+", " ".join(cmd), flush=True)

    try:
        subprocess.run(cmd, check=True, timeout=args.time_budget_s)
        print("train.py finished before timeout")
    except subprocess.TimeoutExpired:
        print(f"train.py reached time budget ({args.time_budget_s}s) and was stopped")
    except subprocess.CalledProcessError as e:
        print(f"train.py failed with exit code {e.returncode}")
        raise


if __name__ == "__main__":
    main()
