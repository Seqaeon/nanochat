#!/usr/bin/env python3
"""One-time setup for autoresearch-on-nanochat.

This wrapper keeps the original nanochat code untouched and prepares the
minimal artifacts needed by `train.py`:
- trains tokenizer (`scripts.tok_train`)
- optionally evaluates tokenizer (`scripts.tok_eval`)
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_module(module: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nanochat artifacts for autoresearch")
    parser.add_argument("--skip-tok-eval", action="store_true", help="skip tokenizer evaluation")
    parser.add_argument("--tok-train-arg", action="append", default=[], help="extra arg passed through to scripts.tok_train")
    parser.add_argument("--tok-eval-arg", action="append", default=[], help="extra arg passed through to scripts.tok_eval")
    args = parser.parse_args()

    run_module("scripts.tok_train", args.tok_train_arg)
    if not args.skip_tok_eval:
        run_module("scripts.tok_eval", args.tok_eval_arg)

    print("prepare.py finished successfully")


if __name__ == "__main__":
    main()
