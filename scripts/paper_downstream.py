"""Group B1: zero-shot downstream evaluation for Table 4.

Evaluates one or more checkpoints on the CORE benchmark bundle via the existing
harness in scripts/base_eval.py, and prints a table with MST alongside the two
dense reference points that bracket it (iso-parameter and iso-FLOP).

    python -m scripts.paper_downstream \\
        --ckpt mst_d32=out/p07/S7_COMBO_A_D32 \\
        --ckpt dense_isoparam=out/dense/d20 \\
        --ckpt dense_isoflop=out/dense/d22 \\
        --out scratch/paper_downstream.json

Pass --max-per-task to cap examples while you are debugging the plumbing; leave
it at -1 for the numbers that go in the paper.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.paper_lib import load_any_model
from scripts.base_eval import evaluate_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    metavar="LABEL=DIR", help="repeatable; e.g. mst_d32=out/.../D32")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--max-per-task", type=int, default=-1)
    ap.add_argument("--out", default="scratch/paper_downstream.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for spec in args.ckpt:
        if "=" not in spec:
            print(f"--ckpt wants LABEL=DIR, got {spec!r}"); sys.exit(1)
        label, ckpt_dir = spec.split("=", 1)
        print(f"\n=== {label}: {ckpt_dir} ===")
        model, tokenizer, cfg, _ = load_any_model(
            ckpt_dir, device, step=args.step, tokenizer_dir=args.tokenizer_dir)
        r = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
        results[label] = dict(
            ckpt=ckpt_dir, n_layer=cfg.n_layer, n_embd=cfg.n_embd,
            use_mst=bool(getattr(cfg, "use_mst", False)),
            total_params=sum(p.numel() for p in model.parameters()),
            core_metric=r.get("core_metric"),
            results=r.get("results"), centered=r.get("centered_results"),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # table, tasks as rows so it maps onto Table 4 directly
    labels = list(results)
    tasks = sorted({t for l in labels for t in (results[l]["results"] or {})})
    print(f"\n{'task':<24}" + "".join(f"{l:>18}" for l in labels))
    for t in tasks:
        row = f"{t:<24}"
        for l in labels:
            v = (results[l]["results"] or {}).get(t)
            row += f"{v*100:>17.1f} " if isinstance(v, float) else f"{'-':>18}"
        print(row)
    print(f"{'CORE metric':<24}" + "".join(
        f"{results[l]['core_metric']:>17.3f} " if isinstance(results[l]['core_metric'], float)
        else f"{'-':>18}" for l in labels))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
