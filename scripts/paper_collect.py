"""Aggregate the p32 ablation runs into the tables the paper needs.

Each run writes `results_depth_<D>.tsv` (model_name, val_bpb) into its own run
directory, so this walks the sweep output, groups the per-seed runs by
condition, and reports mean +/- s.d. plus a delta against a chosen reference.

    python -m scripts.paper_collect --dir out/p32_paper_ablations
    python -m scripts.paper_collect --dir out/p32_paper_ablations --latex
    python -m scripts.paper_collect --dir out/p32_paper_ablations \\
        --ref C1_full_msw --only C1_

Conditions are the run directory names with the trailing `_s<N>` seed suffix
stripped, i.e. exactly the tags in p32_paper_ablations.sh.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys


def read_run(run_dir):
    """Return {model_name: val_bpb} for one run directory, or {} if unfinished."""
    out = {}
    for tsv in glob.glob(os.path.join(run_dir, "**", "results_depth_*.tsv"),
                         recursive=True):
        with open(tsv) as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                name, val = parts
                if val == "FAILED":
                    out[name] = None
                    continue
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
    return out


def collect(root):
    """{condition: {seed: bpb}} over every run directory under root."""
    conditions = {}
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        if not os.path.isdir(run_dir):
            continue
        m = re.match(r"^(.*)_s(\d+)$", entry)
        if not m:
            continue
        cond, seed = m.group(1), int(m.group(2))
        vals = read_run(run_dir)
        if not vals:
            conditions.setdefault(cond, {})[seed] = None   # started, no result
            continue
        # one model per run in this sweep; take the single value
        finite = [v for v in vals.values() if isinstance(v, float)]
        conditions.setdefault(cond, {})[seed] = min(finite) if finite else None
    return conditions


def summarize(conditions):
    rows = {}
    for cond, seeds in conditions.items():
        got = [v for v in seeds.values() if isinstance(v, float)]
        rows[cond] = dict(
            n_done=len(got), n_started=len(seeds),
            mean=statistics.mean(got) if got else None,
            sd=statistics.stdev(got) if len(got) > 1 else (0.0 if got else None),
            seeds={k: v for k, v in sorted(seeds.items())},
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="sweep output root")
    ap.add_argument("--ref", default=None,
                    help="condition to take deltas against (default: none)")
    ap.add_argument("--only", default=None, help="prefix filter, e.g. C1_")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    ap.add_argument("--out", default=None, help="also write JSON here")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"no such directory: {args.dir}"); sys.exit(1)

    rows = summarize(collect(args.dir))
    if args.only:
        rows = {k: v for k, v in rows.items() if k.startswith(args.only)}
    if not rows:
        print(f"no completed runs found under {args.dir}")
        state = os.path.join(args.dir, "p32_state_d8.json")
        if os.path.exists(state):
            done = json.load(open(state)).get("completed", {})
            print(f"state file lists {len(done)} completed tags; "
                  f"if that is 0 the sweep did not run any conditions")
        sys.exit(0)

    ref_mean = rows.get(args.ref, {}).get("mean") if args.ref else None
    if args.ref and ref_mean is None:
        print(f"warning: reference {args.ref!r} has no result; deltas omitted")

    width = max(len(k) for k in rows)
    print(f"{'condition':<{width}}  {'seeds':>7}  {'bpb':>9}  {'s.d.':>7}  {'delta':>8}")
    print("-" * (width + 38))
    for cond in sorted(rows):
        r = rows[cond]
        mean = f"{r['mean']:.4f}" if r["mean"] is not None else "-"
        sd = f"{r['sd']:.4f}" if r["sd"] is not None else "-"
        if ref_mean is not None and r["mean"] is not None and cond != args.ref:
            delta = f"{r['mean'] - ref_mean:+.4f}"
        else:
            delta = "ref" if cond == args.ref else "-"
        print(f"{cond:<{width}}  {r['n_done']}/{r['n_started']:>5}  "
              f"{mean:>9}  {sd:>7}  {delta:>8}")

    incomplete = [c for c, r in rows.items() if r["n_done"] < r["n_started"]]
    if incomplete:
        print(f"\nincomplete ({len(incomplete)}): {', '.join(sorted(incomplete))}")

    if args.latex:
        print("\n% ---- LaTeX rows ----")
        for cond in sorted(rows):
            r = rows[cond]
            if r["mean"] is None:
                print(f" & {cond.replace('_', ' ')} & -- & -- \\\\"); continue
            d = (f"{r['mean'] - ref_mean:+.4f}"
                 if ref_mean is not None and cond != args.ref else "n/a")
            print(f" & {cond.replace('_', ' ')} & "
                  f"${r['mean']:.4f}\\pm{r['sd']:.4f}$ & {d} \\\\")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
