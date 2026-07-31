"""Power-law fits and iso-cost multipliers for the MST paper, from the recomputed
cost table produced by scripts/mst_paper_tables.py.

    python -m scripts.mst_paper_tables && python -m scripts.mst_paper_fits
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(HERE, "scratch", "mst_paper_tables.json")

AXES = [("total_params", "Total parameters"),
        ("flops_per_token", "FLOPs / token"),
        ("train_flops", "Training FLOPs")]

# The headline fit uses MST L>=16; smaller models sit below the amortization
# threshold and are reported separately (see paper, Section 5.2).
MST_FIT_MIN_DEPTH = 16


def fit(points):
    """OLS in log-log space: y = a * x^b. Returns (a, b, R^2)."""
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    ss_res = sum((y - (math.log(a) + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


def main():
    rows = json.load(open(TABLE))
    have = [r for r in rows if r["bpb"] is not None]
    dense = [r for r in have if r["arm"] == "dense"]
    mst_all = [r for r in have if r["arm"] == "mst"]
    mst_fit = [r for r in mst_all if r["depth"] >= MST_FIT_MIN_DEPTH]

    out = {"fits": {}, "ratios": {}}
    for key, label in AXES:
        da, db, dr2 = fit([(r[key], r["bpb"]) for r in dense])
        ma, mb, mr2 = fit([(r[key], r["bpb"]) for r in mst_fit])
        aa, ab, ar2 = fit([(r[key], r["bpb"]) for r in mst_all])
        out["fits"][key] = dict(dense=(da, db, dr2), mst=(ma, mb, mr2), mst_all=(aa, ab, ar2))

        print(f"\n### {label}")
        print(f"  dense  (L={dense[0]['depth']}-{dense[-1]['depth']}, n={len(dense)}): "
              f"bpb = {da:.4g} x^{db:+.4f}   R2={dr2:.4f}")
        print(f"  MST    (L>={MST_FIT_MIN_DEPTH}, n={len(mst_fit)}): "
              f"bpb = {ma:.4g} x^{mb:+.4f}   R2={mr2:.4f}")
        print(f"  MST    (all, n={len(mst_all)}): "
              f"bpb = {aa:.4g} x^{ab:+.4f}   R2={ar2:.4f}   [reported as sensitivity]")

        # Iso-quality multiplier: resource a dense model needs to reach each MST bpb.
        ratios = []
        print(f"  {'L':>4} {'x':>12} {'bpb':>8} {'dense needs':>13} {'ratio':>7}")
        for r in mst_all:
            need = (r["bpb"] / da) ** (1.0 / db)
            ratio = need / r[key]
            ratios.append((r["depth"], ratio))
            flag = "" if r["depth"] >= MST_FIT_MIN_DEPTH else "  (below threshold)"
            print(f"  {r['depth']:>4} {r[key]:>12.4g} {r['bpb']:>8.4f} "
                  f"{need:>13.4g} {ratio:>7.2f}x{flag}")
        headline = [x for d, x in ratios if d >= MST_FIT_MIN_DEPTH]
        print(f"  mean over L>={MST_FIT_MIN_DEPTH}: {sum(headline)/len(headline):.3f}x "
              f"(min {min(headline):.2f}, max {max(headline):.2f})")
        out["ratios"][key] = dict(per_depth=ratios,
                                  mean=sum(headline) / len(headline),
                                  lo=min(headline), hi=max(headline))

    dest = os.path.join(HERE, "scratch", "mst_paper_fits.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")

    # LaTeX bodies
    print("\n% ---- Table 2 body (iso-quality multipliers) ----")
    depths = [r["depth"] for r in mst_fit]
    print("MST $L$ & " + " & ".join(str(d) for d in depths) + r" & \textbf{mean} \\")
    names = {"total_params": "Total parameters", "flops_per_token": "FLOPs / token",
             "train_flops": "Training FLOPs"}
    for key, _ in AXES:
        per = dict(out["ratios"][key]["per_depth"])
        cells = " & ".join(f"{per[d]:.2f}" for d in depths)
        print(f"{names[key]:<16} & {cells} & "
              f"\\textbf{{{out['ratios'][key]['mean']:.2f}}}$\\times$ \\\\")

    print("\n% ---- Table 3 body (power-law fits) ----")
    for key, label in AXES:
        d, m = out["fits"][key]["dense"], out["fits"][key]["mst"]
        print(f"{label:<16} & ${d[0]:.4g}\\,x^{{{d[1]:.4f}}}$ & {d[2]:.3f} & "
              f"${m[0]:.4g}\\,x^{{{m[1]:.4f}}}$ & {m[2]:.3f} \\\\")


if __name__ == "__main__":
    main()
