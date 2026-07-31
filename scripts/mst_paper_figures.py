"""Figure 1 for the MST paper: three-panel scaling comparison.

    python -m scripts.mst_paper_tables
    python -m scripts.mst_paper_figures

Writes MST_iclr2027/fig_scaling.pdf (vector, Type-1 fonts, serif to match the
body text).

Design notes
  form      scatter + fitted power law, log-log; the job is "does one curve sit
            below the other over the measured range", so points must stay visible
            and the fit must not hide them.
  color     two categorical slots only (blue = dense, orange = MST). Validated
            all-pairs: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white.
            MST points below the amortization threshold reuse the SAME hue with
            a hollow marker, because "below threshold" is a state of the MST
            series, not a third identity.
  labels    selective: endpoints of each series only. No number on every point.
  grid      recessive; no dual axes; legend once, in panel 1.
"""
import json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
TABLE = os.path.join(HERE, "scratch", "mst_paper_tables.json")
OUT = os.path.join(HERE, "MST_iclr2027", "fig_scaling.pdf")

C_DENSE = "#2a78d6"
C_MST = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

MST_FIT_MIN_DEPTH = 16

AXES = [
    ("total_params", "Total parameters", 1e6, "M"),
    ("flops_per_token", "FLOPs per token", 1e9, "G"),
    ("train_flops", "Training FLOPs", 1e18, r"$\times 10^{18}$"),
]


def fit(points):
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return math.exp(my - b * mx), b


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
    })


def main():
    rows = [r for r in json.load(open(TABLE)) if r["bpb"] is not None]
    dense = sorted([r for r in rows if r["arm"] == "dense"], key=lambda r: r["depth"])
    mst = sorted([r for r in rows if r["arm"] == "mst"], key=lambda r: r["depth"])
    mst_hi = [r for r in mst if r["depth"] >= MST_FIT_MIN_DEPTH]
    mst_lo = [r for r in mst if r["depth"] < MST_FIT_MIN_DEPTH]

    style()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15), sharey=True)

    for ax, (key, xlabel, scale, unit) in zip(axes, AXES):
        da, db = fit([(r[key], r["bpb"]) for r in dense])
        ma, mb = fit([(r[key], r["bpb"]) for r in mst_hi])

        # fitted curves, drawn first so points sit on top
        for a, b, src, col in ((da, db, dense, C_DENSE), (ma, mb, mst_hi, C_MST)):
            lo, hi = min(r[key] for r in src), max(r[key] for r in src)
            xs = [lo * (hi / lo) ** (i / 120) for i in range(121)]
            ax.plot([x / scale for x in xs], [a * x ** b for x in xs],
                    color=col, lw=1.1, alpha=0.55, zorder=2, solid_capstyle="round")

        ax.plot([r[key] / scale for r in dense], [r["bpb"] for r in dense],
                ls="none", marker="o", ms=3.6, mfc=C_DENSE, mec="white", mew=0.5,
                zorder=3, label="Dense baseline")
        ax.plot([r[key] / scale for r in mst_hi], [r["bpb"] for r in mst_hi],
                ls="none", marker="s", ms=3.6, mfc=C_MST, mec="white", mew=0.5,
                zorder=4, label="MST ($N{=}4$)")
        ax.plot([r[key] / scale for r in mst_lo], [r["bpb"] for r in mst_lo],
                ls="none", marker="s", ms=3.6, mfc="white", mec=C_MST, mew=0.9,
                zorder=4, label="MST, below threshold")

        # Selective direct labels: series endpoints only, placed in offset points
        # so the same offsets hold across three different x scales.
        for r, off, col, ha, va in (
            (mst_hi[0],  (-5, -6), C_MST,   "right", "top"),
            (mst_hi[-1], (0, -8),  C_MST,   "center", "top"),
            (dense[0],   (4, 6),   C_DENSE, "left",  "bottom"),
            (dense[-1],  (5, 4),   C_DENSE, "left",  "bottom"),
        ):
            ax.annotate(f"$L{{=}}{r['depth']}$", (r[key] / scale, r["bpb"]),
                        textcoords="offset points", xytext=off,
                        color=col, fontsize=6.2, ha=ha, va=va, zorder=6)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{xlabel} ({unit})" if unit != r"$\times 10^{18}$"
                      else f"{xlabel} ({unit})")
        ax.grid(True, which="major", color=GRID, lw=0.4, alpha=0.9, zorder=0)
        ax.grid(True, which="minor", color=GRID, lw=0.25, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # The headline, stated once per panel, in the empty upper-right wedge.
        ratios = [((r["bpb"] / da) ** (1 / db)) / r[key] for r in mst_hi]
        ax.text(0.965, 0.93, f"dense needs {sum(ratios)/len(ratios):.2f}$\\times$",
                transform=ax.transAxes, fontsize=6.8, color=INK_2,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID, lw=0.4))

    axes[0].set_ylabel("Validation bits per byte")
    for ax in axes:
        ax.set_yticks([0.75, 0.85, 0.95, 1.05])
        ax.set_yticklabels(["0.75", "0.85", "0.95", "1.05"])
        ax.minorticks_off()
        ax.set_ylim(0.695, 1.13)                      # headroom for the top marker
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * 0.80, hi * 1.30)             # room for endpoint labels

    handles = [
        Line2D([], [], ls="none", marker="s", ms=3.6, mfc=C_MST, mec="white",
               mew=0.5, label="MST ($N{=}4$)"),
        Line2D([], [], ls="none", marker="s", ms=3.6, mfc="white", mec=C_MST,
               mew=0.9, label="MST, $L\\leq 9$ (below threshold)"),
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc=C_DENSE, mec="white",
               mew=0.5, label="Dense baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.075),
               ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.4)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
