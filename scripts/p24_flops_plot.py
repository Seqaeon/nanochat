"""p24_flops_plot.py — Analytical active MLP FLOPs per token vs depth for P24 variants.

No model is trained or even run. FLOPs are computed analytically from layer
dimensions. Forward + backward = 6 × matmul params (Chinchilla convention).

Usage (from repo root, with venv active):
    python -m scripts.p24_flops_plot [--out p24_flops.png] [--rs 4] [--min-select 128]
"""
import argparse
import math
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._sweep_utils import model_dims

# ── depths and palette ─────────────────────────────────────────────────────────
DEPTHS = [2, 3, 4, 6, 8, 12, 16, 20]

PALETTE = {
    "Dense":         "#6C63FF",
    "SlicedWeight":  "#00C9A7",
    "FoldedMod":     "#FF6B6B",
    "SequenceGated": "#FFD166",
}
MARKERS = {
    "Dense":         "o",
    "SlicedWeight":  "s",
    "FoldedMod":     "^",
    "SequenceGated": "D",
}

# ── FLOPs helpers ──────────────────────────────────────────────────────────────

def dense_mlp_flops(C: int, n_layers: int) -> int:
    """Forward+backward matmul FLOPs for all MLP layers of a dense model.
    Uses Chinchilla convention: 6 × params = 2 × fwd × (in×out) × 3 dirs.
    Per layer: c_fc (C→4C) + c_proj (4C→C) = 2 × C × 4C + 2 × 4C × C = 16 C²
    ×3 (fwd+bwd) = 48 C² per layer.
    """
    per_layer = 6 * (C * 4 * C + 4 * C * C)   # 6 × (fwd) = fwd+bwd
    return per_layer * n_layers


def sliced_mlp_flops(C: int, n_layers: int, rs: int, min_select: int) -> int:
    """SlicedWeight: selects n_sel columns of the weight per token.
    c_fc:   in=C,   out=4C  →  n_sel_fc  = min(C,   max(min_select, C   // rs))
    c_proj: in=4C,  out=C   →  n_sel_proj = min(4C, max(min_select, 4C  // rs))
    Active matmul: n_sel_fc × 4C  +  n_sel_proj × C  per token per layer.
    """
    n_sel_fc   = min(C,      max(min_select, C   // rs))
    n_sel_proj = min(4 * C,  max(min_select, 4*C // rs))
    per_layer = 6 * (n_sel_fc * 4 * C + n_sel_proj * C)
    return per_layer * n_layers


def folded_mlp_flops(C: int, n_layers: int, rs: int, min_folded: int) -> int:
    """FoldedMod: folds in_features by effective_R, weight is (out, folded_dim).
    folded_dim_raw = in // rs
    if folded_dim_raw < min_folded: effective_R = max(1, in // min_folded)
    else:                           effective_R = rs
    folded_dim = in // effective_R     (≥ 1, ≤ in)
    Active matmul: folded_dim × out  per token.
    """
    def folded(in_f: int) -> int:
        raw = max(1, in_f // rs)
        eff_r = (max(1, in_f // min_folded) if raw < min_folded else rs)
        return max(1, in_f // eff_r)

    fd_fc   = folded(C)        # c_fc folded dim (operates on C, outputs 4C)
    fd_proj = folded(4 * C)    # c_proj folded dim (operates on 4C, outputs C)
    per_layer = 6 * (fd_fc * 4 * C + fd_proj * C)
    return per_layer * n_layers


def seqgated_mlp_flops(C: int, n_layers: int, T: int = 2048) -> int:
    """SequenceGatedLinear: dense weight + amortised gate projection.
    weight is identical to dense (C→4C and 4C→C).
    Gate proj: c_fc gate  = Linear(C,   C)   computed once per sequence → C²/T per token
              c_proj gate = Linear(4C, 4C)   computed once per sequence → 16C²/T per token
    For T=2048 this is negligible vs 16C² dense, but we include it for accuracy.
    """
    dense   = 6 * (C * 4 * C + 4 * C * C)
    # gate projections are amortised over T tokens
    gate_fc   = int(6 * C * C / T)        # (C → C) gate for c_fc, per token
    gate_proj = int(6 * 4*C * 4*C / T)    # (4C → 4C) gate for c_proj, per token
    per_layer = dense + gate_fc + gate_proj
    return per_layer * n_layers


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",           default="p24_active_flops.png")
    parser.add_argument("--rs",            type=int, default=4,
                        help="Reduction scale for SlicedWeight and FoldedMod")
    parser.add_argument("--min-select",    type=int, default=128,
                        help="min_select floor for SlicedWeight")
    parser.add_argument("--min-folded",    type=int, default=128,
                        help="min_folded_dim floor for FoldedMod")
    parser.add_argument("--seq-len",       type=int, default=2048,
                        help="Sequence length (for SequenceGated gate amortisation)")
    args = parser.parse_args()

    rs, min_s, min_fd, T = args.rs, args.min_select, args.min_folded, args.seq_len

    print(f"Analytical P24 MLP FLOPs (Chinchilla ×6)")
    print(f"  RS={rs}  min_select={min_s}  min_folded={min_fd}  seq_len={T}")
    print(f"{'Depth':>6}  {'n_embd':>7}  {'Dense':>12}  {'Sliced':>12}  {'Folded':>12}  {'SeqGated':>12}  {'Sliced%':>8}  {'Folded%':>8}")

    data = {v: [] for v in PALETTE}
    x_labels = []

    for depth in DEPTHS:
        _, _, C, _ = model_dims(depth)
        x_labels.append(f"d{depth}\n(C={C})")

        d = dense_mlp_flops(C, depth)
        s = sliced_mlp_flops(C, depth, rs, min_s)
        f = folded_mlp_flops(C, depth, rs, min_fd)
        g = seqgated_mlp_flops(C, depth, T)

        data["Dense"].append(d)
        data["SlicedWeight"].append(s)
        data["FoldedMod"].append(f)
        data["SequenceGated"].append(g)

        print(f"{depth:>6}  {C:>7}  {d:>12,}  {s:>12,}  {f:>12,}  {g:>12,}  "
              f"{100*s/d:>7.1f}%  {100*f/d:>7.1f}%")

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, (ax_main, ax_pct) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax_main, ax_pct):
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

    x = np.arange(len(DEPTHS))

    # — top panel: absolute FLOPs ———————————————————————————————————————————
    for variant, flops_list in data.items():
        fl = np.array(flops_list, dtype=float)
        ax_main.plot(
            x, fl,
            marker=MARKERS[variant],
            color=PALETTE[variant],
            linewidth=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=variant,
            zorder=3,
        )
        # annotate last point
        ax_main.annotate(
            f"{fl[-1]/1e9:.2f}G",
            xy=(x[-1], fl[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            color=PALETTE[variant],
            fontsize=8.5,
            va="center",
            fontweight="bold",
        )

    # shade saving vs dense
    dense_arr = np.array(data["Dense"], dtype=float)
    best = np.minimum(np.array(data["SlicedWeight"]), np.array(data["FoldedMod"])).astype(float)
    ax_main.fill_between(x, best, dense_arr, color="#6C63FF", alpha=0.07, zorder=1)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_labels, color="#C9D1D9", fontsize=8.5)
    ax_main.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e9:.1f}G"))
    ax_main.set_ylabel("MLP FLOPs / token (fwd+bwd)", color="#C9D1D9",
                        fontsize=11, labelpad=8)
    ax_main.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_main.grid(axis="x", color="#21262D", linewidth=0.4, zorder=0)
    ax_main.tick_params(axis='y', colors="#8B949E")

    legend = ax_main.legend(
        loc="upper left",
        framealpha=0.25,
        facecolor="#21262D",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=11,
    )

    ax_main.set_title(
        f"P24 Variants — Active MLP FLOPs vs Depth  "
        f"(RS={rs}, min_select={min_s}, min_folded={min_fd})",
        color="#E6EDF3",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # — bottom panel: % of dense ———————————————————————————————————————————
    for variant, flops_list in data.items():
        if variant == "Dense":
            continue
        pct = 100 * np.array(flops_list) / dense_arr
        ax_pct.plot(
            x, pct,
            marker=MARKERS[variant],
            color=PALETTE[variant],
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=variant,
            zorder=3,
        )

    ax_pct.axhline(100, color=PALETTE["Dense"], linewidth=1.2,
                   linestyle="--", alpha=0.6, label="Dense (100%)")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(x_labels, color="#C9D1D9", fontsize=8.5)
    ax_pct.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_pct.set_ylabel("% of Dense FLOPs", color="#C9D1D9", fontsize=10, labelpad=8)
    ax_pct.set_xlabel("Model Depth  (n_embd)", color="#C9D1D9", fontsize=11, labelpad=8)
    ax_pct.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_pct.grid(axis="x", color="#21262D", linewidth=0.4, zorder=0)
    ax_pct.tick_params(axis='y', colors="#8B949E")
    ax_pct.legend(loc="upper left", framealpha=0.2, facecolor="#21262D",
                  edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)

    # footer
    footer = (
        f"SlicedWeight RS={rs}: selects max({min_s}, C//RS) input dims/token  │  "
        f"FoldedMod RS={rs}: folds to max({min_fd}, C//RS) dims  │  "
        "SequenceGated: dense weight + amortised gate (≈ Dense)  │  "
        "MLP only (FFN path); attn FLOPs excluded"
    )
    fig.text(0.5, 0.005, footer, ha="center", color="#6E7681", fontsize=7.5)

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✓ Saved: {args.out}")


if __name__ == "__main__":
    main()
