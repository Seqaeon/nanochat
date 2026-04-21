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

DEPTHS = [2, 4, 6, 8, 12, 16, 20]

PALETTE = {
    "Dense":                 "#6C63FF",
    "RemixedLinear(MLP)":    "#FF6B6B",
    "RemixedLinear(Linear)": "#00C9A7",
}
MARKERS = {
    "Dense":                 "o",
    "RemixedLinear(MLP)":    "^",
    "RemixedLinear(Linear)": "s",
}
LINESTYLES = {
    "Dense":                 "-",
    "RemixedLinear(MLP)":    "-",
    "RemixedLinear(Linear)": "-",
}

def get_params(C, depth):
    # DENSE
    dense_per_block = 12 * C * C
    dense_total = dense_per_block * depth

    # REMIXED LINEAR (common structures)
    # 4 layers of D->D (q, k, v, proj)
    # 1 layer of D->4D (c_fc)
    # 1 layer of 4D->D (c_proj)
    # B = min(in, out) // 4 = C // 4
    B = C // 4
    
    def rml_layer_params(in_f, out_f, ctx_dim, mode):
        b = min(in_f, out_f) // 4
        # structural
        struct = (in_f * b) + (out_f * b)
        # output gate
        out_gate = (ctx_dim * 8) + (8 * out_f)
        # basis gate
        if mode == 'mlp':
            hidden = max(ctx_dim // 2, min(b, ctx_dim * 2))
            basis_gate = (ctx_dim * hidden) + (hidden * b)
        elif mode == 'linear':
            basis_gate = ctx_dim * b
        else:
            basis_gate = 0
            
        return struct + out_gate + basis_gate

    # per block
    mlp_qkv_proj = 4 * rml_layer_params(C, C, C, 'mlp')
    mlp_cfc = rml_layer_params(C, 4*C, C, 'mlp')
    mlp_cproj = rml_layer_params(4*C, C, C, 'mlp')
    mlp_per_block = mlp_qkv_proj + mlp_cfc + mlp_cproj
    mlp_total = mlp_per_block * depth

    lin_qkv_proj = 4 * rml_layer_params(C, C, C, 'linear')
    lin_cfc = rml_layer_params(C, 4*C, C, 'linear')
    lin_cproj = rml_layer_params(4*C, C, C, 'linear')
    lin_per_block = lin_qkv_proj + lin_cfc + lin_cproj
    lin_total = lin_per_block * depth

    return dense_total, mlp_total, lin_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="p25_params_plot.png")
    args = parser.parse_args()

    print(f"Analytical P25 Active Parameters (Transformer Blocks Only)")
    print(f"{'Depth':>6}  {'n_embd':>7}  {'Dense':>12}  {'Remix(MLP)':>12}  {'Remix(Lin)':>12}  {'MLP%':>8}  {'Lin%':>8}")

    data = {v: [] for v in PALETTE}
    x_labels = []

    for depth in DEPTHS:
        _, _, C, _ = model_dims(depth)
        x_labels.append(f"d{depth}\n(C={C})")

        d, m, l = get_params(C, depth)

        data["Dense"].append(d)
        data["RemixedLinear(MLP)"].append(m)
        data["RemixedLinear(Linear)"].append(l)

        print(f"{depth:>6}  {C:>7}  {d:>12,}  {m:>12,}  {l:>12,}  "
              f"{100*m/d:>7.1f}%  {100*l/d:>7.1f}%")

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, (ax_main, ax_pct) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax_main, ax_pct):
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

    x = np.arange(len(DEPTHS))

    # — top panel: absolute params ———————————————————————————————————————————
    for variant, p_list in data.items():
        fl = np.array(p_list, dtype=float)
        ax_main.plot(
            x, fl,
            marker=MARKERS[variant],
            linestyle=LINESTYLES[variant],
            color=PALETTE[variant],
            linewidth=2.5,
            markersize=8,
            markerfacecolor=PALETTE[variant],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=variant,
            zorder=3,
        )
        ax_main.annotate(
            f"{fl[-1]/1e6:.1f}M",
            xy=(x[-1], fl[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            color=PALETTE[variant],
            fontsize=9,
            va="center",
            fontweight="bold",
        )

    # shade saving vs dense
    dense_arr = np.array(data["Dense"], dtype=float)
    lin_arr = np.array(data["RemixedLinear(Linear)"], dtype=float)
    ax_main.fill_between(x, lin_arr, dense_arr, color="#00C9A7", alpha=0.1, zorder=1)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_main.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax_main.set_ylabel("Transformer Block Parameters", color="#C9D1D9",
                        fontsize=12, labelpad=8)
    ax_main.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_main.grid(axis="x", color="#21262D", linewidth=0.4, zorder=0)
    ax_main.tick_params(axis='y', colors="#8B949E")

    ax_main.legend(
        loc="upper left",
        framealpha=0.25,
        facecolor="#21262D",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=12,
    )

    ax_main.set_title(
        f"Phase 25 — Parameter Size vs Depth\n(RemixedLinear Gates vs Dense Baseline)",
        color="#E6EDF3",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    # — bottom panel: % of dense ———————————————————————————————————————————
    for variant, p_list in data.items():
        if variant == "Dense":
            continue
        pct = 100 * np.array(p_list) / dense_arr
        ax_pct.plot(
            x, pct,
            marker=MARKERS[variant],
            linestyle=LINESTYLES[variant],
            color=PALETTE[variant],
            linewidth=2,
            markersize=6,
            markerfacecolor=PALETTE[variant],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=variant,
            zorder=3,
        )

    ax_pct.axhline(100, color=PALETTE["Dense"], linewidth=1.2,
                   linestyle="--", alpha=0.6, label="Dense (100%)")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_pct.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_pct.set_ylabel("% of Dense Params", color="#C9D1D9", fontsize=11, labelpad=8)
    ax_pct.set_xlabel("Model Depth  (n_embd)", color="#C9D1D9", fontsize=12, labelpad=8)
    ax_pct.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_pct.grid(axis="x", color="#21262D", linewidth=0.4, zorder=0)
    ax_pct.tick_params(axis='y', colors="#8B949E")
    ax_pct.legend(loc="upper right", framealpha=0.2, facecolor="#21262D",
                  edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✓ Saved: {args.out}")

if __name__ == "__main__":
    main()
