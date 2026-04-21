"""p25_params_plot.py — Full-model parameter count comparison.

Instantiates Dense, RemixedLinear(MLP gate), and RemixedLinear(Linear gate) on
the meta device (no GPU memory) at each research depth and calls
num_scaling_params() for the authoritative breakdown.

Usage (from repo root, venv active):
    python -m scripts.p25_params_plot [--out p25_params_plot.png]
"""
import argparse
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
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

# Stack components in this order (bottom to top)
COMPONENTS = ["wte", "value_embeds", "transformer_matrices", "lm_head"]
COMP_COLORS = {
    "wte":                  "#3D405B",
    "value_embeds":         "#81B29A",
    "transformer_matrices": "#F2CC8F",
    "lm_head":              "#E07A5F",
}
COMP_LABELS = {
    "wte":                  "Token Embed (wte)",
    "value_embeds":         "Value Embeds",
    "transformer_matrices": "Transformer Blocks",
    "lm_head":              "LM Head",
}


def build_config(depth, gate_mode=None):
    """Build a GPTConfig for the given depth and gate mode.

    gate_mode=None  → dense (use_remix_linear=False)
    gate_mode='mlp' → RemixedLinear with MLP basis gate
    gate_mode='linear' → RemixedLinear with Linear basis gate
    """
    _, head_dim, model_dim, _ = model_dims(depth)
    num_heads = model_dim // head_dim

    config = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

    if gate_mode is not None:
        config.use_remix_linear = True
        # context_dim = model_dim (matches what research_compare.py passes)
        config.remix_context_dim = model_dim
        # basis_size seed=64; scale_basis_size=True (default) will apply
        # max(64, min(in, out) // 4) per layer → e.g. 192 at depth=12 (C=768)
        config.remix_basis_size = 64
        config.scale_basis_size = True
        config.remixed_linear_kwargs = {
            "use_basis_gate":  True,
            "use_output_gate": True,
            "use_context":     True,
            "basis_gate_mode": gate_mode,   # 'mlp' or 'linear'
            "output_gate_rank": 8,
            "sparse_gate_k":   0,
            "gate_temperature": 1.0,
        }

    return config


def count_params(config):
    with torch.device("meta"):
        model = GPT(config)
    return model.num_scaling_params()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="p25_params_plot.png")
    args = parser.parse_args()

    variants = {
        "Dense":                 None,
        "RemixedLinear(MLP)":    "mlp",
        "RemixedLinear(Linear)": "linear",
    }

    print(f"{'Depth':>6}  {'C':>6}  {'Variant':<24}  {'Blocks':>12}  {'Total':>12}  {'%Dense':>8}")
    print("─" * 80)

    data = {v: {"total": [], "components": {c: [] for c in COMPONENTS}}
            for v in variants}
    x_labels = []

    for depth in DEPTHS:
        _, _, C, _ = model_dims(depth)
        x_labels.append(f"d{depth}\n(C={C})")

        dense_total = None
        for vname, gate_mode in variants.items():
            cfg = build_config(depth, gate_mode)
            counts = count_params(cfg)
            total = counts["total"]
            if vname == "Dense":
                dense_total = total

            data[vname]["total"].append(total)
            for comp in COMPONENTS:
                data[vname]["components"][comp].append(counts.get(comp, 0))

            pct = 100 * total / dense_total if dense_total else 0
            print(f"{depth:>6}  {C:>6}  {vname:<24}  "
                  f"{counts['transformer_matrices']:>12,}  {total:>12,}  {pct:>7.1f}%")
        print()

    # ─── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_main, ax_pct) = plt.subplots(
        2, 1, figsize=(13, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0F1117")
    for ax in (ax_main, ax_pct):
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

    x = np.arange(len(DEPTHS))
    w = 0.26  # bar width
    offsets = {"Dense": -w, "RemixedLinear(MLP)": 0, "RemixedLinear(Linear)": w}

    # ── Top panel: stacked bars per variant ───────────────────────────────────
    for vname in variants:
        off = offsets[vname]
        bottom = np.zeros(len(DEPTHS))
        for ci, comp in enumerate(COMPONENTS):
            vals = np.array(data[vname]["components"][comp], dtype=float)
            color = COMP_COLORS[comp]
            # Slightly lighter/darker per variant
            alpha = 1.0 if vname == "Dense" else (0.85 if "MLP" in vname else 0.7)
            bar = ax_main.bar(
                x + off, vals, w,
                bottom=bottom,
                color=color,
                alpha=alpha,
                label=COMP_LABELS[comp] if vname == "Dense" else "_nolegend_",
                zorder=3,
            )
            bottom += vals

        # Total line connecting bar tops
        totals = np.array(data[vname]["total"], dtype=float)
        ax_main.plot(
            x + off + w / 2, totals,
            color=PALETTE[vname],
            marker=MARKERS[vname],
            linewidth=0,
            markersize=7,
            markerfacecolor=PALETTE[vname],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=vname,
            zorder=5,
        )
        # Annotate final depth bar
        ax_main.annotate(
            f"{totals[-1]/1e6:.0f}M",
            xy=(x[-1] + off + w / 2, totals[-1]),
            xytext=(0, 7),
            textcoords="offset points",
            color=PALETTE[vname],
            fontsize=8.5,
            ha="center",
            fontweight="bold",
        )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_main.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax_main.set_ylabel("Total Parameters", color="#C9D1D9", fontsize=12, labelpad=8)
    ax_main.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_main.tick_params(axis="y", colors="#8B949E")

    # Two-part legend: variants (markers) then components (fill)
    handles, labels = ax_main.get_legend_handles_labels()
    ax_main.legend(
        handles, labels,
        loc="upper left",
        framealpha=0.3,
        facecolor="#21262D",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=10,
        ncols=2,
    )
    ax_main.set_title(
        "Full Model Parameters vs Depth — Dense vs RemixedLinear(MLP/Linear gate)\n"
        r"(basis_size = max(64, $\min(in,out)$ // 4), scale_basis_size=True)",
        color="#E6EDF3",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # ── Bottom panel: % of dense ───────────────────────────────────────────────
    dense_arr = np.array(data["Dense"]["total"], dtype=float)
    for vname in variants:
        if vname == "Dense":
            continue
        pct = 100 * np.array(data[vname]["total"]) / dense_arr
        ax_pct.plot(
            x, pct,
            marker=MARKERS[vname],
            color=PALETTE[vname],
            linewidth=2,
            markersize=6,
            markerfacecolor=PALETTE[vname],
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=vname,
            zorder=3,
        )
        # Annotate final point
        ax_pct.annotate(
            f"{pct[-1]:.1f}%",
            xy=(x[-1], pct[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            color=PALETTE[vname],
            fontsize=8.5,
            va="center",
            fontweight="bold",
        )

    ax_pct.axhline(100, color=PALETTE["Dense"], linewidth=1.2,
                   linestyle="--", alpha=0.6, label="Dense (100%)")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_pct.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_pct.set_ylabel("% of Dense", color="#C9D1D9", fontsize=11, labelpad=8)
    ax_pct.set_xlabel("Model Depth  (n_embd)", color="#C9D1D9", fontsize=11, labelpad=8)
    ax_pct.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_pct.tick_params(axis="y", colors="#8B949E")
    ax_pct.legend(loc="upper right", framealpha=0.2, facecolor="#21262D",
                  edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✓ Saved → {args.out}")


if __name__ == "__main__":
    main()
