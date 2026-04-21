"""p25_params_plot.py — Full-model parameter count + FLOPs comparison.

Instantiates Dense, RemixedLinear(MLP gate), and RemixedLinear(Linear gate) on
the meta device (no GPU memory) at each research depth and calls
num_scaling_params() + estimate_flops() for authoritative counts.

Usage (from repo root, venv active):
    python scripts/p25_params_plot.py [--out p25_params_plot.png]
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

    gate_mode=None     → dense (use_remix_linear=False)
    gate_mode='mlp'    → RemixedLinear with MLP basis gate
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
            "use_basis_gate":   True,
            "use_output_gate":  True,
            "use_context":      True,
            "basis_gate_mode":  gate_mode,   # 'mlp' or 'linear'
            "output_gate_rank": 8,
            "sparse_gate_k":    0,
            "gate_temperature": 1.0,
        }

    return config


def count_params_and_flops(config):
    """Returns (param_counts_dict, total_flops, active_flops)."""
    with torch.device("meta"):
        model = GPT(config)
    param_counts = model.num_scaling_params()
    total_flops, active_flops = model.estimate_flops()
    return param_counts, total_flops, active_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="p25_params_plot.png")
    args = parser.parse_args()

    variants = {
        "Dense":                 None,
        "RemixedLinear(MLP)":    "mlp",
        "RemixedLinear(Linear)": "linear",
    }

    print(f"{'Depth':>6}  {'C':>6}  {'Variant':<24}  {'Total Params':>14}  "
          f"{'Active GFLOPs':>14}  {'Params%':>8}  {'FLOPs%':>8}")
    print("─" * 90)

    data = {v: {
        "total":        [],
        "active_flops": [],
        "components":   {c: [] for c in COMPONENTS},
    } for v in variants}
    x_labels = []

    for depth in DEPTHS:
        _, _, C, _ = model_dims(depth)
        x_labels.append(f"d{depth}\n(C={C})")

        dense_total = None
        dense_flops = None
        for vname, gate_mode in variants.items():
            cfg = build_config(depth, gate_mode)
            counts, total_flops, active_flops = count_params_and_flops(cfg)
            total = counts["total"]
            if vname == "Dense":
                dense_total = total
                dense_flops = active_flops

            data[vname]["total"].append(total)
            data[vname]["active_flops"].append(active_flops)
            for comp in COMPONENTS:
                data[vname]["components"][comp].append(counts.get(comp, 0))

            param_pct = 100 * total / dense_total if dense_total else 0
            flop_pct  = 100 * active_flops / dense_flops if dense_flops else 0
            print(f"{depth:>6}  {C:>6}  {vname:<24}  {total:>14,}  "
                  f"{active_flops/1e9:>14.3f}  {param_pct:>7.1f}%  {flop_pct:>7.1f}%")
        print()

    # ─── Plot layout: 4 rows ───────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 16),
        gridspec_kw={"height_ratios": [3, 1, 2, 1]},
    )
    ax_params, ax_params_pct, ax_flops, ax_flops_pct = axes

    fig.patch.set_facecolor("#0F1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

    x = np.arange(len(DEPTHS))
    w = 0.26
    offsets = {"Dense": -w, "RemixedLinear(MLP)": 0, "RemixedLinear(Linear)": w}

    # ── Panel 1: Stacked parameter bars ───────────────────────────────────────
    for vname in variants:
        off = offsets[vname]
        bottom = np.zeros(len(DEPTHS))
        for comp in COMPONENTS:
            vals = np.array(data[vname]["components"][comp], dtype=float)
            alpha = 1.0 if vname == "Dense" else (0.85 if "MLP" in vname else 0.7)
            ax_params.bar(
                x + off, vals, w,
                bottom=bottom,
                color=COMP_COLORS[comp],
                alpha=alpha,
                label=COMP_LABELS[comp] if vname == "Dense" else "_nolegend_",
                zorder=3,
            )
            bottom += vals

        totals = np.array(data[vname]["total"], dtype=float)
        ax_params.plot(
            x + off + w / 2, totals,
            color=PALETTE[vname], marker=MARKERS[vname],
            linewidth=0, markersize=7,
            markerfacecolor=PALETTE[vname],
            markeredgecolor="white", markeredgewidth=0.8,
            label=vname, zorder=5,
        )
        ax_params.annotate(
            f"{totals[-1]/1e6:.0f}M",
            xy=(x[-1] + off + w / 2, totals[-1]),
            xytext=(0, 7), textcoords="offset points",
            color=PALETTE[vname], fontsize=8.5, ha="center", fontweight="bold",
        )

    ax_params.set_xticks(x)
    ax_params.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_params.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax_params.set_ylabel("Total Parameters", color="#C9D1D9", fontsize=11, labelpad=8)
    ax_params.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_params.tick_params(axis="y", colors="#8B949E")
    handles, labels = ax_params.get_legend_handles_labels()
    ax_params.legend(handles, labels, loc="upper left",
                     framealpha=0.3, facecolor="#21262D", edgecolor="#30363D",
                     labelcolor="#C9D1D9", fontsize=9, ncols=2)
    ax_params.set_title(
        "Full Model Parameters & Active FLOPs vs Depth\n"
        r"Dense vs RemixedLinear  ·  basis_size = max(64, $\min(in,out)$ // 4)",
        color="#E6EDF3", fontsize=12, fontweight="bold", pad=12,
    )

    # ── Panel 2: Params % of dense ────────────────────────────────────────────
    dense_params_arr = np.array(data["Dense"]["total"], dtype=float)
    for vname in variants:
        if vname == "Dense":
            continue
        pct = 100 * np.array(data[vname]["total"]) / dense_params_arr
        ax_params_pct.plot(x, pct, marker=MARKERS[vname], color=PALETTE[vname],
                           linewidth=2, markersize=6,
                           markerfacecolor=PALETTE[vname],
                           markeredgecolor="white", markeredgewidth=0.7,
                           label=vname, zorder=3)
        ax_params_pct.annotate(
            f"{pct[-1]:.1f}%", xy=(x[-1], pct[-1]),
            xytext=(7, 0), textcoords="offset points",
            color=PALETTE[vname], fontsize=8.5, va="center", fontweight="bold",
        )

    ax_params_pct.axhline(100, color=PALETTE["Dense"], linewidth=1.2,
                          linestyle="--", alpha=0.6, label="Dense (100%)")
    ax_params_pct.set_xticks(x)
    ax_params_pct.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_params_pct.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_params_pct.set_ylabel("Params\n% of Dense", color="#C9D1D9", fontsize=10, labelpad=8)
    ax_params_pct.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_params_pct.tick_params(axis="y", colors="#8B949E")
    ax_params_pct.legend(loc="upper right", framealpha=0.2, facecolor="#21262D",
                         edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)

    # ── Panel 3: Absolute active FLOPs (GFLOPs per token) ────────────────────
    w2 = 0.26
    for vname in variants:
        off = offsets[vname]
        gflops = np.array(data[vname]["active_flops"], dtype=float) / 1e9
        ax_flops.bar(
            x + off, gflops, w2,
            color=PALETTE[vname],
            alpha=0.85 if vname != "Dense" else 1.0,
            label=vname, zorder=3,
        )
        # annotate final bar
        ax_flops.annotate(
            f"{gflops[-1]:.1f}G",
            xy=(x[-1] + off, gflops[-1]),
            xytext=(0, 5), textcoords="offset points",
            color=PALETTE[vname], fontsize=8, ha="center", fontweight="bold",
        )

    ax_flops.set_xticks(x)
    ax_flops.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_flops.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}G"))
    ax_flops.set_ylabel("Active FLOPs / token\n(fwd + bwd)", color="#C9D1D9",
                        fontsize=10, labelpad=8)
    ax_flops.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_flops.tick_params(axis="y", colors="#8B949E")
    ax_flops.legend(loc="upper left", framealpha=0.25, facecolor="#21262D",
                    edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)
    ax_flops.set_title("Active FLOPs per Token (fwd + bwd × 6)",
                       color="#C9D1D9", fontsize=11, pad=8)

    # ── Panel 4: FLOPs % of dense ─────────────────────────────────────────────
    dense_flops_arr = np.array(data["Dense"]["active_flops"], dtype=float)
    for vname in variants:
        if vname == "Dense":
            continue
        pct = 100 * np.array(data[vname]["active_flops"]) / dense_flops_arr
        ax_flops_pct.plot(x, pct, marker=MARKERS[vname], color=PALETTE[vname],
                          linewidth=2, markersize=6,
                          markerfacecolor=PALETTE[vname],
                          markeredgecolor="white", markeredgewidth=0.7,
                          label=vname, zorder=3)
        ax_flops_pct.annotate(
            f"{pct[-1]:.1f}%", xy=(x[-1], pct[-1]),
            xytext=(7, 0), textcoords="offset points",
            color=PALETTE[vname], fontsize=8.5, va="center", fontweight="bold",
        )

    ax_flops_pct.axhline(100, color=PALETTE["Dense"], linewidth=1.2,
                         linestyle="--", alpha=0.6, label="Dense (100%)")
    ax_flops_pct.set_xticks(x)
    ax_flops_pct.set_xticklabels(x_labels, color="#C9D1D9", fontsize=9)
    ax_flops_pct.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_flops_pct.set_ylabel("FLOPs\n% of Dense", color="#C9D1D9", fontsize=10, labelpad=8)
    ax_flops_pct.set_xlabel("Model Depth  (C = n_embd)", color="#C9D1D9",
                            fontsize=11, labelpad=8)
    ax_flops_pct.grid(axis="y", color="#21262D", linewidth=0.8, zorder=0)
    ax_flops_pct.tick_params(axis="y", colors="#8B949E")
    ax_flops_pct.legend(loc="upper right", framealpha=0.2, facecolor="#21262D",
                        edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)

    plt.tight_layout(h_pad=2.0)
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✓ Saved → {args.out}")


if __name__ == "__main__":
    main()
