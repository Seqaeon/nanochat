"""Figure: stream-level analysis, from the paper_probe JSON outputs.

    python -m scripts.paper_probe --only b2,b3,b6,b7,b8 --out scratch/probe.json
    python -m scripts.paper_figures_analysis --probe scratch/probe.json

Writes MST_iclr2027/fig_analysis.pdf. Accepts several --probe files, since the
probes are often run in separate invocations; later files win on conflicts.

Design notes
  form   bars for per-stream magnitudes (identity + magnitude), lines for
         anything indexed by layer or position (change over an ordered axis).
  color  four categorical slots, one per stream, assigned in fixed order and
         never cycled. Every line is direct-labelled, which the palette's relief
         rule requires for the lower-contrast slots on a light surface.
  axes   one scale per panel; reference levels (the -1/(N-1) similarity floor,
         log N for entropy) are drawn as annotated rules, not second axes.
"""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "MST_iclr2027", "fig_analysis.pdf")

# categorical slots 1-4, fixed order, never cycled
STREAM = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
    })


def tidy(ax):
    ax.grid(True, color=GRID, lw=0.4, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_damage(ax, b6, b2):
    """Per-stream damage from deleting that stream."""
    if b6:
        vals = [r["mean_delta"] for r in b6]; lab = r"$\Delta$ token loss (nats)"
    elif b2:
        vals = [r["delta"] for r in b2["per_stream"]]; lab = r"$\Delta$ bpb"
    else:
        return False
    n = len(vals)
    ax.bar(range(n), vals, color=STREAM[:n], width=0.62,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.2, color=INK2)
    ratio = max(vals) / min(vals)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{i}" for i in range(n)])
    ax.set_xlabel("stream deleted"); ax.set_ylabel(lab)
    ax.set_title(f"(a) every stream is load-bearing\nspread only {ratio:.2f}$\\times$",
                 loc="left")
    ax.set_ylim(0, max(vals) * 1.22)
    tidy(ax)
    return True


def panel_profile(ax, b6):
    """Damage by token bucket, normalised by each stream's own mean.

    Normalising removes the overall-importance differences that panel (a)
    already shows, so what is left is the *shape* of each stream's contribution.
    A stream that is uniformly useful is flat at 1.0; a stream that specialises
    slopes. Frequency is preferred over position when both are available: the
    frequency signal is roughly five times larger.
    """
    if not b6:
        return False
    use_freq = "by_freq" in b6[0] and b6[0]["by_freq"]
    key = "by_freq" if use_freq else "by_pos"
    if key not in b6[0] or not b6[0][key]:
        return False
    keys = list(b6[0][key])
    for r in b6:
        y = [r[key][k] / r["mean_delta"] for k in keys]
        c = STREAM[r["stream"] % len(STREAM)]
        ax.plot(range(len(keys)), y, marker="o", ms=3.2, lw=1.5, color=c, zorder=3)
        ax.annotate(f"  {r['stream']}", (len(keys) - 1, y[-1]), color=c,
                    fontsize=6.6, va="center", fontweight="bold")
    ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(len(keys)))
    if use_freq:
        ax.set_xticklabels(["rarest\n25%", "25-50%", "50-75%", "commonest\n25%"])
        ax.set_xlabel("token frequency quartile")
        ax.set_title("(b) streams split by token frequency", loc="left")
    else:
        ax.set_xticklabels(["early", "middle", "late"])
        ax.set_xlabel("position in sequence")
        ax.set_title("(b) positional profile", loc="left")
    ax.set_ylabel("damage / stream's own mean")
    ax.set_xlim(-0.3, len(keys) - 0.5)
    tidy(ax)
    return True


def panel_lens(ax, b7):
    """How much of the output head each stream accounts for on its own."""
    if not b7:
        return False
    vals = [100 * r["top1_agreement"] for r in b7]
    n = len(vals)
    ax.bar(range(n), vals, color=STREAM[:n], width=0.62,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=6.2, color=INK2)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{i}" for i in range(n)])
    ax.set_xlabel("stream, decoded alone")
    ax.set_ylabel("top-1 agreement (%)")
    ax.set_title("(c) no stream reproduces the model", loc="left")
    ax.set_ylim(0, max(vals) * 1.3)
    tidy(ax)
    return True


def panel_cosine(ax, b34, n_subs):
    if not b34 or not b34.get("sub_sim"):
        return False
    sims = {int(k): v for k, v in b34["sub_sim"].items()}
    xs = sorted(sims)
    floor = -1.0 / (n_subs - 1)
    ax.plot(xs, [sims[l] for l in xs], lw=1.4, color=STREAM[0], zorder=3)
    ax.axhline(0.0, color=INK2, lw=0.6, zorder=2)
    ax.axhline(floor, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(xs[-1], floor, f" floor {floor:.2f}", fontsize=5.8, color=INK2,
            va="bottom", ha="right")
    ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise cosine")
    ax.set_title("(d) streams stay near-orthogonal", loc="left")
    tidy(ax)
    return True


def panel_entropy(ax, b34, n_subs):
    if not b34 or not b34.get("route_entropy"):
        return False
    import math
    ents = {int(k): v for k, v in b34["route_entropy"].items()}
    xs = sorted(ents)
    mx = math.log(n_subs)
    ax.plot(xs, [ents[l] for l in xs], lw=1.4, color=STREAM[1], zorder=3)
    ax.axhline(mx, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(xs[0], mx, f" uniform, $\\log N$ = {mx:.2f}", fontsize=5.8,
            color=INK2, va="bottom")
    ax.set_xlabel("layer"); ax.set_ylabel("router entropy (nats)")
    ax.set_title("(e) the router summarises, it does not select", loc="left")
    ax.set_ylim(min(ents.values()) * 0.97, mx * 1.035)
    tidy(ax)
    return True


def panel_weights(ax, b8):
    if not b8:
        return False
    order = [k for k in ("distribute_w", "c_q_w", "c_k_w", "c_v_w", "fc_w") if k in b8]
    for i, k in enumerate(order):
        y = b8[k]
        c = STREAM[i % len(STREAM)] if k == "distribute_w" else INK2
        lw = 1.5 if k == "distribute_w" else 0.8
        al = 1.0 if k == "distribute_w" else 0.55
        ax.plot(range(len(y)), y, lw=lw, color=c, alpha=al, zorder=3)
        if k == "distribute_w":
            ax.annotate("  $W^{D}_i$", (len(y) - 1, y[-1]), color=c,
                        fontsize=6.4, va="center")
    ax.axhline(0.0, color=INK2, lw=0.6, zorder=2)
    ax.text(0.02, 0.02, "Q, K, V, FFN at chance", transform=ax.transAxes,
            fontsize=6.0, color=INK2)
    ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise cosine")
    ax.set_title("(f) the per-stream matrices diverged", loc="left")
    tidy(ax)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    data = {}
    for f in args.probe:
        if not os.path.exists(f):
            print(f"missing: {f}"); sys.exit(1)
        data.update(json.load(open(f)))
    n_subs = data.get("n_subs", 4)

    style()
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9))
    ok = [
        panel_damage(axes[0][0], data.get("b6"), data.get("b2")),
        panel_profile(axes[0][1], data.get("b6")),
        panel_lens(axes[0][2], data.get("b7")),
        panel_cosine(axes[1][0], data.get("b3b4"), n_subs),
        panel_entropy(axes[1][1], data.get("b3b4"), n_subs),
        panel_weights(axes[1][2], data.get("b8")),
    ]
    for a, drawn in zip(axes.ravel(), ok):
        if not drawn:
            a.axis("off")
            a.text(0.5, 0.5, "no data", ha="center", va="center",
                   color=GRID, fontsize=8, transform=a.transAxes)
    print(f"panels drawn: {sum(ok)}/6")
    fig.tight_layout(pad=0.5, w_pad=1.4, h_pad=1.6)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.02, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
