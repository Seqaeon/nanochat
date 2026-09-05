"""Paper figure: the Monarch output head against a dense head and a
cost-matched low-rank head.

The point of the figure is the third row of the annotation block.  All three
heads are drawn at the same FLOP budget except the dense reference, and the
low-rank rank is *derived* from the Monarch cost rather than chosen, so the
comparison cannot drift if the configuration changes.

Run:  python figures/monarch_head_figure.py
Out:  figures/monarch_head.pdf  (vector, for LaTeX)
      figures/monarch_head.png  (raster preview)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ---------------------------------------------------------------- configuration
V, D = 131072, 768          # vocabulary, model width
M, M1 = 1024, 128           # Monarch inner width and per-token capacity
M2 = M // M1                # number of blocks
BLOCK_OUT = V // M2

FLOPS_DENSE = V * D
FLOPS_MONARCH = D * M + V * M1
RANK_LOWRANK = FLOPS_MONARCH // (D + V)          # cost-matched, not chosen
FLOPS_LOWRANK = RANK_LOWRANK * (D + V)

# ---------------------------------------------------------------- paper styling
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed TrueType, not Type 3
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0,
    "axes.linewidth": 0.6,
})

INK = "#1A202C"          # text and outlines
FACTOR = "#38618C"       # learned factor matrices
BLOCK = "#C1440E"        # the block-diagonal factor, the thing that is new
VEC = "#E2E8F0"          # activation vectors
ZERO = "#F7FAFC"         # structural zeros

CY = 0.615               # vertical centre every object is hung from
H_V, H_D, H_M, H_R = 0.50, 0.16, 0.21, 0.09
BELOW = CY - H_V / 2 - 0.030   # y for labels hung under a full-height bar
NOTE_Y = 0.225                 # y for the one-line note under each panel
FOOT_TOP = 0.180               # footer box must stay below NOTE_Y


def box(ax, x, w, h, face, label=None, lw=0.7, edge=INK, zorder=2):
    ax.add_patch(Rectangle((x, CY - h / 2), w, h, facecolor=face,
                           edgecolor=edge, linewidth=lw, zorder=zorder))
    if label:
        ax.text(x + w / 2, CY, label, ha="center", va="center",
                color="white" if face in (FACTOR, BLOCK) else INK,
                fontsize=7.5, zorder=zorder + 1)


def shape(ax, x, w, h, text, dy=0.028):
    ax.text(x + w / 2, CY + h / 2 + dy, text, ha="center", va="bottom",
            fontsize=6.2, color=INK)


def arrow(ax, x0, x1, y=None):
    ax.add_patch(FancyArrowPatch((x0, y or CY), (x1, y or CY),
                                 arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.7, color=INK, zorder=1))


def footer(ax, flops, rank, views, per_word, highlight=False):
    lines = [
        f"FLOPs/token   {flops / 1e6:,.1f}M",
        f"matrix rank   {rank}",
        f"views x capacity   {views} x {per_word}",
    ]
    ax.add_patch(Rectangle((0.02, 0.015), 0.96, FOOT_TOP - 0.015, transform=ax.transAxes,
                           facecolor="#FFF7ED" if highlight else "#F7FAFC",
                           edgecolor=BLOCK if highlight else "#CBD5E0",
                           linewidth=0.7, zorder=0))
    for i, line in enumerate(lines):
        key, _, val = line.partition("   ")
        ax.text(0.06, 0.142 - i * 0.050, key, fontsize=6.3, color="#4A5568",
                ha="left", va="center")
        ax.text(0.94, 0.142 - i * 0.050, val, fontsize=6.3, color=INK,
                ha="right", va="center", fontweight="bold")


fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55))
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ------------------------------------------------------------------ (a) dense
ax = axes[0]
ax.set_title("(a) dense head", fontsize=8, color=INK, pad=2)
box(ax, 0.06, 0.045, H_D, VEC)
shape(ax, 0.06, 0.045, H_D, "$h$")
ax.text(0.083, CY - H_D / 2 - 0.028, f"$d\\!=\\!{D}$", ha="center", va="top", fontsize=6.2)
arrow(ax, 0.115, 0.25)
box(ax, 0.25, 0.36, H_V, FACTOR, "$W$")
shape(ax, 0.25, 0.36, H_V, f"$V \\times d$")
arrow(ax, 0.62, 0.755)
box(ax, 0.755, 0.045, H_V, VEC)
shape(ax, 0.755, 0.045, H_V, "logits")
ax.text(0.778, BELOW, f"$V\\!=\\!{V:,}$".replace(",", "{,}"),
        ha="center", va="top", fontsize=6.2)
ax.text(0.5, NOTE_Y, "every word reads all $d$ features",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.1,
        color="#4A5568", style="italic")
footer(ax, FLOPS_DENSE, min(V, D), 1, min(V, D))

# --------------------------------------------------------------- (b) low-rank
ax = axes[1]
ax.set_title("(b) low-rank head, cost-matched", fontsize=8, color=INK, pad=2)
box(ax, 0.03, 0.04, H_D, VEC)
shape(ax, 0.03, 0.04, H_D, "$h$")
arrow(ax, 0.075, 0.15)
box(ax, 0.15, 0.24, H_R, FACTOR, "$A$")
shape(ax, 0.15, 0.24, H_R, f"$r \\times d$")
arrow(ax, 0.395, 0.455)
box(ax, 0.455, 0.04, H_R, VEC)
ax.text(0.475, CY - H_R / 2 - 0.028, f"$r\\!=\\!{RANK_LOWRANK}$",
        ha="center", va="top", fontsize=6.2)
arrow(ax, 0.50, 0.585)
box(ax, 0.585, 0.115, H_V, FACTOR, "$B$")
shape(ax, 0.585, 0.115, H_V, f"$V \\times r$")
arrow(ax, 0.705, 0.815)
box(ax, 0.815, 0.04, H_V, VEC)
shape(ax, 0.815, 0.04, H_V, "logits")
ax.text(0.5, NOTE_Y, "every word reads the same $r$ features",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.1,
        color="#4A5568", style="italic")
footer(ax, FLOPS_LOWRANK, RANK_LOWRANK, 1, RANK_LOWRANK)

# ---------------------------------------------------------------- (c) monarch
ax = axes[2]
ax.set_title("(c) Monarch head (ours)", fontsize=8, color=INK, pad=2)
box(ax, 0.02, 0.04, H_D, VEC)
shape(ax, 0.02, 0.04, H_D, "$h$")
arrow(ax, 0.065, 0.125)
box(ax, 0.125, 0.20, H_M, FACTOR, "$W_1$")
shape(ax, 0.125, 0.20, H_M, f"$M \\times d$")
arrow(ax, 0.33, 0.385)
box(ax, 0.385, 0.04, H_M, VEC)
shape(ax, 0.385, 0.04, H_M, f"$M\\!=\\!{M}$")
# the stride permutation: free, no FLOPs, but it is what makes the blocks mix
arrow(ax, 0.43, 0.465)
ax.add_patch(Rectangle((0.465, CY - H_M / 2), 0.045, H_M, facecolor="white",
                       edgecolor=INK, linewidth=0.7, linestyle=(0, (2, 1.4))))
ax.text(0.4875, CY, "$P$", ha="center", va="center", fontsize=7.5)
arrow(ax, 0.515, 0.565)

# block-diagonal second factor: the structural zeros are the whole story
bx, bw = 0.565, 0.155
ax.add_patch(Rectangle((bx, CY - H_V / 2), bw, H_V, facecolor=ZERO,
                       edgecolor=INK, linewidth=0.7))
for j in range(M2):
    ax.add_patch(Rectangle((bx + j * bw / M2, CY + H_V / 2 - (j + 1) * H_V / M2),
                           bw / M2, H_V / M2, facecolor=BLOCK,
                           edgecolor="white", linewidth=0.35, zorder=3))
ax.text(bx + bw * 0.80, CY + H_V * 0.30, "0", fontsize=7, color="#A0AEC0",
        ha="center", va="center")
ax.text(bx + bw * 0.26, CY - H_V * 0.27, "0", fontsize=7, color="#A0AEC0",
        ha="center", va="center")
shape(ax, bx, bw, H_V, f"${M2}$ blocks")
arrow(ax, bx + bw + 0.008, 0.875)
box(ax, 0.875, 0.04, H_V, VEC)
shape(ax, 0.875, 0.04, H_V, "logits")
ax.text(0.5, NOTE_Y,
        f"each word reads only its block's $m_1\\!=\\!{M1}$ features",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.1,
        color="#4A5568", style="italic")
footer(ax, FLOPS_MONARCH, min(D, M), M2, M1, highlight=True)

fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.06)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/monarch_head.{ext}", dpi=400 if ext == "png" else None)

print(f"V={V} d={D} M={M} m1={M1} m2={M2} block_out={BLOCK_OUT}")
print(f"dense    {FLOPS_DENSE/1e6:8.1f}M  rank {min(V,D):4d}  1 view  x {min(V,D)}")
print(f"low-rank {FLOPS_LOWRANK/1e6:8.1f}M  rank {RANK_LOWRANK:4d}  1 view  x {RANK_LOWRANK}")
print(f"monarch  {FLOPS_MONARCH/1e6:8.1f}M  rank {min(D,M):4d}  {M2} views x {M1}")
print(f"monarch is {FLOPS_DENSE/FLOPS_MONARCH:.2f}x cheaper than dense")
