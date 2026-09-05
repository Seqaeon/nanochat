"""Paper figure: the full Monarch output head, block-diagonal factor plus the
shared low-rank residual, and where its rank comes from.

Companion to monarch_head_figure.py, which shows the factorisation alone. This one
shows the head that actually beats dense, and answers the question that head invites:
once the residual is 79% of the cost, is this still anything but low-rank?

All numbers are the measured depth-4, V=32,768 configuration, so the figure and the
results table cannot disagree.

Run:  python figures/monarch_full_head_figure.py
Out:  figures/monarch_full_head.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# ------------------------------------------------- measured d4 configuration
V, D = 32768, 256
M, M1, R = 256, 32, 128
M2, BLOCK_OUT = M // M1, V // (M // M1)
BODY = 3.427610e7 - 6 * (D * M + V * M1)          # everything but the head

def model_flops(head):  return BODY + head
MON   = 6 * (D * M + V * M1)
RES   = 6 * R * (D + V)
LOWR  = int((D * M + V * M1) / (D + V)) + R        # cost-matched rank, derived
BARS = [
    ("dense softmax",       min(V, D), model_flops(6 * V * D),        1.1596),
    ("Monarch + residual",  min(D, min(M, D) + R), model_flops(MON + RES), 1.1757),
    ("low-rank, matched cost", min(LOWR, D), model_flops(6 * LOWR * (D + V)), 1.2179),
]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 7.0, "axes.linewidth": 0.6,
})
INK, FACTOR, BLOCK, VEC, ZERO = "#1A202C", "#38618C", "#C1440E", "#E2E8F0", "#F7FAFC"
RESID = "#2F7A5A"          # the residual branch, distinct from both factors

fig = plt.figure(figsize=(6.8, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.22,
                      left=0.005, right=0.965, top=0.89, bottom=0.17)

# =========================================================== (a) architecture
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) the full head", fontsize=8, color=INK, pad=3)
CY, RY = 0.635, 0.215
H_V, H_D, H_M, H_R = 0.44, 0.13, 0.17, 0.075

def box(x, w, h, y, face, label=None, dashed=False):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=face, edgecolor=INK,
                           linewidth=0.7, zorder=2,
                           linestyle=(0, (2, 1.4)) if dashed else "solid"))
    if label:
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=7.5,
                color="white" if face in (FACTOR, RESID) else INK, zorder=3)

def top(x, w, h, y, text, align="center"):
    ax.text(x if align == "left" else x + w / 2, y + h / 2 + 0.025, text,
            ha=align, va="bottom", fontsize=6.2)

def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.7, color=INK,
                                 zorder=1, **kw))

box(0.02, 0.032, H_D, CY, VEC); top(0.02, 0.032, H_D, CY, "$h$")
ax.text(0.036, CY - H_D / 2 - 0.028, f"$d\\!=\\!{D}$",
        ha="center", va="top", fontsize=6.2)
ax.plot([0.062, 0.062], [RY, CY], color=INK, linewidth=0.7, zorder=1)   # the split

# --- Monarch path -----------------------------------------------------------
arrow(0.062, CY, 0.105, CY)
box(0.105, 0.105, H_M, CY, FACTOR, "$W_1$"); top(0.105, 0.105, H_M, CY, "$M \\times d$")
arrow(0.212, CY, 0.243, CY)
box(0.243, 0.030, H_M, CY, VEC); top(0.243, 0.030, H_M, CY, f"$M\\!=\\!{M}$")
arrow(0.275, CY, 0.300, CY)
box(0.300, 0.036, H_M, CY, "white", "$P$", dashed=True)
arrow(0.338, CY, 0.368, CY)
bx, bw = 0.368, 0.115
ax.add_patch(Rectangle((bx, CY - H_V / 2), bw, H_V, facecolor=ZERO,
                       edgecolor=INK, linewidth=0.7))
for j in range(M2):
    ax.add_patch(Rectangle((bx + j * bw / M2, CY + H_V / 2 - (j + 1) * H_V / M2),
                           bw / M2, H_V / M2, facecolor=BLOCK, edgecolor="white",
                           linewidth=0.35, zorder=3))
top(bx, bw, H_V, CY, f"${M2}$ blocks")

# --- residual path ----------------------------------------------------------
arrow(0.062, RY, 0.105, RY)
box(0.105, 0.105, H_R, RY, RESID, "$A$"); top(0.105, 0.105, H_R, RY, "$r \\times d$")
arrow(0.212, RY, 0.243, RY)
box(0.243, 0.030, H_R, RY, VEC)
ax.text(0.258, RY - H_R / 2 - 0.026, f"$r\\!=\\!{R}$", ha="center", va="top", fontsize=6.2)
arrow(0.275, RY, 0.300, RY)
box(0.300, 0.075, 0.22, RY - 0.025, RESID, "$C$")
top(0.300, 0.075, 0.22, RY - 0.025, "$V \\times r$")

# --- sum and output ---------------------------------------------------------
sx = 0.545
arrow(bx + bw, CY, sx - 0.022, CY)
ax.add_patch(FancyArrowPatch((0.375, RY - 0.025), (sx, RY - 0.025), arrowstyle="-",
                             linewidth=0.7, color=INK, zorder=1))
arrow(sx, RY - 0.025, sx, CY - 0.022)
ax.add_patch(Circle((sx, CY), 0.021, facecolor="white", edgecolor=INK,
                    linewidth=0.7, zorder=4))
ax.text(sx, CY, "+", ha="center", va="center", fontsize=8, zorder=5)
arrow(sx + 0.022, CY, 0.600, CY)
box(0.600, 0.032, H_V, CY, VEC)
top(0.600, 0.032, H_V, CY,
    "logits\n" + f"$V\\!=\\!{V:,}$".replace(",", "{,}"))

ax.text(0.83, CY + 0.10,
        f"$m_1\\!=\\!{M1}$ block-private\ndirections per word\n"
        f"$+\\ r\\!=\\!{R}$ shared\nacross all {M2} blocks",
        ha="center", va="center", fontsize=6.3, color="#4A5568", linespacing=1.6)
ax.text(0.83, RY + 0.02,
        f"head costs {(MON + RES) / 1e6:.1f}M MACs,\n"
        f"{RES / (MON + RES):.0%} of it the residual",
        ha="center", va="center", fontsize=6.3, color="#4A5568", linespacing=1.6)

# ============================================================ (b) rank vs cost
ax = fig.add_subplot(gs[1])
ax.set_title("(b) rank reached, at what cost", fontsize=8, color=INK, pad=3)
ys = [2, 1, 0]
cols = ["#94A3B8", BLOCK, RESID]
for y, (lab, rank, fl, bpb), c in zip(ys, BARS, cols):
    ax.barh(y, rank, height=0.46, color=c, edgecolor=INK, linewidth=0.6, zorder=2)
    ax.text(325, y, f"{rank}", va="center", ha="right", fontsize=7.5,
            fontweight="bold", color=INK)
    ax.text(8, y, lab, va="center", ha="left", fontsize=6.8, color="white",
            fontweight="bold", zorder=3)
    ax.text(8, y - 0.32, f"{fl / 1e6:.1f}M FLOPs/token    bpb {bpb:.4f}",
            va="top", ha="left", fontsize=6.1, color="#4A5568")
ax.set_yticks([]); ax.set_ylim(-0.95, 2.6); ax.set_xlim(0, 340)
ax.set_xlabel("rank ceiling of the logit matrix", fontsize=6.8)
ax.tick_params(axis="x", labelsize=6.2, length=2)
ax.axvline(D, color=INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=3)
ax.text(D - 8, -0.90, f"$d\\!=\\!{D}$: the ceiling any linear head has",
        ha="right", va="bottom", fontsize=6.0, color=INK)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/monarch_full_head.{ext}", dpi=400 if ext == "png" else None)
print(f"M={M} m1={M1} m2={M2} block_out={BLOCK_OUT} r={R}  cost-matched low-rank={LOWR}")
for lab, rank, fl, bpb in BARS:
    print(f"  {lab:<22} rank {rank:4d}  {fl/1e6:7.2f}M/token  bpb {bpb:.4f}")
