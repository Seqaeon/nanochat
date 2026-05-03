"""Generate publication-quality scaling curve plot for ACL paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────
# Dense baselines (Feb 8 nanochat runs, d12-d30)
dense = [
    {"depth": 12, "total": 286261730,  "active": 286261730,  "flops": 7.596959e8,  "bpp": 0.9030},
    {"depth": 14, "total": 399114882,  "active": 399114882,  "flops": 1.128533e9,  "bpp": 0.8688},
    {"depth": 16, "total": 536871738,  "active": 536871738,  "flops": 1.585452e9,  "bpp": 0.8364},
    {"depth": 18, "total": 701891594,  "active": 701891594,  "flops": 2.179995e9,  "bpp": 0.8131},
    {"depth": 20, "total": 896533746,  "active": 896533746,  "flops": 2.886213e9,  "bpp": 0.7906},
    {"depth": 22, "total": 1123157490, "active": 1123157490, "flops": 3.763086e9,  "bpp": 0.7714},
    {"depth": 24, "total": 1384122122, "active": 1384122122, "flops": 4.775225e9,  "bpp": 0.7545},
    {"depth": 26, "total": 1681786938, "active": 1681786938, "flops": 5.991051e9,  "bpp": 0.7382},
    {"depth": 28, "total": 2018511234, "active": 2018511234, "flops": 7.365736e9,  "bpp": 0.7246},
    {"depth": 30, "total": 2396654306, "active": 2396654306, "flops": 8.977137e9,  "bpp": 0.7120},
]

# RemixedLinear (P29 chunk64, K=8, aspect_ratio=64)
remix = [
    {"label": "d4",  "total": 55655968,  "active": 37003936,  "flops": 7.967938e7,  "bpp": 1.082087},
    {"label": "d8",  "total": 276100416, "active": 127174464, "flops": 2.943332e8,  "bpp": 0.902843},
    {"label": "d12", "total": 791881440, "active": 289582944, "flops": 7.583897e8,  "bpp": 0.8139},
]

# Standard MoE (P29, d4 only)
moe = [
    {"label": "MoE top-1",   "total": 51388552, "active": 36708488, "flops": 7.790669e7, "bpp": 1.186575},
    {"label": "MoE top-all", "total": 51388552, "active": 51388552, "flops": 1.659871e8, "bpp": 1.186305},
]


def fit_power_law(x, y):
    """Fit y = a * x^b in log-log space."""
    lx, ly = np.log(x), np.log(y)
    b, a_log = np.polyfit(lx, ly, 1)
    return np.exp(a_log), b


# ── Plot setup ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
})

x_keys = [
    ("total",  "Total Parameters"),
    ("active", "Active Parameters"),
    ("flops",  "Active FLOPs"),
]

for ax, (key, xlabel) in zip(axes, x_keys):
    # Dense baseline points + power-law fit
    dx = np.array([d[key] for d in dense])
    dy = np.array([d["bpp"] for d in dense])
    a, b = fit_power_law(dx, dy)

    # Extrapolated curve
    x_min = min(dx.min(), min(r[key] for r in remix), min(m[key] for m in moe)) * 0.5
    x_max = dx.max() * 1.5
    x_fit = np.geomspace(x_min, x_max, 200)
    y_fit = a * x_fit ** b

    ax.plot(x_fit / 1e6, y_fit, '--', color='#888780', linewidth=1.5, label='Dense baseline (power law)', zorder=1)
    ax.scatter(dx / 1e6, dy, c='#888780', s=30, zorder=5, label='Dense (measured)', edgecolors='white', linewidths=0.5)

    # RemixedLinear points
    rx = np.array([r[key] for r in remix])
    ry = np.array([r["bpp"] for r in remix])
    ax.scatter(rx / 1e6, ry, c='#D4537E', s=80, zorder=10, label='RemixedLinear (ours)',
               edgecolors='white', linewidths=1.0, marker='o')
    for r in remix:
        ax.annotate(r["label"], (r[key] / 1e6, r["bpp"]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8, color='#D4537E', fontweight='bold')

    # MoE points
    mx = np.array([m[key] for m in moe])
    my = np.array([m["bpp"] for m in moe])
    ax.scatter(mx / 1e6, my, c='#E24B4A', s=60, zorder=10, label='Standard MoE (d4)',
               edgecolors='white', linewidths=1.0, marker='s')

    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Validation BPP' if key == "total" else '', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(0.65, 1.25)

axes[0].legend(fontsize=7.5, loc='upper right', framealpha=0.9)
fig.suptitle('RemixedLinear vs Dense Baseline: Scaling Curves', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('scaling_curves_acl.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('scaling_curves_acl.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Saved scaling_curves_acl.png and scaling_curves_acl.pdf")
