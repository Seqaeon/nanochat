"""Generate CORE benchmark scaling curve plot for NeurIPS paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────
# Dense baselines — total=active for dense, CORE scores from eval
dense = [
    {"depth": 12, "total": 286261730,  "active": 286261730,  "flops": 7.596959e8,  "bpp": 0.9030, "core": 0.1140},
    {"depth": 14, "total": 399114882,  "active": 399114882,  "flops": 1.128533e9,  "bpp": 0.8688, "core": 0.1480},
    {"depth": 17, "total": 536871738,  "active": 536871738,  "flops": 1.585452e9,  "bpp": 0.8364, "core": 0.1850},
    {"depth": 20, "total": 896533746,  "active": 896533746,  "flops": 2.886213e9,  "bpp": 0.7906, "core": 0.2150},
    {"depth": 25, "total": 1384122122, "active": 1384122122, "flops": 4.775225e9,  "bpp": 0.7545, "core": 0.2570},
    {"depth": 26, "total": 1681786938, "active": 1681786938, "flops": 5.991051e9,  "bpp": 0.7382, "core": 0.2660},
    {"depth": 30, "total": 2396654306, "active": 2396654306, "flops": 8.977137e9,  "bpp": 0.7120, "core": 0.2910},
]

# RemixedLinear (P29 chunk64, K=8)
remix = [
    {"label": "d8",  "total": 276100416, "active": 127174464, "flops": 2.943332e8,  "bpp": 0.9028, "core": 0.1229},
    {"label": "d12", "total": 791881440, "active": 289582944, "flops": 7.583897e8,  "bpp": 0.8139, "core": 0.1717},
]


def fit_power_law(x, y):
    """Fit y = a * x^b in log-log space."""
    lx, ly = np.log(x), np.log(y)
    b, a_log = np.polyfit(lx, ly, 1)
    return np.exp(a_log), b


# ── Plot ────────────────────────────────────────────────────────────────
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
    dy = np.array([d["core"] for d in dense])
    a, b = fit_power_law(dx, dy)

    # Extrapolated curve
    x_min = min(dx.min(), min(r[key] for r in remix)) * 0.3
    x_max = dx.max() * 1.5
    x_fit = np.geomspace(x_min, x_max, 200)
    y_fit = a * x_fit ** b

    ax.plot(x_fit / 1e6, y_fit, '--', color='#888780', linewidth=1.5,
            label='Dense baseline (power law)', zorder=1)
    ax.scatter(dx / 1e6, dy, c='#888780', s=30, zorder=5,
               label='Dense (measured)', edgecolors='white', linewidths=0.5)

    # Annotate dense points
    for d in dense:
        ax.annotate(f"d{d['depth']}", (d[key] / 1e6, d["core"]),
                     textcoords="offset points", xytext=(5, -10),
                     fontsize=7, color='#888780')

    # RemixedLinear points
    rx = np.array([r[key] for r in remix])
    ry = np.array([r["core"] for r in remix])
    ax.scatter(rx / 1e6, ry, c='#D4537E', s=80, zorder=10,
               label='RemixedLinear (ours)',
               edgecolors='white', linewidths=1.0, marker='o')
    for r in remix:
        ax.annotate(r["label"], (r[key] / 1e6, r["core"]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8, color='#D4537E', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('CORE Score (centered accuracy)' if key == "total" else '', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(0.05, 0.35)

axes[0].legend(fontsize=7.5, loc='lower right', framealpha=0.9)
fig.suptitle('CORE Benchmark: RemixedLinear vs Dense Baseline', fontsize=12,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('scaling_curves_core.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('scaling_curves_core.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Saved scaling_curves_core.png and scaling_curves_core.pdf")

# Print where remix points fall relative to the dense power-law
for r in remix:
    for key in ["active", "flops"]:
        dx = np.array([d[key] for d in dense])
        dy = np.array([d["core"] for d in dense])
        a, b = fit_power_law(dx, dy)
        predicted = a * r[key] ** b
        actual = r["core"]
        print(f"  {r['label']} ({key}): predicted={predicted:.4f}, actual={actual:.4f}, "
              f"delta={actual - predicted:+.4f} ({(actual - predicted)/predicted*100:+.1f}%)")
