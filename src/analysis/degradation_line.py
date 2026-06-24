import os
import matplotlib.pyplot as plt
import numpy as np

# Repo-root-anchored output dir (src/analysis/ -> repo root is two levels up)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Results data
fog_levels = [0, 2, 5, 9]
labels = ['Clear\n(β=0.05)', 'Light\n(i=2, β=0.07)', 'Moderate\n(i=5, β=0.10)', 'Heavy\n(i=9, β=0.14)']

baseline_map = [0.562, 0.185, 0.076, 0.024]
fogaware_map = [0.514, 0.463, 0.367, 0.085]

baseline_recall = [0.921, 0.346, 0.074, 0.009]
fogaware_recall = [0.905, 0.827, 0.729, 0.233]

x = np.arange(len(fog_levels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: mAP@0.5:0.95 ---
ax = axes[0]
ax.plot(x, baseline_map, 'o-', color='steelblue', linewidth=2, markersize=7, label='Baseline (clear-sky training)')
ax.plot(x, fogaware_map, 's--', color='darkorange', linewidth=2, markersize=7, label='Fog-aware (mixed training)')

# Reference line at baseline clear performance
ax.axhline(y=0.562, color='steelblue', linestyle=':', alpha=0.4, linewidth=1.5)
ax.text(3.05, 0.565, 'baseline\nclear', fontsize=7, color='steelblue', alpha=0.6)

# Annotate clear-sky cost
ax.annotate('-0.048',
            xy=(0, fogaware_map[0]),
            xytext=(0.1, fogaware_map[0] + 0.02),
            fontsize=8, color='darkorange')

# Annotate delta at each fog level
for i in range(1, len(fog_levels)):
    delta = fogaware_map[i] - baseline_map[i]
    ax.annotate(f'+{delta:.3f}',
                xy=(x[i], fogaware_map[i]),
                xytext=(x[i] + 0.05, fogaware_map[i] + 0.02),
                fontsize=8, color='darkorange')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('mAP@0.5:0.95')
ax.set_title('mAP@0.5:0.95 vs Fog Severity')
ax.legend()
ax.set_ylim(0, 0.65)
ax.grid(True, alpha=0.3)

# --- Plot 2: Recall ---
ax = axes[1]
ax.plot(x, baseline_recall, 'o-', color='steelblue', linewidth=2, markersize=7, label='Baseline (clear-sky training)')
ax.plot(x, fogaware_recall, 's--', color='darkorange', linewidth=2, markersize=7, label='Fog-aware (mixed training)')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Recall')
ax.set_title('Recall vs Fog Severity')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fog_degradation_curve.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved to {out_path}")