import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# mAP@0.5:0.95 grid -- rows: i=2,5,9 ; cols: A=0.3,0.5,0.7
baseline = {2:[0.287,0.189,0.133], 5:[0.106,0.077,0.053], 9:[0.036,0.024,0.017]}
fogaware = {2:[0.466,0.466,0.485], 5:[0.356,0.368,0.389], 9:[0.093,0.093,0.048]}
A_vals = [0.3,0.5,0.7]; levels=[2,5,9]
A_colors = ["#4575b4","#999999","#d73027"]

fig, axes = plt.subplots(1,2, figsize=(11,4.2), sharey=True)
for ax, (data,title) in zip(axes, [(baseline,"Baseline"),(fogaware,"Fog-aware")]):
    x = np.arange(len(levels)); w=0.25
    for j,A in enumerate(A_vals):
        vals=[data[i][j] for i in levels]
        ax.bar(x+(j-1)*w, vals, w, label=f"$A={A}$", color=A_colors[j], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"$i={i}$" for i in levels])
    ax.set_title(title); ax.set_xlabel("Fog severity")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("mAP@0.5:0.95")
axes[1].legend(title="Atmospheric light", frameon=True)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "sensitivity.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"saved {out_path}")