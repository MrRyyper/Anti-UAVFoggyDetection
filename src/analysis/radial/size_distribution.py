# coding: utf-8
"""
size_distribution.py
--------------------
Plots the overall bounding-box scale distribution with size-category boundaries
and a per-split instance count breakdown.

Outputs (saved next to this script):
    size_distribution.png

Run from any directory:
    python radial_analysis/size_distribution.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths (resolved relative to this file) ──────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ANN_DIR   = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
SPLITS    = ["train", "val", "test"]
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures", "radial")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Size thresholds on geometric-mean scale  sqrt(w*h)  in pixels ───────────
TINY_MAX   = 10
SMALL_MAX  = 50
MEDIUM_MAX = 90   # >= 90 → Large

BOUNDARIES = [TINY_MAX, SMALL_MAX, MEDIUM_MAX]
REGIONS = [
    ("Tiny",   0,          TINY_MAX),
    ("Small",  TINY_MAX,   SMALL_MAX),
    ("Medium", SMALL_MAX,  MEDIUM_MAX),
    ("Large",  MEDIUM_MAX, 280),          # 280 px ≈ upper end of dataset
]
REGION_COLORS = ["#4393c3", "#92c5de", "#f4a582", "#d6604d"]


def size_category(scale):
    if scale < TINY_MAX:
        return "Tiny"
    if scale < SMALL_MAX:
        return "Small"
    if scale < MEDIUM_MAX:
        return "Medium"
    return "Large"


# ── Data loading ─────────────────────────────────────────────────────────────
def load_scales():
    """Return list of (split, scale) for every valid annotated frame."""
    records = []
    for split in SPLITS:
        split_dir = os.path.join(ANN_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: not found – {split_dir}")
            continue
        for root, _, files in os.walk(split_dir):
            if "visible.json" not in files:
                continue
            try:
                with open(os.path.join(root, "visible.json")) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Could not load {root}: {e}")
                continue
            for exist, rect in zip(data.get("exist", []), data.get("gt_rect", [])):
                if not exist:
                    continue
                if rect is None or len(rect) != 4:
                    continue
                x, y, w, h = rect
                if w <= 0 or h <= 0:
                    continue
                records.append((split, np.sqrt(w * h)))
    return records


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot(records):
    splits_arr = np.array([r[0] for r in records])
    scales_arr = np.array([r[1] for r in records])
    cats_arr   = np.array([size_category(s) for s in scales_arr])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left: scale histogram with category shading ──────────────────────────
    ax = axes[0]

    # shade category regions first (behind bars)
    for (label, lo, hi), color in zip(REGIONS, REGION_COLORS):
        ax.axvspan(lo, hi, alpha=0.15, color=color, zorder=0)

    ax.hist(scales_arr, bins=100, range=(0, 270),
            color="steelblue", edgecolor="none", zorder=2)

    for thresh in BOUNDARIES:
        ax.axvline(thresh, color="crimson", linestyle="--", linewidth=1.2, zorder=3)

    # category labels near the top
    ymax = ax.get_ylim()[1]
    for (label, lo, hi), color in zip(REGIONS, REGION_COLORS):
        mid = (lo + hi) / 2
        ax.text(mid, ymax * 0.93, label, ha="center", va="top",
                fontsize=9, fontweight="bold", color="black")

    ax.set_xlabel("Scale √(w×h)  [pixels]")
    ax.set_ylabel("Frame count")
    ax.set_title("Bounding-box scale distribution\nwith size-category boundaries")
    ax.set_xlim(0, 270)

    # ── Right: instance count per category per split ──────────────────────────
    ax = axes[1]
    cat_order   = ["Tiny", "Small", "Medium", "Large"]
    split_order = ["train", "val", "test"]
    bar_colors  = dict(zip(cat_order, REGION_COLORS))

    n_splits = len(split_order)
    n_cats   = len(cat_order)
    x        = np.arange(n_splits)
    width    = 0.18

    for i, cat in enumerate(cat_order):
        counts = [
            int(np.sum((splits_arr == sp) & (cats_arr == cat)))
            for sp in split_order
        ]
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(x + offset, counts, width=width,
                      color=bar_colors[cat], label=cat, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ax.get_ylim()[1] * 0.005,
                        f"{val:,}", ha="center", va="bottom",
                        fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in split_order])
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of instances")
    ax.set_title("Instance count per size category per split")
    ax.legend(title="Size category")

    # overall totals as subtitle
    totals = {cat: int(np.sum(cats_arr == cat)) for cat in cat_order}
    total_str = "  |  ".join(f"{c}: {totals[c]:,}" for c in cat_order)
    fig.suptitle(f"Total: {len(records):,} instances     [{total_str}]",
                 fontsize=9, y=1.01)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "size_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # print table
    print(f"\n{'Category':<10} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 45)
    for cat in cat_order:
        row = {sp: int(np.sum((splits_arr == sp) & (cats_arr == cat)))
               for sp in split_order}
        total = sum(row.values())
        print(f"{cat:<10} {row['train']:>10,} {row['val']:>10,} {row['test']:>10,} {total:>10,}")
    print(f"{'TOTAL':<10} ", end="")
    for sp in split_order:
        print(f"{int(np.sum(splits_arr == sp)):>10,}", end="")
    print(f" {len(records):>10,}")

    print(f"\nScale statistics (sqrt(w*h), px):")
    print(f"  median : {np.median(scales_arr):.1f}")
    print(f"  mean   : {np.mean(scales_arr):.1f}")
    for sp in ["train", "val", "test"]:
        m = np.median(scales_arr[splits_arr == sp])
        print(f"  median ({sp}) : {m:.1f}")
if __name__ == "__main__":
    print(f"Loading annotations from: {ANN_DIR}")
    records = load_scales()
    if not records:
        print("ERROR: no annotations found.")
    else:
        print(f"Loaded {len(records):,} instances.")
        plot(records)
