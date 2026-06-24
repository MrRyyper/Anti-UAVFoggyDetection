# coding: utf-8
"""
radial_prior_analysis.py
------------------------
Analyses whether bounding box size correlates with distance from image center
across the Anti-UAV300 RGB annotation files.

Outputs (saved to ./radial_analysis/)
--------------------------------------
- radial_prior_scatter.png          : scatter of bbox size vs radial distance
- radial_prior_binned.png           : mean bbox size per radial distance bin
- radial_prior_within_seq.png       : distribution of per-sequence correlations
- radial_prior_heatmap_by_size.png  : spatial heatmap of tiny vs large targets
- radial_prior_stats.txt            : all statistics

Place this script inside thesis-1/ and run:
    python radial_prior_analysis.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Config
_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ANN_DIR   = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
SPLITS    = ["train", "val", "test"]
IMG_W     = 1920
IMG_H     = 1080
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures", "radial")

# Size thresholds (geometric mean scale sqrt(w*h) in pixels) -- matches thesis:
#   tiny [0,10), small [10,50), medium [50,90), large [90, inf)
TINY_MAX   = 10
SMALL_MAX  = 50
MEDIUM_MAX = 90
# >= 90 = large


# Annotation loading
def load_annotations():
    """
    Returns:
        all_instances : list of dicts with keys:
                        scale, dist, bx_norm, by_norm, seq_id
        seq_data      : dict seq_id -> list of (scale, dist)
    """
    all_instances = []
    seq_data      = {}
    seq_id        = 0

    cx, cy   = IMG_W / 2.0, IMG_H / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    for split in SPLITS:
        split_dir = os.path.join(ANN_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: split dir not found: {split_dir}")
            continue

        for root, _, files in os.walk(split_dir):
            if "visible.json" not in files:
                continue

            ann_path = os.path.join(root, "visible.json")
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Could not load {ann_path}: {e}")
                continue

            exist_list = data.get("exist", [])
            rect_list  = data.get("gt_rect", [])
            seq_instances = []

            for exist, rect in zip(exist_list, rect_list):
                if not exist:
                    continue
                if rect is None or len(rect) != 4:
                    continue

                x, y, w, h = rect
                if w <= 0 or h <= 0:
                    continue

                scale    = np.sqrt(w * h)
                bx       = x + w / 2.0
                by       = y + h / 2.0
                dist     = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) / max_dist
                bx_norm  = bx / IMG_W
                by_norm  = by / IMG_H

                instance = dict(scale=scale, dist=dist,
                                bx_norm=bx_norm, by_norm=by_norm,
                                seq_id=seq_id)
                all_instances.append(instance)
                seq_instances.append((scale, dist))

            if seq_instances:
                seq_data[seq_id] = seq_instances
                seq_id += 1

    return all_instances, seq_data


# Plot 1: scatter (pooled)
def plot_scatter(instances):
    scales = np.array([i["scale"] for i in instances])
    dists  = np.array([i["dist"]  for i in instances])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dists, scales, alpha=0.10, s=3, color="steelblue")
    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Bounding box scale sqrt(w*h) in pixels")
    ax.set_title("UAV bounding box scale vs radial distance (pooled)")

    m, b, r, p, _ = stats.linregress(dists, scales)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=2,
            label=f"r = {r:.3f},  p = {p:.2e}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return r, p


# Plot 2: binned (pooled)
def plot_binned(instances, n_bins=10):
    scales = np.array([i["scale"] for i in instances])
    dists  = np.array([i["dist"]  for i in instances])

    bins        = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    means, stds, counts = [], [], []

    for i in range(n_bins):
        mask = (dists >= bins[i]) & (dists < bins[i + 1])
        counts.append(mask.sum())
        if mask.sum() == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.nanmean(scales[mask]))
            stds.append(np.nanstd(scales[mask]))

    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bin_centres, means, width=0.08, color="steelblue",
                  alpha=0.7, label="Mean bbox scale")
    ax.errorbar(bin_centres, means, yerr=stds, fmt="none",
                color="black", capsize=4)

    # annotate with instance counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                str(count), ha="center", va="bottom",
                fontsize=7, color="white", rotation=90)

    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Mean bounding box scale sqrt(w*h) in pixels")
    ax.set_title("Mean UAV scale per radial distance bin (n per bar shown)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_binned.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# Plot 3: per-sequence correlation distribution
def plot_within_seq(seq_data):
    """
    Compute Pearson r between scale and radial dist within each sequence.
    Plot distribution of per-sequence r values.
    This removes between-sequence zoom bias.
    """
    r_values = []
    for seq_id, instances in seq_data.items():
        if len(instances) < 10:
            continue
        scales = np.array([s for s, d in instances])
        dists  = np.array([d for s, d in instances])
        # skip if no variance
        if scales.std() < 1e-6 or dists.std() < 1e-6:
            continue
        r, _ = stats.pearsonr(dists, scales)
        r_values.append(r)

    r_values = np.array(r_values)
    mean_r   = np.mean(r_values)
    median_r = np.median(r_values)
    pct_neg  = (r_values < 0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r_values, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(mean_r,   color="crimson",   linewidth=2,
               linestyle="--", label=f"Mean r = {mean_r:.3f}")
    ax.axvline(median_r, color="darkorange", linewidth=2,
               linestyle=":",  label=f"Median r = {median_r:.3f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_xlabel("Pearson r (scale vs radial distance) per sequence")
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"Per-sequence correlation distribution\n"
                 f"({len(r_values)} sequences, {pct_neg:.1f}% with r < 0)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_within_seq.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return mean_r, median_r, pct_neg, len(r_values)


# Plot 4: spatial heatmap by size category
def plot_heatmap_by_size(instances, grid=50):
    """
    2D spatial heatmap showing where tiny vs large targets appear.
    Tiny = scale < TINY_MAX (10px); Large = scale >= MEDIUM_MAX (90px).
    """
    tiny  = [i for i in instances if i["scale"] <  TINY_MAX]
    small = [i for i in instances if TINY_MAX  <= i["scale"] < SMALL_MAX]
    med   = [i for i in instances if SMALL_MAX <= i["scale"] < MEDIUM_MAX]
    large = [i for i in instances if i["scale"] >= MEDIUM_MAX]

    categories = [
        (tiny,  f"Tiny (scale < {TINY_MAX}px)",      "Blues"),
        (large, f"Large (scale >= {MEDIUM_MAX}px)",  "Reds"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (subset, title, cmap) in zip(axes, categories):
        if len(subset) == 0:
            ax.set_title(f"{title}\n(no instances)")
            continue

        bx = np.array([i["bx_norm"] for i in subset])
        by = np.array([i["by_norm"] for i in subset])

        heatmap, xedges, yedges = np.histogram2d(
            bx, by, bins=grid, range=[[0, 1], [0, 1]]
        )
        heatmap = heatmap.T  # transpose for imshow

        im = ax.imshow(heatmap, origin="upper", cmap=cmap,
                       extent=[0, 1, 1, 0], aspect="auto")
        ax.set_xlabel("Normalised x")
        ax.set_ylabel("Normalised y")
        ax.set_title(f"{title}\n(n = {len(subset):,})")

        # mark image centre
        ax.plot(0.5, 0.5, "y+", markersize=15, markeredgewidth=2,
                label="Image centre")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, label="Instance count")

    fig.suptitle("Spatial distribution of UAV targets by size category",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_heatmap_by_size.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # also print size breakdown
    print(f"\nSize breakdown:")
    print(f"  Tiny   (< {TINY_MAX}px)        : {len(tiny):,}")
    print(f"  Small  ({TINY_MAX}-{SMALL_MAX}px)      : {len(small):,}")
    print(f"  Medium ({SMALL_MAX}-{MEDIUM_MAX}px)     : {len(med):,}")
    print(f"  Large  (>= {MEDIUM_MAX}px)       : {len(large):,}")


# Stats file
def save_stats(instances, seq_data, pooled_r, pooled_p,
               mean_r_seq, median_r_seq, pct_neg, n_seqs):

    scales = np.array([i["scale"] for i in instances])
    dists  = np.array([i["dist"]  for i in instances])
    rho, p_spearman = stats.spearmanr(dists, scales)

    lines = [
        f"Total instances analysed : {len(instances):,}",
        f"Total sequences analysed : {n_seqs}",
        f"",
        f"--- Pooled correlation (all instances) ---",
        f"Pearson r            : {pooled_r:.4f}   p = {pooled_p:.4e}",
        f"Spearman rho         : {rho:.4f}   p = {p_spearman:.4e}",
        f"Variance explained   : {pooled_r**2*100:.2f}% (r^2)",
        f"",
        f"--- Per-sequence correlation (removes zoom bias) ---",
        f"Mean r across seqs   : {mean_r_seq:.4f}",
        f"Median r across seqs : {median_r_seq:.4f}",
        f"Sequences with r < 0 : {pct_neg:.1f}%",
        f"",
        f"Interpretation:",
        f"  Negative r/rho -> larger boxes near centre (supports radial prior)",
        f"  Pooled r^2 < 1% -> prior explains almost no variance in scale",
        f"  Per-sequence analysis removes between-sequence zoom/distance bias",
        f"  Majority of sequences with r < 0 supports directional consistency",
        f"  Overall: weak empirical support; prior best treated as heuristic",
    ]

    out = os.path.join(OUT_DIR, "radial_prior_stats.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {out}")


# Main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading annotations from : {ANN_DIR}")
    print(f"Splits                   : {SPLITS}")

    instances, seq_data = load_annotations()

    if len(instances) == 0:
        print("ERROR: no annotations loaded. Check ANN_DIR and SPLITS.")
        return

    print(f"\nLoaded {len(instances):,} instances from {len(seq_data)} sequences.")

    pooled_r, pooled_p             = plot_scatter(instances)
    plot_binned(instances)
    mean_r, median_r, pct_neg, n   = plot_within_seq(seq_data)
    plot_heatmap_by_size(instances)
    save_stats(instances, seq_data, pooled_r, pooled_p,
               mean_r, median_r, pct_neg, n)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()