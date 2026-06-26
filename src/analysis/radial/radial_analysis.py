"""Correlate UAV bbox size with radial distance from the image centre,
to test the depth prior used in the fog simulation."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ANN_DIR   = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
SPLITS    = ["train", "val", "test"]
IMG_W     = 1920
IMG_H     = 1080
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures", "radial")


def load_annotations():
    """Return (bbox_sizes, radial_dists) over all visible.json instances."""
    bbox_sizes   = []
    radial_dists = []

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

            for exist, rect in zip(exist_list, rect_list):
                if not exist:
                    continue
                if rect is None or len(rect) != 4:
                    continue

                x, y, w, h = rect
                if w <= 0 or h <= 0:
                    continue

                scale = np.sqrt(w * h)
                bx   = x + w / 2.0
                by   = y + h / 2.0
                dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) / max_dist

                bbox_sizes.append(scale)
                radial_dists.append(dist)

    return np.array(bbox_sizes), np.array(radial_dists)


def plot_scatter(bbox_sizes, radial_dists):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(radial_dists, bbox_sizes, alpha=0.15, s=4, color="steelblue")
    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Bounding box scale sqrt(w*h) in pixels")
    ax.set_title("UAV bounding box scale vs radial distance from image centre")

    m, b, r, p, _ = stats.linregress(radial_dists, bbox_sizes)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=2,
            label=f"r = {r:.3f},  p = {p:.2e}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_binned(bbox_sizes, radial_dists, n_bins=10):
    bins        = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    means, stds = [], []

    for i in range(n_bins):
        mask = (radial_dists >= bins[i]) & (radial_dists < bins[i + 1])
        if mask.sum() == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.nanmean(bbox_sizes[mask]))
            stds.append(np.nanstd(bbox_sizes[mask]))

    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_centres, means, width=0.08, color="steelblue", alpha=0.7,
           label="Mean bbox scale")
    ax.errorbar(bin_centres, means, yerr=stds, fmt="none",
                color="black", capsize=4)
    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Mean bounding box scale sqrt(w*h) in pixels")
    ax.set_title("Mean UAV scale per radial distance bin")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "radial_prior_binned.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def save_stats(bbox_sizes, radial_dists):
    r,   p_pearson  = stats.pearsonr(radial_dists, bbox_sizes)
    rho, p_spearman = stats.spearmanr(radial_dists, bbox_sizes)

    lines = [
        f"Total instances analysed : {len(bbox_sizes)}",
        f"",
        f"Pearson r   (linear)  : {r:.4f}   p = {p_pearson:.4e}",
        f"Spearman rho (rank)   : {rho:.4f}   p = {p_spearman:.4e}",
        f"",
        f"Interpretation:",
        f"  Negative r/rho -> larger boxes near centre (supports radial prior)",
        f"  Positive r/rho -> larger boxes toward periphery (contradicts prior)",
        f"  |r| > 0.2 with p < 0.05 considered meaningful",
    ]

    out = os.path.join(OUT_DIR, "radial_prior_stats.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bbox_sizes, radial_dists = load_annotations()

    if len(bbox_sizes) == 0:
        print("ERROR: no annotations loaded. Check ANN_DIR and SPLITS at top of script.")
        return

    print(f"\nLoaded {len(bbox_sizes):,} instances.")

    plot_scatter(bbox_sizes, radial_dists)
    plot_binned(bbox_sizes, radial_dists)
    save_stats(bbox_sizes, radial_dists)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()