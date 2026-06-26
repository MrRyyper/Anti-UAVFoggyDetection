"""Spatial density heatmaps and radial-prior analysis for Anti-UAV-RGBT."""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ANN_DIR   = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
SPLITS    = ["train", "val", "test"]
OUT_DIR   = os.path.join(REPO_ROOT, "results", "figures", "radial")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_W, IMG_H = 1920, 1080

# size thresholds on geometric-mean scale sqrt(w*h), px: tiny/small/medium/large
TINY_MAX   = 10
SMALL_MAX  = 50
MEDIUM_MAX = 90
GRID = 50          # heatmap bins per axis


def load_annotations():
    """Return (all_instances, seq_data) keyed by sequence."""
    all_instances = []
    seq_data      = {}
    seq_id        = 0

    cx_img, cy_img = IMG_W / 2.0, IMG_H / 2.0
    max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)

    for split in SPLITS:
        split_dir = os.path.join(ANN_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: not found - {split_dir}")
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

            seq_instances = []
            for exist, rect in zip(data.get("exist", []), data.get("gt_rect", [])):
                if not exist:
                    continue
                if rect is None or len(rect) != 4:
                    continue
                x, y, w, h = rect
                if w <= 0 or h <= 0:
                    continue

                scale   = np.sqrt(w * h)
                bx      = x + w / 2.0
                by      = y + h / 2.0
                dist    = np.sqrt((bx - cx_img) ** 2 + (by - cy_img) ** 2) / max_dist
                bx_norm = bx / IMG_W
                by_norm = by / IMG_H

                inst = dict(scale=scale, dist=dist,
                            bx_norm=bx_norm, by_norm=by_norm,
                            seq_id=seq_id)
                all_instances.append(inst)
                seq_instances.append((scale, dist))

            if seq_instances:
                seq_data[seq_id] = seq_instances
                seq_id += 1

    return all_instances, seq_data


# spatial heatmap: Tiny+Small vs Medium+Large
def plot_heatmap(instances):
    small = [i for i in instances if i["scale"] <  SMALL_MAX]
    large = [i for i in instances if i["scale"] >= SMALL_MAX]

    categories = [
        (small, f"Tiny + Small (scale < {SMALL_MAX}px)",    "Blues"),
        (large, f"Medium + Large (scale $\\geq$ {SMALL_MAX}px)", "Reds"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (subset, title, cmap) in zip(axes, categories):
        if len(subset) == 0:
            ax.set_title(f"{title}\n(no instances)")
            continue

        bx = np.array([i["bx_norm"] for i in subset])
        by = np.array([i["by_norm"] for i in subset])

        heatmap, _, _ = np.histogram2d(bx, by, bins=GRID, range=[[0, 1], [0, 1]])
        heatmap = heatmap.T

        im = ax.imshow(heatmap, origin="upper", cmap=cmap,
                       extent=[0, 1, 1, 0], aspect="auto")
        ax.set_xlabel("Normalised x")
        ax.set_ylabel("Normalised y")
        ax.set_title(f"{title}\n(n = {len(subset):,})")
        ax.plot(0.5, 0.5, "y+", markersize=15, markeredgewidth=2,
                label="Image centre")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, label="Instance count")

    fig.suptitle("Spatial distribution of UAV targets by size category",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "radial_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# scatter: scale vs radial distance (pooled)
def plot_scatter(instances):
    scales = np.array([i["scale"] for i in instances])
    dists  = np.array([i["dist"]  for i in instances])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dists, scales, alpha=0.10, s=3, color="steelblue")
    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Bounding box scale  $\\sqrt{w \\cdot h}$  [px]")
    ax.set_title("UAV bounding box scale vs radial distance (pooled)")

    m, b, r, p, _ = stats.linregress(dists, scales)
    rho, p_s = stats.spearmanr(dists, scales)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=2,
            label=f"Pearson r = {r:.3f}  (p = {p:.2e})\nSpearman $\\rho$ = {rho:.3f}  (p = {p_s:.2e})")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "radial_prior_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return r, p, rho, p_s


# binned mean scale per radial-distance bin
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

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                str(count), ha="center", va="bottom",
                fontsize=7, color="white", rotation=90)

    ax.set_xlabel("Normalised radial distance from image centre")
    ax.set_ylabel("Mean bounding box scale  $\\sqrt{w \\cdot h}$  [px]")
    ax.set_title("Mean UAV scale per radial distance bin  (n per bar shown)")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "radial_prior_binned.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# per-sequence Pearson r distribution
def plot_within_seq(seq_data):
    r_values   = []
    rho_values = []
    for seq_instances in seq_data.values():
        if len(seq_instances) < 10:
            continue
        scales = np.array([s for s, d in seq_instances])
        dists  = np.array([d for s, d in seq_instances])
        if scales.std() < 1e-6 or dists.std() < 1e-6:
            continue
        r,   _ = stats.pearsonr(dists, scales)
        rho, _ = stats.spearmanr(dists, scales)
        r_values.append(r)
        rho_values.append(rho)

    r_values   = np.array(r_values)
    rho_values = np.array(rho_values)
    mean_r     = np.mean(r_values)
    median_r   = np.median(r_values)
    mean_rho   = np.mean(rho_values)
    median_rho = np.median(rho_values)
    pct_neg    = (r_values < 0).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, label, color in [
        (axes[0], r_values,   "Pearson r",   "steelblue"),
        (axes[1], rho_values, "Spearman $\\rho$",  "mediumseagreen"),
    ]:
        mean_v   = np.mean(vals)
        median_v = np.median(vals)
        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(mean_v,   color="crimson",   linewidth=2, linestyle="--",
                   label=f"Mean = {mean_v:.3f}")
        ax.axvline(median_v, color="darkorange", linewidth=2, linestyle=":",
                   label=f"Median = {median_v:.3f}")
        ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
        ax.set_xlabel(f"{label}  (scale vs radial distance)  per sequence")
        ax.set_ylabel("Number of sequences")
        ax.set_title(f"Per-sequence {label} distribution\n"
                     f"({len(vals)} sequences,  {(vals < 0).mean()*100:.1f}% with {label.split()[1]} < 0)")
        ax.legend()

    fig.tight_layout()

    out = os.path.join(OUT_DIR, "radial_prior_within_seq.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return mean_r, median_r, mean_rho, median_rho, pct_neg, len(r_values)


if __name__ == "__main__":
    instances, seq_data = load_annotations()

    if not instances:
        print("ERROR: no annotations found.")
    else:
        print(f"Loaded {len(instances):,} instances from {len(seq_data)} sequences.\n")

        tiny  = [i for i in instances if i["scale"] <  TINY_MAX]
        small = [i for i in instances if TINY_MAX  <= i["scale"] < SMALL_MAX]
        med   = [i for i in instances if SMALL_MAX <= i["scale"] < MEDIUM_MAX]
        large = [i for i in instances if i["scale"] >= MEDIUM_MAX]
        print(f"  Tiny   (< {TINY_MAX}px)            : {len(tiny):,}")
        print(f"  Small  ({TINY_MAX}-{SMALL_MAX}px)          : {len(small):,}")
        print(f"  Medium ({SMALL_MAX}-{MEDIUM_MAX}px)         : {len(med):,}")
        print(f"  Large  (>= {MEDIUM_MAX}px)           : {len(large):,}\n")

        plot_heatmap(instances)
        r, p, rho, p_s = plot_scatter(instances)
        plot_binned(instances)
        mean_r, median_r, mean_rho, median_rho, pct_neg, n_seqs = plot_within_seq(seq_data)

        print(f"\n--- Summary ---")
        print(f"Pooled Pearson r       : {r:.4f}  (p = {p:.2e})")
        print(f"Pooled Spearman rho    : {rho:.4f}  (p = {p_s:.2e})")
        print(f"Per-seq mean r         : {mean_r:.4f}    median r   : {median_r:.4f}")
        print(f"Per-seq mean rho       : {mean_rho:.4f}    median rho : {median_rho:.4f}")
        print(f"Seqs with r < 0        : {pct_neg:.1f}%  ({n_seqs} seqs total)")
