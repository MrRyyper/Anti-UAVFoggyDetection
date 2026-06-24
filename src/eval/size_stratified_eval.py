#!/usr/bin/env python3
# coding: utf-8
"""
size_stratified_eval.py
-----------------------
Compute RECALL stratified by UAV size category for a YOLOv5m model across all
fog severity levels (answers SQ2 / H2).

Why this is needed: YOLOv5 val.py reports only aggregate metrics. Size-stratified
recall requires per-box predictions matched to ground truth and bucketed by GT
size. Fog is applied post-hoc, so GT is identical across fog levels; only the
predictions change per level.

Size categories (geometric mean scale s = sqrt(w_px * h_px)), matching the thesis:
    tiny   [0, 10)
    small  [10, 50)
    medium [50, 90)
    large  [90, inf)

Output: one CSV row per condition (clear + each fog level) with per-category
GT count, detected count, and recall, plus an overall row.

Diagnostic modes:
    --list   list the dataset folders found (+ image counts), then exit
    --scan   print the GT size-category distribution for the test split, then exit
             (use this first: the 27 tiny instances are across the WHOLE RGB set;
              the test split has fewer, which is why tiny recall is coarse)

Run inside the yolo venv on a GPU node:
    module load 2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
    source $HOME/envs/yolo/bin/activate
    python size_stratified_eval.py            # full run
    python size_stratified_eval.py --scan     # inspect GT size distribution first
"""

import argparse
import glob
import os
import re
import sys

import cv2
import torch

# ============================ CONFIG ============================
YOLOV5_DIR = os.path.expanduser("~/thesis-1/yolo/yolov5")
WEIGHTS    = os.path.expanduser("~/thesis-1/yolo/yolov5/runs/train/results8/weights/best.pt")
# For the fog-aware model, point WEIGHTS at its best.pt and change OUT_CSV.

DATA_BASE    = "/scratch-shared/glevybirkental/prepared_dataset"
CLEAR_FOLDER = "visible"                # clear-sky test set
FOG_GLOB     = "visible_fog_i*_beta*"   # per-level fog folders
SPLIT        = "test"                   # images/<SPLIT>, labels/<SPLIT>
IMG_EXTS     = (".jpg", ".jpeg", ".png")

IMG_SIZE    = 640    # inference resolution; set 1280 for the resolution experiment
CONF_THRESH = 0.25   # operating point (matches the qualitative detection figures)
NMS_IOU     = 0.45   # NMS IoU during inference
MATCH_IOU   = 0.50   # IoU threshold for counting a GT as detected

# label, lower px (inclusive), upper px (exclusive)
SIZE_BINS = [
    ("tiny",   0,   10),
    ("small",  10,  50),
    ("medium", 50,  90),
    ("large",  90,  1e9),
]

OUT_CSV = os.path.expanduser("~/thesis-1/results/metrics/size_stratified_recall.csv")
# ================================================================


def list_images(folder):
    img_dir = os.path.join(folder, "images", SPLIT)
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(img_dir, "*" + ext))
    return sorted(imgs)


def label_path_for(img_path, folder):
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(folder, "labels", SPLIT, base + ".txt")


def load_gt(label_path):
    """Return list of (xc, yc, w, h) normalized boxes."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                boxes.append(tuple(float(x) for x in p[1:5]))
    return boxes


def xywh_to_xyxy(b):
    xc, yc, w, h = b
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def size_category(w_px, h_px):
    s = (w_px * h_px) ** 0.5
    for name, lo, hi in SIZE_BINS:
        if lo <= s < hi:
            return name
    return SIZE_BINS[-1][0]


def parse_i_beta(name):
    m = re.search(r"_i(\d+)_beta([0-9.]+)", name)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def discover_fog_folders():
    folders = glob.glob(os.path.join(DATA_BASE, FOG_GLOB))
    return sorted(folders, key=lambda p: parse_i_beta(os.path.basename(p))[0] or 0)


def eval_folder(model, folder):
    """Greedy match preds -> GT; count recalled GT per size category."""
    cat_gt  = {n: 0 for n, _, _ in SIZE_BINS}
    cat_det = {n: 0 for n, _, _ in SIZE_BINS}

    for img_path in list_images(folder):
        im = cv2.imread(img_path)
        if im is None:
            print(f"  [warn] cannot read {img_path}", file=sys.stderr)
            continue
        H, W = im.shape[:2]

        gt_raw   = load_gt(label_path_for(img_path, folder))
        gt_xyxy  = [xywh_to_xyxy(b) for b in gt_raw]
        gt_cat   = [size_category(b[2] * W, b[3] * H) for b in gt_raw]
        gt_taken = [False] * len(gt_raw)
        for c in gt_cat:
            cat_gt[c] += 1

        results = model(img_path, size=IMG_SIZE)
        pred = results.xyxyn[0].cpu().numpy()  # x1,y1,x2,y2,conf,cls (normalized)
        if pred.size:
            pred = pred[pred[:, 4] >= CONF_THRESH]
            pred = pred[pred[:, 4].argsort()[::-1]]  # high conf first

        for p in pred:
            pb = (p[0], p[1], p[2], p[3])
            best_j, best_iou = -1, MATCH_IOU
            for j in range(len(gt_xyxy)):
                if gt_taken[j]:
                    continue
                v = iou(gt_xyxy[j], pb)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                gt_taken[best_j] = True
                cat_det[gt_cat[best_j]] += 1

    return cat_gt, cat_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list dataset folders and exit")
    ap.add_argument("--scan", action="store_true", help="print GT size distribution and exit")
    args = ap.parse_args()

    fog_folders = discover_fog_folders()
    clear_folder = os.path.join(DATA_BASE, CLEAR_FOLDER)

    if args.list:
        print(f"clear: {clear_folder}  (images: {len(list_images(clear_folder))})")
        for f in fog_folders:
            i, b = parse_i_beta(os.path.basename(f))
            print(f"  i={i} beta={b}  {f}  (images: {len(list_images(f))})")
        return

    if args.scan:
        counts = {n: 0 for n, _, _ in SIZE_BINS}
        for img_path in list_images(clear_folder):
            im = cv2.imread(img_path)
            if im is None:
                continue
            H, W = im.shape[:2]
            for b in load_gt(label_path_for(img_path, clear_folder)):
                counts[size_category(b[2] * W, b[3] * H)] += 1
        total = sum(counts.values())
        print(f"GT size-category distribution ({SPLIT} split, n={total}):")
        for n, _, _ in SIZE_BINS:
            pct = 100 * counts[n] / max(total, 1)
            print(f"  {n:7s}: {counts[n]:7d}  ({pct:.2f}%)")
        return

    print(f"Loading model: {WEIGHTS}")
    model = torch.hub.load(YOLOV5_DIR, "custom", path=WEIGHTS, source="local")
    model.conf = CONF_THRESH
    model.iou = NMS_IOU

    header = ["condition", "i", "beta"]
    for n, _, _ in SIZE_BINS:
        header += [f"{n}_n", f"{n}_det", f"{n}_recall"]
    header += ["overall_n", "overall_det", "overall_recall"]

    conditions = [("clear", None, None, clear_folder)]
    for f in fog_folders:
        i, b = parse_i_beta(os.path.basename(f))
        conditions.append((f"fog{i}", i, b, f))

    rows = []
    for cond, i, b, folder in conditions:
        print(f"[eval] {cond}  ({folder})")
        cat_gt, cat_det = eval_folder(model, folder)
        row = [cond, i if i is not None else "", b if b is not None else ""]
        tot_gt = tot_det = 0
        for n, _, _ in SIZE_BINS:
            g, d = cat_gt[n], cat_det[n]
            row += [g, d, round(d / g, 4) if g else 0.0]
            tot_gt += g
            tot_det += d
        row += [tot_gt, tot_det, round(tot_det / tot_gt, 4) if tot_gt else 0.0]
        rows.append(row)
        print("   " + "  ".join(f"{n}={cat_det[n]}/{cat_gt[n]}" for n, _, _ in SIZE_BINS))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()