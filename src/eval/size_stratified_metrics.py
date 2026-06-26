#!/usr/bin/env python3
"""Precision, Recall, F1 and mAP@0.5:0.95 stratified by UAV size category for a
YOLOv5m model across all fog levels, plus an 'all' row (no size filter) that
reproduces the aggregate metrics as a sanity check.

Size categories use geometric-mean scale sqrt(w*h):
    tiny [0,10), small [10,50), medium [50,90), large [90,inf).
Per-category attribution follows COCO area-range AP semantics: a TP is counted
under the GT category it matched; an unmatched prediction is an FP under its own
box-size category. Inference mirrors val.py (conf 0.001, NMS IoU 0.6).

Modes: --list (dataset folders), --scan (GT size counts), default (run).
"""

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import torch

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YOLOV5_DIR = os.path.join(REPO_ROOT, "yolov5")
WEIGHTS    = os.path.join(YOLOV5_DIR, "runs", "train", "results8", "weights", "best.pt")
# Fog-aware model: point WEIGHTS at its best.pt and change OUT_CSV.

DATA_BASE    = os.path.join(REPO_ROOT, "prepared_dataset")
CLEAR_FOLDER = "visible"
FOG_GLOB     = "visible_fog_i*_beta*"
SPLIT        = "test"
IMG_EXTS     = (".jpg", ".jpeg", ".png")

IMG_SIZE    = 640      # inference resolution; set 1280 for the resolution experiment
CONF_INFER  = 0.001    # low threshold to capture the full PR curve (val.py default)
NMS_IOU     = 0.60     # NMS IoU (val.py default)
MAX_DET     = 300
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # 0.5, 0.55, ..., 0.95

SIZE_BINS = [
    ("tiny",   0,   10),
    ("small",  10,  50),
    ("medium", 50,  90),
    ("large",  90,  1e9),
]

OUT_CSV = os.path.join(REPO_ROOT, "results", "metrics", "size_stratified_metrics.csv")

CATS = [n for n, _, _ in SIZE_BINS] + ["all"]


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
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                b = tuple(float(x) for x in p[1:5])
                if b[2] > 0 and b[3] > 0:   # skip degenerate (zero-area) GT boxes
                    boxes.append(b)
    return boxes


def xywh_to_xyxy(b):
    xc, yc, w, h = b
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
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


def discover_fog_folders(A=None):
    """Return fog folders for the requested atmospheric light.
    A=None  -> main A=0.5 run: bare folders WITHOUT an _A suffix (i00..i09).
    A='0.30'/'0.70' -> sensitivity run: only folders carrying that _A suffix.
    """
    folders = glob.glob(os.path.join(DATA_BASE, FOG_GLOB))
    if A is None:
        folders = [f for f in folders if "_A" not in os.path.basename(f)]
    else:
        suffix = f"_A{A}"
        folders = [f for f in folders if os.path.basename(f).endswith(suffix)]
    return sorted(folders, key=lambda p: parse_i_beta(os.path.basename(p))[0] or 0)


def compute_ap(recall, precision):
    """101-point interpolated AP with precision envelope (YOLOv5 val.py method)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def pr_curve(tp_confs, fp_confs, npos):
    """Cumulative precision/recall ordered by descending confidence."""
    entries = [(c, 1) for c in tp_confs] + [(c, 0) for c in fp_confs]
    if not entries:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])
    entries.sort(key=lambda e: -e[0])
    tp = fp = 0
    rec, prec, confs = [], [], []
    for c, is_tp in entries:
        tp += is_tp
        fp += 1 - is_tp
        rec.append(tp / npos if npos else 0.0)
        prec.append(tp / (tp + fp))
        confs.append(c)
    return np.array(rec), np.array(prec), np.array(confs)


def best_f1(rec, prec):
    f1 = 2 * prec * rec / (prec + rec + 1e-16)
    k = int(np.argmax(f1))
    return float(prec[k]), float(rec[k]), float(f1[k])


def eval_folder(model, folder, img_size=IMG_SIZE):
    n_iou = len(IOU_THRESHOLDS)
    tp = [{c: [] for c in CATS} for _ in range(n_iou)]
    fp = [{c: [] for c in CATS} for _ in range(n_iou)]
    npos = {c: 0 for c in CATS}

    for img_path in list_images(folder):
        im = cv2.imread(img_path)
        if im is None:
            print(f"  [warn] cannot read {img_path}", file=sys.stderr)
            continue
        H, W = im.shape[:2]

        gt_raw = load_gt(label_path_for(img_path, folder))
        gxy = [xywh_to_xyxy(b) for b in gt_raw]
        gcat = [size_category(b[2] * W, b[3] * H) for b in gt_raw]
        for c in gcat:
            npos[c] += 1
            npos["all"] += 1

        results = model(img_path, size=img_size)
        pr = results.xyxyn[0].cpu().numpy()  # x1,y1,x2,y2,conf,cls (normalized)
        preds = []
        for row in pr:
            x1, y1, x2, y2, conf = row[0], row[1], row[2], row[3], row[4]
            pcat = size_category((x2 - x1) * W, (y2 - y1) * H)
            preds.append((float(conf), (x1, y1, x2, y2), pcat))
        preds.sort(key=lambda p: -p[0])

        for ti, t in enumerate(IOU_THRESHOLDS):
            matched = [False] * len(gxy)
            for conf, pxy, pcat in preds:
                best_j, best = -1, float(t)
                for j in range(len(gxy)):
                    if matched[j]:
                        continue
                    v = iou(gxy[j], pxy)
                    if v >= best:
                        best, best_j = v, j
                if best_j >= 0:
                    matched[best_j] = True
                    gc = gcat[best_j]
                    tp[ti][gc].append(conf)
                    tp[ti]["all"].append(conf)
                else:
                    fp[ti][pcat].append(conf)
                    fp[ti]["all"].append(conf)

    # aggregate metrics per category
    out = {}
    for c in CATS:
        if npos[c] == 0:
            out[c] = (0, None, None, None, None)
            continue
        aps = []
        for ti in range(n_iou):
            rec, prc, _ = pr_curve(tp[ti][c], fp[ti][c], npos[c])
            aps.append(compute_ap(rec, prc))
        mAP = float(np.mean(aps))
        rec0, prc0, _ = pr_curve(tp[0][c], fp[0][c], npos[c])  # IoU 0.5
        P, R, F1 = best_f1(rec0, prc0)
        out[c] = (npos[c], P, R, F1, mAP)
    return out


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--weights", default=WEIGHTS, help="path to model best.pt")
    ap.add_argument("--out", default=OUT_CSV, help="output CSV path")
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE, help="inference resolution")
    ap.add_argument("--A", default=None, choices=["0.30", "0.70"],
                    help="atmospheric light for sensitivity run; omit for main A=0.5 run")
    args = ap.parse_args()

    weights = args.weights
    out_csv = args.out
    img_size = args.imgsz

    fog_folders = discover_fog_folders(A=args.A)
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
            print(f"  {n:7s}: {counts[n]:7d}  ({100 * counts[n] / max(total, 1):.2f}%)")
        return

    print(f"Loading model: {weights}  (img={img_size}, conf={CONF_INFER}, nms_iou={NMS_IOU})")
    model = torch.hub.load(YOLOV5_DIR, "custom", path=weights, source="local")
    model.conf = CONF_INFER
    model.iou = NMS_IOU
    model.max_det = MAX_DET

    conditions = []
    if args.A is None:
        conditions.append(("clear", None, None, clear_folder))  # clear only in main run
    for f in fog_folders:
        i, b = parse_i_beta(os.path.basename(f))
        conditions.append((f"fog{i}", i, b, f))

    rows = []
    for cond, i, b, folder in conditions:
        print(f"[eval] {cond}  ({folder})")
        res = eval_folder(model, folder, img_size=img_size)
        for c in CATS:
            n_gt, P, R, F1, mAP = res[c]
            rows.append([cond, i if i is not None else "", b if b is not None else "",
                         c, n_gt, fmt(P), fmt(R), fmt(F1), fmt(mAP)])
        a = res["all"]
        print(f"   all: n={a[0]} P={fmt(a[1])} R={fmt(a[2])} F1={fmt(a[3])} mAP={fmt(a[4])}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ["condition", "i", "beta", "category", "n_gt", "P", "R", "F1", "mAP@0.5:0.95"]
    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()