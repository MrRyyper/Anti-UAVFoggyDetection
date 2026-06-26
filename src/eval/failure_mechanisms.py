#!/usr/bin/env python3
"""Three failure-mechanism analyses (TP confidence, FP rate, IoU stability) for
the baseline and fog-aware models across clear + fog_i00..i09, from a single
GT-matching pass over YOLOv5 COCO best_predictions.json files."""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parents[2]
VAL_DIR       = REPO_ROOT / "yolov5" / "runs" / "val"
GT_LABELS_DIR = REPO_ROOT / "prepared_dataset" / "visible" / "labels" / "test"
OUT_DIR       = REPO_ROOT / "results" / "figures"   # PNG figures
CSV_DIR       = REPO_ROOT / "results" / "metrics"   # summary CSV
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H  = 1920, 1080      # Anti-UAV300 RGB frame size
CONF_THRES    = 0.25            # detections below this are ignored for TP/FP
IOU_THRES     = 0.50            # IoU needed to count a prediction as a TP

FOG_LEVELS = ['clear'] + [f'fog_i{i:02d}' for i in range(10)]

# folder-name prefixes per model (latest matching folder is auto-selected)
PREFIX = {
    'baseline': {'clear': 'clear',                   'fog': 'fog_i{ii}'},
    'fogaware': {'clear': 'fogaware_mixedval_clear', 'fog': 'fogaware_mixedval_fog_i{ii}'},
}


def latest_run_folder(prefix):
    """Most recently modified runs/val folder matching prefix, excluding
    A-sensitivity variants (A0.30 / A0.70)."""
    candidates = [p for p in glob.glob(str(VAL_DIR / f"{prefix}*"))
                  if os.path.isdir(p) and "A0." not in os.path.basename(p)]
    if not candidates:
        return None
    return Path(max(candidates, key=os.path.getmtime))


def run_folder(model, level):
    if level == 'clear':
        return latest_run_folder(PREFIX[model]['clear'])
    ii = level.split('fog_i')[1]
    return latest_run_folder(PREFIX[model]['fog'].format(ii=ii))


def load_gt():
    """Load GT boxes once -> {image_id: (N,4) xyxy px}."""
    gt = {}
    for fp in glob.glob(str(GT_LABELS_DIR / "*.txt")):
        img_id = Path(fp).stem
        boxes = []
        with open(fp) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([(cx-w/2)*IMG_W, (cy-h/2)*IMG_H,
                              (cx+w/2)*IMG_W, (cy+h/2)*IMG_H])
        gt[img_id] = np.array(boxes, dtype=float) if boxes else np.empty((0, 4))
    return gt


def iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,))
    xx1 = np.maximum(box[0], boxes[:, 0]); yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2]); yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None)
    area_b = (box[2]-box[0])*(box[3]-box[1])
    area_s = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area_b + area_s - inter
    return np.where(union > 0, inter/union, 0.0)


def match_condition(pred_file, gt):
    """Greedy GT matching. Returns tp_conf, tp_iou, n_tp, n_fp."""
    with open(pred_file) as f:
        preds = json.load(f)

    by_img = {}
    for d in preds:
        if d['score'] < CONF_THRES:
            continue
        x, y, w, h = d['bbox']
        by_img.setdefault(d['image_id'], []).append((d['score'], [x, y, x+w, y+h]))

    tp_conf, tp_iou = [], []
    n_tp = n_fp = 0
    for img_id, dets in by_img.items():
        dets.sort(key=lambda t: t[0], reverse=True)
        gboxes = gt.get(img_id, np.empty((0, 4)))
        used = np.zeros(len(gboxes), dtype=bool)
        for score, box in dets:
            ious = iou_xyxy(np.array(box), gboxes)
            if len(ious):
                ious_masked = np.where(used, -1, ious)
                j = int(np.argmax(ious_masked)); best = ious_masked[j]
            else:
                best = -1
            if best >= IOU_THRES:
                n_tp += 1; used[j] = True
                tp_conf.append(score); tp_iou.append(best)
            else:
                n_fp += 1
    return tp_conf, tp_iou, n_tp, n_fp


def main():
    print("Loading ground-truth labels...")
    gt = load_gt()
    print(f"  GT images: {len(gt)}  (total boxes: {sum(len(v) for v in gt.values())})")
    if len(gt) == 0:
        print(f"  ERROR: no GT label files in {GT_LABELS_DIR} - fix GT_LABELS_DIR.")
        return

    stats = {m: {} for m in ('baseline', 'fogaware')}
    for model in ('baseline', 'fogaware'):
        for level in FOG_LEVELS:
            folder = run_folder(model, level)
            if folder is None:
                print(f"  {model} {level}: no folder found"); continue
            pred_file = folder / "best_predictions.json"
            if not pred_file.exists():
                print(f"  {model} {level}: best_predictions.json missing in {folder.name}"); continue
            tp_conf, tp_iou, n_tp, n_fp = match_condition(pred_file, gt)
            fp_rate = n_fp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
            stats[model][level] = {
                'tp_conf': tp_conf, 'tp_iou': tp_iou, 'n_tp': n_tp, 'n_fp': n_fp,
                'fp_rate': fp_rate,
                'iou_mean': float(np.mean(tp_iou)) if tp_iou else np.nan,
                'iou_std':  float(np.std(tp_iou))  if tp_iou else np.nan,
                'conf_mean': float(np.mean(tp_conf)) if tp_conf else np.nan,
            }
            print(f"  {model:9s} {level:8s} [{folder.name:28s}] "
                  f"TP={n_tp:5d} FP={n_fp:5d} FPrate={fp_rate:.3f} "
                  f"IoU={stats[model][level]['iou_mean']:.3f}")

    # Figure 1: confidence distributions
    fig, axes = plt.subplots(2, len(FOG_LEVELS), figsize=(22, 6), sharex=True, sharey=True)
    for r, model in enumerate(('baseline', 'fogaware')):
        for c, level in enumerate(FOG_LEVELS):
            ax = axes[r, c]; s = stats[model].get(level)
            if s and s['tp_conf']:
                ax.hist(s['tp_conf'], bins=20, range=(0, 1),
                        color='steelblue' if model == 'baseline' else 'crimson', alpha=0.75)
            if r == 0:
                ax.set_title('Clear' if level == 'clear' else f'i={int(level[-2:])}', fontsize=9)
            if c == 0:
                ax.set_ylabel(model.capitalize(), fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=6)
    fig.suptitle(f'True-positive confidence distributions across fog severity '
                 f'(conf>={CONF_THRES}, IoU>={IOU_THRES})', fontsize=13, fontweight='bold')
    fig.supxlabel('Confidence score')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "confidence_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'confidence_distributions.png'}")

    # Figure 2: FP rate
    x = np.arange(len(FOG_LEVELS)); width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    bl = [stats['baseline'].get(l, {}).get('fp_rate', np.nan) for l in FOG_LEVELS]
    fa = [stats['fogaware'].get(l, {}).get('fp_rate', np.nan) for l in FOG_LEVELS]
    ax.bar(x - width/2, bl, width, label='Baseline', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, fa, width, label='Fog-aware', color='crimson', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([l if l == 'clear' else l[-3:] for l in FOG_LEVELS])
    ax.set_xlabel('Fog level'); ax.set_ylabel('False-positive rate  FP/(TP+FP)')
    ax.set_title(f'False-positive rate across fog severity (conf>={CONF_THRES}, IoU>={IOU_THRES})', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fp_rate_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fp_rate_analysis.png'}")

    # Figure 3: IoU stability
    fig, ax = plt.subplots(figsize=(13, 6))
    blm = [stats['baseline'].get(l, {}).get('iou_mean', np.nan) for l in FOG_LEVELS]
    bls = [stats['baseline'].get(l, {}).get('iou_std',  np.nan) for l in FOG_LEVELS]
    fam = [stats['fogaware'].get(l, {}).get('iou_mean', np.nan) for l in FOG_LEVELS]
    fas = [stats['fogaware'].get(l, {}).get('iou_std',  np.nan) for l in FOG_LEVELS]
    ax.errorbar(x - width/2, blm, yerr=bls, fmt='o-', color='steelblue', capsize=4, linewidth=2, label='Baseline')
    ax.errorbar(x + width/2, fam, yerr=fas, fmt='s-', color='crimson', capsize=4, linewidth=2, label='Fog-aware')
    ax.set_xticks(x); ax.set_xticklabels([l if l == 'clear' else l[-3:] for l in FOG_LEVELS])
    ax.set_xlabel('Fog level'); ax.set_ylabel('Mean IoU of true positives'); ax.set_ylim(0, 1)
    ax.set_title(f'IoU stability of surviving detections (conf>={CONF_THRES}, IoU>={IOU_THRES})', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iou_stability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'iou_stability.png'}")

    # CSV summary
    csv_path = CSV_DIR / "failure_mechanisms_summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,level,n_tp,n_fp,fp_rate,iou_mean,iou_std,conf_mean\n")
        for model in ('baseline', 'fogaware'):
            for level in FOG_LEVELS:
                s = stats[model].get(level)
                if not s:
                    continue
                f.write(f"{model},{level},{s['n_tp']},{s['n_fp']},{s['fp_rate']:.4f},"
                        f"{s['iou_mean']:.4f},{s['iou_std']:.4f},{s['conf_mean']:.4f}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()