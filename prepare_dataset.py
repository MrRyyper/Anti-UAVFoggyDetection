"""
Generates YOLOv5 labels from pre-extracted frames in Anti-UAV-RGBT/framecuts/.
Run framecut.py first to extract frames, then run this script.

Reads frames from:  Anti-UAV-RGBT/framecuts/{split}/{sequence}/{variant}/
Writes images to:   prepared_dataset/{variant}/images/{split}/
Writes labels to:   prepared_dataset/{variant}/labels/{split}/

Each variant (visible, visible_fog_beta0.05, etc.) gets its own subfolder,
so you can point a YAML at just the baseline or any fog variant independently.

Label format: 0 cx cy w h  (normalised, class 0 = uav)
"""

import os
import glob
import json
import shutil
import cv2

RAW_ROOT      = "Anti-UAV-RGBT"
FRAMECUT_ROOT = os.path.join(RAW_ROOT, "framecuts")
OUT_ROOT      = "prepared_dataset"
SPLITS        = ["train", "val", "test"]


def process_variant(frames_dir, ann, img_dir, lbl_dir, seq_name, variant):
    exist   = ann["exist"]
    gt_rect = ann["gt_rect"]

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        print(f"  [skip] no frames in {frames_dir}")
        return 0

    first = cv2.imread(frames[0])
    img_h, img_w = first.shape[:2]

    for frame_path in frames:
        frame_idx = int(os.path.splitext(os.path.basename(frame_path))[0])
        stem      = f"{seq_name}__{frame_idx:06d}"

        # Skip if image and label already exist
        if os.path.exists(os.path.join(img_dir, stem + ".jpg")) and \
           os.path.exists(os.path.join(lbl_dir, stem + ".txt")):
            continue

        shutil.copy2(frame_path, os.path.join(img_dir, stem + ".jpg"))

        lbl_file = os.path.join(lbl_dir, stem + ".txt")
        if frame_idx < len(exist) and exist[frame_idx] == 1:
            x, y, w, h = gt_rect[frame_idx]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            with open(lbl_file, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            open(lbl_file, "w").close()

    return len(frames)


def process_sequence(seq_framecut_dir, seq_raw_dir, split, seq_name):
    ann_path = os.path.join(seq_raw_dir, "visible.json")
    if not os.path.exists(ann_path):
        print(f"  [skip] no visible.json for {seq_name}")
        return

    with open(ann_path) as f:
        ann = json.load(f)

    for variant_dir in sorted(os.listdir(seq_framecut_dir)):
        frames_dir = os.path.join(seq_framecut_dir, variant_dir)
        if not os.path.isdir(frames_dir):
            continue

        img_dir = os.path.join(OUT_ROOT, variant_dir, "images", split)
        lbl_dir = os.path.join(OUT_ROOT, variant_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n = process_variant(frames_dir, ann, img_dir, lbl_dir, seq_name, variant_dir)
        print(f"  {seq_name}  {variant_dir:<25}: {n} frames")


def main():
    for split in SPLITS:
        fc_split_dir  = os.path.join(FRAMECUT_ROOT, split)
        raw_split_dir = os.path.join(RAW_ROOT, split)

        if not os.path.isdir(fc_split_dir):
            print(f"[warn] {fc_split_dir} not found — run framecut.py first")
            continue

        print(f"\n=== {split} ===")
        for seq_name in sorted(os.listdir(fc_split_dir)):
            seq_fc_dir  = os.path.join(fc_split_dir, seq_name)
            seq_raw_dir = os.path.join(raw_split_dir, seq_name)
            if os.path.isdir(seq_fc_dir):
                process_sequence(seq_fc_dir, seq_raw_dir, split, seq_name)


if __name__ == "__main__":
    main()
