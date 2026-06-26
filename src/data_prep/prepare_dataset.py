"""Organise extracted frames into a YOLOv5-ready dataset.

framecuts/{split}/{sequence}/{variant}/  ->  prepared_dataset/{variant}/{images,labels}/{split}/

Each fog variant gets its own subfolder. Labels are identical across variants
(fog does not move the bounding boxes). Existing files are skipped.
"""

import os
import glob
import json
import shutil
import cv2

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_ROOT      = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
FRAMECUT_ROOT = os.path.join(REPO_ROOT, "framecuts")
OUT_ROOT      = os.path.join(REPO_ROOT, "prepared_dataset")
SPLITS        = ["train", "val", "test"]


def process_variant(frames_dir, ann, img_dir, lbl_dir, seq_name):
    exist   = ann["exist"]
    gt_rect = ann["gt_rect"]

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        return 0

    first = cv2.imread(frames[0])
    if first is None:
        return 0
    img_h, img_w = first.shape[:2]

    count = 0
    for frame_path in frames:
        frame_idx = int(os.path.splitext(os.path.basename(frame_path))[0])
        stem      = f"{seq_name}__{frame_idx:06d}"

        img_out = os.path.join(img_dir, stem + ".jpg")
        lbl_out = os.path.join(lbl_dir, stem + ".txt")

        if os.path.exists(img_out) and os.path.exists(lbl_out):
            continue

        shutil.copy2(frame_path, img_out)

        if frame_idx < len(exist) and exist[frame_idx] == 1:
            x, y, w, h = gt_rect[frame_idx]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            with open(lbl_out, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            open(lbl_out, "w").close()

        count += 1

    return count


def process_sequence(seq_framecut_dir, seq_raw_dir, split, seq_name):
    ann_path = os.path.join(seq_raw_dir, "visible.json")
    if not os.path.exists(ann_path):
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

        n = process_variant(frames_dir, ann, img_dir, lbl_dir, seq_name)
        if n > 0:
            print(f"  {seq_name}  {variant_dir:<35}: {n} frames")


def main():
    for split in SPLITS:
        fc_split_dir  = os.path.join(FRAMECUT_ROOT, split)
        raw_split_dir = os.path.join(RAW_ROOT, split)
        if not os.path.isdir(fc_split_dir):
            continue

        print(f"=== {split} ===")
        for seq_name in sorted(os.listdir(fc_split_dir)):
            seq_fc_dir  = os.path.join(fc_split_dir, seq_name)
            seq_raw_dir = os.path.join(raw_split_dir, seq_name)
            if os.path.isdir(seq_fc_dir):
                process_sequence(seq_fc_dir, seq_raw_dir, split, seq_name)


if __name__ == "__main__":
    main()
