# coding: utf-8
import os
import random
import shutil
from pathlib import Path

BASE   = Path("/scratch-shared/glevybirkental/prepared_dataset")
OUTPUT = Path("/scratch-shared/glevybirkental/prepared_dataset/fog_aware_all")
SEED   = 42
random.seed(SEED)

CLEAR_TRAIN = BASE / "visible/images/train"

# All 10 fog levels
FOG_LEVELS = []
for i in range(10):
    beta = 0.01 * i + 0.05
    FOG_LEVELS.append((i, BASE / f"visible_fog_i{i:02d}_beta{beta:.2f}/images/train"))

# Get all clear training images
clear_images = list(CLEAR_TRAIN.glob("*.jpg"))
n_clear = len(clear_images)
print(f"Clear images: {n_clear}")

# Sample equal fog images per level (total fog = total clear)
n_per_level = n_clear // len(FOG_LEVELS)
fog_images = []
for i, fog_dir in FOG_LEVELS:
    if not fog_dir.exists():
        print(f"WARNING: {fog_dir} not found, skipping")
        continue
    level_images = list(fog_dir.glob("*.jpg"))
    sampled = random.sample(level_images, min(n_per_level, len(level_images)))
    fog_images.extend([(i, p) for p in sampled])
    print(f"i={i:02d}: sampled {len(sampled)}")

print(f"Total fog images   : {len(fog_images)}")
print(f"Total train images : {n_clear + len(fog_images)}")

# Create output directories
for split in ["images/train", "labels/train"]:
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

# Copy clear images and labels
for img_path in clear_images:
    dst = OUTPUT / "images/train" / img_path.name
    if not dst.exists():
        shutil.copy(img_path, dst)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
    lbl_dst = OUTPUT / "labels/train" / (img_path.stem + ".txt")
    if label_path.exists() and not lbl_dst.exists():
        shutil.copy(label_path, lbl_dst)

print("Clear images copied.")

# Copy fog images with prefix to avoid name collision
for fog_level, img_path in fog_images:
    new_name = f"fog_i{fog_level:02d}__{img_path.name}"
    dst = OUTPUT / "images/train" / new_name
    if not dst.exists():
        shutil.copy(img_path, dst)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
    lbl_dst = OUTPUT / "labels/train" / f"fog_i{fog_level:02d}__{img_path.stem}.txt"
    if label_path.exists() and not lbl_dst.exists():
        shutil.copy(label_path, lbl_dst)

print("Fog images copied.")

final_count = len(list((OUTPUT / "images/train").glob("*.jpg")))
print(f"Final training image count: {final_count}")
print("Done.")