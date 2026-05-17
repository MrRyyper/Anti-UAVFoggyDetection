import os
import random
import shutil
from pathlib import Path

# Paths
BASE = Path("prepared_dataset")
OUTPUT = Path("prepared_dataset/fog_aware")
CLEAR_TRAIN = BASE / "visible/images/train"
FOG_LEVELS = [BASE / f"visible_fog_i0{i}_beta0.{b}/images/train" for i, b in zip([2, 5, 9], ["07", "10", "14"])]
SEED = 42

random.seed(SEED)

# Get all clear training images
clear_images = list(CLEAR_TRAIN.glob("*.jpg"))
n_clear = len(clear_images)
print(f"Clear images: {n_clear}")

# Sample equal fog images per level (total fog = total clear)
n_per_level = n_clear // 3
fog_images = []
for fog_dir in FOG_LEVELS:
    level_images = list(fog_dir.glob("*.jpg"))
    sampled = random.sample(level_images, min(n_per_level, len(level_images)))
    fog_images.extend(sampled)
    print(f"{fog_dir.name}: sampled {len(sampled)}")

print(f"Total fog images: {len(fog_images)}")
print(f"Total training images: {n_clear + len(fog_images)}")

# Create output directories
for split in ["images/train", "labels/train"]:
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

# Copy clear images and labels
for img_path in clear_images:
    shutil.copy(img_path, OUTPUT / "images/train" / img_path.name)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
    if label_path.exists():
        shutil.copy(label_path, OUTPUT / "labels/train" / (img_path.stem + ".txt"))

# Copy fog images and labels
for img_path in fog_images:
    shutil.copy(img_path, OUTPUT / "images/train" / img_path.name)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
    if label_path.exists():
        shutil.copy(label_path, OUTPUT / "labels/train" / (img_path.stem + ".txt"))

print("Done — fog_aware training set created.")