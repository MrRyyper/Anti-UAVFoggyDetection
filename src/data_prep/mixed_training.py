import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE   = REPO_ROOT / "prepared_dataset"
OUTPUT = BASE / "fog_aware"
SEED   = 42
random.seed(SEED)

CLEAR_TRAIN = BASE / "visible/images/train"

FOG_LEVELS = []
for i in range(10):
    beta = 0.01 * i + 0.05
    FOG_LEVELS.append(BASE / f"visible_fog_i{i:02d}_beta{beta:.2f}/images/train")

clear_images = list(CLEAR_TRAIN.glob("*.jpg"))
n_clear = len(clear_images)
print(f"Clear images: {n_clear}")

# sample equal fog images per level so total fog ~= total clear
n_per_level = n_clear // len(FOG_LEVELS)
fog_images = []
for fog_dir in FOG_LEVELS:
    if not fog_dir.exists():
        print(f"WARNING: {fog_dir} not found, skipping")
        continue
    level_images = list(fog_dir.glob("*.jpg"))
    sampled = random.sample(level_images, min(n_per_level, len(level_images)))
    fog_images.extend(sampled)
    print(f"{fog_dir.parent.parent.name}: sampled {len(sampled)}")

print(f"Total fog images   : {len(fog_images)}")
print(f"Total train images : {n_clear + len(fog_images)}")

for split in ["images/train", "labels/train"]:
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

for img_path in clear_images + fog_images:
    shutil.copy(img_path, OUTPUT / "images/train" / img_path.name)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
    if label_path.exists():
        shutil.copy(label_path, OUTPUT / "labels/train" / (img_path.stem + ".txt"))

print("Done - fog_aware training set created.")
