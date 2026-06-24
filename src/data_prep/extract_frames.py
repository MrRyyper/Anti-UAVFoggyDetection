# coding: utf-8
"""
extract_frames.py
-----------------
Reads every visible*.avi and visible*.mp4 in Anti-UAV-RGBT/
and saves each frame as a numbered .jpg into framecuts/.

Skips sequences that already have frames extracted.

Place in Anti-UAV-RGBT/ and run via submit_extract.sh on Snellius,
or locally with: python extract_frames.py
"""

import os
import glob
import cv2
from multiprocessing import Pool, cpu_count

ROOT_DIR     = "/scratch-shared/glevybirkental/Anti-UAV-RGBT"
FRAMECUT_DIR = "/scratch-shared/glevybirkental/framecuts"


def video2jpg(video_path):
    """Extract frames from one video, skipping if already done."""
    rel        = os.path.relpath(video_path, ROOT_DIR)   # train/seq/visible_fog_i03_beta0.08.avi
    rel_noext  = os.path.splitext(rel)[0]                # train/seq/visible_fog_i03_beta0.08
    save_dir   = os.path.join(FRAMECUT_DIR, rel_noext)   # framecuts/train/seq/visible_fog_i03_beta0.08

    # skip if already extracted (check for at least one frame)
    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"  Skipping (already extracted): {rel}")
        return

    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"Extracting: {rel}")
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(save_dir, f"{frame_count:06d}.jpg"), frame)
        frame_count += 1

    cap.release()
    print(f"  -> {frame_count} frames saved: {rel_noext}")


def get_videos(root, pattern):
    """Find all visible*.mp4 and visible*.avi, skipping infrared."""
    videos = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    matched = sorted(v for v in videos if os.path.basename(v).startswith(pattern))
    return matched


def main():
    # -- What to extract ---------------------------------------------------
    # "visible" catches everything: clear mp4, all fog avi, all A variants
    # Already-extracted directories are skipped automatically
    videos = get_videos(ROOT_DIR, "visible")
    print(f"Found {len(videos)} visible videos")

    # show breakdown
    clear = [v for v in videos if "fog" not in os.path.basename(v)]
    fog   = [v for v in videos if "fog"     in os.path.basename(v)]
    print(f"  Clear  : {len(clear)}")
    print(f"  Fog    : {len(fog)}")
    print(f"  Workers: {cpu_count()}\n")

    with Pool(cpu_count()) as pool:
        pool.map(video2jpg, videos)

    print("\nAll done.")


if __name__ == "__main__":
    main()