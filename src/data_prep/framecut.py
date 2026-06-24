# coding: utf-8
"""
framecut.py
-----------
Extracts frames from all visible*.mp4 and visible*.avi in Anti-UAV-RGBT/
and saves them as numbered .jpg files into framecuts/.

Skips sequences that already have frames extracted.
Samples every Nth frame (default N=5, matching existing dataset).

Usage on Snellius:
    python framecut.py
or via submit_prepare.sh
"""

import os
import glob
import cv2
from multiprocessing import Pool, cpu_count

RAW_ROOT      = "/gpfs/home4/glevybirkental/thesis-1/Anti-UAV-RGBT"
FRAMECUT_ROOT = "/scratch-shared/glevybirkental/framecuts"
FRAME_STEP    = 5   # sample every 5th frame, matching existing dataset


def video2jpg(video_path):
    """Extract every Nth frame, skip if already done."""
    rel       = os.path.relpath(video_path, RAW_ROOT)  # train/1/visible_fog_i03_beta0.08.avi
    rel_noext = os.path.splitext(rel)[0]               # train/1/visible_fog_i03_beta0.08
    save_dir  = os.path.join(FRAMECUT_ROOT, rel_noext) # framecuts/train/1/visible_fog_i03_beta0.08

    # skip if already extracted
    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"  Skipping (exists): {rel}")
        return

    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx   = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % FRAME_STEP == 0:
            cv2.imwrite(os.path.join(save_dir, f"{frame_idx:06d}.jpg"), frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted {saved_count} frames: {rel_noext}")


def get_videos(root):
    """Find all visible*.mp4 and visible*.avi, skipping infrared."""
    videos = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(v for v in videos if os.path.basename(v).startswith("visible"))


def main():
    videos = get_videos(RAW_ROOT)

    clear = [v for v in videos if "fog" not in os.path.basename(v)]
    fog   = [v for v in videos if "fog"     in os.path.basename(v)]

    print(f"Found {len(videos)} visible videos")
    print(f"  Clear     : {len(clear)}")
    print(f"  Fog       : {len(fog)}")
    print(f"  Frame step: every {FRAME_STEP} frames")
    print(f"  Workers   : {cpu_count()}\n")

    with Pool(cpu_count()) as pool:
        pool.map(video2jpg, videos)

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()