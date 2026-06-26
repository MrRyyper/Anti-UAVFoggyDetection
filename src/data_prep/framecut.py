"""Extract every Nth frame of each visible*.avi / visible*.mp4 into framecuts/.

Already-extracted sequences are skipped. Sampling step matches the dataset (N=5).
"""

import os
import glob
import cv2
from multiprocessing import Pool, cpu_count

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_ROOT      = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
FRAMECUT_ROOT = os.path.join(REPO_ROOT, "framecuts")
FRAME_STEP    = 5


def video2jpg(video_path):
    rel       = os.path.relpath(video_path, RAW_ROOT)
    rel_noext = os.path.splitext(rel)[0]
    save_dir  = os.path.join(FRAMECUT_ROOT, rel_noext)

    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
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
    print(f"{rel_noext}: {saved_count} frames")


def get_videos(root):
    videos = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(v for v in videos if os.path.basename(v).startswith("visible"))


def main():
    videos = get_videos(RAW_ROOT)
    print(f"Found {len(videos)} visible videos (step {FRAME_STEP})")
    with Pool(cpu_count()) as pool:
        pool.map(video2jpg, videos)


if __name__ == "__main__":
    main()
