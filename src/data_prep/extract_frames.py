"""Extract every frame of each visible*.avi / visible*.mp4 into framecuts/.

Already-extracted sequences are skipped.
"""

import os
import glob
import cv2
from multiprocessing import Pool, cpu_count

REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_DIR     = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
FRAMECUT_DIR = os.path.join(REPO_ROOT, "framecuts")


def video2jpg(video_path):
    rel        = os.path.relpath(video_path, ROOT_DIR)
    rel_noext  = os.path.splitext(rel)[0]
    save_dir   = os.path.join(FRAMECUT_DIR, rel_noext)

    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(save_dir, f"{frame_count:06d}.jpg"), frame)
        frame_count += 1

    cap.release()
    print(f"{rel_noext}: {frame_count} frames")


def get_videos(root, pattern):
    videos = []
    for ext in ("*.mp4", "*.avi"):
        videos.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(v for v in videos if os.path.basename(v).startswith(pattern))


def main():
    videos = get_videos(ROOT_DIR, "visible")
    print(f"Found {len(videos)} visible videos")
    with Pool(cpu_count()) as pool:
        pool.map(video2jpg, videos)


if __name__ == "__main__":
    main()
