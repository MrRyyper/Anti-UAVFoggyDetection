import os
import cv2

DATASET_ROOT = r"C:\Users\Gur Levy\Desktop\UVA\MASTER\THESIS\thesis-1\dataset\Anti-UAV-RGBT"
SPLIT = "train"  # change to val/test if needed
VISIBLE_NAME = "visible.mp4"

def check_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, "cannot_open"

    ok, frame = cap.read()
    cap.release()

    if (not ok) or (frame is None):
        return False, "cannot_decode_first_frame"

    return True, "ok"

def main():
    split_dir = os.path.join(DATASET_ROOT, SPLIT)
    fails = []
    total = 0

    for root, _, files in os.walk(split_dir):
        if VISIBLE_NAME in files:
            total += 1
            vid = os.path.join(root, VISIBLE_NAME)
            ok, reason = check_video(vid)
            if not ok:
                fails.append((vid, reason))
                print("FAIL:", reason, vid)

    print("\nTotal videos:", total)
    print("Failed videos:", len(fails))

    # Save report
    report_path = os.path.join(DATASET_ROOT, f"decode_failures_{SPLIT}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for vid, reason in fails:
            f.write(f"{reason}\t{vid}\n")

    print("Report saved to:", report_path)

if __name__ == "__main__":
    main()