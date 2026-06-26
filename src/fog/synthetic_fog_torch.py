import os
import cv2
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dist_cache(H, W):
    """Distance-from-centre grid, computed once per video resolution."""
    y = torch.arange(H, device=device).float().unsqueeze(1)
    x = torch.arange(W, device=device).float().unsqueeze(0)
    cy, cx = H / 2.0, W / 2.0
    return torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)


def add_fog_gpu(img_bgr, beta, A, dist_cache):
    """GPU fog. I_fog = I*t + A*(1-t), t = exp(-beta*d)."""
    img = torch.from_numpy(img_bgr).to(device).float() / 255.0  # (H, W, 3)

    H, W = img_bgr.shape[:2]
    d = -0.04 * dist_cache + float(np.sqrt(max(H, W)))
    t = torch.exp(-beta * d).clamp(0.0, 1.0).unsqueeze(-1)  # (H, W, 1)

    fog = img * t + A * (1.0 - t)
    return fog.clamp(0, 1).mul(255).byte().cpu().numpy()


def fog_video(in_path, out_path, beta, A):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write video: {out_path}")

    dist_cache = build_dist_cache(h, w)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(add_fog_gpu(frame, beta=beta, A=A, dist_cache=dist_cache))
        n += 1

    cap.release()
    writer.release()
    print(f"Saved: {out_path} ({n} frames)")


def process_split(split_dir, beta, A, visible_name="visible.mp4", fog_level=None):
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Split dir not found: {split_dir}")

    count = 0
    for root, _, files in os.walk(split_dir):
        if visible_name not in files:
            continue
        in_vid = os.path.join(root, visible_name)

        if fog_level is None:
            out_vid = os.path.join(root, f"visible_fog_beta{beta:.2f}.avi")
        else:
            out_vid = os.path.join(root, f"visible_fog_i{fog_level:02d}_beta{beta:.2f}.avi")

        if os.path.exists(out_vid):
            continue

        fog_video(in_vid, out_vid, beta=beta, A=A)
        count += 1

    print(f"Processed {count} video(s) in {split_dir}")
