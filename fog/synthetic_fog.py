import os 
import cv2
import numpy as np

def add_fog_bgr(img_bgr, beta, A):
     
     """
        Takes one frame and return foggy version
        Fog model: I_fog = I*t + A*(1-t), with t=exp(-beta*d) **
        img_bgr= frame in OpenCV format
        beta= fog density
        A= how bright/grey fog veil
     """
     img = img_bgr.astype(np.float32) / 255.0 # convert frame to float 0-1, easier to do math in normalized floats
     H,W = img.shape[:2] # height/width of image
    
    # create a grid where every pixel knows dist from vcenter of image
     y = np.arange(H,dtype=np.float32)[:,None]
     x= np.arange(W,dtype=np.float32)[None,:]
     cy,cx = H / 2.0, W/2.0
     dist = np.sqrt((y-cy)**2 + (x-cx)**2)

     # compute fake depth
     size = np.sqrt(max(H,W))
     d = -0.04 * dist + size # value goes down towards corners & shifts everything positive
     # d -> bigger = heavier fog effect

     # compute transmission -> how much from og image survives through the fog
     t= np.exp(-beta * d)
     t = np.clip(t,0.0,1.0)
     # t smaller when fog is strong -> image gets washed out
     # beta controls strenght of fog

     # mix frame with fog veil
     # foggy_pixel = og_pixel * t + fog_color * (1-t)
     # t = 1 -> og image / t = 0 -> fog veil
     # t[...] apply to all three RGB
     fog = img * t[..., None] + A * (1.0 - t[..., None])
     return np.clip(fog*255.0,0,255).astype(np.uint8)

def fog_video(in_path, out_path, beta, A):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v is usually fine; if writer issues, switch to XVID + .avi
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    #if not writer.isOpened():
     #   cap.release()
      #  raise RuntimeError(f"Cannot write video: {out_path}")
    #out_path = os.path.splitext(out_path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Writer failed to open")  

    n = 0
    while True:
        ok, frame = cap.read()

        # HARD GUARD: never process invalid frames
        if not ok or frame is None:
            print(f"Stopped reading {os.path.basename(in_path)} at frame {n}. ok={ok}, frame_is_none={frame is None}")
            break

        foggy = add_fog_bgr(frame, beta=beta, A=A)
        writer.write(foggy)
        n += 1

        if n % 300 == 0:
            print(f"  {os.path.basename(in_path)}: processed {n} frames...")

    cap.release()
    writer.release()

    if n == 0:
        # This should basically never happen now, but if it does, we’ll know.
        print("WARNING: wrote 0 frames for", in_path)
    else:
        print(f"Saved: {out_path} ({n} frames)")

def process_split(split_dir, beta, A, visible_name="visible.mp4", fog_level=None):
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Split dir not found: {split_dir}")

    print("Processing split:", split_dir)
    count = 0

    for root, _, files in os.walk(split_dir):
        if visible_name in files:
            in_vid = os.path.join(root, visible_name)

            if fog_level is None:
                out_vid = os.path.join(root, f"visible_fog_beta{beta:.2f}.avi")
            else:
                out_vid = os.path.join(root, f"visible_fog_i{fog_level:02d}_beta{beta:.2f}.avi")

            if os.path.exists(out_vid):
                print("Skipping (already exists):", out_vid)
                continue

            print("Fogging:", in_vid)
            fog_video(in_vid, out_vid, beta=beta, A=A)
            count += 1

    print(f"Done. Processed {count} videos in {split_dir}")

def main():
     dataset_root = "C:\\Users\\Gur Levy\\Desktop\\UVA\\MASTER\\THESIS\\thesis-1\\dataset\\Anti-UAV-RGBT"
     #visible_name = "visible.mp4"
     
     A = 0.5
     
     split = "train" # change to test or val
     split_dir = os.path.join(dataset_root,split)
     for i in [2,5,9]:  # i = 0..9
        beta = 0.01 * i + 0.05
        print(f"Generating fog level {i} with beta={beta:.3f}")

        process_split(split_dir, visible_name="visible.mp4", beta=beta ,A=A)

if __name__  == "__main__":
    main()