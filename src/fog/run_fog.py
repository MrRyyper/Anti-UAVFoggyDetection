"""Entrypoint for synthetic_fog_torch: generate the ten standard fog levels at
A=0.5 plus the A-sensitivity variants, named as the configs expect
(visible_fog_i05_beta0.10.avi, visible_fog_i05_beta0.10_A0.30.avi, ...)."""

import os

from synthetic_fog_torch import fog_video   # run from inside src/fog/

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_ROOT  = os.path.join(REPO_ROOT, "Anti-UAV-RGBT")
A_MAIN    = 0.5
SPLITS    = ["train", "val", "test"]   # train builds the fog-aware set; val/test for eval
A_SWEEP   = [(i, A) for i in (2, 5, 9) for A in (0.30, 0.70)]


def beta_for(i):
    return 0.01 * i + 0.05


def visibles(root):
    for split in SPLITS:
        for r, _, files in os.walk(os.path.join(root, split)):
            if "visible.mp4" in files:
                yield os.path.join(r, "visible.mp4")


def gen(in_vid, i, A, suffix=""):
    beta = beta_for(i)
    out = os.path.join(os.path.dirname(in_vid),
                       f"visible_fog_i{i:02d}_beta{beta:.2f}{suffix}.avi")
    if not os.path.exists(out):
        fog_video(in_vid, out, beta=beta, A=A)


if __name__ == "__main__":
    for in_vid in visibles(RAW_ROOT):
        for i in range(10):                 # 10 standard levels, A=0.5
            gen(in_vid, i, A_MAIN)
        for i, A in A_SWEEP:                 # A-sensitivity variants
            gen(in_vid, i, A, suffix=f"_A{A:.2f}")
