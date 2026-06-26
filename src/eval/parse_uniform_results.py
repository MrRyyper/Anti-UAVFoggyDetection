#!/usr/bin/env python3
"""
parse_uniform_results.py (fixed)
---------------------------------
Extracts all ten 'all' rows sequentially from the single uniform-fog
val.py stderr log and writes them to uniform_fog_results.csv.

All ten val.py runs landed in one .err file (sequential SLURM job),
so we read ALL matching 'all' rows in order and pair them with i=0..9.

Usage:
    python parse_uniform_results.py
    python parse_uniform_results.py --logdir logs --out uniform_fog_results.csv
"""

import os
import re
import csv
import glob
import argparse

BETAS = {i: round(0.01 * i + 0.05, 2) for i in range(10)}


def find_uniform_log(logdir):
    """Find the .err log containing uniform_fog results."""
    pattern = os.path.join(logdir, "*.err")
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f) as fh:
                content = fh.read()
            if "foguniform_i" in content:
                return f
        except Exception:
            continue
    return None


def parse_all_rows(log_path):
    """
    Extract ALL 'all' rows from the log in order.
    Returns a list of dicts, one per fog level (i=0..9).
    """
    all_pattern = re.compile(
        r"^\s*all\s+"
        r"[\d]+\s+"        # images
        r"[\d]+\s+"        # instances
        r"([\d.]+)\s+"     # P
        r"([\d.]+)\s+"     # R
        r"([\d.]+)\s+"     # mAP@0.5
        r"([\d.]+)"        # mAP@0.5:0.95
    )
    rows = []
    with open(log_path) as f:
        for line in f:
            m = all_pattern.match(line)
            if m:
                P, R, map50, map5095 = [float(x) for x in m.groups()]
                F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                rows.append({
                    "P":              round(P, 4),
                    "R":              round(R, 4),
                    "F1":             round(F1, 4),
                    "mAP@0.5":        round(map50, 4),
                    "mAP@0.5:0.95":   round(map5095, 4),
                })
    return rows


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    default_out = os.path.join(
        REPO_ROOT, "results", "metrics", "uniform_fog_results.csv")
    default_logdir = os.path.join(REPO_ROOT, "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=default_logdir)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    logdir = os.path.expanduser(args.logdir)
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    log_path = find_uniform_log(logdir)

    if log_path is None:
        print(f"ERROR: no uniform-fog log found in {logdir}")
        return

    print(f"Found log: {log_path}\n")
    metrics_list = parse_all_rows(log_path)

    if len(metrics_list) != 10:
        print(f"WARNING: expected 10 'all' rows, found {len(metrics_list)}.")
        print("Check the log for incomplete runs.")

    fieldnames = ["fog_level", "beta", "P", "R", "F1",
                  "mAP@0.5", "mAP@0.5:0.95", "log_file"]
    rows = []

    for i in range(10):
        beta = BETAS[i]
        if i < len(metrics_list):
            m = metrics_list[i]
            print(f"  i={i:02d} (beta={beta:.2f}):  "
                  f"P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}  "
                  f"mAP50={m['mAP@0.5']:.4f}  mAP5095={m['mAP@0.5:0.95']:.4f}")
            rows.append({
                "fog_level": i, "beta": beta,
                **m,
                "log_file": os.path.basename(log_path)
            })
        else:
            print(f"  i={i:02d} (beta={beta:.2f}):  MISSING")
            rows.append({
                "fog_level": i, "beta": beta,
                "P": "", "R": "", "F1": "",
                "mAP@0.5": "", "mAP@0.5:0.95": "",
                "log_file": "MISSING"
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()