# coding: utf-8
"""
extract_results.py
------------------
Parses YOLOv5 val.py log files and extracts results into a clean CSV.

The runs appear in the log in the same order as the inference script.
This script maps them accordingly.

Place in ~/thesis-1/ and run:
    python extract_results.py
"""

import re
import csv
from pathlib import Path

LOGS_DIR = Path.home() / "logs"
OUT_DIR  = Path.home() / "thesis-1" / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order must match submit_inference_baseline.sh
BASELINE_ORDER = [
    ("baseline", "clear",    0,   0.5, 640),
    ("baseline", "fog",      0,   0.5, 640),
    ("baseline", "fog",      1,   0.5, 640),
    ("baseline", "fog",      2,   0.5, 640),
    ("baseline", "fog",      3,   0.5, 640),
    ("baseline", "fog",      4,   0.5, 640),
    ("baseline", "fog",      5,   0.5, 640),
    ("baseline", "fog",      6,   0.5, 640),
    ("baseline", "fog",      7,   0.5, 640),
    ("baseline", "fog",      8,   0.5, 640),
    ("baseline", "fog",      9,   0.5, 640),
    ("baseline", "fog",      2,   0.3, 640),
    ("baseline", "fog",      5,   0.3, 640),
    ("baseline", "fog",      9,   0.3, 640),
    ("baseline", "fog",      2,   0.7, 640),
    ("baseline", "fog",      5,   0.7, 640),
    ("baseline", "fog",      9,   0.7, 640),
    ("baseline", "clear",    0,   0.5, 1280),
    ("baseline", "fog",      2,   0.5, 1280),
    ("baseline", "fog",      5,   0.5, 1280),
    ("baseline", "fog",      9,   0.5, 1280),
]

# Order must match submit_inference_fogaware.sh
FOGAWARE_ORDER = [
    ("fogaware", "clear",    0,   0.5, 640),
    ("fogaware", "fog",      0,   0.5, 640),
    ("fogaware", "fog",      1,   0.5, 640),
    ("fogaware", "fog",      2,   0.5, 640),
    ("fogaware", "fog",      3,   0.5, 640),
    ("fogaware", "fog",      4,   0.5, 640),
    ("fogaware", "fog",      5,   0.5, 640),
    ("fogaware", "fog",      6,   0.5, 640),
    ("fogaware", "fog",      7,   0.5, 640),
    ("fogaware", "fog",      8,   0.5, 640),
    ("fogaware", "fog",      9,   0.5, 640),
    ("fogaware", "fog",      2,   0.3, 640),
    ("fogaware", "fog",      5,   0.3, 640),
    ("fogaware", "fog",      9,   0.3, 640),
    ("fogaware", "fog",      2,   0.7, 640),
    ("fogaware", "fog",      5,   0.7, 640),
    ("fogaware", "fog",      9,   0.7, 640),
]

HEADERS = ["model", "condition", "fog_level", "A", "imgsz",
           "images", "labels", "P", "R", "mAP50", "mAP5095"]

PATTERN = re.compile(
    r'\s+all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
)


def parse_log(log_path):
    results = []
    # try .err first since YOLOv5 prints to stderr
    for suffix in ['.err', '.out']:
        p = log_path.with_suffix(suffix)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    results.append([
                        int(m.group(1)),   # images
                        int(m.group(2)),   # labels
                        float(m.group(3)), # P
                        float(m.group(4)), # R
                        float(m.group(5)), # mAP50
                        float(m.group(6)), # mAP5095
                    ])
        if results:
            break
    return results


def process(log_stem, order, out_csv):
    log_files = sorted(LOGS_DIR.glob(f"{log_stem}*.out"))
    if not log_files:
        print(f"No log found for {log_stem}")
        return

    log_file = log_files[0]
    print(f"Parsing: {log_file.name}")
    metrics = parse_log(log_file)

    if len(metrics) != len(order):
        print(f"  WARNING: expected {len(order)} results, got {len(metrics)}")

    rows = []
    for i, (model, cond, fog_i, A, imgsz) in enumerate(order):
        if i < len(metrics):
            row = [model, cond, fog_i, A, imgsz] + metrics[i]
        else:
            row = [model, cond, fog_i, A, imgsz, "", "", "", "", "", ""]
        rows.append(row)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows to {out_csv}")
    return rows


def main():
    all_rows = []

    b_rows = process(
        "inference_baseline",
        BASELINE_ORDER,
        OUT_DIR / "results_baseline.csv"
    )
    if b_rows:
        all_rows.extend(b_rows)

    f_rows = process(
        "inference_fogaware",
        FOGAWARE_ORDER,
        OUT_DIR / "results_fogaware.csv"
    )
    if f_rows:
        all_rows.extend(f_rows)

    # combined
    if all_rows:
        with open(OUT_DIR / "results_all.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            writer.writerows(all_rows)
        print(f"\nCombined CSV: {OUT_DIR / 'results_all.csv'}")


if __name__ == "__main__":
    main()