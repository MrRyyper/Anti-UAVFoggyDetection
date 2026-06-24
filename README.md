# Fog-Robust UAV Detection

Master's thesis project investigating whether **fog-aware training** (mixing clear
and synthetic-fog imagery) improves YOLOv5 robustness to fog, evaluated on the
Anti-UAV300 (RGB / visible modality) dataset.

The pipeline: extract frames from the videos → synthesise fog at 10 severity
levels → build YOLOv5 datasets → train a baseline (clear-only) and a fog-aware
(mixed) model → evaluate across fog levels with aggregate, size-stratified, and
failure-mechanism analyses.

## Repository layout

```
thesis-1/
├── configs/                 # YOLOv5 data configs (one .yaml per evaluation condition)
│   ├── anti_uav.yaml              # baseline clear test set
│   ├── fog_aware.yaml             # mixed training set (clear + all fog levels)
│   ├── anti_uav_fog_i00..i09.yaml # individual fog severity levels
│   └── anti_uav_fog_i{02,05,09}_A{0.30,0.70}.yaml  # atmospheric-light sensitivity
│
├── src/
│   ├── data_prep/           # frame extraction & dataset assembly
│   │   ├── extract_frames.py        # full-frame extraction (cluster paths)
│   │   ├── framecut.py              # every-Nth-frame extraction
│   │   ├── prepare_dataset.py       # frames -> YOLOv5 images/labels structure
│   │   ├── create_fogaware_dataset.py  # build fog_aware_all training set
│   │   └── mixed_training.py        # build fog_aware training set
│   ├── fog/                 # synthetic fog (Koschmieder model)
│   │   ├── synthetic_fog.py         # CPU / OpenCV
│   │   └── synthetic_fog_torch.py   # GPU / PyTorch
│   ├── eval/                # evaluation & metrics
│   │   ├── extract_results.py       # parse val.py logs -> results CSVs
│   │   ├── size_stratified_eval.py  # recall by UAV size category
│   │   ├── size_stratified_metrics.py  # P/R/F1/mAP by size category
│   │   └── failure_mechanisms.py    # confidence / FP-rate / IoU analyses
│   └── analysis/            # plotting
│       ├── degradation_line.py      # mAP & recall vs fog severity
│       ├── plot_sensitivity.py      # atmospheric-light sensitivity grid
│       └── radial/                  # spatial / radial-prior EDA
│
├── notebooks/               # EDA.ipynb, final_EDA.ipynb, yolov5.ipynb
│
├── results/
│   ├── metrics/             # all output CSVs (three_seed_*, size_metrics_*, ...)
│   └── figures/             # all output PNG/PDF figures (+ pr_curves/, radial/)
│
├── slurm/                   # HPC (Snellius) job scripts
├── docs/                    # experiment_diagram.html, final_EDA.html, style.css
│
├── Anti-UAV-RGBT/           # dataset mount (gitignored: train/ val/ test/ framecuts/ label_new/)
├── yolo/yolov5/             # YOLOv5 checkout (gitignored)
└── runs/                    # training/val outputs (gitignored)
```

## Paths: local vs cluster

Most scripts were written to run on the **Snellius HPC cluster** and contain
absolute cluster paths that are *not* repo-relative — these are intentionally
left as-is:

- **Dataset / weights** live on the cluster, e.g.
  `/scratch-shared/glevybirkental/prepared_dataset/...`,
  `~/thesis-1/yolo/yolov5/runs/train/<run>/weights/best.pt`.
- **Outputs** (figures and CSVs) are written under `results/figures/` and
  `results/metrics/` (relative to the repo root, `~/thesis-1` on the cluster).

Run analysis/plotting scripts **from the repo root** so the `results/` paths
resolve correctly, e.g.:

```bash
python src/analysis/degradation_line.py
python src/analysis/plot_sensitivity.py
python src/analysis/radial/radial_analysis.py
```

The data-prep and eval scripts (`src/data_prep/*`, `src/eval/*`) require the
cluster dataset and a trained model, so they are intended to run on Snellius.

## Pipeline (cluster)

1. `src/data_prep/extract_frames.py` (or `framecut.py`) — videos → frames
2. `src/fog/synthetic_fog_torch.py` — synthesise fog levels i00..i09
3. `src/data_prep/prepare_dataset.py` — frames → YOLOv5 `images/`+`labels/`
4. `src/data_prep/create_fogaware_dataset.py` — build the mixed training set
5. Train YOLOv5 with `configs/anti_uav.yaml` (baseline) / `configs/fog_aware.yaml`
   (fog-aware); validate with the per-condition `configs/anti_uav_fog_*.yaml`
6. `src/eval/*` and `src/analysis/*` — metrics and figures into `results/`
