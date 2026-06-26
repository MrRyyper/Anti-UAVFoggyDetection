# Fog-Robust UAV Detection

Master's thesis code. The question: does **fog-aware training** (mixing clear and
synthetic-fog images) make YOLOv5 better at spotting UAVs in fog? Tested on the
Anti-UAV300 RGB dataset.


## Layout

```
configs/      # YOLOv5 data .yaml files (one per eval condition)
src/
  data_prep/  # frame extraction + dataset assembly
  fog/        # synthetic fog (Koschmieder model), CPU and GPU versions
  eval/       # metrics: aggregate, size-stratified, failure mechanisms
  analysis/   # plotting (degradation curves, sensitivity, radial EDA)
notebooks/    # EDA notebooks
results/
  metrics/    # output CSVs
  figures/    # output PNG/PDF figures
slurm/        # HPC job scripts
docs/         # HTML diagrams / EDA exports
```

Gitignored (not in the repo): `Anti-UAV-RGBT/` (dataset), `yolov5/` (checkout),
`prepared_dataset/`, `framecuts/`, `runs/`.

## Dataset

The dataset can be found at `https://github.com/ZhaoJ9014/Anti-UAV`. 

## YOLOv5

Training/validation use Ultralytics YOLOv5 (`train.py` / `val.py`), which is not
in this repo. Clone it at the repo root and grab the pretrained backbone:

```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
wget -P yolov5 https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
```

Models are fine-tuned from `yolov5m.pt`.

## Pipeline

1. `src/data_prep/extract_frames.py` (or `framecut.py`) — videos -> frames
2. `src/fog/synthetic_fog_torch.py` — synthesise fog levels i00..i09
3. `src/data_prep/prepare_dataset.py` — frames -> YOLOv5 `images/`+`labels/`
4. `src/data_prep/create_fogaware_dataset.py` — build the mixed training set
5. Train YOLOv5 with `configs/anti_uav.yaml` (baseline) / `configs/fog_aware.yaml`
   (fog-aware); validate with the per-condition `configs/anti_uav_fog_*.yaml`
6. `src/eval/*` and `src/analysis/*` — metrics and figures into `results/`
