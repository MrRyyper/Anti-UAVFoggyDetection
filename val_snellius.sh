#!/bin/bash
#SBATCH --job-name=yolov5_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate your virtual environment
source $HOME/envs/yolo/bin/activate

# Disable Ultralytics network calls (compute nodes have restricted internet)
export YOLO_OFFLINE=1
export ULTRALYTICS_OFFLINE=1
export YOLO_CONFIG_DIR=$HOME/.config/Ultralytics

# Go to yolov5 directory
cd $HOME/thesis-1/yolo/yolov5

# Evaluate on test set
python val.py \
    --weights runs/train/results/weights/best.pt \
    --data ../../anti_uav.yaml \
    --imgsz 640 \
    --task test \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name results
