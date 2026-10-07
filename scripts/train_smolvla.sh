#!/usr/bin/env bash
# This shell script train a SmolVLA policy according to the config file and extract the training metrics

PROJECT_ROOT="$HOME/ML/robot-learning-project1"
TRAIN_ROOT="$HOME/ML/robot-learning-project1/external/lerobot"

cd "$TRAIN_ROOT"

lerobot-train \
  --config_path=$HOME/ML/robot-learning-project1/configs/smolvla_bowl_plate_lora.yaml \
  2>&1 | tee \
  $HOME/ML/robot-learning-project1/results/logs/smolvla_bowl_plate_lora_baseline.log


cd "$PROJECT_ROOT"

python scripts/extract_training_metrics.py
