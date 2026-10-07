#!/usr/bin/env bash


PROJECT_ROOT="$HOME/ML/robot-learning-project1"
TRAIN_ROOT="$HOME/ML/robot-learning-project1/external/lerobot"

cd "$TRAIN_ROOT"

#lerobot-train \
#  --config_path=$HOME/ML/robot-learning-project1/configs/act_bowl_plate_baseline.yaml \
#  2>&1 | tee \
#  $HOME/ML/robot-learning-project1/results/logs/act_bowl_plate_baseline.log

lerobot-train \
  --config_path=$HOME/ML/robot-learning-project1/results/act_baseline3/checkpoints/001000/pretrained_model/train_config.json \
  --resume=true \
  --steps=10000 \
  --save_freq=1000 \
  2>&1 | tee \
  $HOME/ML/robot-learning-project1/results/logs/act_bowl_plate_baseline.log

cd "$PROJECT_ROOT"

python scripts/extract_training_metrics.py
