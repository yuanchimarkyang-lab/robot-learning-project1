#!/usr/bin/env bash


PROJECT_ROOT="$HOME/ML/robot-learning-project1"
TRAIN_ROOT="$HOME/ML/robot-learning-project1/external/lerobot"

cd "$TRAIN_ROOT"

lerobot-train \
  --config_path=$HOME/ML/robot-learning-project1/configs/act_bowl_plate_baseline.yaml \
  2>&1 | tee \
  $HOME/ML/robot-learning-project1/results/logs/act_bowl_plate_baseline.log

cd "$PROJECT_ROOT"
