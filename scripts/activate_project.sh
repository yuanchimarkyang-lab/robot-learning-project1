#!/usr/bin/env bash

PROJECT_ROOT="$HOME/ML/robot-learning-project1"

source "$PROJECT_ROOT/external/lerobot/.venv/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$PROJECT_ROOT"

