#!/usr/bin/env bash

PROJECT_ROOT="$HOME/ML/robot-learning-project1"

source "$PROJECT_ROOT/external/lerobot/.venv/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"


cd "$PROJECT_ROOT"

