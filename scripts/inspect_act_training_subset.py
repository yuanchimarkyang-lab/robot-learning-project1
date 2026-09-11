from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset


PROJECT_ROOT = Path("~/ML/robot-learning-project1").expanduser()

DATASET_ROOT = (
    PROJECT_ROOT
    / "data/nvidia_libero_v3/libero_goal"
)

REPO_ID = "nvidia/LIBERO_LeRobot_v3"
NVIDIA_TASK_INDEX = 0


dataset = LeRobotDataset(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    download_videos=False,
)


# --------------------------------------------------
# Verify task identity
# --------------------------------------------------

tasks = dataset.meta.tasks.reset_index()

row = tasks[
    tasks["task_index"] == NVIDIA_TASK_INDEX
]

assert len(row) == 1

print("Task:")
print(row.to_string(index=False))


# --------------------------------------------------
# Find episodes
# --------------------------------------------------

task_indices = np.asarray(
    dataset.hf_dataset["task_index"]
)

episode_indices = np.asarray(
    dataset.hf_dataset["episode_index"]
)

target_episode_ids = sorted(
    np.unique(
        episode_indices[
            task_indices == NVIDIA_TASK_INDEX
        ]
    ).astype(int).tolist()
)

print("\nNumber of target episodes:")
print(len(target_episode_ids))

print("\nEpisode IDs:")
print(target_episode_ids)


# --------------------------------------------------
# Episode statistics
# --------------------------------------------------

lengths = np.asarray([
    np.sum(episode_indices == ep)
    for ep in target_episode_ids
])

print("\nEpisode length:")
print("min:", lengths.min())
print("median:", np.median(lengths))
print("mean:", lengths.mean())
print("max:", lengths.max())

print(
    "median duration [s]:",
    np.median(lengths) / dataset.meta.fps,
)


# --------------------------------------------------
# YAML-ready list
# --------------------------------------------------

print("\nCopy this into dataset.episodes:")
for ep in target_episode_ids:
    print(f"    - {ep}")

