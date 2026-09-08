"""
    This script generates random initial states for TARGET_TASK_ID in SUITE_NAME, with seeds starting 10000 to (10000 + N_STATES -1)
"""

import os
from pathlib import Path

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITE_NAME = "libero_goal"
TARGET_TASK_ID = 8

N_STATES = 10
SEEDS = list(range(10_000, 10_000 + N_STATES))

OUTPUT = Path(
    "data/eval_init_states/libero_goal_bowl_plate_dev10.npy"
)


# --------------------------------------------------
# Get task definition
# --------------------------------------------------

suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
task = suite.get_task(TARGET_TASK_ID)

bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

print("Task:", task.language)
print("BDDL:", bddl_file)


# --------------------------------------------------
# Create simulator
# --------------------------------------------------

env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    camera_heights=256,
    camera_widths=256,
    control_freq=20, # doesn't matter if we only generate the initial state
    hard_reset=True,
)


# --------------------------------------------------
# Generate reproducible initial states
# --------------------------------------------------

states = []

for seed in SEEDS:
    print(f"Generating seed {seed}")

    env.seed(seed)
    env.reset()

    state = np.asarray(
        env.get_sim_state()
    ).copy()

    states.append(state)


states = np.stack(states)

print("Generated shape:", states.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(OUTPUT, states)

print("Saved:", OUTPUT)

env.close()
