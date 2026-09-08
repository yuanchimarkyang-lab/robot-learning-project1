# Milestone 0 — LIBERO + LeRobot Setup

## System

OS: Ubuntu under WSL2

GPU: RTX 5060 Laptop 8 GB

Project root: ~/ML/robot-learning-project1

LeRobot commit: fbb811fca92504439792b97d216f0d00c2268382

PyTorch: 2.11.0+cu128

CUDA: 13.2

## Target Task

Suite: LIBERO-Goal

Task: Put the bowl on the plate

Task ID: 8

## Observations

Main camera: 
shape = (3,256,256)

Wrist camera:
shape = (3,256,256)

Robot state:
shape = 8
meaning = [
    "eef_x",
    "eef_y",
    "eef_z",
    "axis_angle_x",
    "axis_angle_y",
    "axis_angle_z",
    "gripper_0",
    "gripper_1",
]

## Actions
control mode: relative

dimensions: 7

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rot_1",
    "delta_rot_2",
    "delta_rot_3",
    "gripper",
]

control frequency:
20 Hz

## Dataset

repo:
`nvidia/LIBERO_LeRobot_v3`

number of relevant demonstrations: 49

FPS: 20

episode length: 
- Mean length: 93.1
- Min length: 79
- Max length: 126
- Mean duration: 4.66 seconds

## Evaluation

success condition: Use the simulator's definition, i.e. info["is_success"] sent back by env.step()

held-out initial state strategy:
- training: 10 generated initial state at ~/ML/robot-learning-project1/data/eval_init_states/libero_goal_bowl_plate_dev10.npy
- test: 50 official initial states


Data Record Schema:
- location: results/evaluation/ 
- columns: 
    - policy: ACT, diffusion, or VLA, 
    - checkpoint: which checkpoint the model is, including which trial
    - epoch: how many epochs have the checkpoint been trained on. 
    - task: which task is it?
    - init_state_id: to identify the initial state
    - seed: the random seed.
    - success: 1 of succeeded, 0 otherwise
    - steps: the episode length
    - failure_mode: manual annotation of the failure mode

| Failure Mode | Description |
| :----------- | :-----------|
| reach | robot never gets into a viable grasp pose |
| grasp | reaches object but fails to acquire it |
| transport | grasp succeeds but object is dropped/lost |
| placement | reaches target region but fails placement |
| release | object reaches plate but gripper does not release correctly |
| recovery | initial mistake occurs and policy cannot recover |
| timeout | behavior never reaches task completion|



## Open Questions
- **How to create more initial state in terms of evaluation both while training and in test time?** We can generate the initial state randomly through scripts/generate_dev_init_states.py
- **What is the default maximum episode length?** It is 300 for libero_goal. It is defined using TASK_SUITE_MAX_STEPS inside src/lerobot/envs/libero.py 
- What is the seed for?