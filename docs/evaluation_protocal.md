# Evaluation Protocol

## Task
Suite: LIBERO-Goal

Task: Put the bowl on the plate

Task ID: 
- Benchmark: 8
- Demo: 0

## Dataset / Environment Adapter

Training dataset:
`nvidia/LIBERO_LeRobot_v3`

Temporal rate:
20 Hz

Camera mapping during evaluation:
`observation.images.image2` → `observation.images.wrist_image`

Camera orientation:
Verified visually to match between NVIDIA demonstrations and live LIBERO; no rotation correction is applied.

Action mapping:
The first six action dimensions are passed through unchanged. After policy output is unnormalized to NVIDIA dataset units, the gripper action is converted as:

`g_LIBERO = 1 - 2 * g_NVIDIA`


## Environment
Control mode: relative

Control frequency: 20 Hz

Observation cameras:
- external
- wrist

Maximum Episode Length: 300 (*defined using TASK_SUITE_MAX_STEPS inside src/lerobot/envs/libero.py*)

## Primary metric
Closed-loop task success rate: defined as success / rollouts

A rollout is successful iff LIBERO reports
`info["is_success"] == True`. The episode will be recorded as fail when the rollout reach 300 steps without success.

## Development evaluation
- 10 fixed benchmark initial states, generated with seeds 10000-10009
- same initial states for all checkpoints/policies

## Final evaluation
- 50 official LIBERO fixed initial states 0-49
- otherwise all available benchmark states
- identical state set across ACT / Diffusion / VLA

## Episode termination
- success
- environment termination
- maximum episode length: 300 as 

## Recorded fields
- policy: ACT, diffusion, or VLA, 
- checkpoint: which checkpoint the model is, including which trial
- epoch: how many epochs have the checkpoint been trained on. 
- task: which task is it?
- init_state_id: to identify the initial state
- seed: the random seed.
- success: 1 of succeeded, 0 otherwise
- steps: the episode length
- failure_mode: manual annotation of the failure mode

## Failure categories
| Failure Mode | Description |
| :----------- | :-----------|
| reach | robot never gets into a viable grasp pose |
| grasp | reaches object but fails to acquire it |
| transport | grasp succeeds but object is dropped/lost |
| placement | reaches target region but fails placement |
| release | object reaches plate but gripper does not release correctly |
| recovery | initial mistake occurs and policy cannot recover |
| timeout | behavior never reaches task completion|
