from pathlib import Path
import time

import numpy as np
import torch

import pandas as pd

from lerobot.envs import (
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.configs import LiberoEnv
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION

from src.libero_nvidia_adapter import (
    nvidia_action_to_libero,
    rename_libero_observation,
)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path("~/ML/robot-learning-project1").expanduser()

EXP_NAME = "diffusion_baseline"

MODEL_ROOT = (PROJECT_ROOT / f"results/{EXP_NAME}")

DEV_STATES_PATH = (PROJECT_ROOT / "data/eval_init_states/libero_goal_bowl_plate_dev10.npy")

#CHECKPOINT_LIST = ["000500","001000","001500","002000","002500","003000","003500","004000","004500","005000"]
CHECKPOINT_LIST = ["006000","007000","008000","009000","010000"]

LIBERO_SUITE = "libero_goal"
LIBERO_TASK_ID = 8

TASK_LANGUAGE = "put the bowl on the plate"

FPS = 20
MAX_EPISODE_STEPS = 300
EVAL_SEED = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE = "test" # "eval" "test"

# ============================================================
# Small helpers
# ============================================================

def first_bool(value):
    """Convert scalar/vector Gym result to one bool."""
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0])


def info_bool(info, key):
    """Extract one boolean from Gymnasium vector info."""
    if key not in info:
        return False

    return first_bool(info[key])


def print_tensor_info(name, value):
    if torch.is_tensor(value):
        print(
            f"{name:40s}"
            f" shape={tuple(value.shape)}"
            f" dtype={value.dtype}"
            f" device={value.device}"
        )
    else:
        print(
            f"{name:40s}"
            f" type={type(value)}"
        )

def CHECKPOINT_PATH_func(CHECKPOINT="010000"):

    return (MODEL_ROOT / f"checkpoints/{CHECKPOINT}/pretrained_model") 


def print_gpu_memory(tag):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"[{tag}] "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"peak={peak:.2f} GB"
        )
    

        




# ============================================================
# 1. Sanity checks
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # 0. INITIAL SET-UP
    # ============================================================

    print("=" * 70)
    print("Diffusion -> LIBERO baseline evaluation")
    print("=" * 70)

    print("Device:", device)
    print("Suite:", LIBERO_SUITE)
    print("LIBERO task ID:", LIBERO_TASK_ID)
    print("Task:", TASK_LANGUAGE)
    print("FPS:", FPS)
    print()


    # ============================================================
    # 1. Configure LIBERO
    # ============================================================

    env_cfg = LiberoEnv(
        task=LIBERO_SUITE,
        task_ids=[LIBERO_TASK_ID],

        # Make all temporal assumptions explicit.
        fps=FPS,
        episode_length=MAX_EPISODE_STEPS,

        # Match NVIDIA demonstrations.
        observation_height=256,
        observation_width=256,

        obs_type="pixels_agent_pos",

        init_states=True,
        hard_reset=True,

        # Keep LeRobot's standard names here:
        # image + image2.
        # Our adapter will rename image2 -> wrist_image.
        camera_name_mapping={
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        },
    )


    # ============================================================
    # 2. Build a single synchronous environment
    # ============================================================

    envs = make_env(
        env_cfg,
        n_envs=1,
        use_async_envs=False,
    )

    env = envs[LIBERO_SUITE][LIBERO_TASK_ID]

    libero_env = env.envs[0]

    print("Official init states:",libero_env._init_states.shape)

    if MODE == "eval":
        # ============================================================
        # 3. LOAD dev_states if in evaluation model
        # ============================================================
        dev_states = np.load(DEV_STATES_PATH,allow_pickle=False)
        assert dev_states.shape[0] == 10
        assert dev_states.ndim == 2

        libero_env._init_states = dev_states
        libero_env._reset_stride = 1

        print("Custom dev states installed:",libero_env._init_states.shape)


    

    for CHECKPOINT in CHECKPOINT_LIST:
        eva_list = []

        CHECKPOINT_PATH = CHECKPOINT_PATH_func(CHECKPOINT)

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint does not exist:\n{CHECKPOINT_PATH}")
        
        # ============================================================
        # 4. Load Diffusion checkpoint
        # ============================================================

        print("Loading Diffusion policy...")

        policy = DiffusionPolicy.from_pretrained(CHECKPOINT_PATH)
        policy.to(device)
        policy.eval()

        print("Policy loaded.")
        print("n_obs_steps:", policy.config.n_obs_steps)
        print("horizon:", policy.config.horizon)
        print("n_action_steps:", policy.config.n_action_steps)

        print("\nPolicy input features:")
        for name, feature in policy.config.input_features.items():
            print(" ", name, "->", feature)

        print("\nPolicy output features:")
        for name, feature in policy.config.output_features.items():
            print(" ", name, "->", feature)

        # ============================================================
        # 5. Restore ACT's saved pre/postprocessors
        # ============================================================

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(CHECKPOINT_PATH),
            preprocessor_overrides={
                "device_processor": {
                    "device": str(device)
                }
            },
        )

        print("\nPolicy processors loaded.")

        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_cfg,
                policy_cfg=policy.config,
            )

        for init_state_id in range(libero_env._init_states.shape[0]):
            local_time = time.localtime()
            # Format and print the time (HH:MM:SS)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

            print(f"rollout on {init_state_id} using {CHECKPOINT} starts at {current_time}")
            print_gpu_memory(f"before state {init_state_id}")
            # Reset it at the start of every episode.
            eval_start = time.perf_counter()
            libero_env.init_state_id = init_state_id
            policy.reset()
            obs, info = env.reset(seed=EVAL_SEED)
            success = False
            done = False
            step = 0

            # ============================================================
            # 6. Rollout at init state using policy given by CHECKPOINT
            # ============================================================

            while ((step < MAX_EPISODE_STEPS) and (not success) and (not done)):
                # ----------------------------------------------------
                # A. Convert raw Gym observation to LeRobot tensors
                # ----------------------------------------------------
                policy_obs = preprocess_observation(obs)

                # ACT does not use language, but adding it keeps this
                # compatible with LeRobot's standard evaluation pipeline.
                policy_obs["task"] = [TASK_LANGUAGE]


                # ----------------------------------------------------
                # B. LIBERO-specific observation processing
                #
                # Converts structured robot_state into the 8-D
                # observation.state used during training.
                # ----------------------------------------------------
                policy_obs = env_preprocessor(policy_obs)

                # ----------------------------------------------------
                # C. NVIDIA camera-name compatibility
                #
                # Live:
                # observation.images.image2
                #
                # NVIDIA training:
                # observation.images.wrist_image
                # ----------------------------------------------------

                policy_obs = rename_libero_observation(policy_obs)

                # ----------------------------------------------------
                # D. Policy preprocessing
                #
                # Includes normalization and device transfer using
                # the saved training pipeline.
                # ----------------------------------------------------

                policy_input = preprocessor(policy_obs)

                # ----------------------------------------------------
                # E. ACT inference
                # ----------------------------------------------------

                

                with torch.inference_mode():
                    action = policy.select_action(
                        policy_input
                    )

                

                # ----------------------------------------------------
                # F. Policy postprocessing
                #
                # This converts normalized model output back into
                # NVIDIA dataset action units.
                # ----------------------------------------------------
                action_nvidia = postprocessor(action)
                # ----------------------------------------------------
                # G. NVIDIA -> LIBERO gripper convention
                #
                # g_LIBERO = 1 - 2 * g_NVIDIA
                # ----------------------------------------------------
                action_libero = nvidia_action_to_libero(action_nvidia)

                # ----------------------------------------------------
                # H. Environment action postprocessor
                #
                # Currently essentially identity for LIBERO,
                # but keeping it preserves LeRobot's standard order.
                # ----------------------------------------------------
                transition = {ACTION: action_libero}

                transition = env_postprocessor(transition)

                action_env = transition[ACTION]

                # ----------------------------------------------------
                # I. Validation before touching the simulator
                # ----------------------------------------------------

                if torch.is_tensor(action_env):
                    action_np = (action_env.detach().cpu().numpy())
                else:
                    action_np = np.asarray(action_env)

                # ----------------------------------------------------
                # K. Step simulator
                # ----------------------------------------------------

                obs, reward, terminated, truncated, info = env.step(action_np)

                success = info_bool(info,"is_success")

                done = (first_bool(terminated) or first_bool(truncated))

                step = step + 1


            eva_dict = {
                "policy": "Diffusion",
                "experiment_name": EXP_NAME,
                "checkpoint": CHECKPOINT,
                "task": LIBERO_TASK_ID,
                "init_state_id": init_state_id,
                "seed": EVAL_SEED,
                "success": success,
                "steps": step,
                "failure_mode": "NA",
            }

            eva_list.append(eva_dict)
            eval_ms = (time.perf_counter() - eval_start) * 1000.0

            print("\n" + "=" * 70)
            print("rollout complete.")
            print_gpu_memory(f"after state {init_state_id}")
            print("Steps attempted:", step) # on 2026/9/12 changed from step+1 to step to match with the correct step number.
            print("Success:", success)
            print("=" * 70)
    

        # ============================================================
        # 7. SAVE Data
        # ============================================================


        df = pd.DataFrame(eva_list)

        print(df)
        success_rate = (df["success"].mean())
        print(f"\nSuccess rate: {success_rate:.1%}")
        if MODE == "eval":
            output_path = (MODEL_ROOT/ f"evaluation/dev_{CHECKPOINT}.csv")
        else:
            output_path = (MODEL_ROOT/ f"evaluation/test_{CHECKPOINT}.csv")

        output_path.parent.mkdir(parents=True,exist_ok=True)

        df.to_csv(output_path, index=False)

        print("Saved:", output_path)






    