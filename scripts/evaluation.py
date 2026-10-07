"""
This script implements evaluation with video output.

Note
1. To improve reproducibility, the random seeds for each policy roll-out is supplied as POLICY_SEED_BASE + init_state_id.
2. Configurations to be selected include:
    (1) POLICY_MODE: ACT, Diffusion, or SmolVLA
    (2) EXP_NAME: the name of the experiment output folder
    (3) CHECKPOINT_LIST: a list of checkpoint desired to be evaluated
    (4) VIDEO_MODE= "all": save videos for all cases.
                    "failure-only": save the videos for the failed case.
                    "None" will not save videos. 
                    "Read": read the previous evaluation results, rollout on the failed cases, and save videos.
3. The video will be stored at /results/<EXP_NAME>/video
4. The evaluation results will be stored at /results/<EXP_NAME>/evaluation
"""

from pathlib import Path
import time

import random
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

from lerobot.policies.act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors, make_policy

from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

from src.libero_nvidia_adapter import (
    nvidia_action_to_libero,
    rename_libero_observation,
)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path("~/ML/robot-learning-project1").expanduser()
DEV_STATES_PATH = (PROJECT_ROOT / "data/eval_init_states/libero_goal_bowl_plate_dev10.npy")

LIBERO_SUITE = "libero_goal"
LIBERO_TASK_ID = 8

TASK_LANGUAGE = "put the bowl on the plate"

FPS = 20
VIDEO_FPS = 20
MAX_EPISODE_STEPS = 300
EVAL_SEED = 1000
POLICY_SEED_BASE = 20_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POLICY_MODE = "ACT" # "ACT", "Diffusion", "SmolVLA"
EXP_NAME = "act_baseline" # "act_baseline" "diffusion_baseline" "smolvla_lora_baseline"
MODEL_ROOT = (PROJECT_ROOT / f"results/{EXP_NAME}")
output_dir = MODEL_ROOT/ "video"
output_dir.mkdir(parents=True,exist_ok=True)
 
# For ACT
CHECKPOINT_LIST = ["000100", "000200", "000300", "000400", "000500","000600","000700","000800","000900","001000",
                   "002000", "003000", "004000", "005000", "006000","007000","008000","009000","010000"]

# For Diffusion
#CHECKPOINT_LIST = ["000500","001000","001500","002000","002500","003000","003500","004000","004500","005000",
#                   "005500","006000","006500","007000","007500","008000","008500","009000","009500","010000",
#                   "010500","011000","011500","012000","012500","013000","013500","014000","014500","015000"]


# For SmolVLA
#CHECKPOINT_LIST = ["000500","001000","001500","002000","002500","003000","003500","004000","004500","005000"]


Eval_MODE = "dev"
Video_MODE = "all" # "failure-only" "all" "None" "Read"







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


def tensor_image_to_uint8(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    # Remove batch dimension: BCHW -> CHW
    if image.ndim == 4:
        image = image[0]

    # CHW -> HWC
    if (
        image.ndim == 3
        and image.shape[0] in (1, 3, 4)
    ):
        image = np.transpose(
            image,
            (1, 2, 0),
        )

    # float [0,1] -> uint8
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).round().astype(np.uint8)

    else:
        image = image.astype(np.uint8)

    return np.ascontiguousarray(image)


def set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# make sure cuda use deterministic path for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Sanity checks
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # INITIAL SET-UP
    # ============================================================

    print("=" * 70)
    print(f"{POLICY_MODE} -> Export LIBERO baseline Video")
    print("=" * 70)

    print("Device:", device)
    print("Suite:", LIBERO_SUITE)
    print("LIBERO task ID:", LIBERO_TASK_ID)
    print("Task:", TASK_LANGUAGE)
    print("Eval_MODE:", Eval_MODE)
    print("POLICY_MODE:", POLICY_MODE)
    print("Video_MODE:", Video_MODE)
    print("FPS:", FPS)
    print()


    # ============================================================
    # Configure LIBERO
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
    # Build a single synchronous environment
    # ============================================================

    envs = make_env(
        env_cfg,
        n_envs=1,
        use_async_envs=False,
    )

    env = envs[LIBERO_SUITE][LIBERO_TASK_ID]

    libero_env = env.envs[0]

    print("Official init states:",libero_env._init_states.shape)



    

    for CHECKPOINT in CHECKPOINT_LIST:
        eva_list = []
        CHECKPOINT_PATH = CHECKPOINT_PATH_func(CHECKPOINT)

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint does not exist:\n{CHECKPOINT_PATH}")
        
        # ============================================================
        # Load checkpoint accoding to different POLICY
        # ============================================================

        print(f"Loading {POLICY_MODE} policy...")

        if POLICY_MODE=="ACT":

            policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
            policy.to(device)
            policy.eval()
            print("Policy loaded.")
            print("chunk_size:", policy.config.chunk_size)
            print("n_action_steps:", policy.config.n_action_steps)

        elif POLICY_MODE=="Diffusion":

            policy = DiffusionPolicy.from_pretrained(CHECKPOINT_PATH)
            policy.to(device)
            policy.eval()
            print("Policy loaded.")
            print("n_obs_steps:", policy.config.n_obs_steps)
            print("horizon:", policy.config.horizon)
            print("n_action_steps:", policy.config.n_action_steps)

        elif POLICY_MODE=="SmolVLA":
            policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT_PATH)
            policy_cfg.pretrained_path = CHECKPOINT_PATH
            policy_cfg.device = device

            print("Policy loaded.")
            print("policy type:", policy_cfg.type)
            print("use_peft:", policy_cfg.use_peft)
            print("pretrained_path:", policy_cfg.pretrained_path)

            print("n_obs_steps:", policy_cfg.n_obs_steps)
            print("chunk_size:", policy_cfg.chunk_size)
            print("n_action_steps:", policy_cfg.n_action_steps)


            POLICY_RENAME_MAP = {
                "observation.images.image2":
                    "observation.images.wrist_image",
            }

            policy = make_policy(
                cfg=policy_cfg,
                env_cfg=env_cfg,
                rename_map=POLICY_RENAME_MAP,
            )

            policy.eval()

            print("\nPolicy Type:",type(policy))

        print("\nPolicy input features:")
        for name, feature in policy.config.input_features.items():
            print(" ", name, "->", feature)

        print("\nPolicy output features:")
        for name, feature in policy.config.output_features.items():
            print(" ", name, "->", feature)


        # ============================================================
        # Restore saved pre/postprocessors
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


        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_cfg,
                policy_cfg=policy.config,
            )

        
        #======================================================================
        # Load INITIAL_STATES_LIST
        #======================================================================
        if Video_MODE == "Read":
            df = pd.read_csv(MODEL_ROOT/ f"evaluation/{Eval_MODE}_{CHECKPOINT}.csv")
            INITIAL_STATES_LIST = df[df["success"]==False]["init_state_id"].to_list()
        else:
            INITIAL_STATES_LIST = list(range(libero_env._init_states.shape[0]))



        for init_state_id in INITIAL_STATES_LIST:
            local_time = time.localtime()
            # Format and print the time (HH:MM:SS)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

            print(f"rollout on {init_state_id} using {CHECKPOINT} starts at {current_time}")
            # Reset it at the start of every episode.
            eval_start = time.perf_counter()
            
            frames = []
            frames_wrist = []
            libero_env.init_state_id = init_state_id
            policy.reset()
            # set seeds so that the results can be reproduced
            obs, info = env.reset(seed=EVAL_SEED)
            # the random seeds are supplied to improve reproducibility.
            set_eval_seed(POLICY_SEED_BASE + init_state_id)
            success = False
            done = False
            step = 0

            # ============================================================
            # Rollout at init state using policy given by CHECKPOINT
            # ============================================================

            while ((step < MAX_EPISODE_STEPS) and (not success) and (not done)):
                
                policy_obs = preprocess_observation(obs)
                policy_obs["task"] = [TASK_LANGUAGE]
                policy_obs = env_preprocessor(policy_obs)
                policy_obs = rename_libero_observation(policy_obs)
                
                frame = tensor_image_to_uint8(policy_obs["observation.images.image"])
                frames.append(frame)
                frame_wrist = tensor_image_to_uint8(policy_obs["observation.images.wrist_image"])
                frames_wrist.append(frame_wrist)

                policy_input = preprocessor(policy_obs)

                with torch.inference_mode():
                    action = policy.select_action(policy_input)


                action_nvidia = postprocessor(action)
                action_libero = nvidia_action_to_libero(action_nvidia)
                transition = {ACTION: action_libero}
                transition = env_postprocessor(transition)
                action_env = transition[ACTION]

                if torch.is_tensor(action_env):
                    action_np = (action_env.detach().cpu().numpy())
                else:
                    action_np = np.asarray(action_env)

                obs, reward, terminated, truncated, info = env.step(action_np)
                
                success = info_bool(info,"is_success")
                done = (first_bool(terminated) or first_bool(truncated))
                step = step + 1

            if success:
                # Hold the successful final frame for 1.5 seconds
                frames.extend([frame.copy() for _ in range(int(1.5 * FPS))])
                frames_wrist.extend([frame_wrist.copy() for _ in range(int(1.5 * FPS))])


            eva_dict = {
                "policy": POLICY_MODE,
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
            print("Steps attempted:", step) 
            print("Success:", success)
            print("=" * 70)

            # ============================================================
            # SAVE Video
            # ============================================================

            success_tag = "success" if success else "failure"
            if (Video_MODE == "failure-only" and success_tag == "success") or Video_MODE == "None":
                print("No video is saved!")
            else:
                output_path = (MODEL_ROOT/ "video" /
                                (
                                f"{Eval_MODE}_"
                                f"checkpoint_{CHECKPOINT}_"
                                f"state_{init_state_id:02d}_"
                                f"{success_tag}_image.mp4"
                            )
                        )
                write_video(str(output_path),np.stack(frames),VIDEO_FPS)

                output_path = (MODEL_ROOT/ "video" /
                                (
                                f"{Eval_MODE}_"
                                f"checkpoint_{CHECKPOINT}_"
                                f"state_{init_state_id:02d}_"
                                f"{success_tag}_wrist_image.mp4"
                            )
                        )
                write_video(str(output_path),np.stack(frames_wrist),VIDEO_FPS)

                print(f"Video Saved Successfully for checkpoint = {CHECKPOINT},"
                    f" Eval_MODE = {Eval_MODE}, state_id = {init_state_id} at {str(output_path)}")

        # ============================================================
        # SAVE Evaluation Statustics
        # ============================================================

        if Video_MODE != "Read":
            df = pd.DataFrame(eva_list)

            print(df)
            success_rate = (df["success"].mean())
            print(f"\nSuccess rate: {success_rate:.1%}")

            output_path = (MODEL_ROOT/ f"evaluation/{Eval_MODE}_{CHECKPOINT}.csv")
            output_path.parent.mkdir(parents=True,exist_ok=True)

            df.to_csv(output_path, index=False)

            print("Saved:", output_path)






    