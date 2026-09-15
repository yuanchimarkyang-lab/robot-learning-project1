from pathlib import Path
import time

import numpy as np
import torch

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

CHECKPOINT = (PROJECT_ROOT / "results/diffusion_smoke2/checkpoints/last/pretrained_model") 
# this act_smoke2 is trained with batchsize = 8

LIBERO_SUITE = "libero_goal"
LIBERO_TASK_ID = 8

TASK_LANGUAGE = "put the bowl on the plate"

FPS = 20
MAX_EPISODE_STEPS = 300

# For Session 1D.2 we intentionally run only a short rollout.
SMOKE_STEPS = 50

SEED = 1000


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


# ============================================================
# 1. Sanity checks
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint does not exist:\n{CHECKPOINT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("Diffusion -> LIBERO smoke evaluation")
print("=" * 70)

print("Checkpoint:", CHECKPOINT)
print("Device:", device)
print("Suite:", LIBERO_SUITE)
print("LIBERO task ID:", LIBERO_TASK_ID)
print("Task:", TASK_LANGUAGE)
print("FPS:", FPS)
print()


# ============================================================
# 2. Load Diffusion checkpoint
# ============================================================

print("Loading Diffusion policy...")

policy = DiffusionPolicy.from_pretrained(CHECKPOINT)
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
# 3. Restore Diffusion's saved pre/postprocessors
# ============================================================

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=str(CHECKPOINT),
    preprocessor_overrides={
        "device_processor": {
            "device": str(device)
        }
    },
)

print("\nPolicy processors loaded.")


# ============================================================
# 4. Configure LIBERO
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

env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy.config,
    )


# ============================================================
# 5. Build a single synchronous environment
# ============================================================

envs = make_env(
    env_cfg,
    n_envs=1,
    use_async_envs=False,
)

env = envs[LIBERO_SUITE][LIBERO_TASK_ID]


# ============================================================
# 6. One smoke rollout
# ============================================================

try:
    # Important:
    # Diffusion maintains an internal action queue.
    # Reset it at the start of every episode.
    policy.reset()

    obs, info = env.reset(seed=SEED)

    print("\nEnvironment reset successful.")
    print("Initial info:", info)

    success = False

    for step in range(SMOKE_STEPS):

        # ----------------------------------------------------
        # A. Convert raw Gym observation to LeRobot tensors
        # ----------------------------------------------------

        policy_obs = preprocess_observation(obs)

        # Difussion does not use language, but adding it keeps this
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

        if step == 0:
            print("\nKeys BEFORE NVIDIA rename:")
            for key in policy_obs:
                print(" ", key)

        policy_obs = rename_libero_observation(
            policy_obs
        )

        if step == 0:
            print("\nKeys AFTER NVIDIA rename:")

            for key, value in policy_obs.items():
                print_tensor_info(key, value)

            # Fail early if our policy contract is wrong.
            assert (
                "observation.images.image"
                in policy_obs
            )

            assert (
                "observation.images.wrist_image"
                in policy_obs
            )

            assert (
                "observation.images.image2"
                not in policy_obs
            )

            assert (
                "observation.state"
                in policy_obs
            )

            assert (
                policy_obs["observation.state"].shape[-1]
                == 8
            )


        # ----------------------------------------------------
        # D. Policy preprocessing
        #
        # Includes normalization and device transfer using
        # the saved training pipeline.
        # ----------------------------------------------------

        policy_input = preprocessor(
            policy_obs
        )


        # ----------------------------------------------------
        # E. inference
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        with torch.inference_mode():
            action = policy.select_action(
                policy_input
            )

        inference_ms = (time.perf_counter() - inference_start) * 1000.0


        # ----------------------------------------------------
        # F. postprocessing
        #
        # This converts normalized model output back into
        # NVIDIA dataset action units.
        # ----------------------------------------------------

        action_nvidia = postprocessor(
            action
        )


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

        transition = {
            ACTION: action_libero
        }

        transition = env_postprocessor(
            transition
        )

        action_env = transition[ACTION]


        # ----------------------------------------------------
        # I. Validation before touching the simulator
        # ----------------------------------------------------

        if torch.is_tensor(action_env):
            action_np = (
                action_env
                .detach()
                .cpu()
                .numpy()
            )
        else:
            action_np = np.asarray(
                action_env
            )

        if action_np.shape != (1, 7):
            raise RuntimeError(
                "Unexpected action shape: "
                f"{action_np.shape}; expected (1, 7)"
            )

        if not np.isfinite(action_np).all():
            raise RuntimeError(
                f"Non-finite action at step {step}: "
                f"{action_np}"
            )


        # ----------------------------------------------------
        # J. Print several actions for inspection
        # ----------------------------------------------------

        if step < 300:
            nvidia_np = (
                action_nvidia
                .detach()
                .cpu()
                .numpy()
                if torch.is_tensor(action_nvidia)
                else np.asarray(action_nvidia)
            )

            print(f"\nStep {step}")

            print("NVIDIA action:", np.round(nvidia_np[0],3,),)

            print("LIBERO action:", np.round(action_np[0],3,),)

            print(
                f"select_action time: "
                f"{inference_ms:.1f} ms"
            )


        # ----------------------------------------------------
        # K. Step simulator
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = (
            env.step(action_np)
        )

        success = info_bool(
            info,
            "is_success",
        )

        done = (
            first_bool(terminated)
            or first_bool(truncated)
        )


        # Print progress occasionally.
        if (
            step % 10 == 0
            or success
            or done
        ):
            print(
                f"step={step:03d} | "
                f"reward={float(np.asarray(reward)[0]):.3f} | "
                f"success={success} | "
                f"terminated={first_bool(terminated)} | "
                f"latency={inference_ms:.1f} ms"
            )


        if success:
            print(
                "\nUnexpected but welcome: "
                "smoke checkpoint solved the task!"
            )
            break

        if done:
            print(
                "\nEnvironment terminated "
                "before smoke horizon."
            )
            break


    print("\n" + "=" * 70)
    print("Smoke rollout complete.")
    print("Steps attempted:", step + 1)
    print("Success:", success)
    print("=" * 70)


finally:
    close_envs(envs)

