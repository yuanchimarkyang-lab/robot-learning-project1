import numpy as np
import torch


def rename_libero_observation(obs):
    """
    This function rename the wrist_image (NVIDIA Convention) into image2 (Lerobot Convention)
    """
    obs = dict(obs)

    obs["observation.images.wrist_image"] = obs.pop("observation.images.image2")

    return obs


def nvidia_action_to_libero(action):
    """
    This function convert the grip convention from
        NVIDIA Convention: 1: open, 0: close
        Lerobot Convention: -1: open, 1: close
    """
    if torch.is_tensor(action):
        action = action.clone()
        action[..., -1] = (
            1.0 - 2.0 * action[..., -1]
        )
        return action

    action = np.array(action, copy=True)
    action[..., -1] = (
        1.0 - 2.0 * action[..., -1]
    )
    return action
