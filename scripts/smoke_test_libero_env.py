import numpy as np

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env


TARGET_TASK_ID = 8  # replace with your discovered ID


cfg = LiberoEnv(
    task="libero_goal",
    task_ids=[TARGET_TASK_ID],
    init_states=True,
    hard_reset=True,
)

envs = make_env(
    cfg,
    n_envs=1,
    use_async_envs=False,
)

env = envs["libero_goal"][TARGET_TASK_ID]

print("Observation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

print("\nPredefined Episode Length:",cfg.episode_length)

obs, info = env.reset(seed=0)

print("\nObservation keys:")
print(obs.keys())

print("\nReset info:")
print(info)

action = np.zeros(env.action_space.shape, dtype=np.float32)

obs, reward, terminated, truncated, info = env.step(action)

print("\nAfter one zero-action step:")
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
print("info:", info)

for step in range(20):
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step:02d}, "
        f"reward={reward}, "
        f"success={info.get('is_success')}"
    )

    if np.any(terminated) or np.any(truncated):
        break



def print_nested(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print_nested(value, prefix + str(key) + ".")
    else:
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        print(prefix[:-1], "shape=", shape, "dtype=", dtype)


print_nested(obs)




env.close()
