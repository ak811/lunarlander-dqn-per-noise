from __future__ import annotations

import os
from typing import Optional, Callable

import gymnasium as gym
import numpy as np


def make_env(env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    return env


def record_episode_video(
    env_name: str,
    video_dir: str,
    name_prefix: str,
    policy_fn: Callable[[np.ndarray], int],
    seed: Optional[int] = None,
    max_steps: int = 1000,
) -> str:
    os.makedirs(video_dir, exist_ok=True)
    env = make_env(env_name, seed=seed, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = int(policy_fn(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    env.close()
    return video_dir
