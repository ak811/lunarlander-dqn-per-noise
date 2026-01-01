from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from lunarlander_dqn.envs.lunarlander import make_env


def evaluate(agent, env_name: str, episodes: int = 10, seed: Optional[int] = None, max_steps: int = 1000) -> Dict[str, float]:
    env = make_env(env_name, seed=seed, render_mode=None)
    rewards = []

    # force deterministic greedy evaluation
    agent.cfg.exploration_mode = "none"

    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = agent.select_action(s, exploration_model=None)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += float(r)
            steps += 1
        rewards.append(total)

    env.close()
    arr = np.asarray(rewards, dtype=np.float32)
    return {
        "eval_mean_reward": float(arr.mean()),
        "eval_std_reward": float(arr.std()),
        "eval_min_reward": float(arr.min()),
        "eval_max_reward": float(arr.max()),
    }
