from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt


def plot_rewards(
    episodes: List[int],
    rewards: List[float],
    rolling: Optional[List[float]],
    out_path: str,
    title: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="episode reward")
    if rolling is not None:
        plt.plot(episodes, rolling, label="rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
