from __future__ import annotations

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lunarlander_dqn.agent.dqn import DQNAgent, DQNConfig
from lunarlander_dqn.envs.lunarlander import make_env
from lunarlander_dqn.training.evaluate import evaluate


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def main():
    p = argparse.ArgumentParser(description="Evaluate a saved DQN model.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(str(cfg.get("device", "auto")))
    env = make_env(cfg["env_name"], seed=int(cfg.get("seed", 42)), render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    agent_cfg = DQNConfig(
        gamma=float(cfg["gamma"]),
        learning_rate=float(cfg["learning_rate"]),
        batch_size=int(cfg["batch_size"]),
        train_every_steps=int(cfg["train_every_steps"]),
        target_update_every_episodes=int(cfg["target_update_every_episodes"]),
        replay_capacity=int(cfg["replay_capacity"]),
        min_replay_size=int(cfg["min_replay_size"]),
        replay_mode=str(cfg.get("replay_mode", "per")),
        per_alpha=float(cfg.get("per_alpha", 0.6)),
        per_beta_start=float(cfg.get("per_beta_start", 0.4)),
        per_beta_frames=int(cfg.get("per_beta_frames", 200000)),
        per_eps=float(cfg.get("per_eps", 1e-6)),
        exploration_mode="none",
        noise_scale=float(cfg.get("noise_scale", 0.1)),
        epsilon_start=float(cfg.get("epsilon_start", 1.0)),
        epsilon_final=float(cfg.get("epsilon_final", 0.05)),
        epsilon_decay_frames=int(cfg.get("epsilon_decay_frames", 200000)),
        max_grad_norm=float(cfg.get("max_grad_norm", 10.0)),
    )

    agent = DQNAgent(state_dim, action_dim, agent_cfg, device=device)
    agent.load(args.checkpoint)

    metrics = evaluate(
        agent,
        env_name=cfg["env_name"],
        episodes=args.episodes,
        seed=int(cfg.get("seed", 42)) + 555,
        max_steps=int(cfg.get("max_steps_per_episode", 1000)),
    )

    print("Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
