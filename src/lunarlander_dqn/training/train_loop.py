from __future__ import annotations

import os
import time
from typing import Any, Dict

import numpy as np
import torch
from tqdm import trange

from lunarlander_dqn.agent.dqn import DQNAgent, DQNConfig
from lunarlander_dqn.envs.lunarlander import make_env, record_episode_video
from lunarlander_dqn.training.evaluate import evaluate
from lunarlander_dqn.utils.io import get_run_dir, dump_config
from lunarlander_dqn.utils.logging import setup_logging
from lunarlander_dqn.utils.metrics import RollingMean
from lunarlander_dqn.utils.plotting import plot_rewards


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def train(cfg: Dict[str, Any]) -> str:
    device = _resolve_device(str(cfg.get("device", "auto")))

    run_dir = get_run_dir(
        experiments_dir=str(cfg.get("experiments_dir", "experiments")),
        project_name=str(cfg.get("project_name", "lunarlander-dqn-per-noise")),
        run_name=str(cfg.get("run_name", "")),
    )

    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    videos_dir = os.path.join(run_dir, "videos")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    logger = setup_logging(os.path.join(logs_dir, "train.log"))

    env = make_env(cfg["env_name"], seed=int(cfg.get("seed", 42)), render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

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
        exploration_mode=str(cfg.get("exploration_mode", "param_noise")),
        noise_scale=float(cfg.get("noise_scale", 0.1)),
        epsilon_start=float(cfg.get("epsilon_start", 1.0)),
        epsilon_final=float(cfg.get("epsilon_final", 0.05)),
        epsilon_decay_frames=int(cfg.get("epsilon_decay_frames", 200000)),
        max_grad_norm=float(cfg.get("max_grad_norm", 10.0)),
    )

    agent = DQNAgent(state_dim, action_dim, agent_cfg, device=device)

    num_episodes = int(cfg["num_episodes"])
    max_steps = int(cfg.get("max_steps_per_episode", 1000))
    save_best = bool(cfg.get("save_best", True))
    save_every = int(cfg.get("save_every_episodes", 100))
    eval_every = int(cfg.get("eval_every_episodes", 50))
    eval_eps = int(cfg.get("eval_episodes", 10))

    record_on_best = bool(cfg.get("record_video_on_best", True))
    record_every_mult = int(cfg.get("record_video_every_best_multiple", 50))

    best_reward = float("-inf")
    rolling = RollingMean(window=100)

    episodes, rewards, rolling_rewards = [], [], []
    start_time = time.time()

    for ep in trange(1, num_episodes + 1, desc="Training"):
        s, _ = env.reset()
        done = False
        total = 0.0
        steps = 0

        exploration_model = agent.make_exploration_model()
        ep_losses = []

        while not done and steps < max_steps:
            a = agent.select_action(s, exploration_model=exploration_model)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            done_f = float(done)

            if agent.cfg.replay_mode == "online":
                info = agent.online_update(s, a, float(r), ns, done_f)
                ep_losses.append(info["loss"])
            else:
                agent.store((s, a, float(r), ns, done_f))
                if agent.total_steps % agent.cfg.train_every_steps == 0:
                    info = agent.train_step()
                    if info.get("loss", 0.0) != 0.0:
                        ep_losses.append(info["loss"])

            s = ns
            total += float(r)
            steps += 1
            agent.total_steps += 1

        if ep % agent.cfg.target_update_every_episodes == 0:
            agent.update_target()

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        episodes.append(ep)
        rewards.append(total)
        rolling_rewards.append(rolling.update(total))

        elapsed = time.time() - start_time
        logger.info(
            f"Episode {ep:4d} | reward={total:8.2f} | steps={steps:4d} | loss={mean_loss:8.4f} | "
            f"rolling100={rolling_rewards[-1]:7.2f} | elapsed={elapsed:7.1f}s"
        )

        if save_best and total > best_reward:
            best_reward = total
            best_path = os.path.join(models_dir, "best_model.pth")
            agent.save(best_path)
            logger.info(f"*** New best reward: {best_reward:.2f} (saved {best_path})")

            if record_on_best and (ep % max(1, record_every_mult) == 0):
                record_episode_video(
                    env_name=cfg["env_name"],
                    video_dir=videos_dir,
                    name_prefix=f"best_ep{ep}_reward{best_reward:.1f}",
                    policy_fn=lambda obs: agent.select_action(obs, exploration_model=None),
                    seed=int(cfg.get("seed", 42)) + 777,
                    max_steps=max_steps,
                )
                logger.info(f"Recorded best-policy video into {videos_dir}")

            plot_rewards(
                episodes, rewards, rolling_rewards,
                out_path=os.path.join(plots_dir, "best_rewards.png"),
                title=f"Best-so-far reward curve (best={best_reward:.1f})",
            )

        if ep % eval_every == 0:
            metrics = evaluate(
                agent,
                env_name=cfg["env_name"],
                episodes=eval_eps,
                seed=int(cfg.get("seed", 42)) + 999,
                max_steps=max_steps,
            )
            logger.info(f"[EVAL] mean={metrics['eval_mean_reward']:.2f} std={metrics['eval_std_reward']:.2f}")
            plot_rewards(
                episodes, rewards, rolling_rewards,
                out_path=os.path.join(plots_dir, "rewards.png"),
                title="Training reward curve",
            )

        if ep % save_every == 0:
            ckpt_path = os.path.join(models_dir, f"checkpoint_ep{ep}.pth")
            agent.save(ckpt_path)

    agent.save(os.path.join(models_dir, "final_model.pth"))
    plot_rewards(episodes, rewards, rolling_rewards, os.path.join(plots_dir, "rewards.png"), "Training reward curve")

    env.close()
    return run_dir
