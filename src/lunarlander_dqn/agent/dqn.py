from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lunarlander_dqn.agent.networks import QNetwork
from lunarlander_dqn.buffers.prioritized import PrioritizedReplayBuffer
from lunarlander_dqn.buffers.replay import UniformReplayBuffer
from lunarlander_dqn.exploration.param_noise import perturb_model
from lunarlander_dqn.exploration.epsilon import EpsilonSchedule


@dataclass
class DQNConfig:
    gamma: float
    learning_rate: float
    batch_size: int
    train_every_steps: int
    target_update_every_episodes: int
    replay_capacity: int
    min_replay_size: int

    replay_mode: str  # "per" | "uniform" | "online"

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200000
    per_eps: float = 1e-6

    exploration_mode: str = "param_noise"  # "param_noise" | "epsilon_greedy" | "none"
    noise_scale: float = 0.1

    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_frames: int = 200000

    max_grad_norm: float = 10.0


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig, device: torch.device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device

        self.online = QNetwork(state_dim, action_dim).to(device)
        self.target = QNetwork(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # per-sample huber

        if cfg.replay_mode == "per":
            self.buffer = PrioritizedReplayBuffer(cfg.replay_capacity, alpha=cfg.per_alpha, eps=cfg.per_eps)
        elif cfg.replay_mode == "uniform":
            self.buffer = UniformReplayBuffer(cfg.replay_capacity)
        else:
            self.buffer = None

        self.eps_sched = EpsilonSchedule(cfg.epsilon_start, cfg.epsilon_final, cfg.epsilon_decay_frames)
        self.total_steps = 0

    @torch.no_grad()
    def q_values(self, state: np.ndarray, model: Optional[nn.Module] = None) -> torch.Tensor:
        m = model if model is not None else self.online
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return m(s).squeeze(0)

    def make_exploration_model(self) -> Optional[nn.Module]:
        if self.cfg.exploration_mode == "param_noise":
            return perturb_model(self.online, noise_scale=self.cfg.noise_scale, device=self.device)
        return None

    def select_action(self, state: np.ndarray, exploration_model: Optional[nn.Module] = None) -> int:
        mode = self.cfg.exploration_mode

        if mode == "epsilon_greedy":
            eps = self.eps_sched.value(self.total_steps)
            if np.random.rand() < eps:
                return int(np.random.randint(self.action_dim))
            q = self.q_values(state)
            return int(torch.argmax(q).item())

        if mode == "param_noise" and exploration_model is not None:
            q = self.q_values(state, model=exploration_model)
            return int(torch.argmax(q).item())

        q = self.q_values(state)
        return int(torch.argmax(q).item())

    def store(self, transition: Tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        if self.buffer is not None:
            self.buffer.add(transition)

    def _td_targets(self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1)[0]
            return rewards + self.cfg.gamma * next_q * (1.0 - dones)

    def train_step(self) -> Dict[str, float]:
        if self.buffer is None:
            return {"loss": 0.0}
        if len(self.buffer) < self.cfg.min_replay_size:
            return {"loss": 0.0}

        beta = None
        if self.cfg.replay_mode == "per":
            beta = min(
                1.0,
                self.cfg.per_beta_start
                + self.total_steps * (1.0 - self.cfg.per_beta_start) / max(1, self.cfg.per_beta_frames),
            )
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
                self.cfg.batch_size, device=self.device, beta=beta
            )
        else:
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
                self.cfg.batch_size, device=self.device
            )

        q = self.online(states).gather(1, actions).squeeze(1)
        targets = self._td_targets(rewards, next_states, dones)
        per_sample = self.loss_fn(q, targets)
        loss = (weights * per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.cfg.max_grad_norm)
        self.optimizer.step()

        if self.cfg.replay_mode == "per" and indices is not None:
            td_err = (q - targets).detach().abs().cpu().numpy()
            self.buffer.update_priorities(indices, td_err + self.cfg.per_eps)

        return {"loss": float(loss.item()), "beta": float(beta) if beta is not None else float("nan")}

    def online_update(self, state, action, reward, next_state, done) -> Dict[str, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ns = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor([[action]], dtype=torch.long, device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        d = torch.tensor([done], dtype=torch.float32, device=self.device)

        q = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=1)[0]
            target = r + self.cfg.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(q, target).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.cfg.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.online.to(self.device)
        self.online.eval()
        self.update_target()
