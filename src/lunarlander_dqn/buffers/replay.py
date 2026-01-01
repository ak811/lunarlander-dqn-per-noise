from __future__ import annotations

from collections import deque
from typing import Deque, Tuple
import random

import numpy as np
import torch

Transition = Tuple[np.ndarray, int, float, np.ndarray, float]


class UniformReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.vstack([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.vstack([b[3] for b in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
        weights = torch.ones_like(rewards, device=device)
        indices = None
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities) -> None:
        return
