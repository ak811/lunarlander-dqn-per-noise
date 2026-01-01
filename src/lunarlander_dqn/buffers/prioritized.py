from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch

Transition = Tuple[np.ndarray, int, float, np.ndarray, float]


class PrioritizedReplayBuffer:
    """Simple PER buffer with array priorities."""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        max_p = float(self.priorities.max()) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4):
        assert len(self.buffer) > 0, "Cannot sample from an empty buffer."
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[: len(self.buffer)]
        scaled = prios ** self.alpha
        probs = scaled / (scaled.sum() + 1e-12)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-12)

        states = torch.tensor(np.vstack([s[0] for s in samples]), dtype=torch.float32, device=device)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.vstack([s[3] for s in samples]), dtype=torch.float32, device=device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32, device=device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, weights_t, indices

    def update_priorities(self, indices, priorities) -> None:
        for idx, p in zip(indices, priorities):
            self.priorities[int(idx)] = float(p) + self.eps
