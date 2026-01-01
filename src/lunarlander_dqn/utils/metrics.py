from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque


@dataclass
class RollingMean:
    window: int = 100

    def __post_init__(self):
        self.buf: Deque[float] = deque(maxlen=int(self.window))

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        return self.value()

    def value(self) -> float:
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)
