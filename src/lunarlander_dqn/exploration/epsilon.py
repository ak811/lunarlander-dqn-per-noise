from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    final: float = 0.05
    decay_frames: int = 200_000

    def value(self, frame: int) -> float:
        frame = max(0, int(frame))
        if self.decay_frames <= 0:
            return float(self.final)
        frac = min(1.0, frame / float(self.decay_frames))
        return float(self.start + (self.final - self.start) * frac)
