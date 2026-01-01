from __future__ import annotations

import copy
import torch
import torch.nn as nn


@torch.no_grad()
def perturb_model(model: nn.Module, noise_scale: float, device: torch.device) -> nn.Module:
    m = copy.deepcopy(model).to(device)
    for p in m.parameters():
        p.add_(torch.randn_like(p) * float(noise_scale))
    return m
