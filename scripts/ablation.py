from __future__ import annotations

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lunarlander_dqn.training.train_loop import train


def main():
    p = argparse.ArgumentParser(description="Run PER/noise ablations from configs/ablation.yaml.")
    p.add_argument("--base", required=True)
    p.add_argument("--ablation", default="configs/ablation.yaml")
    args = p.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.ablation, "r", encoding="utf-8") as f:
        ab_cfg = yaml.safe_load(f)

    variants = ab_cfg.get("variants", [])
    if not variants:
        raise ValueError("No variants found in ablation
