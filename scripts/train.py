from __future__ import annotations

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lunarlander_dqn.training.train_loop import train


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        k, v = item.split("=", 1)
        v_strip = v.strip()
        if v_strip.lower() in ("true", "false"):
            vv = v_strip.lower() == "true"
        else:
            try:
                if "." in v_strip or "e" in v_strip.lower():
                    vv = float(v_strip)
                else:
                    vv = int(v_strip)
            except Exception:
                vv = v_strip
        cfg[k] = vv
    return cfg


def main():
    p = argparse.ArgumentParser(description="Train DQN on LunarLander with PER + exploration options.")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--set", nargs="*", default=[], help="Overrides like key=value")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_overrides(cfg, args.set)
    run_dir = train(cfg)
    print(f"Run complete. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
