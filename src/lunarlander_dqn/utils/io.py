from __future__ import annotations

import os
import time
from typing import Any, Dict

import yaml


def get_run_dir(experiments_dir: str, project_name: str, run_name: str = "") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_project = project_name.replace(" ", "_")
    safe_name = run_name.replace(" ", "_").strip()
    folder = f"{safe_project}_{safe_name}_{ts}" if safe_name else f"{safe_project}_{ts}"
    return os.path.join(experiments_dir, folder)


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
