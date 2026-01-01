from __future__ import annotations

import logging
import os


def setup_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("lunarlander_dqn")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
