"""Logging a consola y a logs/futbol_YYYYMMDD.log."""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> Path:
    global _CONFIGURED
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"futbol_{dt.date.today():%Y%m%d}.log"
    if _CONFIGURED:
        return log_file
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    fileh = logging.FileHandler(log_file, encoding="utf-8")
    fileh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root.addHandler(console)
    root.addHandler(fileh)
    # Silenciar librerías ruidosas
    for noisy in ("urllib3", "matplotlib", "soccerdata", "rich", "PIL", "tls_requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
    return log_file
