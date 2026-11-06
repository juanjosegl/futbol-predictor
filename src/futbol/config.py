"""Carga de configuración y utilidades de temporadas/rutas."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Raíz del proyecto: FUTBOL_HOME, el cwd si contiene config.yaml, o la carpeta del repo."""
    env = os.environ.get("FUTBOL_HOME")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    path = Path(path) if path else root / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(root)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["_root"]) / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def current_season_code(today: dt.date | None = None) -> str:
    """Código football-data de la temporada en curso (las temporadas empiezan en julio)."""
    today = today or dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_start_year(code: str) -> int:
    return 2000 + int(code[:2])


def season_label(code: str) -> str:
    y = season_start_year(code)
    return f"{y}-{(y + 1) % 100:02d}"


def season_codes(cfg: dict) -> list[str]:
    """Lista de códigos de temporada ('1920', '2021', ...) en orden cronológico."""
    last = cfg["seasons"].get("last_season", "auto")
    last = current_season_code() if last in (None, "auto") else str(last)
    end = season_start_year(last)
    n = int(cfg["seasons"]["n_seasons"])
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(end - n + 1, end + 1)]


def league_codes(cfg: dict) -> list[str]:
    return [lg["code"] for lg in cfg["leagues"]]


def league_by_code(cfg: dict) -> dict[str, dict]:
    return {lg["code"]: lg for lg in cfg["leagues"]}
