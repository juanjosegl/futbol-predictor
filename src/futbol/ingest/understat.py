"""xG por partido desde Understat.

Orden de intentos por liga/temporada:
1. Caché local (data/raw/understat/{liga}_{temporada}.json)
2. Librería `soccerdata` (Understat.read_schedule)
3. Endpoint JSON que usa la propia web (getLeagueData) o el formato antiguo embebido en el HTML

Si Understat no responde, se registra un warning y el pipeline sigue (los xG quedan vacíos, y
se usan los xG de football-data cuando estén disponibles).
"""
from __future__ import annotations

import codecs
import json
import logging
import re
from pathlib import Path

import pandas as pd

from ..config import get_path, season_codes, season_start_year
from .http import SourceUnavailable, import_soccerdata, make_session

log = logging.getLogger(__name__)

COLS = ["league", "season", "date", "home_team", "away_team", "home_xg", "away_xg"]


def _cache_dir(cfg: dict) -> Path:
    d = get_path(cfg, "raw") / "understat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _from_soccerdata(cfg: dict, lg: dict, season: str) -> list[dict]:
    sd = import_soccerdata(get_path(cfg, "raw") / "soccerdata")
    us = sd.Understat(leagues=lg["soccerdata"], seasons=season)
    sched = us.read_schedule().reset_index()
    sched = sched[sched["home_xg"].notna()]
    return [
        {"date": str(pd.Timestamp(r["date"]).date()), "home": r["home_team"], "away": r["away_team"],
         "home_xg": float(r["home_xg"]), "away_xg": float(r["away_xg"])}
        for _, r in sched.iterrows()
    ]


def _from_direct(cfg: dict, lg: dict, season: str) -> list[dict]:
    session = make_session(cfg)
    year = season_start_year(season)
    league_url = f"https://understat.com/league/{lg['understat']}/{year}"
    timeout = cfg["http"]["timeout"]
    dates = None
    r = session.get(f"https://understat.com/getLeagueData/{lg['understat']}/{year}", timeout=timeout,
                    headers={"X-Requested-With": "XMLHttpRequest", "Referer": league_url})
    if r.ok and r.headers.get("content-type", "").startswith("application/json"):
        dates = r.json().get("dates")
    if dates is None:  # formato antiguo: datos embebidos en el HTML
        r = session.get(league_url, timeout=timeout)
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", r.text) if r.ok else None
        if not m:
            raise SourceUnavailable(f"Understat sin datos para {lg['understat']} {year} (HTTP {r.status_code})")
        dates = json.loads(codecs.decode(m.group(1), "unicode_escape"))
    out = []
    for d in dates:
        if not d.get("isResult"):
            continue
        out.append({"date": d["datetime"][:10], "home": d["h"]["title"], "away": d["a"]["title"],
                    "home_xg": float(d["xG"]["h"]), "away_xg": float(d["xG"]["a"])})
    return out


def download(cfg: dict, refresh: bool = False) -> str:
    """Devuelve 'ok', 'parcial', 'no disponible' o 'desactivado'."""
    if not cfg["sources"].get("understat", True):
        log.info("Understat desactivado en config.")
        return "desactivado"
    site_down = False
    for lg in cfg["leagues"]:
        if not lg.get("understat"):
            continue
        for season in season_codes(cfg):
            dest = _cache_dir(cfg) / f"{lg['code']}_{season}.json"
            # la temporada en curso se considera incompleta: solo se refresca con --refresh
            if dest.exists() and not refresh:
                continue
            if site_down:
                continue
            rows, err = None, None
            for fn in (_from_soccerdata, _from_direct):
                try:
                    rows = fn(cfg, lg, season)
                    if rows:
                        break
                except Exception as e:  # noqa: BLE001
                    err = e
                    log.debug("Understat %s %s vía %s falló: %s", lg["code"], season, fn.__name__, e)
            if rows:
                dest.write_text(json.dumps(rows), encoding="utf-8")
                log.info("Understat %s %s: %d partidos", lg["code"], season, len(rows))
            else:
                log.warning("Understat no disponible (%s %s): %s. Se continúa sin esa fuente.",
                            lg["code"], season, str(err)[:200] if err else "sin datos")
                site_down = True  # cortocircuito: no insistir con un sitio caído
    n_files = len(list(_cache_dir(cfg).glob("*.json")))
    if n_files == 0:
        return "no disponible"
    return "parcial" if site_down else "ok"


def load(cfg: dict) -> pd.DataFrame:
    rows = []
    for lg in cfg["leagues"]:
        for season in season_codes(cfg):
            p = _cache_dir(cfg) / f"{lg['code']}_{season}.json"
            if not p.exists():
                continue
            for r in json.loads(p.read_text(encoding="utf-8")):
                rows.append({"league": lg["code"], "season": season, "date": pd.Timestamp(r["date"]),
                             "home_team": r["home"], "away_team": r["away"],
                             "home_xg": r["home_xg"], "away_xg": r["away_xg"]})
    return pd.DataFrame(rows, columns=COLS)
