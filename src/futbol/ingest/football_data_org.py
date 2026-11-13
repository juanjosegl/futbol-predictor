"""Respaldo: resultados y calendario desde la API de football-data.org (v4, plan gratuito).

Sirve cuando football-data.co.uk no es accesible (p. ej. bloqueado por el proveedor de internet):
- Añade al dataset los partidos JUGADOS de la temporada en curso que falten en la caché de
  football-data.co.uk (solo resultado: sin tiros ni cuotas). Así el Elo, la forma y Dixon-Coles
  quedan al día.
- Genera data/fixtures/upcoming.csv con los próximos partidos (sin cuotas: añade las de tu casa).

Necesita un token gratuito (https://www.football-data.org/client/register) en la variable de
entorno FOOTBALL_DATA_ORG_TOKEN (o en un archivo `.env` en la raíz del proyecto). Sin token se omite.
Plan gratuito: 10 peticiones/minuto -> pausa de 6.5 s entre ligas.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from ..config import current_season_code, get_path, season_start_year
from .http import make_session

log = logging.getLogger(__name__)

API = "https://api.football-data.org/v4/competitions/{comp}/matches?season={year}"
COLS = ["league", "season", "date", "home_team", "away_team", "fthg", "ftag", "status"]


def get_token(cfg: dict) -> str | None:
    tok = os.environ.get("FOOTBALL_DATA_ORG_TOKEN")
    if tok:
        return tok.strip()
    env = Path(cfg["_root"]) / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("FOOTBALL_DATA_ORG_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _dir(cfg: dict) -> Path:
    d = get_path(cfg, "raw") / "football_data_org"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(cfg: dict, refresh: bool = True) -> str:
    """Descarga la temporada en curso de cada liga con código `fdorg`. Devuelve un estado."""
    if not cfg["sources"].get("football_data_org", True):
        return "desactivado"
    token = get_token(cfg)
    if not token:
        log.info("football-data.org: sin token (FOOTBALL_DATA_ORG_TOKEN); se omite el respaldo.")
        return "sin token"
    session = make_session(cfg)
    session.headers["X-Auth-Token"] = token
    season = current_season_code()
    year = season_start_year(season)
    ok = 0
    for i, lg in enumerate(cfg["leagues"]):
        comp = lg.get("fdorg")
        if not comp:
            continue
        dest = _dir(cfg) / f"{lg['code']}_{season}.json"
        if dest.exists() and not refresh:
            ok += 1
            continue
        if i:
            time.sleep(6.5)
        try:
            r = session.get(API.format(comp=comp, year=year), timeout=cfg["http"]["timeout"])
            if r.status_code in (401, 403):
                log.warning("football-data.org rechazó el token (HTTP %s): %s", r.status_code, r.text[:150])
                return "token inválido"
            r.raise_for_status()
            dest.write_text(json.dumps(r.json()), encoding="utf-8")
            ok += 1
            log.info("football-data.org %s %s: %d partidos", lg["code"], season, len(r.json().get("matches", [])))
        except Exception as e:  # noqa: BLE001
            log.warning("football-data.org %s falló: %s", lg["code"], str(e)[:200])
    return "ok" if ok else "no disponible"


def parse(payload: dict, league: str, season: str) -> pd.DataFrame:
    rows = []
    for m in payload.get("matches", []):
        ft = (m.get("score") or {}).get("fullTime") or {}
        date = pd.Timestamp(m["utcDate"]).tz_convert("Europe/London").tz_localize(None).normalize()
        for side in ("homeTeam", "awayTeam"):
            m[side] = m.get(side) or {}
        rows.append({"league": league, "season": season, "date": date,
                     "home_names": [m["homeTeam"].get("shortName"), m["homeTeam"].get("name")],
                     "away_names": [m["awayTeam"].get("shortName"), m["awayTeam"].get("name")],
                     "fthg": ft.get("home"), "ftag": ft.get("away"), "status": m.get("status")})
    return pd.DataFrame(rows)


def _map(mapper, names, league):
    for n in names:
        if n:
            t = mapper.map(n, league, "football-data.org")
            if t:
                return t
    return None


def load(cfg: dict, mapper) -> pd.DataFrame:
    """Partidos (jugados y por jugar) de la caché, con nombres canónicos de football-data.co.uk."""
    frames = []
    season = current_season_code()
    for lg in cfg["leagues"]:
        p = _dir(cfg) / f"{lg['code']}_{season}.json"
        if p.exists():
            frames.append(parse(json.loads(p.read_text(encoding="utf-8")), lg["code"], season))
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    df["home_team"] = [_map(mapper, n, lg) for n, lg in zip(df["home_names"], df["league"])]
    df["away_team"] = [_map(mapper, n, lg) for n, lg in zip(df["away_names"], df["league"])]
    bad = df["home_team"].isna() | df["away_team"].isna()
    if bad.any():
        log.warning("football-data.org: %d partidos con equipos sin emparejar (se omiten)", int(bad.sum()))
    return df[~bad][COLS].reset_index(drop=True)


def missing_played(matches: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Partidos jugados de `extra` que no están en `matches` (cada cruce local-visitante es único por temporada)."""
    if extra.empty:
        return extra
    played = extra[(extra["status"] == "FINISHED") & extra["fthg"].notna() & extra["ftag"].notna()]
    have = set(zip(matches["league"], matches["season"], matches["home_team"], matches["away_team"]))
    keep = [(lg, s, h, a) not in have for lg, s, h, a in
            zip(played["league"], played["season"], played["home_team"], played["away_team"])]
    out = played[keep].drop(columns="status").copy()
    out["fthg"], out["ftag"] = out["fthg"].astype(float), out["ftag"].astype(float)
    return out


def write_upcoming(cfg: dict, extra: pd.DataFrame, days: int = 10) -> Path | None:
    """data/fixtures/upcoming.csv con los partidos de los próximos `days` días (cuotas vacías)."""
    if extra.empty:
        return None
    today = pd.Timestamp(dt.date.today())
    up = extra[extra["status"].isin(["SCHEDULED", "TIMED"]) & (extra["date"] >= today)
               & (extra["date"] <= today + pd.Timedelta(days=days))]
    if up.empty:
        return None
    out = pd.DataFrame({"date": up["date"].dt.strftime("%Y-%m-%d"), "league": up["league"],
                        "home": up["home_team"], "away": up["away_team"],
                        "odds_h": "", "odds_d": "", "odds_a": "", "odds_over": "", "odds_under": ""})
    path = get_path(cfg, "fixtures") / "upcoming.csv"
    odds_cols = ["odds_h", "odds_d", "odds_a", "odds_over", "odds_under"]
    if path.exists():  # conserva las cuotas que ya hubieras escrito para esos partidos
        try:
            prev = pd.read_csv(path).drop_duplicates(["league", "home", "away"])
            prev = prev[["league", "home", "away"] + [c for c in odds_cols if c in prev.columns]]
            out = out.drop(columns=[c for c in odds_cols if c in prev.columns]).merge(
                prev, on=["league", "home", "away"], how="left")
            out = out[["date", "league", "home", "away"] + odds_cols]
        except Exception as e:  # noqa: BLE001
            log.warning("No se pudieron conservar las cuotas de %s: %s", path, e)
    out.sort_values(["date", "league"]).to_csv(path, index=False)
    log.info("Próximos partidos (football-data.org) -> %s (%d). Rellena las cuotas de tu casa de apuestas.",
             path, len(out))
    return path
