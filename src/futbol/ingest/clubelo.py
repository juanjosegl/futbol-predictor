"""Rating Elo histórico desde ClubElo (vía soccerdata, con alternativa directa a la API).

Si ClubElo no responde, se registra un warning y el pipeline usa solo el Elo propio que se
calcula a partir de los resultados de football-data (features/engineering.py).
"""
from __future__ import annotations

import datetime as dt
import io
import logging
from pathlib import Path

import pandas as pd

from ..config import get_path, season_codes, season_start_year
from .http import SourceUnavailable, cached_get, import_soccerdata, make_session

log = logging.getLogger(__name__)

API = "http://api.clubelo.com/{q}"


def _dir(cfg: dict) -> Path:
    d = get_path(cfg, "raw") / "clubelo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_api_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"club": "club"})
    for c in ("from", "to"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _get(cfg: dict, q: str, dest: Path, refresh: bool, timeout: float | None = None) -> pd.DataFrame:
    session = make_session(cfg)
    content = cached_get(API.format(q=q.replace(" ", "")), dest, session=session, refresh=refresh,
                         timeout=timeout or cfg["http"]["timeout"], retries=cfg["http"]["retries"],
                         pause=cfg["http"]["pause_seconds"])
    return _parse_api_csv(content)


def _snapshot(cfg: dict, date: dt.date, refresh: bool, timeout: float | None = None) -> pd.DataFrame:
    """Ranking de todos los clubes en una fecha (soccerdata primero, API directa después)."""
    dest = _dir(cfg) / f"snapshot_{date:%Y-%m-%d}.csv"
    if dest.exists() and not refresh:
        return _parse_api_csv(dest.read_bytes())
    try:
        sd = import_soccerdata(get_path(cfg, "raw") / "soccerdata")
        df = sd.ClubElo().read_by_date(date.isoformat()).reset_index()
        df = df.rename(columns={"team": "club"})
        df[["rank", "club", "country", "level", "elo", "from", "to"]].to_csv(dest, index=False)
        return _parse_api_csv(dest.read_bytes())
    except Exception as e:  # noqa: BLE001
        log.debug("soccerdata ClubElo falló (%s); probando API directa", e)
    return _get(cfg, date.isoformat(), dest, refresh, timeout)


def download(cfg: dict, mapper_factory, refresh: bool = False) -> str:
    """Descarga el histórico Elo de cada club de las ligas configuradas.

    `mapper_factory()` devuelve un TeamNameMapper con los nombres canónicos.
    Devuelve 'ok', 'no disponible' o 'desactivado'.
    """
    if not cfg["sources"].get("clubelo", True):
        log.info("ClubElo desactivado en config.")
        return "desactivado"
    seasons = season_codes(cfg)
    first = dt.date(season_start_year(seasons[0]), 8, 1)
    if not (_dir(cfg) / f"snapshot_{first:%Y-%m-%d}.csv").exists() or refresh:
        # comprobación rápida de disponibilidad (sin reintentos largos)
        try:
            r = make_session(cfg).get(API.format(q=first.isoformat()), timeout=15)
            r.raise_for_status()
            if not r.text.startswith("Rank"):
                raise SourceUnavailable("respuesta inesperada")
        except Exception as e:  # noqa: BLE001
            log.warning("ClubElo no disponible (%s). Se usará solo el Elo propio.", str(e)[:200])
            return "no disponible"
    mapper = mapper_factory()
    countries = {lg["clubelo_country"]: lg["code"] for lg in cfg["leagues"] if lg.get("clubelo_country")}
    clubs: dict[str, str] = {}  # nombre ClubElo -> nombre canónico
    for s in seasons:
        try:
            snap = _snapshot(cfg, dt.date(season_start_year(s), 8, 1), refresh)
        except Exception as e:  # noqa: BLE001
            log.warning("ClubElo snapshot %s falló: %s", s, e)
            continue
        snap = snap[snap["country"].isin(countries) & (snap["level"] == 1)]
        for _, r in snap.iterrows():
            canon = mapper.map(r["club"], countries[r["country"]], source="clubelo")
            if canon:
                clubs[r["club"]] = canon
    ok = 0
    for club in sorted(clubs):
        dest = _dir(cfg) / f"team_{club.replace(' ', '')}.csv"
        try:
            _get(cfg, club, dest, refresh)
            ok += 1
        except SourceUnavailable as e:
            log.warning("ClubElo histórico %s falló: %s", club, e)
    pd.Series(clubs, name="canonical").rename_axis("clubelo").to_csv(_dir(cfg) / "name_map.csv")
    log.info("ClubElo: históricos de %d/%d clubes", ok, len(clubs))
    return "ok" if ok else "no disponible"


def load(cfg: dict) -> pd.DataFrame:
    """Devuelve [team (canónico), elo, from, to] o vacío si no hay datos."""
    map_path = _dir(cfg) / "name_map.csv"
    if not map_path.exists():
        return pd.DataFrame(columns=["team", "elo", "from", "to"])
    name_map = pd.read_csv(map_path).set_index("clubelo")["canonical"].to_dict()
    frames = []
    for club, canon in name_map.items():
        p = _dir(cfg) / f"team_{club.replace(' ', '')}.csv"
        if p.exists():
            h = _parse_api_csv(p.read_bytes())
            h["team"] = canon
            frames.append(h[["team", "elo", "from", "to"]])
    if not frames:
        return pd.DataFrame(columns=["team", "elo", "from", "to"])
    return pd.concat(frames, ignore_index=True)


def elo_before(history: pd.DataFrame, matches: pd.DataFrame, team_col: str) -> pd.Series:
    """Elo de ClubElo vigente el día ANTERIOR al partido (sin leakage).

    ClubElo publica un rating por intervalo [from, to]; el rating que cubre el día del partido
    ya puede incluir ese partido, por eso se usa el del día previo.
    """
    if history.empty:
        return pd.Series(float("nan"), index=matches.index)
    h = history.dropna(subset=["from", "elo"]).sort_values("from")[["team", "from", "elo"]]
    h = h.rename(columns={"team": team_col})
    m = matches[[team_col, "date"]].copy()
    m["_lookup"] = m["date"] - pd.Timedelta(days=1)
    m["_idx"] = range(len(m))
    m = m.sort_values("_lookup")
    merged = pd.merge_asof(m, h, left_on="_lookup", right_on="from", by=team_col, direction="backward")
    merged = merged.sort_values("_idx")
    return pd.Series(merged["elo"].values, index=matches.index)
