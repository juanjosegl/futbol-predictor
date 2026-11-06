"""football-data.co.uk: resultados y cuotas históricas (fuente principal y obligatoria)."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import current_season_code, get_path, league_codes, season_codes
from .http import SourceUnavailable, cached_get, make_session

log = logging.getLogger(__name__)

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Alias de columnas antiguas -> nombres actuales
_ALIASES = {
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbAv>2.5": "Avg>2.5", "BbAv<2.5": "Avg<2.5",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
}


def _raw_dir(cfg: dict) -> Path:
    d = get_path(cfg, "raw") / "football_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(cfg: dict, refresh: bool = False, refresh_current: bool = False) -> list[Path]:
    """Descarga los CSV de todas las ligas/temporadas configuradas (con caché)."""
    session = make_session(cfg)
    http = cfg["http"]
    current = current_season_code()
    paths: list[Path] = []
    for code in league_codes(cfg):
        for season in season_codes(cfg):
            dest = _raw_dir(cfg) / f"{code}_{season}.csv"
            force = refresh or (refresh_current and season == current)
            cached = dest.exists() and not force
            try:
                cached_get(
                    SEASON_URL.format(season=season, code=code), dest, session=session,
                    refresh=force, timeout=http["timeout"], retries=http["retries"],
                    pause=http["pause_seconds"],
                )
                paths.append(dest)
                log.info("football-data %s %s %s", code, season, "(caché)" if cached else "descargado")
            except SourceUnavailable as e:
                log.warning("football-data: no disponible %s %s: %s", code, season, e)
    if not paths:
        raise RuntimeError("No se pudo descargar ningún CSV de football-data.co.uk (fuente obligatoria).")
    return paths


def download_fixtures(cfg: dict, refresh: bool = False) -> Path | None:
    """Descarga fixtures.csv (próximos partidos con cuotas) y lo convierte al formato simple."""
    session = make_session(cfg)
    dest = _raw_dir(cfg) / "fixtures.csv"
    try:
        content = cached_get(FIXTURES_URL, dest, session=session, refresh=refresh,
                             timeout=cfg["http"]["timeout"], retries=cfg["http"]["retries"])
    except SourceUnavailable as e:
        log.warning("football-data fixtures no disponible: %s", e)
        return None
    df = _read_csv_bytes(content)
    df = df[df["Div"].isin(league_codes(cfg))]
    out_path = get_path(cfg, "fixtures") / "upcoming.csv"
    if df.empty:
        log.info("fixtures.csv no contiene partidos de las ligas configuradas por ahora.")
        if out_path.exists():
            out_path.unlink()
        return None
    simple = pd.DataFrame({
        "date": _parse_dates(df["Date"]).dt.strftime("%Y-%m-%d"),
        "league": df["Div"].values,
        "home": df["HomeTeam"].values,
        "away": df["AwayTeam"].values,
        "odds_h": pd.to_numeric(df.get("AvgH"), errors="coerce").values,
        "odds_d": pd.to_numeric(df.get("AvgD"), errors="coerce").values,
        "odds_a": pd.to_numeric(df.get("AvgA"), errors="coerce").values,
    })
    simple.to_csv(out_path, index=False)
    log.info("Próximos partidos guardados en %s (%d)", out_path, len(simple))
    return out_path


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.rename(columns=_ALIASES)
    df = df.dropna(how="all")
    if "HomeTeam" in df.columns:
        df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return d.fillna(d2)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _prefer(df: pd.DataFrame, primary: list[str], fallback: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Por fila, usa las cuotas `primary` si están completas; si no, `fallback`."""
    p = pd.concat([_num(df, c) for c in primary], axis=1)
    f = pd.concat([_num(df, c) for c in fallback], axis=1)
    p.columns = f.columns = range(len(primary))
    ok = p.notna().all(axis=1) & (p > 1).all(axis=1)
    out = f.where(~ok, p)
    src = pd.Series(np.where(ok, "pinnacle", "avg"), index=df.index)
    src[~ok & ~(f.notna().all(axis=1))] = "none"
    return out, src


def standardize(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Convierte un CSV crudo de football-data al esquema interno."""
    out = pd.DataFrame(index=df.index)
    out["league"] = league
    out["season"] = season
    out["date"] = _parse_dates(df["Date"])
    out["time"] = df["Time"].astype(str) if "Time" in df.columns else ""
    out["home_team"] = df["HomeTeam"].astype(str).str.strip()
    out["away_team"] = df["AwayTeam"].astype(str).str.strip()
    out["fthg"] = _num(df, "FTHG")
    out["ftag"] = _num(df, "FTAG")
    out["ftr"] = df["FTR"].astype(str).where(df["FTR"].notna(), None) if "FTR" in df.columns else None
    for src, dst in [("HS", "hs"), ("AS", "as_"), ("HST", "hst"), ("AST", "ast"), ("HC", "hc"), ("AC", "ac")]:
        out[dst] = _num(df, src)
    out["fd_hxg"] = _num(df, "HxG")
    out["fd_axg"] = _num(df, "AxG")

    # Cuotas brutas que guardamos por transparencia
    for c in ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "AvgH", "AvgD", "AvgA",
              "AvgCH", "AvgCD", "AvgCA", "MaxH", "MaxD", "MaxA"]:
        out[f"odds_{c.lower()}"] = _num(df, c)

    # Apertura (pre-cierre) y cierre: Pinnacle si existe, si no promedio de mercado
    o, osrc = _prefer(df, ["PSH", "PSD", "PSA"], ["AvgH", "AvgD", "AvgA"])
    c, csrc = _prefer(df, ["PSCH", "PSCD", "PSCA"], ["AvgCH", "AvgCD", "AvgCA"])
    out["open_h"], out["open_d"], out["open_a"] = o[0], o[1], o[2]
    out["close_h"], out["close_d"], out["close_a"] = c[0], c[1], c[2]
    out["open_src"], out["close_src"] = osrc, csrc

    ou_o, _ = _prefer(df, ["P>2.5", "P<2.5"], ["Avg>2.5", "Avg<2.5"])
    ou_c, ousrc = _prefer(df, ["PC>2.5", "PC<2.5"], ["AvgC>2.5", "AvgC<2.5"])
    out["open_over"], out["open_under"] = ou_o[0], ou_o[1]
    out["close_over"], out["close_under"] = ou_c[0], ou_c[1]
    out["close_ou_src"] = ousrc
    out = out[out["date"].notna()]
    return out.reset_index(drop=True)


def load_all(cfg: dict) -> pd.DataFrame:
    """Lee todos los CSV cacheados y devuelve un DataFrame estandarizado."""
    frames = []
    for code in league_codes(cfg):
        for season in season_codes(cfg):
            p = _raw_dir(cfg) / f"{code}_{season}.csv"
            if not p.exists():
                continue
            raw = _read_csv_bytes(p.read_bytes())
            if raw.empty:
                continue
            frames.append(standardize(raw, code, season))
    if not frames:
        raise RuntimeError("No hay CSV de football-data en caché. Ejecuta primero `download`.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    return df
