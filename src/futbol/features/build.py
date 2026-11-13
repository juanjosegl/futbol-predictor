"""Construcción del dataset unificado data/processed/matches.parquet."""
from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from ..config import get_path
from ..ingest import clubelo, football_data, football_data_org, understat
from ..ingest.team_names import TeamNameMapper
from .engineering import add_targets, build_features
from .odds import add_market_probs

log = logging.getLogger(__name__)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def make_match_id(df: pd.DataFrame) -> pd.Series:
    return (df["league"] + "_" + df["date"].dt.strftime("%Y%m%d") + "_"
            + df["home_team"].map(_slug) + "_" + df["away_team"].map(_slug))


def merge_understat(matches: pd.DataFrame, us: pd.DataFrame, mapper: TeamNameMapper) -> pd.DataFrame:
    matches = matches.copy()
    matches["us_hxg"] = np.nan
    matches["us_axg"] = np.nan
    if us.empty:
        return matches
    us = us.copy()
    us["home_team"] = [mapper.map(n, lg, "understat") for n, lg in zip(us["home_team"], us["league"])]
    us["away_team"] = [mapper.map(n, lg, "understat") for n, lg in zip(us["away_team"], us["league"])]
    us = us.dropna(subset=["home_team", "away_team"])
    # cada cruce local-visitante ocurre una vez por temporada y liga
    us = us.drop_duplicates(["league", "season", "home_team", "away_team"])
    m = matches.drop(columns=["us_hxg", "us_axg"]).merge(
        us[["league", "season", "home_team", "away_team", "home_xg", "away_xg"]],
        on=["league", "season", "home_team", "away_team"], how="left")
    m = m.rename(columns={"home_xg": "us_hxg", "away_xg": "us_axg"})
    m.index = matches.index
    return m


def build_dataset(cfg: dict) -> pd.DataFrame:
    matches = football_data.load_all(cfg)
    log.info("football-data: %d partidos", len(matches))
    by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in matches.groupby("league")}
    mapper = TeamNameMapper(by_league, threshold=cfg["team_names"]["fuzzy_threshold"])
    matches["source"] = "football-data.co.uk"

    # Respaldo: partidos jugados que aún no están en football-data.co.uk (p. ej. si está bloqueado)
    if cfg["sources"].get("football_data_org", True):
        extra = football_data_org.missing_played(matches, football_data_org.load(cfg, mapper))
        if not extra.empty:
            extra["source"] = "football-data.org"
            matches = pd.concat([matches, extra], ignore_index=True)
            log.info("football-data.org: +%d partidos recientes (solo resultado, sin tiros ni cuotas); "
                     "último %s", len(extra), extra["date"].max().date())

    # xG: Understat si existe; si no, xG de football-data (disponible desde 2026/27)
    us = understat.load(cfg) if cfg["sources"].get("understat", True) else pd.DataFrame()
    matches = merge_understat(matches, us, mapper)
    matches["hxg"] = matches["us_hxg"].fillna(matches["fd_hxg"])
    matches["axg"] = matches["us_axg"].fillna(matches["fd_axg"])
    matches["xg_source"] = np.where(matches["us_hxg"].notna(), "understat",
                                    np.where(matches["fd_hxg"].notna(), "football-data", "none"))

    # Elo de ClubElo (si se pudo descargar)
    hist = clubelo.load(cfg) if cfg["sources"].get("clubelo", True) else pd.DataFrame()
    if not hist.empty:
        matches["clubelo_h"] = clubelo.elo_before(hist, matches, "home_team")
        matches["clubelo_a"] = clubelo.elo_before(hist, matches, "away_team")
    else:
        log.info("Sin datos de ClubElo: se usa solo el Elo propio.")

    matches = add_market_probs(matches, cfg["features"]["margin_method"])
    matches = matches.sort_values(["date", "league", "home_team"], kind="mergesort").reset_index(drop=True)
    matches["match_id"] = make_match_id(matches)
    dup = matches["match_id"].duplicated()
    if dup.any():
        log.warning("Eliminando %d partidos duplicados", int(dup.sum()))
        matches = matches[~dup].reset_index(drop=True)

    matches = build_features(matches, cfg)
    matches = add_targets(matches)
    out = get_path(cfg, "processed") / "matches.parquet"
    matches.to_parquet(out, index=False)

    rep_dir = get_path(cfg, "reports")
    unmatched = mapper.save_report(rep_dir / "unmatched_teams.csv")
    summary = {
        "n_matches": int(len(matches)),
        "n_played": int(matches["played"].sum()),
        "date_min": str(matches["date"].min().date()),
        "date_max": str(matches["date"].max().date()),
        "by_league_season": {f"{k[0]}_{k[1]}": int(v) for k, v in matches.groupby(["league", "season"]).size().items()},
        "xg_coverage": matches["xg_source"].value_counts().to_dict(),
        "clubelo_coverage": float(matches["f_clubelo_h"].notna().mean()) if "f_clubelo_h" in matches else 0.0,
        "close_odds_source": matches["close_src"].value_counts().to_dict(),
        "match_source": matches["source"].value_counts().to_dict(),
        "n_features": int(sum(c.startswith("f_") for c in matches.columns)),
        "unmatched_team_names": int(len(unmatched)),
    }
    (rep_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Dataset: %d partidos, %d features -> %s", len(matches), summary["n_features"], out)
    log.info("Cobertura xG: %s", summary["xg_coverage"])
    return matches


def load_dataset(cfg: dict) -> pd.DataFrame:
    p = get_path(cfg, "processed") / "matches.parquet"
    if not p.exists():
        raise FileNotFoundError("No existe matches.parquet. Ejecuta primero `build`.")
    return pd.read_parquet(p)
