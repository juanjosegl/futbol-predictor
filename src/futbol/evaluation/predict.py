"""Predicción de próximos partidos a partir de un CSV simple.

Formato de entrada (cabecera obligatoria; cuotas y liga opcionales):
    date,home,away[,odds_h,odds_d,odds_a][,league]
    2026-11-08,Arsenal,Chelsea,2.10,3.50,3.60,E0
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import current_season_code, get_path
from ..features.build import load_dataset
from ..features.engineering import build_features
from ..features.odds import add_market_probs
from ..ingest.team_names import TeamNameMapper
from ..models.dixon_coles import DC_COLS, fit_latest
from ..models.lgbm import prepare_X

log = logging.getLogger(__name__)


def _season_for(date: pd.Timestamp) -> str:
    return current_season_code(date.date())


def read_fixtures(path: Path) -> pd.DataFrame:
    fx = pd.read_csv(path)
    fx.columns = [c.strip().lower() for c in fx.columns]
    missing = {"date", "home", "away"} - set(fx.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
    fx["date"] = pd.to_datetime(fx["date"], dayfirst=False, errors="coerce")
    for c in ("odds_h", "odds_d", "odds_a"):
        fx[c] = pd.to_numeric(fx[c], errors="coerce") if c in fx.columns else np.nan
    if "league" not in fx.columns:
        fx["league"] = None
    return fx.dropna(subset=["date"])


def predict(cfg: dict, fixtures_path: Path, out_path: Path | None = None) -> pd.DataFrame:
    hist = load_dataset(cfg)
    bundle = joblib.load(get_path(cfg, "models") / "lgbm_final.joblib")
    fx = read_fixtures(Path(fixtures_path))
    last_season = hist["season"].max()
    current = hist[hist["season"] == last_season]
    by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in current.groupby("league")}
    all_by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in hist.groupby("league")}
    mapper = TeamNameMapper(by_league, threshold=cfg["team_names"]["fuzzy_threshold"])
    mapper_all = TeamNameMapper(all_by_league, threshold=cfg["team_names"]["fuzzy_threshold"])

    rows = []
    for _, r in fx.iterrows():
        lg = r["league"] if isinstance(r["league"], str) and r["league"] in all_by_league else None
        if lg is None:  # inferir la liga a partir del equipo local
            for cand in by_league:
                if mapper.map(r["home"], cand, "fixtures") and mapper.map(r["away"], cand, "fixtures"):
                    lg = cand
                    break
        home = (mapper.map(r["home"], lg, "fixtures") or mapper_all.map(r["home"], lg, "fixtures")) if lg else None
        away = (mapper.map(r["away"], lg, "fixtures") or mapper_all.map(r["away"], lg, "fixtures")) if lg else None
        if not (lg and home and away):
            log.warning("No se pudo emparejar el partido %s - %s (liga=%s); se omite", r["home"], r["away"], r["league"])
            continue
        rows.append({"league": lg, "season": _season_for(r["date"]), "date": r["date"], "home_team": home,
                     "away_team": away, "open_h": r["odds_h"], "open_d": r["odds_d"], "open_a": r["odds_a"],
                     "in_home": r["home"], "in_away": r["away"]})
    if not rows:
        raise ValueError("Ningún partido del CSV pudo emparejarse con equipos conocidos.")
    new = pd.DataFrame(rows)
    for c in ("open_over", "open_under", "close_h", "close_d", "close_a", "close_over", "close_under"):
        new[c] = np.nan
    new = add_market_probs(new, cfg["features"]["margin_method"])

    base_cols = [c for c in hist.columns if not c.startswith(("f_", "y_")) and c not in ("played",)]
    combined = pd.concat([hist[base_cols], new.drop(columns=["in_home", "in_away"])], ignore_index=True)
    feats = build_features(combined, cfg)
    target = feats.iloc[len(hist):].copy()
    target.index = new.index

    # Dixon-Coles con todo lo jugado hasta la fecha del partido
    dc = pd.DataFrame(np.nan, index=target.index, columns=DC_COLS)
    for (lg, d), g in target.groupby(["league", "date"]):
        model = fit_latest(hist, lg, cfg["dixon_coles"], d)
        if model is not None:
            dc.loc[g.index, DC_COLS] = model.predict(g["home_team"], g["away_team"])[DC_COLS].to_numpy()
    target = pd.concat([target, dc], axis=1)

    X = prepare_X(target, bundle["features"], bundle["leagues"])
    p = bundle["m1x2"].predict_proba(X)
    q = bundle["mou"].predict_proba(X)[:, 1]
    thr = float(cfg["backtest"]["threshold"])
    out = pd.DataFrame({
        "date": target["date"].dt.strftime("%Y-%m-%d"), "league": target["league"],
        "home": target["home_team"], "away": target["away_team"],
        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "p_over25": q,
        "fair_home": 1 / p[:, 0], "fair_draw": 1 / p[:, 1], "fair_away": 1 / p[:, 2],
        "dc_p_home": target["dc_ph"], "dc_p_draw": target["dc_pd"], "dc_p_away": target["dc_pa"],
        "odds_home": new["open_h"], "odds_draw": new["open_d"], "odds_away": new["open_a"],
    })
    edges = np.column_stack([out["p_home"] * out["odds_home"], out["p_draw"] * out["odds_draw"],
                             out["p_away"] * out["odds_away"]]) - 1
    out["edge_home"], out["edge_draw"], out["edge_away"] = edges[:, 0], edges[:, 1], edges[:, 2]
    best = np.nanargmax(np.where(np.isfinite(edges), edges, -np.inf), axis=1)
    best_edge = edges[np.arange(len(out)), best]
    has_odds = np.isfinite(edges).any(axis=1)
    out["value_bet"] = np.where(has_odds & (best_edge > thr), np.array(["1", "X", "2"])[best], "")
    o_best = np.column_stack([out["odds_home"], out["odds_draw"], out["odds_away"]])[np.arange(len(out)), best]
    p_best = p[np.arange(len(out)), best]
    kelly = np.clip((p_best * o_best - 1) / (o_best - 1), 0, None) * cfg["backtest"]["kelly_fraction"]
    out["kelly_stake_frac"] = np.where(out["value_bet"] != "", np.minimum(kelly, cfg["backtest"]["kelly_max_fraction"]), 0.0)
    out = out.round(4)
    out_path = Path(out_path) if out_path else get_path(cfg, "reports") / "predictions.csv"
    out.to_csv(out_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log.info("Predicciones (%d partidos) -> %s\n%s", len(out), out_path,
                 out[["date", "league", "home", "away", "p_home", "p_draw", "p_away", "fair_home", "fair_draw",
                      "fair_away", "value_bet"]].to_string(index=False))
    return out
