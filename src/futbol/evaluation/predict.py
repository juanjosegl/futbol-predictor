"""Predicción de próximos partidos a partir de un CSV simple.

Formato de entrada (cabecera obligatoria; cuotas y liga opcionales):
    date,home,away[,odds_h,odds_d,odds_a][,odds_over,odds_under][,league]
    2026-11-08,Arsenal,Chelsea,2.10,3.50,3.60,1.85,2.00,E0

Las cuotas deben ser las de TU casa de apuestas en el momento de predecir: el modelo principal
(`model.main: rlm`) las usa como información (sin margen) y además calcula el valor esperado
de apostar a esa cuota. Sin cuotas se usa el LightGBM como respaldo.
Cada predicción se guarda también en reports/ledger.csv para liquidarla después (`futbol settle`).
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
from ..models.reglog import predict_reglogs
from .informe import confidence, write_report
from .ledger import append_to_ledger

log = logging.getLogger(__name__)

MODEL_LABEL = {"rlm": "RegLog+mercado", "rl": "RegLog stats", "lgbm": "LightGBM"}


def _season_for(date: pd.Timestamp) -> str:
    return current_season_code(date.date())


def read_fixtures(path: Path) -> pd.DataFrame:
    fx = pd.read_csv(path)
    fx.columns = [c.strip().lower() for c in fx.columns]
    missing = {"date", "home", "away"} - set(fx.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
    fx["date"] = pd.to_datetime(fx["date"], dayfirst=False, errors="coerce")
    for c in ("odds_h", "odds_d", "odds_a", "odds_over", "odds_under"):
        fx[c] = pd.to_numeric(fx[c], errors="coerce") if c in fx.columns else np.nan
    if "league" not in fx.columns:
        fx["league"] = None
    return fx.dropna(subset=["date"])


def _kelly(p, o, cfg):
    k = np.clip((p * o - 1) / (o - 1), 0, None) * cfg["backtest"]["kelly_fraction"]
    return np.minimum(k, cfg["backtest"]["kelly_max_fraction"])


def predict(cfg: dict, fixtures_path: Path, out_path: Path | None = None, record: bool = True,
            only_future: bool = False) -> pd.DataFrame:
    """`only_future`: descarta los partidos con fecha anterior a hoy (para listas automáticas que pueden estar viejas)."""
    hist = load_dataset(cfg)
    bundle = joblib.load(get_path(cfg, "models") / "lgbm_final.joblib")
    fx = read_fixtures(Path(fixtures_path))
    if only_future:
        old = fx["date"] < pd.Timestamp.today().normalize()
        if old.any():
            log.info("%d partidos de %s ya pasaron y se omiten.", int(old.sum()), fixtures_path)
        fx = fx[~old]
        if fx.empty:
            raise ValueError(f"{fixtures_path} no tiene partidos de hoy en adelante.")
    last_season = hist["season"].max()
    current = hist[hist["season"] == last_season]
    by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in current.groupby("league")}
    all_by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in hist.groupby("league")}
    mapper = TeamNameMapper(by_league, threshold=cfg["team_names"]["fuzzy_threshold"])
    mapper_all = TeamNameMapper(all_by_league, threshold=cfg["team_names"]["fuzzy_threshold"])

    rows = []
    for _, r in fx.iterrows():
        lg = r["league"] if isinstance(r["league"], str) and r["league"] in all_by_league else None
        if lg is None:  # inferir la liga a partir de los equipos
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
                     "open_over": r["odds_over"], "open_under": r["odds_under"]})
    if not rows:
        raise ValueError("Ningún partido del CSV pudo emparejarse con equipos conocidos.")
    new = pd.DataFrame(rows)
    for c in ("close_h", "close_d", "close_a", "close_over", "close_under"):
        new[c] = np.nan
    new = add_market_probs(new, cfg["features"]["margin_method"])

    base_cols = [c for c in hist.columns if not c.startswith(("f_", "y_")) and c not in ("played",)]
    combined = pd.concat([hist[base_cols], new], ignore_index=True)
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
    probs = {"lgbm": (bundle["m1x2"].predict_proba(X), bundle["mou"].predict_proba(X)[:, 1])}
    if "reglog" in bundle:
        r = predict_reglogs(bundle["reglog"], target)
        probs["rl"] = (r[["rl_ph", "rl_pd", "rl_pa"]].to_numpy(), r["rl_pover"].to_numpy())
        probs["rlm"] = (r[["rlm_ph", "rlm_pd", "rlm_pa"]].to_numpy(), r["rlm_pover"].to_numpy())

    # Modelo principal con respaldo fila a fila (rlm necesita cuotas; si faltan, LightGBM)
    main = cfg["model"].get("main", "rlm")
    main = main if main in probs else "lgbm"
    p, q = probs[main][0].copy(), probs[main][1].copy()
    used = np.full(len(p), MODEL_LABEL[main], dtype=object)
    miss = ~np.isfinite(p).all(axis=1)
    p[miss], used[miss] = probs["lgbm"][0][miss], MODEL_LABEL["lgbm"]
    missq = ~np.isfinite(q)
    q[missq] = probs["lgbm"][1][missq]

    thr = float(cfg["backtest"]["threshold"])
    out = pd.DataFrame({
        "date": target["date"].dt.strftime("%Y-%m-%d"), "league": target["league"],
        "home": target["home_team"], "away": target["away_team"], "modelo": used,
        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "p_over25": q,
        "fair_home": 1 / p[:, 0], "fair_draw": 1 / p[:, 1], "fair_away": 1 / p[:, 2], "fair_over25": 1 / q,
        "mkt_p_home": new["mkt_open_h"], "mkt_p_draw": new["mkt_open_d"], "mkt_p_away": new["mkt_open_a"],
        "lgbm_p_home": probs["lgbm"][0][:, 0], "lgbm_p_draw": probs["lgbm"][0][:, 1],
        "lgbm_p_away": probs["lgbm"][0][:, 2],
        "dc_p_home": target["dc_ph"], "dc_p_draw": target["dc_pd"], "dc_p_away": target["dc_pa"],
        "odds_home": new["open_h"], "odds_draw": new["open_d"], "odds_away": new["open_a"],
        "odds_over": new["open_over"], "odds_under": new["open_under"],
    })
    odds = out[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    # Pronóstico = resultado más probable; confianza según esa probabilidad (config tracking.*)
    i = np.arange(len(out))
    top = p.argmax(axis=1)
    out.insert(5, "pick", np.array(["1", "X", "2"])[top])
    out.insert(6, "p_pick", p[i, top])
    out.insert(7, "confianza", confidence(p[i, top], cfg))
    out.insert(8, "cuota_pick", odds[i, top])
    out.insert(9, "pick_ou", np.where(q > 0.5, "Over", "Under"))
    out.insert(10, "p_pick_ou", np.where(q > 0.5, q, 1 - q))
    edges = p * odds - 1
    out["edge_home"], out["edge_draw"], out["edge_away"] = edges[:, 0], edges[:, 1], edges[:, 2]
    safe = np.where(np.isfinite(edges), edges, -np.inf)
    best = np.argmax(safe, axis=1)
    best_edge = safe[i, best]
    out["value_bet"] = np.where(best_edge > thr, np.array(["1", "X", "2"])[best], "")
    out["kelly_stake_frac"] = np.where(out["value_bet"] != "", _kelly(p[i, best], odds[i, best], cfg), 0.0)

    ou_odds = out[["odds_over", "odds_under"]].to_numpy(float)
    ou_p = np.column_stack([q, 1 - q])
    ou_edges = ou_p * ou_odds - 1
    out["edge_over"], out["edge_under"] = ou_edges[:, 0], ou_edges[:, 1]
    ou_safe = np.where(np.isfinite(ou_edges), ou_edges, -np.inf)
    ob = np.argmax(ou_safe, axis=1)
    out["value_bet_ou"] = np.where(ou_safe[i, ob] > thr, np.array(["Over", "Under"])[ob], "")
    out["kelly_stake_frac_ou"] = np.where(out["value_bet_ou"] != "", _kelly(ou_p[i, ob], ou_odds[i, ob], cfg), 0.0)

    out = out.round(4)
    out_path = Path(out_path) if out_path else get_path(cfg, "reports") / "predictions.csv"
    out.to_csv(out_path, index=False)
    if record:
        write_report(cfg, append_to_ledger(cfg, out))
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        log.info("Predicciones (%d partidos, ordenadas de más a menos segura) -> %s\n%s", len(out), out_path,
                 out.sort_values("p_pick", ascending=False)[
                     ["date", "league", "home", "away", "pick", "p_pick", "confianza", "cuota_pick", "pick_ou",
                      "p_pick_ou", "value_bet", "value_bet_ou", "modelo"]].to_string(index=False))
    n_val = int((out["value_bet"] != "").sum() + (out["value_bet_ou"] != "").sum())
    log.info("%d pronósticos de confianza alta. %d apuestas con valor esperado > %.0f%% según el modelo. "
             "Ojo: 'más probable' no es 'rentable'; la cuota justa mínima de cada pronóstico es 1/p_pick.",
             int((out["confianza"] == "Alta").sum()), n_val, thr * 100)
    return out
