"""Evaluación fuera de muestra: Dixon-Coles vs LightGBM vs mercado (apertura y cierre)."""
from __future__ import annotations

import json
import logging

import pandas as pd

from ..config import get_path
from ..models.training import load_oos
from . import metrics as M
from .plots import reliability_diagram

log = logging.getLogger(__name__)

MODELS_1X2 = {
    "LightGBM (calibrado)": ["lgbm_ph", "lgbm_pd", "lgbm_pa"],
    "LightGBM (sin calibrar)": ["lgbmraw_ph", "lgbmraw_pd", "lgbmraw_pa"],
    "Dixon-Coles": ["dc_ph", "dc_pd", "dc_pa"],
    "Regresión logística (stats)": ["rl_ph", "rl_pd", "rl_pa"],
    "Regresión logística + mercado": ["rlm_ph", "rlm_pd", "rlm_pa"],
    "Mercado apertura": ["mkt_open_h", "mkt_open_d", "mkt_open_a"],
    "Mercado cierre": ["mkt_close_h", "mkt_close_d", "mkt_close_a"],
}
MODELS_OU = {
    "LightGBM (calibrado)": "lgbm_pover",
    "LightGBM (sin calibrar)": "lgbmraw_pover",
    "Dixon-Coles": "dc_pover",
    "Regresión logística (stats)": "rl_pover",
    "Regresión logística + mercado": "rlm_pover",
    "Mercado apertura": "mkt_open_over",
    "Mercado cierre": "mkt_close_over",
}
BENCH = "Mercado cierre"


def _table_1x2(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y_1x2"].to_numpy()
    rows = []
    ref = M.per_match_logloss(y, df[MODELS_1X2[BENCH]].to_numpy())
    for name, cols in MODELS_1X2.items():
        p = df[cols].to_numpy()
        d, lo, hi = M.paired_bootstrap_diff(M.per_match_logloss(y, p), ref)
        rows.append({"modelo": name, "n": len(df), "log_loss": M.log_loss_multi(y, p),
                     "brier": M.brier_multi(y, p), "accuracy": M.accuracy(y, p),
                     "dif_logloss_vs_cierre": d, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(rows)


def _table_ou(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y_over25"].to_numpy()
    ref = M.per_match_logloss(y, M.binary_to_multi(df[MODELS_OU[BENCH]].to_numpy()))
    rows = []
    for name, col in MODELS_OU.items():
        p = M.binary_to_multi(df[col].to_numpy())
        d, lo, hi = M.paired_bootstrap_diff(M.per_match_logloss(y, p), ref)
        rows.append({"modelo": name, "n": len(df), "log_loss": M.log_loss_multi(y, p),
                     "brier": M.brier_multi(y, p) / 2, "accuracy": M.accuracy(y, p),
                     "dif_logloss_vs_cierre": d, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(rows)


def evaluate(cfg: dict) -> dict:
    oos = load_oos(cfg)
    rep = get_path(cfg, "reports")
    need = [c for cols in MODELS_1X2.values() for c in cols]
    df = oos.dropna(subset=need).copy()
    log.info("Evaluación 1X2 sobre %d partidos fuera de muestra (de %d; se exige que todos los modelos "
             "tengan predicción)", len(df), len(oos))
    t1 = _table_1x2(df)
    dfo = oos.dropna(subset=list(MODELS_OU.values()))
    t2 = _table_ou(dfo)
    by_season = []
    for s, g in df.groupby("season"):
        t = _table_1x2(g)
        t.insert(0, "season", s)
        by_season.append(t)
    by_season = pd.concat(by_season)
    by_league = []
    for lg, g in df.groupby("league"):
        t = _table_1x2(g)
        t.insert(0, "league", lg)
        by_league.append(t)
    by_league = pd.concat(by_league)

    t1.round(5).to_csv(rep / "metrics_1x2.csv", index=False)
    t2.round(5).to_csv(rep / "metrics_ou25.csv", index=False)
    by_season.round(5).to_csv(rep / "metrics_1x2_by_season.csv", index=False)
    by_league.round(5).to_csv(rep / "metrics_1x2_by_league.csv", index=False)

    # Reliability diagrams (3 series: modelo principal, baseline DC, mercado de cierre)
    y = df["y_1x2"].to_numpy()
    show = ["LightGBM (calibrado)", "Dixon-Coles", "Regresión logística + mercado", "Mercado cierre"]
    reliability_diagram(
        {"Local (1)": (y == 0).astype(int), "Empate (X)": (y == 1).astype(int), "Visitante (2)": (y == 2).astype(int)},
        {m: dict(zip(["Local (1)", "Empate (X)", "Visitante (2)"], df[MODELS_1X2[m]].to_numpy().T)) for m in show},
        rep / "calibration_1x2.png", f"Calibración 1X2 fuera de muestra (n={len(df)})")
    reliability_diagram(
        {"Over 2.5": dfo["y_over25"].to_numpy()},
        {m: {"Over 2.5": dfo[MODELS_OU[m]].to_numpy()} for m in show},
        rep / "calibration_ou25.png", f"Calibración Over 2.5 (n={len(dfo)})")

    beats = bool(t1.set_index("modelo").loc["LightGBM (calibrado)", "ic95_sup"] < 0)
    models_only = t1[~t1["modelo"].str.startswith("Mercado")]
    best = models_only.sort_values("log_loss").iloc[0]
    summary = {
        "n_1x2": len(df), "n_ou25": len(dfo), "seasons": sorted(df["season"].unique().tolist()),
        "metrics_1x2": t1.round(5).to_dict(orient="records"),
        "metrics_ou25": t2.round(5).to_dict(orient="records"),
        "lgbm_beats_closing_market_1x2": beats,
        "mejor_modelo_1x2": best["modelo"],
        "mejor_modelo_supera_cierre": bool(best["ic95_sup"] < 0),
        "nota": "dif_logloss_vs_cierre < 0 significa mejor que el mercado de cierre; "
                "IC 95% por bootstrap pareado.",
    }
    (rep / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log.info("Métricas 1X2:\n%s", t1.round(4).to_string(index=False))
        log.info("Métricas Over/Under 2.5:\n%s", t2.round(4).to_string(index=False))
    log.info("Mejor modelo 1X2: %s (log loss %.4f; vs cierre %+.4f, IC95 %+.4f a %+.4f) -> %s", best["modelo"],
             best["log_loss"], best["dif_logloss_vs_cierre"], best["ic95_inf"], best["ic95_sup"],
             "SUPERA al mercado de cierre" if best["ic95_sup"] < 0 else "NO supera al mercado de cierre")
    return summary

