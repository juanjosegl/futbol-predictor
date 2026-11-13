"""Entrenamiento walk-forward por temporada (nunca split aleatorio).

Para cada temporada de test S (a partir de `min_train_seasons`):
- base LightGBM: temporadas < S-1
- early stopping + calibración: temporada S-1
- predicción: temporada S (nunca vista)
Dixon-Coles se reentrena cada semana con datos anteriores (models/dixon_coles.py).
El resultado es un conjunto de predicciones 100% fuera de muestra para evaluar y hacer backtest.
"""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from ..config import get_path, league_codes
from ..features.build import load_dataset
from ..features.engineering import feature_columns
from .dixon_coles import DC_COLS, dc_walk_forward
from .lgbm import CalibratedLGBM, prepare_X
from .reglog import fit_reglogs, predict_reglogs

log = logging.getLogger(__name__)

KEEP_COLS = ["match_id", "league", "season", "date", "home_team", "away_team", "fthg", "ftag",
             "y_1x2", "y_over25", "close_h", "close_d", "close_a", "close_over", "close_under",
             "open_h", "open_d", "open_a", "open_over", "open_under", "close_src", "open_src",
             "mkt_close_h", "mkt_close_d", "mkt_close_a", "mkt_close_over",
             "mkt_open_h", "mkt_open_d", "mkt_open_a", "mkt_open_over", "odds_maxh", "odds_maxd", "odds_maxa"]


def with_dc(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = dc_walk_forward(df, cfg["dixon_coles"])
    return pd.concat([df.drop(columns=[c for c in DC_COLS if c in df.columns]), dc], axis=1)


def _fit_pair(base_tr, cal, feats, cfg, leagues):
    mcfg = cfg["model"]
    use = [f for f in feats if base_tr[f].notna().any()]
    Xb, Xc = prepare_X(base_tr, use, leagues), prepare_X(cal, use, leagues)
    m1 = CalibratedLGBM(mcfg["lgbm_params"], mcfg["calibration"], mcfg["early_stopping_rounds"], "multiclass")
    m1.fit(Xb, base_tr["y_1x2"], Xc, cal["y_1x2"])
    m2 = CalibratedLGBM(mcfg["lgbm_params"], mcfg["calibration"], mcfg["early_stopping_rounds"], "binary")
    m2.fit(Xb, base_tr["y_over25"], Xc, cal["y_over25"])
    return m1, m2, use


def _predict_pair(m1, m2, use, df, leagues) -> pd.DataFrame:
    X = prepare_X(df, use, leagues)
    p, pr = m1.predict_proba(X), m1.predict_proba_raw(X)
    q, qr = m2.predict_proba(X)[:, 1], m2.predict_proba_raw(X)[:, 1]
    return pd.DataFrame({
        "lgbm_ph": p[:, 0], "lgbm_pd": p[:, 1], "lgbm_pa": p[:, 2], "lgbm_pover": q,
        "lgbmraw_ph": pr[:, 0], "lgbmraw_pd": pr[:, 1], "lgbmraw_pa": pr[:, 2], "lgbmraw_pover": qr,
    }, index=df.index)


def _reglog_c(cfg: dict) -> float:
    return float(cfg["model"].get("reglog_C", 0.05))


def walk_forward(cfg: dict) -> pd.DataFrame:
    df = with_dc(load_dataset(cfg), cfg)
    extra = DC_COLS if cfg["features"].get("use_dc_probs", True) else []
    feats = feature_columns(df, cfg, extra)
    leagues = league_codes(cfg)
    seasons = sorted(df["season"].unique())
    k = int(cfg["model"]["min_train_seasons"])
    if len(seasons) <= k:
        raise ValueError(f"Se necesitan más de {k} temporadas para walk-forward (hay {len(seasons)}).")
    played = df[df["played"]]
    preds, folds, importances = [], [], []
    for S in seasons[k:]:
        train = played[played["season"] < S]
        cal_s = train["season"].max()
        base_tr, cal = train[train["season"] < cal_s], train[train["season"] == cal_s]
        test = played[played["season"] == S]
        if test.empty:
            continue
        m1, m2, use = _fit_pair(base_tr, cal, feats, cfg, leagues)
        p = _predict_pair(m1, m2, use, test, leagues)
        rlm = fit_reglogs(train, feats, C=_reglog_c(cfg))  # sin calibración aparte: usa todo < S
        r = predict_reglogs(rlm, test)
        part = pd.concat([test[[c for c in KEEP_COLS if c in test.columns] + DC_COLS], p, r], axis=1)
        part["fold_train_seasons"] = f"{base_tr['season'].min()}-{base_tr['season'].max()}"
        part["fold_cal_season"] = cal_s
        preds.append(part)
        folds.append({"test_season": S, "train": part["fold_train_seasons"].iloc[0], "calibration": cal_s,
                      "n_train": len(base_tr), "n_cal": len(cal), "n_test": len(test),
                      "best_iter_1x2": int(m1.best_iteration or 0), "best_iter_ou": int(m2.best_iteration or 0),
                      "n_features": len(use)})
        importances.append(m1.feature_importance().rename(S))
        log.info("Fold %s: train=%d cal=%d (%s) test=%d | iters 1x2=%s ou=%s", S, len(base_tr), len(cal),
                 cal_s, len(test), m1.best_iteration, m2.best_iteration)
    oos = pd.concat(preds).sort_values("date").reset_index(drop=True)
    proc, rep = get_path(cfg, "processed"), get_path(cfg, "reports")
    oos.to_parquet(proc / "predictions_oos.parquet", index=False)
    pd.DataFrame(folds).to_csv(rep / "walk_forward_folds.csv", index=False)
    imp = pd.concat(importances, axis=1).fillna(0)
    imp["mean_gain"] = imp.mean(axis=1)
    imp.sort_values("mean_gain", ascending=False).to_csv(rep / "feature_importance.csv")
    log.info("Predicciones fuera de muestra: %d partidos (%s)", len(oos), ", ".join(seasons[k:]))
    train_final(cfg, df, feats, leagues)
    return oos


def train_final(cfg: dict, df: pd.DataFrame, feats: list[str], leagues: list[str]) -> None:
    """Modelo para predecir próximos partidos: base hasta hace 365 días, calibración en el último año."""
    played = df[df["played"]]
    cutoff = played["date"].max() - pd.Timedelta(days=365)
    base_tr, cal = played[played["date"] < cutoff], played[played["date"] >= cutoff]
    m1, m2, use = _fit_pair(base_tr, cal, feats, cfg, leagues)
    rl = fit_reglogs(played, feats, C=_reglog_c(cfg))
    mdir = get_path(cfg, "models")
    joblib.dump({"m1x2": m1, "mou": m2, "features": use, "leagues": leagues, "reglog": rl,
                 "stat_features": feats}, mdir / "lgbm_final.joblib")
    coefs = rl["rlm_1x2"].coefficients().T.round(4)
    coefs.to_csv(get_path(cfg, "reports") / "reglog_mercado_coeficientes.csv")
    rl["rl_1x2"].coefficients().T.round(4).to_csv(get_path(cfg, "reports") / "reglog_stats_coeficientes.csv")
    meta = {"trained_until": str(played["date"].max().date()), "calibration_from": str(cutoff.date()),
            "n_base": len(base_tr), "n_cal": len(cal), "features": use,
            "calibration": cfg["model"]["calibration"]}
    (mdir / "lgbm_final_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Modelo final guardado (datos hasta %s, %d features)", meta["trained_until"], len(use))


def load_oos(cfg: dict) -> pd.DataFrame:
    p = get_path(cfg, "processed") / "predictions_oos.parquet"
    if not p.exists():
        raise FileNotFoundError("No hay predicciones. Ejecuta primero `train`.")
    return pd.read_parquet(p)
