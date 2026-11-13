"""Regresión logística (multinomial para 1X2, binaria para Over/Under 2.5).

Dos variantes:
- `rl`  (estadísticas): las mismas features pre-partido que el LightGBM + probabilidades Dixon-Coles.
        Es el modelo "de clase": muestra cuánto se puede predecir solo con estadísticas.
- `rlm` (anclada al mercado): log-probabilidades de APERTURA sin margen + Elo + Dixon-Coles.
        En los experimentos walk-forward es el modelo más preciso (empata con el mercado de apertura).
        Necesita cuotas del partido; si faltan, su predicción es NaN.

Las cuotas de CIERRE nunca entran (información posterior al momento de apostar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MKT_1X2 = ["mk_lo_h", "mk_lo_d", "mk_lo_a"]
MKT_OU = ["mk_lo_over"]
RLM_EXTRA_1X2 = ["f_elo_diff", "dc_ph", "dc_pa"]
RLM_EXTRA_OU = ["dc_pover", "dc_lh", "dc_la"]
RL_COLS = ["rl_ph", "rl_pd", "rl_pa", "rl_pover"]
RLM_COLS = ["rlm_ph", "rlm_pd", "rlm_pa", "rlm_pover"]


def add_market_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Log-probabilidades de apertura sin margen (1X2) y logit de over 2.5."""
    df = df.copy()
    for c in ("h", "d", "a"):
        col = f"mkt_open_{c}"
        df[f"mk_lo_{c}"] = np.log(df[col].clip(1e-4)) if col in df else np.nan
    if "mkt_open_over" in df:
        p = df["mkt_open_over"].clip(1e-4, 1 - 1e-4)
        df["mk_lo_over"] = np.log(p / (1 - p))
    else:
        df["mk_lo_over"] = np.nan
    return df


class RegLog:
    """Pipeline imputación (mediana) -> estandarización -> regresión logística L2.

    `required`: columnas que deben existir en la fila (no se imputan); si faltan la predicción es NaN.
    """

    def __init__(self, features: list[str], required: list[str] | None = None, C: float = 0.05):
        self.features, self.required, self.C = list(features), list(required or []), C

    def _ok(self, df: pd.DataFrame) -> np.ndarray:
        if not self.required:
            return np.ones(len(df), bool)
        return df[self.required].notna().all(axis=1).to_numpy()

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "RegLog":
        ok = self._ok(df)
        self.features = [f for f in self.features if df.loc[ok, f].notna().any()]
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=self.C, max_iter=3000))
        self.model.fit(df.loc[ok, self.features], np.asarray(y)[ok])
        self.n_train = int(ok.sum())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        k = len(self.model.classes_)
        out = np.full((len(df), k), np.nan)
        ok = self._ok(df)
        if ok.any():
            out[ok] = self.model.predict_proba(df.loc[ok, self.features])
        return out

    def coefficients(self) -> pd.DataFrame:
        lr = self.model[-1]
        idx = [f"clase_{c}" for c in lr.classes_] if lr.coef_.shape[0] > 1 else ["over"]
        return pd.DataFrame(lr.coef_, index=idx, columns=self.features)


def fit_reglogs(train: pd.DataFrame, stat_feats: list[str], C: float = 0.05) -> dict:
    """Ajusta las 4 regresiones (rl/rlm x 1X2/OU) con los partidos jugados de `train`."""
    train = add_market_logs(train)
    stat = [f for f in stat_feats if f in train.columns]
    return {
        "rl_1x2": RegLog(stat, C=C).fit(train, train["y_1x2"]),
        "rl_ou": RegLog(stat, C=C).fit(train, train["y_over25"]),
        "rlm_1x2": RegLog(MKT_1X2 + RLM_EXTRA_1X2, required=MKT_1X2, C=C).fit(train, train["y_1x2"]),
        "rlm_ou": RegLog(MKT_OU + RLM_EXTRA_OU, required=MKT_OU, C=C).fit(train, train["y_over25"]),
    }


def predict_reglogs(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    df = add_market_logs(df)
    a, b = models["rl_1x2"].predict_proba(df), models["rl_ou"].predict_proba(df)
    c, d = models["rlm_1x2"].predict_proba(df), models["rlm_ou"].predict_proba(df)
    return pd.DataFrame({
        "rl_ph": a[:, 0], "rl_pd": a[:, 1], "rl_pa": a[:, 2], "rl_pover": b[:, 1],
        "rlm_ph": c[:, 0], "rlm_pd": c[:, 1], "rlm_pa": c[:, 2], "rlm_pover": d[:, 1],
    }, index=df.index)
