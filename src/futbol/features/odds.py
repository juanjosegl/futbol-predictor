"""Probabilidades implícitas de cuotas decimales quitando el margen de la casa."""
from __future__ import annotations

import numpy as np
import pandas as pd


def overround(odds: np.ndarray) -> np.ndarray:
    """Suma de inversas - 1 (margen de la casa). `odds` con forma (n, k)."""
    odds = np.asarray(odds, dtype=float)
    return (1.0 / odds).sum(axis=1) - 1.0


def remove_margin_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Normalización proporcional: p_i = (1/o_i) / sum_j (1/o_j)."""
    odds = np.atleast_2d(np.asarray(odds, dtype=float))
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def _shin_p(z: float, pi: np.ndarray, s: float) -> np.ndarray:
    return (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))


def remove_margin_shin(odds: np.ndarray) -> np.ndarray:
    """Método de Shin (1993): corrige el sesgo favorito-longshot.

    Se busca z (proporción de apostadores informados) tal que las probabilidades sumen 1.
    """
    from scipy.optimize import brentq

    odds = np.atleast_2d(np.asarray(odds, dtype=float))
    out = np.full_like(odds, np.nan)
    for r in range(odds.shape[0]):
        o = odds[r]
        if not np.all(np.isfinite(o)) or np.any(o <= 1):
            continue
        pi = 1.0 / o
        s = pi.sum()
        if s <= 1:  # sin margen (o negativo): no hay nada que corregir
            out[r] = pi / s
            continue
        f = lambda z: _shin_p(z, pi, s).sum() - 1.0  # noqa: E731  (decreciente en z)
        z = brentq(f, 0.0, 0.999) if f(0.999) < 0 else 0.999
        p = _shin_p(z, pi, s)
        out[r] = p / p.sum()
    return out


def implied_probs(odds: np.ndarray | pd.DataFrame, method: str = "multiplicative") -> np.ndarray:
    """Probabilidades sin margen; filas con cuotas faltantes o <=1 devuelven NaN."""
    arr = np.atleast_2d(np.asarray(odds, dtype=float))
    valid = np.all(np.isfinite(arr), axis=1) & np.all(arr > 1, axis=1)
    out = np.full_like(arr, np.nan)
    if valid.any():
        fn = remove_margin_shin if method == "shin" else remove_margin_multiplicative
        out[valid] = fn(arr[valid])
    return out


def fair_odds(probs: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 1.0 / np.asarray(probs, dtype=float)


def add_market_probs(df: pd.DataFrame, method: str = "multiplicative") -> pd.DataFrame:
    """Añade mkt_{open,close}_{h,d,a} y mkt_{open,close}_over a partir de las cuotas."""
    df = df.copy()
    for when in ("open", "close"):
        p = implied_probs(df[[f"{when}_h", f"{when}_d", f"{when}_a"]].to_numpy(), method)
        df[f"mkt_{when}_h"], df[f"mkt_{when}_d"], df[f"mkt_{when}_a"] = p[:, 0], p[:, 1], p[:, 2]
        q = implied_probs(df[[f"{when}_over", f"{when}_under"]].to_numpy(), method)
        df[f"mkt_{when}_over"] = q[:, 0]
    return df
