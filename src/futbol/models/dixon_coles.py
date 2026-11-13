"""Modelo Dixon-Coles (Poisson bivariado con corrección de marcadores bajos y decaimiento temporal).

Ajuste en dos pasos (rápido y estable):
1. Ataques, defensas y ventaja local con una regresión de Poisson ponderada por tiempo
   (pesos exp(-xi * días)) y regularización L2 leve (promovidos -> cerca de la media).
2. rho (corrección de Dixon-Coles para 0-0, 1-0, 0-1, 1-1) por máxima verosimilitud 1D.

`dc_walk_forward` reentrena cada semana con datos estrictamente anteriores a esa semana.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger(__name__)


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, xi: float = 0.0019, alpha: float = 0.002, max_goals: int = 10):
        self.xi, self.alpha, self.max_goals = xi, alpha, max_goals
        self.teams: list[str] = []
        self.rho = 0.0

    def fit(self, home, away, hg, ag, dates, ref_date) -> "DixonColes":
        home, away = np.asarray(home), np.asarray(away)
        hg, ag = np.asarray(hg, float), np.asarray(ag, float)
        days = (pd.Timestamp(ref_date) - pd.to_datetime(pd.Series(dates))).dt.days.to_numpy()
        w = np.exp(-self.xi * np.clip(days, 0, None))
        self.teams = sorted(set(home) | set(away))
        idx = {t: i for i, t in enumerate(self.teams)}
        n, T = len(home), len(self.teams)
        X = np.zeros((2 * n, 2 * T + 1))
        hi = np.array([idx[t] for t in home])
        ai = np.array([idx[t] for t in away])
        r = np.arange(n)
        X[r, hi] = 1            # ataque local
        X[r, T + ai] = 1        # defensa visitante
        X[r, 2 * T] = 1         # ventaja local
        X[n + r, ai] = 1        # ataque visitante
        X[n + r, T + hi] = 1    # defensa local
        y = np.concatenate([hg, ag])
        sw = np.concatenate([w, w])
        self.glm = PoissonRegressor(alpha=self.alpha / max(sw.mean(), 1e-9), max_iter=1000, tol=1e-6)
        self.glm.fit(X, y, sample_weight=sw)
        coef = self.glm.coef_
        self.attack = coef[:T]
        self.defense = coef[T:2 * T]
        self.home_adv = coef[2 * T]
        self.intercept = self.glm.intercept_
        # equipos desconocidos (p. ej. recién ascendidos): por debajo de la media
        self.unk_attack = float(np.quantile(self.attack, 0.25))
        self.unk_defense = float(np.quantile(self.defense, 0.75))
        lh, la = self._lambdas(home, away)
        hgi, agi = hg.astype(int), ag.astype(int)

        def nll(rho):
            t = _tau(hgi, agi, lh, la, rho)
            if np.any(t <= 0):
                return 1e12
            return -np.sum(w * np.log(t))

        self.rho = float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)
        return self

    def _lambdas(self, home, away):
        idx = {t: i for i, t in enumerate(self.teams)}
        att = lambda t: self.attack[idx[t]] if t in idx else self.unk_attack  # noqa: E731
        de = lambda t: self.defense[idx[t]] if t in idx else self.unk_defense  # noqa: E731
        lh = np.exp(self.intercept + self.home_adv + np.array([att(h) + de(a) for h, a in zip(home, away)]))
        la = np.exp(self.intercept + np.array([att(a) + de(h) for h, a in zip(home, away)]))
        return lh, la

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        m[0, 0] *= 1 - lh * la * self.rho
        m[0, 1] *= 1 + lh * self.rho
        m[1, 0] *= 1 + la * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def predict(self, home, away) -> pd.DataFrame:
        lh, la = self._lambdas(np.asarray(home), np.asarray(away))
        rows = []
        tot = np.add.outer(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        for a, b in zip(lh, la):
            m = self.score_matrix(a, b)
            rows.append((np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m[tot > 2.5].sum(), a, b))
        return pd.DataFrame(rows, columns=["dc_ph", "dc_pd", "dc_pa", "dc_pover", "dc_lh", "dc_la"])


DC_COLS = ["dc_ph", "dc_pd", "dc_pa", "dc_pover", "dc_lh", "dc_la"]


def dc_walk_forward(matches: pd.DataFrame, dc_cfg: dict) -> pd.DataFrame:
    """Predicciones Dixon-Coles fuera de muestra para cada partido.

    Para cada liga y cada semana (lunes a domingo) se ajusta el modelo con los partidos
    jugados estrictamente ANTES del lunes de esa semana (ventana de `window_days`).
    Devuelve un DataFrame alineado con `matches.index` con columnas DC_COLS (NaN si no hay
    historial suficiente).
    """
    out = pd.DataFrame(np.nan, index=matches.index, columns=DC_COLS)
    window = pd.Timedelta(days=int(dc_cfg["window_days"]))
    for lg, ldf in matches.groupby("league"):
        ldf = ldf.sort_values("date")
        played = ldf[ldf["fthg"].notna() & ldf["ftag"].notna()]
        week = (ldf["date"] - pd.to_timedelta(ldf["date"].dt.day_of_week, unit="D")).dt.normalize()
        n_fit = 0
        for wk, target in ldf.groupby(week):
            train = played[(played["date"] < wk) & (played["date"] >= wk - window)]
            if len(train) < int(dc_cfg["min_matches"]):
                continue
            model = DixonColes(xi=dc_cfg["xi"], alpha=dc_cfg["alpha"], max_goals=dc_cfg["max_goals"])
            model.fit(train["home_team"], train["away_team"], train["fthg"], train["ftag"], train["date"], wk)
            pred = model.predict(target["home_team"], target["away_team"])
            out.loc[target.index, DC_COLS] = pred[DC_COLS].to_numpy()
            n_fit += 1
        log.info("Dixon-Coles %s: %d reentrenamientos semanales", lg, n_fit)
    return out


def fit_latest(matches: pd.DataFrame, league: str, dc_cfg: dict, ref_date) -> DixonColes | None:
    """Ajusta un DC con todos los partidos jugados de la liga hasta `ref_date` (para predecir)."""
    window = pd.Timedelta(days=int(dc_cfg["window_days"]))
    ref = pd.Timestamp(ref_date)
    p = matches[(matches["league"] == league) & matches["fthg"].notna() & (matches["date"] < ref)
                & (matches["date"] >= ref - window)]
    if len(p) < int(dc_cfg["min_matches"]):
        return None
    m = DixonColes(xi=dc_cfg["xi"], alpha=dc_cfg["alpha"], max_goals=dc_cfg["max_goals"])
    return m.fit(p["home_team"], p["away_team"], p["fthg"], p["ftag"], p["date"], ref)
