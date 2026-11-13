"""Tipos de apuesta simple además del ganador (fútbol): doble oportunidad, total de goles y ambos marcan.

Cómo se calcula la probabilidad de cada tipo de apuesta:

1. Goles esperados de cada equipo (`run_goals`): cada equipo tiene un nivel de ataque y otro de defensa que se
   actualizan tras cada partido (un único recorrido cronológico, igual que el Elo: solo se usa el pasado).
2. Con esos goles esperados se construye la matriz de marcadores (Poisson con la corrección de Dixon-Coles para
   0-0, 1-0, 0-1 y 1-1) y se reparte entre los dos equipos de forma que sea coherente con el pronóstico 1X2
   (`reconcile`); si ESPN publica la línea de goles con sus cuotas, el total también se acerca al del mercado.
3. De la matriz sale la probabilidad de cada apuesta (`market_probs`): más/menos de N goles, ambos marcan...
   Ganador y doble oportunidad salen directamente del pronóstico 1X2.

`recommend` elige, para cada partido, la apuesta recomendada: el ganador, salvo que otro tipo de apuesta sea
más probable, todavía pague algo (`mercados.prob_max`) y en la validación con años no vistos haya acertado lo
que prometía.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10
ETA_GRID = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
BIN = 0.05  # tramos de probabilidad de la calibración por mercado (50-55%, 55-60%...)

# código -> (tipo, nombre, ¿se cumple? dados los goles del local y del visitante)
MARKETS: dict[str, tuple] = {
    "1": ("Ganador", "Gana {home}", lambda h, a: h > a),
    "X": ("Ganador", "Empate", lambda h, a: h == a),
    "2": ("Ganador", "Gana {away}", lambda h, a: h < a),
    "1X": ("Doble oportunidad", "{home} gana o empata", lambda h, a: h >= a),
    "X2": ("Doble oportunidad", "{away} gana o empata", lambda h, a: h <= a),
    "12": ("Doble oportunidad", "No hay empate (gana cualquiera)", lambda h, a: h != a),
    **{f"O{n}.5": ("Total de goles", f"Más de {n}.5 goles ({n + 1} o más)", lambda h, a, n=n: h + a > n)
       for n in (0, 1, 2, 3)},
    **{f"U{n}.5": ("Total de goles", f"Menos de {n}.5 goles ({n} o menos)", lambda h, a, n=n: h + a <= n)
       for n in (1, 2, 3, 4)},
    "BTTS": ("Ambos marcan", "Ambos equipos marcan: Sí", lambda h, a: (h > 0) & (a > 0)),
    "NOBTTS": ("Ambos marcan", "Ambos equipos marcan: No", lambda h, a: (h == 0) | (a == 0)),
}
CODES = list(MARKETS)
WINNER = ("1", "X", "2")
ALIASES = {"Over": "O2.5", "Under": "U2.5"}  # códigos antiguos del ledger
DEFAULT_ACTIVE = ["1X", "X2", "O1.5", "O2.5", "U2.5", "U3.5", "BTTS", "NOBTTS"]  # además del ganador


def col(code: str) -> str:
    """Columna del ledger / hoy.csv con la probabilidad de ese mercado."""
    return f"m_{code}"


def tipo(code) -> str:
    return MARKETS[code][0] if code in MARKETS else ""


def nombre(code, home: str = "local", away: str = "visitante") -> str:
    code = ALIASES.get(code, code)
    return MARKETS[code][1].format(home=home, away=away) if code in MARKETS else str(code)


def won(code: str, hg, ag) -> np.ndarray:
    """¿Se cumplió la apuesta `code` con ese marcador?"""
    return np.asarray(MARKETS[ALIASES.get(code, code)][2](np.asarray(hg, float), np.asarray(ag, float)), bool)


def parse_code(text: str) -> str | None:
    """Lo que escribe el usuario en `futbol apuesta --a` -> código de mercado (None si no es ninguno)."""
    t = str(text).strip().upper().replace(",", ".").replace(" ", "")
    extra = {"EMPATE": "X", "OVER": "Over", "UNDER": "Under", "AMBOS": "BTTS", "AMBOSMARCAN": "BTTS",
             "AMBOSNO": "NOBTTS", "NOAMBOS": "NOBTTS", "SI": "BTTS"}
    for a, b in (("MASDE", "O"), ("MÁSDE", "O"), ("MENOSDE", "U")):
        if t.startswith(a):
            t = b + t[len(a):]
    if t in extra:
        return extra[t]
    return t if t in MARKETS else None


# ------------------------------------------------------------------------------------------------
# Goles esperados
# ------------------------------------------------------------------------------------------------
def run_goals(ev: pd.DataFrame, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Goles esperados (local, visitante) de cada partido ANTES de jugarse, en un recorrido cronológico.

    Ataque y defensa de cada equipo (escala logarítmica) se mueven tras cada partido en proporción a la
    diferencia entre los goles reales y los esperados (paso de gradiente de la verosimilitud de Poisson).
    Los partidos sin resultado reciben su predicción y no actualizan nada.
    """
    hid, aid = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    gh, ga = ev["home_score"].to_numpy(float), ev["away_score"].to_numpy(float)
    done = ev["terminado"].to_numpy(bool) & np.isfinite(gh) & np.isfinite(ga)
    neutral = ev["neutral"].to_numpy(bool)
    att: dict = defaultdict(float)
    dfn: dict = defaultdict(float)
    n: dict = defaultdict(int)
    bh, ba = math.log(1.45), math.log(1.15)  # medias de partida; se corrigen solas con los datos
    lam, mu = np.empty(len(ev)), np.empty(len(ev))
    for i in range(len(ev)):
        h, a = hid[i], aid[i]
        b1, b2 = ((bh + ba) / 2,) * 2 if neutral[i] else (bh, ba)
        l = math.exp(min(max(b1 + att[h] - dfn[a], -2.5), 1.6))
        m = math.exp(min(max(b2 + att[a] - dfn[h], -2.5), 1.6))
        lam[i], mu[i] = l, m
        if not done[i]:
            continue
        rh, ra = min(gh[i], 6.0) - l, min(ga[i], 6.0) - m  # una goleada no cuenta más que un 6-0
        kh, ka = eta * (1 + 2 * math.exp(-n[h] / 10)), eta * (1 + 2 * math.exp(-n[a] / 10))  # nuevos: aprenden rápido
        att[h] += kh * rh
        dfn[h] -= kh * ra
        att[a] += ka * ra
        dfn[a] -= ka * rh
        if not neutral[i]:
            bh += 0.003 * rh
            ba += 0.003 * ra
        n[h] += 1
        n[a] += 1
    return lam, mu


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    return np.where((x == 1) & (y == 1), 1 - rho, t)


def tune_goals(ev: pd.DataFrame) -> dict:
    """Velocidad de aprendizaje (`g_eta`) y corrección de marcadores bajos (`g_rho`) por máxima verosimilitud.
    El primer año no cuenta: los niveles todavía se están asentando."""
    gh, ga = ev["home_score"].to_numpy(float), ev["away_score"].to_numpy(float)
    ok = (ev["terminado"].to_numpy(bool) & np.isfinite(gh) & np.isfinite(ga)
          & (ev["inicio_utc"] > ev["inicio_utc"].min() + pd.Timedelta(days=365)).to_numpy())
    if ok.sum() < 100:
        return {"g_eta": 0.02, "g_rho": -0.05}
    best = None
    for eta in ETA_GRID:
        lam, mu = run_goals(ev, eta)
        nll = float(np.mean(lam[ok] + mu[ok] - gh[ok] * np.log(lam[ok]) - ga[ok] * np.log(mu[ok])))
        if best is None or nll < best[0]:
            best = (nll, eta, lam[ok], mu[ok])
    _, eta, lam, mu = best
    x, y = gh[ok].astype(int), ga[ok].astype(int)

    def nll_rho(rho):
        t = _tau(x, y, lam, mu, rho)
        return 1e12 if np.any(t <= 0) else -float(np.log(t).sum())

    rho = float(minimize_scalar(nll_rho, bounds=(-0.2, 0.1), method="bounded").x)
    return {"g_eta": eta, "g_rho": round(rho, 4)}


# ------------------------------------------------------------------------------------------------
# Matriz de marcadores y probabilidades por mercado
# ------------------------------------------------------------------------------------------------
def score_matrix(lam, mu, rho: float = 0.0) -> np.ndarray:
    """P(local marca i, visitante marca j): array (n, MAX_GOALS+1, MAX_GOALS+1) que suma 1 por partido."""
    lam, mu = np.atleast_1d(np.asarray(lam, float)), np.atleast_1d(np.asarray(mu, float))
    g = np.arange(MAX_GOALS + 1)
    M = poisson.pmf(g[None, :], lam[:, None])[:, :, None] * poisson.pmf(g[None, :], mu[:, None])[:, None, :]
    M[:, 0, 0] *= np.clip(1 - lam * mu * rho, 0.01, None)
    M[:, 0, 1] *= np.clip(1 + lam * rho, 0.01, None)
    M[:, 1, 0] *= np.clip(1 + mu * rho, 0.01, None)
    M[:, 1, 1] *= max(1 - rho, 0.01)
    return M / M.sum(axis=(1, 2), keepdims=True)


_I, _J = np.meshgrid(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1), indexing="ij")


def _p_over(M: np.ndarray, line) -> np.ndarray:
    return (M * ((_I + _J)[None] > np.asarray(line, float).reshape(-1, 1, 1))).sum(axis=(1, 2))


def reconcile(lam, mu, p_home, p_away, rho: float = 0.0, line=None, p_over=None) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta los goles esperados para que la matriz de marcadores cuente la misma historia que el 1X2.

    Se conserva el total de goles esperado y se reparte entre los dos equipos hasta que
    P(gana local) / P(gana local o visitante) coincide con el pronóstico. Si se da `line` y `p_over` (línea de
    goles y probabilidad objetivo de superarla; NaN = no tocar), el total se mueve hasta cumplirla.
    """
    lam, mu = np.asarray(lam, float), np.asarray(mu, float)
    T = lam + mu
    ph, pa = np.asarray(p_home, float), np.asarray(p_away, float)
    target = np.where(ph + pa > 0, ph / np.where(ph + pa > 0, ph + pa, 1), 0.5)
    share = lam / T
    fix_total = (np.isfinite(np.asarray(line, float)) & np.isfinite(np.asarray(p_over, float))
                 if line is not None and p_over is not None else np.zeros(len(T), bool))
    for _ in range(3 if fix_total.any() else 1):
        lo, hi = np.full(len(T), 0.02), np.full(len(T), 0.98)
        for _ in range(30):  # bisección: más parte del total para el local -> más probable que gane
            share = (lo + hi) / 2
            M = score_matrix(T * share, T * (1 - share), rho)
            h, a = np.tril(M, -1).sum(axis=(1, 2)), np.triu(M, 1).sum(axis=(1, 2))
            low = h / (h + a) < target
            lo, hi = np.where(low, share, lo), np.where(low, hi, share)
        if not fix_total.any():
            break
        lo, hi = np.full(len(T), 0.3), np.full(len(T), 7.0)
        for _ in range(30):  # bisección: más goles en total -> más probable superar la línea
            t = (lo + hi) / 2
            low = _p_over(score_matrix(t * share, t * (1 - share), rho), line) < p_over
            lo, hi = np.where(low, t, lo), np.where(low, hi, t)
        T = np.where(fix_total, (lo + hi) / 2, T)
    return T * share, T * (1 - share)


def market_probs(P: np.ndarray, M: np.ndarray | None = None) -> pd.DataFrame:
    """Probabilidad de cada mercado (columnas = códigos). `P`: 1X2 (n × 3); `M`: matriz de marcadores o None
    (entonces solo se rellenan ganador y doble oportunidad)."""
    P = np.asarray(P, float)
    out = {"1": P[:, 0], "X": P[:, 1], "2": P[:, 2], "1X": P[:, 0] + P[:, 1], "X2": P[:, 1] + P[:, 2],
           "12": P[:, 0] + P[:, 2]}
    nan = np.full(len(P), np.nan)
    for code in CODES:
        if code in out:
            continue
        if M is None:
            out[code] = nan
        elif code[0] in "OU" and code[1].isdigit():
            over = _p_over(M, np.full(len(P), float(code[1:])))
            out[code] = over if code[0] == "O" else 1 - over
        else:
            both = M[:, 1:, 1:].sum(axis=(1, 2))
            out[code] = both if code == "BTTS" else 1 - both
    return pd.DataFrame(out)[CODES]


def tramo(p) -> np.ndarray:
    """Borde inferior del tramo de 5 puntos en el que cae una probabilidad (0.73 -> 0.70; 1.0 -> 0.95)."""
    p = np.asarray(p, float)
    return np.round(np.clip(np.floor(np.nan_to_num(p, nan=0.0) / BIN + 1e-9) * BIN, 0.0, 1 - BIN), 2)


def calibrate(probs: pd.DataFrame, hg, ag, liga: str) -> pd.DataFrame:
    """Calibración por mercado y tramo de probabilidad: cuando el sistema dijo 70-75%, ¿cuánto acertó?
    Solo cuenta el lado al que se apostaría (probabilidad ≥ 50%)."""
    rows = []
    for code in probs.columns:
        p = probs[code].to_numpy(float)
        ok = np.isfinite(p) & (p >= 0.5)
        if not ok.any():
            continue
        d = pd.DataFrame({"tramo": tramo(p[ok]), "p": p[ok], "hit": won(code, hg, ag)[ok]})
        for t, g in d.groupby("tramo"):
            rows.append({"liga": liga, "mercado": code, "tipo": tipo(code), "tramo": float(t), "partidos": len(g),
                         "acierto": float(g["hit"].mean()), "prob_media": float(g["p"].mean())})
    return pd.DataFrame(rows, columns=["liga", "mercado", "tipo", "tramo", "partidos", "acierto", "prob_media"])


# ------------------------------------------------------------------------------------------------
# Apuesta recomendada
# ------------------------------------------------------------------------------------------------
def recommend(probs: pd.DataFrame, hist: pd.DataFrame, n_hist: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Elige la apuesta recomendada de cada partido.

    `probs`, `hist`, `n_hist`: una fila por partido y una columna por mercado (probabilidad, % de acierto de
    ese mercado en ese tramo en la validación y nº de casos; NaN / 0 si no hay validación).

    Regla: el punto de partida es el ganador más probable (nunca el empate, salvo `seleccion.permitir_empate`).
    Otro tipo de apuesta de `mercados.activos` lo sustituye solo si es más probable que él y además:
    - no es una apuesta casi segura que no paga (probabilidad > `prob_max`),
    - y en la validación, con al menos `n_hist_min` casos, no acertó `margen_calibracion` puntos menos de lo
      que el sistema decía (ese tipo de apuesta no es fiable en esa liga).
    """
    m = cfg.get("mercados") or {}
    win = ["1", "2"] + (["X"] if (cfg.get("seleccion") or {}).get("permitir_empate", False) else [])
    active = [c for c in (m.get("activos") or DEFAULT_ACTIVE) if c in probs.columns and c not in WINNER]
    pw = np.nan_to_num(probs[win].to_numpy(float), nan=-1.0)
    code, best = np.array(win, dtype=object)[pw.argmax(axis=1)], pw.max(axis=1)
    if active:
        p = probs[active].to_numpy(float)
        h, n = hist[active].to_numpy(float), n_hist[active].to_numpy(float)
        bad = (n >= int(m.get("n_hist_min", 100))) & (h < p - float(m.get("margen_calibracion", 0.05)))
        ok = np.isfinite(p) & (p <= float(m.get("prob_max", 0.80)) + 1e-9) & ~bad
        alt = np.where(ok, p, -1.0)
        k = alt.argmax(axis=1)
        code = np.where(alt.max(axis=1) > best + 1e-9, np.array(active, dtype=object)[k], code)
    i = np.arange(len(probs))
    j = probs.columns.get_indexer(code)
    return pd.DataFrame({"rec": code, "rec_tipo": [tipo(c) for c in code],
                         "p_rec": probs.to_numpy(float)[i, j], "hist_rec": hist.to_numpy(float)[i, j],
                         "n_hist_rec": n_hist.to_numpy(float)[i, j].astype(int)}, index=probs.index)
