"""Modelo genérico para cualquier deporte de dos contrincantes: Elo + estado de forma + regresión logística.

Para cada partido se calculan variables SOLO con partidos anteriores (un único recorrido cronológico):

- `elo`:      diferencia de rating Elo (se actualiza tras cada partido; K y peso del margen se ajustan por liga).
- `forma5/10`: diferencia de rendimiento en los últimos 5 y 10 partidos (victoria 1, empate 0.5, derrota 0).
- `margen10`: diferencia del margen medio (goles/puntos/sets) en los últimos 10.
- `descanso`: diferencia de días de descanso (tope 10).
- `carga`:    diferencia de partidos jugados en los últimos 7 días (cansancio; back-to-back en NBA/NHL).
- `h2h`:      historial entre ambos (suavizado).
- `exp`:      diferencia de experiencia (log de partidos jugados; importa sobre todo en tenis).
- `local`:    1 si hay ventaja de campo (0 en tenis y campo neutral).

Con esas variables se ajusta una regresión logística (multinomial si el deporte admite empate).
Validación walk-forward por año: para predecir el año A se entrena solo con lo anterior a A.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import get_path
from ..evaluation.informe import LEVELS, confidence, wilson

log = logging.getLogger(__name__)

FEATS = ["elo", "forma5", "forma10", "margen10", "descanso", "carga", "h2h", "exp", "local"]
K_GRID = [4, 6, 8, 12, 16, 24, 32, 48, 64]
REGRESS = {"soccer": 0.15, "tennis": 0.0}      # regresión a la media al cambiar de temporada
NEW_OFFSET = {"soccer": 60.0, "tennis": 100.0}  # los que llegan después (ascendidos, debutantes) entran por debajo


def outcome(ev: pd.DataFrame, draw: bool) -> np.ndarray:
    """0 = gana local/jugador 1, 1 = empate, 2 = gana visitante/jugador 2, -1 = sin resultado válido."""
    w = ev["ganador"].to_numpy()
    y = np.where(w == "home", 0, np.where(w == "away", 2, np.where(w == "draw", 1, -1)))
    return np.where(ev["terminado"].to_numpy(), y, -1) if draw else np.where(y == 1, -1, np.where(ev["terminado"].to_numpy(), y, -1))


def run(ev: pd.DataFrame, lg: dict, k: float, mov: float, hfa: float, scale: float,
        collect: bool = True) -> tuple[np.ndarray, float]:
    """Recorre los partidos en orden. Devuelve (features por partido, error cuadrático medio del Elo).

    Los partidos sin resultado (próximos) reciben features con el estado acumulado y no actualizan nada.
    """
    sport = lg["sport"]
    regress, new_off = REGRESS.get(sport, 0.25), NEW_OFFSET.get(sport, 0.0)
    rating: dict = {}
    n_played: dict = defaultdict(int)
    hist: dict = defaultdict(lambda: deque(maxlen=10))    # (resultado, margen)
    days: dict = defaultdict(lambda: deque(maxlen=12))    # días (ordinal) de los últimos partidos
    season: dict = {}
    h2h: dict = defaultdict(lambda: [0.0, 0])             # clave (a, b) con a < b: [victorias de a − de b, n]
    hid, aid = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    t = ((ev["inicio_utc"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy(float)  # días
    seas = ev["temporada"].to_numpy()
    neutral = ev["neutral"].to_numpy()
    y = outcome(ev, bool(lg.get("empate")))
    margin = (ev["home_score"].fillna(0) - ev["away_score"].fillna(0)).to_numpy(float)
    t0 = t[0] if len(t) else 0.0
    F = np.zeros((len(ev), len(FEATS))) if collect else None
    se, ns = 0.0, 0

    def prep(team, day, s):
        if team not in rating:
            rating[team] = 1500.0 - (new_off if day - t0 > 365 else 0.0)
            season[team] = s
        elif season[team] != s and s == s:  # cambio de temporada
            rating[team] = 1500.0 + (rating[team] - 1500.0) * (1 - regress)
            season[team] = s

    def form(team, day):
        h = hist[team]
        if not h:
            return 0.5, 0.5, 0.0, 10.0, 0.0
        r = [x[0] for x in h]
        d = days[team]
        return (sum(r[-5:]) / len(r[-5:]), sum(r) / len(r), sum(x[1] for x in h) / len(h),
                min(max(day - d[-1], 0.0), 10.0), float(sum(1 for x in d if day - x <= 7)))

    for i in range(len(ev)):
        h, a, day = hid[i], aid[i], t[i]
        prep(h, day, seas[i])
        prep(a, day, seas[i])
        adv = 0.0 if neutral[i] else hfa
        e = 1.0 / (1.0 + 10 ** (-(rating[h] - rating[a] + adv) / 400.0))
        if collect:
            f5h, f10h, mh, rh, lh = form(h, day)
            f5a, f10a, ma, ra, la = form(a, day)
            key, sign = ((h, a), 1.0) if h < a else ((a, h), -1.0)
            hh = h2h[key]
            F[i] = [(rating[h] - rating[a]) / 400.0, f5h - f5a, f10h - f10a, (mh - ma) / scale, rh - ra, lh - la,
                    sign * hh[0] / (hh[1] + 2.0), math.log1p(n_played[h]) - math.log1p(n_played[a]),
                    0.0 if neutral[i] else 1.0]
        if y[i] < 0:
            continue
        s = 1.0 - y[i] / 2.0  # 0 -> 1, 1 -> 0.5, 2 -> 0
        se += (s - e) ** 2
        ns += 1
        mult = 1.0 + mov * math.log1p(abs(margin[i]) / scale)
        delta = mult * (s - e)
        rating[h] += k * (1.0 + 3.0 * math.exp(-n_played[h] / 15.0)) * delta
        rating[a] -= k * (1.0 + 3.0 * math.exp(-n_played[a] / 15.0)) * delta
        for team, res, mg in ((h, s, margin[i]), (a, 1.0 - s, -margin[i])):
            hist[team].append((res, mg))
            days[team].append(day)
            n_played[team] += 1
        if collect:
            hh[0] += sign * (s - 0.5) * 2.0
            hh[1] += 1
    return F, (se / ns if ns else np.nan)


def tune(ev: pd.DataFrame, lg: dict) -> dict:
    """Ventaja de campo, escala del margen y (K, peso del margen) que minimizan el error del Elo."""
    y = outcome(ev, bool(lg.get("empate")))
    ok = y >= 0
    m = (ev["home_score"] - ev["away_score"]).abs()[ok]
    scale = float(max(m[m > 0].median(), 1.0)) if (m > 0).any() else 1.0
    nn = ok & ~ev["neutral"].to_numpy()
    share = float(np.mean(1.0 - y[nn] / 2.0)) if nn.sum() > 50 else 0.5
    hfa = float(np.clip(400 * math.log10(share / (1 - share)), 0, 200)) if 0 < share < 1 else 0.0
    best = None
    for mov in (0.0, 1.0):
        for k in K_GRID:
            _, mse = run(ev, lg, k, mov, hfa, scale, collect=False)
            if best is None or mse < best[0]:
                best = (mse, k, mov)
    return {"k": best[1], "mov": best[2], "hfa": round(hfa, 1), "scale": scale}


def _clf(draw: bool):
    lr = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=draw)  # sin empate: simétrico, p=0.5 si todo es 0
    return make_pipeline(StandardScaler(with_mean=False), lr)


def fit_predict(F: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, draw: bool) -> np.ndarray:
    """Probabilidades (n_test × 3: local, empate, visitante) entrenando con `train` (máscaras booleanas)."""
    m = _clf(draw).fit(F[train], y[train])
    p = m.predict_proba(F[test])
    out = np.zeros((int(test.sum()), 3))
    for j, c in enumerate(m[-1].classes_):
        out[:, int(c)] = p[:, j]
    return out


def _burn_in(ev: pd.DataFrame) -> np.ndarray:
    """Los primeros 365 días no se usan para entrenar: el Elo todavía se está asentando."""
    return (ev["inicio_utc"] > ev["inicio_utc"].min() + pd.Timedelta(days=365)).to_numpy()


def evaluate_league(ev: pd.DataFrame, lg: dict, cfg: dict) -> tuple[dict, pd.DataFrame, dict] | None:
    """Walk-forward por año natural. Devuelve (métricas, calibración por confianza, parámetros) o None."""
    dcfg = cfg.get("deportes") or {}
    draw = bool(lg.get("empate"))
    ev = ev[ev["terminado"]].reset_index(drop=True)
    y = outcome(ev, draw)
    if (y >= 0).sum() < int(dcfg.get("min_partidos", 600)):
        return None
    years = ev["inicio_utc"].dt.year.to_numpy()
    counts = pd.Series(years[y >= 0]).value_counts()
    test_years = sorted(counts[counts >= 50].index)[-int(dcfg.get("anios_test", 3)):]
    first = pd.Timestamp(f"{test_years[0]}-01-01", tz="UTC")
    params = tune(ev[ev["inicio_utc"] < first], lg)  # los hiperparámetros se eligen sin mirar los años de test
    F, _ = run(ev, lg, params["k"], params["mov"], params["hfa"], params["scale"])
    valid = (y >= 0) & _burn_in(ev)
    P = np.full((len(ev), 3), np.nan)
    for yr in test_years:
        tr, te = valid & (years < yr), (y >= 0) & (years == yr)
        if tr.sum() < 300 or not te.any():
            continue
        P[te] = fit_predict(F, y, tr, te, draw)
    ok = np.isfinite(P).all(axis=1)
    if not ok.any():
        return None
    p, yy = P[ok], y[ok]
    i = np.arange(len(yy))
    pick, pp = p.argmax(axis=1), p.max(axis=1)
    hit = pick == yy
    elo_pick = np.where(F[ok, 0] + (params["hfa"] / 400.0) * F[ok, 8] >= 0, 0, 2)
    onehot = np.zeros_like(p)
    onehot[i, yy] = 1
    metrics = {"liga": lg["id"], "nombre": lg["nombre"], "deporte": lg["deporte"], "partidos_test": int(ok.sum()),
               "anios_test": f"{test_years[0]}-{test_years[-1]}", "acierto": float(hit.mean()),
               "acierto_local_siempre": float((yy == 0).mean()), "acierto_solo_elo": float((elo_pick == yy).mean()),
               "log_loss": float(-np.log(np.clip(p[i, yy], 1e-12, None)).mean()),
               "brier": float(((p - onehot) ** 2).sum(axis=1).mean()), **{f"elo_{k}": v for k, v in params.items()}}
    conf = confidence(pp, cfg)
    rows = []
    for level in LEVELS:
        m = conf == level
        lo, hi = wilson(float(hit[m].sum()), int(m.sum()))
        rows.append({"liga": lg["id"], "confianza": level, "partidos": int(m.sum()),
                     "pct_del_total": float(m.mean()), "acierto": float(hit[m].mean()) if m.any() else np.nan,
                     "prob_media_modelo": float(pp[m].mean()) if m.any() else np.nan, "ic95_inf": lo, "ic95_sup": hi})
        metrics[f"acierto_{level.lower()}"] = rows[-1]["acierto"]
        metrics[f"n_{level.lower()}"] = int(m.sum())
    return metrics, pd.DataFrame(rows), params


def evaluate(cfg: dict, ev: pd.DataFrame, only: list[str] | None = None) -> pd.DataFrame:
    """Valida todas las ligas y guarda reports/deportes_metricas.csv, deportes_calibracion.csv y los parámetros."""
    from .espn import leagues

    rep, mdir = get_path(cfg, "reports"), get_path(cfg, "models")
    mets, cals, params = [], [], load_params(cfg)
    for lg in leagues(cfg, only):
        res = evaluate_league(ev[ev["liga"] == lg["id"]], lg, cfg)
        if res is None:
            log.warning("%s: historial insuficiente; no se darán pronósticos de esta liga.", lg["nombre"])
            params.pop(lg["id"], None)
            continue
        m, c, p = res
        mets.append(m)
        cals.append(c)
        params[lg["id"]] = p
        log.info("%-28s test %s  n=%5d  acierto %.1f%% (local siempre %.1f%%)  confianza alta: %.1f%% en %d",
                 lg["nombre"], m["anios_test"], m["partidos_test"], 100 * m["acierto"],
                 100 * m["acierto_local_siempre"], 100 * (m["acierto_alta"] if m["n_alta"] else np.nan), m["n_alta"])
    if not mets:
        raise RuntimeError("Ninguna liga tiene historial suficiente. Ejecuta `futbol deportes-descargar`.")
    old = [pd.read_csv(rep / f) for f in ("deportes_metricas.csv", "deportes_calibracion.csv") if only and (rep / f).exists()]
    met, cal = pd.DataFrame(mets), pd.concat(cals, ignore_index=True)
    if only and len(old) == 2:  # evaluación parcial: conservar las demás ligas
        met = pd.concat([old[0][~old[0]["liga"].isin(met["liga"])], met], ignore_index=True)
        cal = pd.concat([old[1][~old[1]["liga"].isin(cal["liga"])], cal], ignore_index=True)
    met.sort_values("acierto", ascending=False).round(4).to_csv(rep / "deportes_metricas.csv", index=False)
    cal.round(4).to_csv(rep / "deportes_calibracion.csv", index=False)
    (mdir / "deportes_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return met


def load_params(cfg: dict) -> dict:
    p = get_path(cfg, "models") / "deportes_params.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def predict_upcoming(ev: pd.DataFrame, lg: dict, params: dict) -> pd.DataFrame:
    """Probabilidades del modelo para los partidos programados de una liga (entrena con todo lo jugado)."""
    draw = bool(lg.get("empate"))
    ev = ev[ev["terminado"] | ev["programado"]].reset_index(drop=True)
    up = ev["programado"].to_numpy() & ~ev["terminado"].to_numpy()
    if not up.any():
        return pd.DataFrame()
    y = outcome(ev, draw)
    F, _ = run(ev, lg, params["k"], params["mov"], params["hfa"], params["scale"])
    P = fit_predict(F, y, (y >= 0) & _burn_in(ev), up, draw)
    out = ev[up].copy()
    out["mod_p_home"], out["mod_p_draw"], out["mod_p_away"] = P[:, 0], P[:, 1], P[:, 2]
    for j, f in enumerate(FEATS):
        out[f"x_{f}"] = F[up, j]
    return out
