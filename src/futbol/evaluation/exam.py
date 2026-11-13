"""Examen sobre partidos ya jugados: "¿qué habría predicho el modelo antes de esos partidos?".

Para un rango [desde, hasta]:
- Todos los modelos se entrenan SOLO con partidos anteriores a `desde`
  (LightGBM: base hasta desde-365 días y calibración en el último año; regresiones: todo < desde).
- Dixon-Coles es walk-forward semanal (cada semana se ajusta con lo anterior a esa semana).
- Las features de cada partido usan solo partidos previos (igual que en producción).
- Se compara con el resultado real, con el mercado de apertura (lo que sabías al apostar) y el de cierre.
- Se simulan las apuestas con valor a cuota de apertura (stake fijo de 1 unidad).

Salida: reports/examen_<desde>_<hasta>.csv (partido a partido) y examen_<desde>_<hasta>.json (resumen).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import get_path, league_codes
from ..features.build import load_dataset
from ..features.engineering import feature_columns
from ..models.dixon_coles import DC_COLS
from ..models.reglog import fit_reglogs, predict_reglogs
from ..models.training import _fit_pair, _predict_pair, with_dc
from . import metrics as M
from .informe import write_report

log = logging.getLogger(__name__)

MODELS = {
    "Reg. logística + mercado": ("rlm_ph", "rlm_pd", "rlm_pa", "rlm_pover"),
    "Reg. logística (stats)": ("rl_ph", "rl_pd", "rl_pa", "rl_pover"),
    "LightGBM": ("lgbm_ph", "lgbm_pd", "lgbm_pa", "lgbm_pover"),
    "Dixon-Coles": ("dc_ph", "dc_pd", "dc_pa", "dc_pover"),
    "Mercado apertura": ("mkt_open_h", "mkt_open_d", "mkt_open_a", "mkt_open_over"),
    "Mercado cierre": ("mkt_close_h", "mkt_close_d", "mkt_close_a", "mkt_close_over"),
}
SEL = np.array(["1", "X", "2"])


def run_exam(cfg: dict, desde: str, hasta: str, leagues: list[str] | None = None,
             threshold: float | None = None, df: pd.DataFrame | None = None) -> dict:
    d0, d1 = pd.Timestamp(desde), pd.Timestamp(hasta)
    thr = float(cfg["backtest"]["threshold"] if threshold is None else threshold)
    df = with_dc(load_dataset(cfg), cfg) if df is None else df
    feats = feature_columns(df, cfg, DC_COLS if cfg["features"].get("use_dc_probs", True) else [])
    lgs = league_codes(cfg)
    played = df[df["played"]]
    hist = played[played["date"] < d0]
    target = played[(played["date"] >= d0) & (played["date"] <= d1)]
    if leagues:
        target = target[target["league"].isin(leagues)]
    if target.empty:
        raise ValueError(f"No hay partidos jugados entre {desde} y {hasta} en los datos (¿falta `download`/`build`?).")
    if len(hist) < 2000:
        raise ValueError("Muy poco historial antes de la fecha de inicio para entrenar.")
    cut = d0 - pd.Timedelta(days=365)
    base, cal = hist[hist["date"] < cut], hist[hist["date"] >= cut]
    log.info("Examen %s → %s: %d partidos. Entrenamiento con %d partidos anteriores (calibración desde %s)",
             desde, hasta, len(target), len(hist), cut.date())
    m1, m2, use = _fit_pair(base, cal, feats, cfg, lgs)
    p = _predict_pair(m1, m2, use, target, lgs)
    r = predict_reglogs(fit_reglogs(hist, feats, C=float(cfg["model"].get("reglog_C", 0.05))), target)
    t = pd.concat([target, p, r], axis=1)

    main = {"rlm": "Reg. logística + mercado", "rl": "Reg. logística (stats)", "lgbm": "LightGBM"}.get(
        cfg["model"].get("main", "rlm"), "Reg. logística + mercado")
    cols = MODELS[main]
    P = t[list(cols[:3])].to_numpy(float).copy()
    nomain = ~np.isfinite(P).all(axis=1)  # sin cuotas de apertura -> respaldo LightGBM
    P[nomain] = t.loc[nomain, list(MODELS["LightGBM"][:3])].to_numpy(float)
    y = t["y_1x2"].to_numpy()
    res = SEL[y]
    odds = t[["open_h", "open_d", "open_a"]].to_numpy(float)
    edges = P * odds - 1
    safe = np.where(np.isfinite(edges), edges, -np.inf)
    best = safe.argmax(axis=1)
    i = np.arange(len(t))
    vb = np.where(safe[i, best] > thr, SEL[best], "")
    won = vb == res
    profit = np.where(vb == "", np.nan, np.where(won, odds[i, best] - 1, -1.0))
    mk = t[list(MODELS["Mercado apertura"][:3])].to_numpy(float)

    detail = pd.DataFrame({
        "fecha": t["date"].dt.strftime("%Y-%m-%d"), "liga": t["league"], "local": t["home_team"],
        "visitante": t["away_team"], "marcador": t["fthg"].astype(int).astype(str) + "-" + t["ftag"].astype(int).astype(str),
        "resultado": res, "p_1": P[:, 0], "p_X": P[:, 1], "p_2": P[:, 2], "pronostico": SEL[P.argmax(axis=1)],
        "acierto": (SEL[P.argmax(axis=1)] == res).astype(int),
        "mercado_p_1": mk[:, 0], "mercado_p_X": mk[:, 1], "mercado_p_2": mk[:, 2],
        "pronostico_mercado": np.where(np.isfinite(mk).all(axis=1), SEL[np.nan_to_num(mk, nan=-1).argmax(axis=1)], ""),
        "cuota_1": odds[:, 0], "cuota_X": odds[:, 1], "cuota_2": odds[:, 2],
        "apuesta_valor": vb, "edge": np.where(vb == "", np.nan, safe[i, best]), "beneficio_u": profit,
    }).round(4)
    detail["acierto_mercado"] = (detail["pronostico_mercado"] == detail["resultado"]).astype(int)

    # Métricas por modelo (solo partidos con todas las probabilidades disponibles)
    need = [c for cols_ in MODELS.values() for c in cols_[:3]]
    ok = t[need].notna().all(axis=1).to_numpy()
    ref = M.per_match_logloss(y[ok], t.loc[ok, list(MODELS["Mercado apertura"][:3])].to_numpy())
    rows = []
    for name, cols_ in MODELS.items():
        Pm = t.loc[ok, list(cols_[:3])].to_numpy(float)
        ll = M.per_match_logloss(y[ok], Pm)
        d, lo, hi = M.paired_bootstrap_diff(ll, ref) if ok.sum() > 1 else (np.nan,) * 3
        q = t.loc[ok, cols_[3]].to_numpy(float)
        yo = t.loc[ok, "y_over25"].to_numpy()
        qok = np.isfinite(q)
        rows.append({"modelo": name, "n": int(ok.sum()), "log_loss": ll.mean(), "brier": M.brier_multi(y[ok], Pm),
                     "accuracy": M.accuracy(y[ok], Pm), "vs_apertura": d, "ic95_inf": lo, "ic95_sup": hi,
                     "accuracy_over25": float(np.mean((q[qok] > 0.5) == yo[qok])) if qok.any() else np.nan})
    table = pd.DataFrame(rows)
    nb = int((vb != "").sum())
    summary = {
        "desde": desde, "hasta": hasta, "partidos": len(t), "modelo_principal": main, "umbral": thr,
        "aciertos_modelo": int(detail["acierto"].sum()), "accuracy_modelo": float(detail["acierto"].mean()),
        "accuracy_mercado": float(detail.loc[detail["pronostico_mercado"] != "", "acierto_mercado"].mean()),
        "apuestas_valor": {"n": nb, "ganadas": int(won[vb != ""].sum()), "beneficio_u": float(np.nansum(profit)),
                           "yield": float(np.nansum(profit) / nb) if nb else None},
        "metricas": table.round(5).to_dict(orient="records"),
        "nota": "vs_apertura < 0 = mejor que el mercado de apertura; IC 95% bootstrap. "
                "Con pocos partidos los resultados son muy ruidosos.",
    }
    rep = get_path(cfg, "reports")
    stem = f"examen_{d0:%Y%m%d}_{d1:%Y%m%d}"
    detail.to_csv(rep / f"{stem}.csv", index=False)
    # Mismo documento de seguimiento que el ledger (aciertos por confianza, liga, semana...) para el examen
    q = t[cols[3]].to_numpy(float).copy()
    q[~np.isfinite(q)] = t["lgbm_pover"].to_numpy(float)[~np.isfinite(q)]
    like = pd.DataFrame({
        "date": detail["fecha"], "league": detail["liga"], "home": detail["local"], "away": detail["visitante"],
        "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2], "p_over25": q,
        "mkt_p_home": mk[:, 0], "mkt_p_draw": mk[:, 1], "mkt_p_away": mk[:, 2],
        "odds_home": odds[:, 0], "odds_draw": odds[:, 1], "odds_away": odds[:, 2],
        "odds_over": t["open_over"].to_numpy(float), "odds_under": t["open_under"].to_numpy(float),
        "value_bet": vb, "value_bet_ou": "", "fthg": t["fthg"].to_numpy(float), "ftag": t["ftag"].to_numpy(float),
        "predicted_at": f"examen (entrenado con datos anteriores a {desde})",
    }).reset_index(drop=True)
    like["value_bet"] = like["value_bet"].where(like["value_bet"] != "", None).astype(object)
    like["value_bet_ou"] = pd.Series([None] * len(like), dtype=object)
    write_report(cfg, like, rep / f"{stem}.xlsx",
                 f"Examen {desde} → {hasta}: predicciones hechas solo con datos anteriores a {desde}")
    (rep / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 60):
        log.info("Partido a partido (primeros 40):\n%s", detail[["fecha", "liga", "local", "visitante", "marcador",
                 "p_1", "p_X", "p_2", "pronostico", "acierto", "apuesta_valor", "beneficio_u"]].head(40).to_string(index=False))
        log.info("Métricas del examen:\n%s", table.round(4).to_string(index=False))
    log.info("Modelo principal (%s): %d/%d aciertos (%.1f%%) | mercado %.1f%% | apuestas con valor: %d, "
             "beneficio %+.2f u (yield %s)", main, summary["aciertos_modelo"], len(t),
             100 * summary["accuracy_modelo"], 100 * summary["accuracy_mercado"], nb, np.nansum(profit),
             f"{100 * summary['apuestas_valor']['yield']:+.1f}%" if nb else "—")
    log.info("Detalle -> %s", rep / f"{stem}.csv")
    return summary
