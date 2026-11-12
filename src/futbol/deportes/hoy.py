"""`futbol hoy`: un solo comando que actualiza datos, liquida lo pendiente y da los pronósticos del día.

Para cada partido programado (hoy y los próximos `deportes.dias_adelante` días) de todas las ligas:

- `p_modelo`:  probabilidad del modelo Elo + forma (deportes/modelo.py).
- `p_mercado`: probabilidad implícita en las cuotas actuales de ESPN (sin el margen de la casa), si las hay.
- `prob`:      combinación de ambas (peso `deportes.peso_mercado` para el mercado; solo modelo si no hay cuotas).
               El mercado ya incorpora lesiones, alineaciones y noticias que el modelo no ve.
- `acierto_historico`: cuánto acertó ese nivel de confianza en esa liga en la validación (años no vistos).

La lista sale ordenada de más a menos probable, se guarda en reports/hoy.csv, se registra en el ledger
y actualiza reports/seguimiento.xlsx.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..config import get_path
from ..evaluation.informe import confidence, save_workbook, thresholds, write_report
from ..evaluation.ledger import append_to_ledger, load_ledger, settle
from . import espn, modelo

log = logging.getLogger(__name__)

SEL = np.array(["1", "X", "2"])


def market_probs(odds: np.ndarray) -> np.ndarray:
    """Cuotas decimales (n × 3; empate NaN si no existe) -> probabilidades sin margen (NaN si faltan)."""
    inv = 1.0 / np.asarray(odds, float)
    two_way = ~np.isfinite(inv[:, 1])
    inv[two_way, 1] = 0.0
    tot = inv.sum(axis=1, keepdims=True)
    p = inv / tot
    p[~np.isfinite(p).all(axis=1) | (tot[:, 0] < 1.0) | (tot[:, 0] > 1.4)] = np.nan  # cuotas incompletas o absurdas
    return p


def blend(p_model: np.ndarray, p_market: np.ndarray, w: float) -> np.ndarray:
    """Media geométrica ponderada (w = peso del mercado). Sin mercado, queda el modelo."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp((1 - w) * np.log(np.clip(p_model, 1e-9, 1)) + w * np.log(np.clip(p_market, 1e-9, 1)))
    g[p_model == 0] = 0.0
    g = g / g.sum(axis=1, keepdims=True)
    no_mkt = ~np.isfinite(p_market).all(axis=1)
    g[no_mkt] = p_model[no_mkt]
    return g


def select(out: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Selección del día: los pocos pronósticos pensados para apostar (apuesta simple al ganador).

    Entra un partido solo si: la probabilidad es alta (`seleccion.prob_min`), ese nivel de confianza ha
    acertado de verdad en esa liga en años no vistos (`acierto_hist_min`, con al menos `n_hist_min` casos),
    no es un empate, y, si hay cuotas, el mercado tiene al mismo favorito. Como mucho `max_por_dia` por día.
    """
    s = cfg.get("seleccion") or {}
    ok = ((out["p_pick"] >= float(s.get("prob_min", thresholds(cfg)[0])))
          & (out["acierto_historico"] >= float(s.get("acierto_hist_min", 0.70)))
          & (out["n_historico"] >= int(s.get("n_hist_min", 100))))
    if not s.get("permitir_empate", False):
        ok &= out["pick"] != "X"
    if s.get("exigir_mercado_de_acuerdo", True):
        mk = out[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float)
        has = np.isfinite(mk).all(axis=1)
        ok &= ~has | (SEL[np.nan_to_num(mk, nan=-1).argmax(axis=1)] == out["pick"].to_numpy())
    rank = out[ok].groupby("date")["p_pick"].rank(ascending=False, method="first")
    ok.loc[rank.index[rank > int(s.get("max_por_dia", 5))]] = False
    return ok.to_numpy()


def friendly(out: pd.DataFrame) -> pd.DataFrame:
    """Columnas legibles para reports/hoy.xlsx."""
    i = np.arange(len(out))
    k = out["pick"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    pm = out[["mod_p_home", "mod_p_draw", "mod_p_away"]].to_numpy(float)[i, k] if len(out) else []
    mk = out[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float)[i, k] if len(out) else []
    cuota = out["cuota_pick"].to_numpy(float)
    return pd.DataFrame({
        "fecha": out["date"].to_numpy(), "hora": out["hora"].to_numpy(), "deporte": out["deporte"].to_numpy(),
        "liga": out["liga_nombre"].to_numpy(), "partido": (out["home"] + " vs " + out["away"]).to_numpy(),
        "apostar_a": out["pick_nombre"].to_numpy(), "probabilidad": out["p_pick"].to_numpy(),
        "confianza": out["confianza"].to_numpy(), "cuota_minima": out["cuota_minima"].to_numpy(),
        "cuota_espn": cuota,
        "supera_minima": np.where(~np.isfinite(cuota), "sin cuota", np.where(cuota >= out["cuota_minima"], "Sí", "No")),
        "acierto_historico": out["acierto_historico"].to_numpy(), "casos_historicos": out["n_historico"].to_numpy(),
        "prob_modelo": pm, "prob_mercado": mk,
        "nota": np.where(np.isfinite(cuota), "", "Sin cuotas en ESPN: solo modelo; mira la cuota de tu casa"),
    })


NOTES = ["Solo apuesta simple al ganador del partido (nada de combinadas). Si la lista está vacía, hoy no se apuesta.",
         "cuota_minima = 1 / probabilidad: si tu casa paga menos que eso, a la larga se pierde aunque se acierte mucho."]


def ensure_data(cfg: dict, only: list[str] | None = None) -> pd.DataFrame:
    """Descarga lo que falte (la primera vez, todo el historial: unos minutos) y valida las ligas sin parámetros."""
    lgs = espn.leagues(cfg, only)
    if not lgs:
        raise ValueError("No hay ligas configuradas en `deportes.ligas` (config.yaml).")
    first = not any((get_path(cfg, "raw") / "espn" / lg["id"]).glob("*.csv") for lg in lgs)
    if first:
        log.info("Primera ejecución: descargando el historial de %d ligas desde ESPN (tarda unos minutos)...", len(lgs))
    status = espn.download(cfg, only=only)
    down = [k for k, v in status.items() if v < 0]
    if down:
        log.warning("Sin datos nuevos de: %s (se usa la caché).", ", ".join(down))
    ev = espn.load_events(cfg, only)
    if ev.empty:
        raise RuntimeError("No se pudo descargar ningún dato de ESPN y no hay caché.")
    params = modelo.load_params(cfg)
    missing = [lg["id"] for lg in lgs if lg["id"] not in params]
    if missing:
        log.info("Validando el modelo en %d ligas (walk-forward)...", len(missing))
        modelo.evaluate(cfg, ev, missing if params else only)
    return ev


def predict_all(cfg: dict, ev: pd.DataFrame, now: pd.Timestamp, days: int, only: list[str] | None = None) -> pd.DataFrame:
    params = modelo.load_params(cfg)
    tz = dt.datetime.now().astimezone().tzinfo
    end = (now.tz_convert(tz).normalize() + pd.Timedelta(days=days + 1)).tz_convert("UTC")
    cal_path = get_path(cfg, "reports") / "deportes_calibracion.csv"
    cal = pd.read_csv(cal_path).set_index(["liga", "confianza"]) if cal_path.exists() else None
    w = float((cfg.get("deportes") or {}).get("peso_mercado", 0.7))
    thr = float(cfg["backtest"]["threshold"])
    frames = []
    for lg in espn.leagues(cfg, only):
        if lg["id"] not in params:
            continue
        g = ev[ev["liga"] == lg["id"]]
        soon = g["programado"] & ~g["terminado"] & (g["inicio_utc"] > now) & (g["inicio_utc"] < end)
        if not soon.any():
            continue
        # solo se dejan como "programados" los de la ventana: el resto del calendario no hace falta
        up = modelo.predict_upcoming(g[g["terminado"] | soon], lg, params[lg["id"]])
        if up.empty:
            continue
        pm = up[["mod_p_home", "mod_p_draw", "mod_p_away"]].to_numpy(float)
        odds = up[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
        mk = market_probs(odds)
        p = blend(pm, mk, w)
        i, top = np.arange(len(up)), p.argmax(axis=1)
        has_mkt = np.isfinite(mk).all(axis=1)
        local = up["inicio_utc"].dt.tz_convert(tz)
        edges = p * odds - 1
        safe = np.where(np.isfinite(edges), edges, -np.inf)
        best = safe.argmax(axis=1)
        conf = confidence(p[i, top], cfg)
        names = np.column_stack([up["home"].to_numpy(), np.full(len(up), "Empate"), up["away"].to_numpy()])
        hist = [float(cal.loc[(lg["id"], c), "acierto"]) if cal is not None and (lg["id"], c) in cal.index else np.nan
                for c in conf]
        n_hist = [int(cal.loc[(lg["id"], c), "partidos"]) if cal is not None and (lg["id"], c) in cal.index else 0
                  for c in conf]
        frames.append(pd.DataFrame({
            "date": local.dt.strftime("%Y-%m-%d").to_numpy(), "hora": local.dt.strftime("%H:%M").to_numpy(),
            "deporte": lg["deporte"], "league": lg["id"], "liga_nombre": lg["nombre"],
            "home": up["home"].to_numpy(), "away": up["away"].to_numpy(),
            "modelo": np.where(has_mkt, "Elo+forma+mercado", "Elo+forma"),
            "pick": SEL[top], "pick_nombre": names[i, top], "p_pick": p[i, top], "confianza": conf,
            "cuota_pick": odds[i, top], "cuota_minima": 1.0 / p[i, top],
            "acierto_historico": hist, "n_historico": n_hist,
            "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "p_over25": np.nan,
            "mod_p_home": pm[:, 0], "mod_p_draw": pm[:, 1], "mod_p_away": pm[:, 2],
            "mkt_p_home": mk[:, 0], "mkt_p_draw": mk[:, 1], "mkt_p_away": mk[:, 2],
            "odds_home": odds[:, 0], "odds_draw": odds[:, 1], "odds_away": odds[:, 2],
            "odds_over": np.nan, "odds_under": np.nan,
            "edge_pick": edges[i, top], "value_bet": np.where(safe[i, best] > thr, SEL[best], ""), "value_bet_ou": "",
            "casa_cuotas": up["casa"].to_numpy(), "torneo": up["torneo"].to_numpy(), "ronda": up["ronda"].to_numpy(),
            "event_id": up["event_id"].to_numpy(),
            "inicio_utc": up["inicio_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            **{c: up[c].to_numpy() for c in up.columns if c.startswith("x_")},
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["p_pick", "inicio_utc"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def hoy(cfg: dict, days: int | None = None, top: int = 25, record: bool = True, only: list[str] | None = None,
        now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    days = int((cfg.get("deportes") or {}).get("dias_adelante", 1) if days is None else days)
    cfg.setdefault("deportes", {})["dias_adelante"] = days  # la descarga cubre la misma ventana
    ev = ensure_data(cfg, only)
    if record and not load_ledger(cfg).empty:
        log.info("===== Liquidando predicciones anteriores =====")
        settle(cfg, events=ev)
    out = predict_all(cfg, ev, now, days, only)
    rep = get_path(cfg, "reports")
    if out.empty:
        log.warning("No hay partidos programados en los próximos %d días en las ligas configuradas.", days)
        return out
    out.insert(out.columns.get_loc("confianza") + 1, "seleccion", select(out, cfg))
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    out.to_csv(rep / "hoy.csv", index=False)
    sel = out[out["seleccion"]]
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    save_workbook({"Selección del día": friendly(sel), "Todos los partidos": friendly(out).assign(
        seleccion=np.where(out["seleccion"], "Sí", "No"))}, rep / "hoy.xlsx",
        f"Selección del día — generada el {stamp} ({len(sel)} de {len(out)} partidos)", NOTES)
    if record:
        write_report(cfg, append_to_ledger(cfg, out.drop(columns=[c for c in out.columns if c.startswith("x_")])))
    show = out.head(top)[["date", "hora", "deporte", "liga_nombre", "home", "away", "pick_nombre", "p_pick",
                          "confianza", "cuota_pick", "acierto_historico", "modelo"]].rename(columns={
                              "date": "fecha", "liga_nombre": "liga", "home": "local", "away": "visitante",
                              "pick_nombre": "pronóstico", "p_pick": "prob", "cuota_pick": "cuota",
                              "acierto_historico": "acierto_hist"})
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 28):
        log.info("Pronósticos de los próximos %d días: %d partidos de %d ligas. Los %d más probables:\n%s",
                 days, len(out), out["league"].nunique(), len(show), show.to_string(index=False))
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40):
        if len(sel):
            log.info("SELECCIÓN DEL DÍA (%d): apuesta simple al ganador, solo estos:\n%s", len(sel),
                     friendly(sel)[["fecha", "hora", "deporte", "partido", "apostar_a", "probabilidad", "cuota_minima",
                                    "cuota_espn", "acierto_historico", "nota"]].to_string(index=False))
        else:
            log.info("SELECCIÓN DEL DÍA: ningún partido cumple los criterios. Hoy no se apuesta.")
    log.info("Para verlo: %s (selección y todos los partidos) y %s (aciertos y fallos). "
             "Más probable no significa rentable: apuesta solo si tu casa paga al menos la cuota_minima.",
             rep / "hoy.xlsx", rep / "seguimiento.xlsx")
    return out
