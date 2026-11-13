"""`futbol hoy`: un solo comando que actualiza datos, liquida lo pendiente y da los pronósticos del día.

Para cada partido programado (hoy y los próximos `deportes.dias_adelante` días) de todas las ligas:

- `p_modelo`:  probabilidad del modelo Elo + forma (deportes/modelo.py).
- `p_mercado`: probabilidad implícita en las cuotas actuales de ESPN (sin el margen de la casa), si las hay.
- `prob`:      combinación de ambas (peso `deportes.peso_mercado` para el mercado; solo modelo si no hay cuotas).
               El mercado ya incorpora lesiones, alineaciones y noticias que el modelo no ve.
- `acierto_historico`: cuánto acertó ese nivel de confianza en esa liga en la validación (años no vistos).
- `rec`:       la apuesta recomendada del partido. En fútbol se comparan varios tipos de apuesta simple
               (ganador, doble oportunidad, más/menos goles, ambos marcan; ver mercados.py) y se elige la más
               probable que todavía paga algo; en los demás deportes es el ganador.

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
from . import espn, mercados, modelo

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
    """Selección del día: las pocas apuestas pensadas para apostar (siempre apuesta simple, una por partido).

    Entra la apuesta recomendada de un partido solo si: su probabilidad es alta (`seleccion.prob_min`), ese
    tipo de apuesta con esa probabilidad ha acertado de verdad en esa liga en años no vistos
    (`acierto_hist_min`, con al menos `n_hist_min` casos), no es un empate, y, si es una apuesta al ganador y
    hay cuotas, el mercado tiene al mismo favorito. Como mucho `max_por_dia` por día.
    """
    s = cfg.get("seleccion") or {}
    new = "rec" in out  # sin apuesta recomendada (versiones anteriores): el pronóstico al ganador
    rec, p = (out["rec"], out["p_rec"]) if new else (out["pick"], out["p_pick"])
    hist, n_hist = (out["hist_rec"], out["n_hist_rec"]) if new else (out["acierto_historico"], out["n_historico"])
    ok = ((p >= float(s.get("prob_min", thresholds(cfg)[0])))
          & (hist >= float(s.get("acierto_hist_min", 0.70)))
          & (n_hist >= int(s.get("n_hist_min", 100))))
    if not s.get("permitir_empate", False):
        ok &= rec != "X"
    if s.get("exigir_mercado_de_acuerdo", True):
        mk = out[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float)
        has = np.isfinite(mk).all(axis=1) & rec.isin(mercados.WINNER).to_numpy()
        ok &= ~has | (SEL[np.nan_to_num(mk, nan=-1).argmax(axis=1)] == rec.to_numpy())
    rank = p[ok].groupby(out.loc[ok, "date"]).rank(ascending=False, method="first")
    ok.loc[rank.index[rank > int(s.get("max_por_dia", 5))]] = False
    return ok.to_numpy()


def friendly(out: pd.DataFrame) -> pd.DataFrame:
    """Columnas legibles para reports/hoy.xlsx: lo que hace falta para decidir, primero."""
    i = np.arange(len(out))
    k = out["pick"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    pm = out[["mod_p_home", "mod_p_draw", "mod_p_away"]].to_numpy(float)[i, k] if len(out) else []
    mk = out[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float)[i, k] if len(out) else []
    cuota = out["cuota_rec"].to_numpy(float)
    xg = out[["xg_home", "xg_away"]].to_numpy(float)
    return pd.DataFrame({
        "fecha": out["date"].to_numpy(), "hora": out["hora"].to_numpy(), "deporte": out["deporte"].to_numpy(),
        "partido": (out["home"] + " vs " + out["away"]).to_numpy(),
        "tipo_apuesta": out["rec_tipo"].to_numpy(), "apostar_a": out["rec_nombre"].to_numpy(),
        "probabilidad": out["p_rec"].to_numpy(), "cuota_minima": out["cuota_min_rec"].to_numpy(),
        "cuota_espn": cuota,
        "supera_minima": np.where(~np.isfinite(cuota), "sin cuota", np.where(cuota >= out["cuota_min_rec"], "Sí", "No")),
        "acierto_historico": out["hist_rec"].to_numpy(), "casos_historicos": out["n_hist_rec"].to_numpy(),
        "ganador_probable": out["pick_nombre"].to_numpy(), "prob_ganador": out["p_pick"].to_numpy(),
        "goles_esperados": [f"{a:.1f} - {b:.1f}" if np.isfinite(a) else "" for a, b in xg],
        "confianza": out["confianza"].to_numpy(), "liga": out["liga_nombre"].to_numpy(),
        "prob_modelo": pm, "prob_mercado": mk,
        "nota": np.where(np.isfinite(cuota), "", "ESPN no da cuota para esta apuesta: mira la de tu casa"),
    })


def markets_sheet(out: pd.DataFrame) -> pd.DataFrame:
    """Hoja «Tipos de apuesta»: para cada partido de fútbol, todas las apuestas con probabilidad ≥ 50%."""
    rows = []
    for d in out[out[mercados.col("O2.5")].notna()].to_dict("records"):
        for code in mercados.CODES:
            p = d.get(mercados.col(code), np.nan)
            if not np.isfinite(p) or p < 0.5:
                continue
            is_rec = code == d["rec"]
            rows.append({"fecha": d["date"], "hora": d["hora"], "partido": f"{d['home']} vs {d['away']}",
                         "tipo_apuesta": mercados.tipo(code), "apuesta": mercados.nombre(code, d["home"], d["away"]),
                         "probabilidad": p, "cuota_minima": 1 / p,
                         "acierto_historico": d.get(f"x_h_{code}", np.nan), "casos_historicos": d.get(f"x_n_{code}", 0),
                         "recomendada": "Sí" if is_rec else "",
                         "en_seleccion": "Sí" if is_rec and d.get("seleccion") else ""})
    cols = ["fecha", "hora", "partido", "tipo_apuesta", "apuesta", "probabilidad", "cuota_minima",
            "acierto_historico", "casos_historicos", "recomendada", "en_seleccion"]
    t = pd.DataFrame(rows, columns=cols)
    return t.sort_values(["fecha", "hora", "partido", "probabilidad"], ascending=[True, True, True, False],
                         kind="mergesort").reset_index(drop=True)


NOTES = ["Siempre apuesta simple, una por partido (nada de combinadas). `apostar_a` es el tipo de apuesta que el "
         "sistema ve más probable en ese partido: ganador, doble oportunidad, más/menos goles o ambos marcan.",
         "Si la selección está vacía, hoy no se apuesta.",
         "cuota_minima = 1 / probabilidad: si tu casa paga menos que eso, a la larga se pierde aunque se acierte mucho.",
         "acierto_historico: cuántas veces acertó ese tipo de apuesta con esa probabilidad en esa liga (o en todas "
         "las ligas de fútbol, si en esa hay pocos casos), en años que el sistema no había visto."]


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
    missing = [lg["id"] for lg in lgs if lg["id"] not in params
               or (lg["sport"] == "soccer" and "g_eta" not in params[lg["id"]])]  # falta validar los goles
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
    mcal_path = get_path(cfg, "reports") / "deportes_mercados.csv"
    mcal, pool = {}, {}  # (liga, mercado, tramo) -> (acierto, partidos); `pool`: todas las ligas de fútbol juntas
    if mcal_path.exists():
        m = pd.read_csv(mcal_path, dtype={"mercado": str})
        mcal = dict(zip(zip(m["liga"], m["mercado"], m["tramo"].round(2)), zip(m["acierto"], m["partidos"])))
        g = m.assign(ok=m["acierto"] * m["partidos"], tramo=m["tramo"].round(2)).groupby(["mercado", "tramo"])[["ok", "partidos"]].sum()
        pool = {k: (float(r["ok"] / r["partidos"]), int(r["partidos"])) for k, r in g.iterrows()}
    n_min = int((cfg.get("mercados") or {}).get("n_hist_min", 100))
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
        if not up.empty:  # tenis: cuadro sin decidir ("TBD"); todavía no se sabe quién juega
            up = up[(up["home"] != "TBD") & (up["away"] != "TBD")]
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
        # --- tipos de apuesta y apuesta recomendada ---
        line = up["total"].to_numpy(float)
        o_ou = up[["odds_over", "odds_under"]].to_numpy(float)
        xg = np.full((len(up), 2), np.nan)
        if lg["sport"] == "soccer" and "xg_home" in up:
            rho = float(params[lg["id"]].get("g_rho", 0.0))
            lam, mu = mercados.reconcile(up["xg_home"].to_numpy(float), up["xg_away"].to_numpy(float), p[:, 0], p[:, 2], rho)
            half = np.isfinite(line) & (np.abs(line % 1 - 0.5) < 1e-9)  # solo líneas x.5 (sin devolución)
            mk_ou = market_probs(np.column_stack([o_ou[:, 0], np.full(len(up), np.nan), o_ou[:, 1]]))
            if (half & np.isfinite(mk_ou[:, 0])).any():  # el total de goles también se acerca al del mercado
                mod_over = mercados._p_over(mercados.score_matrix(lam, mu, rho), np.where(half, line, 2.5))
                target = blend(np.column_stack([mod_over, np.zeros(len(up)), 1 - mod_over]), mk_ou, w)[:, 0]
                target = np.where(half & np.isfinite(mk_ou[:, 0]), target, np.nan)
                lam, mu = mercados.reconcile(lam, mu, p[:, 0], p[:, 2], rho, np.where(half, line, np.nan), target)
            probs = mercados.market_probs(p, mercados.score_matrix(lam, mu, rho))
            xg = np.column_stack([lam, mu])
        else:
            probs = mercados.market_probs(p, None)
        if not lg.get("empate"):
            probs[["X", "1X", "X2", "12"]] = np.nan
        H = pd.DataFrame(np.nan, index=probs.index, columns=probs.columns)
        N = pd.DataFrame(0, index=probs.index, columns=probs.columns)
        for code in probs.columns:
            found = [mcal.get((lg["id"], code, t)) for t in mercados.tramo(probs[code])]
            if lg["sport"] == "soccer":  # pocos casos en esta liga: vale la validación de todas las ligas de fútbol
                found = [pool.get((code, t), f) if f is None or f[1] < n_min else f
                         for f, t in zip(found, mercados.tramo(probs[code]))]
            if code in ("1", "2"):  # sin validación por tramos (otros deportes): la del nivel de confianza
                lvl = confidence(probs[code].to_numpy(float), cfg)
                found = [f or ((float(cal.loc[(lg["id"], c), "acierto"]), int(cal.loc[(lg["id"], c), "partidos"]))
                               if cal is not None and (lg["id"], c) in cal.index else None)
                         for f, c in zip(found, lvl)]
            H[code] = [f[0] if f else np.nan for f in found]
            N[code] = [int(f[1]) if f else 0 for f in found]
        rec = mercados.recommend(probs, H, N, cfg)
        ou_code = np.where(np.isfinite(line), [f"{x:g}" for x in np.nan_to_num(line)], "")
        cuota_rec = np.select([rec["rec"] == "1", rec["rec"] == "X", rec["rec"] == "2",
                               rec["rec"] == np.char.add("O", ou_code), rec["rec"] == np.char.add("U", ou_code)],
                              [odds[:, 0], odds[:, 1], odds[:, 2], o_ou[:, 0], o_ou[:, 1]], np.nan)
        is25 = np.isfinite(line) & (line == 2.5)
        e_ou = np.column_stack([probs["O2.5"], probs["U2.5"]]) * o_ou - 1
        s_ou = np.where(np.isfinite(e_ou) & is25[:, None], e_ou, -np.inf)
        frames.append(pd.DataFrame({
            "date": local.dt.strftime("%Y-%m-%d").to_numpy(), "hora": local.dt.strftime("%H:%M").to_numpy(),
            "deporte": lg["deporte"], "league": lg["id"], "liga_nombre": lg["nombre"],
            "home": up["home"].to_numpy(), "away": up["away"].to_numpy(),
            "modelo": np.where(has_mkt, "Elo+forma+mercado", "Elo+forma"),
            "pick": SEL[top], "pick_nombre": names[i, top], "p_pick": p[i, top], "confianza": conf,
            "cuota_pick": odds[i, top], "cuota_minima": 1.0 / p[i, top],
            "acierto_historico": hist, "n_historico": n_hist,
            "rec": rec["rec"].to_numpy(), "rec_tipo": rec["rec_tipo"].to_numpy(),
            "rec_nombre": [mercados.nombre(c, h, a) for c, h, a in zip(rec["rec"], up["home"], up["away"])],
            "p_rec": rec["p_rec"].to_numpy(), "cuota_min_rec": 1.0 / rec["p_rec"].to_numpy(), "cuota_rec": cuota_rec,
            "hist_rec": rec["hist_rec"].to_numpy(), "n_hist_rec": rec["n_hist_rec"].to_numpy(),
            "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "p_over25": probs["O2.5"].to_numpy(),
            "xg_home": xg[:, 0], "xg_away": xg[:, 1], "linea_goles": line,
            "cuota_over_linea": o_ou[:, 0], "cuota_under_linea": o_ou[:, 1],
            "mod_p_home": pm[:, 0], "mod_p_draw": pm[:, 1], "mod_p_away": pm[:, 2],
            "mkt_p_home": mk[:, 0], "mkt_p_draw": mk[:, 1], "mkt_p_away": mk[:, 2],
            "odds_home": odds[:, 0], "odds_draw": odds[:, 1], "odds_away": odds[:, 2],
            "odds_over": np.where(is25, o_ou[:, 0], np.nan), "odds_under": np.where(is25, o_ou[:, 1], np.nan),
            "edge_pick": edges[i, top], "value_bet": np.where(safe[i, best] > thr, SEL[best], ""),
            "value_bet_ou": np.where(s_ou.max(axis=1) > thr, np.array(["Over", "Under"])[s_ou.argmax(axis=1)], ""),
            "casa_cuotas": up["casa"].to_numpy(), "torneo": up["torneo"].to_numpy(), "ronda": up["ronda"].to_numpy(),
            "event_id": up["event_id"].to_numpy(),
            "inicio_utc": up["inicio_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            **{mercados.col(c): probs[c].to_numpy() for c in probs.columns},
            **{c: up[c].to_numpy() for c in up.columns if c.startswith("x_")},
            **{f"x_h_{c}": H[c].to_numpy() for c in H.columns}, **{f"x_n_{c}": N[c].to_numpy() for c in N.columns},
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["p_rec", "p_pick", "inicio_utc"], ascending=[False, False, True],
                           kind="mergesort").reset_index(drop=True)


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
        seleccion=np.where(out["seleccion"], "Sí", "No")), "Tipos de apuesta (fútbol)": markets_sheet(out)},
        rep / "hoy.xlsx", f"Selección del día — generada el {stamp} ({len(sel)} de {len(out)} partidos)", NOTES)
    if record:
        write_report(cfg, append_to_ledger(cfg, out.drop(columns=[c for c in out.columns if c.startswith("x_")])))
    show = out.head(top)[["date", "hora", "deporte", "liga_nombre", "home", "away", "rec_nombre", "p_rec",
                          "cuota_rec", "hist_rec", "pick_nombre", "p_pick"]].rename(columns={
                              "date": "fecha", "liga_nombre": "liga", "home": "local", "away": "visitante",
                              "rec_nombre": "apuesta recomendada", "p_rec": "prob", "cuota_rec": "cuota",
                              "hist_rec": "acierto_hist", "pick_nombre": "ganador probable", "p_pick": "prob_ganador"})
    with pd.option_context("display.width", 280, "display.max_columns", 20, "display.max_colwidth", 34):
        log.info("Pronósticos de los próximos %d días: %d partidos de %d ligas. Las %d apuestas más probables:\n%s",
                 days, len(out), out["league"].nunique(), len(show), show.to_string(index=False))
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40):
        if len(sel):
            log.info("SELECCIÓN DEL DÍA (%d): apuesta simple, solo estas:\n%s", len(sel),
                     friendly(sel)[["fecha", "hora", "deporte", "partido", "tipo_apuesta", "apostar_a", "probabilidad",
                                    "cuota_minima", "cuota_espn", "acierto_historico"]].to_string(index=False))
        else:
            log.info("SELECCIÓN DEL DÍA: ningún partido cumple los criterios. Hoy no se apuesta.")
    log.info("Para verlo en el navegador: `futbol ver` (abre %s y %s; también hay .xlsx). "
             "Más probable no significa rentable: apuesta solo si tu casa paga al menos la cuota_minima.",
             rep / "hoy.html", rep / "seguimiento.html")
    return out
