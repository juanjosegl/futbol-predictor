"""Registro de predicciones (reports/ledger.csv) y su liquidación con resultados reales.

Flujo de prueba en tiempo real ("paper trading"):
1. `futbol predict partidos.csv`  -> cada predicción se añade al ledger con la fecha/hora en que se hizo.
2. (Opcional) editas el ledger y rellenas `apuesta_real`, `cuota_real`, `stake_real` si apostaste.
3. Tras los partidos: `futbol download --refresh-current && futbol build && futbol settle`
   -> rellena resultados, aciertos, log loss y beneficio, y resume el rendimiento acumulado.

Solo cuentan para las métricas las predicciones hechas ANTES del día del partido (`antes_del_partido`).
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from ..config import get_path
from ..deportes import mercados

log = logging.getLogger(__name__)

USER_COLS = ["apuesta_real", "cuota_real", "stake_real", "apuesta_tipo"]  # apuesta_tipo: real | papel
RESULT_COLS = ["fthg", "ftag", "resultado", "over25", "acierto", "acierto_mercado", "acierto_ou", "acierto_rec",
               "ll_modelo", "ll_mercado", "profit_pick", "profit_rec", "profit_value_bet", "profit_value_bet_ou",
               "profit_real", "fuente_resultado"]
TEXT_COLS = ["value_bet", "value_bet_ou", "apuesta_real", "resultado", "fuente_resultado", "pick", "pick_ou",
             "confianza", "event_id", "deporte", "liga_nombre", "pick_nombre", "hora", "inicio_utc", "torneo", "ronda",
             "casa_cuotas", "league", "modelo", "seleccion", "nota", "apuesta_tipo", "rec", "rec_tipo", "rec_nombre"]
SEL = np.array(["1", "X", "2"])
SEL_OU = np.array(["Over", "Under"])


def _path(cfg: dict):
    return get_path(cfg, "reports") / "ledger.csv"


def load_ledger(cfg: dict) -> pd.DataFrame:
    p = _path(cfg)
    if not p.exists():
        return pd.DataFrame()
    return _ensure_cols(pd.read_csv(p, dtype={c: str for c in TEXT_COLS}))


def _ensure_cols(led: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas que falten (ledgers creados con versiones anteriores) con el tipo correcto."""
    for c in USER_COLS + RESULT_COLS + ["event_id"]:
        if c not in led:
            led[c] = pd.Series([None] * len(led), index=led.index, dtype=object) if c in TEXT_COLS else np.nan
    return led


def append_to_ledger(cfg: dict, preds: pd.DataFrame, now: dt.datetime | None = None) -> pd.DataFrame:
    """Añade predicciones. Si el partido ya estaba sin liquidar, se reemplaza por la nueva predicción
    (conservando lo que el usuario haya rellenado en las columnas de apuesta real)."""
    now = now or dt.datetime.now()
    new = preds.copy()
    pid = new["league"] + "_" + new["date"].astype(str) + "_" + new["home"] + "_" + new["away"]
    before = pd.Series(pd.to_datetime(new["date"]).dt.date > now.date(), index=new.index)
    if "event_id" in new:  # partidos con id y hora de inicio (ESPN): vale si se predijo antes de empezar
        has = new["event_id"].notna()
        pid = pid.where(~has, new["league"] + "_" + new["event_id"].astype(str))
        start = pd.to_datetime(new.get("inicio_utc"), utc=True, errors="coerce")
        now_utc = pd.Timestamp(now).tz_localize(dt.datetime.now().astimezone().tzinfo).tz_convert("UTC") \
            if pd.Timestamp(now).tzinfo is None else pd.Timestamp(now).tz_convert("UTC")
        before = before.where(start.isna(), start > now_utc)
    new.insert(0, "pred_id", pid)
    new.insert(1, "predicted_at", now.isoformat(timespec="seconds"))
    new.insert(2, "antes_del_partido", before)
    old = load_ledger(cfg)
    n_new = int((~new["pred_id"].isin(old["pred_id"])).sum()) if not old.empty else len(new)
    if not old.empty:
        settled = old["resultado"].notna() if "resultado" in old else pd.Series(False, index=old.index)
        replace = old["pred_id"].isin(new["pred_id"]) & ~settled
        keep_user = old.loc[replace].set_index("pred_id")[[c for c in USER_COLS if c in old]]
        if "seleccion" in old and "seleccion" in new:  # si estuvo en la selección del día, lo sigue estando
            known = (old["home"] != "TBD") & (old["away"] != "TBD")  # cuadro de tenis sin decidir: no contaba
            was = set(old.loc[replace & known & (old["seleccion"].astype(str).str.lower() == "true"), "pred_id"])
            new["seleccion"] = new["seleccion"].astype(bool) | new["pred_id"].isin(was)
        new = new[~new["pred_id"].isin(old.loc[settled, "pred_id"])]  # no tocar partidos ya liquidados
        new = new.set_index("pred_id")
        for c in keep_user.columns:
            new.loc[new.index.intersection(keep_user.index), c] = keep_user[c]
        new = new.reset_index()
        old = old[~replace]
    new = _ensure_cols(new)
    led = pd.concat([old, new], ignore_index=True) if not old.empty else new
    led = led.sort_values(["date", "league", "home"], kind="mergesort")
    led.to_csv(_path(cfg), index=False)
    log.info("Ledger: %d predicciones registradas (%d nuevas, %d actualizadas) -> %s", len(led), n_new,
             len(new) - n_new, _path(cfg))
    return led


def _settle_bets(sel: pd.Series, names: list[str], won: np.ndarray, odds: np.ndarray, stake=1.0) -> np.ndarray:
    """Beneficio de cada apuesta. `sel`: selección por fila (vacío = sin apuesta); `won`: matriz booleana
    (filas × len(names)); `odds`: matriz con la cuota de cada selección, o un vector con la cuota tomada."""
    idx = sel.map(lambda v: names.index(v.strip()) if isinstance(v, str) and v.strip() in names else -1).to_numpy(int)
    has = idx >= 0
    i, k = np.arange(len(idx)), np.where(has, idx, 0)
    o = np.asarray(odds, float)
    o = o[i, k] if o.ndim == 2 else o
    profit = np.where(won[i, k], stake * (o - 1.0), -stake).astype(float)
    profit[~has | ~np.isfinite(o) | ~np.isfinite(profit)] = np.nan
    return profit


def score(led: pd.DataFrame) -> pd.DataFrame:
    """Calcula resultado, aciertos, log loss y beneficios de las filas que ya tienen marcador (`fthg`, `ftag`).

    - `acierto`: el pronóstico (resultado más probable según el modelo) coincide con el resultado.
    - `profit_pick`: beneficio de apostar 1 unidad al pronóstico a la cuota registrada.
    - `acierto_rec` / `profit_rec`: lo mismo para la apuesta recomendada (`rec`: ganador, total de goles,
      ambos marcan...; en las filas antiguas, sin `rec`, es el pronóstico al ganador).
    - `profit_value_bet(_ou)`: beneficio de 1 unidad en la apuesta de valor (si la hubo).
    - `profit_real`: beneficio de la apuesta que el usuario dice haber hecho (`apuesta_real`, `cuota_real`, `stake_real`).
    """
    led = _ensure_cols(led.copy())
    n = len(led)
    hg, ag = pd.to_numeric(led["fthg"], errors="coerce").to_numpy(float), pd.to_numeric(led["ftag"], errors="coerce").to_numpy(float)
    done = np.isfinite(hg) & np.isfinite(ag)
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    over = (hg + ag) > 2.5
    i = np.arange(n)

    def cols(names):
        return np.column_stack([pd.to_numeric(led[c], errors="coerce").to_numpy(float) if c in led
                                else np.full(n, np.nan) for c in names])

    P = cols(["p_home", "p_draw", "p_away"])
    mk = cols(["mkt_p_home", "mkt_p_draw", "mkt_p_away"])
    q = cols(["p_over25"])[:, 0]
    odds = cols(["odds_home", "odds_draw", "odds_away"])
    odds_ou = cols(["odds_over", "odds_under"])
    okP, okM = np.isfinite(P).all(axis=1), np.isfinite(mk).all(axis=1)
    pick = np.where(okP, np.nan_to_num(P, nan=-1).argmax(axis=1), 0)
    won = np.column_stack([y == 0, y == 1, y == 2])
    won_ou = np.column_stack([over, ~over])

    def put(col, values, ok):
        led[col] = np.where(done & ok, values, np.nan)

    led["resultado"] = pd.Series(np.where(done, SEL[y], None), index=led.index, dtype=object)
    put("over25", over.astype(float), True)
    put("acierto", (pick == y).astype(float), okP)
    put("acierto_mercado", (np.nan_to_num(mk, nan=-1).argmax(axis=1) == y).astype(float), okM)
    put("acierto_ou", ((q > 0.5) == over).astype(float), np.isfinite(q))
    put("ll_modelo", -np.log(np.clip(np.nan_to_num(P[i, y], nan=1.0), 1e-15, None)), okP)
    put("ll_mercado", -np.log(np.clip(np.nan_to_num(mk[i, y], nan=1.0), 1e-15, None)), okM)
    put("profit_pick", _settle_bets(pd.Series(SEL[pick]), list(SEL), won, odds), okP)
    rec, cuota_rec = rec_bets(led)
    names = mercados.CODES + list(SEL_OU)  # todos los tipos de apuesta + los códigos antiguos Over/Under (2.5)
    won_all = np.column_stack([mercados.won(c, np.nan_to_num(hg), np.nan_to_num(ag)) for c in names])
    k = np.array([names.index(c) if c in names else -1 for c in rec])
    put("acierto_rec", won_all[i, np.maximum(k, 0)].astype(float), k >= 0)
    put("profit_rec", _settle_bets(pd.Series(rec), names, won_all, cuota_rec), k >= 0)
    put("profit_value_bet", _settle_bets(led["value_bet"], list(SEL), won, odds), True)
    put("profit_value_bet_ou", _settle_bets(led["value_bet_ou"], list(SEL_OU), won_ou, odds_ou), True)
    stake = pd.to_numeric(led["stake_real"], errors="coerce").fillna(1.0).to_numpy(float)
    put("profit_real", _settle_bets(led["apuesta_real"], names, won_all,
                                    pd.to_numeric(led["cuota_real"], errors="coerce").to_numpy(float), stake), True)
    return led


def rec_bets(led: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(código, cuota registrada) de la apuesta recomendada de cada fila. Las predicciones anteriores a los
    tipos de apuesta no tienen `rec`: su apuesta recomendada era el pronóstico al ganador."""
    n = len(led)

    def num(c):
        return pd.to_numeric(led[c], errors="coerce").to_numpy(float) if c in led else np.full(n, np.nan)

    P = np.column_stack([num("p_home"), num("p_draw"), num("p_away")])
    okP = np.isfinite(P).all(axis=1)
    pick = np.nan_to_num(P, nan=-1).argmax(axis=1)
    old = led["rec"].isna().to_numpy() if "rec" in led else np.ones(n, bool)
    rec = np.where(old, np.where(okP, SEL[pick], ""), led["rec"].fillna("").to_numpy() if "rec" in led else "")
    odds = np.column_stack([num("odds_home"), num("odds_draw"), num("odds_away")])
    return rec.astype(object), np.where(old, odds[np.arange(n), pick], num("cuota_rec"))


def _norm(s) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower().strip()


def record_bet(cfg: dict, equipo: str, cuota: float, monto: float | None = None, a: str | None = None,
               fecha: str | None = None, papel: bool = False) -> dict:
    """Anota una apuesta (real o de papel) sobre un partido pendiente del ledger, buscándolo por nombre.

    Por defecto se apuesta a la apuesta recomendada del sistema; `a` permite otra: 1, X, 2, el nombre de uno de
    los dos contrincantes, o un tipo de apuesta (1X, X2, 12, O1.5, O2.5, U2.5, U3.5, BTTS, NOBTTS...).
    Siempre es una apuesta simple a un solo partido.
    """
    from .informe import write_report

    if not (cuota and cuota > 1):
        raise ValueError("La cuota debe ser decimal y mayor que 1 (p. ej. 1.35).")
    if not papel and not (monto and monto > 0):
        raise ValueError("Indica --monto (lo apostado) o usa --papel para una apuesta simulada sin dinero.")
    led = load_ledger(cfg)
    if led.empty:
        raise FileNotFoundError("No hay predicciones registradas. Ejecuta primero `futbol hoy`.")
    q = _norm(equipo)
    m = led["resultado"].isna() & (led["home"].map(_norm).str.contains(q, regex=False)
                                   | led["away"].map(_norm).str.contains(q, regex=False))
    if fecha:
        m &= led["date"].astype(str) == fecha
    cand = led[m]
    if cand.empty:
        raise ValueError(f"No hay ningún partido pendiente con '{equipo}'. Mira reports/hoy.xlsx o ejecuta `futbol hoy`.")
    if len(cand) > 1:
        lst = "; ".join(f"{r.date} {r.home} vs {r.away}" for r in cand.itertuples())
        raise ValueError(f"Hay {len(cand)} partidos pendientes con '{equipo}': {lst}. Sé más específico o añade --fecha AAAA-MM-DD.")
    i = cand.index[0]
    r = led.loc[i]
    p = np.array([r["p_home"], r["p_draw"], r["p_away"]], float)
    sel = r["rec"] if "rec" in led and isinstance(r["rec"], str) and r["rec"] else SEL[int(np.nan_to_num(p, nan=-1).argmax())]
    if a:
        an = _norm(a)
        sel = mercados.parse_code(an) or ("1" if an in _norm(r["home"]) else "2" if an in _norm(r["away"]) else None)
        if sel is None:
            raise ValueError(f"No entiendo --a '{a}'. Usa el nombre de {r['home']} / {r['away']}, o un tipo de apuesta: "
                             "1, X, 2, 1X, X2, 12, O1.5, O2.5, O3.5, U2.5, U3.5, BTTS (ambos marcan), NOBTTS.")
    led.loc[i, ["apuesta_real", "apuesta_tipo"]] = [sel, "papel" if papel else "real"]
    led.loc[i, ["cuota_real", "stake_real"]] = [float(cuota), 1.0 if papel else float(monto)]
    led.to_csv(_path(cfg), index=False)
    write_report(cfg, led)
    name = bet_name(sel, r["home"], r["away"])
    code = mercados.ALIASES.get(sel, sel)
    prob = {"1": p[0], "X": p[1], "2": p[2]}.get(sel, np.nan)
    if not np.isfinite(prob) and mercados.col(code) in led:
        prob = pd.to_numeric(r[mercados.col(code)], errors="coerce")
    info = {"partido": f"{r['home']} vs {r['away']}", "fecha": r["date"], "apuesta_a": name, "cuota": float(cuota),
            "monto": None if papel else float(monto), "tipo": "papel" if papel else "real",
            "probabilidad_modelo": float(prob) if np.isfinite(prob) else None,
            "cuota_minima": float(1 / prob) if np.isfinite(prob) and prob > 0 else None,
            "en_seleccion": str(r.get("seleccion")).lower() == "true"}
    start = pd.to_datetime(r.get("inicio_utc"), utc=True, errors="coerce")
    msg = "Apuesta %s anotada: %s (%s) a cuota %.2f%s." % (
        info["tipo"], info["apuesta_a"], info["partido"], cuota, "" if papel else f", monto {monto:g}")
    if info["cuota_minima"]:
        msg += " Probabilidad según el sistema %.1f%% -> cuota mínima %.2f: %s." % (
            100 * prob, info["cuota_minima"], "la supera" if cuota >= info["cuota_minima"] else
            "NO la alcanza (a la larga esta cuota pierde aunque se acierte a menudo)")
    if not info["en_seleccion"]:
        msg += " Ojo: este partido NO está en la selección del día."
    if pd.notna(start) and start < pd.Timestamp.now(tz="UTC"):
        msg += " Ojo: el partido ya había empezado."
    log.info(msg)
    return info


def bet_name(code, home: str, away: str) -> str:
    """Nombre legible de una apuesta: el contrincante si es al ganador; si no, el tipo de apuesta."""
    return {"1": home, "X": "Empate", "2": away}.get(code) or mercados.nombre(code, home, away)


def _is_valid(led: pd.DataFrame) -> pd.Series:
    """Predicciones hechas antes del día del partido (las únicas que cuentan para las métricas)."""
    return led["antes_del_partido"].astype(str).str.lower() == "true"


def settle(cfg: dict, events: pd.DataFrame | None = None) -> dict:
    """Rellena los marcadores de los partidos pendientes y recalcula aciertos y beneficios.

    `events`: partidos de ESPN ya cargados (los pasa `futbol hoy`); si no, se leen de la caché.

    El marcador sale de los datos descargados (`futbol download` + `build`). Si no hay datos nuevos
    (p. ej. football-data bloqueado y sin token de football-data.org) puedes escribir a mano `fthg` y
    `ftag` (goles del local y del visitante) en reports/ledger.csv y volver a ejecutar `futbol settle`.
    """
    from ..features.build import load_dataset
    from .informe import analyze, write_report

    led = load_ledger(cfg)
    if led.empty:
        raise FileNotFoundError("No hay ledger todavía. Usa primero `futbol predict`.")
    pending = led["resultado"].isna()
    key = None
    if (pending & led["event_id"].isna()).any():  # predicciones del pipeline de fútbol (football-data)
        try:
            hist = load_dataset(cfg)
            hist = hist[hist["played"]]
            key = hist.set_index(["league", "home_team", "away_team"]).sort_index()
        except FileNotFoundError:
            log.warning("No hay dataset (`futbol build`); solo se liquidan los marcadores escritos a mano.")
    finals = None
    if led["event_id"].notna().any():  # predicciones multideporte: el marcador sale de ESPN por id de partido
        try:
            from ..deportes import espn

            finals = espn.final_scores(espn.load_events(cfg) if events is None else events)
        except Exception as e:  # noqa: BLE001
            log.warning("No se pudieron leer los resultados de ESPN: %s", e)
    dates = pd.to_datetime(led["date"])
    for i in led.index[pending]:
        r = led.loc[i]
        cand = pd.DataFrame()
        eid = f"{r['league']}_{r['event_id']}" if pd.notna(r["event_id"]) else None
        if eid is not None:
            if finals is not None and eid in finals.index:
                led.loc[i, ["fthg", "ftag"]] = finals.loc[eid, ["fthg", "ftag"]].to_numpy(float)
                led.loc[i, "fuente_resultado"] = "ESPN"
            elif pd.notna(r["fthg"]) and pd.notna(r["ftag"]):
                led.loc[i, "fuente_resultado"] = "manual"
            continue
        if key is not None and (r["league"], r["home"], r["away"]) in key.index:
            cand = key.loc[[(r["league"], r["home"], r["away"])]]
            cand = cand[(cand["date"] - dates[i]).abs() <= pd.Timedelta(days=3)]
        if not cand.empty:
            led.loc[i, ["fthg", "ftag"]] = [float(cand.iloc[0]["fthg"]), float(cand.iloc[0]["ftag"])]
            led.loc[i, "fuente_resultado"] = "datos"
        elif pd.notna(r["fthg"]) and pd.notna(r["ftag"]):
            led.loc[i, "fuente_resultado"] = "manual"
    led = score(led)
    n_new = int((pending & led["resultado"].notna()).sum())
    led.to_csv(_path(cfg), index=False)

    s = led[led["resultado"].notna()]
    honest = s[_is_valid(s)]
    summary = {"liquidadas_ahora": n_new, "total_liquidadas": len(s), "validas_antes_del_partido": len(honest),
               "pendientes": int(led["resultado"].isna().sum())}
    if len(honest):
        both = honest.dropna(subset=["ll_mercado"])
        pk = honest["profit_pick"].dropna()
        vb = honest["profit_value_bet"].dropna()
        vbo = honest["profit_value_bet_ou"].dropna()
        real = s.loc[s["apuesta_tipo"].fillna("real") != "papel", "profit_real"].dropna()  # dinero real: todas
        summary.update({
            "accuracy_modelo": float(honest["acierto"].mean()),
            "accuracy_mercado": float(both["acierto_mercado"].mean()) if len(both) else None,
            "accuracy_over_under": float(honest["acierto_ou"].mean()),
            "log_loss_modelo": float(honest["ll_modelo"].mean()),
            "log_loss_mercado_mismas": float(both["ll_mercado"].mean()) if len(both) else None,
            "log_loss_modelo_mismas": float(both["ll_modelo"].mean()) if len(both) else None,
            "pronosticos_1u": {"n": len(pk), "beneficio_u": float(pk.sum()),
                               "yield": float(pk.mean()) if len(pk) else None},
            "apuesta_recomendada": {"n": int(honest["acierto_rec"].notna().sum()),
                                    "aciertos": int(honest["acierto_rec"].sum()),
                                    "accuracy": float(honest["acierto_rec"].mean()),
                                    "beneficio_u": float(honest["profit_rec"].sum()),
                                    "apuestas_con_cuota": int(honest["profit_rec"].notna().sum())},
            "value_bets_1x2": {"n": len(vb), "beneficio_u": float(vb.sum()),
                               "yield": float(vb.mean()) if len(vb) else None},
            "value_bets_ou": {"n": len(vbo), "beneficio_u": float(vbo.sum()),
                              "yield": float(vbo.mean()) if len(vbo) else None},
            "apuestas_reales": {"n": len(real), "beneficio": float(real.sum()),
                                "apostado": float(s.loc[real.index, "stake_real"].fillna(1).astype(float).sum())},
            "por_confianza": analyze(led, cfg)["Por confianza"].to_dict(orient="records"),
        })
    (get_path(cfg, "reports") / "ledger_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log.info("Ledger liquidado: %s", json.dumps({k: v for k, v in summary.items() if k != "por_confianza"},
                                                 ensure_ascii=False, indent=2, default=float))
    write_report(cfg, led)
    if len(honest) < 200:
        log.info("Ojo: con %d partidos la muestra es pequeña; el azar domina. Hacen falta cientos para concluir.",
                 len(honest))
    return summary
