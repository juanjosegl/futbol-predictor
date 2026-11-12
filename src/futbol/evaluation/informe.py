"""Documento de seguimiento (Excel): cada predicción, si acertó o falló, y el análisis de los aciertos.

Se genera a partir del ledger (`reports/seguimiento.xlsx`, tras `predict` y `settle`) y a partir de un
examen sobre partidos pasados (`reports/examen_<desde>_<hasta>.xlsx`). Hojas:

- Resumen:        totales, % de acierto (con IC 95%), comparación con el mercado y beneficio.
- Predicciones:   una fila por partido: pronóstico, probabilidad, confianza, cuota, marcador y estado.
- Selección:      rendimiento de la "selección del día" (lo único pensado para apostar) frente al resto.
- Modelo vs mercado: ¿funciona el modelo local? Modelo solo, mercado y combinación sobre los mismos partidos.
- Por confianza:  ¿acierta más cuando dice estar más seguro? ¿y da beneficio?
- Por deporte, Por liga, Por pronóstico (1/X/2), Por semana, Over-Under 2.5.

"Pronóstico" = resultado más probable según el modelo. "Confianza" depende de esa probabilidad
(config `tracking.confianza_alta` / `tracking.confianza_media`). Acertar mucho NO implica ganar dinero:
los favoritos pagan poco; por eso cada tabla trae también el beneficio de apostar 1 unidad por partido.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_path
from .ledger import SEL, _is_valid, load_ledger, score

log = logging.getLogger(__name__)

LEVELS = ["Alta", "Media", "Baja"]


def thresholds(cfg: dict) -> tuple[float, float]:
    t = cfg.get("tracking", {}) or {}
    return float(t.get("confianza_alta", 0.70)), float(t.get("confianza_media", 0.55))


def confidence(p, cfg: dict) -> np.ndarray:
    """Etiqueta de confianza según la probabilidad del pronóstico."""
    hi, mid = thresholds(cfg)
    p = np.asarray(p, float)
    return np.where(p >= hi, "Alta", np.where(p >= mid, "Media", np.where(np.isfinite(p), "Baja", "")))


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza 95% (Wilson) para una proporción k/n."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tracking_frame(led: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Convierte un ledger (ya puntuado con `score`) en la tabla legible del documento."""
    n = len(led)
    i = np.arange(n)

    def num(c):
        return pd.to_numeric(led[c], errors="coerce").to_numpy(float) if c in led else np.full(n, np.nan)

    P = np.column_stack([num("p_home"), num("p_draw"), num("p_away")])
    ok = np.isfinite(P).all(axis=1)
    pick = np.where(ok, np.nan_to_num(P, nan=-1).argmax(axis=1), 0)
    odds = np.column_stack([num("odds_home"), num("odds_draw"), num("odds_away")])
    mk = np.column_stack([num("mkt_p_home"), num("mkt_p_draw"), num("mkt_p_away")])
    q = num("p_over25")
    over = q > 0.5
    done = led["resultado"].notna().to_numpy()
    hit = num("acierto")
    hg, ag = num("fthg"), num("ftag")
    marcador = [f"{int(a)}-{int(b)}" if d else "" for a, b, d in zip(np.nan_to_num(hg), np.nan_to_num(ag), done)]
    t = pd.DataFrame({
        "fecha": pd.to_datetime(led["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "deporte": (led["deporte"].fillna("Fútbol") if "deporte" in led else pd.Series("Fútbol", index=led.index)).to_numpy(),
        "liga": (led["liga_nombre"].fillna(led["league"]) if "liga_nombre" in led else led["league"]).to_numpy(),
        "local": led["home"].to_numpy(), "visitante": led["away"].to_numpy(),
        "pronostico": np.where(ok, SEL[pick], ""),
        "pronostico_nombre": np.where(~ok, "", np.column_stack([led["home"].to_numpy(), np.full(n, "Empate"),
                                                                 led["away"].to_numpy()])[i, pick]),
        "prob": np.where(ok, P[i, pick], np.nan),
        "seleccion": np.where((led["seleccion"].astype(str).str.lower() == "true").to_numpy(), "Sí", "No")
        if "seleccion" in led else "No",
        "confianza": confidence(np.where(ok, P[i, pick], np.nan), cfg), "cuota": odds[i, pick],
        "p_1": P[:, 0], "p_X": P[:, 1], "p_2": P[:, 2],
        "pronostico_mercado": np.where(np.isfinite(mk).all(axis=1), SEL[np.nan_to_num(mk, nan=-1).argmax(axis=1)], ""),
        "pronostico_ou": np.where(np.isfinite(q), np.where(over, "Over 2.5", "Under 2.5"), ""),
        "prob_ou": np.where(over, q, 1 - q), "cuota_ou": np.where(over, num("odds_over"), num("odds_under")),
        "apuesta_valor": led["value_bet"].fillna("").to_numpy() if "value_bet" in led else "",
        "apuesta_valor_ou": led["value_bet_ou"].fillna("").to_numpy() if "value_bet_ou" in led else "",
        "marcador": marcador, "resultado": led["resultado"].fillna("").to_numpy(),
        "estado": np.where(~done, "Pendiente", np.where(hit == 1, "Acertó", "Falló")),
        "acierto": hit, "acierto_mercado": num("acierto_mercado"), "acierto_ou": num("acierto_ou"),
        "beneficio_pronostico": num("profit_pick"), "beneficio_valor": num("profit_value_bet"),
        "beneficio_valor_ou": num("profit_value_bet_ou"),
        "apuesta_real": led["apuesta_real"].fillna("").to_numpy(),
        "apuesta_tipo": np.where(led["apuesta_real"].isna(), "", led["apuesta_tipo"].fillna("real")),
        "cuota_real": num("cuota_real"),
        "stake_real": num("stake_real"), "beneficio_real": num("profit_real"),
        "ll_modelo": num("ll_modelo"), "ll_mercado": num("ll_mercado"),
        "predicho_el": led["predicted_at"].to_numpy() if "predicted_at" in led else "",
        "valida": _is_valid(led).to_numpy() if "antes_del_partido" in led else True,
    })
    return t.sort_values(["fecha", "liga", "local"], kind="mergesort").reset_index(drop=True)


def _group(d: pd.DataFrame, by, order: list | None = None) -> pd.DataFrame:
    """Tabla de aciertos y beneficio por grupo (apostando 1 unidad al pronóstico de cada partido)."""
    rows = []
    for key, g in d.groupby(by, sort=True, observed=True):
        n, k = len(g), float(g["acierto"].sum())
        lo, hi = wilson(k, n)
        b = g["beneficio_pronostico"].dropna()
        rows.append({by if isinstance(by, str) else "grupo": key, "partidos": n, "aciertos": int(k),
                     "pct_acierto": k / n, "ic95_inf": lo, "ic95_sup": hi, "prob_media_modelo": g["prob"].mean(),
                     "pct_acierto_mercado": g["acierto_mercado"].mean(), "cuota_media": g["cuota"].mean(),
                     "apuestas_con_cuota": len(b), "beneficio_u": b.sum() if len(b) else np.nan,
                     "yield": b.mean() if len(b) else np.nan})
    out = pd.DataFrame(rows, columns=[by, "partidos", "aciertos", "pct_acierto", "ic95_inf", "ic95_sup",
                                      "prob_media_modelo", "pct_acierto_mercado", "cuota_media",
                                      "apuestas_con_cuota", "beneficio_u", "yield"])
    if order and len(out):
        out = out.set_index(by).reindex([o for o in order if o in set(out[by])]).reset_index()
    return out


def analyze(led: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Tablas de análisis. Solo cuentan los partidos con resultado y predicción hecha antes del día del partido."""
    scored = score(led)
    t = tracking_frame(scored, cfg)
    d = t[(t["estado"] != "Pendiente") & t["valida"]].copy()
    n, k = len(d), float(d["acierto"].sum())
    lo, hi = wilson(k, n)
    both = d.dropna(subset=["ll_mercado"])
    pk, vb, vbo, real = (d[c].dropna() for c in ("beneficio_pronostico", "beneficio_valor", "beneficio_valor_ou",
                                                  "beneficio_real"))
    # Apuestas anotadas con `futbol apuesta`: cuentan todas las liquidadas, reales y de papel por separado
    bets = t[t["apuesta_real"] != ""].copy()
    bets["apostado_a"] = [bets.loc[i, "local"] if a == "1" else bets.loc[i, "visitante"] if a == "2" else
                          {"X": "Empate"}.get(a, a) for i, a in zip(bets.index, bets["apuesta_real"])]
    bets["estado_apuesta"] = np.where(bets["beneficio_real"].isna(), "Pendiente",
                                      np.where(bets["beneficio_real"] > 0, "Ganada", "Perdida"))
    done_b = bets[bets["beneficio_real"].notna()]
    real_b, paper_b = done_b[done_b["apuesta_tipo"] != "papel"], done_b[done_b["apuesta_tipo"] == "papel"]

    def bet_line(b):
        if not len(b):
            return "ninguna liquidada todavía"
        st = b["stake_real"].fillna(1.0)
        y = b["beneficio_real"] / st
        err = f" ± {100 * 1.96 * y.std() / np.sqrt(len(b)):.1f}" if len(b) > 1 else ""
        return (f"{int((b['beneficio_real'] > 0).sum())} ganadas de {len(b)}; beneficio {b['beneficio_real'].sum():+.2f} "
                f"con {st.sum():.2f} apostado (yield {100 * b['beneficio_real'].sum() / st.sum():.1f}{err}%)")
    alta = d[d["confianza"] == "Alta"]
    sel = d[d["seleccion"] == "Sí"]
    sel_lo, sel_hi = wilson(float(sel["acierto"].sum()), len(sel))

    def pct(x):
        return f"{100 * x:.1f}%" if np.isfinite(x) else "—"

    def money(s):
        if not len(s):
            return "sin apuestas"
        err = f" ± {100 * 1.96 * s.std() / np.sqrt(len(s)):.1f}" if len(s) > 1 else ""
        return f"{s.sum():+.2f} u en {len(s)} apuestas (yield {100 * s.mean():.1f}{err}%)"

    hi_t, mid_t = thresholds(cfg)
    resumen = [
        ("Predicciones registradas", len(t)),
        ("Con resultado", int((t["estado"] != "Pendiente").sum())),
        ("Pendientes", int((t["estado"] == "Pendiente").sum())),
        ("Válidas para medir (hechas antes del partido)", n),
        ("Aciertos 1X2 del modelo", f"{int(k)} de {n} ({pct(k / n) if n else '—'})"),
        ("  intervalo de confianza 95% del % de acierto", f"{pct(lo)} – {pct(hi)}" if n else "—"),
        ("Aciertos 1X2 del mercado (favorito de las cuotas)", pct(both["acierto_mercado"].mean()) if len(both) else "—"),
        ("Log loss modelo / mercado (menor = mejor)",
         f"{both['ll_modelo'].mean():.4f} / {both['ll_mercado'].mean():.4f}" if len(both) else "—"),
        ("Aciertos Over/Under 2.5", pct(d["acierto_ou"].mean()) if d["acierto_ou"].notna().any() else "—"),
        ("Apostar 1 u al pronóstico de cada partido", money(pk)),
        (f"Solo confianza alta (prob ≥ {hi_t:.0%})",
         f"{int(alta['acierto'].sum())} de {len(alta)} ({pct(alta['acierto'].mean()) if len(alta) else '—'}); "
         + money(alta["beneficio_pronostico"].dropna())),
        ("SELECCIÓN DEL DÍA (lo único pensado para apostar)",
         f"{int(sel['acierto'].sum())} de {len(sel)} ({pct(sel['acierto'].mean())}, IC 95% {pct(sel_lo)} – {pct(sel_hi)}); "
         + money(sel["beneficio_pronostico"].dropna()) if len(sel) else "sin partidos liquidados todavía"),
        ("Apuestas de valor 1X2 (1 u)", money(vb)),
        ("Apuestas de valor Over/Under (1 u)", money(vbo)),
        ("MIS APUESTAS REALES", bet_line(real_b)),
        ("Mis apuestas de papel (sin dinero, 1 u cada una)", bet_line(paper_b)),
        ("", ""),
        ("Cómo leerlo", "Acertar mucho no es ganar dinero: los favoritos pagan poco. Mira el yield "
                        "(beneficio / apostado), no solo el % de acierto."),
        ("Tamaño de muestra", "El '±' del yield es su margen de error (95%): si el intervalo incluye el 0, el "
                              "resultado es compatible con el puro azar. Hacen falta cientos o miles de apuestas "
                              "para distinguir habilidad de suerte."),
        ("Niveles de confianza", f"Alta: prob ≥ {hi_t:.0%} · Media: {mid_t:.0%}–{hi_t:.0%} · Baja: < {mid_t:.0%}"),
    ]
    d["semana"] = (pd.to_datetime(d["fecha"]) - pd.to_timedelta(pd.to_datetime(d["fecha"]).dt.day_of_week, unit="D")
                   ).dt.strftime("%Y-%m-%d")
    semana = _group(d, "semana")
    if len(semana):
        semana["beneficio_acumulado_u"] = semana["beneficio_u"].fillna(0).cumsum()
        semana["pct_acierto_acumulado"] = semana["aciertos"].cumsum() / semana["partidos"].cumsum()
    ou = d[d["acierto_ou"].notna()].copy()
    ou["prob_modelo"] = pd.cut(ou["prob_ou"], [0.5, 0.55, 0.6, 0.65, 0.7, 1.0], include_lowest=True).astype(str)
    b_ou = np.where(ou["acierto_ou"] == 1, ou["cuota_ou"] - 1, -1.0)
    ou["b"] = np.where(np.isfinite(ou["cuota_ou"]), b_ou, np.nan)
    ou_tab = ou.groupby("prob_modelo", observed=True).agg(
        partidos=("acierto_ou", "size"), aciertos=("acierto_ou", "sum"), pct_acierto=("acierto_ou", "mean"),
        prob_media_modelo=("prob_ou", "mean"), beneficio_u=("b", "sum"), **{"yield": ("b", "mean")}).reset_index()
    return {
        "Resumen": pd.DataFrame(resumen, columns=["indicador", "valor"]),
        "Predicciones": t.drop(columns=["ll_modelo", "ll_mercado"]),
        "Mis apuestas": bets[["fecha", "deporte", "liga", "local", "visitante", "apostado_a", "apuesta_tipo",
                              "cuota_real", "stake_real", "estado_apuesta", "beneficio_real", "seleccion", "prob",
                              "marcador"]].rename(columns={"estado_apuesta": "estado", "prob": "prob_pronostico"}),
        "Selección": _group(d, "seleccion", ["Sí", "No"]),
        "Modelo vs mercado": model_vs_market(scored),
        "Por confianza": _group(d, "confianza", LEVELS),
        "Por deporte": _group(d, "deporte"),
        "Por liga": _group(d, "liga"),
        "Por pronóstico": _group(d, "pronostico", list(SEL)),
        "Por semana": semana,
        "Over-Under 2.5": ou_tab,
    }


def model_vs_market(scored: pd.DataFrame) -> pd.DataFrame:
    """¿Funciona el modelo local? Compara, sobre los mismos partidos ya jugados, el modelo solo (Elo + forma),
    el mercado (cuotas sin margen) y la combinación que se usa para pronosticar. Menor log loss = mejor."""
    cols = ["ambito", "fuente", "partidos", "pct_acierto", "log_loss", "prob_media_del_pronostico"]
    if "mod_p_home" not in scored or scored.empty:
        return pd.DataFrame(columns=cols)
    ok = scored["resultado"].notna() & (_is_valid(scored) if "antes_del_partido" in scored else True)
    s = scored[ok]
    y = s["resultado"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    src = {"Modelo local (Elo + forma)": ["mod_p_home", "mod_p_draw", "mod_p_away"],
           "Mercado (cuotas sin margen)": ["mkt_p_home", "mkt_p_draw", "mkt_p_away"],
           "Combinado (el pronóstico usado)": ["p_home", "p_draw", "p_away"]}
    P = {k: s[v].apply(pd.to_numeric, errors="coerce").to_numpy(float) for k, v in src.items()}
    fin = {k: np.isfinite(v).all(axis=1) for k, v in P.items()}
    rows = []
    for ambito, mask, names in (
            ("Partidos con cuotas", fin["Modelo local (Elo + forma)"] & fin["Mercado (cuotas sin margen)"], list(src)),
            ("Todos los partidos", fin["Modelo local (Elo + forma)"], ["Modelo local (Elo + forma)"])):
        for name in names:
            p = P[name][mask]
            yy = y[mask]
            if not len(yy):
                rows.append({"ambito": ambito, "fuente": name, "partidos": 0})
                continue
            i = np.arange(len(yy))
            rows.append({"ambito": ambito, "fuente": name, "partidos": len(yy),
                         "pct_acierto": float((p.argmax(axis=1) == yy).mean()),
                         "log_loss": float(-np.log(np.clip(p[i, yy], 1e-12, None)).mean()),
                         "prob_media_del_pronostico": float(p.max(axis=1).mean())})
    return pd.DataFrame(rows, columns=cols)


PCT = {"probabilidad", "prob_pronostico", "acierto_historico", "prob_modelo", "prob_mercado", "prob_media_del_pronostico", "pct_acierto", "ic95_inf", "ic95_sup", "prob_media_modelo", "pct_acierto_mercado", "yield", "prob", "p_1",
       "p_X", "p_2", "prob_ou", "pct_acierto_acumulado"}
MONEY = {"beneficio_u", "beneficio_acumulado_u", "beneficio_pronostico", "beneficio_valor", "beneficio_valor_ou",
         "beneficio_real"}
FILL = {"Acertó": "C6EFCE", "Falló": "FFC7CE", "Pendiente": "FFF2CC", "Ganada": "C6EFCE", "Perdida": "FFC7CE"}


def save_workbook(tables: dict[str, pd.DataFrame], path: Path, title: str = "", notes: list[str] | None = None) -> Path | None:
    """Escribe un Excel con una hoja por tabla. La primera hoja lleva el título y las notas encima."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    head = PatternFill("solid", fgColor="1F4E78")
    for k, (name, df) in enumerate(tables.items()):
        ws = wb.create_sheet(name)
        r0 = 1
        if k == 0 and title:
            ws.cell(1, 1, title).font = Font(bold=True, size=14)
            for r, note in enumerate(notes or [], 2):
                ws.cell(r, 1, note).font = Font(italic=True)
            r0 = 3 + len(notes or [])
        wide = k == 0 and list(df.columns) == ["indicador", "valor"]
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(r0, j, c)
            cell.font, cell.fill = Font(bold=True, color="FFFFFF"), head
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, row in enumerate(df.itertuples(index=False), r0 + 1):
            for j, (c, v) in enumerate(zip(df.columns, row), 1):
                if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                    v = None
                elif isinstance(v, np.generic):
                    v = v.item()
                cell = ws.cell(i, j, v)
                if c in PCT:
                    cell.number_format = "0.0%"
                elif c in MONEY:
                    cell.number_format = "+0.00;-0.00;0.00"
                elif isinstance(v, float):
                    cell.number_format = "0.00"
                if c == "estado" and v in FILL:
                    cell.fill = PatternFill("solid", fgColor=FILL[v])
        for j, c in enumerate(df.columns, 1):
            width = max([len(str(c))] + [len(str(v)) for v in df[c].head(300)]) + 2
            ws.column_dimensions[get_column_letter(j)].width = min(max(width, 9), 70 if wide else 34)
        ws.freeze_panes = ws.cell(r0 + 1, 6 if name == "Predicciones" else 1)
        if not wide and len(df):
            ws.auto_filter.ref = f"A{r0}:{get_column_letter(len(df.columns))}{r0 + len(df)}"
    try:
        wb.save(path)
    except PermissionError:
        log.warning("No se pudo escribir %s (¿está abierto en Excel?). Ciérralo y vuelve a ejecutar.", path)
        return None
    return Path(path)


def write_report(cfg: dict, led: pd.DataFrame | None = None, path: Path | None = None,
                 title: str = "Seguimiento de predicciones") -> Path | None:
    """Escribe el Excel de seguimiento. Sin argumentos usa reports/ledger.csv -> reports/seguimiento.xlsx."""
    led = load_ledger(cfg) if led is None else led
    if led.empty:
        log.info("No hay predicciones registradas todavía; no se genera el documento de seguimiento.")
        return None
    path = Path(path) if path else get_path(cfg, "reports") / "seguimiento.xlsx"
    out = save_workbook(analyze(led, cfg), path, title)
    if out:
        log.info("Documento de seguimiento -> %s", path)
    return out
