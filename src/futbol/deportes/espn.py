"""Resultados, calendario y cuotas de varios deportes desde la API pública de ESPN (JSON, sin clave).

    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates=AAAA[MM]&limit=1000

- Una petición devuelve todos los eventos de un año (`chunk: year`) o de un mes (`chunk: month`).
- Los eventos pasados traen marcador; los próximos traen además cuotas (moneyline, total) cuando ESPN las
  publica. ESPN no da cuotas históricas: el mercado solo se puede comparar de aquí en adelante (ledger).
- Tenis: cada evento es un torneo con sus partidos dentro (`groupings` → `mens-singles`, `womens-singles`).

Caché: un CSV ya procesado por liga y periodo en data/raw/espn/<liga>/<periodo>.csv. Los periodos cerrados
no se vuelven a pedir; el periodo en curso (y el siguiente, si entra en la ventana) se refrescan siempre.
Cliente HTTP normal e identificado (config `http.user_agent`), con pausa entre peticiones.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_path
from ..ingest.http import make_session

log = logging.getLogger(__name__)

URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
COLS = ["liga", "event_id", "inicio_utc", "temporada", "tipo", "estado", "terminado", "programado", "neutral",
        "home_id", "home", "away_id", "away", "home_score", "away_score", "ganador", "torneo", "ronda",
        "odds_home", "odds_draw", "odds_away", "open_home", "open_draw", "open_away",
        "total", "odds_over", "odds_under", "casa"]
TEXT = ["liga", "event_id", "estado", "home_id", "home", "away_id", "away", "ganador", "torneo", "ronda", "casa"]
DRAW90 = {"STATUS_FINAL_PEN", "STATUS_FINAL_AET"}  # fútbol: a los 90' iban empatados


def leagues(cfg: dict, only: list[str] | None = None) -> list[dict]:
    lgs = (cfg.get("deportes") or {}).get("ligas") or []
    return [lg for lg in lgs if not only or lg["id"] in only]


def american_to_decimal(v) -> float:
    """Cuota americana (+150, -200, 'EVEN') -> decimal (2.50, 1.50, 2.00)."""
    if v is None:
        return np.nan
    s = str(v).strip().upper().replace("−", "-")
    if s in ("EVEN", "EV", "PK"):
        return 2.0
    try:
        a = float(s)
    except ValueError:
        return np.nan
    if a == 0 or not np.isfinite(a):
        return np.nan
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def _ml(o: dict, side: str, when: str) -> float:
    try:
        return american_to_decimal(o["moneyline"][side][when]["odds"])
    except (KeyError, TypeError):
        return np.nan


def _odds(c: dict) -> dict:
    out = {k: np.nan for k in COLS[COLS.index("odds_home"):COLS.index("casa")]}
    out["casa"] = None
    o = next((x for x in (c.get("odds") or []) if x), None)
    if not o:
        return out
    for side in ("home", "draw", "away"):
        out[f"odds_{side}"] = _ml(o, side, "close")
        out[f"open_{side}"] = _ml(o, side, "open")
    if not np.isfinite(out["odds_draw"]):
        out["odds_draw"] = american_to_decimal((o.get("drawOdds") or {}).get("moneyLine"))
    out["total"] = pd.to_numeric(o.get("overUnder"), errors="coerce")
    for side in ("over", "under"):
        try:
            out[f"odds_{side}"] = american_to_decimal(o["total"][side]["close"]["odds"])
        except (KeyError, TypeError):
            pass
    out["casa"] = (o.get("provider") or {}).get("name")
    return out


def _score(x: dict) -> float:
    s = x.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    return pd.to_numeric(s, errors="coerce")


def _row(lg: dict, c: dict, home: dict, away: dict, names: tuple[str, str], scores: tuple[float, float],
         season: dict, torneo=None, ronda=None) -> dict:
    st = (c.get("status") or {}).get("type") or {}
    done = bool(st.get("completed"))
    hs, as_ = scores
    winner = "home" if home.get("winner") else ("away" if away.get("winner") else None)
    if done and lg.get("empate") and (st.get("name") in DRAW90 or hs == as_):
        winner = "draw"
    elif done and winner is None and np.isfinite(hs) and np.isfinite(as_):
        winner = "home" if hs > as_ else ("away" if as_ > hs else "draw")
    return {"liga": lg["id"], "event_id": str(c.get("id")), "inicio_utc": c.get("date"),
            "temporada": season.get("year"), "tipo": season.get("type"),
            "estado": st.get("name"), "terminado": done, "programado": st.get("state") == "pre"
            and st.get("name") == "STATUS_SCHEDULED", "neutral": bool(c.get("neutralSite", False)) or lg["sport"] == "tennis",
            "home_id": str(home.get("id")), "home": names[0], "away_id": str(away.get("id")), "away": names[1],
            "home_score": hs, "away_score": as_, "ganador": winner if done else None, "torneo": torneo,
            "ronda": ronda, **_odds(c)}


def parse_team_events(payload: dict, lg: dict) -> list[dict]:
    rows = []
    for e in payload.get("events", []):
        for c in e.get("competitions", [])[:1]:
            comp = {x.get("homeAway"): x for x in c.get("competitors", [])}
            if "home" not in comp or "away" not in comp:
                continue
            h, a = comp["home"], comp["away"]
            names = tuple((x.get("team") or {}).get("displayName") for x in (h, a))
            if not all(names):
                continue
            h, a = ({**x, "id": (x.get("team") or {}).get("id", x.get("id"))} for x in (h, a))
            rows.append(_row(lg, {**c, "id": e.get("id", c.get("id"))}, h, a, names, (_score(h), _score(a)),
                             e.get("season") or {}))
    return rows


def parse_tennis_events(payload: dict, lg: dict) -> list[dict]:
    """Partidos individuales. El orden de los dos jugadores se fija por la paridad del id del partido,
    para que 'home'/'away' no dependa de quién ganó ni de cómo los ordene ESPN."""
    rows = []
    for e in payload.get("events", []):
        for g in e.get("groupings", []):
            if (g.get("grouping") or {}).get("slug") != lg.get("grouping", "mens-singles"):
                continue
            for c in g.get("competitions", []):
                ps = c.get("competitors", [])
                if len(ps) != 2 or not all((p.get("athlete") or {}).get("displayName") for p in ps):
                    continue
                try:
                    swap = int(str(c.get("id"))[-1]) % 2 == 1
                except ValueError:
                    swap = False
                p1, p2 = (ps[1], ps[0]) if swap else (ps[0], ps[1])
                sets = tuple(float(sum(1 for s in p.get("linescores", []) if s.get("winner"))) for p in (p1, p2))
                rows.append(_row(lg, c, p1, p2, (p1["athlete"]["displayName"], p2["athlete"]["displayName"]), sets,
                                 e.get("season") or {}, e.get("name"),
                                 (c.get("round") or {}).get("displayName")))
    return rows


def parse(payload: dict, lg: dict) -> pd.DataFrame:
    rows = parse_tennis_events(payload, lg) if lg["sport"] == "tennis" else parse_team_events(payload, lg)
    return pd.DataFrame(rows, columns=COLS)


def _dir(cfg: dict, lg: dict) -> Path:
    d = get_path(cfg, "raw") / "espn" / lg["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_keys(lg: dict, start_year: int, until: dt.date) -> list[str]:
    if lg.get("chunk", "year") == "year":
        return [str(y) for y in range(start_year, until.year + 1)]
    return [f"{y}{m:02d}" for y in range(start_year, until.year + 1) for m in range(1, 13)
            if (y, m) <= (until.year, until.month)]


def chunk_end(key: str) -> dt.date:
    if len(key) == 4:
        return dt.date(int(key), 12, 31)
    y, m = int(key[:4]), int(key[4:])
    return (dt.date(y + (m == 12), m % 12 + 1, 1)) - dt.timedelta(days=1)


def download(cfg: dict, refresh: bool = False, only: list[str] | None = None, today: dt.date | None = None) -> dict:
    """Descarga lo que falte. Devuelve {liga: nº de periodos descargados} (−1 si la liga falló)."""
    dcfg = cfg.get("deportes") or {}
    today = today or dt.date.today()
    until = today + dt.timedelta(days=int(dcfg.get("dias_adelante", 2)) + 1)
    session = make_session(cfg)
    pause = float(dcfg.get("pausa_segundos", 0.25))
    status = {}
    for lg in leagues(cfg, only):
        n = 0
        try:
            for key in chunk_keys(lg, int(dcfg.get("desde", 2016)), until):
                dest = _dir(cfg, lg) / f"{key}.csv"
                closed = chunk_end(key) + dt.timedelta(days=int(lg.get("dias_cierre", 25 if lg["sport"] == "tennis" else 4))) < today
                if dest.exists() and closed and not refresh:
                    continue
                r = None
                for attempt in range(3):
                    try:
                        r = session.get(URL.format(sport=lg["sport"], league=lg["league"]),
                                        params={"dates": key, "limit": 1000}, timeout=cfg["http"]["timeout"])
                        if r.status_code < 500:
                            break
                    except Exception as e:  # noqa: BLE001
                        log.debug("ESPN %s %s intento %d: %s", lg["id"], key, attempt + 1, e)
                    time.sleep(1 + attempt)
                if r is None or r.status_code != 200:
                    raise RuntimeError(f"{key}: HTTP {getattr(r, 'status_code', 'sin respuesta')}")
                df = parse(r.json(), lg)
                tmp = dest.with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                tmp.replace(dest)
                n += 1
                time.sleep(pause)
            status[lg["id"]] = n
            if n > 2:
                log.info("ESPN %s: %d periodos descargados", lg["id"], n)
        except Exception as e:  # noqa: BLE001  una liga caída no debe tumbar las demás
            log.warning("ESPN %s falló (%s). Se usa lo que haya en caché.", lg["id"], str(e)[:160])
            status[lg["id"]] = -1
    return status


def load_events(cfg: dict, only: list[str] | None = None) -> pd.DataFrame:
    """Todos los eventos en caché (un partido por fila, sin duplicados, en orden cronológico)."""
    frames = []
    for lg in leagues(cfg, only):
        for p in sorted(_dir(cfg, lg).glob("*.csv")):
            df = pd.read_csv(p, dtype={c: str for c in TEXT})
            if len(df):
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLS)
    ev = pd.concat(frames, ignore_index=True)
    ev["inicio_utc"] = pd.to_datetime(ev["inicio_utc"], utc=True, errors="coerce", format="mixed")
    for c in ("terminado", "programado", "neutral"):
        ev[c] = ev[c].astype(str).str.lower() == "true"
    ev = ev.dropna(subset=["inicio_utc", "home_id", "away_id"])
    wo = ev["estado"] == "STATUS_WALKOVER"  # tenis: no se jugó (un jugador no se presentó) -> sin resultado
    ev.loc[wo, "terminado"] = False
    ev.loc[wo, "ganador"] = None
    skip = {lg["id"]: set(lg.get("excluir_tipos") or []) for lg in leagues(cfg, only)}  # p. ej. pretemporada
    tipo = pd.to_numeric(ev["tipo"], errors="coerce")
    ev = ev[[not (t in skip.get(l, ())) for l, t in zip(ev["liga"], tipo)]]
    # un mismo partido puede venir en dos periodos: gana la versión terminada / más reciente
    ev = ev.sort_values(["terminado", "inicio_utc"]).drop_duplicates(["liga", "event_id"], keep="last")
    return ev.sort_values(["inicio_utc", "liga", "event_id"], kind="mergesort").reset_index(drop=True)


def final_scores(ev: pd.DataFrame) -> pd.DataFrame:
    """Marcador para liquidar el ledger (índice: liga + '_' + event_id). El signo del marcador siempre
    coincide con el ganador oficial (p. ej. tenis con retirada, o fútbol decidido en penaltis = empate a 90')."""
    d = ev[ev["terminado"] & ev["ganador"].notna()].copy()
    hs, as_ = d["home_score"].fillna(0).to_numpy(float), d["away_score"].fillna(0).to_numpy(float)
    w = d["ganador"].to_numpy()
    lo = np.minimum(hs, as_)
    bad_h, bad_a, bad_d = (w == "home") & ~(hs > as_), (w == "away") & ~(as_ > hs), (w == "draw") & (hs != as_)
    hs = np.where(bad_h, lo + 1, np.where(bad_a | bad_d, lo, hs))
    as_ = np.where(bad_a, lo + 1, np.where(bad_h | bad_d, lo, as_))
    return pd.DataFrame({"fthg": hs, "ftag": as_}, index=(d["liga"] + "_" + d["event_id"]).to_numpy())
