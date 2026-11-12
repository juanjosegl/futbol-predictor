"""Tests del módulo multideporte: lectura de ESPN, modelo Elo + forma, mezcla con el mercado y ledger."""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd

from futbol.deportes import espn, modelo
from futbol.deportes.hoy import blend, market_probs
from futbol.evaluation.ledger import append_to_ledger, load_ledger, settle

SOCCER = {"id": "col.1", "nombre": "Liga", "deporte": "Fútbol", "sport": "soccer", "league": "col.1", "empate": True}
NBA = {"id": "nba", "nombre": "NBA", "deporte": "Baloncesto", "sport": "basketball", "league": "nba", "empate": False}
ATP = {"id": "atp", "nombre": "ATP", "deporte": "Tenis", "sport": "tennis", "league": "atp", "empate": False,
       "grouping": "mens-singles"}


def _team(side, tid, name, score, winner=None):
    return {"homeAway": side, "score": str(score), "winner": winner, "team": {"id": tid, "displayName": name}}


def _event(eid, date, home, away, status, completed, odds=None, season=2026, stype=2):
    return {"id": eid, "date": date, "season": {"year": season, "type": stype},
            "competitions": [{"id": eid, "date": date, "neutralSite": False, "competitors": [home, away],
                              "status": {"type": {"name": status, "completed": completed,
                                                  "state": "post" if completed else "pre"}}, "odds": odds}]}


def test_american_to_decimal():
    assert np.isclose(espn.american_to_decimal("+150"), 2.5) and np.isclose(espn.american_to_decimal("-200"), 1.5)
    assert espn.american_to_decimal("EVEN") == 2.0 and np.isnan(espn.american_to_decimal(None))
    assert np.isnan(espn.american_to_decimal("OFF"))


def test_parse_team_events_scores_odds_and_penalties():
    odds = [{"provider": {"name": "X"}, "overUnder": 2.5, "drawOdds": {"moneyLine": 200},
             "moneyline": {"home": {"close": {"odds": "-125"}, "open": {"odds": "-110"}},
                           "away": {"close": {"odds": "+320"}}}}]
    payload = {"events": [
        _event("1", "2026-11-01T20:00Z", _team("home", "10", "Nacional", 2, True), _team("away", "20", "Millonarios", 1, False),
               "STATUS_FULL_TIME", True),
        _event("2", "2026-11-02T20:00Z", _team("home", "30", "Junior", 1, True), _team("away", "40", "Cali", 1, False),
               "STATUS_FINAL_PEN", True),  # decidido en penaltis: a 90' fue empate
        _event("3", "2026-11-20T20:00Z", _team("home", "20", "Millonarios", 0), _team("away", "10", "Nacional", 0),
               "STATUS_SCHEDULED", False, odds),
        _event("4", "2026-11-03T20:00Z", _team("home", "30", "Junior", 0), _team("away", "10", "Nacional", 0),
               "STATUS_POSTPONED", False),
    ]}
    df = espn.parse(payload, SOCCER).set_index("event_id")
    assert df.loc["1", "ganador"] == "home" and df.loc["2", "ganador"] == "draw"
    assert bool(df.loc["3", "programado"]) and not bool(df.loc["4", "programado"]) and not bool(df.loc["4", "terminado"])
    assert np.isclose(df.loc["3", "odds_home"], 1.8) and np.isclose(df.loc["3", "odds_draw"], 3.0)
    assert np.isclose(df.loc["3", "odds_away"], 4.2) and df.loc["3", "casa"] == "X"
    assert pd.isna(df.loc["1", "odds_home"])


def test_parse_tennis_orientation_does_not_depend_on_winner():
    def match(mid, winner_first):
        a = {"id": "1", "winner": winner_first, "athlete": {"displayName": "A"},
             "linescores": [{"winner": winner_first}, {"winner": winner_first}]}
        b = {"id": "2", "winner": not winner_first, "athlete": {"displayName": "B"},
             "linescores": [{"winner": not winner_first}, {"winner": not winner_first}]}
        return {"id": mid, "date": "2026-06-01T10:00Z", "competitors": [a, b], "round": {"displayName": "Final"},
                "status": {"type": {"name": "STATUS_FINAL", "completed": True, "state": "post"}}}

    payload = {"events": [{"name": "Open", "season": {"year": 2026}, "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [match("100", True), match("101", True), match("102", False)]},
        {"grouping": {"slug": "mens-doubles"}, "competitions": [match("900", True)]}]}]}
    df = espn.parse(payload, ATP).set_index("event_id")
    assert len(df) == 3 and df["neutral"].all()
    assert (df.loc["100", "home"], df.loc["100", "ganador"]) == ("A", "home")   # id par: orden original
    assert (df.loc["101", "home"], df.loc["101", "ganador"]) == ("B", "away")   # id impar: se invierte
    assert df.loc["102", "ganador"] == "away" and df.loc["102", "away_score"] == 2.0


def test_final_scores_follow_official_winner():
    ev = pd.DataFrame({"liga": ["atp", "col.1", "nba"], "event_id": ["1", "2", "3"], "terminado": True,
                       "ganador": ["away", "draw", "home"], "home_score": [1.0, 2.0, 110.0], "away_score": [1.0, 1.0, 99.0]})
    fs = espn.final_scores(ev)
    assert fs.loc["atp_1", "ftag"] > fs.loc["atp_1", "fthg"]        # retirada con 1-1 en sets: gana el oficial
    assert fs.loc["col.1_2", "fthg"] == fs.loc["col.1_2", "ftag"]   # prórroga/penaltis: empate a 90'
    assert tuple(fs.loc["nba_3"]) == (110.0, 99.0)


def _synthetic(lg, n_teams=12, n=2400, seed=0, draw=False):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1.0, n_teams)
    rows = []
    for i in range(n):
        h, a = rng.choice(n_teams, 2, replace=False)
        d = strength[h] - strength[a] + 0.3 + rng.normal(0, 1.5)
        hs, as_ = (2.0, 1.0) if d > (0.5 if draw else 0) else ((1.0, 1.0) if draw and d > -0.5 else (1.0, 2.0))
        rows.append({"liga": lg["id"], "event_id": str(i), "inicio_utc": pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=14 * i),
                     "temporada": 2021 + i // 600, "terminado": True, "programado": False, "neutral": False,
                     "home_id": f"T{h}", "home": f"T{h}", "away_id": f"T{a}", "away": f"T{a}", "home_score": hs,
                     "away_score": as_, "ganador": "home" if hs > as_ else ("away" if as_ > hs else "draw")})
    return pd.DataFrame(rows)


def test_features_use_only_past_matches():
    ev = _synthetic(NBA)
    F1, _ = modelo.run(ev, NBA, 20, 1.0, 50, 1.0)
    ev2 = ev.copy()
    ev2.loc[1500:, ["home_score", "away_score"]] = ev2.loc[1500:, ["away_score", "home_score"]].to_numpy()
    ev2.loc[1500:, "ganador"] = ev2.loc[1500:, "ganador"].map({"home": "away", "away": "home"})
    F2, _ = modelo.run(ev2, NBA, 20, 1.0, 50, 1.0)
    assert np.array_equal(F1[:1501], F2[:1501])       # cambiar el futuro no cambia nada anterior
    assert not np.array_equal(F1[1501:], F2[1501:])   # control: sí cambia lo posterior


def test_walk_forward_beats_baseline_and_probabilities_are_valid(cfg):
    c = copy.deepcopy(cfg)
    c["deportes"] = {**c.get("deportes", {}), "min_partidos": 300, "anios_test": 2}
    for lg, draw in ((NBA, False), (SOCCER, True)):
        ev = _synthetic(lg, draw=draw)
        m, cal, params = modelo.evaluate_league(ev, lg, c)
        assert m["acierto"] > m["acierto_local_siempre"] + 0.03 and m["partidos_test"] > 500
        up = ev.tail(5).assign(terminado=False, programado=True, ganador=None)
        out = modelo.predict_upcoming(pd.concat([ev.iloc[:-5], up]), lg, params)
        P = out[["mod_p_home", "mod_p_draw", "mod_p_away"]].to_numpy()
        assert len(out) == 5 and np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
        assert (P[:, 1] > 0).all() if draw else (P[:, 1] == 0).all()


def test_market_probs_and_blend():
    mk = market_probs(np.array([[1.5, np.nan, 2.8], [2.0, 3.4, 3.9], [np.nan, np.nan, np.nan]]))
    assert np.allclose(mk[:2].sum(axis=1), 1) and mk[0, 1] == 0 and np.isnan(mk[2]).all()
    pm = np.array([[0.5, 0.0, 0.5], [0.4, 0.3, 0.3], [0.7, 0.0, 0.3]])
    p = blend(pm, mk, 0.7)
    assert np.allclose(p.sum(axis=1), 1) and p[0, 1] == 0
    assert 0.5 < p[0, 0] < mk[0, 0]                 # entre el modelo y el mercado
    assert np.allclose(p[2], pm[2])                 # sin cuotas: solo modelo
    assert np.allclose(blend(pm, mk, 1.0)[1], mk[1])


def test_ledger_with_event_ids_validity_and_settlement(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["_root"] = str(tmp_path)
    base = {"date": "2026-11-11", "league": "nba", "deporte": "Baloncesto", "modelo": "Elo+forma", "p_draw": 0.0,
            "mkt_p_draw": 0.0, "value_bet": "", "value_bet_ou": ""}
    preds = pd.DataFrame([
        {**base, "home": "Bulls", "away": "Spurs", "event_id": "501", "inicio_utc": "2026-11-12T01:00:00Z",
         "p_home": 0.2, "p_away": 0.8, "mkt_p_home": 0.15, "mkt_p_away": 0.85, "odds_home": 6.0, "odds_away": 1.12},
        {**base, "home": "Knicks", "away": "Bucks", "event_id": "502", "inicio_utc": "2026-11-11T18:00:00Z",
         "p_home": 0.6, "p_away": 0.4, "mkt_p_home": 0.6, "mkt_p_away": 0.4, "odds_home": 1.6, "odds_away": 2.4}])
    now = pd.Timestamp("2026-11-11T20:00:00Z").tz_convert(dt.datetime.now().astimezone().tzinfo).tz_localize(None)
    led = append_to_ledger(c, preds, now=now.to_pydatetime())
    by = led.set_index("home")["antes_del_partido"]
    assert bool(by["Bulls"]) and not bool(by["Knicks"])   # el mismo día vale si aún no ha empezado
    assert set(led["pred_id"]) == {"nba_501", "nba_502"}
    ev = pd.DataFrame({"liga": ["nba"], "event_id": ["501"], "terminado": [True], "ganador": ["away"],
                       "home_score": [101.0], "away_score": [120.0]})
    s = settle(c, events=ev)
    led = load_ledger(c).set_index("home")
    assert led.loc["Bulls", "resultado"] == "2" and led.loc["Bulls", "acierto"] == 1
    assert led.loc["Bulls", "fuente_resultado"] == "ESPN" and np.isclose(led.loc["Bulls", "profit_pick"], 0.12)
    assert pd.isna(led.loc["Knicks", "resultado"]) and s["pendientes"] == 1


def _out_row(**kw):
    base = {"date": "2026-11-12", "pick": "1", "p_pick": 0.80, "acierto_historico": 0.78, "n_historico": 500,
            "mkt_p_home": np.nan, "mkt_p_draw": np.nan, "mkt_p_away": np.nan}
    return {**base, **kw}


def test_daily_selection_rules(cfg):
    from futbol.deportes.hoy import select

    c = copy.deepcopy(cfg)
    c["seleccion"] = {"prob_min": 0.70, "acierto_hist_min": 0.70, "n_hist_min": 100, "max_por_dia": 2,
                      "permitir_empate": False, "exigir_mercado_de_acuerdo": True}
    out = pd.DataFrame([
        _out_row(p_pick=0.90),                                                   # 0: entra
        _out_row(p_pick=0.85, mkt_p_home=0.8, mkt_p_draw=0.0, mkt_p_away=0.2),   # 1: entra (mercado de acuerdo)
        _out_row(p_pick=0.75),                                                   # 2: fuera por max_por_dia
        _out_row(p_pick=0.69),                                                   # 3: probabilidad baja
        _out_row(p_pick=0.95, acierto_historico=0.60),                           # 4: ese nivel no acierta en esa liga
        _out_row(p_pick=0.95, n_historico=20),                                   # 5: muy pocos casos históricos
        _out_row(p_pick=0.95, pick="X"),                                         # 6: empate
        _out_row(p_pick=0.95, mkt_p_home=0.4, mkt_p_draw=0.0, mkt_p_away=0.6),   # 7: el mercado opina lo contrario
        _out_row(p_pick=0.72, date="2026-11-13"),                                # 8: otro día, entra
    ])
    assert list(np.flatnonzero(select(out, c))) == [0, 1, 8]


def test_model_vs_market_and_selection_in_report(cfg, tmp_path):
    from futbol.evaluation.informe import analyze

    c = copy.deepcopy(cfg)
    c["_root"] = str(tmp_path)
    base = {"date": "2026-11-10", "league": "nba", "deporte": "Baloncesto", "modelo": "Elo+forma+mercado",
            "p_draw": 0.0, "mkt_p_draw": 0.0, "mod_p_draw": 0.0, "value_bet": "", "value_bet_ou": "",
            "odds_home": 1.25, "odds_away": 4.0}
    preds = pd.DataFrame([
        {**base, "home": "A", "away": "B", "event_id": "1", "inicio_utc": "2026-11-10T23:00:00Z", "seleccion": True,
         "p_home": 0.8, "p_away": 0.2, "mod_p_home": 0.9, "mod_p_away": 0.1, "mkt_p_home": 0.76, "mkt_p_away": 0.24},
        {**base, "home": "C", "away": "D", "event_id": "2", "inicio_utc": "2026-11-10T23:00:00Z", "seleccion": False,
         "p_home": 0.55, "p_away": 0.45, "mod_p_home": 0.4, "mod_p_away": 0.6, "mkt_p_home": 0.62, "mkt_p_away": 0.38}])
    append_to_ledger(c, preds, now=dt.datetime(2026, 11, 9, 12))
    ev = pd.DataFrame({"liga": ["nba", "nba"], "event_id": ["1", "2"], "terminado": True, "ganador": ["home", "home"],
                       "home_score": [100.0, 100.0], "away_score": [90.0, 95.0]})
    settle(c, events=ev)
    tabs = analyze(load_ledger(c), c)
    sel = tabs["Selección"].set_index("seleccion")
    assert sel.loc["Sí", "partidos"] == 1 and sel.loc["Sí", "aciertos"] == 1 and np.isclose(sel.loc["Sí", "yield"], 0.25)
    mm = tabs["Modelo vs mercado"]
    con = mm[mm["ambito"] == "Partidos con cuotas"].set_index("fuente")
    assert con.loc["Modelo local (Elo + forma)", "pct_acierto"] == 0.5      # el modelo solo falló el segundo
    assert con.loc["Mercado (cuotas sin margen)", "pct_acierto"] == 1.0
    assert con.loc["Mercado (cuotas sin margen)", "log_loss"] < con.loc["Modelo local (Elo + forma)", "log_loss"]


def test_schedule_plist(cfg):
    from futbol import programar as P

    d = P.build_plist(cfg, ["07:00", "17:30"])
    assert d["ProgramArguments"][-1] == "hoy" and d["ProgramArguments"][0].endswith("futbol")
    assert d["StartCalendarInterval"] == [{"Hour": 7, "Minute": 0}, {"Hour": 17, "Minute": 30}]
    import pytest
    with pytest.raises(ValueError):
        P.build_plist(cfg, ["25:00"])


def test_record_bet_real_and_paper(cfg, tmp_path):
    import pytest

    from futbol.evaluation.informe import analyze
    from futbol.evaluation.ledger import record_bet

    c = copy.deepcopy(cfg)
    c["_root"] = str(tmp_path)
    base = {"date": "2026-11-12", "league": "nba", "deporte": "Baloncesto", "modelo": "Elo+forma", "p_draw": 0.0,
            "value_bet": "", "value_bet_ou": ""}
    preds = pd.DataFrame([
        {**base, "home": "Washington Wizards", "away": "Detroit Pistons", "event_id": "1", "seleccion": True,
         "inicio_utc": "2026-11-13T00:00:00Z", "p_home": 0.2, "p_away": 0.8},
        {**base, "home": "Atlético Nacional", "away": "Detroit Lions", "event_id": "2", "seleccion": False,
         "inicio_utc": "2026-11-13T00:00:00Z", "p_home": 0.6, "p_away": 0.4}])
    append_to_ledger(c, preds, now=dt.datetime(2026, 11, 11, 12))
    with pytest.raises(ValueError, match="2 partidos"):
        record_bet(c, "detroit", 1.3, monto=10)                      # ambiguo
    with pytest.raises(ValueError, match="monto"):
        record_bet(c, "pistons", 1.3)                                # ni monto ni --papel
    info = record_bet(c, "PISTONS", 1.30, monto=10000)               # por defecto, al pronóstico
    assert info["apuesta_a"] == "Detroit Pistons" and info["tipo"] == "real" and np.isclose(info["cuota_minima"], 1.25)
    info = record_bet(c, "atletico", 2.5, papel=True, a="Lions")     # sin tilde; otra selección; de papel
    assert info["apuesta_a"] == "Detroit Lions" and info["tipo"] == "papel"
    append_to_ledger(c, preds, now=dt.datetime(2026, 11, 12, 12))    # volver a predecir no borra las apuestas
    ev = pd.DataFrame({"liga": ["nba", "nba"], "event_id": ["1", "2"], "terminado": True, "ganador": ["away", "home"],
                       "home_score": [90.0, 100.0], "away_score": [99.0, 95.0]})
    s = settle(c, events=ev)
    led = load_ledger(c).set_index("event_id")
    assert np.isclose(led.loc["1", "profit_real"], 3000) and np.isclose(led.loc["2", "profit_real"], -1.0)
    real = s["apuestas_reales"]                                      # la de papel no cuenta como dinero real
    assert real["n"] == 1 and np.isclose(real["beneficio"], 3000) and real["apostado"] == 10000.0
    tabs = analyze(load_ledger(c), c)
    mine = tabs["Mis apuestas"].set_index("apostado_a")
    assert mine.loc["Detroit Pistons", "estado"] == "Ganada" and mine.loc["Detroit Lions", "estado"] == "Perdida"
    res = dict(zip(tabs["Resumen"]["indicador"], tabs["Resumen"]["valor"]))
    assert "1 ganadas de 1" in res["MIS APUESTAS REALES"] and "0 ganadas de 1" in res["Mis apuestas de papel (sin dinero, 1 u cada una)"]
