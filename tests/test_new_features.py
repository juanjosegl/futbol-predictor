"""Tests de: respaldo football-data.org, ledger/settle y regresión logística."""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd

from futbol.evaluation.ledger import append_to_ledger, load_ledger, settle
from futbol.ingest import football_data_org as fdo
from futbol.ingest.team_names import TeamNameMapper
from futbol.models.reglog import RegLog


def _tmp_cfg(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["_root"] = str(tmp_path)
    return c


def test_fdorg_parse_map_and_missing():
    payload = {"matches": [
        {"utcDate": "2026-11-01T15:00:00Z", "status": "FINISHED",
         "homeTeam": {"name": "Manchester City FC", "shortName": "Man City"},
         "awayTeam": {"name": "Liverpool FC", "shortName": "Liverpool"},
         "score": {"fullTime": {"home": 2, "away": 1}}},
        {"utcDate": "2026-11-08T20:00:00Z", "status": "TIMED",
         "homeTeam": {"name": "Arsenal FC", "shortName": "Arsenal"},
         "awayTeam": {"name": "Tottenham Hotspur FC", "shortName": "Tottenham"},
         "score": {"fullTime": {"home": None, "away": None}}},
        {"utcDate": "2026-10-25T14:00:00Z", "status": "FINISHED",
         "homeTeam": {"name": "Arsenal FC", "shortName": "Arsenal"},
         "awayTeam": {"name": "Chelsea FC", "shortName": "Chelsea"},
         "score": {"fullTime": {"home": 0, "away": 0}}},
    ]}
    df = fdo.parse(payload, "E0", "2627")
    mapper = TeamNameMapper({"E0": ["Man City", "Liverpool", "Arsenal", "Tottenham", "Chelsea"]})
    df["home_team"] = [fdo._map(mapper, n, "E0") for n in df["home_names"]]
    df["away_team"] = [fdo._map(mapper, n, "E0") for n in df["away_names"]]
    assert list(df["home_team"]) == ["Man City", "Arsenal", "Arsenal"]
    # Arsenal-Chelsea ya está en football-data.co.uk -> no se duplica; el partido futuro tampoco entra
    have = pd.DataFrame({"league": ["E0"], "season": ["2627"], "home_team": ["Arsenal"], "away_team": ["Chelsea"]})
    miss = fdo.missing_played(have, df[fdo.COLS])
    assert len(miss) == 1 and miss.iloc[0]["home_team"] == "Man City" and miss.iloc[0]["fthg"] == 2.0


def _pred_row(date, home, away, p=(0.5, 0.3, 0.2), odds=(2.2, 3.4, 4.0), vb="1"):
    return {"date": date, "league": "E0", "home": home, "away": away, "modelo": "x",
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "p_over25": 0.55,
            "mkt_p_home": 0.45, "mkt_p_draw": 0.29, "mkt_p_away": 0.26,
            "odds_home": odds[0], "odds_draw": odds[1], "odds_away": odds[2], "odds_over": 1.9, "odds_under": 1.9,
            "value_bet": vb, "value_bet_ou": "Over"}


def test_ledger_append_and_settle(cfg, tmp_path):
    c = _tmp_cfg(cfg, tmp_path)
    preds = pd.DataFrame([_pred_row("2026-11-08", "Arsenal", "Chelsea"),
                          _pred_row("2026-11-08", "Leeds", "Everton", vb="")])
    append_to_ledger(c, preds, now=dt.datetime(2026, 11, 7, 12))
    # re-predecir el mismo partido reemplaza (no duplica) y conserva lo que rellenó el usuario
    led = load_ledger(c)
    led.loc[0, ["apuesta_real", "cuota_real", "stake_real"]] = ["1", 2.3, 10]
    led.to_csv(tmp_path / "reports" / "ledger.csv", index=False)
    append_to_ledger(c, preds.iloc[[0]], now=dt.datetime(2026, 11, 7, 18))
    led = load_ledger(c)
    assert len(led) == 2
    assert led.loc[led["home"] == "Arsenal", "cuota_real"].iloc[0] == 2.3

    hist = pd.DataFrame({"league": ["E0", "E0"], "home_team": ["Arsenal", "Leeds"], "away_team": ["Chelsea", "Everton"],
                         "date": pd.to_datetime(["2026-11-08", "2026-11-09"]), "fthg": [2.0, 1.0], "ftag": [0.0, 1.0],
                         "played": [True, True], "season": "2627"})
    (tmp_path / "data" / "processed").mkdir(parents=True)
    hist.to_parquet(tmp_path / "data" / "processed" / "matches.parquet")
    s = settle(c)
    led = load_ledger(c)
    a = led[led["home"] == "Arsenal"].iloc[0]
    assert a["resultado"] == "1" and a["acierto"] == 1
    assert np.isclose(a["profit_value_bet"], 1.2) and np.isclose(a["profit_real"], 13.0)
    assert np.isclose(a["ll_modelo"], -np.log(0.5))
    b = led[led["home"] == "Leeds"].iloc[0]
    assert b["resultado"] == "X" and b["acierto"] == 0 and np.isnan(b["profit_value_bet"])
    assert s["validas_antes_del_partido"] == 2 and s["value_bets_1x2"]["n"] == 1


def test_ledger_flags_predictions_made_after_kickoff(cfg, tmp_path):
    c = _tmp_cfg(cfg, tmp_path)
    led = append_to_ledger(c, pd.DataFrame([_pred_row("2026-11-01", "Arsenal", "Chelsea")]),
                           now=dt.datetime(2026, 11, 7))
    assert not bool(led["antes_del_partido"].iloc[0])


def test_reglog_required_columns_give_nan():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (df["a"] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    df.loc[:9, "b"] = np.nan
    m = RegLog(["a", "b"], required=["b"]).fit(df, y)
    p = m.predict_proba(df)
    assert np.isnan(p[:10]).all() and np.isfinite(p[10:]).all()
    assert np.allclose(p[10:].sum(axis=1), 1)


def test_settle_with_manual_scores_and_report(cfg, tmp_path):
    """Sin datos descargados: el marcador escrito a mano en el ledger basta para liquidar y generar el Excel."""
    from openpyxl import load_workbook

    from futbol.evaluation.informe import analyze, confidence

    c = _tmp_cfg(cfg, tmp_path)
    preds = pd.DataFrame([_pred_row("2026-11-08", "Arsenal", "Chelsea", p=(0.75, 0.15, 0.10), odds=(1.3, 5.0, 9.0), vb=""),
                          _pred_row("2026-11-08", "Leeds", "Everton", p=(0.40, 0.30, 0.30), vb=""),
                          _pred_row("2026-11-09", "Fulham", "Brentford", vb="")])
    append_to_ledger(c, preds, now=dt.datetime(2026, 11, 7, 12))
    led = load_ledger(c)
    led.loc[led["home"] == "Arsenal", ["fthg", "ftag"]] = [3, 1]
    led.loc[led["home"] == "Leeds", ["fthg", "ftag"]] = [0, 2]
    led.loc[led["home"] == "Arsenal", ["apuesta_real", "cuota_real"]] = ["Over", 1.8]  # sin stake -> 1 unidad
    led.to_csv(tmp_path / "reports" / "ledger.csv", index=False)
    s = settle(c)
    led = load_ledger(c)
    a, b, f = (led[led["home"] == h].iloc[0] for h in ("Arsenal", "Leeds", "Fulham"))
    assert a["resultado"] == "1" and a["fuente_resultado"] == "manual" and a["acierto"] == 1
    assert np.isclose(a["profit_pick"], 0.3) and np.isclose(a["profit_real"], 0.8) and a["acierto_ou"] == 1
    assert b["resultado"] == "2" and b["acierto"] == 0 and np.isclose(b["profit_pick"], -1.0)
    assert pd.isna(f["resultado"]) and pd.isna(f["acierto"])  # sin marcador sigue pendiente
    assert s["total_liquidadas"] == 2 and s["pendientes"] == 1 and s["pronosticos_1u"]["n"] == 2
    tabs = analyze(led, c)
    conf = tabs["Por confianza"].set_index("confianza")
    assert conf.loc["Alta", "aciertos"] == 1 and conf.loc["Baja", "aciertos"] == 0
    assert list(tabs["Predicciones"]["estado"]) == ["Acertó", "Falló", "Pendiente"]
    wb = load_workbook(tmp_path / "reports" / "seguimiento.xlsx")
    assert {"Resumen", "Predicciones", "Por confianza", "Por liga", "Por semana"} <= set(wb.sheetnames)
    assert list(confidence([0.8, 0.6, 0.4], c)) == ["Alta", "Media", "Baja"]
    web = (tmp_path / "reports" / "seguimiento.html").read_text(encoding="utf-8")  # versión sin Excel
    assert "Mis apuestas" in web and 'class="Acertó"' in web and 'class="Pendiente"' in web and "Arsenal" in web
