import numpy as np
import pandas as pd
import pytest

from futbol.evaluation.backtest import (kelly_fraction, max_drawdown, max_drawdown_units, run_flat, run_kelly,
                                        select_bets, summarize)


def _matches():
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3", "m4"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-15"]),
        "season": "2324", "league": "E0",
        "p_h": [0.60, 0.40, 0.30, 0.50], "p_d": [0.25, 0.30, 0.30, 0.25], "p_a": [0.15, 0.30, 0.40, 0.25],
        "o_h": [2.00, 2.40, 3.00, 2.00], "o_d": [3.50, 3.30, 3.40, 3.60], "o_a": [6.00, 3.20, 2.20, 4.50],
        "y": [0, 2, 1, 2],
    })


def _bets(df, threshold=0.05, one=True, max_odds=None):
    won = [df["y"] == 0, df["y"] == 1, df["y"] == 2]
    return select_bets(df, ["p_h", "p_d", "p_a"], ["o_h", "o_d", "o_a"], won, ["H", "D", "A"],
                       threshold, one_per_match=one, max_odds=max_odds)


def test_threshold_rule():
    bets = _bets(_matches(), one=False)
    # se apuesta solo si p*o > 1 + umbral
    assert (bets["prob"] * bets["odds"] > 1.05).all()
    # m1: H 0.6*2.0=1.20 sí; D 0.875 no; A 0.9 no
    m1 = bets[bets["match_id"] == "m1"]
    assert list(m1["selection"]) == ["H"]
    # m3: A 0.4*2.2 = 0.88 no; D 0.3*3.4=1.02 no (<1.05); H 0.3*3.0 = 0.90 no
    assert "m3" not in set(bets["match_id"])


def test_edge_exactly_at_threshold_is_not_bet():
    df = _matches().iloc[[0]].copy()
    df["p_h"], df["o_h"] = 0.525, 2.0  # p*o = 1.05 exactamente
    assert _bets(df).empty


def test_one_bet_per_match_takes_max_edge():
    df = _matches()
    df.loc[1, ["p_h", "p_a"]] = [0.47, 0.35]  # m2: H 1.128 y A 1.12 -> ambas value
    many = _bets(df, one=False)
    assert (many["match_id"] == "m2").sum() == 2
    one = _bets(df, one=True)
    assert one.loc[one["match_id"] == "m2", "selection"].item() == "H"
    assert one["match_id"].is_unique


def test_flat_stake_profit():
    bets = _bets(_matches())
    led = run_flat(bets, stake=1.0, initial=100)
    expected = sum((r.odds - 1) if r.won else -1 for r in bets.itertuples())
    assert led["profit"].sum() == pytest.approx(expected)
    assert led["bankroll"].iloc[-1] == pytest.approx(100 + expected)
    s = summarize(led, 100)
    assert s["n_bets"] == len(bets)
    assert s["yield"] == pytest.approx(expected / len(bets))
    assert s["roi"] == pytest.approx(expected / 100)


def test_kelly_formula_and_cap():
    assert kelly_fraction(np.array([0.6]), np.array([2.0]))[0] == pytest.approx(0.2)
    assert kelly_fraction(np.array([0.4]), np.array([2.0]))[0] == 0.0  # sin ventaja, sin apuesta
    bets = pd.DataFrame({"match_id": ["a"], "date": pd.to_datetime(["2024-01-01"]), "season": "x",
                         "prob": [0.6], "odds": [2.0], "edge": [0.2], "won": [True]})
    led = run_kelly(bets, fraction=0.25, initial=100, max_fraction=0.5)
    assert led["stake"].iloc[0] == pytest.approx(100 * 0.25 * 0.2)
    capped = run_kelly(bets, fraction=1.0, initial=100, max_fraction=0.05)
    assert capped["stake"].iloc[0] == pytest.approx(5.0)


def test_kelly_same_day_uses_start_of_day_bankroll():
    bets = pd.DataFrame({"match_id": ["a", "b", "c"],
                         "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]), "season": "x",
                         "prob": [0.6, 0.6, 0.6], "odds": [2.0, 2.0, 2.0], "edge": [0.2] * 3,
                         "won": [False, True, True]})
    led = run_kelly(bets, fraction=0.25, initial=100, max_fraction=1.0)
    assert led["stake"].iloc[0] == pytest.approx(5.0) and led["stake"].iloc[1] == pytest.approx(5.0)
    # día 1: -5 + 5 = 0 -> el día 2 vuelve a apostar 5% de 100
    assert led["stake"].iloc[2] == pytest.approx(5.0)


def test_max_drawdown():
    assert max_drawdown([110, 120, 90, 100, 130], initial=100) == pytest.approx(30 / 120)
    assert max_drawdown_units([110, 120, 90, 100, 130], initial=100) == pytest.approx(30)
    assert max_drawdown([100, 101, 102]) == 0.0


def test_clv():
    df = _matches().iloc[[0]].copy()
    df["fair_h"], df["fair_d"], df["fair_a"] = 0.55, 0.25, 0.20
    won = [df["y"] == 0, df["y"] == 1, df["y"] == 2]
    bets = select_bets(df, ["p_h", "p_d", "p_a"], ["o_h", "o_d", "o_a"], won, ["H", "D", "A"], 0.05,
                       fair_close_cols=["fair_h", "fair_d", "fair_a"])
    # cuota 2.00 tomada; cuota justa de cierre 1/0.55 = 1.818 -> CLV = 2.0*0.55 - 1 = +10%
    assert bets["clv"].iloc[0] == pytest.approx(0.10)


def test_market_probabilities_never_bet():
    """Si el 'modelo' es el propio mercado (sin margen), no hay apuestas: p*o = 1/(1+margen) < 1."""
    df = _matches()
    true_p = df[["p_h", "p_d", "p_a"]].to_numpy()
    odds = 1 / (true_p * 1.05)  # cuotas con 5% de margen
    df["o_h"], df["o_d"], df["o_a"] = odds[:, 0], odds[:, 1], odds[:, 2]
    inv = 1 / odds
    p = inv / inv.sum(axis=1, keepdims=True)
    df["p_h"], df["p_d"], df["p_a"] = p[:, 0], p[:, 1], p[:, 2]
    assert _bets(df, threshold=0.0).empty


def test_oracle_makes_money():
    df = _matches()
    onehot = np.eye(3)[df["y"]] * 0.97 + 0.01
    df["p_h"], df["p_d"], df["p_a"] = onehot[:, 0], onehot[:, 1], onehot[:, 2]
    led = run_flat(_bets(df), 1.0, 100)
    assert led["won"].all() and led["profit"].sum() > 0


def test_max_odds_filter():
    bets = _bets(_matches(), one=False, max_odds=2.1)
    assert (bets["odds"] <= 2.1).all()
