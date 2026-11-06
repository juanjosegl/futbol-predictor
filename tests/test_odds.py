import numpy as np
import pandas as pd
import pytest

from futbol.features.odds import add_market_probs, fair_odds, implied_probs, overround


def test_overround():
    assert overround(np.array([[2.0, 2.0]]))[0] == pytest.approx(0.0)
    assert overround(np.array([[1.9, 1.9]]))[0] == pytest.approx(2 / 1.9 - 1)


def test_multiplicative_sums_to_one_and_known_values():
    odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.2, 7.0]])
    p = implied_probs(odds, "multiplicative")
    assert np.allclose(p.sum(axis=1), 1)
    inv = 1 / odds[0]
    assert p[0] == pytest.approx(inv / inv.sum())
    # sin margen, las probabilidades son exactamente 1/cuota
    fair = np.array([[2.0, 4.0, 4.0]])
    assert implied_probs(fair)[0] == pytest.approx([0.5, 0.25, 0.25])


def test_shin_sums_to_one_and_shrinks_longshots():
    odds = np.array([[1.3, 5.5, 11.0]])
    mult = implied_probs(odds, "multiplicative")[0]
    shin = implied_probs(odds, "shin")[0]
    assert shin.sum() == pytest.approx(1)
    assert shin[0] > mult[0]  # Shin asigna más al favorito
    assert shin[2] < mult[2]  # y menos al longshot


def test_shin_two_way_market():
    p = implied_probs(np.array([[1.8, 2.05]]), "shin")[0]
    assert p.sum() == pytest.approx(1)
    assert p[0] > p[1]


def test_invalid_rows_are_nan():
    p = implied_probs(np.array([[np.nan, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 3.5, 4.0]]))
    assert np.isnan(p[0]).all() and np.isnan(p[1]).all()
    assert np.isfinite(p[2]).all()


def test_fair_odds_inverse():
    assert fair_odds(np.array([0.5, 0.25])) == pytest.approx([2.0, 4.0])


def test_add_market_probs_columns():
    df = pd.DataFrame({"open_h": [2.0], "open_d": [3.4], "open_a": [3.9], "close_h": [1.9], "close_d": [3.5],
                       "close_a": [4.2], "open_over": [1.9], "open_under": [1.95], "close_over": [1.85],
                       "close_under": [2.0]})
    out = add_market_probs(df)
    assert out[["mkt_close_h", "mkt_close_d", "mkt_close_a"]].sum(axis=1).iloc[0] == pytest.approx(1)
    assert 0 < out["mkt_close_over"].iloc[0] < 1
