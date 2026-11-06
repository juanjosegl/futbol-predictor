"""Ninguna feature de un partido puede usar datos de ese partido ni de partidos posteriores.

Estrategia: se calculan las features, se ALTERAN todos los datos post-partido (goles, xG,
tiros, cuotas de cierre) del partido objetivo y de todos los posteriores, y se recalculan.
Las features del partido objetivo (y de todos los anteriores o del mismo día) deben ser
idénticas. También se comprueba lo contrario: alterar el pasado sí cambia las features
(para que el test no sea trivialmente verdadero).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futbol.features.engineering import POST_MATCH_COLS, build_features, feature_columns
from futbol.models.dixon_coles import dc_walk_forward

from .conftest import make_synthetic_matches

PERTURB = ["fthg", "ftag", "hxg", "axg", "hs", "as_", "hst", "ast",
           "close_h", "close_d", "close_a", "close_over", "close_under"]


def _perturb(df: pd.DataFrame, mask: pd.Series, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = int(mask.sum())
    for c in PERTURB:
        if c not in out.columns:
            continue
        if c in ("fthg", "ftag", "hs", "as_", "hst", "ast"):
            out.loc[mask, c] = rng.integers(0, 9, n).astype(float)
        else:
            out.loc[mask, c] = rng.uniform(1.2, 9.0, n)
    return out


def _feature_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = build_features(df, cfg)
    cols = [c for c in feats.columns if c.startswith("f_")]
    return feats[cols]


@pytest.mark.parametrize("cut_frac", [0.15, 0.5, 0.85])
def test_features_ignore_current_and_future_matches(cfg, cut_frac):
    df = make_synthetic_matches()
    cut_date = df["date"].iloc[int(len(df) * cut_frac)]
    base = _feature_frame(df, cfg)

    future_or_same = df["date"] >= cut_date
    altered = _feature_frame(_perturb(df, future_or_same), cfg)

    rows = df.index[df["date"] <= cut_date]  # partidos del día de corte y anteriores
    pd.testing.assert_frame_equal(base.loc[rows], altered.loc[rows], check_exact=True)


def test_perturbing_past_changes_features(cfg):
    """Control: si se altera el pasado, las features de partidos posteriores SÍ cambian."""
    df = make_synthetic_matches()
    cut_date = df["date"].iloc[len(df) // 2]
    base = _feature_frame(df, cfg)
    altered = _feature_frame(_perturb(df, df["date"] < cut_date), cfg)
    later = df.index[df["date"] >= cut_date]
    assert not base.loc[later].equals(altered.loc[later])


def test_feature_columns_exclude_post_match_info(cfg):
    df = build_features(make_synthetic_matches(), cfg)
    cols = feature_columns(df, {**cfg, "features": {**cfg["features"], "use_opening_odds": True}})
    assert cols, "debe haber features"
    assert not set(cols) & set(POST_MATCH_COLS)
    assert not any(c.startswith("mkt_close") or "close" in c for c in cols)


def test_first_match_of_each_team_has_no_history(cfg):
    df = make_synthetic_matches()
    feats = build_features(df, cfg)
    first = feats.groupby("home_team").head(1)
    first_ever = first[first["f_h_n_prev"] == 0]
    assert len(first_ever) > 0
    assert first_ever["f_h_gf_l5"].isna().all()
    assert first_ever["f_h_rest"].isna().all()


def test_rolling_value_matches_manual_computation(cfg):
    df = make_synthetic_matches()
    feats = build_features(df, cfg)
    team = df["home_team"].iloc[0]
    idx = feats.index[(feats["home_team"] == team)][8]
    d = feats.loc[idx, "date"]
    prev = df[((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < d)].tail(5)
    gf = np.where(prev["home_team"] == team, prev["fthg"], prev["ftag"])
    assert feats.loc[idx, "f_h_gf_l5"] == pytest.approx(gf.mean())


def test_dixon_coles_walk_forward_has_no_leakage(cfg):
    df = make_synthetic_matches(n_teams=8)
    dc_cfg = {**cfg["dixon_coles"], "min_matches": 40}
    base = dc_walk_forward(df, dc_cfg)
    cut_date = df["date"].iloc[int(len(df) * 0.7)]
    # el reentrenamiento es semanal: todo lo de la semana del corte en adelante se altera
    week_start = (cut_date - pd.Timedelta(days=cut_date.weekday())).normalize()
    altered = dc_walk_forward(_perturb(df, df["date"] >= week_start), dc_cfg)
    # los partidos de esa semana se predicen con un modelo entrenado solo con el pasado,
    # así que alterar su propio resultado (y el futuro) no puede cambiar su predicción
    week = (df["date"] >= week_start) & (df["date"] < week_start + pd.Timedelta(days=7))
    rows = df.index[week & base["dc_ph"].notna()]
    assert len(rows) > 0
    pd.testing.assert_frame_equal(base.loc[rows], altered.loc[rows])
    # control: las semanas siguientes sí cambian (entrenan con el pasado alterado)
    later = df.index[(df["date"] >= week_start + pd.Timedelta(days=7)) & base["dc_ph"].notna()]
    assert not base.loc[later].equals(altered.loc[later])


def test_real_dataset_no_leakage(cfg):
    """Mismo test sobre los datos reales (una liga, para que sea rápido) si el dataset existe."""
    from futbol.config import get_path

    p = get_path(cfg, "processed") / "matches.parquet"
    if not p.exists():
        pytest.skip("matches.parquet no existe todavía")
    real = pd.read_parquet(p)
    real = real[real["league"] == real["league"].iloc[0]].reset_index(drop=True)
    real = real[[c for c in real.columns if not c.startswith(("f_", "y_"))]]
    cut_date = real["date"].iloc[int(len(real) * 0.6)]
    base = _feature_frame(real, cfg)
    altered = _feature_frame(_perturb(real, real["date"] >= cut_date), cfg)
    rows = real.index[real["date"] <= cut_date]
    pd.testing.assert_frame_equal(base.loc[rows], altered.loc[rows], check_exact=True)
