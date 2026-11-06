"""Features pre-partido (CERO data leakage).

Regla: cada feature de un partido se calcula SOLO con partidos anteriores. Para los rolling
se agrupa por equipo, se hace `shift(1)` (excluye el partido actual) y después `rolling`.
El Elo se registra ANTES de actualizarlo con el resultado. Las cuotas de cierre nunca son
feature: solo se usan como benchmark y en el backtest.

Todas las columnas de features empiezan por `f_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columnas que contienen información posterior al pitido inicial (resultados, stats, cierre).
POST_MATCH_COLS = [
    "fthg", "ftag", "ftr", "hxg", "axg", "fd_hxg", "fd_axg", "us_hxg", "us_axg",
    "hs", "as_", "hst", "ast", "hc", "ac",
    "close_h", "close_d", "close_a", "close_over", "close_under",
    "odds_psch", "odds_pscd", "odds_psca", "odds_avgch", "odds_avgcd", "odds_avgca",
]

STATS = ["gf", "ga", "xgf", "xga", "sf", "sa", "sotf", "sota", "pts"]


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Formato largo: dos filas por partido (perspectiva local y visitante)."""
    def side(home: bool) -> pd.DataFrame:
        t, o = ("home_team", "away_team") if home else ("away_team", "home_team")
        gf, ga = ("fthg", "ftag") if home else ("ftag", "fthg")
        xf, xa = ("hxg", "axg") if home else ("axg", "hxg")
        sf, sa = ("hs", "as_") if home else ("as_", "hs")
        tf, ta = ("hst", "ast") if home else ("ast", "hst")
        return pd.DataFrame({
            "row": df.index, "date": df["date"].values, "league": df["league"].values,
            "team": df[t].values, "is_home": int(home),
            "gf": df[gf].values, "ga": df[ga].values, "xgf": df[xf].values, "xga": df[xa].values,
            "sf": df[sf].values, "sa": df[sa].values, "sotf": df[tf].values, "sota": df[ta].values,
        })

    long = pd.concat([side(True), side(False)], ignore_index=True)
    long["pts"] = np.where(long["gf"].isna(), np.nan,
                           np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0)))
    long = long.sort_values(["team", "date", "row"], kind="mergesort").reset_index(drop=True)
    return long


def _rolling(long: pd.DataFrame, keys: list[str], windows: list[int], prefix: str) -> pd.DataFrame:
    """Medias móviles de STATS sobre los partidos PREVIOS (shift(1) antes de rolling)."""
    grp_keys = [long[k] for k in keys]
    shifted = long.groupby(keys, sort=False)[STATS].shift(1)  # <- excluye el partido actual
    out = pd.DataFrame(index=long.index)
    for w in windows:
        r = shifted.groupby(grp_keys, sort=False).rolling(w, min_periods=1).mean()
        r = r.reset_index(level=list(range(len(keys))), drop=True).sort_index()
        for s in STATS:
            out[f"{prefix}{s}_l{w}"] = r[s]
    return out


def team_form_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling generales y por condición (local/visitante), días de descanso y nº de partidos previos."""
    long = _team_long(df)
    gen = _rolling(long, ["team"], windows, "")
    ven = _rolling(long, ["team", "is_home"], windows, "venue_")
    g = long.groupby("team", sort=False)
    extra = pd.DataFrame(index=long.index)
    extra["rest"] = g["date"].diff().dt.days.clip(upper=60)  # días desde el partido anterior
    extra["n_prev"] = g.cumcount()                            # partidos previos en el dataset
    feats = pd.concat([long[["row", "is_home"]], gen, ven, extra], axis=1)

    home = feats[feats["is_home"] == 1].set_index("row").drop(columns="is_home")
    away = feats[feats["is_home"] == 0].set_index("row").drop(columns="is_home")
    home = home.rename(columns=lambda c: "f_h_" + c.replace("venue_", "home_"))
    away = away.rename(columns=lambda c: "f_a_" + c.replace("venue_", "away_"))
    return home.join(away, how="outer").reindex(df.index)


def _goal_multiplier(gd: float) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(df: pd.DataFrame, k: float = 20, home_advantage: float = 60, initial: float = 1500,
                season_regression: float = 0.2, promoted_offset: float = -100) -> pd.DataFrame:
    """Elo propio por liga, calculado secuencialmente por fecha.

    Devuelve el rating de cada equipo ANTES del partido (se guarda antes de actualizar).
    - Inicio de temporada: regresión parcial a la media de la liga.
    - Equipo nuevo (ascendido) tras la primera temporada: media de la liga + offset.
    - Partidos sin resultado (futuros) no actualizan el rating.
    """
    order = df.sort_values(["date"], kind="mergesort").index
    ratings: dict[tuple[str, str], float] = {}
    last_season: dict[str, str] = {}
    first_season: dict[str, str] = {}
    elo_h = pd.Series(np.nan, index=df.index)
    elo_a = pd.Series(np.nan, index=df.index)
    lg_arr, s_arr = df["league"].to_numpy(), df["season"].to_numpy()
    ht_arr, at_arr = df["home_team"].to_numpy(), df["away_team"].to_numpy()
    hg_arr, ag_arr = df["fthg"].to_numpy(dtype=float), df["ftag"].to_numpy(dtype=float)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for idx in order:
        i = pos[idx]
        lg, season = lg_arr[i], s_arr[i]
        if lg not in first_season:
            first_season[lg] = season
        if last_season.get(lg) != season:
            if lg in last_season:  # cambio de temporada: regresión a la media
                keys = [key for key in ratings if key[0] == lg]
                if keys:
                    mean = float(np.mean([ratings[key] for key in keys]))
                    for key in keys:
                        ratings[key] = ratings[key] * (1 - season_regression) + mean * season_regression
            last_season[lg] = season
        for team in (ht_arr[i], at_arr[i]):
            key = (lg, team)
            if key not in ratings:
                league_r = [v for kk, v in ratings.items() if kk[0] == lg]
                if season == first_season[lg] or not league_r:
                    ratings[key] = initial
                else:
                    ratings[key] = float(np.mean(league_r)) + promoted_offset
        kh, ka = (lg, ht_arr[i]), (lg, at_arr[i])
        rh, ra = ratings[kh], ratings[ka]
        elo_h.iat[i], elo_a.iat[i] = rh, ra  # <- valor PRE-partido
        hg, ag = hg_arr[i], ag_arr[i]
        if np.isnan(hg) or np.isnan(ag):
            continue
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + home_advantage)) / 400.0))
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = k * _goal_multiplier(hg - ag) * (score - exp_h)
        ratings[kh] = rh + delta
        ratings[ka] = ra - delta
    return pd.DataFrame({"f_elo_h": elo_h, "f_elo_a": elo_a})


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Añade todas las features `f_*` a un DataFrame de partidos (ordenado o no).

    `df` debe tener: league, season, date, home_team, away_team, fthg, ftag, hxg, axg,
    hs, as_, hst, ast y (opcional) mkt_open_*, clubelo_h/clubelo_a.
    """
    fcfg = cfg["features"]
    ecfg = fcfg["elo"]
    df = df.copy()
    for c in ["hxg", "axg", "hs", "as_", "hst", "ast"]:
        if c not in df.columns:
            df[c] = np.nan
    form = team_form_features(df, list(fcfg["windows"]))
    elo = compute_elo(df, k=ecfg["k"], home_advantage=ecfg["home_advantage"], initial=ecfg["initial"],
                      season_regression=ecfg["season_regression"], promoted_offset=ecfg["promoted_offset"])
    out = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("f_")]), form, elo], axis=1)
    ha = ecfg["home_advantage"]
    out["f_elo_diff"] = out["f_elo_h"] - out["f_elo_a"]
    out["f_elo_exp_h"] = 1.0 / (1.0 + 10 ** (-(out["f_elo_diff"] + ha) / 400.0))
    if "clubelo_h" in out.columns:
        out["f_clubelo_h"], out["f_clubelo_a"] = out["clubelo_h"], out["clubelo_a"]
        out["f_clubelo_diff"] = out["clubelo_h"] - out["clubelo_a"]
    for w in fcfg["windows"]:
        out[f"f_d_pts_l{w}"] = out[f"f_h_pts_l{w}"] - out[f"f_a_pts_l{w}"]
        out[f"f_d_gd_l{w}"] = (out[f"f_h_gf_l{w}"] - out[f"f_h_ga_l{w}"]) - (out[f"f_a_gf_l{w}"] - out[f"f_a_ga_l{w}"])
        out[f"f_d_sot_l{w}"] = (out[f"f_h_sotf_l{w}"] - out[f"f_h_sota_l{w}"]) - (out[f"f_a_sotf_l{w}"] - out[f"f_a_sota_l{w}"])
        out[f"f_d_xg_l{w}"] = (out[f"f_h_xgf_l{w}"] - out[f"f_h_xga_l{w}"]) - (out[f"f_a_xgf_l{w}"] - out[f"f_a_xga_l{w}"])
    out["f_rest_diff"] = out["f_h_rest"] - out["f_a_rest"]
    # Probabilidades de APERTURA (información pre-partido). Se filtran por config al entrenar.
    for c in ["h", "d", "a", "over"]:
        if f"mkt_open_{c}" in out.columns:
            out[f"f_mkt_open_{c}"] = out[f"mkt_open_{c}"]
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    played = df["fthg"].notna() & df["ftag"].notna()
    res = np.where(df["fthg"] > df["ftag"], 0, np.where(df["fthg"] == df["ftag"], 1, 2))
    df["y_1x2"] = pd.Series(np.where(played, res, -1), index=df.index).astype(int)
    df["y_over25"] = pd.Series(np.where(played, (df["fthg"] + df["ftag"]) > 2.5, False), index=df.index).astype(int)
    df["played"] = played
    return df


def feature_columns(df: pd.DataFrame, cfg: dict, extra: list[str] | None = None) -> list[str]:
    """Columnas de features para el modelo según la config."""
    cols = [c for c in df.columns if c.startswith("f_")]
    if not cfg["features"].get("use_opening_odds", False):
        cols = [c for c in cols if not c.startswith("f_mkt_open")]
    if extra:
        cols += [c for c in extra if c in df.columns]
    # Defensa en profundidad: nunca columnas de cierre/resultados
    banned = set(POST_MATCH_COLS) | {c for c in df.columns if c.startswith("mkt_close")}
    return [c for c in cols if c not in banned]
