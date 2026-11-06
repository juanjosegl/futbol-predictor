from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from futbol.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_synthetic_matches(seed: int = 0, n_teams: int = 6, seasons=("2122", "2223", "2324"),
                           leagues=("L1", "L2")) -> pd.DataFrame:
    """Liga sintética: doble round-robin por temporada con resultados, xG, tiros y cuotas aleatorias."""
    rng = np.random.default_rng(seed)
    rows = []
    for lg in leagues:
        teams = [f"{lg}_T{i}" for i in range(n_teams)]
        for s_i, season in enumerate(seasons):
            start = pd.Timestamp(f"20{season[:2]}-08-10")
            pairs = list(itertools.permutations(teams, 2))
            rng.shuffle(pairs)
            # 3 partidos por "jornada" -> cada equipo juega como mucho una vez por día
            day = 0
            remaining = pairs[:]
            while remaining:
                used, today = set(), []
                for p in list(remaining):
                    if p[0] not in used and p[1] not in used:
                        today.append(p)
                        used |= set(p)
                        remaining.remove(p)
                for h, a in today:
                    rows.append({"league": lg, "season": season, "date": start + pd.Timedelta(days=7 * day),
                                 "home_team": h, "away_team": a})
                day += 1
    df = pd.DataFrame(rows)
    n = len(df)
    df["fthg"] = rng.poisson(1.5, n).astype(float)
    df["ftag"] = rng.poisson(1.1, n).astype(float)
    df["hxg"] = rng.gamma(2, 0.7, n)
    df["axg"] = rng.gamma(2, 0.55, n)
    df["hs"] = rng.poisson(13, n).astype(float)
    df["as_"] = rng.poisson(10, n).astype(float)
    df["hst"] = rng.poisson(5, n).astype(float)
    df["ast"] = rng.poisson(4, n).astype(float)
    for c, base in [("h", 2.3), ("d", 3.4), ("a", 3.2)]:
        df[f"open_{c}"] = base + rng.normal(0, 0.2, n)
        df[f"close_{c}"] = base + rng.normal(0, 0.2, n)
    df["open_over"], df["open_under"] = 1.9, 1.95
    df["close_over"], df["close_under"] = 1.9 + rng.normal(0, .05, n), 1.95
    from futbol.features.odds import add_market_probs
    df = add_market_probs(df)
    return df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)


@pytest.fixture
def synthetic_matches():
    return make_synthetic_matches()
