"""Backtest de apuestas sobre predicciones fuera de muestra.

- Se apuesta si prob_modelo * cuota > 1 + umbral.
- Estrategias: stake fijo y Kelly fraccional (por defecto 25%, con tope por apuesta).
- Precio principal: cuota de CIERRE (Pinnacle si existe, si no promedio de mercado).
- Escenario adicional a cuota de APERTURA (pre-cierre) para medir CLV contra el cierre.
- Solo temporadas fuera de muestra (cada temporada la predice un modelo entrenado con las anteriores).

Definiciones:
- yield = beneficio / total apostado
- ROI   = beneficio / bankroll inicial
- CLV   = cuota tomada / cuota justa de cierre - 1 (cuota justa = 1 / prob. de cierre sin margen)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import get_path
from ..models.training import load_oos
from .plots import bankroll_curves

log = logging.getLogger(__name__)

MODEL_NAMES = {"lgbm": "LightGBM", "dc": "Dixon-Coles", "rl": "Regresión logística",
               "rlm": "Reg. logística + mercado"}

MARKETS = {
    "1x2": {
        "outcomes": ["H", "D", "A"],
        "won": lambda df: [df["y_1x2"] == 0, df["y_1x2"] == 1, df["y_1x2"] == 2],
        "close": ["close_h", "close_d", "close_a"],
        "open": ["open_h", "open_d", "open_a"],
        "fair_close": ["mkt_close_h", "mkt_close_d", "mkt_close_a"],
        "models": {"lgbm": ["lgbm_ph", "lgbm_pd", "lgbm_pa"], "dc": ["dc_ph", "dc_pd", "dc_pa"],
                   "rl": ["rl_ph", "rl_pd", "rl_pa"], "rlm": ["rlm_ph", "rlm_pd", "rlm_pa"]},
    },
    "ou25": {
        "outcomes": ["Over", "Under"],
        "won": lambda df: [df["y_over25"] == 1, df["y_over25"] == 0],
        "close": ["close_over", "close_under"],
        "open": ["open_over", "open_under"],
        "fair_close": ["mkt_close_over", "_mkt_close_under"],
        "models": {"lgbm": ["lgbm_pover", "_lgbm_punder"], "dc": ["dc_pover", "_dc_punder"],
                   "rl": ["rl_pover", "_rl_punder"], "rlm": ["rlm_pover", "_rlm_punder"]},
    },
}


def select_bets(df: pd.DataFrame, prob_cols: list[str], odds_cols: list[str], won: list[pd.Series],
                outcomes: list[str], threshold: float, one_per_match: bool = True,
                max_odds: float | None = None, fair_close_cols: list[str] | None = None) -> pd.DataFrame:
    """Devuelve una fila por apuesta con prob, cuota, edge, resultado y CLV (si hay cierre)."""
    parts = []
    for k, out in enumerate(outcomes):
        p = df[prob_cols[k]].to_numpy(float)
        o = df[odds_cols[k]].to_numpy(float)
        edge = p * o - 1.0
        part = pd.DataFrame({
            "match_id": df["match_id"].values, "date": df["date"].values, "season": df["season"].values,
            "league": df["league"].values, "selection": out, "prob": p, "odds": o, "edge": edge,
            "won": np.asarray(won[k], bool),
        }, index=df.index)
        if fair_close_cols is not None:
            fc = df[fair_close_cols[k]].to_numpy(float)
            part["clv"] = o * fc - 1.0  # = cuota tomada / (1/prob_cierre) - 1
        parts.append(part)
    allb = pd.concat(parts)
    # p*o > 1 + umbral (estricto; 1e-12 evita que el redondeo flotante convierta "igual" en "mayor")
    ok = np.isfinite(allb["edge"]) & (allb["odds"] > 1) & (allb["edge"] > threshold + 1e-12)
    if max_odds is not None:
        ok &= allb["odds"] <= max_odds
    bets = allb[ok]
    if one_per_match and not bets.empty:
        bets = bets.sort_values("edge", ascending=False).drop_duplicates("match_id")
    return bets.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)


def kelly_fraction(prob: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Fracción de Kelly completa: (p*o - 1) / (o - 1), truncada a >= 0."""
    prob, odds = np.asarray(prob, float), np.asarray(odds, float)
    return np.clip((prob * odds - 1.0) / (odds - 1.0), 0, None)


def run_flat(bets: pd.DataFrame, stake: float = 1.0, initial: float = 100.0) -> pd.DataFrame:
    led = bets.copy()
    led["stake"] = stake
    led["profit"] = np.where(led["won"], stake * (led["odds"] - 1.0), -stake)
    led["bankroll"] = initial + led["profit"].cumsum()
    return led


def run_kelly(bets: pd.DataFrame, fraction: float = 0.25, initial: float = 100.0,
              max_fraction: float = 0.05) -> pd.DataFrame:
    """Kelly fraccional. Las apuestas de un mismo día se dimensionan con el bankroll al inicio del día."""
    led = bets.copy()
    led["stake"] = 0.0
    led["profit"] = 0.0
    bank = initial
    for _, idx in led.groupby("date", sort=True).groups.items():
        g = led.loc[idx]
        f = np.minimum(fraction * kelly_fraction(g["prob"], g["odds"]), max_fraction)
        stakes = bank * f
        profit = np.where(g["won"], stakes * (g["odds"] - 1.0), -stakes)
        led.loc[idx, "stake"] = stakes
        led.loc[idx, "profit"] = profit
        bank += profit.sum()
    led["bankroll"] = initial + led["profit"].cumsum()
    return led


def max_drawdown(bankroll: pd.Series | np.ndarray, initial: float | None = None) -> float:
    """Máxima caída relativa desde un pico (0.25 = -25%). >1 si el bankroll llega a ser negativo."""
    b = np.asarray(bankroll, float)
    if initial is not None:
        b = np.concatenate([[initial], b])
    if len(b) == 0:
        return 0.0
    peak = np.maximum.accumulate(b)
    return float(np.max((peak - b) / peak))


def max_drawdown_units(bankroll: pd.Series | np.ndarray, initial: float | None = None) -> float:
    """Máxima caída absoluta desde un pico, en unidades."""
    b = np.asarray(bankroll, float)
    if initial is not None:
        b = np.concatenate([[initial], b])
    if len(b) == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(b) - b))


def summarize(led: pd.DataFrame, initial: float) -> dict:
    n = len(led)
    staked = float(led["stake"].sum()) if n else 0.0
    profit = float(led["profit"].sum()) if n else 0.0
    out = {
        "n_bets": n, "staked": staked, "profit": profit,
        "yield": profit / staked if staked else 0.0,
        "roi": profit / initial,
        "hit_rate": float(led["won"].mean()) if n else 0.0,
        "avg_odds": float(led["odds"].mean()) if n else 0.0,
        "avg_edge": float(led["edge"].mean()) if n else 0.0,
        "max_drawdown": max_drawdown(led["bankroll"], initial) if n else 0.0,
        "max_drawdown_units": max_drawdown_units(led["bankroll"], initial) if n else 0.0,
        "final_bankroll": float(led["bankroll"].iloc[-1]) if n else initial,
        # quiebra: con stake fijo el bankroll llega a <= 0; con Kelly, a < 1% del inicial
        "bust": bool((led["bankroll"] <= 0.01 * initial).any()) if n else False,
    }
    if "clv" in led.columns and n:
        out["clv_mean"] = float(led["clv"].mean())
        out["clv_pos_rate"] = float((led["clv"] > 0).mean())
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_mkt_close_under"] = 1 - df["mkt_close_over"]
    df["_lgbm_punder"] = 1 - df["lgbm_pover"]
    df["_dc_punder"] = 1 - df["dc_pover"]
    for m in ("rl", "rlm"):
        if f"{m}_pover" in df:
            df[f"_{m}_punder"] = 1 - df[f"{m}_pover"]
    return df


def backtest(cfg: dict) -> pd.DataFrame:
    bcfg = cfg["backtest"]
    oos = _prepare(load_oos(cfg))
    rep = get_path(cfg, "reports")
    init = float(bcfg["initial_bankroll"])
    rows, all_bets = [], []
    seasons = sorted(oos["season"].unique())
    log.info("Backtest sobre temporadas fuera de muestra: %s (umbral=%.2f)", ", ".join(seasons), bcfg["threshold"])
    for market in bcfg["markets"]:
        spec = MARKETS[market]
        for model in bcfg["models"]:
            curves = {}
            if spec["models"][model][0] not in oos.columns:
                log.warning("Modelo %s sin predicciones (¿falta `train`?); se omite", model)
                continue
            for price in ("close", "open"):
                sub = oos.dropna(subset=spec["models"][model][:1] + spec[price])
                bets = select_bets(sub, spec["models"][model], spec[price], spec["won"](sub), spec["outcomes"],
                                   bcfg["threshold"], bcfg["one_bet_per_match"], bcfg["max_odds"],
                                   fair_close_cols=spec["fair_close"] if price == "open" else None)
                for strat, led in (("flat", run_flat(bets, bcfg["flat_stake"], init)),
                                   ("kelly", run_kelly(bets, bcfg["kelly_fraction"], init, bcfg["kelly_max_fraction"]))):
                    s = summarize(led, init)
                    rows.append({"market": market, "model": model, "price": price, "strategy": strat, **s})
                    if not led.empty:
                        all_bets.append(led.assign(market=market, model=model, price=price, strategy=strat))
                        curves[f"{'Cierre' if price == 'close' else 'Apertura'} · {'stake fijo' if strat == 'flat' else 'Kelly 25%'}"] = led["bankroll"].reset_index(drop=True)
                    for season, g in led.groupby("season"):
                        gs = summarize(g.assign(bankroll=init + g["profit"].cumsum()), init)
                        rows.append({"market": market, "model": model, "price": price, "strategy": strat,
                                     "season": season, **gs})
            bankroll_curves(curves, rep / f"bankroll_{market}_{model}.png",
                            f"Bankroll · {market.upper()} · {MODEL_NAMES.get(model, model)}"
                            f" (umbral {bcfg['threshold']:.0%})", init)
    res = pd.DataFrame(rows)
    res["season"] = res.get("season", pd.Series(dtype=object)).fillna("TOTAL")
    cols = ["market", "model", "price", "strategy", "season"] + [c for c in res.columns if c not in
                                                                  ("market", "model", "price", "strategy", "season")]
    res = res[cols]
    res.round(4).to_csv(rep / "backtest_summary.csv", index=False)
    total = res[res["season"] == "TOTAL"]
    (rep / "backtest_summary.json").write_text(
        json.dumps({"seasons": seasons, "threshold": bcfg["threshold"],
                    "results": total.round(4).to_dict(orient="records")}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if all_bets:
        pd.concat(all_bets).to_csv(rep / "backtest_bets.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        show = total[["market", "model", "price", "strategy", "n_bets", "profit", "yield", "roi",
                      "max_drawdown_units", "final_bankroll", "bust"] + [c for c in ("clv_mean",) if c in total]]
        log.info("Backtest (total):\n%s", show.round(4).to_string(index=False))
    return res
