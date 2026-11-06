"""CLI del pipeline: download, build, train, evaluate, backtest, predict, run-all."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer

from .config import get_path, load_config
from .logging_setup import setup_logging

app = typer.Typer(add_completion=False, help="Predicción de resultados de fútbol orientada a apuestas.")
log = logging.getLogger("futbol")

ConfigOpt = typer.Option(None, "--config", "-c", help="Ruta a config.yaml (por defecto el del proyecto).")


def _init(config: Optional[Path]) -> dict:
    cfg = load_config(config)
    log_file = setup_logging(get_path(cfg, "logs"))
    log.info("Log en %s", log_file)
    return cfg


@app.command()
def download(refresh: bool = typer.Option(False, "--refresh", help="Volver a descargar todo aunque esté en caché."),
             refresh_current: bool = typer.Option(False, "--refresh-current",
                                                  help="Re-descargar solo la temporada en curso y los fixtures."),
             config: Optional[Path] = ConfigOpt):
    """Descarga datos (football-data obligatoria; Understat, ClubElo y SofaScore opcionales)."""
    from .ingest import download_all

    download_all(_init(config), refresh=refresh, refresh_current=refresh_current)


@app.command()
def build(config: Optional[Path] = ConfigOpt):
    """Unifica fuentes, calcula features sin leakage y guarda data/processed/matches.parquet."""
    from .features.build import build_dataset

    build_dataset(_init(config))


@app.command()
def train(config: Optional[Path] = ConfigOpt):
    """Walk-forward por temporada (Dixon-Coles + LightGBM calibrado) y modelo final."""
    from .models.training import walk_forward

    walk_forward(_init(config))


@app.command()
def evaluate(config: Optional[Path] = ConfigOpt):
    """Log loss, Brier, accuracy y reliability diagrams vs el mercado de cierre."""
    from .evaluation.evaluate import evaluate as _evaluate

    _evaluate(_init(config))


@app.command()
def backtest(threshold: Optional[float] = typer.Option(None, help="Sobrescribe backtest.threshold."),
             config: Optional[Path] = ConfigOpt):
    """Backtest de apuestas (stake fijo y Kelly fraccional) sobre temporadas no vistas."""
    from .evaluation.backtest import backtest as _backtest

    cfg = _init(config)
    if threshold is not None:
        cfg["backtest"]["threshold"] = threshold
    _backtest(cfg)


@app.command()
def predict(fixtures: Path = typer.Argument(..., help="CSV: date,home,away[,odds_h,odds_d,odds_a][,league]"),
            out: Optional[Path] = typer.Option(None, "--out", "-o", help="CSV de salida (por defecto reports/predictions.csv)"),
            config: Optional[Path] = ConfigOpt):
    """Probabilidades 1X2, cuota justa y value bets para próximos partidos."""
    from .evaluation.predict import predict as _predict

    _predict(_init(config), fixtures, out)


@app.command("run-all")
def run_all(refresh: bool = typer.Option(False, "--refresh", help="Re-descargar todos los datos."),
            fixtures: Optional[Path] = typer.Option(None, help="CSV de próximos partidos a predecir al final."),
            config: Optional[Path] = ConfigOpt):
    """Ejecuta todo el pipeline: download -> build -> train -> evaluate -> backtest -> predict."""
    from .evaluation.backtest import backtest as _backtest
    from .evaluation.evaluate import evaluate as _evaluate
    from .evaluation.predict import predict as _predict
    from .features.build import build_dataset
    from .ingest import download_all
    from .models.training import walk_forward

    cfg = _init(config)
    t0 = time.time()
    steps = [("download", lambda: download_all(cfg, refresh=refresh)), ("build", lambda: build_dataset(cfg)),
             ("train", lambda: walk_forward(cfg)), ("evaluate", lambda: _evaluate(cfg)),
             ("backtest", lambda: _backtest(cfg))]
    for name, fn in steps:
        log.info("===== %s =====", name.upper())
        t = time.time()
        fn()
        log.info("%s completado en %.1fs", name, time.time() - t)
    fx_dir = get_path(cfg, "fixtures")
    candidates = [fixtures] if fixtures else [fx_dir / "upcoming.csv", fx_dir / "ejemplo_fixtures.csv"]
    fx = next((p for p in candidates if p and Path(p).exists()), None)
    if fx:
        log.info("===== PREDICT (%s) =====", fx)
        _predict(cfg, Path(fx))
    else:
        log.info("Sin CSV de próximos partidos: se omite predict.")
    log.info("Pipeline completo en %.1fs. Resultados en %s", time.time() - t0, get_path(cfg, "reports"))


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
