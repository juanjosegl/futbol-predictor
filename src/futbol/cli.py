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
            no_ledger: bool = typer.Option(False, "--no-ledger", help="No guardar en reports/ledger.csv."),
            config: Optional[Path] = ConfigOpt):
    """Probabilidades 1X2 y O/U 2.5, cuota justa y value bets para próximos partidos (se guardan en el ledger)."""
    from .evaluation.predict import predict as _predict

    _predict(_init(config), fixtures, out, record=not no_ledger)


@app.command()
def exam(desde: str = typer.Argument(..., help="Fecha inicial AAAA-MM-DD"),
         hasta: Optional[str] = typer.Argument(None, help="Fecha final AAAA-MM-DD (por defecto, el último partido)"),
         liga: Optional[list[str]] = typer.Option(None, "--liga", "-l", help="Filtrar ligas (E0, SP1...). Repetible."),
         threshold: Optional[float] = typer.Option(None, help="Umbral de valor para simular apuestas."),
         config: Optional[Path] = ConfigOpt):
    """Examen: predice partidos YA JUGADOS entrenando solo con lo anterior y compara con el resultado real."""
    from .evaluation.exam import run_exam

    cfg = _init(config)
    if hasta is None:
        from .features.build import load_dataset

        hasta = str(load_dataset(cfg)["date"].max().date())
    run_exam(cfg, desde, hasta, liga, threshold)


@app.command()
def settle(config: Optional[Path] = ConfigOpt):
    """Liquida el ledger de predicciones con los resultados reales descargados (paper trading)."""
    from .evaluation.ledger import settle as _settle

    _settle(_init(config))


@app.command()
def informe(config: Optional[Path] = ConfigOpt):
    """Regenera reports/seguimiento.xlsx (predicciones, aciertos/fallos y análisis) a partir del ledger."""
    from .evaluation.informe import write_report

    write_report(_init(config))


@app.command()
def actualizar(fixtures: Optional[Path] = typer.Option(None, help="CSV de próximos partidos (por defecto data/fixtures/upcoming.csv)."),
               no_train: bool = typer.Option(False, "--no-train", help="No reentrenar (más rápido)."),
               config: Optional[Path] = ConfigOpt):
    """Rutina semanal: datos nuevos -> reentrenar -> liquidar lo predicho -> predecir lo próximo -> documento."""
    from .evaluation.informe import write_report
    from .evaluation.ledger import load_ledger, settle as _settle
    from .evaluation.predict import predict as _predict
    from .features.build import build_dataset
    from .ingest import download_all
    from .models.training import walk_forward

    cfg = _init(config)
    log.info("===== 1/5 DESCARGA (temporada en curso y próximos partidos) =====")
    status = download_all(cfg, refresh_current=True)
    if cfg.get("_football_data_down") and status.get("football_data_org") != "ok":
        log.warning("SIN DATOS NUEVOS: football-data.co.uk está bloqueado en esta red y no hay token de "
                    "football-data.org (ver README). Se sigue con los datos en caché; los resultados de los "
                    "partidos pendientes puedes escribirlos a mano en reports/ledger.csv (columnas fthg, ftag).")
    log.info("===== 2/5 DATASET =====")
    build_dataset(cfg)
    if not no_train:
        log.info("===== 3/5 ENTRENAMIENTO =====")
        walk_forward(cfg)
    if not load_ledger(cfg).empty:
        log.info("===== 4/5 LIQUIDAR PREDICCIONES ANTERIORES =====")
        _settle(cfg)
    fx = fixtures or get_path(cfg, "fixtures") / "upcoming.csv"
    if Path(fx).exists():
        log.info("===== 5/5 PREDECIR PRÓXIMOS PARTIDOS (%s) =====", fx)
        try:
            _predict(cfg, Path(fx), only_future=True)
        except ValueError as e:
            log.warning("No se predijo nada: %s", e)
    else:
        log.warning("No hay %s: crea un CSV con los próximos partidos y sus cuotas (ver README) y ejecuta "
                    "`futbol predict ARCHIVO.csv`.", fx)
    write_report(cfg)


LigaOpt = typer.Option(None, "--liga", "-l", help="Solo estas ligas (id de config.yaml: nba, atp, col.1...). Repetible.")


@app.command()
def hoy(dias: Optional[int] = typer.Option(None, "--dias", "-d", help="Días hacia adelante (por defecto deportes.dias_adelante)."),
        top: int = typer.Option(25, "--top", "-n", help="Cuántos pronósticos mostrar en pantalla."),
        liga: Optional[list[str]] = LigaOpt,
        no_ledger: bool = typer.Option(False, "--no-ledger", help="No registrar en el ledger ni liquidar."),
        config: Optional[Path] = ConfigOpt):
    """UN SOLO COMANDO: actualiza datos de todos los deportes, liquida lo pendiente y da los pronósticos del día."""
    from .deportes.hoy import hoy as _hoy

    _hoy(_init(config), days=dias, top=top, record=not no_ledger, only=liga)


@app.command()
def programar(hora: Optional[list[str]] = typer.Option(None, "--hora", help="Hora(s) HH:MM de ejecución diaria (por defecto 07:00 y 17:00). Repetible."),
              quitar: bool = typer.Option(False, "--quitar", help="Desactivar la ejecución automática."),
              estado: bool = typer.Option(False, "--estado", help="Mostrar si está activa y su última salida."),
              probar: bool = typer.Option(False, "--probar", help="Lanzarla ahora en segundo plano para comprobar que funciona."),
              config: Optional[Path] = ConfigOpt):
    """Hace que `futbol hoy` se ejecute solo todos los días (macOS)."""
    from . import programar as P

    cfg = _init(config)
    if quitar:
        P.remove()
    elif estado:
        P.status(cfg)
    elif probar:
        P.run_now()
        log.info("Lanzada. En ~30 s mira logs/programado.log o ejecuta `futbol programar --estado`.")
    else:
        P.install(cfg, hora or ["07:00", "17:00"])


@app.command()
def apuesta(equipo: str = typer.Argument(..., help="Nombre (o parte) de uno de los dos contrincantes del partido."),
            cuota: float = typer.Option(..., "--cuota", "-c", help="Cuota decimal a la que apostaste en tu casa (p. ej. 1.35)."),
            monto: Optional[float] = typer.Option(None, "--monto", "-m", help="Dinero apostado."),
            papel: bool = typer.Option(False, "--papel", help="Apuesta simulada, sin dinero (para medir antes de arriesgar)."),
            a: Optional[str] = typer.Option(None, "--a", help="A qué apostaste si no es al pronóstico: 1, X, 2, Over, Under o un nombre."),
            fecha: Optional[str] = typer.Option(None, "--fecha", help="AAAA-MM-DD, si hay varios partidos pendientes de ese equipo."),
            config: Optional[Path] = ConfigOpt):
    """Anota una apuesta simple (real o de papel) sobre un partido pendiente; se liquida sola con el resultado."""
    from .evaluation.ledger import record_bet

    cfg = _init(config)
    try:
        record_bet(cfg, equipo, cuota, monto, a, fecha, papel)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        raise typer.Exit(1) from e


@app.command()
def ver(excel: bool = typer.Option(False, "--excel", help="Abrir los .xlsx en vez de la versión web (necesita Excel o Numbers)."),
        config: Optional[Path] = ConfigOpt):
    """Abre en el navegador la selección del día (reports/hoy.html) y el seguimiento (reports/seguimiento.html)."""
    import subprocess
    import sys
    import webbrowser

    rep = get_path(_init(config), "reports")
    ext = ".xlsx" if excel else ".html"
    files = [f for f in (rep / f"hoy{ext}", rep / f"seguimiento{ext}") if f.exists()]
    if not files:
        log.warning("Todavía no hay documentos: ejecuta primero `futbol hoy`.")
    for f in files:
        log.info("Abriendo %s", f)
        if sys.platform == "darwin":
            r = subprocess.run(["open", str(f)], capture_output=True, text=True)
            if r.returncode != 0:
                log.error("No hay ninguna aplicación para abrir %s. %s", f.name,
                          "Usa `futbol ver` (versión web) o instala Numbers (gratis en el App Store)." if excel
                          else "Ábrelo arrastrándolo a Chrome o Safari.")
        elif sys.platform.startswith("win"):
            __import__("os").startfile(str(f))  # noqa: S606
        else:
            webbrowser.open(f.as_uri())


@app.command("deportes-descargar")
def deportes_descargar(refresh: bool = typer.Option(False, "--refresh", help="Volver a descargar todo el historial."),
                       liga: Optional[list[str]] = LigaOpt, config: Optional[Path] = ConfigOpt):
    """Descarga resultados, calendario y cuotas actuales de las ligas de `deportes.ligas` (API pública de ESPN)."""
    from .deportes import espn

    log.info("Estado: %s", espn.download(_init(config), refresh=refresh, only=liga))


@app.command("deportes-evaluar")
def deportes_evaluar(liga: Optional[list[str]] = LigaOpt, config: Optional[Path] = ConfigOpt):
    """Valida el modelo multideporte en años no vistos: % de acierto y calibración por liga (reports/deportes_*.csv)."""
    from .deportes import espn, modelo

    cfg = _init(config)
    met = modelo.evaluate(cfg, espn.load_events(cfg, liga), liga)
    with __import__("pandas").option_context("display.width", 220, "display.max_columns", 20):
        log.info("Resumen (ordenado por %% de acierto):\n%s", met.sort_values("acierto", ascending=False)[
            ["nombre", "deporte", "partidos_test", "acierto", "acierto_local_siempre", "acierto_alta", "n_alta",
             "log_loss"]].round(3).to_string(index=False))


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
    # Examen automático de las últimas 4 semanas con datos (para tener métricas recientes siempre)
    try:
        import pandas as pd

        from .evaluation.exam import run_exam
        from .features.build import load_dataset

        last = load_dataset(cfg)["date"].max()
        log.info("===== EXAMEN (últimas 4 semanas) =====")
        run_exam(cfg, str((last - pd.Timedelta(days=27)).date()), str(last.date()))
    except Exception as e:  # noqa: BLE001
        log.warning("Examen automático omitido: %s", e)
    fx_dir = get_path(cfg, "fixtures")
    candidates = [fixtures] if fixtures else [fx_dir / "upcoming.csv", fx_dir / "ejemplo_fixtures.csv"]
    fx = next((p for p in candidates if p and Path(p).exists()), None)
    if fx:
        log.info("===== PREDICT (%s) =====", fx)
        try:  # el ejemplo no va al ledger; de upcoming.csv solo los partidos que aún no se han jugado
            _predict(cfg, Path(fx), record=Path(fx).name != "ejemplo_fixtures.csv",
                     only_future=Path(fx).name == "upcoming.csv")
        except ValueError as e:
            log.warning("No se predijo nada: %s", e)
    else:
        log.info("Sin CSV de próximos partidos: se omite predict.")
    log.info("Pipeline completo en %.1fs. Resultados en %s", time.time() - t0, get_path(cfg, "reports"))


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
