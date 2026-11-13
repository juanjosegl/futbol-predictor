"""Registra el mejor run con el alias `challenger`.

Busca los runs terminados de `MLFLOW_EXPERIMENT_NAME` con la etiqueta
`tipo=modelo`, se queda con el de menor log loss y publica ese modelo en el
Model Registry con el alias `MLFLOW_MODEL_ALIAS`.

Es el entry point `register` de `MLproject` (`mlflow run . -e register`).
"""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from . import config


def _uri_modelo(cliente: MlflowClient, run_id: str) -> str:
    """En MLflow 3 el modelo queda como logged model, no como carpeta del run."""
    run = cliente.get_run(run_id)
    salidas = list(run.outputs.model_outputs) if run.outputs is not None and run.outputs.model_outputs else []
    if not salidas:
        raise SystemExit(f"El run {run_id} no tiene un modelo guardado en MLflow.")
    return f"models:/{max(salidas, key=lambda s: s.step).model_id}"


def registrar() -> None:
    config.activar_seguimiento()
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experimento is None:
        raise SystemExit(f"No existe el experimento {config.MLFLOW_EXPERIMENT_NAME}. Ejecuta antes `mlflow run .`.")

    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string=(f"attributes.status = 'FINISHED' and tags.{config.TAG_TIPO} = 'modelo' "
                       f"and metrics.{config.METRIC} >= 0"),
        order_by=[f"metrics.{config.METRIC} ASC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No hay runs de modelo con la métrica {config.METRIC}. Ejecuta antes `mlflow run .`.")

    mejor = runs[0]
    valor = mejor.data.metrics[config.METRIC]
    nombre = mejor.data.tags.get("mlflow.runName", mejor.info.run_id)
    version = mlflow.register_model(
        _uri_modelo(cliente, mejor.info.run_id),
        config.MLFLOW_MODEL_NAME,
        tags={"estrategia": nombre, config.METRIC: f"{valor:.6f}"},
    )
    cliente.update_registered_model(
        config.MLFLOW_MODEL_NAME,
        description="Probabilidades 1X2 (local, empate, visitante) de partidos de las 5 grandes ligas.",
    )
    cliente.update_model_version(
        config.MLFLOW_MODEL_NAME, version.version,
        description=f"{nombre}: {config.METRIC}={valor:.4f} en temporadas >= {config.TEST_FROM_SEASON}.",
    )
    cliente.set_registered_model_alias(config.MLFLOW_MODEL_NAME, config.MLFLOW_MODEL_ALIAS, version.version)
    print(f"Mejor run: {nombre} ({config.METRIC}={valor:.6f}). "
          f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS} -> versión {version.version}")


if __name__ == "__main__":
    registrar()
