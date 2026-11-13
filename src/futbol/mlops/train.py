"""Entrena cuatro modelos 1X2 y los deja en MLflow, un run por modelo.

Runs del experimento `MLFLOW_EXPERIMENT_NAME`:

- `preprocesamiento`: partición temporal, features y muestras.
- `regresion_logistica`, `random_forest`, `hist_gradient_boosting`, `lightgbm`:
  cada uno guarda log loss, Brier y accuracy sobre las temporadas de prueba,
  sus parámetros y el modelo (un `Pipeline` de sklearn: imputación, escalado
  si aplica y clasificador).
- `referencia_mercado_cierre`: las mismas métricas con las probabilidades de
  las cuotas de cierre. Es la vara de medir; no tiene modelo y no se registra.

El modelo se guarda con `pyfunc_predict_fn="predict_proba"`: la API devuelve
las tres probabilidades (local, empate, visitante) y no solo la clase.

Es el entry point `main` de `MLproject` (`mlflow run .`).
"""
from __future__ import annotations

import os
import warnings

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation import metrics as M
from . import config
from .data import MERCADO, cargar_particion, registrar_preprocesamiento


def _modelos() -> list[tuple[str, Pipeline]]:
    s = config.RANDOM_SEED
    imputar = ("imputar", SimpleImputer(strategy="median"))
    return [
        ("regresion_logistica", Pipeline([imputar, ("escalar", StandardScaler()),
                                          ("modelo", LogisticRegression(C=0.05, max_iter=2000, random_state=s))])),
        ("random_forest", Pipeline([imputar, ("modelo", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=40, max_features="sqrt", n_jobs=-1, random_state=s))])),
        ("hist_gradient_boosting", Pipeline([("modelo", HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=5.0, random_state=s))])),
        ("lightgbm", Pipeline([("modelo", lgb.LGBMClassifier(
            objective="multiclass", n_estimators=300, learning_rate=0.02, num_leaves=15,
            min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=5.0, random_state=s, verbose=-1))])),
    ]


def _metricas(y, p) -> dict:
    return {"log_loss": M.log_loss_multi(y, p), "brier": M.brier_multi(y, p), "accuracy": M.accuracy(y, p)}


def _datasets(x_train, y_train, x_test, y_test):
    """Los dos conjuntos que muestra la columna Dataset de cada run."""
    def conjunto(x, y, nombre):
        tabla = x.copy()
        tabla[config.TARGET] = y.to_numpy()
        return mlflow.data.from_pandas(tabla, source=str(config.DATA_PATH), targets=config.TARGET, name=nombre)

    return conjunto(x_train, y_train, "partidos_train"), conjunto(x_test, y_test, "partidos_test")


def _parametros(pipeline: Pipeline) -> dict:
    out = {"pasos": " > ".join(pipeline.named_steps)}
    for clave, valor in pipeline.named_steps["modelo"].get_params().items():
        out[clave] = valor if isinstance(valor, (int, float, str, bool)) else str(valor)
    return out


def entrenar() -> None:
    # `mlflow run` ya abrió un run padre; los modelos van en runs propios para compararlos.
    os.environ.pop("MLFLOW_RUN_ID", None)
    config.activar_seguimiento()
    mlflow.sklearn.autolog(disable=True)

    x_train, x_test, y_train, y_test, test, material = cargar_particion()
    with mlflow.start_run(run_name="preprocesamiento"):
        registrar_preprocesamiento(material)
    r = material["resumen"]
    print(f"Partición temporal: train {r['filas_train']} partidos ({r['temporadas_train'][0]}-"
          f"{r['temporadas_train'][-1]}), test {r['filas_test']} ({', '.join(r['temporadas_test'])}), "
          f"{r['n_features']} features")

    datos_train, datos_test = _datasets(x_train, y_train, x_test, y_test)
    comunes = {"test_from_season": config.TEST_FROM_SEASON, "random_seed": config.RANDOM_SEED,
               "n_features": x_train.shape[1], "filas_train": len(x_train), "filas_test": len(x_test)}
    ejemplo = x_test.head(1)
    y = y_test.to_numpy()

    for nombre, pipeline in _modelos():
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tag(config.TAG_TIPO, "modelo")
            mlflow.log_input(datos_train, context="training")
            mlflow.log_input(datos_test, context="evaluation")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(x_train, y_train)
                p = pipeline.predict_proba(x_test)
            mlflow.log_params({**_parametros(pipeline), **comunes, "estrategia": nombre})
            met = _metricas(y, p)
            mlflow.log_metrics(met)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=ejemplo,
                pyfunc_predict_fn="predict_proba",
                serialization_format="pickle",
            )
            print(f"{nombre:24s} log_loss={met['log_loss']:.4f}  brier={met['brier']:.4f}  "
                  f"accuracy={met['accuracy']:.4f}")

    mercado = test[MERCADO].to_numpy(dtype=float)
    ok = ~np.isnan(mercado).any(axis=1)
    with mlflow.start_run(run_name="referencia_mercado_cierre"):
        mlflow.set_tag(config.TAG_TIPO, "referencia")
        mlflow.log_params({**comunes, "estrategia": "probabilidades implícitas de las cuotas de cierre",
                           "filas_con_cuota": int(ok.sum())})
        met = _metricas(y[ok], mercado[ok])
        mlflow.log_metrics(met)
        print(f"{'referencia_mercado':24s} log_loss={met['log_loss']:.4f}  brier={met['brier']:.4f}  "
              f"accuracy={met['accuracy']:.4f}")


if __name__ == "__main__":
    entrenar()
