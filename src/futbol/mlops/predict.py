"""Carga el modelo por alias y predice probabilidades 1X2.

Resuelve `models:/<MLFLOW_MODEL_NAME>@<MLFLOW_MODEL_ALIAS>`. El modelo es un
`Pipeline` completo, así que el CSV de entrada lleva las features `f_*` sin
transformar (los nulos se imputan dentro).

Es el entry point `predict` de `MLproject` (`mlflow run . -e predict`).
Sin `--input` usa el `input_example` guardado con el modelo.
"""
from __future__ import annotations

import argparse

import mlflow.pyfunc
import pandas as pd
from mlflow.models import Model

from . import config


def _uri() -> str:
    return f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS}"


def predecir(datos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila de probabilidades (local, empate, visitante) por partido."""
    config.activar_seguimiento()
    p = mlflow.pyfunc.load_model(_uri()).predict(datos)
    return pd.DataFrame(p, columns=list(config.CLASES), index=datos.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice probabilidades 1X2 con el modelo del alias.")
    parser.add_argument("--input", default=None, help="CSV con las features. Si se omite, usa el input_example.")
    args = parser.parse_args()
    config.activar_seguimiento()
    if args.input:
        datos = pd.read_csv(args.input)
    else:
        datos = Model.load(_uri()).load_input_example(_uri())
    if datos is None:
        raise SystemExit("El modelo no tiene input_example. Pasa un CSV con --input.")
    print(_uri())
    print(predecir(datos).round(4).to_string())


if __name__ == "__main__":
    main()
