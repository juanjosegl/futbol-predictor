"""Genera un documento con la petición y la respuesta de la API de un modelo registrado.

El archivo queda en `ejemplos-api/<modelo>__<alias>.md`.

    python -m futbol.mlops.ejemplo_api
    python -m futbol.mlops.ejemplo_api --alias challenger --puerto 5002
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import Model
from mlflow.tracking import MlflowClient

from . import config

SALIDA = config.RAIZ / "ejemplos-api"
PUERTO = 5001


def _jsonable(valor):
    if isinstance(valor, dict):
        return {k: _jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(v) for v in valor]
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and math.isnan(valor):
        return None
    return valor


def _payload(ejemplo: pd.DataFrame) -> dict:
    return {"dataframe_split": {"columns": list(ejemplo.columns), "data": _jsonable(ejemplo.to_numpy().tolist())}}


def _esquema(firma, lado: str) -> str:
    tensor = getattr(firma, lado, None) if firma is not None else None
    return "No hay firma registrada." if tensor is None else json.dumps(tensor.to_dict(), indent=2, ensure_ascii=False)


def generar(modelo: str, alias: str, puerto: int = PUERTO) -> Path:
    config.activar_seguimiento()
    uri = f"models:/{modelo}@{alias}"
    version = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI).get_model_version_by_alias(modelo, alias)
    info = mlflow.models.get_model_info(uri)
    ejemplo = Model.load(uri).load_input_example(uri)
    payload = _payload(ejemplo)
    respuesta = {"predictions": _jsonable(mlflow.pyfunc.load_model(uri).predict(ejemplo).tolist())}

    curl = (f"curl http://127.0.0.1:{puerto}/invocations \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(payload, ensure_ascii=False)}'")
    lineas = [
        f"# {modelo} @ {alias}", "",
        f"- URI: `{uri}`",
        f"- Versión: {version.version}",
        f"- Estrategia: {version.tags.get('estrategia', '-')}",
        f"- Endpoint: `POST http://127.0.0.1:{puerto}/invocations`",
        "- Respuesta: una fila por partido con las probabilidades `[local, empate, visitante]`.", "",
        "## Esquema de entrada", "", "```json", _esquema(info.signature, "inputs"), "```", "",
        "## Esquema de salida", "", "```json", _esquema(info.signature, "outputs"), "```", "",
        "## Petición", "", "```bash", curl, "```", "",
        "## Respuesta", "", "```json", json.dumps(respuesta, indent=2), "```", "",
    ]
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / f"{modelo}__{alias}.md"
    destino.write_text("\n".join(lineas), encoding="utf-8")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el ejemplo de petición y respuesta de un modelo registrado.")
    parser.add_argument("--modelo", default=config.MLFLOW_MODEL_NAME)
    parser.add_argument("--alias", action="append", default=None)
    parser.add_argument("--puerto", type=float, default=PUERTO)
    args = parser.parse_args()
    for alias in args.alias or [config.MLFLOW_MODEL_ALIAS]:
        print(generar(args.modelo, alias, int(args.puerto)))


if __name__ == "__main__":
    main()
