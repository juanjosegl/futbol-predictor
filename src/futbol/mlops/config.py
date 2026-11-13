"""Nombres del experimento y del modelo, semilla y decisiones del flujo MLOps.

Los valores salen de `.env`; si falta el archivo se usan los de `.env.example`.
El objetivo es el resultado 1X2 (0 = local, 1 = empate, 2 = visitante) y el
mejor run es el de menor log loss en las temporadas de prueba.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[3]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "futbol-1x2")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "futbol-1x2")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_FROM_SEASON = os.getenv("TEST_FROM_SEASON", "2526")

_ruta = os.getenv("DATA_PATH", "data/processed/matches.parquet")
DATA_PATH = Path(_ruta) if Path(_ruta).is_absolute() else RAIZ / _ruta

# Decisiones fijas del flujo.
TARGET = "y_1x2"
CLASES = ("local", "empate", "visitante")
METRIC = "log_loss"          # menor es mejor
MAX_NULOS = 0.5              # se descartan features con más de 50 % de nulos en train
TAG_TIPO = "tipo"            # "modelo" o "referencia"; register.py solo mira "modelo"


def activar_seguimiento() -> None:
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
