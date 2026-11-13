"""Cargar el dataset de partidos y partirlo por tiempo para el flujo MLOps.

- Solo partidos jugados (con resultado).
- Features `f_*` previas al partido, las mismas que usa el pipeline principal
  (`feature_columns`), sin cuotas de cierre ni columnas posteriores al partido.
- Se descartan las features con más de `MAX_NULOS` de nulos en entrenamiento
  (por ejemplo, xG, que casi no tiene cobertura).
- Partición temporal: entrenamiento con temporadas anteriores a
  `TEST_FROM_SEASON` y prueba desde esa temporada. Nunca aleatoria, para no
  usar información del futuro.

La imputación y el escalado viven dentro de cada `Pipeline` de sklearn, así
que la API recibe los valores de las features tal cual, sin transformar.
"""
from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..features.engineering import feature_columns
from . import config

MUESTRA = 10
INFO = ["match_id", "date", "league", "season", "home_team", "away_team", "fthg", "ftag"]
MERCADO = ["mkt_close_h", "mkt_close_d", "mkt_close_a"]


def _leer() -> pd.DataFrame:
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No está el dataset en {config.DATA_PATH}. Ejecuta `futbol build` o revisa DATA_PATH en .env."
        )
    return pd.read_parquet(config.DATA_PATH)


def cargar_particion():
    """Devuelve x_train, x_test, y_train, y_test, el bloque de prueba completo y el material para MLflow."""
    crudo = _leer()
    datos = crudo[crudo["played"]].sort_values("date", kind="mergesort").reset_index(drop=True)
    candidatas = feature_columns(datos, load_config())

    es_test = datos["season"].astype(str) >= config.TEST_FROM_SEASON
    train, test = datos[~es_test], datos[es_test]
    if train.empty or test.empty:
        raise ValueError(f"La partición con TEST_FROM_SEASON={config.TEST_FROM_SEASON} deja un lado vacío.")

    nulos = train[candidatas].isna().mean()
    features = [c for c in candidatas if nulos[c] <= config.MAX_NULOS]
    descartadas = sorted(set(candidatas) - set(features))

    x_train, x_test = train[features].astype(float), test[features].astype(float)
    y_train, y_test = train[config.TARGET].astype(int), test[config.TARGET].astype(int)

    material = {
        "resumen": {
            "archivo": config.DATA_PATH.name,
            "filas_leidas": int(len(crudo)),
            "filas_jugadas": int(len(datos)),
            "filas_train": int(len(train)),
            "filas_test": int(len(test)),
            "temporadas_train": sorted(train["season"].astype(str).unique().tolist()),
            "temporadas_test": sorted(test["season"].astype(str).unique().tolist()),
            "fecha_train": [str(train["date"].min().date()), str(train["date"].max().date())],
            "fecha_test": [str(test["date"].min().date()), str(test["date"].max().date())],
            "n_features_candidatas": len(candidatas),
            "n_features": len(features),
            "features_descartadas_por_nulos": descartadas,
            "max_nulos": config.MAX_NULOS,
            "distribucion_train": y_train.value_counts(normalize=True).sort_index().round(4).tolist(),
            "distribucion_test": y_test.value_counts(normalize=True).sort_index().round(4).tolist(),
        },
        "features": features,
        "muestra_cruda": test[INFO + features].head(MUESTRA).astype({"date": str}),
        "muestra_features": x_test.head(MUESTRA),
    }
    return x_train, x_test, y_train, y_test, test, material


def registrar_preprocesamiento(material: dict) -> None:
    """Escribe la partición, las features y unas muestras en el run activo."""
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("registrar_preprocesamiento necesita un run de MLflow activo.")
    r = material["resumen"]
    mlflow.set_tag(config.TAG_TIPO, "preprocesamiento")
    mlflow.log_params({
        "archivo": r["archivo"],
        "filas_leidas": r["filas_leidas"],
        "filas_jugadas": r["filas_jugadas"],
        "filas_train": r["filas_train"],
        "filas_test": r["filas_test"],
        "temporadas_train": ",".join(r["temporadas_train"]),
        "temporadas_test": ",".join(r["temporadas_test"]),
        "n_features_candidatas": r["n_features_candidatas"],
        "n_features": r["n_features"],
        "n_features_descartadas": len(r["features_descartadas_por_nulos"]),
        "max_nulos": r["max_nulos"],
        "particion": "temporal",
    })
    mlflow.log_dict({**r, "features": material["features"],
                     "pasos": ["solo partidos jugados",
                               "features pre-partido f_* (sin cuotas de cierre)",
                               f"descartar features con > {int(config.MAX_NULOS * 100)} % de nulos en train",
                               f"train: temporadas < {config.TEST_FROM_SEASON}; test: >= {config.TEST_FROM_SEASON}",
                               "imputación y escalado dentro del Pipeline de cada modelo"]},
                    "preprocesamiento/resumen.json")
    mlflow.log_table(material["muestra_cruda"], "preprocesamiento/muestra_cruda.json")
    mlflow.log_table(material["muestra_features"], "preprocesamiento/muestra_features.json")
