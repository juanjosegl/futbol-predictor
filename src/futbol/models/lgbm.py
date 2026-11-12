"""LightGBM (multiclase 1X2 o binario over/under) con calibración isotónica o Platt."""
from __future__ import annotations

import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

log = logging.getLogger(__name__)


def prepare_X(df: pd.DataFrame, features: list[str], leagues: list[str]) -> pd.DataFrame:
    X = df[features].astype(float).copy()
    X["league"] = pd.Categorical(df["league"].astype(str), categories=leagues)
    return X


class CalibratedLGBM:
    """Base LightGBM entrenado en `train`, early stopping y calibración sobre `cal` (temporal posterior)."""

    def __init__(self, params: dict, calibration: str = "isotonic", early_stopping_rounds: int = 100,
                 task: str = "multiclass"):
        self.params = dict(params)
        self.calibration = calibration
        self.esr = early_stopping_rounds
        self.task = task

    def fit(self, X_train, y_train, X_cal, y_cal) -> "CalibratedLGBM":
        obj = {"objective": "multiclass", "num_class": 3} if self.task == "multiclass" else {"objective": "binary"}
        self.base = lgb.LGBMClassifier(**self.params, **obj)
        self.base.fit(X_train, y_train, eval_X=(X_cal,), eval_y=(y_cal,),
                      callbacks=[lgb.early_stopping(self.esr, verbose=False), lgb.log_evaluation(0)])
        self.best_iteration = self.base.best_iteration_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cal = CalibratedClassifierCV(FrozenEstimator(self.base), method=self.calibration)
            self.cal.fit(X_cal, y_cal)
        self.features = list(X_train.columns)
        return self

    def predict_proba_raw(self, X) -> np.ndarray:
        return self.base.predict_proba(X[self.features])

    def predict_proba(self, X) -> np.ndarray:
        p = self.cal.predict_proba(X[self.features])
        p = np.clip(p, 1e-4, 1)
        return p / p.sum(axis=1, keepdims=True)

    def feature_importance(self) -> pd.Series:
        imp = self.base.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.base.booster_.feature_name()).sort_values(ascending=False)
