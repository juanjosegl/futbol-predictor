"""Métricas probabilísticas: log loss, Brier, accuracy y bootstrap pareado."""
from __future__ import annotations

import numpy as np


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1)
    return p / p.sum(axis=1, keepdims=True)


def log_loss_multi(y: np.ndarray, p: np.ndarray) -> float:
    p = _norm(p)
    return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y, int)])))


def brier_multi(y: np.ndarray, p: np.ndarray) -> float:
    """Brier multiclase: media de sum_k (p_k - 1[y=k])^2 (rango 0-2)."""
    p = _norm(p)
    onehot = np.eye(p.shape[1])[np.asarray(y, int)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == np.asarray(y, int)))


def binary_to_multi(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, float)
    return np.column_stack([1 - p1, p1])


def per_match_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _norm(p)
    return -np.log(p[np.arange(len(y)), np.asarray(y, int)])


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Diferencia media (a - b) de pérdidas por partido con IC 95% bootstrap."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
