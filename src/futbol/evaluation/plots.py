"""Gráficos PNG: reliability diagrams y curvas de bankroll."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta categórica de referencia (slots 1-4, orden fijo) y tokens de superficie/texto
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability_diagram(y_by_class: dict[str, np.ndarray], probs: dict[str, dict[str, np.ndarray]],
                        path: Path, title: str, n_bins: int = 10) -> None:
    """y_by_class: {clase: y binaria}; probs: {modelo: {clase: p}}. Bins por cuantiles."""
    classes = list(y_by_class)
    fig, axes = plt.subplots(1, len(classes), figsize=(4.6 * len(classes), 4.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        _style(ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT2, linewidth=1, label="Calibración perfecta")
        y = y_by_class[cls]
        hi = 0.0
        for i, (model, pc) in enumerate(probs.items()):
            p = pc[cls]
            bins = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
            df = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
            ax.plot(df["p"], df["y"], color=SERIES[i], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=model)
            hi = max(hi, df["p"].max(), df["y"].max())
        lim = min(1.0, hi + 0.05)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(cls, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Probabilidad predicha", color=TEXT2, fontsize=9)
        ax.set_ylabel("Frecuencia observada", color=TEXT2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="upper left")
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def bankroll_curves(curves: dict[str, pd.Series], path: Path, title: str, initial: float) -> None:
    """curves: {etiqueta: serie de bankroll indexada por nº de apuesta}. Máx. 4 series."""
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.axhline(initial, color=TEXT2, linewidth=1, linestyle="--")
    for i, (label, s) in enumerate(list(curves.items())[:4]):
        if s.empty:
            continue
        ax.plot(np.arange(len(s)), s.values, color=SERIES[i], linewidth=1.5, label=label)
        ax.annotate(f"{s.values[-1]:.1f}", (len(s) - 1, s.values[-1]), xytext=(4, 6 * (i % 2) - 3),
                    textcoords="offset points", color=TEXT, fontsize=8.5, va="center")
    ax.set_xlabel("Nº de apuesta (orden cronológico)", color=TEXT2, fontsize=9)
    ax.set_ylabel("Bankroll (unidades)", color=TEXT2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
