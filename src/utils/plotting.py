"""Plot helpers shared by the experiment scripts (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_COLORS = {"HQ": "#1b7837", "AQ": "#d9a400", "LQ": "#b2182b"}
MARKERS = {"HQ": "o", "AQ": "s", "LQ": "^"}
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pca(Z, y, names, ev, path, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for ci, c in enumerate(names):
        m = y == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=28, label=c, c=CLASS_COLORS.get(c), marker=MARKERS.get(c, "o"),
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.2f} %)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.2f} %)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_confusion(cm, names, path, title=""):
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    _save(fig, path)


def bar_with_err(df: pd.DataFrame, x: str, y: str, err: str, path, hue: str | None = None,
                 title="", ylabel="Accuracy", ref: float | None = None, ylim=(0, 1), rotate=0):
    """Grouped bar chart with error bars (mean +- std over CV repeats)."""
    fig_w = max(4.5, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(min(fig_w, 12), 3.4))
    if hue is None:
        ax.bar(df[x].astype(str), df[y], yerr=df[err], color="#4c72b0", capsize=3)
    else:
        groups = list(dict.fromkeys(df[x]))
        hues = list(dict.fromkeys(df[hue]))
        wdt = 0.8 / len(hues)
        palette = plt.get_cmap("tab10")
        for hi, h in enumerate(hues):
            sub = df[df[hue] == h].set_index(x).reindex(groups)
            ax.bar(np.arange(len(groups)) + (hi - (len(hues) - 1) / 2) * wdt, sub[y], wdt,
                   yerr=sub[err], label=str(h), color=palette(hi), capsize=2)
        ax.set_xticks(range(len(groups)), groups)
        ax.legend(frameon=False, fontsize=7, ncol=min(len(hues), 5))
    if ref is not None:
        ax.axhline(ref, color="grey", ls="--", lw=0.8)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    _save(fig, path)
