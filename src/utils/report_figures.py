"""Report-only figure: block diagram of the three fusion levels used in Experiment A.
Colours match the article figures (low = blue, mid = orange, high = aqua).
Run: python -m src.utils.report_figures
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"
SURFACE, INK, INK2, BORDER = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
LEVEL = {"Low level": "#2a78d6", "Mid level": "#eb6834", "High level": "#1baf7a"}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def box(ax, x, y, w, h, text, edge=BORDER, lw=1.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc="white", ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontsize=8.2,
            fontweight=weight, linespacing=1.25)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0))


def fig_fusion_levels():
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = {
        "Low level": ["8 conductance curves\n8 sensors × 300 s", "Concatenate\n2,400 values",
                      "Standardise + PCA\n(10 components, in-fold)", "Classifier\nLR / SVM / RF / k-NN / MLP",
                      "HQ / AQ / LQ"],
        "Mid level": ["8 conductance curves\n8 sensors × 300 s", "15 features per sensor\n(level, slope, area, timing, FFT)",
                      "Concatenate\n120 features", "Classifier\nLR / SVM / RF / k-NN / MLP", "HQ / AQ / LQ"],
        "High level": ["8 conductance curves\n8 sensors × 300 s", "15 features per sensor",
                       "8 per-sensor classifiers\n(one per sensor)", "Combiner (inner CV)\nvote / mean / weighted / stacking",
                       "HQ / AQ / LQ"],
    }
    ws = [0.14, 0.15, 0.15, 0.17, 0.12]
    gap = 0.038
    xs = [0.105]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    h = 0.17
    for r, (name, steps) in enumerate(rows.items()):
        y = 0.74 - r * 0.3
        ax.text(0.005, y + h / 2, name, ha="left", va="center", color=LEVEL[name], fontsize=9.5, fontweight="bold")
        for i, (x, w, t) in enumerate(zip(xs, ws, steps)):
            box(ax, x, y, w, h, t, edge=LEVEL[name] if i in (1, 2, 3) else BORDER,
                lw=1.3 if i in (1, 2, 3) else 1.0, weight="bold" if i == 4 else "normal")
            if i < len(xs) - 1:
                arrow(ax, x + w + 0.006, xs[i + 1] - 0.006, y + h / 2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_fusion_levels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_fusion_levels.png")


if __name__ == "__main__":
    fig_fusion_levels()
