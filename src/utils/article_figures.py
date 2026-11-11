"""Article figures, generated only from logged results (results/*.csv, results/runs/*).

Style follows the dataviz method: validated categorical slots in fixed order
(blue #2a78d6, orange #eb6834, aqua #1baf7a; all-pairs CVD dE >= 9.2), secondary encoding
by marker shape (aqua is < 3:1 on the surface), thin marks, solid hairline grid, text in
ink colours (never series colours). Every figure has a table twin in the article.

Run: python -m src.utils.article_figures [names...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loaders import S1_CLASSES, S1_SENSORS, load_s1

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "article" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated)
CLS = dict(zip(S1_CLASSES, SLOT))                 # HQ, AQ, LQ
MARK = {"HQ": "o", "AQ": "s", "LQ": "^"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    "lines.linewidth": 1.5, "lines.markersize": 6, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- Fig. 1 -----------
def fig_s1_curves():
    """Class-mean sensor responses (resistance, kOhm) - analogue of the paper's Figs. 5-6."""
    X, y, _ = load_s1()
    t = np.arange(X.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), sharex=True)
    for s, ax in enumerate(axes.ravel()):
        for ci, c in enumerate(S1_CLASSES):
            m = X[y == ci, :, s].mean(0)
            ax.plot(t, m, color=CLS[c], lw=1.5, label=f"{c} (n={int((y == ci).sum())})")
        ax.set_title(S1_SENSORS[s], loc="left")
        ax.grid(True, axis="y")
        ax.tick_params(length=0)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Resistance (kΩ)")
    fig.suptitle("S1: class-mean response of the 8 MOX sensors (1 Hz, 300 s)", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0), handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig1_s1_sensor_curves")


# ---------------------------------------------------------------- Fig. 2 -----------
def fig_pca():
    """PCA of the mean-centred static parameter dG - analogue of the paper's Figs. 7 and 9."""
    from sklearn.decomposition import PCA
    from src.features.enose import delta_g
    X, y, _ = load_s1()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, feat, title in [(axes[0], "abs", "ΔG = Gmax − Gmin"), (axes[1], "rel", "ΔG / Gmin")]:
        F = delta_g(X, feat)
        Z = F - F.mean(0)
        p = PCA().fit(Z)
        P = p.transform(Z)
        ev = p.explained_variance_ratio_
        for ci, c in enumerate(S1_CLASSES):
            m = y == ci
            ax.scatter(P[m, 0], P[m, 1], s=34, color=CLS[c], marker=MARK[c], label=c,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.2f} %)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.2f} %)")
        ax.set_title(f"{title}, mean-centred · PC1+PC2 = {100 * ev[:2].sum():.2f} %", loc="left")
        ax.grid(True)
        ax.tick_params(length=0)
    axes[0].legend(loc="upper right", handletextpad=0.2)
    fig.tight_layout()
    _save(fig, "fig2_pca_static_parameter")


# ---------------------------------------------------------------- Fig. 3 -----------
def fusion_summary_rows():
    """Rows for the main comparison (also used for the article table)."""
    base = pd.read_csv(R / "baseline.csv")
    a = pd.read_csv(R / "expA_fusion_levels.csv")
    b = pd.read_csv(R / "expB_s1_s2_fusion.csv")

    def get(df, variant, scheme, col="accuracy_mean"):
        r = df[(df.variant == variant) & (df.scheme == scheme)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    rows = [
        ("Rodríguez et al. 2010, analysis 1 (reported)", "reported", np.nan, np.nan, 1.000),
        ("Rodríguez et al. 2010, analysis 2 (reported)", "reported", np.nan, np.nan, 0.925),
    ]
    spec = [
        ("Baseline reproduction: ΔG/Gmin, MLP(10)", base, "dG-rel|center|sgd"),
        ("A1 low: 8×300 curves → PCA → LR", a, "A1a|LR"),
        ("A1 low: 8-channel 1D-CNN + windows", a, "A1b|CNN+win"),
        ("A2 mid: 120 features → SVM", a, "A2|SVM"),
        ("A3 high: per-sensor SVM → stacking", a, "A3|SVM|stacking"),
    ]
    for label, df, v in spec:
        rows.append((label, "ours", get(df, v, "rskf5"), get(df, v, "rskf5", "accuracy_std"), get(df, v, "loo")))
    for label, v in [("B: S1 only (channel-agnostic, RF)", "S1only|RF"), ("B2 mid: S1+S2 pooled (RF)", "B2|RF"),
                     ("B3 high: S1-only ⊕ pooled, stacking (RF)", "B3|RF|stacking")]:
        rows.append((label, "ours", get(b, v, "rskf5"), get(b, v, "rskf5", "accuracy_std"), np.nan))
    return pd.DataFrame(rows, columns=["method", "source", "cv_mean", "cv_std", "loo"])


def fig_fusion_summary():
    df = fusion_summary_rows()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    yy = np.arange(len(df))[::-1]
    for y_, (_, r) in zip(yy, df.iterrows()):
        if r.source == "reported":
            ax.plot(r.loo, y_, marker="D", ms=6.5, mfc=SURFACE, mec=MUTED, mew=1.4, ls="none", zorder=3)
            continue
        ax.errorbar(r.cv_mean, y_, xerr=r.cv_std, fmt="o", color=SLOT[0], ms=6.5, elinewidth=1.4,
                    capsize=0, mec=SURFACE, mew=0.8, zorder=3)
        if not np.isnan(r.loo):
            ax.plot(r.loo, y_, marker="D", ms=6.5, mfc=SURFACE, mec=SLOT[1], mew=1.4, ls="none", zorder=4)
    ax.axvline(28 / 58, color=INK2, lw=0.8, zorder=1)
    ax.text(28 / 58 + 0.005, yy[-1] - 0.45, "majority class 48.3 %", color=INK2, fontsize=8, va="center")
    for sep in [1.5, 2.5, 6.5]:   # paper | baseline | A | B
        ax.axhline(yy[0] - sep, color=GRID, lw=0.8)
    ax.set_yticks(yy, df.method)
    ax.set_xlim(0.40, 1.03)
    ax.set_ylim(yy[-1] - 0.8, yy[0] + 0.6)
    ax.set_xlabel("Accuracy on S1 (3 classes: HQ / AQ / LQ)")
    ax.grid(True, axis="x")
    ax.tick_params(length=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=SLOT[0], ls="-", lw=1.4, ms=6.5, mec=SURFACE, label="ours, 10×5-fold CV (mean ± SD)"),
         Line2D([], [], marker="D", color=SLOT[1], mfc=SURFACE, mew=1.4, ls="none", ms=6.5, label="ours, leave-one-out"),
         Line2D([], [], marker="D", color=MUTED, mfc=SURFACE, mew=1.4, ls="none", ms=6.5,
                label="reported (different samples & classes)")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=8, handletextpad=0.3,
              columnspacing=1.2)
    fig.tight_layout()
    _save(fig, "fig3_fusion_summary")


# ---------------------------------------------------------------- Fig. 5 -----------
def fig_cqi():
    from src.data.loaders import S3_G1, load_s3
    df = load_s3()
    t1, t2 = df["Total.Cup.Points"].quantile([1 / 3, 2 / 3]).values
    tp = df["Total.Cup.Points"].values
    q = np.where(tp >= t2, "HQ", np.where(tp >= t1, "AQ", "LQ"))
    col = df["Country.of.Origin"] == "Colombia"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    bins = np.arange(74, 91.01, 0.5)
    ax.hist(tp, bins=bins, color="#c3c2b7", label=f"all reviews (n={len(tp)})", rwidth=0.85)
    ax.hist(tp[col.values], bins=bins, color=SLOT[0], label=f"Colombia (n={int(col.sum())})", rwidth=0.85)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    for t, ha, dx in ((t1, "right", -0.15), (t2, "left", 0.15)):
        ax.axvline(t, color=INK2, lw=0.8)
        ax.text(t + dx, top * 1.07, f"{t:.2f}", color=INK2, fontsize=8, ha=ha, va="center")
    ax.set_xlabel("Total Cup Points")
    ax.set_ylabel("Reviews")
    ax.set_title("Tertile cut-offs define LQ | AQ | HQ", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.tick_params(length=0)
    ax.grid(True, axis="y")
    ax = axes[1]
    prof = pd.DataFrame(df[S3_G1].values, columns=S3_G1).assign(q=q).groupby("q").mean().loc[S1_CLASSES]
    x = np.arange(len(S3_G1))
    for c in S1_CLASSES:
        ax.plot(x, prof.loc[c].values, color=CLS[c], marker=MARK[c], ms=5.5, mec=SURFACE, mew=0.8,
                label=f"{c} (n={int((q == c).sum())})")
    ax.set_xticks(x, [s.replace(".", " ") for s in S3_G1], rotation=40, ha="right")
    ax.set_ylabel("Mean attribute score")
    ax.set_title("Cupping profile per quality class", loc="left")
    ax.legend(loc="center left")
    ax.grid(True, axis="y")
    ax.tick_params(length=0)
    fig.tight_layout()
    _save(fig, "fig4_cqi_quality_grades")


# ---------------------------------------------------------------- Fig. 6 -----------
def expD_gain_rows():
    """Paired (same seed) D1 - D0 gain per setting for the MLP family, vs the paper's reported gain."""
    from src.experiments.expD_smellnet import best_lr, load_runs
    df = load_runs()
    df = df[df.gcms.isin(["spec", "none"])]
    paper = pd.read_csv(ROOT / "docs" / "smellnet_table2_reported.csv", comment="#")
    rows = []
    for (m, w, p), g in df.groupby(["model", "w", "p"]):
        d0 = g[(g.system == "D0") & np.isclose(g.lr, best_lr(df, m, w, p, "D0"))].set_index("seed").test_acc1
        d1 = g[(g.system == "D1") & np.isclose(g.lr, best_lr(df, m, w, p, "D1"))].set_index("seed").test_acc1
        diff = (d1 - d0).dropna() * 100
        pp = paper[(paper.model == m) & (paper.w == w) & (paper.p == p)].set_index("system").paper_acc1
        rows.append({"model": m, "w": w, "p": p, "n_seeds": len(diff), "gain_mean": diff.mean(),
                     "gain_std": diff.std(ddof=1), "d0": d0.mean() * 100, "d1": d1.mean() * 100,
                     "paper_d0": pp.get("D0", np.nan), "paper_d1": pp.get("D1", np.nan),
                     "paper_gain": pp.get("D1", np.nan) - pp.get("D0", np.nan)})
    return pd.DataFrame(rows)


def fig_expD():
    df = expD_gain_rows()
    df = df[df.model == "mlp"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(df))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.errorbar(x - 0.08, df.gain_mean, yerr=df.gain_std, fmt="o", color=SLOT[0], ms=6.5, elinewidth=1.4,
                capsize=0, mec=SURFACE, mew=0.8, label="ours: D1 − D0, paired over 3 seeds (mean ± SD)", zorder=3)
    ax.plot(x + 0.08, df.paper_gain, "D", ms=6.5, mfc=SURFACE, mec=SLOT[1], mew=1.4, ls="none",
            label="SmellNet paper, Table 2 (reported, single seed)", zorder=3)
    ax.set_xticks(x, [f"w={w}, p={p}" for w, p in zip(df.w, df.p)])
    ax.set_ylabel("Accuracy gain from GC-MS (points)")
    ax.set_title("Experiment D, MLP: contrastive GC-MS training (D1) vs sensor-only (D0)", loc="left")
    ax.grid(True, axis="y")
    ax.tick_params(length=0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in sorted(l, key=lambda t: not t.startswith("ours"))]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=8)
    ax.set_ylim(-2.5, 9.0)
    fig.tight_layout()
    _save(fig, "fig5_expD_mlp_gcms_gain")


# ---------------------------------------------------------------- Fig. 4 -----------
ABL_SERIES = [("A1a|LR", "low: curves → PCA → LR"), ("A2|SVM", "mid: features → SVM"),
              ("A3|LR|stacking", "high: per-sensor LR → stacking")]


def _abl_pivot(name):
    d = pd.read_csv(R / "ablations" / f"{name}.csv")
    return d.pivot_table(index="setting", columns="variant", values="accuracy_mean")


def _dot_rows(ax, piv, rows, labels, series, ref=None, delta=False):
    yy = np.arange(len(rows))[::-1]
    off = np.linspace(-0.22, 0.22, len(series))
    for k, (v, lab) in enumerate(series):
        if v not in piv:
            continue
        vals = piv.loc[rows, v].values * 100
        if delta:
            vals = vals - piv.loc["full", v] * 100
        ax.plot(vals, yy + off[k], ls="none", marker=["o", "s", "^"][k], color=SLOT[k], ms=5.5,
                mec=SURFACE, mew=0.7, label=lab, zorder=3)
    ax.set_yticks(yy, labels)
    ax.grid(True, axis="x")
    ax.tick_params(length=0)
    if delta:
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)


def fig_ablations():
    """S1 ablations AB1-AB4 for the reference model of each fusion level (5x5 CV)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    # (a) AB1 leave-one-sensor-out
    p1 = _abl_pivot("AB1_sensors")
    rows = [f"without_{s}" for s in S1_SENSORS]
    _dot_rows(axes[0, 0], p1, rows, S1_SENSORS, ABL_SERIES, delta=True)
    axes[0, 0].set_title("a) AB1: remove one sensor (Δ accuracy vs all 8, points)", loc="left")
    axes[0, 0].set_xlabel("Δ accuracy (points)")
    # (b) AB2 duration
    p2 = _abl_pivot("AB2_duration")
    durs = [("first_30s", 30), ("first_60s", 60), ("first_120s", 120), ("first_180s", 180), ("full", 300)]
    ax = axes[0, 1]
    for k, (v, lab) in enumerate(ABL_SERIES):
        if v in p2:
            ax.plot([d for _, d in durs], [p2.loc[s, v] * 100 for s, _ in durs], marker=["o", "s", "^"][k],
                    color=SLOT[k], ms=5.5, mec=SURFACE, mew=0.7, label=lab)
    ax.set_xticks([d for _, d in durs])
    ax.set_xlim(15, 310)
    ax.axvline(33, color=INK2, lw=0.8, zorder=1)   # measured response onset: 32-34 s in all 58 recordings
    ax.text(37, 91.6, "odour arrives at 32–34 s", color=INK2, fontsize=7.5, va="top", ha="left")
    ax.set_xlabel("Signal duration used (s from start of recording)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("b) AB2: signal duration", loc="left")
    ax.grid(True, axis="y")
    ax.tick_params(length=0)
    # (c) AB3 representation
    p3 = _abl_pivot("AB3_representation")
    rows3 = ["full", "rep_R", "rep_RR0", "rep_GG0", "rep_z", "G_diff25", "RR0_diff25"]
    lab3 = ["G = 1/R (default)", "R (raw)", "R/R0 (baseline-normalised)", "G/G0 (baseline-normalised)",
            "z-score per recording", "G, differenced p=25", "R/R0, differenced p=25"]
    _dot_rows(axes[1, 0], p3, [r for r in rows3 if r in p3.index], [l for r, l in zip(rows3, lab3) if r in p3.index],
              ABL_SERIES)
    axes[1, 0].set_title("c) AB3: signal representation (accuracy, %)", loc="left")
    axes[1, 0].set_xlabel("Accuracy (%)")
    # (d) AB4 feature groups (mid and high level only)
    p4 = _abl_pivot("AB4_features")
    rows4 = [r for r in p4.index if r.startswith("no_")]
    _dot_rows(axes[1, 1], p4, rows4, [r.replace("no_", "without ") for r in rows4],
              ABL_SERIES, delta=True)   # A1a has no hand-crafted features; skipped inside, colours stay fixed
    axes[1, 1].set_title("d) AB4: remove one feature type (Δ vs all 15, points)", loc="left")
    axes[1, 1].set_xlabel("Δ accuracy (points)")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, "fig6_s1_ablations")


FIGS = {"curves": fig_s1_curves, "pca": fig_pca, "fusion": fig_fusion_summary, "cqi": fig_cqi, "expD": fig_expD,
        "ablations": fig_ablations}

if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
