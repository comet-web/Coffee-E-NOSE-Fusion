"""Experiment C - fusion of attribute groups within S3 / CQI (RQ4), plus AB5/AB6/AB9 on S3.

Target: 3 quality classes from Total.Cup.Points tertiles (LQ < t1 <= AQ < t2 <= HQ),
mirroring S1's HQ/AQ/LQ. Thresholds are a fixed label definition computed once on the
whole table (documented); everything learned from features is fitted in-fold.

Sources:
  G1 sensory cupping scores   -> leakage: Total.Cup.Points = sum(G1) (verified);
                                 reported ONLY as an upper bound.
  G2 physical / bean data     (Moisture, defects, Quakers, Color, Processing.Method, Species)
  G3 origin data              (Country.of.Origin, Region, altitude)
Main study fuses G2 + G3:
  C1 low-level : concatenate encoded G2 + G3 columns -> one model
  C2 mid-level : per-group encoding -> per-group PCA (in-fold) -> concatenate -> one model
  C3 high-level: one model per group -> vote / mean / weighted / stacking
Run: python -m src.experiments.expC_cqi
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.loaders import S3_G1, S3_G2_CAT, S3_G2_NUM, S3_G3_CAT, S3_G3_NUM, load_s3
from src.evaluation.cv import RESULTS, per_repeat_scores, repeated_skf_splits, run_cv, wilcoxon_paired
from src.fusion.fusion import HighLevelFusion
from src.models.zoo import make_clf
from src.utils.plotting import bar_with_err, plt, _save

warnings.filterwarnings("ignore", message=".*unknown categories.*")
warnings.filterwarnings("ignore", category=FutureWarning)
CLASSES = ["HQ", "AQ", "LQ"]
MODELS = ["LR", "RF", "GB", "SVM", "MLP"]
GROUPS = {"G2": (S3_G2_NUM, S3_G2_CAT), "G3": (S3_G3_NUM, S3_G3_CAT), "G1": (S3_G1, [])}


def make_target(df):
    t1, t2 = df["Total.Cup.Points"].quantile([1 / 3, 2 / 3]).values
    tp = df["Total.Cup.Points"].values
    y = np.where(tp >= t2, 0, np.where(tp >= t1, 1, 2))  # 0=HQ, 1=AQ, 2=LQ
    return y, (float(t1), float(t2))


def encoder(groups):
    """ColumnTransformer for the listed groups (imputation + encoding fitted in-fold)."""
    tr = []
    for g in groups:
        num, cat = GROUPS[g]
        if num:
            tr.append((f"{g}_num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                 StandardScaler()), num))
        if cat:
            tr.append((f"{g}_cat", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)), cat))
    return ColumnTransformer(tr)


def mid_level(groups, n_comp={"G1": 5, "G2": 5, "G3": 10}):
    """Per-group target-free embedding (PCA on the group's encoded columns), concatenated."""
    parts = []
    for g in groups:
        num, cat = GROUPS[g]
        parts.append((g, make_pipeline(encoder([g]), PCA(n_components=n_comp[g], random_state=0)), num + cat))
    return ColumnTransformer(parts)


def low(groups, model, seed):
    return make_pipeline(encoder(groups), make_clf(model, seed))


def mid(groups, model, seed):
    return make_pipeline(mid_level(groups), StandardScaler(), make_clf(model, seed))


def fit_predict_factory(models=MODELS, main_groups=("G2", "G3"), extras=True):
    def fp(Xtr, ytr, Xte, seed):
        out = {}
        for m in models:
            out[f"C1|{m}"] = low(main_groups, m, seed).fit(Xtr, ytr).predict_proba(Xte)
            out[f"C2|{m}"] = mid(main_groups, m, seed).fit(Xtr, ytr).predict_proba(Xte)
            srcs = [(g, (lambda Z: Z), low([g], m, seed)) for g in main_groups]
            hl = HighLevelFusion(srcs, 3, inner_splits=5, seed=seed).fit(Xtr, ytr)
            for comb, P in hl.predict_proba_all(Xte).items():
                out[f"C3|{m}|{comb}"] = P
            if extras:  # AB9 single-group models and the leaky G1 upper bound
                for g in main_groups:
                    out[f"only_{g}|{m}"] = low([g], m, seed).fit(Xtr, ytr).predict_proba(Xte)
                if m in ("LR", "RF"):
                    out[f"UPPER_G1|{m}"] = low(["G1"], m, seed).fit(Xtr, ytr).predict_proba(Xte)
                    out[f"UPPER_G1G2G3|{m}"] = low(["G1", "G2", "G3"], m, seed).fit(Xtr, ytr).predict_proba(Xte)
        return out
    return fp


def describe(df, y, thr):
    """Descriptive use of S3: what HQ / AQ / LQ mean in cupping terms; Colombian subset."""
    d = df.assign(quality=np.array(CLASSES)[y])
    prof = d.groupby("quality")[S3_G1 + ["Total.Cup.Points"] + S3_G2_NUM + S3_G3_NUM].mean().loc[CLASSES]
    prof["n"] = d.quality.value_counts().loc[CLASSES]
    col = d[d["Country.of.Origin"] == "Colombia"]
    prof_col = col.groupby("quality")[S3_G1 + ["Total.Cup.Points"]].mean().reindex(CLASSES)
    prof_col["n"] = col.quality.value_counts().reindex(CLASSES)
    prof.to_csv(RESULTS / "expC_quality_profile_all.csv")
    prof_col.to_csv(RESULTS / "expC_quality_profile_colombia.csv")
    country = (d.groupby("Country.of.Origin")["Total.Cup.Points"].agg(["count", "mean", "median"])
               .query("count >= 10").sort_values("mean", ascending=False))
    country.to_csv(RESULTS / "expC_country_scores.csv")
    corr = d[S3_G1 + ["Total.Cup.Points"] + S3_G2_NUM + S3_G3_NUM].corr()
    corr.to_csv(RESULTS / "expC_correlations.csv")

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    bins = np.arange(59.5, 91.5, 0.5)
    ax[0].hist(d["Total.Cup.Points"], bins=bins, color="#9e9e9e", label="all")
    ax[0].hist(col["Total.Cup.Points"], bins=bins, color="#b2182b", alpha=0.8, label="Colombia")
    for t in thr:
        ax[0].axvline(t, color="k", ls="--", lw=0.8)
    ax[0].set_xlim(74, 91)
    ax[0].set_xlabel("Total Cup Points")
    ax[0].set_ylabel("Reviews")
    ax[0].legend(frameon=False)
    ax[0].set_title(f"Tertile cut-offs {thr[0]:.2f} / {thr[1]:.2f}")
    z = prof[S3_G1].T
    for c, colr in zip(CLASSES, ["#1b7837", "#d9a400", "#b2182b"]):
        ax[1].plot(z.index, z[c], marker="o", label=c, color=colr)
    ax[1].set_ylabel("Mean score")
    ax[1].set_title("Cupping profile per S3 quality class")
    plt.setp(ax[1].get_xticklabels(), rotation=60, ha="right")
    ax[1].legend(frameon=False)
    _save(fig, RESULTS / "figures" / "expC_quality_grades.png")
    return prof, prof_col


def main():
    df = load_s3()
    y, thr = make_target(df)
    print("thresholds", thr, "class counts", np.bincount(y))
    describe(df, y, thr)
    cols = S3_G1 + S3_G2_NUM + S3_G2_CAT + S3_G3_NUM + S3_G3_CAT
    X = df[cols].copy()
    for c in S3_G2_CAT + S3_G3_CAT:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    # 5x5 (not 10x5): N = 1338 gives low fold-to-fold variance and the nested SVM/GB runs are costly.
    splits = repeated_skf_splits(y, 5, 5, seed=0)
    res = run_cv("expC_main", {"target": "Total.Cup.Points tertiles", "t1": thr[0], "t2": thr[1],
                               "groups_main": "G2+G3"}, X, y, fit_predict_factory(), splits, CLASSES,
                 df.index.values)
    s = res.summary.copy()
    s["level"] = s.variant.str.split("|").str[0]
    s.to_csv(RESULTS / "expC_cqi_fusion.csv", index=False)

    # AB5 (combiners), AB6 (family), AB9 (group removal) tables
    abl = RESULTS / "ablations"
    c3 = s[s.level == "C3"].assign(model=lambda d: d.variant.str.split("|").str[1],
                                   combiner=lambda d: d.variant.str.split("|").str[2])
    c3.to_csv(abl / "AB5_combiners_S3.csv", index=False)
    bar_with_err(c3, "combiner", "accuracy_mean", "accuracy_std", RESULTS / "figures" / "AB5_combiners_S3.png",
                 hue="model", title="AB5 (S3): decision-level combiners", ref=1 / 3)
    fam = s[s.level.isin(["C1", "C2"])].assign(model=lambda d: d.variant.str.split("|").str[1])
    fam.to_csv(abl / "AB6_classifier_family_S3.csv", index=False)
    bar_with_err(fam, "model", "accuracy_mean", "accuracy_std", RESULTS / "figures" / "AB6_classifier_family_S3.png",
                 hue="level", title="AB6 (S3): classifier family", ref=1 / 3)
    ab9 = s[s.level.isin(["C1", "only_G2", "only_G3"])].assign(model=lambda d: d.variant.str.split("|").str[1])
    ab9["groups"] = ab9.level.map({"C1": "G2+G3", "only_G2": "G2 only (no G3)", "only_G3": "G3 only (no G2)"})
    ab9.to_csv(abl / "AB9_group_removal_S3.csv", index=False)
    bar_with_err(ab9, "model", "accuracy_mean", "accuracy_std", RESULTS / "figures" / "AB9_group_removal_S3.png",
                 hue="groups", title="AB9 (S3): remove a source group", ref=1 / 3)

    # Paired tests: best per level
    wide = per_repeat_scores(res.predictions, [0, 1, 2])
    best = {lv: s[s.level == lv].sort_values("accuracy_mean").iloc[-1].variant for lv in ["C1", "C2", "C3"]}
    rows = []
    for a, b in [("C1", "C2"), ("C1", "C3"), ("C2", "C3")]:
        _, p = wilcoxon_paired(wide[best[a]], wide[best[b]])
        rows.append({"A": best[a], "B": best[b], "acc_A": wide[best[a]].mean(), "acc_B": wide[best[b]].mean(),
                     "wilcoxon_p": p})
    pd.DataFrame(rows).to_csv(RESULTS / "expC_significance.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
