"""Experiment A - multi-sensor fusion within S1 (RQ2) and S1 ablations (RQ5).

Each of the 8 sensors is treated as a source.
  A1 low-level : (a) flatten 8 x T curves -> scaler -> PCA(10) -> classifier
                 (b) 8-channel 1D-CNN on the curves (optionally sliding-window augmentation)
  A2 mid-level : 15 features per sensor -> concatenate (8 x 15) -> classifier
  A3 high-level: one classifier per sensor (its 15 features) -> vote / mean / weighted / stacking
Default signal representation: conductance G = 1/R over the full 300 s.

Run: python -m src.experiments.expA_s1_fusion [main|cnn|ablations|all]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.data.loaders import S1_CLASSES, S1_SENSORS, load_s1
from src.evaluation.cv import (RESULTS, loo_splits, mcnemar, per_repeat_scores, repeated_skf_splits,
                               run_cv, wilcoxon_paired)
from src.features.enose import FEATURE_GROUPS, FEATURE_NAMES, sensor_features, temporal_diff, to_representation
from src.fusion.fusion import COMBINERS, HighLevelFusion
from src.models.torch_models import TorchSeqClassifier
from src.models.zoo import S1_CLFS, scaled
from src.utils.plotting import bar_with_err

ABL = RESULTS / "ablations"
K = 3
LABELS = [0, 1, 2]


# ------------------------------------------------------------------ inputs ---------
def make_inputs(X, rep="G", dur=300, sensors=None, diff_p=0, feat_keep=None):
    """Return (curves (N,T,C), per-sensor features (N,C,F)) for a configuration."""
    sensors = list(range(8)) if sensors is None else list(sensors)
    S = to_representation(X, rep)[:, :dur, sensors]
    S = temporal_diff(S, diff_p)
    Fe = sensor_features(S, baseline_s=min(20, S.shape[1] // 3))
    if feat_keep is not None:
        Fe = Fe[:, :, [FEATURE_NAMES.index(f) for f in feat_keep]]
    return S, Fe


class Inputs(tuple):
    """(curves, features) pair that run_cv can index by rows."""


def fit_predict_factory(levels=("A1a", "A2", "A3"), clfs=S1_CLFS, a3_bases=("LR", "SVM")):
    def fit_predict(Xtr, ytr, Xte, seed):
        Str, Ftr = Xtr
        Ste, Fte = Xte
        out = {}
        n_comp = min(10, len(ytr) - 1)
        if "A1a" in levels:
            for c in clfs:
                m = scaled(c, seed, pca=n_comp).fit(Str.reshape(len(Str), -1), ytr)
                out[f"A1a|{c}"] = m.predict_proba(Ste.reshape(len(Ste), -1))
        if "A2" in levels:
            for c in clfs:
                m = scaled(c, seed).fit(Ftr.reshape(len(Ftr), -1), ytr)
                out[f"A2|{c}"] = m.predict_proba(Fte.reshape(len(Fte), -1))
        if "A3" in levels:
            C = Ftr.shape[1]
            for b in a3_bases:
                srcs = [(f"s{i}", (lambda Z, i=i: Z[:, i, :]), scaled(b, seed)) for i in range(C)]
                hl = HighLevelFusion(srcs, K, inner_splits=5, seed=seed).fit(Ftr, ytr)
                for comb, P in hl.predict_proba_all(Fte).items():
                    out[f"A3|{b}|{comb}"] = P
        return out
    return fit_predict


def cnn_fit_predict(window=None, epochs=150):
    def fp(Xtr, ytr, Xte, seed):
        Str, _ = Xtr
        Ste, _ = Xte
        m = TorchSeqClassifier("cnn", K, epochs=epochs, window=window, stride=10, seed=seed)
        m.fit(Str, ytr)
        return {f"A1b|CNN{'+win' if window else ''}": m.predict_proba(Ste)}
    return fp


# ------------------------------------------------------------------ main -----------
def run_main(X, y, ids):
    S, Fe = make_inputs(X)
    splits = loo_splits(y) + repeated_skf_splits(y, 5, 10, seed=0)
    res = run_cv("expA_main", {"rep": "G", "dur": 300}, Inputs((S, Fe)), y,
                 fit_predict_factory(), splits, S1_CLASSES, ids)
    return res


def run_cnn(X, y, ids):
    S, Fe = make_inputs(X)
    splits = loo_splits(y) + repeated_skf_splits(y, 5, 10, seed=0)
    r1 = run_cv("expA_cnn", {"rep": "G", "epochs": 150}, Inputs((S, Fe)), y, cnn_fit_predict(None, 150),
                splits, S1_CLASSES, ids)
    r2 = run_cv("expA_cnn_win", {"rep": "G", "window": 250, "stride": 10, "epochs": 60},
                Inputs((S, Fe)), y, cnn_fit_predict(250, 60), splits, S1_CLASSES, ids)
    return r1, r2


def collect_main():
    """Merge main + CNN summaries into results/expA_fusion_levels.csv, add significance."""
    parts = []
    preds = []
    for e in ["expA_main", "expA_cnn", "expA_cnn_win"]:
        d = RESULTS / "runs" / e
        if (d / "summary.csv").exists():
            parts.append(pd.read_csv(d / "summary.csv"))
            preds.append(pd.read_csv(d / "predictions.csv.gz"))
    summ = pd.concat(parts, ignore_index=True)
    summ["level"] = summ.variant.str.split("|").str[0].map(
        {"A1a": "low", "A1b": "low", "A2": "mid", "A3": "high"})
    summ = summ[["exp", "level", "variant", "scheme", "n_repeats", "accuracy_mean", "accuracy_std",
                 "balanced_accuracy_mean", "balanced_accuracy_std", "macro_f1_mean", "macro_f1_std"]]
    summ.to_csv(RESULTS / "expA_fusion_levels.csv", index=False)
    pred = pd.concat(preds, ignore_index=True)

    # Best per level (by repeated 5-fold mean accuracy), then paired tests between levels.
    r = summ[summ.scheme == "rskf5"]
    best = r.loc[r.groupby("level").accuracy_mean.idxmax()].set_index("level")
    wide = per_repeat_scores(pred[pred.scheme == "rskf5"], LABELS)
    loo = pred[pred.scheme == "loo"]
    rows = []
    lv = ["low", "mid", "high"]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = best.loc[lv[i]].variant, best.loc[lv[j]].variant
            _, p_w = wilcoxon_paired(wide[a].values, wide[b].values)
            la = loo[loo.variant == a].sort_values("sample")
            lb = loo[loo.variant == b].sort_values("sample")
            n01, n10, p_m = mcnemar(la.y_true.values, la.y_pred.values, lb.y_pred.values)
            rows.append({"A": a, "B": b, "acc_A_rskf": wide[a].mean(), "acc_B_rskf": wide[b].mean(),
                         "wilcoxon_p_rskf": p_w, "loo_A_only_correct": n01,
                         "loo_B_only_correct": n10, "mcnemar_p_loo": p_m})
    sig = pd.DataFrame(rows)
    sig.to_csv(RESULTS / "expA_significance.csv", index=False)
    print(best[["variant", "accuracy_mean", "accuracy_std", "macro_f1_mean"]])
    print(sig.to_string(index=False))

    # Figure: every variant, rskf mean +- std, coloured by level
    fig_df = r.sort_values(["level", "accuracy_mean"])
    bar_with_err(fig_df, "variant", "accuracy_mean", "accuracy_std",
                 RESULTS / "figures" / "expA_fusion_levels.png", hue="level",
                 title="Experiment A: fusion level comparison (10x5-fold CV)",
                 ref=28 / 58, rotate=60)
    return summ


# ------------------------------------------------------------------ ablations -----
REF = dict(levels=("A1a", "A2", "A3"), clfs=("LR", "SVM"), a3_bases=("LR",))
REF_VARIANTS = ["A1a|LR", "A1a|SVM", "A2|LR", "A2|SVM", "A3|LR|mean", "A3|LR|stacking"]


def _abl(name, X, y, ids, settings, n_repeats=5, **ref):
    """Run the reference variants for every setting; settings = {label: make_inputs kwargs}."""
    kw = REF | ref
    rows = []
    for label, cfg in settings.items():
        S, Fe = make_inputs(X, **cfg)
        res = run_cv(f"{name}_{label}", {"ablation": name, "setting": label, **{k: str(v) for k, v in cfg.items()}},
                     Inputs((S, Fe)), y, fit_predict_factory(**kw),
                     repeated_skf_splits(y, 5, n_repeats, seed=0), S1_CLASSES, ids, verbose=False)
        s = res.summary
        s["setting"] = label
        rows.append(s)
        print(name, label, s.set_index("variant").accuracy_mean.round(3).to_dict())
    df = pd.concat(rows, ignore_index=True)
    df = df[df.variant.isin(REF_VARIANTS) | ~df.variant.str.startswith("A3")]
    full = df[df.setting == "full"].set_index("variant").accuracy_mean if "full" in df.setting.values else None
    if full is not None:
        df["delta_vs_full"] = df.accuracy_mean - df.variant.map(full)
    cols = ["setting", "variant", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean",
            "macro_f1_mean", "macro_f1_std"] + (["delta_vs_full"] if full is not None else [])
    df[cols].to_csv(ABL / f"{name}.csv", index=False)
    return df


def run_ablations(X, y, ids, which=None):
    ABL.mkdir(parents=True, exist_ok=True)
    which = which or ["AB1", "AB2", "AB3", "AB4", "AB10"]
    if "AB1" in which:  # leave-one-sensor-out and single-sensor
        st = {"full": {}}
        st |= {f"without_{s}": {"sensors": [j for j in range(8) if j != i]} for i, s in enumerate(S1_SENSORS)}
        st |= {f"only_{s}": {"sensors": [i]} for i, s in enumerate(S1_SENSORS)}
        df = _abl("AB1_sensors", X, y, ids, st)
        _plot_abl(df, "AB1_sensors", "AB1: leave-one-sensor-out / single sensor")
    if "AB2" in which:  # signal duration
        st = {("full" if d == 300 else f"first_{d}s"): {"dur": d} for d in [30, 60, 120, 180, 300]}
        df = _abl("AB2_duration", X, y, ids, st)
        _plot_abl(df, "AB2_duration", "AB2: signal duration")
    if "AB3" in which:  # signal representation and differencing
        st = {"full": {}}
        st |= {f"rep_{r}": {"rep": r} for r in ["R", "RR0", "GG0", "z"]}
        st |= {"G_diff25": {"diff_p": 25}, "RR0_diff25": {"rep": "RR0", "diff_p": 25}}
        df = _abl("AB3_representation", X, y, ids, st)
        _plot_abl(df, "AB3_representation", "AB3: signal representation")
    if "AB4" in which:  # remove each feature type (affects A2 and A3 only)
        st = {"full": {}}
        for g, names in FEATURE_GROUPS.items():
            st[f"no_{g}"] = {"feat_keep": [f for f in FEATURE_NAMES if f not in names]}
        df = _abl("AB4_features", X, y, ids, st, levels=("A2", "A3"))
        _plot_abl(df, "AB4_features", "AB4: feature-type removal")
    if "AB10" in which:  # sliding-window augmentation for the CNN (from expA_cnn runs)
        d = pd.concat([pd.read_csv(RESULTS / "runs" / e / "summary.csv") for e in ["expA_cnn", "expA_cnn_win"]])
        d.to_csv(ABL / "AB10_window_augmentation.csv", index=False)


def summarise_derived():
    """AB5 (combiners), AB6 (classifier family) and AB7 (LOO vs 5-fold) from the main run."""
    s = pd.read_csv(RESULTS / "expA_fusion_levels.csv")
    a3 = s[s.variant.str.startswith("A3")].copy()
    a3["base"] = a3.variant.str.split("|").str[1]
    a3["combiner"] = a3.variant.str.split("|").str[2]
    a3[["base", "combiner", "scheme", "accuracy_mean", "accuracy_std", "macro_f1_mean"]].to_csv(
        ABL / "AB5_combiners_S1.csv", index=False)
    bar_with_err(a3[a3.scheme == "rskf5"], "combiner", "accuracy_mean", "accuracy_std",
                 RESULTS / "figures" / "AB5_combiners_S1.png", hue="base",
                 title="AB5 (S1): decision-level combiners", ref=28 / 58)
    fam = s[s.variant.str.match(r"A1a|A2") & (s.variant.str.count(r"\|") == 1)].copy()
    fam["level"] = fam.variant.str.split("|").str[0]
    fam["classifier"] = fam.variant.str.split("|").str[1]
    fam[["level", "classifier", "scheme", "accuracy_mean", "accuracy_std", "macro_f1_mean"]].to_csv(
        ABL / "AB6_classifier_family_S1.csv", index=False)
    bar_with_err(fam[fam.scheme == "rskf5"], "classifier", "accuracy_mean", "accuracy_std",
                 RESULTS / "figures" / "AB6_classifier_family_S1.png", hue="level",
                 title="AB6 (S1): classifier family", ref=28 / 58)
    base = pd.read_csv(RESULTS / "baseline.csv")
    both = pd.concat([base.assign(source="baseline"), s.assign(source="expA")])
    piv = both.pivot_table(index=["source", "variant"], columns="scheme", values="accuracy_mean")
    piv["loo_minus_rskf"] = piv["loo"] - piv["rskf5"]
    piv.reset_index().to_csv(ABL / "AB7_validation_scheme.csv", index=False)
    print("AB7 mean LOO - 5fold gap:", piv.loo_minus_rskf.describe().round(3).to_dict())


def _plot_abl(df, name, title):
    sub = df[df.variant.isin(REF_VARIANTS)]
    bar_with_err(sub, "setting", "accuracy_mean", "accuracy_std", RESULTS / "figures" / f"{name}.png",
                 hue="variant", title=title, ref=28 / 58, rotate=60)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    X, y, ids = load_s1()
    if part in ("main", "all"):
        run_main(X, y, ids)
    if part in ("cnn", "all"):
        run_cnn(X, y, ids)
    if part in ("collect", "main", "cnn", "all"):
        collect_main()
    if part in ("ablations", "all"):
        run_ablations(X, y, ids, sys.argv[2:] or None)
    if part in ("derived", "ablations", "all"):
        summarise_derived()
