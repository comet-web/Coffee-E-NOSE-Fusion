"""Generate every number table used in the article from logged results (no hand-typed values).

Writes article/tables.md (Markdown tables) and results/s1_confusion_summary.csv.
Run: python -m src.utils.article_tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
A = R / "ablations"


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}"


def pm(m, s, d=1):
    return "n/a" if pd.isna(m) else (f"{100 * m:.{d}f} ± {100 * s:.{d}f}" if not pd.isna(s) else f"{100 * m:.{d}f}")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():  # model names such as A1a|LR contain '|', which must be escaped in tables
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(out)


def get(df, variant, scheme, col):
    r = df[(df.variant == variant) & (df.scheme == scheme)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ------------------------------------------------------------------ tables --------
def t_baseline():
    b = pd.read_csv(R / "baseline.csv")
    names = {"none": "none", "center": "mean-centring", "std": "autoscaling"}
    rows = []
    for v in b.variant.unique():
        feat, sc, solver = v.split("|")
        rows.append({"Feature": "ΔG = Gmax − Gmin" if feat == "dG-abs" else "ΔG / Gmin",
                     "Input scaling": names[sc], "Solver": solver.upper() if solver == "sgd" else "Adam",
                     "LOO acc. % (3 inits)": pm(get(b, v, "loo", "accuracy_mean"), get(b, v, "loo", "accuracy_std")),
                     "10×5-fold acc. %": pm(get(b, v, "rskf5", "accuracy_mean"), get(b, v, "rskf5", "accuracy_std")),
                     "Macro-F1 (5-fold)": f"{get(b, v, 'rskf5', 'macro_f1_mean'):.3f}"})
    pca = pd.read_csv(R / "baseline_pca.csv")
    pca_t = pd.DataFrame({"Feature": pca.feature.map({"dG-abs": "ΔG = Gmax − Gmin", "dG-rel": "ΔG / Gmin"}),
                          "Scaling": pca.scaling.map({"center": "mean-centring", "std": "autoscaling"}),
                          "PC1 %": (100 * pca.PC1).round(2), "PC2 %": (100 * pca.PC2).round(2),
                          "PC1+PC2 %": (100 * pca["PC1+PC2"]).round(2)})
    return md(pd.DataFrame(rows)), md(pca_t)


def t_main_comparison():
    base = pd.read_csv(R / "baseline.csv")
    a = pd.read_csv(R / "expA_fusion_levels.csv")
    b = pd.read_csv(R / "expB_s1_s2_fusion.csv")
    rows = [
        ["Rodríguez et al. 2010, analysis 1", "16 meas.", "4", "LOO", "100", "n/a", "reported"],
        ["Rodríguez et al. 2010, analysis 2", "35 meas.", "9", "LOO", "92.5", "n/a", "reported"],
    ]
    spec = [("Our baseline reproduction (ΔG/Gmin, mean-centring, MLP 10, SGD)", base, "dG-rel|center|sgd", "S1, 58 meas."),
            ("Best low-level fusion (A1a: curves → PCA(10) → LR)", a, "A1a|LR", "S1"),
            ("Low-level fusion, 1D-CNN with sliding windows (A1b)", a, "A1b|CNN+win", "S1"),
            ("Best mid-level fusion (A2: 120 features → SVM)", a, "A2|SVM", "S1"),
            ("Best high-level fusion (A3: per-sensor SVM → stacking)", a, "A3|SVM|stacking", "S1")]
    for name, df, v, data in spec:
        rows.append([name, data, "3", "LOO / 10×5-fold",
                     f"{pct(get(df, v, 'loo', 'accuracy_mean'))} / {pm(get(df, v, 'rskf5', 'accuracy_mean'), get(df, v, 'rskf5', 'accuracy_std'))}",
                     f"{get(df, v, 'loo', 'macro_f1_mean'):.3f} / {get(df, v, 'rskf5', 'macro_f1_mean'):.3f}", "ours"])
    bb = b[b.variant.str.startswith(("B2", "B3", "B1"))].sort_values("accuracy_mean", ascending=False).iloc[0]
    rows.append([f"Best S1+S2 fusion (B: {bb.variant.replace('|', ' ')})", "S1 + S2", "3", "10×5-fold",
                 pm(bb.accuracy_mean, bb.accuracy_std), f"{bb.macro_f1_mean:.3f}", "ours"])
    return md(pd.DataFrame(rows, columns=["Method", "Data", "Classes", "Validation", "Accuracy %",
                                          "Macro-F1", "Source"]))


def t_expA():
    a = pd.read_csv(R / "expA_fusion_levels.csv")
    r = a[a.scheme == "rskf5"].set_index("variant")
    lo = a[a.scheme == "loo"].set_index("variant")
    rows = []
    for c in ["LR", "SVM", "RF", "kNN", "MLP"]:
        row = {"Classifier": c}
        for lvl, name in [("A1a", "Low (A1a) %"), ("A2", "Mid (A2) %")]:
            v = f"{lvl}|{c}"
            row[name] = f"{pm(r.loc[v, 'accuracy_mean'], r.loc[v, 'accuracy_std'])} (LOO {pct(lo.loc[v, 'accuracy_mean'])})"
        rows.append(row)
    fam = pd.DataFrame(rows)
    rows = []
    for base in ["LR", "SVM"]:
        row = {"Per-sensor model": base}
        for comb in ["vote", "mean", "weighted", "stacking"]:
            v = f"A3|{base}|{comb}"
            row[comb] = f"{pm(r.loc[v, 'accuracy_mean'], r.loc[v, 'accuracy_std'])} (LOO {pct(lo.loc[v, 'accuracy_mean'])})"
        rows.append(row)
    a3 = pd.DataFrame(rows)
    cnn = pd.DataFrame([{"Model": v.replace("A1b|", ""),
                         "10×5-fold %": pm(r.loc[v, "accuracy_mean"], r.loc[v, "accuracy_std"]),
                         "LOO %": pct(lo.loc[v, "accuracy_mean"]), "Macro-F1 (5-fold)": f"{r.loc[v, 'macro_f1_mean']:.3f}"}
                        for v in ["A1b|CNN", "A1b|CNN+win"]])
    sig = pd.read_csv(R / "expA_significance.csv")
    sig_t = pd.DataFrame({"Comparison": sig.A + " vs " + sig.B,
                          "Acc. A / B (5-fold) %": [f"{100 * x:.1f} / {100 * y:.1f}" for x, y in zip(sig.acc_A_rskf, sig.acc_B_rskf)],
                          "Wilcoxon p (10 repeats)": sig.wilcoxon_p_rskf.map(lambda p: f"{p:.3f}"),
                          "LOO: only A / only B correct": [f"{x} / {y}" for x, y in zip(sig.loo_A_only_correct, sig.loo_B_only_correct)],
                          "McNemar p (LOO)": sig.mcnemar_p_loo.map(lambda p: f"{p:.3f}")})
    return md(fam), md(a3), md(cnn), md(sig_t)


def confusion_summary():
    P = pd.concat([pd.read_csv(R / "runs" / e / "predictions.csv.gz") for e in ["baseline", "expA_main", "expA_cnn_win"]])
    rows = []
    for v, sch in [("dG-rel|center|sgd", "loo"), ("A1a|LR", "loo"), ("A1b|CNN+win", "loo"), ("A2|SVM", "loo"),
                   ("A3|SVM|stacking", "loo"), ("A1a|LR", "rskf5"), ("A2|SVM", "rskf5"), ("A3|SVM|stacking", "rskf5")]:
        g = P[(P.variant == v) & (P.scheme == sch)]
        if sch == "loo":
            g = g[g.repeat == 0]
        cm = confusion_matrix(g.y_true, g.y_pred, labels=[0, 1, 2])
        rec = cm.diagonal() / cm.sum(1)
        err = int(cm.sum() - np.trace(cm))
        rows.append({"variant": v, "scheme": sch, "cm_rows_true_HQ_AQ_LQ": cm.tolist(),
                     "recall_HQ": rec[0], "recall_AQ": rec[1], "recall_LQ": rec[2], "errors": err,
                     "AQ_LQ_confusions": int(cm[1, 2] + cm[2, 1])})
    df = pd.DataFrame(rows)
    df.to_csv(R / "s1_confusion_summary.csv", index=False)
    t = pd.DataFrame({"Model": df.variant, "Validation": df.scheme.map({"loo": "LOO", "rskf5": "10×5-fold (pooled)"}),
                      "Recall HQ / AQ / LQ": [f"{a:.2f} / {b:.2f} / {c:.2f}" for a, b, c in zip(df.recall_HQ, df.recall_AQ, df.recall_LQ)],
                      "Errors": df.errors, "of which AQ↔LQ": [f"{x} ({x / e:.0%})" for x, e in zip(df.AQ_LQ_confusions, df.errors)]})
    return md(t)


def t_expB():
    b = pd.read_csv(R / "expB_s1_s2_fusion.csv")
    sig = pd.read_csv(R / "expB_significance.csv")
    rows = []
    for v in b.variant:
        s = sig[sig.variant == v]
        rows.append({"Variant": v.replace("|", " · "), "Level": b.loc[b.variant == v, "level"].iloc[0],
                     "Accuracy %": pm(b.loc[b.variant == v, "accuracy_mean"].iloc[0], b.loc[b.variant == v, "accuracy_std"].iloc[0]),
                     "Macro-F1": f"{b.loc[b.variant == v, 'macro_f1_mean'].iloc[0]:.3f}",
                     "Δ vs S1-only (pts)": f"{100 * s['diff'].iloc[0]:+.1f}" if len(s) else "-",
                     "Wilcoxon p": f"{s['wilcoxon_p'].iloc[0]:.3f}" if len(s) else "-"})
    sec = pd.read_csv(R / "expB_s2_secondary_check.csv")
    return md(pd.DataFrame(rows)), md(sec.assign(s2_day6_acc_50way=(100 * sec.s2_day6_acc_50way).round(1))
                                      .rename(columns={"model": "Pooled model", "s2_day6_acc_50way": "S2 day-6 accuracy % (50-way)"}))


def t_expC():
    c = pd.read_csv(R / "expC_cqi_fusion.csv")
    order = ["C1", "C2", "C3", "only_G2", "only_G3", "UPPER_G1", "UPPER_G1G2G3"]
    names = {"C1": "C1 low (G2+G3 concatenated)", "C2": "C2 mid (per-group PCA)", "C3": "C3 high (per-group models)",
             "only_G2": "G2 only (bean/physical)", "only_G3": "G3 only (origin)",
             "UPPER_G1": "UPPER BOUND: G1 sensory (leaky)", "UPPER_G1G2G3": "UPPER BOUND: G1+G2+G3 (leaky)"}
    rows = []
    for lv in order:
        g = c[c.level == lv].sort_values("accuracy_mean", ascending=False).iloc[0]
        rows.append({"Setting": names[lv], "Best model": g.variant.split("|", 1)[1].replace("|", " · "),
                     "Accuracy %": pm(g.accuracy_mean, g.accuracy_std),
                     "Balanced acc. %": pct(g.balanced_accuracy_mean), "Macro-F1": f"{g.macro_f1_mean:.3f}"})
    fam = []
    for m in ["LR", "SVM", "RF", "GB", "MLP"]:
        row = {"Model": m}
        for lv in ["C1", "C2"]:
            g = c[c.variant == f"{lv}|{m}"].iloc[0]
            row[lv] = pm(g.accuracy_mean, g.accuracy_std)
        g = c[c.variant.str.startswith(f"C3|{m}|")].sort_values("accuracy_mean", ascending=False).iloc[0]
        row["C3 (best combiner)"] = f"{pm(g.accuracy_mean, g.accuracy_std)} ({g.variant.split('|')[2]})"
        fam.append(row)
    sig = pd.read_csv(R / "expC_significance.csv")
    prof = pd.read_csv(R / "expC_quality_profile_all.csv", index_col=0)
    pcols = ["Total.Cup.Points", "Uniformity", "Clean.Cup", "Sweetness", "Category.One.Defects",
             "Category.Two.Defects", "Moisture", "altitude_mean_meters", "n"]
    pt = prof[pcols].T.reset_index().rename(columns={"index": "Mean per class"})
    for col in ["HQ", "AQ", "LQ"]:
        pt[col] = [f"{x:.0f}" if k in ("n", "altitude_mean_meters") else f"{x:.2f}" for k, x in zip(pt["Mean per class"], pt[col])]
    sig_t = pd.DataFrame({"Comparison": sig.A + " vs " + sig.B,
                          "Acc. A / B %": [f"{100 * a:.1f} / {100 * b:.1f}" for a, b in zip(sig.acc_A, sig.acc_B)],
                          "Wilcoxon p (5 repeats)": sig.wilcoxon_p.map(lambda p: f"{p:.3f}")})
    return md(pd.DataFrame(rows)), md(pd.DataFrame(fam)), md(sig_t), md(pt)


def t_expD():
    d = pd.read_csv(R / "expD_sensor_gcms_fusion.csv")
    rows = []
    for (w, p), g in d.groupby(["w", "p"]):
        g = g.set_index("system")
        row = {"Model, window, period": f"MLP, w={w}, p={p}"}
        for s in ["D0", "D1", "D2", "D3"]:
            row[f"{s} ours %"] = pm(g.loc[s, "test_acc1_mean"], g.loc[s, "test_acc1_std"])
        row["Paper sensor-only / cross-modal %"] = f"{g.loc['D0', 'paper_acc1']:.1f} / {g.loc['D1', 'paper_acc1']:.1f}"
        rows.append(row)
    leak = pd.read_csv(R / "expD_leakage_demo.csv")
    leak_t = pd.DataFrame({"System (w=100, p=25, logistic regression)": leak.system,
                           "Day-6 accuracy %": (100 * leak.test_acc1).round(1)})
    mask = pd.read_csv(A / "AB15_channel_masking.csv")
    mcols = [c for c in mask.columns if c.startswith("mask_")]
    mt = mask.groupby("system")[mcols].mean().T * 100
    mt.index = [i.replace("mask_", "").replace("_delta", "") for i in mt.index]
    mt = mt.round(1).reset_index().rename(columns={"index": "Masked channel", "D0": "Δ D0 (pts)", "D1": "Δ D1 (pts)"})
    enc = pd.read_csv(A / "AB12_gcms_encoding.csv")
    enc_t = enc.pivot_table(index=["w", "p"], columns="gcms", values="test_acc1_mean").reset_index()
    n_atom = enc[enc.gcms == "atom"].set_index(["w", "p"]).n
    enc_t["Xatom seeds"] = [int(n_atom.get((w, p), 0)) for w, p in zip(enc_t.w, enc_t.p)]
    enc_t = pd.DataFrame({"Setting (MLP)": [f"w={w}, p={p}" for w, p in zip(enc_t.w, enc_t.p)],
                          "D1 with Xspec %": (100 * enc_t["spec"]).round(1), "D1 with Xatom %": (100 * enc_t["atom"]).round(1),
                          "Xatom seeds": enc_t["Xatom seeds"]})
    return md(pd.DataFrame(rows)), md(leak_t), md(mt), md(enc_t)


def onset_summary():
    """When does the odour reach the sensors? First second at which resistance falls >5 % below
    the mean of the first 20 s (per recording and sensor). Saved to results/s1_response_onset.csv."""
    from src.data.loaders import S1_SENSORS, load_s1
    X, _, ids = load_s1()
    drop = X < 0.95 * X[:, :20].mean(1, keepdims=True)
    onset = np.where(drop.any(1), drop.argmax(1), X.shape[1])
    pd.DataFrame(onset, columns=S1_SENSORS, index=ids).to_csv(R / "s1_response_onset.csv")
    fast = onset[:, [S1_SENSORS.index(s) for s in ["SP-31", "TGS-842", "SP-AQ3", "TGS-823", "ST-31", "TGS-800"]]].min(1)
    t = pd.DataFrame({"Sensor": S1_SENSORS, "Median onset (s)": np.median(onset, 0).astype(int),
                      "Min (s)": onset.min(0), "Max (s)": onset.max(0),
                      "Recordings with no 5 % drop": (onset == X.shape[1]).sum(0)})
    note = (f"Earliest onset among the six fast sensors: {fast.min()}-{fast.max()} s (median {np.median(fast):.0f} s); "
            f"recordings responding before 30 s: {int((fast < 30).sum())} of {len(fast)}.")
    return md(t), note


def t_ablations_s1():
    """Compact S1 ablation table (5x5 CV) for the reference model of each fusion level."""
    ref = [("A1a|LR", "Low (A1a, LR)"), ("A2|SVM", "Mid (A2, SVM)"), ("A3|LR|stacking", "High (A3, LR stacking)")]
    piv = {n: pd.read_csv(A / f"{n}.csv").pivot_table(index="setting", columns="variant", values="accuracy_mean")
           for n in ["AB1_sensors", "AB2_duration", "AB3_representation", "AB4_features"]}
    rows_spec = [
        ("Full configuration (G = 1/R, 300 s, 8 sensors)", "AB1_sensors", "full"),
        ("AB1: without SP-12A", "AB1_sensors", "without_SP-12A"),
        ("AB1: without TGS-813", "AB1_sensors", "without_TGS-813"),
        ("AB1: TGS-800 only", "AB1_sensors", "only_TGS-800"),
        ("AB1: SP-12A only", "AB1_sensors", "only_SP-12A"),
        ("AB2: first 30 s (no odour response yet)", "AB2_duration", "first_30s"),
        ("AB2: first 60 s", "AB2_duration", "first_60s"),
        ("AB2: first 120 s", "AB2_duration", "first_120s"),
        ("AB3: raw resistance R", "AB3_representation", "rep_R"),
        ("AB3: R/R0 (baseline-normalised)", "AB3_representation", "rep_RR0"),
        ("AB3: G/G0 (baseline-normalised)", "AB3_representation", "rep_GG0"),
        ("AB3: z-score per recording (shape only)", "AB3_representation", "rep_z"),
        ("AB3: G, differenced (p = 25)", "AB3_representation", "G_diff25"),
    ]
    rows = []
    for label, tab, setting in rows_spec:
        p = piv[tab]
        rows.append({"Setting": label, **{name: pct(p.loc[setting, v]) if v in p else "n/a" for v, name in ref}})
    p4 = piv["AB4_features"]
    full4 = p4.loc["full"]
    d4 = (p4.drop(index="full") - full4) * 100
    rows.append({"Setting": "AB4: remove one feature type (range of Δ, points)", ref[0][1]: "n/a",
                 ref[1][1]: f"{d4['A2|SVM'].min():+.1f} to {d4['A2|SVM'].max():+.1f}",
                 ref[2][1]: f"{d4['A3|LR|stacking'].min():+.1f} to {d4['A3|LR|stacking'].max():+.1f}"})
    return md(pd.DataFrame(rows))


def t_ab8():
    """AB8: S1-only vs pooled S1+S2 (B2) per setting, with paired Wilcoxon over the 10 CV repeats.
    Saves results/ablations/AB8_significance.csv."""
    from src.evaluation.cv import per_repeat_scores, wilcoxon_paired
    names = {"full": "full harmonisation, 100 % of S2", "s2_frac_10": "10 % of S2 sessions",
             "s2_frac_25": "25 % of S2 sessions", "s2_frac_50": "50 % of S2 sessions",
             "no_polarity(R)": "no polarity correction (S1 as R)", "norm_ratio": "ratio x/x0 instead of z-score",
             "norm_none": "no per-recording normalisation", "no_common_length": "S2 at full length",
             "diff25": "differenced (p = 25)", "first_60s": "first 60 s", "first_120s": "first 120 s",
             "first_180s": "first 180 s"}
    rows = []
    for setting, label in names.items():
        P = pd.read_csv(R / "runs" / f"AB8_{setting}" / "predictions.csv.gz")
        w = per_repeat_scores(P, [0, 1, 2])
        for m in ["LR", "RF", "MLP64"]:
            _, p = wilcoxon_paired(w[f"B2|{m}"], w[f"S1only|{m}"])
            rows.append({"setting": setting, "label": label, "model": m, "s1_only": w[f"S1only|{m}"].mean(),
                         "pooled": w[f"B2|{m}"].mean(), "diff": w[f"B2|{m}"].mean() - w[f"S1only|{m}"].mean(),
                         "wilcoxon_p": p})
    df = pd.DataFrame(rows)
    df.to_csv(A / "AB8_significance.csv", index=False)
    t = df.pivot_table(index=["setting", "label"], columns="model", values="diff", sort=False) * 100
    pv = df.pivot_table(index=["setting", "label"], columns="model", values="wilcoxon_p", sort=False)
    s1 = df.pivot_table(index=["setting", "label"], columns="model", values="s1_only", sort=False) * 100
    out = pd.DataFrame({"Setting (Experiment B2, mid level)": [l for _, l in t.index]})
    for m in ["LR", "RF", "MLP64"]:
        out[f"{m}: S1-only % → Δ pooled (p)"] = [f"{a:.1f} → {d:+.1f} ({p:.3f})" for a, d, p in
                                                 zip(s1[m].values, t[m].values, pv[m].values)]
    return md(out)


def main():
    parts = ["# Article tables (generated by src/utils/article_tables.py from logged results)\n"]
    parts += ["## AB8 (S2 amount and harmonisation)", t_ab8(), ""]
    parts += ["## S1 ablations (5x5 CV, accuracy %)", t_ablations_s1(), ""]
    ot, on = onset_summary()
    parts += ["## S1 response onset", ot, "", on, ""]
    b1, b2 = t_baseline()
    parts += ["## Baseline reproduction", b1, "", "## PCA explained variance", b2, ""]
    parts += ["## Main comparison", t_main_comparison(), ""]
    f, a3, cnn, sig = t_expA()
    parts += ["## Experiment A: classifier family x level (10x5-fold, LOO in brackets)", f, "",
              "## Experiment A3 combiners", a3, "", "## Experiment A1b CNN", cnn, "", "## Experiment A significance", sig, ""]
    parts += ["## S1 confusion summary", confusion_summary(), ""]
    eb, sec = t_expB()
    parts += ["## Experiment B", eb, "", "## Experiment B secondary check", sec, ""]
    c1, c2, c3, c4 = t_expC()
    parts += ["## Experiment C (best per setting)", c1, "", "## Experiment C by model", c2, "",
              "## Experiment C significance", c3, "", "## Experiment C quality profile", c4, ""]
    d1, d2, d3, d4 = t_expD()
    parts += ["## Experiment D (MLP)", d1, "", "## Leakage demonstration", d2, "",
              "## AB15 channel masking (mean over 4 settings)", d3, "", "## AB12 encoding", d4, ""]
    out = ROOT / "article" / "tables.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
