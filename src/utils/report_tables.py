"""Generate the LaTeX tables of the Mini Project-II report from logged results.

Every table body is written to report/tables/<name>.tex (tabular only; captions and labels
live in the main .tex). No number is typed by hand. Output is pure ASCII LaTeX for pdfLaTeX.
Run: python -m src.utils.report_tables
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.cv import per_repeat_scores, wilcoxon_paired

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
A = R / "ablations"
OUT = ROOT / "report" / "tables"

UNI = {"±": r"$\pm$", "→": r"$\rightarrow$", "Δ": r"$\Delta$", "×": r"$\times$", "≤": r"$\leq$",
       "≥": r"$\geq$", "↔": r"$\leftrightarrow$", "·": r"$\cdot$", "−": r"$-$", "–": "--", "—": "---",
       "é": r"\'{e}", "í": r"\'{\i}", "ö": r'\"{o}', "ü": r'\"{u}', "á": r"\'{a}", "ñ": r"\~{n}"}


def tex(x) -> str:
    """Escape a cell for pdfLaTeX (ASCII output)."""
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"),
                 ("|", r"\textbar{}"), ("<", r"$<$"), (">", r"$>$"), ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    s = re.sub(r"(?<![\w$\\])-(?=\d)", r"$-$", s)          # negative numbers -> math minus
    for a, b in UNI.items():
        s = s.replace(a, b)
    if any(ord(ch) > 127 for ch in s):
        raise ValueError(f"non-ASCII left in cell: {s!r}")
    return s


def tabular(df: pd.DataFrame, align: str | None = None, bold_header=True) -> str:
    cols = list(df.columns)
    align = align or ("l" + "c" * (len(cols) - 1))
    head = " & ".join((r"\textbf{" + tex(c) + "}") if bold_header else tex(c) for c in cols)
    rows = [" & ".join(tex(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    return "\n".join([r"\begin{tabular}{" + align + "}", r"\toprule", head + r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"]) + "\n"


def save(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="ascii")
    print("wrote", OUT / f"{name}.tex")


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}"


def pm(m, s, d=1):
    return "n/a" if pd.isna(m) else (f"{100 * m:.{d}f} ± {100 * s:.{d}f}" if not pd.isna(s) else f"{100 * m:.{d}f}")


def get(df, variant, scheme, col):
    r = df[(df.variant == variant) & (df.scheme == scheme)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ------------------------------------------------------------------ tables --------
def t_onset():
    o = pd.read_csv(R / "s1_response_onset.csv", index_col=0)
    T = 300
    df = pd.DataFrame({"Sensor": o.columns, "Median onset (s)": o.median().astype(int).values,
                       "Min (s)": o.min().values, "Max (s)": o.max().values,
                       "Recordings without a 5% drop": (o == T).sum().values})
    save("onset", tabular(df))


def t_pca():
    p = pd.read_csv(R / "baseline_pca.csv")
    df = pd.DataFrame({"Static parameter": p.feature.map({"dG-abs": "ΔG = Gmax − Gmin", "dG-rel": "ΔG / Gmin"}),
                       "Scaling": p.scaling.map({"center": "mean-centring", "std": "autoscaling"}),
                       "PC1 (%)": (100 * p.PC1).map("{:.2f}".format), "PC2 (%)": (100 * p.PC2).map("{:.2f}".format),
                       "PC1+PC2 (%)": (100 * p["PC1+PC2"]).map("{:.2f}".format)})
    save("pca", tabular(df))


def t_baseline():
    b = pd.read_csv(R / "baseline.csv")
    names = {"none": "none", "center": "mean-centring", "std": "autoscaling"}
    rows = []
    for v in b.variant.unique():
        feat, sc, solver = v.split("|")
        rows.append({"Static parameter": "ΔG = Gmax − Gmin" if feat == "dG-abs" else "ΔG / Gmin",
                     "Input scaling": names[sc], "Solver": "SGD" if solver == "sgd" else "Adam",
                     "LOO acc. (%)": pm(get(b, v, "loo", "accuracy_mean"), get(b, v, "loo", "accuracy_std")),
                     "10×5-fold acc. (%)": pm(get(b, v, "rskf5", "accuracy_mean"), get(b, v, "rskf5", "accuracy_std")),
                     "Macro-F1": f"{get(b, v, 'rskf5', 'macro_f1_mean'):.3f}"})
    save("baseline", tabular(pd.DataFrame(rows), "lllccc"))


def t_expA():
    a = pd.read_csv(R / "expA_fusion_levels.csv")
    r = a[a.scheme == "rskf5"].set_index("variant")
    lo = a[a.scheme == "loo"].set_index("variant")
    rows = []
    for c, name in [("LR", "Logistic regression"), ("SVM", "SVM (RBF, nested C)"), ("RF", "Random forest"),
                    ("kNN", "k-NN (k = 5)"), ("MLP", "MLP (10 hidden units)")]:
        rows.append({"Classifier": name,
                     "Low: 10×5-fold": pm(r.loc[f"A1a|{c}", "accuracy_mean"], r.loc[f"A1a|{c}", "accuracy_std"]),
                     "Low: LOO": pct(lo.loc[f"A1a|{c}", "accuracy_mean"]),
                     "Mid: 10×5-fold": pm(r.loc[f"A2|{c}", "accuracy_mean"], r.loc[f"A2|{c}", "accuracy_std"]),
                     "Mid: LOO": pct(lo.loc[f"A2|{c}", "accuracy_mean"])})
    for v, name in [("A1b|CNN", "8-channel 1D-CNN"), ("A1b|CNN+win", "8-channel 1D-CNN + windows")]:
        rows.append({"Classifier": name, "Low: 10×5-fold": pm(r.loc[v, "accuracy_mean"], r.loc[v, "accuracy_std"]),
                     "Low: LOO": pct(lo.loc[v, "accuracy_mean"]), "Mid: 10×5-fold": "--", "Mid: LOO": "--"})
    save("expA_levels", tabular(pd.DataFrame(rows), "lcccc"))
    rows = []
    for base in ["LR", "SVM"]:
        row = {"Per-sensor model": "Logistic regression" if base == "LR" else "SVM"}
        for comb, name in [("vote", "Majority vote"), ("mean", "Mean prob."), ("weighted", "Weighted mean"),
                           ("stacking", "Stacking")]:
            v = f"A3|{base}|{comb}"
            row[name] = f"{pm(r.loc[v, 'accuracy_mean'], r.loc[v, 'accuracy_std'])} ({pct(lo.loc[v, 'accuracy_mean'])})"
        rows.append(row)
    save("expA_highlevel", tabular(pd.DataFrame(rows), "lcccc"))
    sig = pd.read_csv(R / "expA_significance.csv")
    nm = {"A1a|LR": "Low (A1a, LR)", "A2|SVM": "Mid (A2, SVM)", "A3|SVM|stacking": "High (A3, SVM, stacking)"}
    df = pd.DataFrame({"Comparison": [f"{nm[a]} vs {nm[b]}" for a, b in zip(sig.A, sig.B)],
                       "Acc. (%)": [f"{100 * x:.1f} vs {100 * y:.1f}" for x, y in zip(sig.acc_A_rskf, sig.acc_B_rskf)],
                       "Wilcoxon p": sig.wilcoxon_p_rskf.map("{:.3f}".format),
                       "LOO discordant (A / B)": [f"{x} / {y}" for x, y in zip(sig.loo_A_only_correct, sig.loo_B_only_correct)],
                       "McNemar p": sig.mcnemar_p_loo.map("{:.3f}".format)})
    save("expA_significance", tabular(df, "lcccc"))


def t_confusion():
    c = pd.read_csv(R / "s1_confusion_summary.csv")
    nm = {"dG-rel|center|sgd": "Baseline (ΔG/Gmin, MLP)", "A1a|LR": "Low (A1a, LR)", "A1b|CNN+win": "Low (1D-CNN + windows)",
          "A2|SVM": "Mid (A2, SVM)", "A3|SVM|stacking": "High (A3, SVM, stacking)"}
    df = pd.DataFrame({"Model": c.variant.map(nm), "Validation": c.scheme.map({"loo": "LOO", "rskf5": "10×5-fold"}),
                       "Recall HQ": c.recall_HQ.map("{:.2f}".format), "Recall AQ": c.recall_AQ.map("{:.2f}".format),
                       "Recall LQ": c.recall_LQ.map("{:.2f}".format), "Errors": c.errors,
                       "AQ↔LQ errors": [f"{x} ({100 * x / e:.0f}%)" for x, e in zip(c.AQ_LQ_confusions, c.errors)]})
    save("confusion", tabular(df, "llccccc"))


def t_ablation_s1():
    ref = [("A1a|LR", "Low (A1a, LR)"), ("A2|SVM", "Mid (A2, SVM)"), ("A3|LR|stacking", "High (A3, LR, stacking)")]
    piv = {n: pd.read_csv(A / f"{n}.csv").pivot_table(index="setting", columns="variant", values="accuracy_mean")
           for n in ["AB1_sensors", "AB2_duration", "AB3_representation", "AB4_features"]}
    spec = [("Full configuration (G = 1/R, 300 s, 8 sensors)", "AB1_sensors", "full"),
            ("AB1: without SP-12A", "AB1_sensors", "without_SP-12A"),
            ("AB1: without TGS-813", "AB1_sensors", "without_TGS-813"),
            ("AB1: TGS-800 only", "AB1_sensors", "only_TGS-800"),
            ("AB1: SP-12A only", "AB1_sensors", "only_SP-12A"),
            ("AB2: first 30 s (before any odour response)", "AB2_duration", "first_30s"),
            ("AB2: first 60 s", "AB2_duration", "first_60s"),
            ("AB2: first 120 s", "AB2_duration", "first_120s"),
            ("AB3: raw resistance R", "AB3_representation", "rep_R"),
            ("AB3: R/R0 (baseline-normalised)", "AB3_representation", "rep_RR0"),
            ("AB3: G/G0 (baseline-normalised)", "AB3_representation", "rep_GG0"),
            ("AB3: z-score per recording (shape only)", "AB3_representation", "rep_z"),
            ("AB3: G, differenced (p = 25)", "AB3_representation", "G_diff25")]
    rows = [{"Setting": lab, **{name: pct(piv[t].loc[s, v]) if v in piv[t] else "n/a" for v, name in ref}}
            for lab, t, s in spec]
    p4 = piv["AB4_features"]
    d4 = (p4.drop(index="full") - p4.loc["full"]) * 100
    rows.append({"Setting": "AB4: remove one feature type (range of Δ, points)", ref[0][1]: "n/a",
                 ref[1][1]: f"{d4['A2|SVM'].min():+.1f} to {d4['A2|SVM'].max():+.1f}",
                 ref[2][1]: f"{d4['A3|LR|stacking'].min():+.1f} to {d4['A3|LR|stacking'].max():+.1f}"})
    save("ablation_s1", tabular(pd.DataFrame(rows), "lccc"))


def t_expB():
    b = pd.read_csv(R / "expB_s1_s2_fusion.csv")
    sig = pd.read_csv(R / "expB_significance.csv").set_index("variant")
    s = b.set_index("variant")
    rows = []
    for m, name, pooled in [("LR", "Logistic regression", "B2|LR"), ("RF", "Random forest", "B2|RF"),
                            ("MLP64", "MLP (64 hidden units)", "B2|MLP64"), ("CA-CNN", "Channel-agnostic CNN", "B1|CA-CNN")]:
        b3 = s.loc[[v for v in s.index if v.startswith(f"B3|{m}|")]].sort_values("accuracy_mean").index[-1]
        rows.append({"Model": name,
                     "S1 only (%)": pm(s.loc[f"S1only|{m}", "accuracy_mean"], s.loc[f"S1only|{m}", "accuracy_std"]),
                     "Pooled S1+S2 (%)": pm(s.loc[pooled, "accuracy_mean"], s.loc[pooled, "accuracy_std"]),
                     "Δ (p)": f"{100 * sig.loc[pooled, 'diff']:+.1f} ({sig.loc[pooled, 'wilcoxon_p']:.3f})",
                     "Best decision fusion (%)": f"{pm(s.loc[b3, 'accuracy_mean'], s.loc[b3, 'accuracy_std'])} [{b3.split('|')[-1]}]",
                     "Δ (p) ": f"{100 * sig.loc[b3, 'diff']:+.1f} ({sig.loc[b3, 'wilcoxon_p']:.3f})"})
    save("expB", tabular(pd.DataFrame(rows), "lccccc"))
    ab = pd.read_csv(A / "AB8_significance.csv")
    keep = [("full", "Full harmonisation, all S2 sessions"), ("s2_frac_10", "Only 10% of S2 sessions"),
            ("no_polarity(R)", "No polarity correction (S1 as R)"), ("norm_none", "No per-recording normalisation"),
            ("diff25", "Differenced signals (p = 25)")]
    rows = []
    for st, lab in keep:
        row = {"AB8 setting": lab}
        for m, nm in [("LR", "LR"), ("RF", "RF"), ("MLP64", "MLP")]:
            r = ab[(ab.setting == st) & (ab.model == m)].iloc[0]
            row[nm] = f"{100 * r.s1_only:.1f} → {100 * r['diff']:+.1f} ({r.wilcoxon_p:.3f})"
        rows.append(row)
    save("ab8", tabular(pd.DataFrame(rows), "lccc"))


def t_expC():
    c = pd.read_csv(R / "expC_cqi_fusion.csv")
    names = {"C1": "C1, low level (G2+G3 concatenated)", "C2": "C2, mid level (per-group PCA)",
             "C3": "C3, high level (per-group models)", "only_G2": "G2 only (bean/physical)",
             "only_G3": "G3 only (origin)", "UPPER_G1": "Upper bound: G1 sensory scores",
             "UPPER_G1G2G3": "Upper bound: G1+G2+G3"}
    rows = []
    for lv, lab in names.items():
        g = c[c.level == lv].sort_values("accuracy_mean").iloc[-1]
        rows.append({"Setting": lab, "Best model": g.variant.split("|", 1)[1].replace("|", ", "),
                     "Accuracy (%)": pm(g.accuracy_mean, g.accuracy_std),
                     "Balanced acc. (%)": pct(g.balanced_accuracy_mean), "Macro-F1": f"{g.macro_f1_mean:.3f}"})
    save("expC", tabular(pd.DataFrame(rows), "llccc"))
    prof = pd.read_csv(R / "expC_quality_profile_all.csv", index_col=0)
    items = [("Total.Cup.Points", "Total cup points", 2), ("Aroma", "Aroma", 2), ("Flavor", "Flavour", 2),
             ("Uniformity", "Uniformity", 2), ("Clean.Cup", "Clean cup", 2), ("Sweetness", "Sweetness", 2),
             ("Category.One.Defects", "Category-one defects", 2), ("Category.Two.Defects", "Category-two defects", 2),
             ("altitude_mean_meters", "Mean altitude (m)", 0), ("n", "Number of reviews", 0)]
    rows = [{"Attribute (class mean)": lab, **{q: f"{prof.loc[q, k]:.{d}f}" for q in ["HQ", "AQ", "LQ"]}}
            for k, lab, d in items]
    save("cqi_profile", tabular(pd.DataFrame(rows), "lccc"))


def t_expD():
    leak = pd.read_csv(R / "expD_leakage_demo.csv")
    lab = {"DX-gcms_only (LEAKY)": "GC-MS vector of the true class only (leaky)",
           "DX-early_concat (LEAKY)": "Sensor features + GC-MS of the true class (leaky)",
           "DX-early_concat, label-free GC-MS at test": "Same model, label-free GC-MS at test time",
           "sensor-only same features (valid)": "Sensor features only (valid)"}
    save("leakage", tabular(pd.DataFrame({"System (logistic regression, w = 100, p = 25)": leak.system.map(lab),
                                          "Test-day accuracy (%)": (100 * leak.test_acc1).map("{:.1f}".format)}), "lc"))
    d = pd.read_csv(R / "expD_sensor_gcms_fusion.csv")
    rows = []
    for (w, p), g in d.groupby(["w", "p"]):
        g = g.set_index("system")
        row = {"Window w, period p": f"w = {w}, p = {p}"}
        for sname, label in [("D0", "D0 sensor-only"), ("D1", "D1 contrastive"), ("D2", "D2 hybrid"),
                             ("D3", "D3 ensemble")]:
            row[label] = pm(g.loc[sname, "test_acc1_mean"], g.loc[sname, "test_acc1_std"])
        row["Reported D0 / D1 [Feng et al.]"] = f"{g.loc['D0', 'paper_acc1']:.1f} / {g.loc['D1', 'paper_acc1']:.1f}"
        rows.append(row)
    save("expD", tabular(pd.DataFrame(rows), "lccccc"))
    mask = pd.read_csv(A / "AB15_channel_masking.csv")
    mcols = [c for c in mask.columns if c.startswith("mask_")]
    mt = (mask.groupby("system")[mcols].mean().T * 100).round(1)
    mt.index = [i.replace("mask_", "").replace("_delta", "") for i in mt.index]
    df = pd.DataFrame({"Masked channel": mt.index, "Δ D0 (points)": mt["D0"].map("{:+.1f}".format).values,
                       "Δ D1 (points)": mt["D1"].map("{:+.1f}".format).values})
    save("masking", tabular(df, "lcc"))


def t_main():
    base = pd.read_csv(R / "baseline.csv")
    a = pd.read_csv(R / "expA_fusion_levels.csv")
    b = pd.read_csv(R / "expB_s1_s2_fusion.csv")
    rows = [["Rodríguez et al. (2010), analysis 1", "16 meas.", "4", "LOO", "100", "n/a", "reported"],
            ["Rodríguez et al. (2010), analysis 2", "35 meas.", "9", "LOO", "92.5", "n/a", "reported"]]
    for name, df, v in [("Baseline reproduction (ΔG/Gmin, MLP)", base, "dG-rel|center|sgd"),
                        ("Low-level fusion (A1a, LR)", a, "A1a|LR"),
                        ("Low-level fusion (1D-CNN + windows)", a, "A1b|CNN+win"),
                        ("Mid-level fusion (A2, SVM)", a, "A2|SVM"),
                        ("High-level fusion (A3, SVM, stacking)", a, "A3|SVM|stacking")]:
        rows.append([name, "S1, 58 meas.", "3", "LOO / 10×5-fold",
                     f"{pct(get(df, v, 'loo', 'accuracy_mean'))} / {pm(get(df, v, 'rskf5', 'accuracy_mean'), get(df, v, 'rskf5', 'accuracy_std'))}",
                     f"{get(df, v, 'loo', 'macro_f1_mean'):.3f} / {get(df, v, 'rskf5', 'macro_f1_mean'):.3f}", "this work"])
    bb = b[b.variant.str.startswith(("B1", "B2", "B3"))].sort_values("accuracy_mean").iloc[-1]
    rows.append([f"S1+S2 fusion (B3, RF, stacking)" if bb.variant == "B3|RF|stacking" else bb.variant,
                 "S1 + S2", "3", "10×5-fold", pm(bb.accuracy_mean, bb.accuracy_std), f"{bb.macro_f1_mean:.3f}", "this work"])
    ab3 = pd.read_csv(A / "AB3_representation.csv").pivot_table(index="setting", columns="variant", values="accuracy_mean")
    ab3s = pd.read_csv(A / "AB3_representation.csv").pivot_table(index="setting", columns="variant", values="accuracy_std")
    rows.append(["Mid-level fusion on level-free (z-scored) signals (SVM)", "S1, 58 meas.", "3", "5×5-fold",
                 pm(ab3.loc["rep_z", "A2|SVM"], ab3s.loc["rep_z", "A2|SVM"]), "--", "this work"])
    df = pd.DataFrame(rows, columns=["Method", "Data", "Classes", "Validation", "Accuracy (%)", "Macro-F1", "Source"])
    save("main_comparison", tabular(df, "llcclcc"))


if __name__ == "__main__":
    for f in [t_onset, t_pca, t_baseline, t_expA, t_confusion, t_ablation_s1, t_expB, t_expC, t_expD, t_main]:
        f()
