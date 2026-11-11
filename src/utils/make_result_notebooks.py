"""Generate the explanation notebooks for each phase (02-06). They only READ logged results
(results/*.csv, results/figures/*.png); the experiments themselves run from src/experiments.

Run: python -m src.utils.make_result_notebooks
     jupyter nbconvert --to notebook --execute --inplace notebooks/0[2-6]_*.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = code("""import sys, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 180); pd.set_option('display.max_columns', 30); pd.set_option('display.precision', 3)
R = ROOT / 'results'; F = R / 'figures'; A = R / 'ablations'
def show(p, w=650):
    p = Path(p)
    display(Image(filename=str(p), width=w)) if p.exists() else print('missing figure:', p.name)
def table(p, **kw):
    p = Path(p)
    return pd.read_csv(p, **kw) if p.exists() else print('missing table:', p.name)""")

NB = {
"02_baseline_reproduction": [
    md("# 02 · Reproducing the reference article (RQ1)\n\n"
       "**Goal.** Re-implement the pipeline of Rodríguez et al. (2010) on the public S1 data and compare the results.\n\n"
       "**Pipeline.** Resistance R → conductance G = 1/R → per-sensor *static parameter* ΔG = Gmax − Gmin (the paper does not "
       "give the formula, so we also test the relative variant (Gmax − Gmin)/Gmin) → mean-centring → PCA for the 2-D plot "
       "→ MLP with 10 hidden neurons trained by backpropagation with an adaptive learning rate → leave-one-out (LOO).\n\n"
       "**Deviations.** (1) Python/scikit-learn instead of MATLAB 7.5. (2) The paper's network has *one* output neuron, which is "
       "ambiguous for several classes, so we use 3 softmax outputs. (3) Besides LOO we report 10 × repeated stratified "
       "5-fold CV, and we repeat LOO with 3 random MLP initialisations, because a single LOO number on 58 samples is noisy.\n\n"
       "Code: `src/experiments/s1_baseline.py`."),
    SETUP,
    md("## PCA of the static parameter\n`PC1+PC2` should be compared with the paper's 99.62 % (analysis 1) and > 99 % (analysis 2)."),
    code("table(R / 'baseline_pca.csv')"),
    code("show(F / 'baseline_pca_dG-abs_center.png', 420); show(F / 'baseline_pca_dG-abs_std.png', 420)"),
    md("**Why mean-centring gives such high explained variance.** Without scaling, PCA is dominated by the sensors whose ΔG "
       "has the largest spread (here SP-AQ3), so two components capture almost all the variance. After autoscaling (each "
       "sensor to unit variance) the figure drops, because every sensor then counts equally. Explained variance measures "
       "compression, not class separation."),
    md("## MLP accuracy: LOO vs repeated 5-fold\nVariant names: `feature | input scaling | solver`. `none` = raw ΔG, "
       "`center` = mean-centring only (as in the paper), `std` = autoscaling."),
    code("b = table(R / 'baseline.csv')\nb[['variant','scheme','n_repeats','accuracy_mean','accuracy_std','balanced_accuracy_mean','macro_f1_mean']]"),
    md("**How to read it.** The absolute ΔG values are tiny (about 0.001–0.2 mS). With plain SGD and no autoscaling, the "
       "network barely moves away from predicting the majority class (LQ, 48 %): about 52–55 % accuracy. Adam or "
       "autoscaling lift the absolute feature to 69–75 %, and the relative feature ΔG/Gmin, which is scale-free, reaches "
       "79–83 %. Compare `loo` with `rskf5` for each variant: the gap shows how optimistic LOO is on this data (ablation AB7)."),
    code("show(F / 'baseline_confusion_loo.png', 330); show(F / 'baseline_confusion_rskf5.png', 330)"),
],
"03_expA_sensor_fusion": [
    md("# 03 · Experiment A: fusing the 8 sensors of S1 (RQ2)\n\n"
       "Each sensor is treated as a separate source, and we ask at which level combining them works best.\n\n"
       "| Level | What is combined | Our implementation |\n|---|---|---|\n"
       "| Low (A1) | raw curves | (a) flatten 8 × 300 conductance values → scaler → PCA(10) → classifier; (b) 8-channel 1D-CNN |\n"
       "| Mid (A2) | features | 15 features per sensor (baseline, extrema, ΔG, slopes, area, timing, FFT, …) → 120 values → classifier |\n"
       "| High (A3) | decisions | one classifier per sensor → majority vote / mean probability / weighted mean / stacking |\n\n"
       "**Leakage control.** Features are computed per recording. Every fitted step (scaler, PCA, SVM grid search, "
       "combiner weights, stacking meta-learner) is fitted on the training fold only; the weights and the meta-learner use a 5-fold "
       "*inner* CV inside the training fold. Code: `src/experiments/expA_s1_fusion.py`."),
    SETUP,
    code("a = table(R / 'expA_fusion_levels.csv')\na[a.scheme == 'rskf5'].sort_values('accuracy_mean', ascending=False).head(15)"),
    code("show(F / 'expA_fusion_levels.png', 900)"),
    md("## Best model per level and paired tests\nWilcoxon signed-rank on the 10 per-repeat accuracies of 5-fold CV; McNemar's test on the LOO predictions."),
    code("table(R / 'expA_significance.csv')"),
    md("## LOO numbers (for comparison with the paper's validation scheme)"),
    code("a[a.scheme == 'loo'].sort_values('accuracy_mean', ascending=False).head(10)"),
],
"04_expB_s1_s2_fusion": [
    md("# 04 · Experiment B: does a second gas-sensor dataset (SmellNet) help coffee classification? (RQ3)\n\n"
       "S1 (8 Figaro/FIS sensors, resistance) and S2 (6 Grove/MQ channels, raw readings) differ in sensors, units and "
       "channel count. **Harmonisation:** S1 → conductance (so both rise with gas), common length of 300 s, per-measurement "
       "z-score per channel. **Channel-agnostic models:** B1 uses one encoder shared by every channel, pooled across channels; "
       "B2 computes features per channel and aggregates them across channels (mean/std/max/min). Both are trained on 3 coffee + "
       "50 SmellNet classes and evaluated **only on held-out S1 measurements** (argmax over the 3 coffee classes). B3 combines "
       "the S1-only model with the pooled model. S2 only ever enters training. Code: `src/experiments/expB_s1_s2.py`."),
    SETUP,
    code("b = table(R / 'expB_s1_s2_fusion.csv')\nb[['exp','level','variant','accuracy_mean','accuracy_std','balanced_accuracy_mean','macro_f1_mean']]"),
    code("show(F / 'expB_s1_s2_fusion.png', 800)"),
    md("**Paired comparison with the identical S1-only model** (Wilcoxon over CV repeats):"),
    code("table(R / 'expB_significance.csv')"),
    md("**Secondary check:** does the pooled model still recognise SmellNet substances on the unseen day 6 (50-way)?"),
    code("table(R / 'expB_s2_secondary_check.csv')"),
    md("## AB8: amount of S2 and each harmonisation step"),
    code("table(A / 'AB8_s2_gain_by_setting.csv')"),
    code("show(F / 'AB8_s2_fusion.png', 900)"),
],
"05_expC_cqi_fusion": [
    md("# 05 · Experiment C: fusion within the CQI cupping database (RQ4)\n\n"
       "**Target.** Three quality classes from tertiles of the total cup score (HQ ≥ 83.17, AQ 81.67–83.17, LQ < 81.67), "
       "mirroring S1's HQ/AQ/LQ.\n\n**Sources.** G1 = the 10 sensory scores (the total *is* their sum, so G1 is only an "
       "upper bound), G2 = physical/bean data, G3 = origin. **C1** concatenates the encoded G2 + G3 columns, **C2** uses per-group "
       "PCA embeddings, **C3** trains one model per group and combines the decisions. Imputation, encoding and PCA are fitted "
       "in-fold. Code: `src/experiments/expC_cqi.py`."),
    SETUP,
    code("c = table(R / 'expC_cqi_fusion.csv')\nc[['level','variant','accuracy_mean','accuracy_std','balanced_accuracy_mean','macro_f1_mean']].sort_values(['level','accuracy_mean'], ascending=[True, False])"),
    code("table(R / 'expC_significance.csv')"),
    code("show(F / 'AB9_group_removal_S3.png', 700); show(F / 'AB5_combiners_S3.png', 700)"),
    md("## What do the quality grades mean in cupping terms?"),
    code("table(R / 'expC_quality_profile_all.csv', index_col=0)"),
    code("table(R / 'expC_quality_profile_colombia.csv', index_col=0)"),
    code("show(F / 'expC_quality_grades.png', 850)"),
],
"06_expD_sensor_gcms": [
    md("# 06 · Experiment D: sensor + GC-MS fusion in SmellNet, without label leakage (RQ6)\n\n"
       "SmellNet's GC-MS data is **one vector per ingredient**. If a test sample's GC-MS vector is looked up by its label, "
       "the model is handed the answer. The leakage demonstration below shows this directly. The valid systems use GC-MS "
       "only during training (D1, D2) or as the fixed prototypes of *all* 50 classes (D1 prediction), and D3 combines decisions.\n\n"
       "**Protocol.** Train on days 1–4, select learning rate / λ / α / checkpoint on day 5, test once on day 6. "
       "Code: `src/experiments/expD_smellnet.py` (uses SmellNet's own model classes and contrastive loss)."),
    SETUP,
    md("## Leakage demonstration (invalid by design; never reported as a result)"),
    code("table(R / 'expD_leakage_demo.csv')"),
    md("## Results vs the SmellNet paper (Table 2, ICLR 2026)\nOur accuracies are fractions; `paper_acc1` is in %. D0 ↔ "
       "paper's *sensor-only*, D1 ↔ paper's *cross-modal*."),
    code("d = table(R / 'expD_sensor_gcms_fusion.csv')\nd if d is None else d[['model','w','p','system','n_seeds','test_acc1_mean','test_acc1_std','paper_acc1']]"),
    code("table(A / 'AB11_with_without_gcms.csv')"),
    code("table(A / 'AB15_channel_masking.csv')"),
    code("table(A / 'AB12_gcms_encoding.csv')"),
],
"07_ablations": [
    md("# 07 · Ablation studies (RQ5): which components actually matter?\n\n"
       "An ablation changes **one** factor and keeps everything else fixed, then measures the change in accuracy. "
       "For S1 every ablation uses one reference model per fusion level: low (A1a: curves → PCA → LR), "
       "mid (A2: 120 features → SVM), high (A3: per-sensor LR → stacking). Each is evaluated with 5 × 5 repeated "
       "stratified CV (Experiment A's main table uses 10 × 5, so the *full* rows can differ slightly).\n\n"
       "| ID | Question |\n|---|---|\n"
       "| AB1 | Which sensors matter? (leave one sensor out; single sensor only) |\n"
       "| AB2 | How much of the 300-s recording is needed? |\n"
       "| AB3 | Which signal representation works best, and does the pre-exposure baseline drive accuracy? |\n"
       "| AB4 | Which hand-crafted feature types matter? |\n"
       "| AB5 / AB6 | Which combiner / classifier family? (from Experiments A and C) |\n"
       "| AB7 | How optimistic is leave-one-out compared with repeated 5-fold CV? |\n"
       "| AB8 | Does the amount of SmellNet data or any harmonisation step change Experiment B? |\n"
       "| AB9 | Which CQI attribute group carries the signal? |\n"
       "| AB10 | Does sliding-window augmentation help the CNN? |"),
    SETUP,
    md("## AB1: sensors\n`delta_vs_full` is the change in accuracy compared with using all 8 sensors."),
    code("t = table(A / 'AB1_sensors.csv')\nt.pivot_table(index='setting', columns='variant', values='accuracy_mean').round(3)"),
    md("## AB2: signal duration"),
    code("t = table(A / 'AB2_duration.csv')\nt.pivot_table(index='setting', columns='variant', values='accuracy_mean').round(3)"),
    md("## AB3: signal representation\nR = raw resistance, G = conductance, R/R0 and G/G0 divide by the pre-exposure "
       "baseline (first 20 s), z = per-recording z-score; *diff25* applies SmellNet-style differencing x(t) − x(t−25). "
       "If accuracy falls sharply once the baseline is removed, the classifier was relying on the baseline level, "
       "which is measured before the coffee arrives."),
    code("t = table(A / 'AB3_representation.csv')\nt.pivot_table(index='setting', columns='variant', values='accuracy_mean').round(3)"),
    md("## AB4: feature types (mid and high level)"),
    code("t = table(A / 'AB4_features.csv')\nt.pivot_table(index='setting', columns='variant', values='accuracy_mean').round(3)"),
    code("show(R.parent / 'article' / 'figures' / 'fig6_s1_ablations.png', 900)"),
    md("## AB5 / AB6: combiners and classifier families"),
    code("display(table(A / 'AB5_combiners_S1.csv')); display(table(A / 'AB6_classifier_family_S1.csv'))"),
    md("## AB7: leave-one-out vs repeated 5-fold\n`loo_minus_rskf` > 0 means LOO reports a higher accuracy than repeated 5-fold CV for the same pipeline."),
    code("t = table(A / 'AB7_validation_scheme.csv')\nt.sort_values('loo_minus_rskf', ascending=False)"),
    md("## AB8: SmellNet amount and harmonisation (Experiment B)"),
    code("table(A / 'AB8_s2_gain_by_setting.csv')"),
    md("## AB10: sliding-window augmentation for the 1D-CNN"),
    code("table(A / 'AB10_window_augmentation.csv')"),
],
}

for name, cells in NB.items():
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
print("written", list(NB))
