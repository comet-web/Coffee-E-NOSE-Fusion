"""Generate the three EDA notebooks (notebooks/01_eda_s1|s2|s3.ipynb).

Run: python -m src.utils.make_eda_notebooks   then execute them with
     jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_*.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = code("""import sys, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
FIG = ROOT / 'results' / 'figures'""")

s1 = [
    md("# 01 · EDA of Source 1: the coffee e-nose dataset\n\n"
       "**What this notebook does.** It loads the 58 public e-nose measurements (Rodríguez et al., 2010), checks their "
       "format (rows, columns, missing values, class balance), plots the raw sensor curves, compares the classes and "
       "redoes the paper's PCA view. Every finding is copied into `docs/data_notes.md`.\n\n"
       "**Background.** Each file is one measurement: 8 metal-oxide (MOX) gas sensors sampled once per second for 300 s. "
       "A MOX sensor's resistance *drops* when reducing gases (coffee volatiles) reach it, so the curves go down when the odour arrives."),
    SETUP,
    code("from src.data.loaders import load_s1, S1_SENSORS, S1_CLASSES\n"
         "X, y, ids = load_s1()\n"
         "print('shape (measurements, seconds, sensors):', X.shape)\n"
         "print('missing values:', np.isnan(X).sum(), '| min/max kOhm:', X.min().round(2), X.max().round(2))\n"
         "print('class counts:', dict(zip(S1_CLASSES, np.bincount(y))))\n"
         "print('majority-class accuracy (the trivial baseline):', round(np.bincount(y).max() / len(y), 3))"),
    md("**Reading the output.** All files have exactly 300 rows × 8 columns and no missing values. The classes are imbalanced "
       "(LQ 28, HQ 20, AQ 10), so a model that always says *LQ* already gets 48 %. That is why we also report balanced "
       "accuracy and macro-F1, not only accuracy."),
    code("fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)\n"
         "for r, c in enumerate(S1_CLASSES):\n"
         "    for k, i in enumerate(np.where(y == r)[0][:2]):\n"
         "        ax = axes[r, k]\n"
         "        for s in range(8): ax.plot(X[i, :, s], lw=1, label=S1_SENSORS[s])\n"
         "        ax.set_title(f'{ids[i]}'); ax.set_ylabel('R (kOhm)')\n"
         "axes[0, 1].legend(fontsize=7, ncol=2); axes[-1, 0].set_xlabel('time (s)'); axes[-1, 1].set_xlabel('time (s)')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'eda_s1_curves.png', dpi=130); plt.show()"),
    md("**Signal structure.** Every curve is flat for about the first 30 s (the *baseline*, before the coffee headspace "
       "reaches the sensors), drops sharply, reaches a minimum after roughly 50–60 s for most sensors and then recovers slowly. "
       "This matches the measurement phase of the paper's cycle (its Figs. 5–6). We use the first 20 s as the baseline segment."),
    code("t_min = pd.DataFrame(X.argmin(1), columns=S1_SENSORS)\n"
         "print('median time of minimum resistance (s):'); print(t_min.median().to_string())\n"
         "base = pd.DataFrame(X[:, :20].mean(1), columns=S1_SENSORS).assign(cls=np.array(S1_CLASSES)[y])\n"
         "print('\\nmean pre-exposure baseline resistance per class (kOhm):')\n"
         "base.groupby('cls').mean().loc[S1_CLASSES].round(1)"),
    md("**An important observation.** The baseline resistance, measured *before* the coffee arrives, already differs by "
       "class (e.g. SP-12A, mean of the first 20 s: HQ 46.4, AQ 33.9, LQ 38.7 kΩ). Coffee aroma cannot cause that. It points to measurement "
       "sessions or sensor drift, i.e. the classes were probably recorded at different times. A classifier could exploit this "
       "instead of the aroma. Ablation AB3 measures how much accuracy depends on it by comparing representations that remove "
       "the baseline (R/R0, G/G0, z-score) with the raw ones."),
    code("fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True)\n"
         "colors = {'HQ': '#1b7837', 'AQ': '#d9a400', 'LQ': '#b2182b'}\n"
         "for s, ax in enumerate(axes.ravel()):\n"
         "    for r, c in enumerate(S1_CLASSES):\n"
         "        m, sd = X[y == r, :, s].mean(0), X[y == r, :, s].std(0)\n"
         "        ax.plot(m, color=colors[c], label=c); ax.fill_between(range(300), m - sd, m + sd, color=colors[c], alpha=0.15)\n"
         "    ax.set_title(S1_SENSORS[s], fontsize=9)\n"
         "axes[0, 0].legend(); plt.tight_layout(); plt.savefig(FIG / 'eda_s1_class_means.png', dpi=130); plt.show()"),
    md("## The paper's view: static parameter + PCA\nThe paper's feature per sensor is the variation between maximum and "
       "minimum conductance (G = 1/R). We use ΔG = Gmax − Gmin, mean-centre the 58 × 8 matrix and project it on the first "
       "two principal components (the paper's Figs. 7 and 9). This PCA is purely descriptive (fitted on all data, no labels used)."),
    code("from sklearn.decomposition import PCA\nfrom src.features.enose import delta_g\n"
         "F = delta_g(X, 'abs'); Z = F - F.mean(0); p = PCA().fit(Z); P = p.transform(Z)\n"
         "print('explained variance PC1, PC2:', p.explained_variance_ratio_[:2].round(4), '| PC1+PC2 =', p.explained_variance_ratio_[:2].sum().round(4))\n"
         "plt.figure(figsize=(4.5, 4))\n"
         "for r, c in enumerate(S1_CLASSES): plt.scatter(P[y == r, 0], P[y == r, 1], label=c, color=colors[c], s=25)\n"
         "plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.title('dG, mean-centred'); plt.show()\n"
         "pd.Series(p.components_[0], index=S1_SENSORS).round(3).rename('PC1 loading')"),
    md("**Reading the PCA.** PC1+PC2 explain about 99.5 % of the variance, close to the paper's 99.62 %. But this "
       "number is mostly a consequence of mean-centring without scaling: the sensors with the largest ΔG dominate PC1 "
       "(SP-AQ3 has a loading of about 0.85; see the loadings). A high explained variance does not by itself mean the classes are separable."),
]

s2 = [
    md("# 01 · EDA of Source 2: SmellNet-Base\n\n"
       "**What this notebook does.** It checks the SmellNet-Base files (50 substances × 6 days, 6 gas channels at 1 Hz), "
       "the official train/test split (days 1–5 / day 6), recording lengths, and the GC-MS vectors shipped with the dataset. "
       "SmellNet contains no coffee; in this project it is (i) a second MOX gas-sensor dataset to fuse with S1 (Experiment B) "
       "and (ii) the testbed for sensor + GC-MS fusion (Experiment D)."),
    SETUP,
    code("from src.data.loaders import load_s2_sessions, load_s2_gcms, S2_CHANNELS\n"
         "S = load_s2_sessions()\n"
         "meta = pd.DataFrame([{'substance': s['substance'], 'day': s['day'], 'split': s['split'], 'length': len(s['data'])} for s in S])\n"
         "print('sessions:', len(meta), '| substances:', meta.substance.nunique(), '| channels:', S2_CHANNELS)\n"
         "print(meta.groupby('split').agg(sessions=('day', 'size'), days=('day', lambda d: sorted(d.unique()))))\n"
         "meta.length.describe().round(1)"),
    md("**Reading the output.** 300 sessions = 50 substances × 6 days, one CSV each; days 1–5 are in `training/` and day 6 "
       "in `testing/`. Sessions last 517–867 s (median 600 s)."),
    code("A = np.concatenate([s['data'] for s in S])\n"
         "pd.DataFrame({'min': A.min(0), 'median': np.median(A, 0), 'max': A.max(0)}, index=S2_CHANNELS)"),
    code("fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)\n"
         "for ax, sub in zip(axes, ['cinnamon', 'garlic', 'banana']):\n"
         "    s = next(x for x in S if x['substance'] == sub and x['day'] == 1)\n"
         "    for c in range(6): ax.plot(s['data'][:, c] - s['data'][0, c], lw=1, label=S2_CHANNELS[c])\n"
         "    ax.set_title(f'{sub}, day 1 (minus first row)'); ax.set_xlabel('time (s)')\n"
         "axes[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / 'eda_s2_examples.png', dpi=130); plt.show()"),
    md("**Polarity.** Unlike S1's resistance, SmellNet's raw readings *rise* when the odour is present. For the S1 + S2 "
       "fusion we therefore convert S1 to conductance (G = 1/R), which also rises with gas."),
    code("G = load_s2_gcms('spec'); Ga = load_s2_gcms('atom')\n"
         "print('Xspec:', G.shape, '| Xatom:', Ga.shape, '| same 50 substances:', set(G.index) == set(Ga.index) == set(meta.substance))\n"
         "Ga.head()"),
    md("**GC-MS is class-level.** There is exactly one GC-MS vector per substance, derived from the FooDB database, not "
       "measured per session. If a model received the GC-MS vector of a test sample's (true) substance, it would be handed "
       "the answer. Experiment D therefore uses GC-MS only during training, or as the fixed prototypes of *all* 50 classes."),
]

s3 = [
    md("# 01 · EDA of Source 3: CQI coffee-quality database\n\n"
       "**What this notebook does.** It loads the cleaned CQI reviews (arabica + robusta), harmonises the robusta column "
       "names, checks missing values and score distributions, verifies whether the total score is the sum of the cupping "
       "attributes (a leakage risk), and looks at the Colombian subset."),
    SETUP,
    code("from src.data.loaders import load_s3, S3_G1, S3_G2_NUM, S3_G2_CAT, S3_G3_NUM, S3_G3_CAT\n"
         "df = load_s3()\nprint(df.shape); print(df.Species.value_counts().to_dict())\n"
         "df['Total.Cup.Points'].describe().round(2)"),
    code("resid = df['Total.Cup.Points'] - df[S3_G1].sum(axis=1)\n"
         "print('share of rows where total == sum of the 10 attributes (±0.01):', (resid.abs() < 0.011).mean().round(3))\n"
         "print('max |difference|:', resid.abs().max())"),
    md("**Leakage.** The total cup score *is* the sum of the 10 sensory attributes (G1) for 95 % of rows. Predicting a "
       "class derived from the total score using G1 is therefore trivial, so G1 is only reported as an explicitly labelled "
       "upper bound. The real fusion study uses the non-sensory groups G2 (bean/physical) and G3 (origin)."),
    code("cols = S3_G2_NUM + S3_G2_CAT + S3_G3_NUM + S3_G3_CAT\n"
         "pd.DataFrame({'missing_frac': df[cols].isna().mean().round(3), 'n_unique': df[cols].nunique()})"),
    code("t1, t2 = df['Total.Cup.Points'].quantile([1/3, 2/3])\nprint('tertile thresholds:', round(t1, 2), round(t2, 2))\n"
         "col = df[df['Country.of.Origin'] == 'Colombia']\nprint('Colombian rows:', len(col))\n"
         "fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))\n"
         "ax[0].hist(df['Total.Cup.Points'], bins=60, color='grey'); ax[0].hist(col['Total.Cup.Points'], bins=30, color='#b2182b')\n"
         "[ax[0].axvline(t, ls='--', c='k') for t in (t1, t2)]; ax[0].set_xlim(70, 92); ax[0].set_title('Total Cup Points (red: Colombia)')\n"
         "df['Country.of.Origin'].value_counts().head(12)[::-1].plot.barh(ax=ax[1]); ax[1].set_title('Reviews per country (top 12)')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'eda_s3_scores.png', dpi=130); plt.show()"),
    md("**Narrow classes.** The middle tertile (AQ) spans only 81.67–83.17 points, i.e. 1.5 points on a 100-point scale. "
       "Worku et al. (2016) report a mean absolute difference of about 3 points between cupping centres, so the class "
       "boundaries are within the measurement noise of cupping itself. That limits any model's accuracy."),
    code("import seaborn as sns\nc = df[S3_G1 + ['Total.Cup.Points'] + S3_G2_NUM + S3_G3_NUM].corr()\n"
         "plt.figure(figsize=(8, 6.5)); sns.heatmap(c, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=False)\n"
         "plt.title('Correlations: sensory (G1), physical (G2), altitude (G3)'); plt.tight_layout()\n"
         "plt.savefig(FIG / 'eda_s3_correlations.png', dpi=130); plt.show()\n"
         "c['Total.Cup.Points'].drop('Total.Cup.Points').sort_values().round(2)"),
    md("**Reading the correlations.** Every sensory attribute correlates strongly with the total (as expected, since it is "
       "their sum). The non-sensory variables (moisture, defect counts, altitude) are only weakly correlated with the total. "
       "So any model using G2 + G3 alone will perform far below the G1 upper bound."),
]

for name, cells in [("01_eda_s1", s1), ("01_eda_s2", s2), ("01_eda_s3", s3)]:
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
print("written")
