"""Phase 3 - reproduction of the Rodríguez et al. (2010) pipeline on the public S1 data (RQ1).

Pipeline: R (kOhm) -> G = 1/R -> static parameter per sensor (dG = Gmax - Gmin, or the
relative variant) -> mean-centring -> PCA (visualisation) and MLP (10 hidden units,
backprop with adaptive learning rate) -> leave-one-out.

Deviations from the paper (documented in the article):
  * softmax output with 3 units instead of the paper's single output neuron;
  * MATLAB 7.5 -> scikit-learn MLPClassifier;
  * we also report repeated stratified 5-fold CV and several MLP seeds.
Run: python -m src.experiments.s1_baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loaders import S1_CLASSES, S1_SENSORS, load_s1
from src.evaluation.cv import RESULTS, confusion, loo_splits, repeated_skf_splits, run_cv
from src.features.enose import delta_g
from src.utils.plotting import plot_confusion, plot_pca

SEEDS = [0, 1, 2]  # LOO repeated with 3 MLP initialisations


def mlp(solver, seed):
    if solver == "sgd":  # backprop with adaptive learning rate (closest to the paper)
        return MLPClassifier((10,), solver="sgd", learning_rate="adaptive", learning_rate_init=0.05,
                             momentum=0.9, max_iter=5000, random_state=seed)
    return MLPClassifier((10,), solver="adam", learning_rate_init=0.01, max_iter=5000, random_state=seed)


def variants():
    for feat in ["abs", "rel"]:
        for scaling in ["none", "center", "std"]:
            for solver in ["sgd", "adam"]:
                yield f"dG-{feat}|{scaling}|{solver}", feat, scaling, solver


def main():
    X, y, ids = load_s1()
    F = {"abs": delta_g(X, "abs"), "rel": delta_g(X, "rel")}

    # ---------------- PCA (descriptive, whole data set, as in the paper's Figs. 7 and 9)
    pca_rows = []
    for feat in ["abs", "rel"]:
        for scaling in ["center", "std"]:
            Z = StandardScaler(with_std=(scaling == "std")).fit_transform(F[feat])
            p = PCA().fit(Z)
            ev = p.explained_variance_ratio_
            pca_rows.append({"feature": f"dG-{feat}", "scaling": scaling, "PC1": ev[0], "PC2": ev[1],
                             "PC1+PC2": ev[:2].sum()})
            plot_pca(p.transform(Z)[:, :2], y, S1_CLASSES, ev,
                     RESULTS / "figures" / f"baseline_pca_dG-{feat}_{scaling}.png",
                     title=f"S1 PCA, dG-{feat}, {'mean-centred' if scaling == 'center' else 'autoscaled'}")
    pca_df = pd.DataFrame(pca_rows)
    pca_df.to_csv(RESULTS / "baseline_pca.csv", index=False)
    print(pca_df.to_string(index=False))

    # ---------------- MLP, LOO (repeats = MLP seeds) and repeated stratified 5-fold
    def fit_predict(Xtr, ytr, Xte, seed):
        out = {}
        for name, feat, scaling, solver in variants():
            pipe = make_pipeline(StandardScaler(with_mean=(scaling != "none"), with_std=(scaling == "std")),
                                 mlp(solver, seed))
            pipe.fit(Xtr[feat], ytr)
            out[name] = pipe.predict_proba(Xte[feat])
        return out

    Xd = _Dict(F)
    loo = [("loo", s, f) for s in SEEDS for (_, _, f) in loo_splits(y)]
    rskf = repeated_skf_splits(y, 5, 10, seed=0)
    res = run_cv("baseline", {"feature": "dG", "model": "MLP(10)", "seeds": SEEDS}, Xd, y,
                 fit_predict, loo + rskf, S1_CLASSES, ids)
    res.summary.to_csv(RESULTS / "baseline.csv", index=False)
    best = res.summary[res.summary.scheme == "loo"].sort_values("accuracy_mean").iloc[-1].variant
    for scheme in ["loo", "rskf5"]:
        cm = confusion(res.predictions[res.predictions.repeat == 0], best, scheme, [0, 1, 2])
        plot_confusion(cm, S1_CLASSES, RESULTS / "figures" / f"baseline_confusion_{scheme}.png",
                       title=f"Baseline {best}\n{scheme}, repeat 0")


class _Dict(dict):
    """Dict of aligned arrays that supports row indexing (so run_cv can split it)."""

    def __getitem__(self, k):
        if isinstance(k, str):
            return dict.__getitem__(self, k)
        return _Dict({n: v[k] for n, v in self.items()})

    def __len__(self):
        return len(next(iter(self.values())))


if __name__ == "__main__":
    main()
