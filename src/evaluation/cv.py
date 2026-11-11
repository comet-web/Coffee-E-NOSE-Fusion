"""Cross-validation runner, metrics, logging and significance tests.

Design: an experiment supplies `fit_predict(Xtr, ytr, Xte, rng_seed) -> {variant: proba}`.
Everything data-dependent (scalers, PCA, feature selection, tuning, meta-learners) must
happen inside fit_predict using only the training part, so the protocol is leak-free by
construction. One call can return several variants (e.g. several combiners fitted on the
same base models) so they are compared on exactly the same folds.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


# ----------------------------------------------------------------- splits ---------
def loo_splits(y):
    return [("loo", 0, list(LeaveOneOut().split(np.zeros(len(y)), y)))]


def repeated_skf_splits(y, n_splits=5, n_repeats=10, seed=0):
    """List of (scheme, repeat, folds). Repeat r uses seed + r."""
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        out.append((f"rskf{n_splits}", r, list(skf.split(np.zeros(len(y)), y))))
    return out


# ----------------------------------------------------------------- metrics --------
def metrics(y_true, y_pred, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def per_class(y_true, y_pred, labels, names):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f, "support": s})


# ----------------------------------------------------------------- runner ---------
@dataclass
class CVResult:
    exp: str
    config: dict
    predictions: pd.DataFrame            # one row per (variant, scheme, repeat, sample)
    summary: pd.DataFrame = field(default=None)  # one row per (variant, scheme)


def run_cv(exp: str, config: dict, X, y, fit_predict: Callable, split_sets, class_names,
           ids=None, save=True, verbose=True) -> CVResult:
    """Run `fit_predict` over all split sets and return/save predictions + summary.

    X may be any indexable object (ndarray or tuple of arrays); indexing uses `_take`.
    """
    labels = list(range(len(class_names)))
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    done = _load_completed(exp, config)
    if done is not None:  # identical config already finished: reuse instead of recomputing
        if verbose:
            print(f"[{exp}] already complete with this config; loaded saved results "
                  f"(delete results/runs/{exp}/ to recompute)")
        return done
    rows = []
    t0 = time.time()
    # Fold-level checkpoints: each fold's outputs are saved as soon as they are computed, so an
    # interrupted run resumes where it stopped. The cache is keyed by the experiment name and a
    # hash of the config; it is reused only if the fold's test indices match exactly.
    ckpt = RESULTS / "runs" / exp / "folds" / _config_hash(config)
    ckpt.mkdir(parents=True, exist_ok=True)
    n_done = n_cached = 0
    n_total = sum(len(f) for _, _, f in split_sets)
    for scheme, rep, folds in split_sets:
        for k, (tr, te) in enumerate(folds):
            seed = 1000 * rep + k
            f = ckpt / f"{scheme}_r{rep}_k{k}.pkl"
            out = None
            if f.exists():
                with f.open("rb") as fh:
                    saved = pickle.load(fh)
                if np.array_equal(saved["te"], te):
                    out = saved["out"]
                    n_cached += 1
            if out is None:
                out = fit_predict(_take(X, tr), y[tr], _take(X, te), seed)
                tmp = f.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    pickle.dump({"te": np.asarray(te), "out": out}, fh)
                os.replace(tmp, f)  # atomic: a kill never leaves a half-written checkpoint
            n_done += 1
            if verbose and n_done % 10 == 0:
                print(f"[{exp}] fold {n_done}/{n_total} ({n_cached} from checkpoint) "
                      f"{time.time() - t0:.0f}s", flush=True)
            for variant, P in out.items():
                P = np.asarray(P)
                pred = P.argmax(1)
                for j, i in enumerate(te):
                    rows.append({"variant": variant, "scheme": scheme, "repeat": rep, "fold": k,
                                 "sample": ids[i], "y_true": int(y[i]), "y_pred": int(pred[j]),
                                 **{f"p_{c}": float(P[j, ci]) for ci, c in enumerate(class_names)}})
    pred_df = pd.DataFrame(rows)
    summary = summarise(pred_df, labels)
    summary.insert(0, "exp", exp)
    for k_, v_ in config.items():
        if k_ not in summary.columns and not isinstance(v_, (dict, list)):
            summary[k_] = v_
    res = CVResult(exp, config, pred_df, summary)
    if save:
        save_result(res, runtime_s=time.time() - t0)
    if verbose:
        cols = ["variant", "scheme", "accuracy_mean", "accuracy_std", "macro_f1_mean"]
        print(f"[{exp}] {time.time() - t0:.1f}s")
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return res


def summarise(pred_df: pd.DataFrame, labels) -> pd.DataFrame:
    """Metrics per repeat (pooled over folds), then mean/std over repeats."""
    out = []
    for (variant, scheme), g in pred_df.groupby(["variant", "scheme"], sort=False):
        per_rep = pd.DataFrame([metrics(gg.y_true, gg.y_pred, labels) | {"repeat": r}
                                for r, gg in g.groupby("repeat")])
        row = {"variant": variant, "scheme": scheme, "n_repeats": len(per_rep)}
        for m in ["accuracy", "balanced_accuracy", "macro_f1"]:
            row[f"{m}_mean"] = per_rep[m].mean()
            row[f"{m}_std"] = per_rep[m].std(ddof=1) if len(per_rep) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def per_repeat_scores(pred_df: pd.DataFrame, labels, metric="accuracy") -> pd.DataFrame:
    """Wide table: rows = (scheme, repeat), columns = variant, values = metric."""
    rows = []
    for (variant, scheme, rep), g in pred_df.groupby(["variant", "scheme", "repeat"], sort=False):
        rows.append({"variant": variant, "scheme": scheme, "repeat": rep,
                     metric: metrics(g.y_true, g.y_pred, labels)[metric]})
    return pd.DataFrame(rows).pivot_table(index=["scheme", "repeat"], columns="variant", values=metric)


def save_result(res: CVResult, runtime_s: float):
    d = RESULTS / "runs" / res.exp
    d.mkdir(parents=True, exist_ok=True)
    res.predictions.to_csv(d / "predictions.csv.gz", index=False, compression="gzip")
    res.summary.to_csv(d / "summary.csv", index=False)
    meta = {"exp": res.exp, "config": res.config, "runtime_s": round(runtime_s, 2),
            "environment": environment()}
    (d / "config.json").write_text(json.dumps(meta, indent=1, default=str))


def environment():
    import sklearn, scipy
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "scipy": scipy.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    return env


# ----------------------------------------------------------------- significance ---
def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank on paired per-repeat scores. Returns (stat, p) or (nan, nan)."""
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return np.nan, 1.0
    try:
        r = stats.wilcoxon(a, b, zero_method="zsplit")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return np.nan, np.nan


def mcnemar(y, pa, pb):
    """Exact McNemar test on two sets of predictions for the same samples."""
    a_ok, b_ok = (pa == y), (pb == y)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return n01, n10, p


def confusion(pred_df, variant, scheme, labels):
    g = pred_df[(pred_df.variant == variant) & (pred_df.scheme == scheme)]
    return confusion_matrix(g.y_true, g.y_pred, labels=labels)


def _load_completed(exp: str, config: dict):
    """Return the saved CVResult if `exp` already finished with exactly this config, else None."""
    d = RESULTS / "runs" / exp
    files = [d / "config.json", d / "summary.csv", d / "predictions.csv.gz"]
    if not all(f.exists() for f in files):
        return None
    saved = json.loads(files[0].read_text()).get("config")
    if json.dumps(saved, sort_keys=True, default=str) != json.dumps(config, sort_keys=True, default=str):
        return None
    return CVResult(exp, config, pd.read_csv(files[2]), pd.read_csv(files[1]))


def _config_hash(config: dict) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _take(X, idx):
    if isinstance(X, tuple):
        return tuple(_take(x, idx) for x in X)
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return X[idx]
