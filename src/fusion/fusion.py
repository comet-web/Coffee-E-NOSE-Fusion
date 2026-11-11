"""Low-, mid- and high-level fusion building blocks.

Low-level  : concatenate raw (minimally preprocessed) data from all sources -> one model.
Mid-level  : extract features per source -> concatenate -> one model.
High-level : one model per source -> combine decisions (vote / mean / weighted / stacking).

Low- and mid-level fusion are just sklearn pipelines over concatenated inputs; they are
built in the experiment scripts. This module implements the high-level combiners, whose
weights and meta-learner are fitted with an inner CV *inside* the outer training fold.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

COMBINERS = ["vote", "mean", "weighted", "stacking"]


class HighLevelFusion:
    """Decision-level fusion over several sources.

    sources: list of (name, selector, estimator) where selector(X) -> that source's input
             and estimator is an unfitted sklearn classifier with predict_proba.
    """

    def __init__(self, sources: Sequence[tuple[str, Callable, object]], n_classes: int,
                 inner_splits: int = 5, seed: int = 0, meta_C: float = 1.0):
        self.sources = sources
        self.K = n_classes
        self.inner_splits = inner_splits
        self.seed = seed
        self.meta_C = meta_C

    def fit(self, X, y):
        y = np.asarray(y)
        skf = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(y), len(self.sources), self.K))
        for tr, va in skf.split(np.zeros(len(y)), y):
            for s, (_, sel, est) in enumerate(self.sources):
                m = clone(est).fit(_rows(sel(X), tr), y[tr])
                oof[va, s] = _full_proba(m, _rows(sel(X), va), self.K)
        self.inner_acc_ = (oof.argmax(2) == y[:, None]).mean(0)          # (S,)
        w = np.clip(self.inner_acc_ - 1.0 / self.K, 1e-3, None)          # above-chance accuracy
        self.weights_ = w / w.sum()
        self.meta_ = LogisticRegression(C=self.meta_C, max_iter=2000).fit(
            np.log(oof.reshape(len(y), -1) + 1e-6), y)
        self.models_ = [clone(est).fit(sel(X), y) for _, sel, est in self.sources]
        return self

    def source_proba(self, X):
        return np.stack([_full_proba(m, sel(X), self.K)
                         for m, (_, sel, _) in zip(self.models_, self.sources)], 1)  # (N,S,K)

    def predict_proba_all(self, X):
        P = self.source_proba(X)
        mean = P.mean(1)
        votes = np.zeros_like(mean)
        for s in range(P.shape[1]):
            votes[np.arange(len(P)), P[:, s].argmax(1)] += 1
        vote = votes / P.shape[1] + 1e-3 * mean  # ties broken by mean probability
        weighted = np.einsum("nsk,s->nk", P, self.weights_)
        stack = _full_proba(self.meta_, np.log(P.reshape(len(P), -1) + 1e-6), self.K)
        return {"vote": vote, "mean": mean, "weighted": weighted, "stacking": stack}


def _full_proba(model, X, K):
    """predict_proba padded to all K classes (a training fold may miss a rare class)."""
    P = model.predict_proba(X)
    if P.shape[1] == K:
        return P
    out = np.zeros((P.shape[0], K))
    out[:, model.classes_] = P
    return out


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]
