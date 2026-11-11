"""Signal representations and per-sensor feature extraction for gas-sensor time series.

Every function here is *per measurement* (it only looks at one recording), so it can be
applied before the train/test split without leaking information across samples.
Anything fitted on data (scaling, PCA, ...) lives in sklearn pipelines instead.
"""
from __future__ import annotations

import numpy as np

BASELINE_S = 20  # first 20 s are pre-exposure baseline in S1 (see docs/data_notes.md)


# ----------------------------------------------------------- representations ------
def to_representation(X: np.ndarray, rep: str = "G", baseline_s: int = BASELINE_S) -> np.ndarray:
    """Convert raw resistance X (N, T, C) in kOhm to a signal representation.

    rep:
      'R'      raw resistance (kOhm)
      'G'      conductance G = 1/R (mS)
      'RR0'    baseline-normalised resistance R/R0, R0 = mean of first `baseline_s` s
      'GG0'    baseline-normalised conductance G/G0
      'z'      per-measurement, per-channel z-score of R
    """
    X = np.asarray(X, dtype=np.float64)
    if rep == "R":
        return X
    if rep == "G":
        return 1.0 / X
    if rep == "RR0":
        return X / X[:, :baseline_s].mean(1, keepdims=True)
    if rep == "GG0":
        G = 1.0 / X
        return G / G[:, :baseline_s].mean(1, keepdims=True)
    if rep == "z":
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    raise ValueError(rep)


def temporal_diff(X: np.ndarray, p: int) -> np.ndarray:
    """SmellNet-style differencing x_t - x_{t-p} along time (drops the first p steps)."""
    if p <= 0:
        return X
    return X[:, p:] - X[:, :-p]


# ----------------------------------------------------------- baseline feature -----
def delta_g(R: np.ndarray, variant: str = "abs") -> np.ndarray:
    """Rodríguez et al. (2010) 'static parameter': variation between max and min conductance.

    The paper does not give the exact formula. Primary: dG = Gmax - Gmin.
    Variant 'rel': (Gmax - Gmin) / Gmin. Returns (N, C).
    """
    G = 1.0 / np.asarray(R, dtype=np.float64)
    gmax, gmin = G.max(1), G.min(1)
    if variant == "abs":
        return gmax - gmin
    if variant == "rel":
        return (gmax - gmin) / gmin
    raise ValueError(variant)


# ----------------------------------------------------------- per-sensor features --
FEATURE_NAMES = ["baseline", "max", "min", "dG", "rel_change", "mean", "std", "auc",
                 "max_slope", "min_slope", "t_peak", "final", "fft1", "fft2", "fft3"]
FEATURE_GROUPS = {  # used by the feature-type ablation (AB4)
    "baseline": ["baseline"],
    "extrema": ["max", "min", "dG"],
    "relative": ["rel_change"],
    "moments": ["mean", "std"],
    "auc": ["auc"],
    "slope": ["max_slope", "min_slope"],
    "timing": ["t_peak"],
    "final": ["final"],
    "fft": ["fft1", "fft2", "fft3"],
}


def sensor_features(S: np.ndarray, baseline_s: int = BASELINE_S) -> np.ndarray:
    """Per-channel feature set. S is (N, T, C) in any representation (default conductance).

    Returns (N, C, F) with F = len(FEATURE_NAMES):
      baseline   mean of the first `baseline_s` samples
      max, min   extrema over the recording
      dG         max - min (the paper's static parameter when S is conductance)
      rel_change (peak deviation from baseline) / |baseline|
      mean, std  over the whole recording
      auc        area under (S - baseline), trapezoid rule, 1 s steps
      max_slope, min_slope  extrema of the first derivative
      t_peak     time (s) of the largest absolute deviation from baseline
      final      mean of the last 10 samples
      fft1..3    magnitudes of the 3 lowest non-DC FFT bins of (S - baseline), normalised by T
    """
    S = np.asarray(S, dtype=np.float64)
    N, T, C = S.shape
    base = S[:, :baseline_s].mean(1)                      # (N, C)
    dev = S - base[:, None, :]
    peak_idx = np.abs(dev).argmax(1)                      # (N, C)
    peak_dev = np.take_along_axis(dev, peak_idx[:, None, :], 1)[:, 0]
    d1 = np.diff(S, axis=1)
    spec = np.abs(np.fft.rfft(dev, axis=1)) / T
    feats = np.stack([
        base,
        S.max(1),
        S.min(1),
        S.max(1) - S.min(1),
        peak_dev / (np.abs(base) + 1e-12),
        S.mean(1),
        S.std(1),
        np.trapezoid(dev, axis=1),
        d1.max(1),
        d1.min(1),
        peak_idx.astype(float),
        S[:, -10:].mean(1),
        spec[:, 1], spec[:, 2], spec[:, 3],
    ], axis=-1)
    return feats  # (N, C, F)


def flat_feature_names(channels, names=FEATURE_NAMES):
    return [f"{c}:{n}" for c in channels for n in names]


def channel_agnostic(feats: np.ndarray) -> np.ndarray:
    """Aggregate per-channel features (N, C, F) across channels -> (N, 4F): mean, std, max, min.

    The output length does not depend on C, so S1 (8 channels) and S2 (6 channels)
    end up in the same feature space (used by Experiment B, mid-level fusion).
    """
    return np.concatenate([feats.mean(1), feats.std(1), feats.max(1), feats.min(1)], axis=1)
