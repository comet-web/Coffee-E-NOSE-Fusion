"""Experiment B - fusion of S1 (coffee e-nose) with S2 (SmellNet-Base) (RQ3) + AB8, AB2/AB3 on B.

Harmonisation (each step can be switched off for AB8):
  polarity : S1 resistance -> conductance G = 1/R, so that, as for S2's raw readings, the
             signal rises when gas is present ('R' keeps resistance).
  length   : both 1 Hz; use the first 300 s of every S2 session ('full' keeps whole sessions).
  norm     : per-measurement, per-channel z-score ('z'); alternatives: 'ratio' x / max(x0, 1)
             with x0 = mean of the first 20 s, or 'none'.
Channel mismatch (8 vs 6, no one-to-one correspondence) -> channel-agnostic designs:
  B1 low-level : shared univariate 1D-CNN encoder per channel, mean+max pooled across channels.
  B2 mid-level : 15 per-channel features, aggregated across channels (mean/std/max/min) -> 60-d.
  B3 high-level: combine coffee-class probabilities of an S1-only model and a pooled S1+S2 model.
Joint label space = 3 coffee classes + 50 SmellNet substances. Evaluation ONLY on held-out S1
measurements (argmax over the 3 coffee classes). S1 folds are fixed first; S2 sessions (days
1-5) only ever enter training; S2 day 6 is a secondary check for the pooled models.
Run: python -m src.experiments.expB_s1_s2 [b2|b1|ablations|collect|all]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loaders import S1_CLASSES, load_s1, load_s2_sessions
from src.evaluation.cv import RESULTS, per_repeat_scores, repeated_skf_splits, run_cv, wilcoxon_paired
from src.features.enose import channel_agnostic, sensor_features
from src.models.torch_models import TorchSeqClassifier, _predict, _train, set_seed, ChannelAgnosticNet
from src.models.zoo import make_clf
from src.utils.plotting import bar_with_err

import torch

KC = 3  # coffee classes


# ------------------------------------------------------------------ harmonisation --
def harmonise(series, norm="z", base_s=20):
    """series: list of (T, C) arrays -> list of normalised arrays (per measurement, per channel)."""
    out = []
    for x in series:
        x = np.asarray(x, dtype=np.float64)
        if norm == "z":
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        elif norm == "ratio":
            x = x / np.maximum(x[:base_s].mean(0), 1.0)
        elif norm != "none":
            raise ValueError(norm)
        out.append(x)
    return out


def build(polarity="G", length="300", norm="z", dur=300, s2_frac=1.0, seed=0, diff_p=0):
    """Return harmonised S1 list, y1, ids, S2 train list, y2 (3..52), S2 day-6 list, y2 test."""
    X1, y1, ids = load_s1()
    s1 = [(1.0 / x if polarity == "G" else x)[:dur] for x in X1]
    sess = load_s2_sessions()
    subs = sorted({s["substance"] for s in sess})
    cmap = {c: KC + i for i, c in enumerate(subs)}
    tr = [s for s in sess if s["day"] <= 5]
    te = [s for s in sess if s["day"] == 6]
    if s2_frac < 1.0:  # random subset of S2 training sessions (AB8)
        rng = np.random.default_rng(seed)
        tr = [tr[i] for i in sorted(rng.choice(len(tr), int(round(s2_frac * len(tr))), replace=False))]
    cut = (lambda a: a[:dur]) if length == "300" else (lambda a: a)
    s2tr = [cut(s["data"]) for s in tr]
    s2te = [cut(s["data"]) for s in te]
    if diff_p:
        s1, s2tr, s2te = ([x[diff_p:] - x[:-diff_p] for x in L] for L in (s1, s2tr, s2te))
    return (harmonise(s1, norm), y1, ids, harmonise(s2tr, norm), np.array([cmap[s["substance"]] for s in tr]),
            harmonise(s2te, norm), np.array([cmap[s["substance"]] for s in te]))


def feats(series):
    """Channel-agnostic feature vector per measurement (works for 8 or 6 channels)."""
    return np.stack([channel_agnostic(sensor_features(x[None]))[0] for x in series])


# ------------------------------------------------------------------ B2 / B3 --------
def b_models():
    return {
        "LR": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
        "RF": lambda s: make_clf("RF", s),
        "MLP64": lambda s: make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=1e-3, max_iter=3000, random_state=s)),
    }


def coffee_proba(model, X):
    """Probabilities restricted to the coffee classes (columns 0..2), renormalised."""
    P = model.predict_proba(X)
    cols = [list(model.classes_).index(c) for c in range(KC)]
    P = P[:, cols]
    return P / P.sum(1, keepdims=True)


def b2_factory(F2, y2, models=None, b3=True):
    models = models or b_models()

    def fp(Ftr, ytr, Fte, seed):
        out = {}
        for name, mk in models.items():
            m1 = mk(seed).fit(Ftr, ytr)
            mp = mk(seed).fit(np.r_[Ftr, F2], np.r_[ytr, y2])
            P1, Pp = coffee_proba(m1, Fte), coffee_proba(mp, Fte)
            out[f"S1only|{name}"] = P1
            out[f"B2|{name}"] = Pp
            if b3:
                # weights / stacking from inner CV on the S1 training fold (S2 always in training)
                skf = StratifiedKFold(5, shuffle=True, random_state=seed)
                o1, op = np.zeros((len(ytr), KC)), np.zeros((len(ytr), KC))
                for a, b in skf.split(Ftr, ytr):
                    o1[b] = coffee_proba(mk(seed).fit(Ftr[a], ytr[a]), Ftr[b])
                    op[b] = coffee_proba(mk(seed).fit(np.r_[Ftr[a], F2], np.r_[ytr[a], y2]), Ftr[b])
                acc = np.array([(o1.argmax(1) == ytr).mean(), (op.argmax(1) == ytr).mean()])
                w = np.clip(acc - 1 / KC, 1e-3, None)
                w = w / w.sum()
                meta = LogisticRegression(max_iter=2000).fit(np.log(np.c_[o1, op] + 1e-6), ytr)
                out[f"B3|{name}|mean"] = (P1 + Pp) / 2
                out[f"B3|{name}|weighted"] = w[0] * P1 + w[1] * Pp
                out[f"B3|{name}|stacking"] = meta.predict_proba(np.log(np.c_[P1, Pp] + 1e-6))
        return out
    return fp


def s2_secondary(F1, y1, F2, y2, F2te, y2te, models=None):
    """Pooled model trained on all S1 + S2 days 1-5, scored on S2 day 6 (50-way)."""
    rows = []
    for name, mk in (models or b_models()).items():
        m = mk(0).fit(np.r_[F1, F2], np.r_[y1, y2])
        P = m.predict_proba(F2te)
        cls = np.array(m.classes_)
        sub = cls >= KC
        rows.append({"model": name, "s2_day6_acc_50way": float((cls[sub][P[:, sub].argmax(1)] == y2te).mean())})
    return rows


def run_b2(tag="expB_b2", n_repeats=10, b3=True, **hcfg):
    s1, y1, ids, s2, y2, s2te, y2te = build(**hcfg)
    F1, F2 = feats(s1), feats(s2)
    res = run_cv(tag, {"part": "B2/B3", **{k: str(v) for k, v in hcfg.items()}, "n_s2": len(y2)}, F1, y1,
                 b2_factory(F2, y2, b3=b3), repeated_skf_splits(y1, 5, n_repeats, seed=0), S1_CLASSES, ids,
                 verbose=(tag == "expB_b2"))
    return res, (F1, y1, F2, y2, feats(s2te), y2te)


# ------------------------------------------------------------------ B1 -------------
def b1_factory(s2, y2, epochs=150, width=16):
    """Channel-agnostic CNN: S1-only vs pooled S1+S2 (+ B3 mean of the two)."""
    X2 = torch.tensor(np.transpose(np.stack(s2), (0, 2, 1)), dtype=torch.float32)  # common 300 s length
    y2t = torch.tensor(y2)

    class Cfg:
        pass

    def fp(Xtr, ytr, Xte, seed):
        Xtr_t = torch.tensor(np.transpose(Xtr, (0, 2, 1)), dtype=torch.float32)
        Xte_t = torch.tensor(np.transpose(Xte, (0, 2, 1)), dtype=torch.float32)
        cfg = Cfg()
        cfg.seed, cfg.lr, cfg.weight_decay, cfg.batch_size, cfg.epochs = seed, 3e-3, 1e-3, 16, epochs
        # Unweighted CE and 150 epochs: chosen from TRAINING-set fit only (balanced weights over
        # the 53-class joint space and 60 epochs under-fitted the coffee classes; docs/data_notes.md)
        cfg.class_weight = None
        set_seed(seed)
        m1 = ChannelAgnosticNet(KC, width)
        _train(m1, [(Xtr_t, torch.tensor(ytr))], cfg)
        P1 = _predict(m1, [Xte_t])
        set_seed(seed)
        mp = ChannelAgnosticNet(KC + 50, width)
        _train(mp, [(Xtr_t, torch.tensor(ytr)), (X2, y2t)], cfg)
        Pp = _predict(mp, [Xte_t])[:, :KC]
        Pp = Pp / Pp.sum(1, keepdims=True)
        return {"S1only|CA-CNN": P1, "B1|CA-CNN": Pp, "B3|CA-CNN|mean": (P1 + Pp) / 2}
    return fp


def run_b1(n_repeats=5):
    s1, y1, ids, s2, y2, s2te, y2te = build()
    X1 = np.stack(s1)
    return run_cv("expB_b1", {"part": "B1", "model": "channel-agnostic CNN", "epochs": 150, "n_s2": len(y2)},
                  X1, y1, b1_factory(s2, y2), repeated_skf_splits(y1, 5, n_repeats, seed=0), S1_CLASSES, ids)


# ------------------------------------------------------------------ AB8 ------------
def run_ablations():
    rows = []
    settings = {"full": {}}
    settings |= {f"s2_frac_{int(f * 100)}": {"s2_frac": f} for f in [0.1, 0.25, 0.5]}
    settings |= {"no_polarity(R)": {"polarity": "R"}, "norm_ratio": {"norm": "ratio"},
                 "norm_none": {"norm": "none"}, "no_common_length": {"length": "full"},
                 "diff25": {"diff_p": 25}}
    settings |= {f"first_{d}s": {"dur": d} for d in [60, 120, 180]}
    for label, cfg in settings.items():
        res, _ = run_b2(tag=f"AB8_{label}", n_repeats=10, b3=False, **cfg)
        s = res.summary.assign(setting=label)
        rows.append(s)
        print("AB8", label, s.set_index("variant").accuracy_mean.round(3).to_dict(), flush=True)
    df = pd.concat(rows, ignore_index=True)
    full = df[df.setting == "full"].set_index("variant").accuracy_mean
    df["delta_vs_full"] = df.accuracy_mean - df.variant.map(full)
    pooled_gain = df.pivot_table(index="setting", columns="variant", values="accuracy_mean")
    for m in b_models():
        pooled_gain[f"gain_{m}"] = pooled_gain[f"B2|{m}"] - pooled_gain[f"S1only|{m}"]
    df.to_csv(RESULTS / "ablations" / "AB8_s2_fusion.csv", index=False)
    pooled_gain.reset_index().to_csv(RESULTS / "ablations" / "AB8_s2_gain_by_setting.csv", index=False)
    sub = df[df.variant.str.startswith("B2")]
    bar_with_err(sub, "setting", "accuracy_mean", "accuracy_std", RESULTS / "figures" / "AB8_s2_fusion.png",
                 hue="variant", title="AB8: S1+S2 pooled (B2) under harmonisation / S2-fraction ablations",
                 ref=28 / 58, rotate=60)


def collect():
    parts, preds = [], []
    for e in ["expB_b2", "expB_b1"]:
        d = RESULTS / "runs" / e
        if (d / "summary.csv").exists():
            parts.append(pd.read_csv(d / "summary.csv"))
            preds.append(pd.read_csv(d / "predictions.csv.gz").assign(exp=e))
    s = pd.concat(parts, ignore_index=True)
    s["level"] = s.variant.str.split("|").str[0].map({"S1only": "none (S1 only)", "B1": "low", "B2": "mid", "B3": "high"})
    # paired tests: each fused variant vs its S1-only counterpart on the same folds
    rows = []
    for p in preds:
        exp = p.exp.iloc[0]
        wide = per_repeat_scores(p, [0, 1, 2])
        for v in wide.columns:
            if v.startswith("S1only"):
                continue
            base = "S1only|" + v.split("|")[1]
            if base in wide:
                _, pv = wilcoxon_paired(wide[v], wide[base])
                rows.append({"exp": exp, "variant": v, "vs": base, "acc": wide[v].mean(), "acc_vs": wide[base].mean(),
                             "diff": wide[v].mean() - wide[base].mean(), "wilcoxon_p": pv})
    sig = pd.DataFrame(rows)
    s.to_csv(RESULTS / "expB_s1_s2_fusion.csv", index=False)
    sig.to_csv(RESULTS / "expB_significance.csv", index=False)
    print(s[["exp", "variant", "accuracy_mean", "accuracy_std", "macro_f1_mean"]].to_string(index=False))
    print(sig.to_string(index=False))
    bar_with_err(s.sort_values("variant"), "variant", "accuracy_mean", "accuracy_std",
                 RESULTS / "figures" / "expB_s1_s2_fusion.png", hue="level",
                 title="Experiment B: S1 with vs without S2 (S1 test folds only)", ref=28 / 58, rotate=60)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("b2", "all"):
        res, (F1, y1, F2, y2, F2te, y2te) = run_b2()
        sec = pd.DataFrame(s2_secondary(F1, y1, F2, y2, F2te, y2te))
        sec.to_csv(RESULTS / "expB_s2_secondary_check.csv", index=False)
        print(sec)
    if part in ("b1", "all"):
        run_b1()
    if part in ("ablations", "all"):
        run_ablations()
    if part in ("collect", "b2", "b1", "all"):
        collect()
