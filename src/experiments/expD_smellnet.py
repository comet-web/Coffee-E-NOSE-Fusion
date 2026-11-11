"""Experiment D - sensor + GC-MS fusion inside SmellNet-Base (RQ6), leakage-free.

Reuses SmellNet's own model classes (models/models.py) and contrastive loss (models/loss.py)
from the cloned repository, but with a validation protocol: train on days 1-4, select the
learning rate / lambda / checkpoint on day 5, test once on day 6 (official test day).

Systems (test-time input is ALWAYS sensor data only):
  D0 sensor-only classifier (cross-entropy)
  D1 cross-modal contrastive training (SmellNet's method); prediction = most similar of
     the 50 GC-MS class prototypes (all classes, never only the true one)
  D2 hybrid: CE + lambda * contrastive, prediction from the classifier head
  D3 decision-level ensemble: alpha * softmax(D0 logits) + (1-alpha) * softmax(D1 sims / tau)
  DX leakage demonstration ONLY (invalid): GC-MS of the true class given at test time.

Run examples:
  python -m src.experiments.expD_smellnet sweep --models mlp cnn --stage lr
  python -m src.experiments.expD_smellnet collect
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data.loaders import RAW, S2_CHANNELS, load_s2_gcms, load_s2_sessions
from src.evaluation.cv import RESULTS, environment

SN = RAW / "s2" / "SmellNet-code" / "models"
OUT = RESULTS / "runs" / "expD"
LRS = [3e-4, 1e-3, 3e-3]
LAMBDAS = [0.1, 0.5, 1.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 7, 123]
TAU = 0.07
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"smellnet_{name}", SN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snm = _load("models")
snl = _load("loss")


# ------------------------------------------------------------------ data -----------
_CACHE = {}


def get_data(w: int, p: int, gcms: str = "spec", val_day: int = 5):
    """Windows for train (days 1-4), val (day 5), test (day 6), standardised with train stats.

    Preprocessing follows SmellNet's run.py: subtract the first row of every session,
    optional differencing x_t - x_{t-p}, sliding windows (stride w/2) cut AFTER splitting.
    """
    key = (w, p, gcms, val_day)
    if key in _CACHE:
        return _CACHE[key]
    G = load_s2_gcms(gcms)
    classes = sorted(G.index)                          # same order as SmellNet's LabelEncoder
    cidx = {c: i for i, c in enumerate(classes)}
    gvec = StandardScaler().fit_transform(G.loc[classes].values).astype(np.float32)  # class-level prior
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for s in load_s2_sessions(subtract_first_row=True):
        x = s["data"]
        if p > 0:
            x = x[p:] - x[:-p]
        split = "test" if s["day"] == 6 else ("val" if s["day"] == val_day else "train")
        for st in range(0, len(x) - w + 1, w // 2):
            parts[split][0].append(x[st:st + w])
            parts[split][1].append(cidx[s["substance"]])
    arr = {k: (np.stack(v[0]).astype(np.float64), np.array(v[1])) for k, v in parts.items()}
    Xtr = arr["train"][0]
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    out = {k: (sc.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).astype(np.float32), y)
           for k, (X, y) in arr.items()}
    out["gcms"] = gvec
    out["classes"] = classes
    _CACHE[key] = out
    return out


# ------------------------------------------------------------------ models ---------
def make_model(name, C, K, w):
    """Same constructors and defaults as SmellNet's run.py:get_model."""
    if name == "mlp":
        return snm.MLPClassifier(in_features=C, num_classes=K, hidden_sizes=(256, 256), dropout=0.2, pool="mean")
    if name == "cnn":
        return snm.CNN1DClassifier(in_channels=C, num_classes=K, channels=(64, 128, 256), kernel_size=5, dropout=0.2)
    if name == "lstm":
        return snm.LSTMNet(input_dim=C, hidden_dim=256, embedding_dim=256, num_classes=K, num_layers=1)
    if name == "transformer":
        return snm.Transformer(input_dim=C, model_dim=256, num_classes=K, num_heads=8, num_layers=3, dropout=0.1)
    raise ValueError(name)


def head_from_features(model, feats):
    if isinstance(model, (snm.MLPClassifier, snm.CNN1DClassifier)):
        return model.head(feats)
    if isinstance(model, snm.LSTMNet):
        return model.classifier(feats)
    if isinstance(model, snm.Transformer):
        return model.classifier(model.dropout(feats))
    raise TypeError(type(model))


def unique_class_batches(y, bs, rng):
    """Equivalent of SmellNet's UniqueGCMSampler: every batch has distinct classes, so the
    in-batch contrastive negatives are all other ingredients."""
    queues = {c: list(rng.permutation(np.where(y == c)[0])) for c in np.unique(y)}
    batches, cur = [], []
    while any(queues.values()):
        for c in rng.permutation(list(queues)):
            if queues[c]:
                cur.append(queues[c].pop())
                if len(cur) == bs:
                    batches.append(cur)
                    cur = []
    if cur:
        batches.append(cur)
    return batches


# ------------------------------------------------------------------ train / eval ---
@torch.no_grad()
def scores(system, model, genc, X, gvec, bs=512):
    """Class scores (N, K): logits for D0/D2, cosine similarity to all 50 prototypes for D1."""
    model.eval()
    if genc is not None:
        genc.eval()
    out = []
    zg = F.normalize(genc.forward_features(torch.from_numpy(gvec).to(DEV)), dim=1) if system == "D1" else None
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        if system == "D1":
            out.append((F.normalize(model.forward_features(xb), dim=1) @ zg.T).cpu())
        else:
            out.append(model(xb).cpu())
    return torch.cat(out).numpy()


def acc_at(S, y, k=1):
    top = np.argsort(-S, 1)[:, :k]
    return float((top == y[:, None]).any(1).mean())


def train_run(model_name, w, p, system, lr, seed, lam=0.0, gcms="spec", epochs=90, patience=25,
              bs=32, mask_eval=True, verbose=False):
    d = get_data(w, p, gcms)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = d["train"], d["val"], d["test"]
    gvec = d["gcms"]
    K, C = len(d["classes"]), Xtr.shape[2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(model_name, C, K, w).to(DEV)
    genc = snm.GCMSMLPEncoder(in_features=gvec.shape[1], embedding_dim=256).to(DEV) if system in ("D1", "D2") else None
    params = list(model.parameters()) + (list(genc.parameters()) if genc is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    Xtr_t, ytr_t, G_t = torch.from_numpy(Xtr).to(DEV), torch.from_numpy(ytr).to(DEV), torch.from_numpy(gvec).to(DEV)

    best = (-1.0, None, 0)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if genc is not None:
            genc.train()
        if system == "D0":
            perm = rng.permutation(len(ytr))
            batches = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        else:
            batches = unique_class_batches(ytr, bs, rng)
        for b in batches:
            if len(b) < 2:
                continue
            b = torch.as_tensor(np.asarray(b), device=DEV)
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad()
            if system == "D0":
                loss = F.cross_entropy(model(xb), yb)
            else:
                zs = model.forward_features(xb)
                zg = genc.forward_features(G_t[yb])  # training-time pairing only
                lc = snl.cross_modal_contrastive_loss(zg, zs, TAU)
                loss = lc if system == "D1" else F.cross_entropy(head_from_features(model, zs), yb) + lam * lc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        va = acc_at(scores(system, model, genc, Xva, gvec), yva)
        hist.append(va)
        if va > best[0]:
            best = (va, ({k: v.clone() for k, v in model.state_dict().items()},
                         {k: v.clone() for k, v in genc.state_dict().items()} if genc else None), ep)
        if verbose:
            print(f"ep {ep} val {va:.3f}")
        if ep - best[2] >= patience:
            break
    model.load_state_dict(best[1][0])
    if genc is not None:
        genc.load_state_dict(best[1][1])
    Sva = scores(system, model, genc, Xva, gvec)
    Ste = scores(system, model, genc, Xte, gvec)
    res = {"model": model_name, "w": w, "p": p, "system": system, "lr": lr, "seed": seed, "lam": lam,
           "gcms": gcms if system != "D0" else "none", "epochs_run": len(hist), "best_epoch": best[2],
           "val_acc1": acc_at(Sva, yva), "test_acc1": acc_at(Ste, yte), "test_acc5": acc_at(Ste, yte, 5),
           "test_macro_f1": float(f1_score(yte, Ste.argmax(1), average="macro")),
           "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
           "train_time_s": round(time.time() - t0, 1), "device": DEV.type, "val_curve": hist}
    if mask_eval:  # AB15: zero one standardised channel at test time (as SmellNet's --ablate-channels)
        for c, name in enumerate(S2_CHANNELS):
            Xm = Xte.copy()
            Xm[:, :, c] = 0.0
            res[f"mask_{name}_acc1"] = acc_at(scores(system, model, genc, Xm, gvec), yte)
    return res, Sva, Ste


def run_id(r):
    return f"{r['model']}_w{r['w']}_p{r['p']}_{r['system']}_lr{r['lr']}_lam{r['lam']}_g{r['gcms']}_s{r['seed']}"


def run_and_save(**kw):
    tag = {"model": kw["model_name"], "w": kw["w"], "p": kw["p"], "system": kw["system"], "lr": kw["lr"],
           "lam": kw.get("lam", 0.0), "gcms": kw.get("gcms", "spec") if kw["system"] != "D0" else "none",
           "seed": kw["seed"]}
    rid = run_id(tag)
    f = OUT / f"{rid}.json"
    if f.exists():
        return json.loads(f.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    res, Sva, Ste = train_run(**kw)
    res["run_id"] = rid
    np.savez_compressed(OUT / f"{rid}_scores.npz", val=Sva.astype(np.float32), test=Ste.astype(np.float32))
    f.write_text(json.dumps(res | {"env": environment()}, indent=1))
    print(f"{rid}: val {res['val_acc1']:.3f} test {res['test_acc1']:.3f} ep {res['best_epoch']}/{res['epochs_run']} "
          f"{res['train_time_s']}s", flush=True)
    return res


# ------------------------------------------------------------------ sweep ----------
def load_runs():
    rows = [json.loads(f.read_text()) for f in OUT.glob("*.json")] if OUT.exists() else []
    return pd.DataFrame(rows)


def best_lr(df, model, w, p, system, lam=0.0, gcms="spec"):
    m = df[(df.model == model) & (df.w == w) & (df.p == p) & (df.system == system) & (df.seed == 42)
           & (np.isclose(df.lam, lam)) & (df.gcms == (gcms if system != "D0" else "none"))]
    return float(m.sort_values(["val_acc1", "lr"], ascending=[False, True]).iloc[0].lr)


def sweep(models, ws, ps, stage, gcms="spec"):
    """stage 'lr'   : D0 and D1 over the LR grid (seed 42)
       stage 'lam'  : D2 over lambda grid at D0's selected LR (seed 42)
       stage 'seeds': D0, D1 at selected LR and D2 at selected (LR, lambda) for the other seeds
       stage 'seeds01': only the D0/D1 part of 'seeds' (no lambda search needed; run first)
       stage 'seeds2' : only the D2 part of 'seeds' (needs stage 'lam')
       stage 'atom' : D1 with Xatom encoding at D1's selected LR, all seeds (AB12)
    All stages skip runs whose result file already exists, so any order/restart is safe."""
    for m in models:
        for w in ws:
            for p in ps:
                base = dict(model_name=m, w=w, p=p)
                if stage == "lr":
                    for system in ["D0", "D1"]:
                        for lr in LRS:
                            run_and_save(**base, system=system, lr=lr, seed=42, gcms=gcms)
                    continue
                df = load_runs()
                lr0, lr1 = best_lr(df, m, w, p, "D0"), best_lr(df, m, w, p, "D1", gcms=gcms)
                if stage == "lam":
                    for lam in LAMBDAS:
                        run_and_save(**base, system="D2", lr=lr0, seed=42, lam=lam, gcms=gcms)
                elif stage in ("seeds", "seeds01", "seeds2"):
                    lam = best_lam(df, m, w, p, gcms) if stage != "seeds01" else None
                    for seed in SEEDS[1:]:
                        if stage != "seeds2":
                            run_and_save(**base, system="D0", lr=lr0, seed=seed)
                            run_and_save(**base, system="D1", lr=lr1, seed=seed, gcms=gcms)
                        if stage != "seeds01":
                            run_and_save(**base, system="D2", lr=lr0, seed=seed, lam=lam, gcms=gcms)
                elif stage == "atom":
                    for seed in SEEDS:
                        run_and_save(**base, system="D1", lr=lr1, seed=seed, gcms="atom")


def best_lam(df, m, w, p, gcms="spec"):
    s = df[(df.model == m) & (df.w == w) & (df.p == p) & (df.system == "D2") & (df.seed == 42) & (df.gcms == gcms)]
    return float(s.sort_values(["val_acc1", "lam"], ascending=[False, True]).iloc[0].lam)


# ------------------------------------------------------------------ D3 + collect ----
def softmax(S, t=1.0):
    S = S / t
    S = S - S.max(1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(1, keepdims=True)


def collect():
    df = load_runs()
    df = df[df.gcms.isin(["spec", "none"])]
    rows, abl16 = [], []
    for (m, w, p), g in df.groupby(["model", "w", "p"]):
        lr0 = best_lr(df, m, w, p, "D0")
        lr1 = best_lr(df, m, w, p, "D1")
        sel = {"D0": g[(g.system == "D0") & np.isclose(g.lr, lr0)],
               "D1": g[(g.system == "D1") & np.isclose(g.lr, lr1)]}
        if (g.system == "D2").any():
            lam = best_lam(df, m, w, p)
            sel["D2"] = g[(g.system == "D2") & np.isclose(g.lr, lr0) & np.isclose(g.lam, lam)]
            for lam_, gg in g[(g.system == "D2") & (g.seed == 42)].groupby("lam"):
                abl16.append({"model": m, "w": w, "p": p, "param": "lambda", "value": lam_,
                              "val_acc1": gg.val_acc1.iloc[0], "test_acc1": gg.test_acc1.iloc[0]})
        for s, gg in sel.items():
            rows.append(_agg(m, w, p, s, gg, {"lr": gg.lr.iloc[0], "lam": gg.lam.iloc[0]}))
        # D3: alpha tuned on day-5 validation scores, seed by seed
        both = sel["D0"].merge(sel["D1"], on="seed", suffixes=("_0", "_1"))
        d = get_data(w, p)
        yva, yte = d["val"][1], d["test"][1]
        d3 = []
        for _, r in both.iterrows():
            z0 = np.load(OUT / f"{r.run_id_0}_scores.npz")
            z1 = np.load(OUT / f"{r.run_id_1}_scores.npz")
            va = {a: acc_at(a * softmax(z0["val"]) + (1 - a) * softmax(z1["val"], TAU), yva) for a in ALPHAS}
            a_best = max(ALPHAS, key=lambda a: (va[a], -abs(a - 0.5)))
            P = a_best * softmax(z0["test"]) + (1 - a_best) * softmax(z1["test"], TAU)
            d3.append({"seed": r.seed, "alpha": a_best, "val_acc1": va[a_best], "test_acc1": acc_at(P, yte),
                       "test_acc5": acc_at(P, yte, 5),
                       "test_macro_f1": float(f1_score(yte, P.argmax(1), average="macro"))})
            if r.seed == 42:
                for a in ALPHAS:
                    Pa = a * softmax(z0["test"]) + (1 - a) * softmax(z1["test"], TAU)
                    abl16.append({"model": m, "w": w, "p": p, "param": "alpha", "value": a, "val_acc1": va[a],
                                  "test_acc1": acc_at(Pa, yte)})
        d3 = pd.DataFrame(d3)
        rows.append(_agg(m, w, p, "D3", d3, {"alpha": ",".join(map(str, d3.alpha))}))
    out = pd.DataFrame(rows)
    out = out.merge(paper_table2(), on=["model", "w", "p", "system"], how="left")
    out.to_csv(RESULTS / "expD_sensor_gcms_fusion.csv", index=False)
    pd.DataFrame(abl16).to_csv(RESULTS / "ablations" / "AB16_lambda_alpha.csv", index=False)
    ablation_tables(df, out)
    print(out[["model", "w", "p", "system", "n_seeds", "test_acc1_mean", "test_acc1_std", "paper_acc1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out


def _agg(m, w, p, system, g, extra):
    r = {"model": m, "w": w, "p": p, "system": system, "n_seeds": len(g)}
    for c in ["val_acc1", "test_acc1", "test_acc5", "test_macro_f1"]:
        r[f"{c}_mean"] = g[c].mean()
        r[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
    return r | extra


def paper_table2():
    """Reported Acc@1 (%) from the ICLR 2026 SmellNet paper, Table 2, filled in by hand from
    the paper in docs/smellnet_table2_reported.csv (columns model,w,p,system,paper_acc1)."""
    f = Path(__file__).resolve().parents[2] / "docs" / "smellnet_table2_reported.csv"
    if f.exists():
        return pd.read_csv(f, comment="#")
    return pd.DataFrame(columns=["model", "w", "p", "system", "paper_acc1"])


def ablation_tables(df, out):
    abl = RESULTS / "ablations"
    # AB11 with vs without GC-MS, AB13 w/p, AB14 model family: views of the main table
    piv = out.pivot_table(index=["model", "w", "p"], columns="system", values="test_acc1_mean")
    for s in ["D1", "D2", "D3"]:
        if s in piv:
            piv[f"{s}-D0"] = piv[s] - piv["D0"]
    piv.reset_index().to_csv(abl / "AB11_with_without_gcms.csv", index=False)
    out.groupby(["system", "w", "p"]).test_acc1_mean.mean().reset_index().to_csv(
        abl / "AB13_window_differencing.csv", index=False)
    out.groupby(["system", "model"]).test_acc1_mean.mean().reset_index().to_csv(
        abl / "AB14_model_family.csv", index=False)
    # AB15 channel masking (seed-averaged, selected configurations)
    mcols = [c for c in df.columns if c.startswith("mask_")]
    sel = []
    for (m, w, p), g in df.groupby(["model", "w", "p"]):
        for s in ["D0", "D1"]:
            lr = best_lr(df, m, w, p, s)
            gg = g[(g.system == s) & np.isclose(g.lr, lr) & (g.gcms != "atom")]
            r = {"model": m, "w": w, "p": p, "system": s, "baseline_acc1": gg.test_acc1.mean()}
            for c in mcols:
                r[c.replace("_acc1", "_delta")] = gg[c].mean() - gg.test_acc1.mean()
            sel.append(r)
    pd.DataFrame(sel).to_csv(abl / "AB15_channel_masking.csv", index=False)
    # AB12 Xspec vs Xatom
    atom = load_runs()
    atom = atom[(atom.system == "D1")]
    if (atom.gcms == "atom").any():
        # like-for-like: both encodings at D1's selected learning rate (Xatom runs only exist there)
        keep = [(r.gcms == "atom") or np.isclose(r.lr, best_lr(atom, r.model, r.w, r.p, "D1"))
                for r in atom.itertuples()]
        atom = atom[keep]
        t = atom.groupby(["model", "w", "p", "gcms"]).agg(test_acc1_mean=("test_acc1", "mean"),
                                                          test_acc1_std=("test_acc1", "std"),
                                                          n=("seed", "nunique")).reset_index()
        keys = atom[atom.gcms == "atom"][["model", "w", "p"]].drop_duplicates()
        t = t.merge(keys)
        # Xspec rows restricted to the selected LR so the comparison is like-for-like
        t.to_csv(abl / "AB12_gcms_encoding.csv", index=False)


# ------------------------------------------------------------------ DX (leaky) ------
def leakage_demo(w=100, p=25, seed=42):
    """INVALID by design: shows what happens if the true class's GC-MS vector is a test input.

    DX-gcms_only : logistic regression on the GC-MS vector alone.
    DX-early     : logistic regression on [mean-pooled sensor window, GC-MS vector of its label].
    DX-honest    : the same early-fusion model, but at test time every window gets the
                   average GC-MS vector (the only label-free choice) -> shows the gain was leakage.
    """
    from sklearn.linear_model import LogisticRegression
    d = get_data(w, p)
    (Xtr, ytr), (Xte, yte) = d["train"], d["test"]
    G = d["gcms"]
    ftr, fte = np.c_[Xtr.mean(1), Xtr.std(1)], np.c_[Xte.mean(1), Xte.std(1)]
    rows = []
    lr = LogisticRegression(max_iter=3000)
    rows.append({"system": "DX-gcms_only (LEAKY)", "test_acc1": lr.fit(G[ytr], ytr).score(G[yte], yte)})
    lr2 = LogisticRegression(max_iter=3000).fit(np.c_[ftr, G[ytr]], ytr)
    rows.append({"system": "DX-early_concat (LEAKY)", "test_acc1": lr2.score(np.c_[fte, G[yte]], yte)})
    rows.append({"system": "DX-early_concat, label-free GC-MS at test",
                 "test_acc1": lr2.score(np.c_[fte, np.tile(G.mean(0), (len(yte), 1))], yte)})
    lr3 = LogisticRegression(max_iter=3000).fit(ftr, ytr)
    rows.append({"system": "sensor-only same features (valid)", "test_acc1": lr3.score(fte, yte)})
    df = pd.DataFrame(rows).assign(w=w, p=p)
    df.to_csv(RESULTS / "expD_leakage_demo.csv", index=False)
    print(df.to_string(index=False))


def time_epoch():
    for m in ["mlp", "cnn", "lstm", "transformer"]:
        for w in [50, 100]:
            t = time.time()
            train_run(m, w, 25, "D1", 1e-3, 42, epochs=1, patience=99, mask_eval=False)
            print(m, w, f"{time.time() - t:.1f}s/epoch (D1)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "collect", "leak", "time"])
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn", "lstm", "transformer"])
    ap.add_argument("--ws", nargs="+", type=int, default=[50, 100])
    ap.add_argument("--ps", nargs="+", type=int, default=[0, 25])
    ap.add_argument("--stage", default="lr", choices=["lr", "lam", "seeds", "seeds01", "seeds2", "atom"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.cmd == "sweep":
        sweep(a.models, a.ws, a.ps, a.stage)
    elif a.cmd == "collect":
        collect()
    elif a.cmd == "leak":
        leakage_demo()
    else:
        time_epoch()
