"""Small PyTorch models for S1 / S1+S2 and an sklearn-style wrapper.

Hyperparameters are fixed in advance (not tuned on test folds). Per-channel
standardisation statistics are computed on the training data inside `fit`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNN1D(nn.Module):
    """Multichannel 1D-CNN: all sensors enter as input channels (low-level fusion)."""

    def __init__(self, in_ch: int, n_classes: int, width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(width, 2 * width, 5, padding=2), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, x):  # x: (N, C, T)
        return self.head(self.net(x))


class ChannelAgnosticNet(nn.Module):
    """Shared univariate encoder applied to every channel, then pooled across channels.

    Works for any number of channels, so S1 (8 sensors) and S2 (6 channels) can be
    pooled into one training set (Experiment B, low-level fusion).
    """

    def __init__(self, n_classes: int, width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(width, 2 * width, 5, padding=2), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * width, n_classes)

    def forward(self, x):  # x: (N, C, T), C may vary between batches
        N, C, T = x.shape
        h = self.enc(x.reshape(N * C, 1, T)).reshape(N, C, -1)
        h = torch.cat([h.mean(1), h.amax(1)], 1)
        return self.head(self.drop(h))


class TorchSeqClassifier:
    """sklearn-like wrapper. X is (N, T, C) numpy; inputs are standardised per channel
    (train-fold statistics) unless `scale='none'`.

    window: if set, train on sliding windows (length `window`, stride `stride`) cut from
            each training measurement (augmentation, AB10); at test time the class
            probabilities of all windows of a measurement are averaged. Windows never
            cross measurements, so fold membership is preserved.
    """

    def __init__(self, arch="cnn", n_classes=3, epochs=150, lr=3e-3, weight_decay=1e-3,
                 batch_size=16, window=None, stride=20, seed=0, class_weight=None, width=16,
                 scale="channel"):
        self.__dict__.update(locals()); del self.__dict__["self"]

    # -- helpers
    def _windows(self, X):
        if self.window is None or self.window >= X.shape[1]:
            return X, np.arange(len(X))
        starts = range(0, X.shape[1] - self.window + 1, self.stride)
        W = np.concatenate([X[:, s:s + self.window] for s in starts])
        owner = np.tile(np.arange(len(X)), len(starts))
        return W, owner

    def _prep(self, X):
        if self.scale == "channel":
            X = (X - self.mu_) / self.sd_
        return torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)

    def fit(self, X, y, groups_mixed_channels=None):
        set_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean((0, 1), keepdims=True)
        self.sd_ = X.std((0, 1), keepdims=True) + 1e-8
        W, owner = self._windows(X)
        yw = np.asarray(y)[owner]
        self.model_ = (CNN1D(X.shape[2], self.n_classes, self.width) if self.arch == "cnn"
                       else ChannelAgnosticNet(self.n_classes, self.width))
        _train(self.model_, [(self._prep(W), torch.tensor(yw))], self)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        W, owner = self._windows(X)
        P = _predict(self.model_, [self._prep(W)])
        out = np.zeros((len(X), self.n_classes))
        np.add.at(out, owner, P)
        return out / np.bincount(owner, minlength=len(X))[:, None]


def _train(model, groups, cfg):
    """groups: list of (X tensor (N,C,T), y tensor). Batches are drawn within a group so
    tensors with different channel counts (S1: 8, S2: 6) never share a batch."""
    g = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    w = None
    if cfg.class_weight is not None:
        w = torch.tensor(cfg.class_weight, dtype=torch.float32)
    model.train()
    for _ in range(cfg.epochs):
        batches = []
        for gi, (Xg, yg) in enumerate(groups):
            perm = torch.randperm(len(Xg), generator=g)
            batches += [(gi, perm[i:i + cfg.batch_size]) for i in range(0, len(Xg), cfg.batch_size)]
        order = torch.randperm(len(batches), generator=g).tolist()
        for bi in order:
            gi, idx = batches[bi]
            if len(idx) < 2:
                continue  # BatchNorm needs >1 sample
            Xg, yg = groups[gi]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xg[idx]), yg[idx], weight=w)
            loss.backward()
            opt.step()
    model.eval()


@torch.no_grad()
def _predict(model, Xs):
    model.eval()
    return np.concatenate([F.softmax(model(X), 1).numpy() for X in Xs])
