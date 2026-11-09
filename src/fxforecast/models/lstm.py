"""LSTM forecaster (PyTorch, CPU) with early stopping and a seed ensemble.

Per refit, using the training window only:
  1. Standardise inputs (per column) and target on the training rows.
  2. Hold out the last ``val_fraction`` of training rows for early stopping,
     with an (h-1)-row gap before them (same embargo as the outer split).
  3. Pick (lookback, hidden_units) by validation loss with the first seed.
  4. Train one network per seed with that architecture. The forecast is the
     mean of the seeds; per-seed forecasts are kept to report seed variability.

Network: LSTM(1 layer) -> dropout -> linear head on the last time step.
Training: Adam, MSE loss, mini-batches shuffled within the fitting rows only
(the validation block is never trained on), best-validation weights restored.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from fxforecast.features.supervised import to_sequences


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


@dataclass
class TrainResult:
    net: LSTMNet
    best_val: float
    best_epoch: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_net(x_tr, y_tr, x_val, y_val, *, hidden: int, dropout: float, lr: float, batch_size: int,
              max_epochs: int, patience: int, seed: int) -> TrainResult:
    """Train one network with early stopping on (x_val, y_val); returns best weights."""
    set_seed(seed)
    net = LSTMNet(x_tr.shape[2], hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
    gen = torch.Generator().manual_seed(seed)
    best = (np.inf, 0, copy.deepcopy(net.state_dict()))
    for epoch in range(1, max_epochs + 1):
        net.train()
        for idx in torch.randperm(len(xt), generator=gen).split(batch_size):
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(xv), yv))
        if val < best[0] - 1e-6:
            best = (val, epoch, copy.deepcopy(net.state_dict()))
        elif epoch - best[1] >= patience:
            break
    net.load_state_dict(best[2])
    net.eval()
    return TrainResult(net, best[0], best[1])


class LSTMForecaster:
    name = "lstm"

    def __init__(self, horizon: int, lookback=(6, 12), hidden_units=(16, 32), dropout: float = 0.2,
                 max_epochs: int = 300, patience: int = 20, seeds=(0, 1, 2, 3, 4), lr: float = 1e-3,
                 batch_size: int = 32, val_fraction: float = 0.2):
        self.horizon = horizon
        self.lookbacks, self.hiddens = list(lookback), list(hidden_units)
        self.seeds = list(seeds)
        self.train_kw = dict(dropout=dropout, lr=lr, batch_size=batch_size,
                             max_epochs=max_epochs, patience=patience)
        self.val_fraction = val_fraction

    # -- scaling --------------------------------------------------------------
    def _scale_x(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMForecaster":
        torch.set_num_threads(1)   # tiny networks: threading overhead outweighs gains, and keeps runs reproducible
        self.x_mean_, self.x_std_ = X.mean(), X.std(ddof=0).replace(0, 1.0)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std(ddof=0)) or 1.0
        Xs = self._scale_x(X)
        ys = ((y.to_numpy() - self.y_mean_) / self.y_std_).astype(np.float32)

        n_val = max(int(len(X) * self.val_fraction), 12)
        n_fit = len(X) - n_val - (self.horizon - 1)          # embargo before the validation block
        if n_fit < 24:
            raise ValueError(f"Too few training rows ({len(X)}) for LSTM validation split")
        val_slice = slice(len(X) - n_val, len(X))

        def data(lookback):
            seq = to_sequences(Xs, lookback)
            return seq[:n_fit], ys[:n_fit], seq[val_slice], ys[val_slice]

        # architecture choice with the first seed
        trials = {}
        for lb in self.lookbacks:
            for hid in self.hiddens:
                trials[(lb, hid)] = train_net(*data(lb), hidden=hid, seed=self.seeds[0], **self.train_kw)
        (self.lookback_, self.hidden_), first = min(trials.items(), key=lambda kv: kv[1].best_val)

        self.members_ = [first] + [
            train_net(*data(self.lookback_), hidden=self.hidden_, seed=s, **self.train_kw) for s in self.seeds[1:]
        ]
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """(n_seeds, n_rows) forecasts on the original y scale."""
        seq = torch.from_numpy(to_sequences(self._scale_x(X), self.lookback_))
        with torch.no_grad():
            out = np.stack([m.net(seq).numpy() for m in self.members_])
        return out * self.y_std_ + self.y_mean_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X).mean(axis=0)

    def info(self) -> dict:
        return {
            "lookback": self.lookback_, "hidden": self.hidden_,
            "best_epochs": [m.best_epoch for m in self.members_],
        }
