"""Benchmark models every other model has to beat.

    RandomWalk      y_hat = 0               (next level = current level)
    Drift           y_hat = mean training y (long-run average depreciation)
    AutoRegressive  direct h-step AR(p) on past FX changes, p chosen by AIC/BIC on training rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FX_LAG_PREFIX = "dlog_fx_lag"


class RandomWalk:
    name = "random_walk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalk":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class Drift:
    name = "drift"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Drift":
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class AutoRegressive:
    """y_h(t) = c + sum_{k<p} b_k * dlog_fx(t-k), fitted by OLS.

    This is a direct forecast: one regression per horizon, no iteration. p = 0
    reduces to the drift model. All candidate p use the same rows, so the
    information criteria are comparable.
    """

    name = "ar"

    def __init__(self, max_lag: int = 12, criterion: str = "aic"):
        if criterion not in ("aic", "bic"):
            raise ValueError("criterion must be 'aic' or 'bic'")
        self.max_lag, self.criterion = max_lag, criterion

    def _design(self, X: pd.DataFrame, p: int) -> np.ndarray:
        cols = [f"{FX_LAG_PREFIX}{k}" for k in range(p)]
        missing = set(cols) - set(X.columns)
        if missing:
            raise KeyError(f"AR({p}) needs columns {sorted(missing)}")
        return np.column_stack([np.ones(len(X)), X[cols].to_numpy()]) if p else np.ones((len(X), 1))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoRegressive":
        n, target = len(y), y.to_numpy()
        best = None
        for p in range(self.max_lag + 1):
            design = self._design(X, p)
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            rss = float(np.sum((target - design @ coef) ** 2))
            penalty = 2 if self.criterion == "aic" else np.log(n)
            score = n * np.log(rss / n) + penalty * (p + 1)
            if best is None or score < best[0]:
                best = (score, p, coef)
        _, self.p_, self.coef_ = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._design(X, self.p_) @ self.coef_
