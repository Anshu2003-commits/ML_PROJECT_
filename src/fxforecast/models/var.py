"""Econometric multivariate models: VAR on stationary differentials, VECM on log levels.

Both need the full time series rather than a row of lagged features, so they
are built with the series table and use X only for its index (the forecast
origins). Look-ahead is prevented explicitly:
    fit      uses series rows up to T = last training origin + h (targets known by T)
    predict  forecasts from each origin t using rows up to t only, with the
             parameters fitted at T (the same rule every other model follows)

Forecasts are iterated h steps ahead and turned into y = log FX(t+h) - log FX(t).

    VARForecaster   dlog_fx + 7 macro differentials; non-stationary ones are
                    differenced by ADF on the training window (as in M2); p by AIC.
    VECMForecaster  monetary-model levels (log FX, relative prices, output, money,
                    rate differential); cointegration rank by Johansen trace on the
                    training window; lag order by AIC on the levels VAR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

from fxforecast.features.diagnostics import with_freq, integration_order, var_lag_selection
from fxforecast.features.transforms import TARGET, difference


def iterate_var(history: np.ndarray, coefs: np.ndarray, const: np.ndarray, steps: int) -> np.ndarray:
    """Iterate y_t = const + sum_i A_i y_{t-i} forward.

    history: (>= p, K) most recent rows last; coefs: (p, K, K); returns (steps, K).
    """
    p = coefs.shape[0]
    if len(history) < p:
        raise ValueError(f"Need at least {p} rows of history, got {len(history)}")
    hist = [row for row in history[-p:]]
    out = []
    for _ in range(steps):
        nxt = const + sum(coefs[i] @ hist[-1 - i] for i in range(p))
        hist.append(nxt)
        out.append(nxt)
    return np.asarray(out)


def _refit_date(X: pd.DataFrame, horizon: int) -> pd.Timestamp:
    return X.index.max() + pd.DateOffset(months=horizon)


class VARForecaster:
    name = "var"

    def __init__(self, series: pd.DataFrame, horizon: int, max_lag: int = 6, criterion: str = "aic"):
        if TARGET not in series.columns:
            raise ValueError(f"series must contain {TARGET}")
        self.series, self.horizon, self.max_lag, self.criterion = series, horizon, max_lag, criterion

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VARForecaster":
        train = self.series.loc[: _refit_date(X, self.horizon)]
        self.orders_ = {c: 0 if c == TARGET else integration_order(train[c]) for c in train.columns}
        z = difference(train, self.orders_)
        self.p_ = max(var_lag_selection(z, self.max_lag)[self.criterion], 1)
        res = VAR(with_freq(z)).fit(self.p_)
        self.coefs_, self.const_ = res.coefs, res.intercept
        self.target_pos_ = z.columns.get_loc(TARGET)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = difference(self.series.loc[: X.index.max()], self.orders_)   # differencing uses past rows only
        out = []
        for t in X.index:
            path = iterate_var(z.loc[:t].to_numpy(), self.coefs_, self.const_, self.horizon)
            out.append(path[:, self.target_pos_].sum())   # sum of monthly log changes = h-month change
        return np.asarray(out)

    def info(self) -> dict:
        return {"p": int(self.p_), "differenced": sorted(c for c, d in self.orders_.items() if d)}


class VECMForecaster:
    name = "vecm"
    LOG_FX = "log_fx"

    def __init__(self, levels: pd.DataFrame, horizon: int, max_lag: int = 6,
                 deterministic: str = "co", signif: float = 0.05):
        if self.LOG_FX not in levels.columns:
            raise ValueError(f"levels must contain {self.LOG_FX}")
        self.levels, self.horizon, self.max_lag = levels, horizon, max_lag
        self.deterministic, self.signif = deterministic, signif

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VECMForecaster":
        train = self.levels.loc[: _refit_date(X, self.horizon)]
        self.k_ar_diff_ = max(var_lag_selection(train, self.max_lag)["aic"] - 1, 1)
        rank = select_coint_rank(train.to_numpy(), det_order=0, k_ar_diff=self.k_ar_diff_,
                                 method="trace", signif=self.signif).rank
        self.rank_ = int(min(rank, train.shape[1] - 1))     # full rank would mean all levels stationary
        res = VECM(train.to_numpy(), k_ar_diff=self.k_ar_diff_, coint_rank=self.rank_,
                   deterministic=self.deterministic).fit()
        const = np.zeros(train.shape[1])
        if res.det_coef.size:
            const += res.det_coef[:, 0]
        if self.rank_ and res.det_coef_coint.size:
            const += (res.alpha @ res.det_coef_coint).ravel()
        self.coefs_, self.const_ = res.var_rep, const
        self.fx_pos_ = train.columns.get_loc(self.LOG_FX)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = []
        for t in X.index:
            hist = self.levels.loc[:t].to_numpy()
            path = iterate_var(hist, self.coefs_, self.const_, self.horizon)
            out.append(path[-1, self.fx_pos_] - hist[-1, self.fx_pos_])
        return np.asarray(out)

    def info(self) -> dict:
        return {"k_ar_diff": int(self.k_ar_diff_), "rank": self.rank_}
