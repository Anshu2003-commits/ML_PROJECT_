"""Frame the feature table as a supervised learning problem (X at origin t -> y over t..t+h).

Rows are indexed by the **forecast origin** t: the last month whose data is
known when the forecast is made.

    y_h(t)  = log FX(t+h) - log FX(t)        (cumulative log change over h months)
    X(t)    = lags 0..n_lags-1 of the chosen columns, all observed at or before t

A level forecast is FX(t) * exp(y_hat). The random walk is y_hat = 0.

Two feature sets answer RQ2 by comparison:
    fx_only        lags of dlog_fx only (univariate information)
    fx_plus_macro  lags of dlog_fx plus lags of the 7 macro differentials

``publication_lag`` delays the macro columns by that many months to mimic the
fact that, at month t, macro data for month t is not yet released.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxforecast.features.transforms import MACRO_FEATURES, TARGET

FEATURE_SETS = ("fx_only", "fx_plus_macro")


def feature_columns(feature_set: str) -> list[str]:
    """Base columns (before lagging) used by a feature set."""
    if feature_set == "fx_only":
        return [TARGET]
    if feature_set == "fx_plus_macro":
        return [TARGET, *MACRO_FEATURES]
    raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")


def select_lags(X: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Keep columns ``<name>_lag<k>`` with k < n_lags (all base columns, fewer lags)."""
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    keep = [c for c in X.columns if int(c.rsplit("_lag", 1)[1]) < n_lags]
    return X[keep]


def to_sequences(X: pd.DataFrame, lookback: int) -> np.ndarray:
    """Reshape lag columns into (rows, lookback, variables), oldest step first.

    Step j of each sequence holds lag (lookback - 1 - j), so the last step is
    the origin month: the layout an LSTM expects.
    """
    bases = list(dict.fromkeys(c.rsplit("_lag", 1)[0] for c in X.columns))
    missing = [f"{b}_lag{k}" for b in bases for k in range(lookback) if f"{b}_lag{k}" not in X.columns]
    if missing:
        raise KeyError(f"lookback {lookback} needs columns {missing[:3]}...")
    steps = [np.column_stack([X[f"{b}_lag{k}"].to_numpy() for b in bases]) for k in reversed(range(lookback))]
    return np.stack(steps, axis=1).astype(np.float32)


def h_step_target(dlog_fx: pd.Series, horizon: int) -> pd.Series:
    """y_h(t) = sum of dlog_fx over t+1..t+h = log FX(t+h) - log FX(t)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return dlog_fx.rolling(horizon).sum().shift(-horizon).rename(f"y_h{horizon}")


def make_supervised(
    feats: pd.DataFrame,
    feature_set: str,
    *,
    n_lags: int,
    horizon: int,
    publication_lag: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned on forecast origin, with incomplete rows dropped.

    Column ``<name>_lag<k>`` holds the value observed k months before the origin
    (k = 0 is the origin month itself, shifted further back by publication_lag
    for macro columns).
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    if publication_lag < 0:
        raise ValueError("publication_lag must be >= 0")
    columns = {}
    for col in feature_columns(feature_set):
        base = feats[col] if col == TARGET else feats[col].shift(publication_lag)
        for k in range(n_lags):
            columns[f"{col}_lag{k}"] = base.shift(k)
    X = pd.DataFrame(columns, index=feats.index)
    y = h_step_target(feats[TARGET], horizon)
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep], y[keep]
    X.index.name = y.index.name = "origin"
    return X, y
