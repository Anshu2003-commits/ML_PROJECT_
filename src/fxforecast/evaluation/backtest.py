"""Run a model over a list of folds and collect out-of-sample forecasts.

A fresh model is built for every fold, so nothing learned in one fold carries
into the next except through the (expanding) training data itself.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

from fxforecast.evaluation.splits import Fold
from fxforecast.models import Forecaster

log = logging.getLogger(__name__)


def backtest(
    make_model: Callable[[], Forecaster],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
) -> pd.DataFrame:
    """Fit on each fold's train rows, predict its test rows.

    Returns one row per test origin: origin, refit_date, y_true, y_pred, fit_seconds,
    model_info (JSON of settings the model chose on that fold, e.g. lag order).
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    out = []
    for fold in folds:
        started = time.perf_counter()
        model = make_model().fit(X.loc[fold.train], y.loc[fold.train])
        pred = np.asarray(model.predict(X.loc[fold.test]), dtype=float)
        seconds = time.perf_counter() - started
        if pred.shape != (len(fold.test),) or not np.isfinite(pred).all():
            raise ValueError(f"{model.name}: bad predictions for refit {fold.refit_date:%Y-%m}")
        log.debug("%s refit %s: train=%d test=%d %.3fs", model.name, f"{fold.refit_date:%Y-%m}",
                  len(fold.train), len(fold.test), seconds)
        info = model.info() if hasattr(model, "info") else {}
        out.append(pd.DataFrame({
            "origin": fold.test,
            "refit_date": fold.refit_date,
            "y_true": y.loc[fold.test].to_numpy(),
            "y_pred": pred,
            "fit_seconds": seconds,
            "model_info": json.dumps(info, sort_keys=True),
        }))
    return pd.concat(out, ignore_index=True)


def attach_levels(preds: pd.DataFrame, fx: pd.Series, horizon: int) -> pd.DataFrame:
    """Add FX levels: fx_origin, fx_true = FX(t+h), fx_pred = FX(t) * exp(y_pred)."""
    out = preds.copy()
    out["target_date"] = out["origin"] + pd.DateOffset(months=horizon)
    out["fx_origin"] = fx.reindex(out["origin"]).to_numpy()
    out["fx_true"] = fx.reindex(out["target_date"]).to_numpy()
    if out[["fx_origin", "fx_true"]].isna().any().any():
        raise ValueError("FX level missing for some origin or target date")
    out["fx_pred"] = out["fx_origin"] * np.exp(out["y_pred"])
    return out
