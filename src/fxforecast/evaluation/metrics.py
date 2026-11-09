"""Forecast accuracy metrics.

Level metrics (RMSE, MAE, MAPE, MPE, "accuracy") are computed on the INR/USD
level, as in the base paper. Directional accuracy uses the predicted vs actual
sign of the h-month change. Theil's U compares RMSE with the random walk on
the same points: U < 1 means the model beats the "no change" forecast.

Paper definitions: MAPE = mean(|y_hat - y| / y), MPE = mean((y_hat - y) / y),
Accuracy = 100 % - MAPE x 100 %.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Need two non-empty arrays of equal shape, got {a.shape} and {b.shape}")


def rmse(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    _check(a, p)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mae(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    _check(a, p)
    return float(np.mean(np.abs(p - a)))


def mape(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    _check(a, p)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs(p - a) / a))


def mpe(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    _check(a, p)
    return float(np.mean((p - a) / a))


def hit_rate(actual_change, predicted_change) -> float:
    """Share of correctly predicted directions, over forecasts that predict a direction.

    A zero forecast (the random walk) has no direction, so it returns NaN.
    """
    a, p = np.asarray(actual_change, float), np.asarray(predicted_change, float)
    _check(a, p)
    called = p != 0
    if not called.any():
        return float("nan")
    return float(np.mean(np.sign(p[called]) == np.sign(a[called])))


def summarize(preds: pd.DataFrame) -> dict[str, float]:
    """Metrics for one model on one set of forecasts.

    ``preds`` needs columns fx_true, fx_pred, y_true, y_pred.
    """
    m = mape(preds["fx_true"], preds["fx_pred"])
    return {
        "n": len(preds),
        "rmse": rmse(preds["fx_true"], preds["fx_pred"]),
        "mae": mae(preds["fx_true"], preds["fx_pred"]),
        "mape": m,
        "mpe": mpe(preds["fx_true"], preds["fx_pred"]),
        "accuracy_pct": 100 * (1 - m),
        "hit_rate": hit_rate(preds["y_true"], preds["y_pred"]),
    }


def member_spread(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """RMSE (INR) of each ensemble member separately, summarised as mean, std, min, max.

    Uses the ``y_pred_members`` column (JSON list per row); groups without it are skipped.
    """
    if "y_pred_members" not in preds.columns:
        return pd.DataFrame()
    rows = {}
    for key, g in preds.dropna(subset=["y_pred_members"]).groupby(by, sort=False):
        members = np.array([json.loads(s) for s in g["y_pred_members"]])        # (rows, seeds)
        fx_members = g["fx_origin"].to_numpy()[:, None] * np.exp(members)
        scores = [rmse(g["fx_true"], fx_members[:, k]) for k in range(members.shape[1])]
        rows[key] = {"seeds": len(scores), "rmse_mean": np.mean(scores), "rmse_std": np.std(scores, ddof=1),
                     "rmse_min": np.min(scores), "rmse_max": np.max(scores), "rmse_ensemble": rmse(g["fx_true"], g["fx_pred"])}
    table = pd.DataFrame(rows).T
    if len(table):
        table.index.names = by
        table["seeds"] = table["seeds"].astype(int)
    return table


def metrics_table(preds: pd.DataFrame, by: list[str], benchmark: str = "random_walk") -> pd.DataFrame:
    """Summarize every group in ``by`` (which must include 'model') and add Theil's U.

    U = RMSE(model) / RMSE(benchmark) on the same forecast origins. The benchmark
    is matched on every grouping column except 'model' and 'feature_set'. Raises
    if any forecast has no benchmark forecast for the same origin.
    """
    if "model" not in by:
        raise ValueError("'by' must include 'model'")
    table = pd.DataFrame({key: summarize(g) for key, g in preds.groupby(by, sort=False)}).T
    table.index.names = by
    table["n"] = table["n"].astype(int)

    context = [c for c in by if c not in ("model", "feature_set")] + ["origin"]
    bench = preds.loc[preds["model"] == benchmark, context + ["fx_pred"]]
    if bench.duplicated(context).any():
        raise ValueError(f"Benchmark '{benchmark}' has more than one forecast per origin")
    merged = preds.merge(bench, on=context, how="left", suffixes=("", "_bench"))
    if merged["fx_pred_bench"].isna().any():
        raise ValueError(f"Some forecasts have no '{benchmark}' forecast for the same origin")
    table["theil_u"] = pd.Series({
        key: rmse(g["fx_true"], g["fx_pred"]) / rmse(g["fx_true"], g["fx_pred_bench"])
        for key, g in merged.groupby(by, sort=False)
    })
    return table
