"""Tests of equal forecast accuracy.

Both work on forecast errors of the target y = log FX(t+h) - log FX(t) with
squared-error loss, and allow for the MA(h-1) autocorrelation that h-step
forecast errors have.

diebold_mariano  any two models (Diebold & Mariano 1995), with the
                 Harvey, Leybourne & Newbold (1997) small-sample correction
                 and Student-t p-values. Two-sided.
clark_west       nested models (Clark & West 2007): the restricted model
                 (e.g. FX only) is a special case of the unrestricted one
                 (FX + macro). DM is undersized in that case; CW adjusts
                 for the noise of estimating extra parameters. One-sided:
                 H1 = the unrestricted model is more accurate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    mean_loss_diff: float   # mean of (loss model 1 - loss model 2); > 0 means model 2 is better
    n: int


def _long_run_variance(d: np.ndarray, lags: int) -> float:
    """Autocovariance sum up to ``lags``. Uses the rectangular (DM) kernel,
    falling back to Bartlett weights if that is not positive."""
    d = d - d.mean()
    n = len(d)
    gammas = [d @ d / n] + [d[k:] @ d[:-k] / n for k in range(1, lags + 1)]
    lrv = gammas[0] + 2 * sum(gammas[1:])
    if lrv <= 0:
        lrv = gammas[0] + 2 * sum((1 - k / (lags + 1)) * g for k, g in enumerate(gammas[1:], start=1))
    return float(lrv)


def _check(*arrays: np.ndarray) -> int:
    n = len(arrays[0])
    if any(len(a) != n for a in arrays) or n < 10:
        raise ValueError("Need equally long error series with at least 10 observations")
    return n


def diebold_mariano(e1, e2, horizon: int) -> TestResult:
    """H0: equal expected squared error. Positive statistic = model 2 more accurate."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    n = _check(e1, e2)
    d = e1 ** 2 - e2 ** 2
    lrv = _long_run_variance(d, horizon - 1)
    if lrv <= 0:
        return TestResult(0.0, 1.0, float(d.mean()), n)   # identical forecasts
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * hln)
    return TestResult(stat, float(2 * stats.t.sf(abs(stat), df=n - 1)), float(d.mean()), n)


def clark_west(y_true, pred_restricted, pred_unrestricted, horizon: int) -> TestResult:
    """H0: equal accuracy; H1: unrestricted (bigger, nesting) model more accurate."""
    y, fr, fu = (np.asarray(a, float) for a in (y_true, pred_restricted, pred_unrestricted))
    n = _check(y, fr, fu)
    er, eu = y - fr, y - fu
    f = er ** 2 - (eu ** 2 - (fr - fu) ** 2)
    lrv = _long_run_variance(f, horizon - 1)
    if lrv <= 0:
        return TestResult(0.0, 0.5, float(np.mean(er ** 2 - eu ** 2)), n)
    stat = float(f.mean() / np.sqrt(lrv / n))
    return TestResult(stat, float(stats.norm.sf(stat)), float(np.mean(er ** 2 - eu ** 2)), n)
