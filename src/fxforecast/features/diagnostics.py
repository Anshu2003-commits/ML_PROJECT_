"""Statistical diagnostics: stationarity, Granger causality, cointegration, autocorrelation.

Each function takes a DataFrame or Series and returns a tidy table. Nothing
here writes files or plots; that belongs to the orchestration and reporting layers.

Run diagnostics on the training period only when they drive modelling
choices (differencing, VAR vs VECM, lag order). Otherwise the test period
leaks into the model specification.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ALPHA = 0.05


def _with_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the inferred frequency (e.g. monthly 'MS') so statsmodels does not have to guess."""
    df = df.dropna().copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is None and len(df) >= 3:
        df.index.freq = pd.infer_freq(df.index)
    return df


def adf_kpss(series: pd.Series, alpha: float = ALPHA) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) with a combined verdict.

    stationary     - ADF rejects and KPSS does not
    non-stationary - ADF does not reject and KPSS rejects
    inconclusive   - the two tests disagree
    KPSS p-values are only tabulated between 0.01 and 0.10, so they are clipped there.
    """
    x = series.dropna().to_numpy()
    adf_stat, adf_p, adf_lags, *_ = adfuller(x, regression="c", autolag="AIC", result_object=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto", result_object=False)
    adf_rejects, kpss_rejects = adf_p < alpha, kpss_p < alpha
    if adf_rejects and not kpss_rejects:
        verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"
    return {
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": adf_lags,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "verdict": verdict,
    }


def stationarity_table(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """ADF + KPSS for every column."""
    return pd.DataFrame({col: adf_kpss(df[col], alpha) for col in df.columns}).T.rename_axis("series")


def integration_order(series: pd.Series, max_d: int = 2, alpha: float = ALPHA) -> int:
    """Smallest d such that the d-th difference passes the ADF test (H0 unit root rejected)."""
    x = series.dropna()
    for d in range(max_d + 1):
        if adfuller(x.to_numpy(), regression="c", autolag="AIC", result_object=False)[1] < alpha:
            return d
        x = x.diff().dropna()
    raise ValueError(f"{series.name}: still non-stationary after {max_d} differences")


def granger_table(df: pd.DataFrame, target: str, lags: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    """p-values of 'X does not Granger-cause target' for each other column and lag.

    Uses a bivariate VAR(lag) on [target, X] and an F-test. Inputs must be stationary.
    """
    rows = {}
    for col in df.columns.drop(target):
        pair = _with_freq(df[[target, col]])
        rows[col] = {
            f"p_lag{p}": VAR(pair).fit(p).test_causality(target, [col], kind="f").pvalue
            for p in lags
        }
    return pd.DataFrame(rows).T.rename_axis("causing")


def var_lag_selection(df: pd.DataFrame, max_lags: int = 12) -> dict[str, int]:
    """Lag order chosen by AIC, BIC, HQIC and FPE for a VAR on all columns."""
    return {k: int(v) for k, v in VAR(_with_freq(df)).select_order(max_lags).selected_orders.items()}


def johansen_table(df: pd.DataFrame, k_ar_diff: int = 2, det_order: int = 0) -> tuple[pd.DataFrame, int]:
    """Johansen trace test. Returns (table, cointegration rank at 5 %).

    The rank is the number of leading hypotheses 'rank <= r' rejected in sequence.
    det_order=0 (unrestricted constant) assumes the levels trend, as log USD/INR
    does. On driftless data the final 'rank <= n-1' test over-rejects.
    """
    res = coint_johansen(df.dropna().to_numpy(), det_order, k_ar_diff)
    crit_5 = res.cvt[:, 1]
    table = pd.DataFrame(
        {"trace_stat": res.lr1, "crit_5pct": crit_5, "reject": res.lr1 > crit_5},
        index=pd.Index([f"rank <= {r}" for r in range(len(res.lr1))], name="H0"),
    )
    rank = int(np.argmin(np.append(table["reject"].to_numpy(), False)))
    return table, rank


def ljung_box(series: pd.Series, lags: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation (H0: no autocorrelation up to lag k)."""
    return acorr_ljungbox(series.dropna(), lags=list(lags)).rename_axis("lag")
