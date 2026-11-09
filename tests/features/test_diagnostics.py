"""Diagnostics are checked on synthetic data where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from fxforecast.features import diagnostics as dg

N = 400


def _idx(n=N):
    return pd.date_range("1990-01-01", periods=n, freq="MS")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_white_noise_is_stationary(rng):
    s = pd.Series(rng.normal(size=N), index=_idx())
    assert dg.adf_kpss(s)["verdict"] == "stationary"
    assert dg.integration_order(s) == 0


def test_random_walk_is_non_stationary_and_I1(rng):
    s = pd.Series(rng.normal(size=N).cumsum(), index=_idx())
    assert dg.adf_kpss(s)["verdict"] == "non-stationary"
    assert dg.integration_order(s) == 1


def test_stationarity_table_has_row_per_column(rng):
    df = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N).cumsum()}, index=_idx())
    table = dg.stationarity_table(df)
    assert list(table.index) == ["a", "b"]
    assert {"adf_p", "kpss_p", "verdict"} <= set(table.columns)


def test_granger_detects_leading_variable(rng):
    x = rng.normal(size=N)
    y = np.r_[0, 0.8 * x[:-1]] + 0.3 * rng.normal(size=N)   # y_t depends on x_{t-1}
    noise = rng.normal(size=N)
    df = pd.DataFrame({"y": y, "x": x, "noise": noise}, index=_idx())
    table = dg.granger_table(df, "y", lags=(1, 3))
    assert table.loc["x", "p_lag1"] < 0.01
    assert table.loc["noise", "p_lag1"] > 0.05


# det_order=0 critical values assume trending (drifting) levels, as USD/INR has,
# so the synthetic walks carry a drift too. Without drift the last test is oversized.
def test_johansen_finds_cointegration(rng):
    common = (0.3 + rng.normal(size=N)).cumsum()
    df = pd.DataFrame({"a": common + rng.normal(size=N), "b": 2 * common + rng.normal(size=N)}, index=_idx())
    _, rank = dg.johansen_table(df, k_ar_diff=1)
    assert rank == 1


def test_johansen_no_cointegration_for_independent_walks(rng):
    df = pd.DataFrame({"a": (0.3 + rng.normal(size=N)).cumsum(), "b": (0.2 + rng.normal(size=N)).cumsum()},
                      index=_idx())
    _, rank = dg.johansen_table(df, k_ar_diff=1)
    assert rank == 0


def test_var_lag_selection_recovers_ar1(rng):
    e = rng.normal(size=(N, 2))
    x = np.zeros((N, 2))
    for t in range(1, N):
        x[t] = 0.6 * x[t - 1] + e[t]
    orders = dg.var_lag_selection(pd.DataFrame(x, index=_idx(), columns=["a", "b"]), max_lags=6)
    assert orders["bic"] == 1
    assert all(isinstance(v, int) for v in orders.values())


def test_ljung_box_flags_autocorrelation(rng):
    e = rng.normal(size=N)
    ar = np.zeros(N)
    for t in range(1, N):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    assert dg.ljung_box(pd.Series(ar, index=_idx()))["lb_pvalue"].iloc[0] < 0.01
    assert dg.ljung_box(pd.Series(e, index=_idx()))["lb_pvalue"].iloc[0] > 0.05
