import numpy as np
import pandas as pd
import pytest

from fxforecast.features.supervised import select_lags
from fxforecast.models.svr import LagSelector, SVRForecaster, inner_cv

SMALL_GRID = dict(n_lags=(1, 2), C=(1, 10), gamma=(0.1,), epsilon=(0.1,), cv_splits=3, n_jobs=1)


@pytest.fixture
def nonlinear():
    """y depends nonlinearly on the first lag only."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({f"dlog_fx_lag{k}": rng.uniform(-2, 2, n) for k in range(3)},
                     index=pd.date_range("1990-01-01", periods=n, freq="MS"))
    y = pd.Series(np.sin(2 * X["dlog_fx_lag0"]) + 0.05 * rng.normal(size=n), index=X.index)
    return X, y


def test_select_lags_keeps_all_variables_with_fewer_lags():
    X = pd.DataFrame(columns=["dlog_fx_lag0", "dlog_fx_lag1", "dlog_fx_lag2", "infl_diff_lag0", "infl_diff_lag1"])
    assert list(select_lags(X, 1).columns) == ["dlog_fx_lag0", "infl_diff_lag0"]
    assert LagSelector(2).fit(X).transform(X).shape[1] == 4


@pytest.mark.parametrize("h", [1, 3, 6])
def test_inner_cv_validates_only_on_later_rows_with_embargo(h):
    for train, val in inner_cv(h, 5).split(np.zeros((200, 1))):
        assert train.max() + h <= val.min()        # train origin + h never reaches validation origins
        assert np.all(np.diff(val) == 1)           # each validation block is contiguous


def test_svr_learns_nonlinear_signal(nonlinear):
    X, y = nonlinear
    m = SVRForecaster(1, **SMALL_GRID).fit(X.iloc[:250], y.iloc[:250])
    y_test = y.iloc[250:].to_numpy()
    rmse_svr = np.sqrt(np.mean((m.predict(X.iloc[250:]) - y_test) ** 2))
    # linear least squares on the same inputs cannot capture sin()
    A = np.column_stack([np.ones(250), X.iloc[:250].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.iloc[:250].to_numpy(), rcond=None)
    rmse_lin = np.sqrt(np.mean((np.column_stack([np.ones(50), X.iloc[250:].to_numpy()]) @ coef - y_test) ** 2))
    assert rmse_svr < 0.6 * rmse_lin
    assert m.info()["n_lags"] in (1, 2) and set(m.info()) == {"n_lags", "C", "gamma", "epsilon"}


def test_svr_is_deterministic(nonlinear):
    X, y = nonlinear
    a = SVRForecaster(1, **SMALL_GRID).fit(X.iloc[:200], y.iloc[:200]).predict(X.iloc[200:])
    b = SVRForecaster(1, **SMALL_GRID).fit(X.iloc[:200], y.iloc[:200]).predict(X.iloc[200:])
    assert np.array_equal(a, b)


def test_svr_tuning_depends_only_on_training_rows(nonlinear):
    """Changing rows after the training window must not change the fitted model."""
    X, y = nonlinear
    m1 = SVRForecaster(1, **SMALL_GRID).fit(X.iloc[:200], y.iloc[:200])
    y2 = y.copy()
    y2.iloc[200:] += 100
    m2 = SVRForecaster(1, **SMALL_GRID).fit(X.iloc[:200], y2.iloc[:200])
    assert m1.info() == m2.info()
    assert np.array_equal(m1.predict(X.iloc[200:]), m2.predict(X.iloc[200:]))
