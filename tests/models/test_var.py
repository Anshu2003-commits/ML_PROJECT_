import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from fxforecast.models.var import VARForecaster, VECMForecaster, iterate_var

N = 400
IDX = pd.date_range("1990-01-01", periods=N, freq="MS")
A_TRUE = np.array([[0.4, 0.3], [0.0, 0.5]])


@pytest.fixture
def var_series():
    """Stationary VAR(1): dlog_fx depends on its own lag and on x's lag."""
    rng = np.random.default_rng(0)
    z = np.zeros((N, 2))
    for t in range(1, N):
        z[t] = A_TRUE @ z[t - 1] + rng.normal(scale=0.01, size=2)
    return pd.DataFrame(z, index=IDX, columns=["dlog_fx", "x"])


@pytest.fixture
def levels():
    """log_fx cointegrated with a drifting common trend; one independent walk."""
    rng = np.random.default_rng(1)
    trend = (0.002 + rng.normal(scale=0.01, size=N)).cumsum()
    return pd.DataFrame({
        "log_fx": 4 + trend + rng.normal(scale=0.005, size=N),
        "rel_price": 2 * trend + rng.normal(scale=0.005, size=N),
        "other": (0.001 + rng.normal(scale=0.01, size=N)).cumsum(),
    }, index=IDX)


def origins_frame(start, end):
    idx = IDX[(IDX >= start) & (IDX <= end)]
    return pd.DataFrame(index=idx)


def test_iterate_var_matches_statsmodels(var_series):
    res = VAR(var_series.to_numpy()).fit(2)
    mine = iterate_var(var_series.to_numpy(), res.coefs, res.intercept, 5)
    assert np.allclose(mine, res.forecast(var_series.to_numpy()[-2:], 5))


def test_iterate_var_needs_enough_history():
    with pytest.raises(ValueError):
        iterate_var(np.zeros((1, 2)), np.zeros((2, 2, 2)), np.zeros(2), 1)


def test_var_recovers_coefficients(var_series):
    X = origins_frame(IDX[0], IDX[N - 2])
    m = VARForecaster(var_series, horizon=1, max_lag=4, criterion="bic").fit(X, None)
    assert m.p_ == 1 and m.orders_ == {"dlog_fx": 0, "x": 0}
    assert np.allclose(m.coefs_[0], A_TRUE, atol=0.1)


@pytest.mark.parametrize("h", [1, 3])
def test_var_prediction_is_sum_of_iterated_fx_changes(var_series, h):
    train = origins_frame(IDX[0], IDX[300])
    m = VARForecaster(var_series, horizon=h, max_lag=2).fit(train, None)
    test = origins_frame(IDX[310], IDX[312])
    pred = m.predict(test)
    t = test.index[0]
    path = iterate_var(var_series.loc[:t].to_numpy(), m.coefs_, m.const_, h)
    assert pred.shape == (3,) and pred[0] == pytest.approx(path[:, 0].sum())


def _perturbed_after(df, cut):
    out = df.copy()
    out.loc[out.index > cut] += 5.0
    return out


def test_var_never_looks_ahead(var_series):
    h = 3
    train, test = origins_frame(IDX[0], IDX[300]), origins_frame(IDX[303], IDX[320])
    refit = IDX[300] + pd.DateOffset(months=h)
    base = VARForecaster(var_series, h).fit(train, None)
    future_changed = VARForecaster(_perturbed_after(var_series, refit), h).fit(train, None)
    assert np.allclose(base.coefs_, future_changed.coefs_)   # fit ignores data after refit date
    origin = test.index[5]
    shocked = VARForecaster(_perturbed_after(var_series, origin), h).fit(train, None)
    assert base.predict(test.loc[:origin])[-1] == pytest.approx(shocked.predict(test.loc[:origin])[-1])


def test_vecm_finds_rank_and_matches_statsmodels_at_sample_end(levels):
    h = 4
    train = origins_frame(IDX[0], IDX[300 - h])           # refit date = IDX[300]
    m = VECMForecaster(levels, h, max_lag=3).fit(train, None)
    assert m.rank_ == 1
    res = VECM(levels.loc[:IDX[300]].to_numpy(), k_ar_diff=m.k_ar_diff_, coint_rank=1, deterministic="co").fit()
    expected = res.predict(steps=h)[-1, 0] - levels["log_fx"].iloc[300]
    assert m.predict(origins_frame(IDX[300], IDX[300]))[0] == pytest.approx(expected)


def test_vecm_never_looks_ahead(levels):
    h = 2
    train, test = origins_frame(IDX[0], IDX[298]), origins_frame(IDX[300], IDX[310])
    base = VECMForecaster(levels, h).fit(train, None)
    origin = test.index[4]
    shocked = VECMForecaster(_perturbed_after(levels, origin), h).fit(train, None)
    assert np.allclose(base.coefs_, shocked.coefs_)
    assert base.predict(test.loc[:origin])[-1] == pytest.approx(shocked.predict(test.loc[:origin])[-1])


def test_models_require_target_column(var_series, levels):
    with pytest.raises(ValueError):
        VARForecaster(var_series.rename(columns={"dlog_fx": "fx"}), 1)
    with pytest.raises(ValueError):
        VECMForecaster(levels.rename(columns={"log_fx": "fx"}), 1)
