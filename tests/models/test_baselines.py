import numpy as np
import pandas as pd
import pytest

from fxforecast.models.baselines import AutoRegressive, Drift, RandomWalk


def lagged(series: np.ndarray, n_lags: int) -> tuple[pd.DataFrame, pd.Series]:
    s = pd.Series(series)
    X = pd.DataFrame({f"dlog_fx_lag{k}": s.shift(k) for k in range(n_lags)})
    y = s.shift(-1).rename("y_h1")
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def test_random_walk_predicts_no_change():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert list(RandomWalk().fit(X, pd.Series([5.0, 6, 7])).predict(X)) == [0, 0, 0]


def test_drift_predicts_training_mean():
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    pred = Drift().fit(X, pd.Series([0.01, 0.02, 0.03])).predict(X.iloc[:2])
    assert pred == pytest.approx([0.02, 0.02])


def test_ar_recovers_ar1_coefficient():
    rng = np.random.default_rng(0)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.001 + 0.5 * x[t - 1] + 0.01 * rng.normal()
    X, y = lagged(x, 6)
    model = AutoRegressive(max_lag=6, criterion="bic").fit(X, y)
    assert model.p_ == 1
    assert model.coef_[1] == pytest.approx(0.5, abs=0.05)


def test_ar_on_white_noise_picks_no_lags_with_bic():
    X, y = lagged(np.random.default_rng(1).normal(size=1000), 6)
    assert AutoRegressive(max_lag=6, criterion="bic").fit(X, y).p_ == 0


def test_ar_prediction_matches_coefficients():
    X, y = lagged(np.random.default_rng(2).normal(size=300), 3)
    m = AutoRegressive(max_lag=3).fit(X, y)
    row = X.iloc[[0]]
    manual = m.coef_[0] + sum(m.coef_[k + 1] * row[f"dlog_fx_lag{k}"].iloc[0] for k in range(m.p_))
    assert m.predict(row)[0] == pytest.approx(manual)


def test_ar_needs_lag_columns():
    with pytest.raises(KeyError):
        AutoRegressive(max_lag=2).fit(pd.DataFrame({"x": [1.0, 2, 3, 4]}), pd.Series([1.0, 2, 3, 4]))


def test_ar_rejects_unknown_criterion():
    with pytest.raises(ValueError):
        AutoRegressive(criterion="r2")
