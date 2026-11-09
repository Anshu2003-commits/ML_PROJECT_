import numpy as np
import pandas as pd
import pytest

from fxforecast.features.supervised import to_sequences
from fxforecast.models.lstm import LSTMForecaster, train_net

FAST = dict(lookback=(3,), hidden_units=(8,), max_epochs=40, patience=10, seeds=(0, 1), batch_size=16)


@pytest.fixture
def data():
    """y depends on the two most recent lags of x, so an LSTM can learn it."""
    rng = np.random.default_rng(0)
    n = 160
    x = rng.normal(size=n + 4)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    X = pd.DataFrame({f"dlog_fx_lag{k}": x[4 - k: 4 - k + n] for k in range(4)}, index=idx)
    X["infl_diff_lag0"], X["infl_diff_lag1"], X["infl_diff_lag2"] = rng.normal(size=(3, n))
    y = pd.Series(0.8 * X["dlog_fx_lag0"] - 0.5 * X["dlog_fx_lag1"] + 0.05 * rng.normal(size=n), index=idx)
    return X, y


def test_to_sequences_orders_oldest_first(data):
    X, _ = data
    seq = to_sequences(X, 3)
    assert seq.shape == (len(X), 3, 2)                       # 2 variables: dlog_fx, infl_diff
    assert seq[5, -1, 0] == pytest.approx(X["dlog_fx_lag0"].iloc[5])   # last step = origin month
    assert seq[5, 0, 0] == pytest.approx(X["dlog_fx_lag2"].iloc[5])
    assert seq[5, 0, 1] == pytest.approx(X["infl_diff_lag2"].iloc[5])


def test_to_sequences_needs_enough_lags(data):
    with pytest.raises(KeyError):
        to_sequences(data[0], 4)                             # infl_diff only has 3 lags


def test_network_can_overfit_a_tiny_batch():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 4, 2)).astype(np.float32)
    y = rng.normal(size=16).astype(np.float32)
    res = train_net(x, y, x, y, hidden=32, dropout=0.0, lr=1e-2, batch_size=16,
                    max_epochs=400, patience=400, seed=0)
    assert res.best_val < 0.02 * float(np.var(y))


def test_lstm_learns_signal_and_beats_zero_forecast(data):
    X, y = data
    m = LSTMForecaster(1, **{**FAST, "max_epochs": 150, "patience": 30}).fit(X.iloc[:130], y.iloc[:130])
    pred = m.predict(X.iloc[130:])
    mse_model = np.mean((pred - y.iloc[130:].to_numpy()) ** 2)
    mse_zero = np.mean(y.iloc[130:].to_numpy() ** 2)
    assert mse_model < 0.5 * mse_zero


def test_same_seeds_give_identical_forecasts(data):
    X, y = data
    a = LSTMForecaster(1, **FAST).fit(X.iloc[:120], y.iloc[:120]).predict(X.iloc[120:])
    b = LSTMForecaster(1, **FAST).fit(X.iloc[:120], y.iloc[:120]).predict(X.iloc[120:])
    assert np.array_equal(a, b)


def test_members_average_to_prediction_and_info(data):
    X, y = data
    m = LSTMForecaster(1, **FAST).fit(X.iloc[:120], y.iloc[:120])
    members = m.predict_members(X.iloc[120:])
    assert members.shape == (2, 40)
    assert np.allclose(members.mean(axis=0), m.predict(X.iloc[120:]))
    assert m.info()["lookback"] == 3 and len(m.info()["best_epochs"]) == 2


def test_fit_ignores_rows_it_is_not_given(data):
    """The model only sees the training rows passed in; later data cannot affect it."""
    X, y = data
    y_shocked = y.copy()
    y_shocked.iloc[120:] += 50
    a = LSTMForecaster(1, **FAST).fit(X.iloc[:120], y.iloc[:120]).predict(X.iloc[120:])
    b = LSTMForecaster(1, **FAST).fit(X.iloc[:120], y_shocked.iloc[:120]).predict(X.iloc[120:])
    assert np.array_equal(a, b)


def test_too_few_rows_raises(data):
    X, y = data
    with pytest.raises(ValueError, match="Too few"):
        LSTMForecaster(6, **FAST).fit(X.iloc[:30], y.iloc[:30])
