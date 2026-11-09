import numpy as np
import pandas as pd
import pytest

from fxforecast.evaluation.backtest import attach_levels, backtest
from fxforecast.evaluation.splits import walk_forward_folds
from fxforecast.models.baselines import Drift, RandomWalk

IDX = pd.date_range("2000-01-01", periods=60, freq="MS")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"dlog_fx_lag0": rng.normal(size=60)}, index=IDX)
    y = pd.Series(rng.normal(size=60), index=IDX, name="y_h1")
    return X, y


class Spy(Drift):
    """Records the training index it was given."""
    seen: list = []

    def fit(self, X, y):
        Spy.seen.append(X.index)
        return super().fit(X, y)


def test_backtest_uses_only_fold_training_rows(data):
    X, y = data
    folds = walk_forward_folds(X.index, pd.Timestamp("2003-01-01"), 12, 1)
    Spy.seen = []
    out = backtest(Spy, X, y, folds)
    assert [list(s) for s in Spy.seen] == [list(f.train) for f in folds]
    assert list(out["origin"]) == [d for f in folds for d in f.test]
    expected_first = y.loc[folds[0].train].mean()
    assert out["y_pred"].iloc[0] == pytest.approx(expected_first)


def test_backtest_rejects_misaligned_inputs(data):
    X, y = data
    with pytest.raises(ValueError):
        backtest(RandomWalk, X, y.iloc[1:], [])


def test_attach_levels_random_walk_equals_last_value():
    fx = pd.Series([70.0, 71.0, 72.0, 73.0], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    preds = pd.DataFrame({"origin": fx.index[:2], "y_pred": [0.0, 0.0],
                          "y_true": np.log([71 / 70, 72 / 71])})
    out = attach_levels(preds, fx, horizon=1)
    assert list(out["fx_pred"]) == [70.0, 71.0]        # "no change"
    assert list(out["fx_true"]) == [71.0, 72.0]
    assert np.allclose(out["fx_origin"] * np.exp(out["y_true"]), out["fx_true"])


def test_attach_levels_needs_target_in_data():
    fx = pd.Series([70.0, 71.0], index=pd.date_range("2020-01-01", periods=2, freq="MS"))
    preds = pd.DataFrame({"origin": fx.index[1:], "y_pred": [0.0], "y_true": [0.0]})
    with pytest.raises(ValueError):
        attach_levels(preds, fx, horizon=1)
