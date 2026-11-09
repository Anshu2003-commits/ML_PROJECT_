import numpy as np
import pandas as pd
import pytest

from fxforecast.evaluation.metrics import hit_rate, mae, mape, metrics_table, mpe, rmse

ACTUAL = [100.0, 200.0]
PRED = [110.0, 190.0]


def test_hand_computed_values():
    assert rmse(ACTUAL, PRED) == pytest.approx(10.0)
    assert mae(ACTUAL, PRED) == pytest.approx(10.0)
    assert mape(ACTUAL, PRED) == pytest.approx((0.10 + 0.05) / 2)
    assert mpe(ACTUAL, PRED) == pytest.approx((0.10 - 0.05) / 2)


def test_hit_rate_counts_only_called_directions():
    assert hit_rate([0.1, -0.2, 0.3], [0.5, 0.1, 0.0]) == pytest.approx(0.5)
    assert np.isnan(hit_rate([0.1, -0.2], [0.0, 0.0]))


def test_errors_on_bad_input():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        mape([0.0, 1.0], [1.0, 1.0])


def _preds(model, fx_pred, y_pred, origins=("2020-01-01", "2020-02-01")):
    return pd.DataFrame({
        "origin": pd.to_datetime(origins), "scheme": "s", "horizon": 1, "feature_set": "fx_only",
        "model": model, "fx_true": [100.0, 102.0], "fx_pred": fx_pred,
        "y_true": [0.01, 0.02], "y_pred": y_pred,
    })


def test_metrics_table_theil_u_relative_to_random_walk():
    preds = pd.concat([_preds("random_walk", [98.0, 100.0], [0.0, 0.0]),
                       _preds("good", [99.0, 101.0], [0.01, 0.01])])
    table = metrics_table(preds, ["scheme", "horizon", "feature_set", "model"])
    assert table.loc[("s", 1, "fx_only", "random_walk"), "theil_u"] == pytest.approx(1.0)
    assert table.loc[("s", 1, "fx_only", "good"), "theil_u"] == pytest.approx(0.5)
    assert table["n"].dtype.kind == "i"


def test_metrics_table_requires_benchmark_on_same_origins():
    preds = pd.concat([_preds("random_walk", [98.0, 100.0], [0.0, 0.0]),
                       _preds("other", [99.0, 101.0], [0.01, 0.01], origins=("2021-01-01", "2021-02-01"))])
    with pytest.raises(ValueError, match="same origin"):
        metrics_table(preds, ["scheme", "horizon", "feature_set", "model"])
