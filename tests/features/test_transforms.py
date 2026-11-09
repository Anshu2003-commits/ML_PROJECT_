import numpy as np
import pandas as pd
import pytest

from fxforecast.features.transforms import (
    MACRO_FEATURES,
    TARGET,
    difference,
    log_diff,
    macro_differentials,
    relative_levels,
    trade_balance_ratio,
)

RAW_COLUMNS = ["fx"] + [f"{v}_{c}" for v in
                        ("cpi", "ip", "rate", "m3", "reserves", "equity", "exports", "imports")
                        for c in ("us", "in")]


@pytest.fixture
def panel():
    idx = pd.date_range("2000-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(np.exp(rng.normal(0, 0.01, (36, len(RAW_COLUMNS))).cumsum(axis=0)) * 100,
                      index=idx, columns=RAW_COLUMNS)
    df["rate_us"], df["rate_in"] = 2.0, 6.0
    return df


def test_log_diff_matches_formula():
    s = pd.Series([100.0, 110.0, 121.0])
    assert log_diff(s).iloc[2] == pytest.approx(np.log(121 / 110))
    assert log_diff(s, 2).iloc[2] == pytest.approx(np.log(1.21))


def test_log_diff_rejects_non_positive():
    with pytest.raises(ValueError):
        log_diff(pd.Series([1.0, 0.0, 2.0], name="x"))


def test_trade_balance_ratio_is_bounded_and_unit_free():
    x, m = pd.Series([3.0, 1.0]), pd.Series([1.0, 3.0])
    assert list(trade_balance_ratio(x, m)) == [0.5, -0.5]
    assert list(trade_balance_ratio(x * 1e9, m * 1e9)) == [0.5, -0.5]


def test_macro_differentials_shape_and_values(panel):
    out = macro_differentials(panel)
    assert list(out.columns) == [TARGET, *MACRO_FEATURES]
    assert out.index[0] == panel.index[12]           # first 12 months dropped
    assert not out.isna().any().any()
    t = panel.index[20]
    assert out.loc[t, "rate_diff"] == -4.0
    expected_infl = (np.log(panel.loc[t, "cpi_us"] / panel["cpi_us"].shift(12)[t])
                     - np.log(panel.loc[t, "cpi_in"] / panel["cpi_in"].shift(12)[t]))
    assert out.loc[t, "infl_diff"] == pytest.approx(expected_infl)


def test_features_use_only_past_data(panel):
    """Changing a future value must not change any earlier feature row (no look-ahead)."""
    base = macro_differentials(panel)
    shocked = panel.copy()
    shocked.iloc[30:] *= 3
    after = macro_differentials(shocked)
    cutoff = panel.index[29]
    pd.testing.assert_frame_equal(base.loc[:cutoff], after.loc[:cutoff])


def test_difference_applies_per_column_orders():
    df = pd.DataFrame({"a": [1.0, 2.0, 4.0, 7.0], "b": [5.0, 5.0, 5.0, 5.0]})
    out = difference(df, {"a": 1})
    assert list(out["a"]) == [1.0, 2.0, 3.0]
    assert list(out["b"]) == [5.0, 5.0, 5.0]


def test_relative_levels_columns(panel):
    out = relative_levels(panel)
    assert list(out.columns) == ["log_fx", "rel_price", "rel_output", "rel_money", "rate_diff"]
    assert out["log_fx"].iloc[0] == pytest.approx(np.log(panel["fx"].iloc[0]))
