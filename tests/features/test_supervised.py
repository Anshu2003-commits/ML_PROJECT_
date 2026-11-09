import numpy as np
import pandas as pd
import pytest

from fxforecast.features.supervised import feature_columns, h_step_target, make_supervised
from fxforecast.features.transforms import MACRO_FEATURES, TARGET


@pytest.fixture
def feats():
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    rng = np.random.default_rng(1)
    cols = [TARGET, *MACRO_FEATURES]
    return pd.DataFrame(rng.normal(size=(40, len(cols))), index=idx, columns=cols)


def test_h_step_target_equals_log_level_change():
    log_fx = pd.Series(np.log([10.0, 11, 12, 13, 15]), index=pd.date_range("2000-01-01", periods=5, freq="MS"))
    y = h_step_target(log_fx.diff(), 2)
    assert y.iloc[1] == pytest.approx(np.log(13 / 11))   # origin Feb -> target Apr
    assert y.iloc[-2:].isna().all()                       # no target past the data


def test_fx_only_has_only_fx_lags(feats):
    X, _ = make_supervised(feats, "fx_only", n_lags=3, horizon=1)
    assert list(X.columns) == ["dlog_fx_lag0", "dlog_fx_lag1", "dlog_fx_lag2"]


def test_fx_plus_macro_columns(feats):
    X, _ = make_supervised(feats, "fx_plus_macro", n_lags=2, horizon=1)
    assert X.shape[1] == 2 * (1 + len(MACRO_FEATURES))
    assert "infl_diff_lag1" in X.columns


def test_lag_values_and_alignment(feats):
    X, y = make_supervised(feats, "fx_plus_macro", n_lags=3, horizon=3)
    t = X.index[5]
    pos = feats.index.get_loc(t)
    assert X.loc[t, "dlog_fx_lag2"] == feats[TARGET].iloc[pos - 2]
    assert X.loc[t, "rate_diff_lag0"] == feats["rate_diff"].iloc[pos]
    assert y.loc[t] == pytest.approx(feats[TARGET].iloc[pos + 1: pos + 4].sum())
    assert X.index.equals(y.index) and not X.isna().any().any()


def test_publication_lag_shifts_macro_only(feats):
    X, _ = make_supervised(feats, "fx_plus_macro", n_lags=1, horizon=1, publication_lag=2)
    t = X.index[0]
    pos = feats.index.get_loc(t)
    assert X.loc[t, "dlog_fx_lag0"] == feats[TARGET].iloc[pos]
    assert X.loc[t, "infl_diff_lag0"] == feats["infl_diff"].iloc[pos - 2]


def test_inputs_never_use_data_after_origin(feats):
    """Perturbing the future must leave every past input row unchanged."""
    X0, _ = make_supervised(feats, "fx_plus_macro", n_lags=4, horizon=6)
    shocked = feats.copy()
    cut = feats.index[25]
    shocked.loc[shocked.index > cut] += 100
    X1, _ = make_supervised(shocked, "fx_plus_macro", n_lags=4, horizon=6)
    pd.testing.assert_frame_equal(X0.loc[:cut], X1.loc[:cut])


@pytest.mark.parametrize("bad", [{"n_lags": 0, "horizon": 1}, {"n_lags": 1, "horizon": 0},
                                 {"n_lags": 1, "horizon": 1, "publication_lag": -1}])
def test_invalid_arguments(feats, bad):
    with pytest.raises(ValueError):
        make_supervised(feats, "fx_only", **bad)


def test_select_lags_rejects_zero():
    from fxforecast.features.supervised import select_lags
    with pytest.raises(ValueError):
        select_lags(pd.DataFrame(columns=["dlog_fx_lag0"]), 0)


def test_unknown_feature_set():
    with pytest.raises(ValueError):
        feature_columns("everything")
