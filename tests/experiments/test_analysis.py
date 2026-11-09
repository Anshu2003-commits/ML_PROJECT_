import numpy as np
import pandas as pd
import pytest

from fxforecast.experiments import analysis as an

ORIGINS = pd.date_range("2010-01-01", "2022-12-01", freq="MS")
MODELS = [("fx_only", "random_walk"), ("fx_only", "drift"), ("fx_only", "ar"), ("fx_plus_macro", "var"),
          ("fx_plus_macro", "vecm"), ("fx_only", "svr"), ("fx_plus_macro", "svr"),
          ("fx_only", "lstm"), ("fx_plus_macro", "lstm")]


@pytest.fixture
def preds():
    """Synthetic forecasts: 'var' sees a real signal, every other model is noise around 0."""
    rng = np.random.default_rng(0)
    signal = rng.normal(scale=0.01, size=len(ORIGINS))
    y = signal + rng.normal(scale=0.005, size=len(ORIGINS))
    fx0 = 70 * np.exp(np.cumsum(rng.normal(scale=0.01, size=len(ORIGINS))))
    rows = []
    for fs, model in MODELS:
        if model == "random_walk":
            yp = np.zeros_like(y)
        elif model == "var":
            yp = signal
        else:
            yp = rng.normal(scale=0.002, size=len(y))
        rows.append(pd.DataFrame({
            "scheme": "walk_forward", "horizon": 1, "feature_set": fs, "model": model,
            "origin": ORIGINS, "target_date": ORIGINS + pd.DateOffset(months=1),
            "y_true": y, "y_pred": yp, "fx_origin": fx0,
            "fx_true": fx0 * np.exp(y), "fx_pred": fx0 * np.exp(yp),
        }))
    return pd.concat(rows, ignore_index=True)


def test_vs_benchmarks_flags_the_informed_model(preds):
    t = an.vs_benchmarks(preds)
    var = t.loc[("walk_forward", 1, "var (FX+macro)")]
    assert var["theil_u"] < 0.7 and var["dm_rw"] > 0 and var["p_rw"] < 0.01
    assert np.isnan(t.loc[("walk_forward", 1, "drift (FX only)"), "p_drift"])
    assert not any("random_walk" in m for m in t.index.get_level_values("model"))


def test_macro_value_detects_var_gain_over_ar(preds):
    t = an.macro_value(preds).droplevel(["scheme", "horizon"])
    assert t.loc["econometric (AR -> VAR)", "macro_helps"] == "yes"
    assert t.loc["econometric (AR -> VAR)", "rmse_change_pct"] < 0
    assert set(t.index) == set(an.NESTED)


def test_multivariate_pairs_covers_all_pairs(preds):
    t = an.multivariate_pairs(preds)
    assert len(t) == 6                                  # 4 models -> 6 pairs
    row = t.loc[("walk_forward", 1, "var (FX+macro)", "vecm (FX+macro)")]
    assert row["dm_stat"] < 0 and row["more_accurate"].startswith("var")


def test_subperiods_has_three_periods_and_no_benchmark_row(preds):
    t = an.subperiods(preds)
    assert list(t.columns) == list(an.PERIODS)
    assert t.loc[(1, "var (FX+macro)")].lt(1).all()
    assert not any(m.startswith("random_walk") for m in t.index.get_level_values("model"))


def test_publication_lag_table(preds):
    lagged = preds.copy()
    lagged = lagged[lagged.model.isin(["random_walk", "svr", "lstm"])]
    t = an.publication_lag(preds, {1: lagged})
    assert list(t.columns) == ["lag 0", "lag 1"]
    assert ("walk_forward", 1, "svr (FX+macro)") in t.index


def test_latest_run_picks_newest_completed(tmp_path):
    for name in ("final-20260101T000000Z-aaaa", "final-20260102T000000Z-bbbb", "final-20260103T000000Z-cccc"):
        (tmp_path / name).mkdir()
    for name in ("final-20260101T000000Z-aaaa", "final-20260102T000000Z-bbbb"):
        (tmp_path / name / "predictions.csv").write_text("x\n", encoding="utf-8")
    assert an.latest_run("final", tmp_path).name == "final-20260102T000000Z-bbbb"   # newest *completed*
    with pytest.raises(FileNotFoundError):
        an.latest_run("nothing", tmp_path)
