import numpy as np
import pandas as pd

from fxforecast.reporting import plots
from fxforecast.reporting.tables import to_markdown


def test_to_markdown_formats_numbers():
    df = pd.DataFrame({"p": [0.12345, 1e-9], "ok": [True, False], "n": [3, 4]},
                      index=pd.Index(["a", "b"], name="series"))
    md = to_markdown(df).splitlines()
    assert md[0] == "| series | p | ok | n |"
    assert md[2] == "| a | 0.123 | True | 3 |"
    assert md[3] == "| b | 1.0e-09 | False | 4 |"


def test_to_markdown_multiindex_and_int_column():
    df = pd.DataFrame({"n": [29], "rmse": [0.84699]},
                      index=pd.MultiIndex.from_tuples([("fx_only", "ar")], names=["feature_set", "model"]))
    md = to_markdown(df, 2).splitlines()
    assert md[0] == "| feature_set | model | n | rmse |"
    assert md[2] == "| fx_only | ar | 29 | 0.85 |"


def test_plots_write_png_files(tmp_path):
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 4)), index=idx, columns=list("abcd"))
    paths = [
        plots.small_multiples(df, tmp_path / "sm.png", title="t", zero_line=True,
                              shade=("2001-01-01", "2002-01-01", "test")),
        plots.correlation_heatmap(df, tmp_path / "corr.png", title="t"),
        plots.acf_plot(df["a"], tmp_path / "acf.png", title="t", nlags=12),
    ]
    paths += [
        plots.forecast_paths(df["a"], {"x": df["b"], "y": df["c"]}, tmp_path / "paths.png", title="t"),
        plots.theil_u_dots(pd.DataFrame({"h=1": [0.9, 1.1], "h=3": [1.0, 0.95]},
                                        index=pd.MultiIndex.from_tuples([("m", "FX only"), ("m", "FX + macro")])),
                           tmp_path / "dots.png", title="t", reference={"h=1": 0.98}),
        plots.ratio_heatmap(pd.DataFrame({"p1": [0.9, 1.2]}, index=["a", "b"]), tmp_path / "heat.png", title="t"),
        plots.grouped_bars(pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]}, index=["a", "b"]),
                           tmp_path / "bars.png", title="t", ylabel="v"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
