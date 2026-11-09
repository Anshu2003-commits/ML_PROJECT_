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


def test_plots_write_png_files(tmp_path):
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 4)), index=idx, columns=list("abcd"))
    paths = [
        plots.small_multiples(df, tmp_path / "sm.png", title="t", zero_line=True,
                              shade=("2001-01-01", "2002-01-01", "test")),
        plots.correlation_heatmap(df, tmp_path / "corr.png", title="t"),
        plots.acf_plot(df["a"], tmp_path / "acf.png", title="t", nlags=12),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
