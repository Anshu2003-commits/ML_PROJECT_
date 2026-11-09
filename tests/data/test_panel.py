import pandas as pd
import pytest

from fxforecast.data.panel import build_panel, common_window
from fxforecast.data.validate import ValidationError


def monthly(start, n, name):
    return pd.Series(range(n), index=pd.date_range(start, periods=n, freq="MS"), name=name, dtype=float)


@pytest.fixture
def series():
    return {
        "fx": monthly("2000-01-01", 24, "A"),   # 2000-01 .. 2001-12
        "cpi_us": monthly("2000-03-01", 12, "B"),  # 2000-03 .. 2001-02
    }


def test_common_window(series):
    assert common_window(series) == (pd.Timestamp("2000-03-01"), pd.Timestamp("2001-02-01"))


def test_default_end_is_last_common_month(series):
    panel = build_panel(series, "2000-03-01")
    assert panel.index[0] == pd.Timestamp("2000-03-01")
    assert panel.index[-1] == pd.Timestamp("2001-02-01")
    assert list(panel.columns) == ["fx", "cpi_us"]
    assert not panel.isna().any().any()


def test_explicit_end(series):
    panel = build_panel(series, "2000-03-01", "2000-06-01")
    assert len(panel) == 4


def test_start_before_coverage_names_the_late_series(series):
    with pytest.raises(ValidationError, match="cpi_us"):
        build_panel(series, "2000-01-01")


def test_end_after_coverage_names_the_short_series(series):
    with pytest.raises(ValidationError, match="cpi_us"):
        build_panel(series, "2000-03-01", "2001-06-01")


def test_non_overlapping_series_fail():
    with pytest.raises(ValidationError, match="overlap"):
        common_window({"a": monthly("2000-01-01", 3, "A"), "b": monthly("2010-01-01", 3, "B")})
