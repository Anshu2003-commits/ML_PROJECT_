from datetime import date

import pandas as pd
import pytest

from fxforecast.data.validate import ValidationError, coverage_report, validate_in_window, validate_series


def monthly(start="2020-01-01", n=6, name="X"):
    return pd.Series(range(n), index=pd.date_range(start, periods=n, freq="MS"), name=name, dtype=float)


def test_clean_series_passes():
    validate_series(monthly(), today=date(2026, 1, 1))


def test_missing_month_is_caught():
    s = monthly().drop(pd.Timestamp("2020-03-01"))
    with pytest.raises(ValidationError, match="missing month"):
        validate_series(s, today=date(2026, 1, 1))


def test_duplicate_date_is_caught():
    s = monthly()
    s = pd.concat([s, s.iloc[[1]]]).sort_index()
    with pytest.raises(ValidationError, match="duplicate"):
        validate_series(s, today=date(2026, 1, 1))


def test_non_month_start_is_caught():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-15", "2020-02-15"]), name="X")
    with pytest.raises(ValidationError, match="start-of-month"):
        validate_series(s, today=date(2026, 1, 1))


def test_future_date_is_caught():
    with pytest.raises(ValidationError, match="future"):
        validate_series(monthly("2025-11-01"), today=date(2026, 1, 1))


def test_unsorted_is_caught():
    with pytest.raises(ValidationError, match="sorted"):
        validate_series(monthly().iloc[::-1], today=date(2026, 1, 1))


def test_empty_is_caught():
    with pytest.raises(ValidationError, match="no observations"):
        validate_series(pd.Series([], dtype=float, index=pd.DatetimeIndex([]), name="X"))


def test_gap_outside_window_is_reported_not_raised():
    s = monthly(n=12).drop(pd.Timestamp("2020-11-01"))
    issues = validate_in_window({"x": s}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"),
                                today=date(2026, 1, 1))
    assert len(issues) == 1 and "missing month" in issues[0]


def test_gap_inside_window_raises():
    s = monthly(n=12).drop(pd.Timestamp("2020-03-01"))
    with pytest.raises(ValidationError, match="missing month"):
        validate_in_window({"x": s}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"),
                           today=date(2026, 1, 1))


def test_coverage_report():
    rep = coverage_report({"fx": monthly(name="EXINUS", n=3)})
    assert rep.loc["fx"].to_dict() == {"fred_id": "EXINUS", "first": "2020-01", "last": "2020-03", "n_obs": 3}
