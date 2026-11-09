"""Checks every raw series must pass before it is used.

These are deterministic assertions: a series that fails stops the pipeline
with a clear message instead of silently producing wrong results.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


class ValidationError(ValueError):
    """Raised when a series breaks one of the data rules."""


def validate_series(series: pd.Series, *, today: date | None = None) -> None:
    """Check that a monthly series is clean.

    Rules: non-empty, numeric, sorted, no duplicate dates, every date is the
    first of a month, no date after today, and no missing month between the
    first and last observation.
    """
    name = series.name or "series"
    if series.empty:
        raise ValidationError(f"{name}: no observations")
    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(f"{name}: values are not numeric")
    if series.isna().any():
        raise ValidationError(f"{name}: contains NaN values")
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValidationError(f"{name}: index is not a DatetimeIndex")
    if idx.has_duplicates:
        raise ValidationError(f"{name}: duplicate dates {list(idx[idx.duplicated()][:3])}")
    if not idx.is_monotonic_increasing:
        raise ValidationError(f"{name}: dates are not sorted")
    if (idx.day != 1).any():
        raise ValidationError(f"{name}: not monthly start-of-month data")
    today = today or date.today()
    if idx[-1].date() > today:
        raise ValidationError(f"{name}: has a future date {idx[-1].date()}")
    expected = pd.date_range(idx[0], idx[-1], freq="MS")
    missing = expected.difference(idx)
    if len(missing):
        raise ValidationError(
            f"{name}: {len(missing)} missing month(s), e.g. {[d.strftime('%Y-%m') for d in missing[:3]]}"
        )


def validate_in_window(
    series_by_column: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    today: date | None = None,
) -> list[str]:
    """Strictly validate each series inside [start, end]; return issues found outside it.

    A gap in, say, 1951 or late 2025 should not block an analysis of 1994-2023,
    but it is still reported so the user knows about it.
    """
    outside_issues = []
    for column, s in series_by_column.items():
        validate_series(s.loc[start:end], today=today)
        try:
            validate_series(s, today=today)
        except ValidationError as exc:
            outside_issues.append(f"{column}: {exc}")
    return outside_issues


def coverage_report(series_by_column: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per column: FRED ID, first and last month, number of observations."""
    rows = [
        {
            "column": col,
            "fred_id": s.name,
            "first": s.index[0].strftime("%Y-%m"),
            "last": s.index[-1].strftime("%Y-%m"),
            "n_obs": len(s),
        }
        for col, s in series_by_column.items()
    ]
    return pd.DataFrame(rows).set_index("column")
