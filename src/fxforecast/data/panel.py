"""Align validated series into one monthly panel (one row per month)."""
from __future__ import annotations

import pandas as pd

from fxforecast.data.validate import ValidationError


def common_window(series_by_column: dict[str, pd.Series]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Latest first month and earliest last month across all series."""
    start = max(s.index[0] for s in series_by_column.values())
    end = min(s.index[-1] for s in series_by_column.values())
    if start > end:
        raise ValidationError("Series do not overlap in time")
    return start, end


def build_panel(
    series_by_column: dict[str, pd.Series],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Combine series into a DataFrame over [start, end] with no missing values.

    ``end=None`` means the last month for which every series has data. Raises
    if a requested window falls outside what the data covers, rather than
    filling gaps silently.
    """
    first_common, last_common = common_window(series_by_column)
    start = pd.Timestamp(start)
    end = last_common if end is None else pd.Timestamp(end)
    if start < first_common:
        late = [c for c, s in series_by_column.items() if s.index[0] > start]
        raise ValidationError(f"Requested start {start:%Y-%m} is before data begins for {late}")
    if end > last_common:
        short = [c for c, s in series_by_column.items() if s.index[-1] < end]
        raise ValidationError(f"Requested end {end:%Y-%m} is after data ends for {short}")
    if start > end:
        raise ValidationError(f"Start {start:%Y-%m} is after end {end:%Y-%m}")

    panel = pd.DataFrame(series_by_column).loc[start:end]
    panel.index.name = "date"
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise ValidationError(f"Panel has missing values in {bad}")
    return panel
