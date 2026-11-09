"""RBI weighted average call money rate (WACR), from a manually retrieved DBIE table.

RBI's servers refuse automated downloads, so the table is retrieved once by
hand (see ``data/raw/rbi/rbi_wacr_monthly_table.meta.json``) and committed.
This module only parses that file; it never touches the network.

The table is laid out by Indian fiscal year (April-March): one row per
fiscal year and range (High / Low / Average), one column per month.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SERIES_ID = "RBI_WACR"
FILE = "rbi/rbi_wacr_monthly_table.csv"
MONTHS = ["apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar"]


def fiscal_month_to_date(fiscal_year: str, month: str) -> pd.Timestamp:
    """'1994-95' + 'apr' -> 1994-04-01; '1994-95' + 'jan' -> 1995-01-01."""
    start = int(fiscal_year[:4])
    pos = MONTHS.index(month)
    year = start if pos < 9 else start + 1
    return pd.Timestamp(year=year, month=(pos + 3) % 12 + 1, day=1)


def parse_wacr_table(table: pd.DataFrame, value: str = "Average") -> pd.Series:
    """Turn the fiscal-year x month table into a monthly series (missing cells dropped)."""
    expected = {"fiscal_year", "range", *MONTHS}
    if not expected <= set(table.columns):
        raise ValueError(f"RBI table is missing columns {sorted(expected - set(table.columns))}")
    rows = table[table["range"] == value]
    if rows.empty:
        raise ValueError(f"No '{value}' rows in the RBI table")
    if rows["fiscal_year"].duplicated().any():
        raise ValueError("Duplicate fiscal years in the RBI table")
    points = {
        fiscal_month_to_date(fy, m): v
        for fy, row in rows.set_index("fiscal_year").iterrows()
        for m, v in row[MONTHS].items()
        if pd.notna(v)
    }
    series = pd.Series(points, name=SERIES_ID, dtype=float).sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series


def load_rbi_wacr(raw_dir: Path, value: str = "Average") -> pd.Series:
    path = Path(raw_dir) / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see the .meta.json next to it for how it was retrieved")
    return parse_wacr_table(pd.read_csv(path, dtype={"fiscal_year": str}), value)


# Series that are read from committed local files instead of being fetched from FRED.
LOCAL_SERIES = {SERIES_ID: load_rbi_wacr}
