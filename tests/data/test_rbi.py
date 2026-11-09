import hashlib
import json

import pandas as pd
import pytest

from fxforecast.config import resolve_path
from fxforecast.data.rbi import FILE, fiscal_month_to_date, load_rbi_wacr, parse_wacr_table
from fxforecast.data.validate import validate_series

RAW = resolve_path("data/raw")
MONTHS = ["apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar"]


def test_fiscal_year_mapping():
    assert fiscal_month_to_date("1994-95", "apr") == pd.Timestamp("1994-04-01")
    assert fiscal_month_to_date("1994-95", "dec") == pd.Timestamp("1994-12-01")
    assert fiscal_month_to_date("1994-95", "jan") == pd.Timestamp("1995-01-01")
    assert fiscal_month_to_date("1994-95", "mar") == pd.Timestamp("1995-03-01")


def _table(rows):
    return pd.DataFrame(rows, columns=["fiscal_year", "range", *MONTHS, "annual"])


def test_parse_uses_average_row_and_drops_blank_months():
    t = _table([["2026-27", "High", *[9.0] * 12, 9.0],
                ["2026-27", "Average", 5.0, 5.5, *[None] * 10, None]])
    s = parse_wacr_table(t)
    assert list(s.index) == [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")]
    assert list(s) == [5.0, 5.5]


def test_parse_rejects_duplicates_and_bad_columns():
    row = ["2020-21", "Average", *[1.0] * 12, 1.0]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_wacr_table(_table([row, row]))
    with pytest.raises(ValueError, match="missing columns"):
        parse_wacr_table(pd.DataFrame({"fiscal_year": ["2020-21"], "range": ["Average"]}))


@pytest.fixture(scope="module")
def wacr():
    return load_rbi_wacr(RAW)


def test_committed_file_matches_recorded_hash():
    meta = json.loads((RAW / "rbi" / "rbi_wacr_monthly_table.meta.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((RAW / FILE).read_bytes()).hexdigest() == meta["sha256"]


def test_committed_series_is_clean_and_covers_sample(wacr):
    validate_series(wacr)                                   # monthly, no gaps, no future dates
    assert wacr.index[0] <= pd.Timestamp("1994-04-01") and wacr.index[-1] >= pd.Timestamp("2023-01-01")
    assert (wacr > 0).all() and wacr.max() < 60


def test_known_episodes(wacr):
    assert wacr["1995-11-01"] == pytest.approx(34.83)      # Nov 1995 liquidity squeeze
    assert wacr["1998-01-01"] == pytest.approx(28.70)      # Jan 1998 rupee defence
    assert wacr["2008-10-01"] == pytest.approx(10.70)      # Oct 2008 global crisis


def test_matches_handbook_table_174(wacr):
    """Cross-check against the Handbook 2025-26 PDF (covers 2020-21 onwards)."""
    pdfplumber = pytest.importorskip("pdfplumber")
    pdf_path = RAW / "rbi" / "rbi_handbook_2025-26_table174.pdf"
    if not pdf_path.exists():
        pytest.skip("Handbook PDF not present")
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
    lines = text.splitlines()
    checked = 0
    for i, line in enumerate(lines):
        if line.strip() in {f"{y}-{str(y + 1)[2:]}" for y in range(2020, 2027)}:
            fy = line.strip()
            avg = next(l for l in lines[i + 1:i + 4] if l.startswith("Average")).split()[1:13]
            for month, value in zip(MONTHS, avg):
                if value != "-":
                    assert wacr[fiscal_month_to_date(fy, month)] == pytest.approx(float(value))
                    checked += 1
    assert checked >= 70
