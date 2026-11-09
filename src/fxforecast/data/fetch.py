"""Download FRED series as CSV and cache them in data/raw/.

Each series is saved as ``<ID>.csv`` (FRED's own format) next to a
``<ID>.meta.json`` that records when and where it was fetched plus a SHA-256
hash, so every experiment can say exactly which data snapshot it used.

Downloads are sequential: FRED tends to drop parallel requests.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

log = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# (url, timeout_seconds) -> CSV text. Injectable so tests never touch the network.
HttpGet = Callable[[str, float], str]


class FetchError(RuntimeError):
    """Raised when a series cannot be downloaded after all retries."""


def _requests_get(url: str, timeout: float) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_series(
    series_id: str,
    raw_dir: Path,
    *,
    refresh: bool = False,
    http_get: HttpGet = _requests_get,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 30.0,
) -> Path:
    """Make sure ``<raw_dir>/<series_id>.csv`` exists and return its path.

    Uses the cached file unless ``refresh`` is True. Retries with exponential
    backoff, then raises FetchError.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{series_id}.csv"
    if csv_path.exists() and not refresh:
        log.debug("cache hit %s", series_id)
        return csv_path

    url = FRED_CSV_URL.format(series_id=series_id)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            text = http_get(url, timeout)
            _parse_fred_csv(text, series_id)  # reject HTML error pages / bad payloads before caching
        except Exception as exc:  # noqa: BLE001 - any failure is retried, then surfaced
            last_error = exc
            log.warning("fetch %s attempt %d/%d failed: %s", series_id, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff_seconds * 2 ** (attempt - 1))
            continue
        csv_path.write_text(text, encoding="utf-8")
        _write_meta(csv_path, series_id, url, text)
        log.info("fetched %s in %.2fs", series_id, time.perf_counter() - started)
        return csv_path
    raise FetchError(f"Could not download {series_id} after {retries} attempts: {last_error}")


def _write_meta(csv_path: Path, series_id: str, url: str, text: str) -> None:
    series = _parse_fred_csv(text, series_id)
    meta = {
        "series_id": series_id,
        "url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_obs": int(series.notna().sum()),
        "first": series.first_valid_index().strftime("%Y-%m-%d"),
        "last": series.last_valid_index().strftime("%Y-%m-%d"),
    }
    csv_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse FRED's two-column CSV (date, value). Missing values are '.' or empty."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2:
        raise ValueError(f"{series_id}: expected 2 columns, got {list(df.columns)}")
    dates = pd.to_datetime(df.iloc[:, 0], errors="raise")
    values = pd.to_numeric(df.iloc[:, 1], errors="raise")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=series_id)
    if series.notna().sum() == 0:
        raise ValueError(f"{series_id}: no observations")
    return series


def load_series(series_id: str, raw_dir: Path) -> pd.Series:
    """Read a cached series. Missing observations are dropped."""
    csv_path = raw_dir / f"{series_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run the fetch step first")
    return _parse_fred_csv(csv_path.read_text(encoding="utf-8"), series_id).dropna()
