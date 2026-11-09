"""Command-line entry point for the data layer.

    python -m fxforecast.data             # use cached raw files, fetch only missing ones
    python -m fxforecast.data --refresh   # re-download everything from FRED

Fetches every series in the config, validates it, prints a coverage report and
writes data/processed/panel.parquet.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from fxforecast.config import DEFAULT_CONFIG, load_config, resolve_path, series_ids
from fxforecast.data.fetch import fetch_series, load_series
from fxforecast.data.panel import build_panel, common_window
from fxforecast.data.validate import coverage_report, validate_in_window

log = logging.getLogger("fxforecast.data")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--refresh", action="store_true", help="re-download all series")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    processed_dir = resolve_path(cfg["data"]["processed_dir"])

    raw = {}
    for column, fred_id in series_ids(cfg).items():
        fetch_series(fred_id, raw_dir, refresh=args.refresh)
        raw[column] = load_series(fred_id, raw_dir)
    print(coverage_report(raw).to_string())

    # Strict checks apply to the analysis window; problems outside it are only reported.
    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(cfg["data"]["end"]) if cfg["data"].get("end") else common_window(raw)[1]
    for issue in validate_in_window(raw, start, end):
        log.warning("outside-window issue (ignored): %s", issue)

    panel = build_panel(raw, start, end)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "panel.parquet"
    panel.to_parquet(out)
    print(f"\nPanel: {panel.index[0]:%Y-%m} to {panel.index[-1]:%Y-%m}, "
          f"{len(panel)} months x {panel.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
