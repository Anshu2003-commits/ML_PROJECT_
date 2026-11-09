"""Load and validate the YAML experiment config.

The rest of the code receives plain values from here, so no module hard-codes
dates, series IDs or paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
COUNTRIES = ("us", "in")


class ConfigError(ValueError):
    """Raised when the config file is missing required fields or is malformed."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config and check the fields the pipeline depends on.

    A config may start with ``base: <other.yaml>`` (relative to its own folder):
    the base is loaded first and this file's values override it key by key.
    """
    cfg = _read(Path(path), seen=set())
    _validate(cfg)
    return cfg


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    if path.resolve() in seen:
        raise ConfigError(f"Circular 'base' reference at {path}")
    seen.add(path.resolve())
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    return _merge(_read(path.parent / base, seen), cfg) if base else cfg


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _validate(cfg: dict[str, Any]) -> None:
    data = cfg.get("data")
    if not isinstance(data, dict):
        raise ConfigError("Config needs a 'data' section")
    for key in ("raw_dir", "processed_dir", "start", "target", "series"):
        if key not in data:
            raise ConfigError(f"Config 'data' section is missing '{key}'")
    series = data["series"]
    if not isinstance(series, dict) or not series:
        raise ConfigError("'data.series' must be a non-empty mapping")
    for var, ids in series.items():
        if not isinstance(ids, dict) or set(ids) != set(COUNTRIES):
            raise ConfigError(f"'data.series.{var}' must have exactly the keys {COUNTRIES}")
    for key in ("start", "end"):
        if data.get(key) is not None:
            try:
                pd.Timestamp(data[key])
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"'data.{key}' is not a valid date: {data[key]!r}") from exc


def resolve_path(relative: str | Path) -> Path:
    """Turn a config path (relative to the project root) into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def series_ids(cfg: dict[str, Any]) -> dict[str, str]:
    """Map panel column name -> FRED ID, e.g. {'fx': 'EXINUS', 'cpi_us': 'CPIAUCSL', ...}."""
    data = cfg["data"]
    ids = {"fx": data["target"]}
    for var, by_country in data["series"].items():
        for country in COUNTRIES:
            ids[f"{var}_{country}"] = by_country[country]
    return ids
