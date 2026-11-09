import pytest

from fxforecast.config import ConfigError, load_config, series_ids


def test_project_config_loads():
    cfg = load_config()
    ids = series_ids(cfg)
    assert ids["fx"] == "EXINUS"
    assert ids["cpi_us"] == "CPIAUCSL"
    assert len(ids) == 1 + 2 * len(cfg["data"]["series"])


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_missing_country_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  raw_dir: r\n  processed_dir: p\n  start: 2000-01-01\n  target: EXINUS\n"
        "  series:\n    cpi: {us: CPIAUCSL}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cpi"):
        load_config(p)


def test_bad_date_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  raw_dir: r\n  processed_dir: p\n  start: not-a-date\n  target: EXINUS\n"
        "  series:\n    cpi: {us: A, in: B}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="start"):
        load_config(p)
