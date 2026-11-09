import pytest

from fxforecast.config import PROJECT_ROOT, ConfigError, load_config, series_ids


def test_project_config_loads():
    cfg = load_config()
    ids = series_ids(cfg)
    assert ids["fx"] == "EXINUS"
    assert ids["cpi_us"] == "CPIAUCSL"
    assert len(ids) == 1 + 2 * len(cfg["data"]["series"])


def test_base_config_is_inherited_and_overridden(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "data:\n  raw_dir: r\n  processed_dir: p\n  start: 2000-01-01\n  target: EXINUS\n"
        "  series:\n    cpi: {us: A, in: B}\n    rate: {us: C, in: D}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\ndata:\n  processed_dir: p2\n  series:\n    rate: {us: C, in: RBI}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["data"]["processed_dir"] == "p2"
    assert cfg["data"]["raw_dir"] == "r"                         # inherited
    assert cfg["data"]["series"]["cpi"] == {"us": "A", "in": "B"}  # sibling key kept
    assert cfg["data"]["series"]["rate"]["in"] == "RBI"
    assert "base" not in cfg


def test_project_rbi_variant_config():
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment_rbi.yaml")
    assert series_ids(cfg)["rate_in"] == "RBI_WACR"
    assert series_ids(cfg)["cpi_in"] == "INDCPIALLMINMEI"


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
