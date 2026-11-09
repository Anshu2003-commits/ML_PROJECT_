"""End-to-end smoke test of the experiment runner on the real (committed) data snapshot."""
import json

import pandas as pd
import pytest

from fxforecast.config import load_config, resolve_path
from fxforecast.experiments import run as runner
from fxforecast.experiments.tracking import Run

PANEL = resolve_path("data/processed/panel.parquet")


@pytest.mark.skipif(not PANEL.exists(), reason="run `python -m fxforecast.data` first")
def test_baseline_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(Run, "root", tmp_path)
    cfg = load_config()
    cfg["experiments"]["walk_forward"]["horizons"] = [1]
    table, tracker = runner.run(cfg, "smoke", ["drift"])

    # random walk is added automatically as the benchmark
    models = set(table.index.get_level_values("model"))
    assert models == {"random_walk", "drift"}
    rep = table.xs("replication", level="scheme")
    assert set(rep["n"]) == {29}
    assert (table.xs("random_walk", level="model")["theil_u"] == 1.0).all()

    preds = pd.read_csv(tracker.dir / "predictions.csv")
    counts = preds.groupby(["scheme", "model"]).size().unstack()
    assert (counts["drift"] == counts["random_walk"]).all()   # identical origins per model
    info = json.loads((tracker.dir / "run.json").read_text(encoding="utf-8"))
    assert info["data_sha256"]["EXINUS"] and info["git_commit"]

    report = runner.write_report(table, tracker, tmp_path / "report", ["drift"])
    assert "Theil" in report.read_text(encoding="utf-8") or "theil_u" in report.read_text(encoding="utf-8")


def test_unknown_model_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Run, "root", tmp_path)
    with pytest.raises(ValueError, match="Unknown models"):
        runner.run(load_config(), "smoke", ["prophet"])
