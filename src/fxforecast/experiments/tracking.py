"""Minimal experiment tracking: every run gets a folder that records how its numbers were made.

results/runs/<run_id>/
    config.yaml       exact config used
    run.json          run id, git commit, dirty flag, library versions, data hashes, timings
    predictions.csv   every out-of-sample forecast
    metrics.csv       summary table
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from fxforecast.config import PROJECT_ROOT

TRACKED_PACKAGES = ("numpy", "pandas", "statsmodels", "scikit-learn", "torch")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _data_hashes(raw_dir: Path) -> dict[str, str]:
    return {p.stem.replace(".meta", ""): json.loads(p.read_text(encoding="utf-8"))["sha256"][:12]
            for p in sorted(raw_dir.glob("*.meta.json"))}


@dataclass
class Run:
    stage: str
    cfg: dict
    raw_dir: Path
    root: Path = PROJECT_ROOT / "results" / "runs"
    started: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.run_id = f"{self.stage}-{self.started:%Y%m%dT%H%M%SZ}-{config_hash(self.cfg)}"
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), encoding="utf-8")

    def finish(self, **extra) -> dict:
        info = {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_utc": self.started.isoformat(timespec="seconds"),
            "seconds": round((datetime.now(timezone.utc) - self.started).total_seconds(), 2),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
            "versions": _versions(),
            "data_sha256": _data_hashes(self.raw_dir),
            **extra,
        }
        (self.dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        return info
