"""Run models through the replication and walk-forward evaluations.

    python -m fxforecast.experiments.run --stage baselines --models random_walk drift ar

For each scheme x horizon x feature set x model: build (X, y), make folds,
backtest, convert to FX levels, then summarize. Every model in a run forecasts
exactly the same origins (checked), so the metrics are directly comparable.

Outputs:
    results/runs/<run_id>/   full predictions, metrics, config, provenance (git-ignored)
    results/<stage>/README.md + metrics.csv   summary committed to the repo
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from fxforecast.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, resolve_path
from fxforecast.evaluation.backtest import attach_levels, backtest
from fxforecast.evaluation.metrics import member_spread, metrics_table
from fxforecast.evaluation.splits import replication_folds, walk_forward_folds
from fxforecast.experiments.tracking import Run
from fxforecast.features.supervised import make_supervised
from fxforecast.features.transforms import macro_differentials, relative_levels
from fxforecast.models import Forecaster
from fxforecast.models.baselines import AutoRegressive, Drift, RandomWalk
from fxforecast.models.lstm import LSTMForecaster
from fxforecast.models.svr import SVRForecaster
from fxforecast.models.var import VARForecaster, VECMForecaster
from fxforecast.reporting.tables import to_markdown

log = logging.getLogger("fxforecast.run")
GROUP = ["scheme", "horizon", "feature_set", "model"]

# Base paper (Giri & Kaushik 2020), replication period, h = 1, Bloomberg data.
PAPER = pd.DataFrame(
    {"rmse": [2.9381, 2.5585, 1.6872], "mape": [0.0369, 0.0283, 0.0217], "mpe": [0.0286, -0.0261, 0.0031],
     "accuracy_pct": [96.31, 97.17, 97.83]},
    index=pd.Index(["VAR (paper)", "SVM (paper)", "LSTM (paper)"], name="model"),
)


ModelFactory = Callable[[int], Forecaster]   # horizon -> fresh model


def model_registry(cfg: dict, feats: pd.DataFrame, levels: pd.DataFrame) -> dict[str, tuple[ModelFactory, list[str]]]:
    """name -> (factory taking the horizon, feature sets the model is run on).

    VAR/VECM read the series tables directly (they guard against look-ahead themselves);
    the feature set label records which information they use.
    """
    m = cfg["models"]
    return {
        "random_walk": (lambda h: RandomWalk(), ["fx_only"]),
        "drift": (lambda h: Drift(), ["fx_only"]),
        "ar": (lambda h: AutoRegressive(**m["ar"]), ["fx_only"]),
        "var": (lambda h: VARForecaster(feats, h, **m["var"]), ["fx_plus_macro"]),
        "vecm": (lambda h: VECMForecaster(levels, h, **m["vecm"]), ["fx_plus_macro"]),
        "svr": (lambda h: SVRForecaster(h, **m["svr"]), ["fx_only", "fx_plus_macro"]),
        "lstm": (lambda h: LSTMForecaster(h, **m["lstm"]), ["fx_only", "fx_plus_macro"]),
    }


def _schemes(cfg: dict) -> list[tuple[str, int, Callable]]:
    rep, wf = cfg["experiments"]["replication"], cfg["experiments"]["walk_forward"]
    out = [("replication", h, lambda o, h: replication_folds(
        o, pd.Timestamp(rep["train_end"]), pd.Timestamp(rep["test_end"]), h)) for h in rep["horizons"]]
    out += [("walk_forward", h, lambda o, h: walk_forward_folds(
        o, pd.Timestamp(wf["first_origin"]), wf["refit_every_months"], h)) for h in wf["horizons"]]
    return out


def run(cfg: dict, stage: str, model_names: list[str], publication_lag: int = 0) -> tuple[pd.DataFrame, Run]:
    panel = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "panel.parquet")
    feats = macro_differentials(panel)
    registry = model_registry(cfg, feats, relative_levels(panel))
    unknown = set(model_names) - set(registry)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; available: {sorted(registry)}")
    if publication_lag and {"var", "vecm"} & set(model_names):
        raise ValueError("publication lag is not implemented for var/vecm")
    if "random_walk" not in model_names:
        model_names = ["random_walk", *model_names]   # needed as the Theil's U benchmark

    tracker = Run(stage, cfg, resolve_path(cfg["data"]["raw_dir"]))
    n_lags = cfg["features"]["max_lags"]
    feature_sets = sorted({fs for name in model_names for fs in registry[name][1]})

    all_preds = []
    for scheme, h, make_folds in _schemes(cfg):
        data = {fs: make_supervised(feats, fs, n_lags=n_lags, horizon=h, publication_lag=publication_lag)
                for fs in feature_sets}
        # identical origins for every feature set, so all models face the same test points
        common = sorted(set.intersection(*(set(X.index) for X, _ in data.values())))
        common = pd.DatetimeIndex(common)
        folds = make_folds(common, h)
        for name in model_names:
            factory, sets = registry[name]
            for fs in sets:
                X, y = data[fs]
                preds = attach_levels(backtest(lambda: factory(h), X.loc[common], y.loc[common], folds),
                                      panel["fx"], h)
                all_preds.append(preds.assign(scheme=scheme, horizon=h, feature_set=fs, model=name))
                log.info("%-12s h=%d %-13s %-12s %d forecasts", scheme, h, fs, name, len(preds))

    preds = pd.concat(all_preds, ignore_index=True)
    table = metrics_table(preds, GROUP)
    choices = (preds.groupby(GROUP + ["refit_date"], sort=False)["model_info"].first()
               .loc[lambda s: s != "{}"].rename("chosen_settings"))
    choices.to_csv(tracker.dir / "model_choices.csv")
    spread = member_spread(preds, GROUP)
    if len(spread):
        spread.to_csv(tracker.dir / "seed_spread.csv")
    preds.to_csv(tracker.dir / "predictions.csv", index=False)
    table.to_csv(tracker.dir / "metrics.csv")
    tracker.finish(models=model_names, publication_lag=publication_lag, n_forecasts=len(preds))
    return table, tracker


def write_report(table: pd.DataFrame, tracker: Run, out_dir: Path, models: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "metrics.csv")
    shutil.copyfile(tracker.dir / "model_choices.csv", out_dir / "model_choices.csv")
    spread_md = ""
    if (tracker.dir / "seed_spread.csv").exists():
        shutil.copyfile(tracker.dir / "seed_spread.csv", out_dir / "seed_spread.csv")
        spread = pd.read_csv(tracker.dir / "seed_spread.csv", index_col=list(range(len(GROUP))))
        spread_md = ("### Seed variability (each seed scored on its own; the model above is the seed average)\n\n"
                     + to_markdown(spread, 4) + "\n")
    cols = ["n", "rmse", "mae", "mape", "accuracy_pct", "hit_rate", "theil_u"]
    sections = []
    for (scheme, h), part in table.groupby(level=["scheme", "horizon"], sort=False):
        t = part.droplevel(["scheme", "horizon"])[cols].sort_values("rmse")
        sections.append(f"### {scheme.replace('_', ' ').title()}, h = {h} month(s)\n\n{to_markdown(t, 4)}")
    md = f"""# Results: {tracker.stage}

Run `{tracker.run_id}`. Reproduce with `python -m fxforecast.experiments.run --stage {tracker.stage} --models {' '.join(models)}`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

{chr(10).join(s + chr(10) for s in sections)}
{spread_md}
### Base paper, for reference (replication period, h = 1, Bloomberg data)

{to_markdown(PAPER, 4)}
"""
    report = out_dir / "README.md"
    report.write_text(md, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", required=True, help="name for the results folder, e.g. baselines")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--publication-lag", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    table, tracker = run(cfg, args.stage, args.models, args.publication_lag)
    print(write_report(table, tracker, PROJECT_ROOT / "results" / args.stage, args.models))


if __name__ == "__main__":
    main()
