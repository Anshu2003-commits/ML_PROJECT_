"""Significance tests, sub-period results, robustness and ablation for the final run (M8).

    python -m fxforecast.experiments.analysis

Reads the latest run of each stage from results/runs/ (``final``, ``publag1``,
``publag2``) and writes results/final/analysis.md plus CSV tables:

    vs_benchmarks     Diebold-Mariano of every model vs random walk and vs drift   (RQ1)
    multivariate      pairwise DM between VAR, VECM, SVR+macro, LSTM+macro          (RQ1)
    macro_value       Clark-West: FX-only vs FX+macro within each model family      (RQ2)
    subperiods        Theil's U by period (2010-14, 2015-19, 2020-23)
    publication_lag   Theil's U of SVR/LSTM+macro with macro data delayed 0/1/2 months
    var_ablation      VAR walk-forward Theil's U with each macro differential removed
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fxforecast.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, resolve_path
from fxforecast.evaluation.backtest import attach_levels, backtest
from fxforecast.evaluation.metrics import rmse
from fxforecast.evaluation.splits import walk_forward_folds
from fxforecast.evaluation.stat_tests import clark_west, diebold_mariano
from fxforecast.features.supervised import make_supervised
from fxforecast.features.transforms import MACRO_FEATURES, macro_differentials
from fxforecast.models.baselines import RandomWalk
from fxforecast.models.var import VARForecaster
from fxforecast.reporting.tables import to_markdown

log = logging.getLogger("fxforecast.analysis")
RUNS = PROJECT_ROOT / "results" / "runs"
CASES = ["scheme", "horizon"]
PERIODS = {"2010-2014": ("2010-01-01", "2014-12-01"), "2015-2019": ("2015-01-01", "2019-12-01"),
           "2020-2023": ("2020-01-01", "2023-12-01")}
NESTED = {   # family: (restricted FX-only model, unrestricted FX+macro model)
    "econometric (AR -> VAR)": (("fx_only", "ar"), ("fx_plus_macro", "var")),
    "machine learning (SVR)": (("fx_only", "svr"), ("fx_plus_macro", "svr")),
    "deep learning (LSTM)": (("fx_only", "lstm"), ("fx_plus_macro", "lstm")),
}
MULTIVARIATE = [("fx_plus_macro", m) for m in ("var", "vecm", "svr", "lstm")]


def latest_run(stage: str, runs_dir: Path = RUNS) -> Path:
    candidates = sorted(p for p in runs_dir.glob(f"{stage}-*") if (p / "predictions.csv").exists())
    if not candidates:
        raise FileNotFoundError(f"No completed run for stage '{stage}' in {runs_dir}")
    return candidates[-1]


def load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv", parse_dates=["origin", "target_date", "refit_date"])


def _series(preds: pd.DataFrame, scheme: str, h: int, fs: str, model: str) -> pd.DataFrame:
    sel = preds[(preds.scheme == scheme) & (preds.horizon == h) & (preds.feature_set == fs) & (preds.model == model)]
    return sel.set_index("origin").sort_index()


def _label(fs: str, model: str) -> str:
    return f"{model} ({'FX only' if fs == 'fx_only' else 'FX+macro'})"


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def vs_benchmarks(preds: pd.DataFrame) -> pd.DataFrame:
    """DM of each model against random walk and drift. stat > 0 = model more accurate."""
    rows = []
    for (scheme, h), g in preds.groupby(CASES, sort=False):
        rw = _series(g, scheme, h, "fx_only", "random_walk")
        dr = _series(g, scheme, h, "fx_only", "drift")
        for (fs, model), m in g.groupby(["feature_set", "model"], sort=False):
            if model in ("random_walk",):
                continue
            s = _series(g, scheme, h, fs, model)
            e = s.y_true - s.y_pred
            row = {"scheme": scheme, "horizon": h, "model": _label(fs, model),
                   "theil_u": rmse(s.fx_true, s.fx_pred) / rmse(rw.fx_true, rw.fx_pred)}
            for name, bench in (("rw", rw), ("drift", dr)):
                if model == "drift" and name == "drift":
                    row[f"dm_{name}"], row[f"p_{name}"] = np.nan, np.nan
                    continue
                r = diebold_mariano(bench.y_true - bench.y_pred, e, h)
                row[f"dm_{name}"], row[f"p_{name}"] = r.statistic, r.p_value
            rows.append(row)
    return pd.DataFrame(rows).set_index(CASES + ["model"])


def multivariate_pairs(preds: pd.DataFrame) -> pd.DataFrame:
    """Pairwise DM between the multivariate models. stat > 0 = model B more accurate than A."""
    rows = []
    for (scheme, h), g in preds.groupby(CASES, sort=False):
        for a, b in itertools.combinations(MULTIVARIATE, 2):
            sa, sb = _series(g, scheme, h, *a), _series(g, scheme, h, *b)
            r = diebold_mariano(sa.y_true - sa.y_pred, sb.y_true - sb.y_pred, h)
            better = _label(*b) if r.statistic > 0 else _label(*a)
            rows.append({"scheme": scheme, "horizon": h, "A": _label(*a), "B": _label(*b),
                         "dm_stat": r.statistic, "p_value": r.p_value,
                         "more_accurate": better + _stars(r.p_value)})
    return pd.DataFrame(rows).set_index(CASES + ["A", "B"])


def macro_value(preds: pd.DataFrame) -> pd.DataFrame:
    """Clark-West test per family: does adding macro differentials improve accuracy? (RQ2)"""
    rows = []
    for (scheme, h), g in preds.groupby(CASES, sort=False):
        rw = _series(g, scheme, h, "fx_only", "random_walk")
        rmse_rw = rmse(rw.fx_true, rw.fx_pred)
        for family, (restricted, unrestricted) in NESTED.items():
            r, u = _series(g, scheme, h, *restricted), _series(g, scheme, h, *unrestricted)
            cw = clark_west(r.y_true, r.y_pred, u.y_pred, h)
            u_r, u_u = rmse(r.fx_true, r.fx_pred) / rmse_rw, rmse(u.fx_true, u.fx_pred) / rmse_rw
            rows.append({"scheme": scheme, "horizon": h, "family": family,
                         "theil_u_fx_only": u_r, "theil_u_fx_macro": u_u,
                         "rmse_change_pct": 100 * (u_u / u_r - 1),
                         "cw_stat": cw.statistic, "cw_p_one_sided": cw.p_value,
                         "macro_helps": ("yes" if cw.p_value < 0.05 else "weak" if cw.p_value < 0.10 else "no")})
    return pd.DataFrame(rows).set_index(CASES + ["family"])


def subperiods(preds: pd.DataFrame) -> pd.DataFrame:
    """Theil's U by target-date period for the walk-forward forecasts."""
    wf = preds[preds.scheme == "walk_forward"]
    out = {}
    for period, (start, end) in PERIODS.items():
        part = wf[(wf.target_date >= start) & (wf.target_date <= end)]
        for (h, fs, model), g in part.groupby(["horizon", "feature_set", "model"], sort=False):
            rw = part[(part.horizon == h) & (part.model == "random_walk")].set_index("origin").loc[g.origin]
            out.setdefault((h, _label(fs, model)), {})[period] = rmse(g.fx_true, g.fx_pred) / rmse(rw.fx_true, rw.fx_pred)
    table = pd.DataFrame(out).T
    table.index.names = ["horizon", "model"]
    return table.drop(index=[i for i in table.index if i[1].startswith("random_walk")])


def publication_lag(final: pd.DataFrame, lagged: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Theil's U of the macro SVR/LSTM when macro inputs are delayed by 0, 1, 2 months."""
    out = {}
    for lag, preds in {0: final, **lagged}.items():
        for (scheme, h), g in preds.groupby(CASES, sort=False):
            rw = _series(g, scheme, h, "fx_only", "random_walk")
            for model in ("svr", "lstm"):
                s = _series(g, scheme, h, "fx_plus_macro", model)
                if len(s):
                    ref = rw.loc[s.index]
                    out.setdefault((scheme, h, _label("fx_plus_macro", model)), {})[f"lag {lag}"] = (
                        rmse(s.fx_true, s.fx_pred) / rmse(ref.fx_true, ref.fx_pred))
    table = pd.DataFrame(out).T
    table.index.names = CASES + ["model"]
    return table


def var_ablation(cfg: dict) -> pd.DataFrame:
    """Walk-forward Theil's U of the VAR with all macro variables vs with one removed."""
    panel = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "panel.parquet")
    feats = macro_differentials(panel)
    wf = cfg["experiments"]["walk_forward"]
    out = {}
    for h in wf["horizons"]:
        X, y = make_supervised(feats, "fx_plus_macro", n_lags=cfg["features"]["max_lags"], horizon=h)
        folds = walk_forward_folds(X.index, pd.Timestamp(wf["first_origin"]), wf["refit_every_months"], h)
        rw = attach_levels(backtest(RandomWalk, X, y, folds), panel["fx"], h)
        rmse_rw = rmse(rw.fx_true, rw.fx_pred)
        for dropped in ("none (all 7)", *MACRO_FEATURES):
            series = feats if dropped.startswith("none") else feats.drop(columns=dropped)
            p = attach_levels(backtest(lambda: VARForecaster(series, h, **cfg["models"]["var"]), X, y, folds),
                              panel["fx"], h)
            out.setdefault(dropped, {})[f"h={h}"] = rmse(p.fx_true, p.fx_pred) / rmse_rw
        log.info("VAR ablation h=%d done", h)
    table = pd.DataFrame(out).T
    table.index.name = "macro variable removed"
    return table


def run(cfg: dict, out_dir: Path) -> Path:
    final_dir = latest_run("final")
    final = load_predictions(final_dir)
    lagged = {}
    for lag in (1, 2):
        try:
            lagged[lag] = load_predictions(latest_run(f"publag{lag}"))
        except FileNotFoundError:
            log.warning("no publag%d run found; skipping that column", lag)

    tables = {
        "vs_benchmarks": vs_benchmarks(final),
        "multivariate": multivariate_pairs(final),
        "macro_value": macro_value(final),
        "subperiods": subperiods(final),
        "publication_lag": publication_lag(final, lagged),
        "var_ablation": var_ablation(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")

    info = json.loads((final_dir / "run.json").read_text(encoding="utf-8"))
    md = f"""# Significance, robustness and ablation (M8)

Generated by `python -m fxforecast.experiments.analysis` from run `{info['run_id']}` (commit `{info['git_commit'][:7]}`).
All tests use squared errors of the h-month log change. Stars: * p<0.10, ** p<0.05, *** p<0.01.
Many tests are reported, so read individual p-values with caution (no multiple-testing correction).

## RQ1: does any model beat the benchmarks? (Diebold-Mariano)
`dm_rw` / `dm_drift` > 0 means the model is more accurate than the random walk / drift. p-values are two-sided.

{to_markdown(tables['vs_benchmarks'], 3)}

## RQ1: multivariate models against each other (Diebold-Mariano)
{to_markdown(tables['multivariate'], 3)}

## RQ2: does adding macro data help? (Clark-West, nested models)
Each family's FX-only model is compared with the same family using FX + macro differentials. `cw_p_one_sided` < 0.05 means macro data significantly improves accuracy.

{to_markdown(tables['macro_value'], 3)}

## Sub-periods: Theil's U by period (walk-forward)
{to_markdown(tables['subperiods'], 3)}

## Robustness: macro data delayed by publication lags (Theil's U)
Lag k means the macro inputs at origin t are the values for month t-k (FX itself is not delayed).

{to_markdown(tables['publication_lag'], 3)}

## Ablation: VAR with one macro differential removed (walk-forward Theil's U)
Higher than the "none" row = that variable was helping.

{to_markdown(tables['var_ablation'], 3)}
"""
    report = out_dir / "analysis.md"
    report.write_text(md, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "final"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(run(load_config(args.config), Path(args.out)))


if __name__ == "__main__":
    main()
