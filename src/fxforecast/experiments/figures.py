"""Final report figures (M9), built from the latest ``final`` run and the M8 analysis tables.

    python -m fxforecast.experiments.figures

Writes results/final/figures/:
    paper_comparison.png   replication RMSE: base paper vs this project
    replication_paths.png  actual USD/INR vs forecasts on the paper's test window
    theil_u.png            Theil's U for every model, FX only vs FX + macro, by horizon (RQ1 + RQ2)
    subperiods.png         Theil's U by period (walk-forward)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fxforecast.config import PROJECT_ROOT
from fxforecast.experiments.analysis import latest_run, load_predictions
from fxforecast.experiments.run import PAPER
from fxforecast.reporting import plots

MODEL_ORDER = ["drift", "ar", "var", "vecm", "svr", "lstm"]
SET_LABEL = {"fx_only": "FX only", "fx_plus_macro": "FX + macro"}


def run(out_dir: Path) -> list[Path]:
    final_dir = latest_run("final")
    preds = load_predictions(final_dir)
    metrics = pd.read_csv(final_dir / "metrics.csv")
    fig_dir = out_dir / "figures"
    paths = []

    # 1. paper vs ours, replication RMSE
    rep = metrics[metrics.scheme == "replication"].set_index(["model", "feature_set"])["rmse"]
    ours = {"Random walk": rep[("random_walk", "fx_only")], "VAR": rep[("var", "fx_plus_macro")],
            "SVR / SVM": rep[("svr", "fx_plus_macro")], "LSTM": rep[("lstm", "fx_plus_macro")]}
    paper = {"Random walk": np.nan, "VAR": PAPER.loc["VAR (paper)", "rmse"],
             "SVR / SVM": PAPER.loc["SVM (paper)", "rmse"], "LSTM": PAPER.loc["LSTM (paper)", "rmse"]}
    bars = pd.DataFrame({"Giri & Kaushik (2020), Bloomberg": paper, "This project, FRED": ours})
    paths.append(plots.grouped_bars(bars, fig_dir / "paper_comparison.png", ylabel="RMSE (INR per USD)",
                                    title="Replication window 2016-08 to 2018-12, 1 month ahead "
                                          "(models use FX + macro): lower is better"))

    # 2. forecast paths on the replication window
    r = preds[preds.scheme == "replication"]
    pick = lambda m, fs: r[(r.model == m) & (r.feature_set == fs)].set_index("target_date")["fx_pred"]  # noqa: E731
    actual = r[r.model == "random_walk"].set_index("target_date")["fx_true"]
    paths.append(plots.forecast_paths(
        actual, {"random walk": pick("random_walk", "fx_only"), "VAR": pick("var", "fx_plus_macro"),
                 "SVR": pick("svr", "fx_plus_macro"), "LSTM": pick("lstm", "fx_plus_macro")},
        fig_dir / "replication_paths.png",
        title="USD/INR: actual vs 1-month-ahead forecasts (FX + macro models), paper's test window"))

    # 3. Theil's U dot plot: walk-forward by horizon + replication
    m = metrics[metrics.model != "random_walk"].copy()
    m["case"] = np.where(m.scheme == "replication", "replication h=1", "walk-forward h=" + m.horizon.astype(str))
    m["feature_set"] = m.feature_set.map(SET_LABEL)
    table = m.pivot_table(index=["model", "feature_set"], columns="case", values="theil_u")
    order = [(mo, fs) for mo in MODEL_ORDER for fs in SET_LABEL.values() if (mo, fs) in table.index]
    table = table.loc[order, ["replication h=1", "walk-forward h=1", "walk-forward h=3", "walk-forward h=6"]]
    drift = table.xs("drift", level="model").iloc[0].to_dict()
    paths.append(plots.theil_u_dots(table, fig_dir / "theil_u.png", reference=drift,
                                    title="Theil's U by model, information set and horizon"))

    # 4. sub-periods heatmap (h = 1 and 6)
    sub = pd.read_csv(out_dir / "subperiods.csv", index_col=[0, 1])
    sub = sub.loc[[i for i in sub.index if i[0] in (1, 6)]]
    sub.index = [f"h={h}  {name}" for h, name in sub.index]
    paths.append(plots.ratio_heatmap(sub, fig_dir / "subperiods.png",
                                     title="Theil's U by period, walk-forward (blue < 1 beats random walk, red > 1 loses)"))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "final"))
    args = parser.parse_args(argv)
    for p in run(Path(args.out)):
        print(p)


if __name__ == "__main__":
    main()
