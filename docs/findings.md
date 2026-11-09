# Running findings log

Numbers that will go into the final report, with where they come from. Updated at each milestone.

## M2: data and diagnostics ([results/eda](../results/eda/README.md))
- Only **IP growth** and **trade balance** differentials Granger-cause monthly USD/INR changes at 5%. Inflation, interest-rate, money, reserves and equity differentials do not.
- One cointegrating relationship (Johansen rank 1) links log FX with relative prices, output, money and the rate differential. This argues for a VECM.
- Monthly FX changes have lag-1 autocorrelation of about 0.30. Part of this is an artefact of monthly averaging (Working, 1960).
- India's "interest rate" on FRED is the RBI policy rate (flat at 6% from 2003 to 2012).

## M4: baselines ([results/baselines](../results/baselines/README.md))

**The paper's headline numbers do not beat a "no change" forecast.** On the paper's own test window (2016-08 to 2018-12, h = 1):

| | RMSE (INR) | "Accuracy" |
|---|---|---|
| Random walk, monthly-average data (EXINUS) | **0.95** | **98.93%** |
| Random walk, end-of-month data (DEXINUS) | **1.28** | **98.69%** |
| Paper: LSTM (best) | 1.69 | 97.83% |
| Paper: SVM | 2.56 | 97.17% |
| Paper: VAR | 2.94 | 96.31% |

The data differ (FRED vs Bloomberg), but the gap is large and holds under both FX definitions. This is the Meese–Rogoff result in miniature: the bar is "no change", and an "accuracy" around 97% is far from impressive for monthly FX.

**Univariate baselines (FX history only):**
- **Replication window:** AR beats the random walk (Theil's U = 0.89, 72% correct direction). Drift is about equal to RW (U = 0.99).
- **Walk-forward 2010–2023 (156 forecasts):** AR does **not** beat RW at any horizon (U = 1.02 / 1.04 / 1.02 at h = 1 / 3 / 6). Its replication-window advantage does not generalise.
- **Drift beats RW at every walk-forward horizon** (U = 0.98 / 0.96 / 0.92 at h = 1 / 3 / 6), and the gain grows with horizon. This reflects the rupee's steady long-run depreciation. Drift is therefore the strongest simple benchmark at longer horizons.

## M5: VAR and VECM ([results/var](../results/var/README.md))

Theil's U vs random walk (below 1 = beats "no change"):

| | Replication h=1 | Walk-fwd h=1 | h=3 | h=6 |
|---|---|---|---|---|
| Drift (FX only) | 0.99 | **0.98** | **0.96** | **0.92** |
| AR (FX only) | **0.89** | 1.02 | 1.04 | 1.02 |
| VAR (FX + 7 macro differentials) | 0.97 | 1.06 | 0.99 | 0.93 |
| VECM (FX + monetary-model levels) | 0.92 | 1.18 | 1.20 | 1.23 |

- **Our VAR's replication RMSE is 0.92 INR vs the paper's 2.94.** With a proper stationarity treatment (differencing chosen by ADF on training data only) and FRED data, the econometric model is much stronger than the paper reports. The paper's "ML beats VAR" conclusion may partly reflect a weak VAR setup.
- **RQ2, econometric family (AR = FX only vs VAR = FX + macro):** macro information **hurts at 1 month** (U 1.06 vs 1.02) but **helps at 3 and 6 months** (0.99 vs 1.04, and 0.93 vs 1.02). This fits the literature that fundamentals matter more at longer horizons (e.g. Mark, 1995). Significance is still to be tested (M8).
- **Neither VAR nor VECM beats drift** in the walk-forward evaluation. The rupee's steady trend is still the hardest thing to beat.
- **VECM is unstable:** the Johansen rank chosen at each refit varies (rank 0 in 2 folds, 1 in 8, 2 in 3). It is the worst model out-of-sample (U 1.18 to 1.23), even though it looks good on the paper's short window (0.92). The long-run relationship is not stable enough to forecast with.
- **VAR settings:** AIC picks 1 lag on every refit. Most macro differentials are differenced; which ones varies slightly by fold (`model_choices.csv`).

## M6: SVR ([results/svr](../results/svr/README.md))

Theil's U vs random walk (below 1 = beats "no change"):

| | Replication h=1 | Walk-fwd h=1 | h=3 | h=6 |
|---|---|---|---|---|
| Drift (FX only) | 0.99 | **0.98** | **0.96** | **0.92** |
| SVR, FX only | 0.92 | 1.01 | 0.99 | 0.96 |
| SVR, FX + macro | 1.10 | 0.99 | 0.96 | 1.03 |
| VAR, FX + macro | 0.97 | 1.06 | 0.99 | 0.93 |

- **Replication:** our FX-only SVR has RMSE 0.87 INR vs the paper's SVM 2.56. Adding macro data makes it **worse** (RMSE 1.04, U 1.10).
- **RQ2, ML family (SVR FX only vs SVR FX + macro):** macro data helps a little at 1 and 3 months (U 1.01 → 0.99, and 0.99 → 0.96) and hurts at 6 months (0.96 → 1.03). That is the opposite horizon pattern from the VAR. No consistent benefit; significance tests come in M8.
- **RQ1 so far:** in the walk-forward test the best multivariate model differs by horizon: SVR+macro at 1 and 3 months, VAR at 6 months. None beats drift, although SVR+macro at 3 months (0.961) is close to drift (0.956).
- **The SVR prefers an almost linear fit.** Inner-CV tuning chose the smallest gamma in 50 of 80 fits on grid v1 and 30 of 80 on the widened grid. Widening the grid (gamma down to 0.0001, C up to 10000) changed Theil's U by at most 0.006, so the results are not sensitive to the grid. There is little nonlinear structure for the RBF kernel to exploit at monthly frequency.
- **Cost:** the full SVR stage (2 feature sets, 4 scheme/horizon cases, 216-point grid × 5 inner splits per refit) takes about 12 minutes on a laptop CPU. VAR takes seconds.

**Implication for RQ1/RQ2:** a multivariate model must beat both the random walk *and* drift to show that macro information helps. Any claim of an improvement needs a significance test (M8), because 29 test months is a small sample.
