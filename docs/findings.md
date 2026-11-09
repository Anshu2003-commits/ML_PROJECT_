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

**Implication for RQ1/RQ2:** a multivariate model must beat both the random walk *and* drift to show that macro information helps. Any claim of an improvement needs a significance test (M8), because 29 test months is a small sample.
