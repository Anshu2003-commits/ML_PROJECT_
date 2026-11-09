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

## M7: LSTM ([results/lstm](../results/lstm/README.md))

Theil's U vs random walk (LSTM = average of 5 seeds):

| | Replication h=1 | Walk-fwd h=1 | h=3 | h=6 |
|---|---|---|---|---|
| Drift (FX only) | 0.99 | **0.98** | **0.96** | 0.92 |
| LSTM, FX only | 0.96 | 1.00 | 0.97 | 0.95 |
| LSTM, FX + macro | 0.98 | 1.00 | 1.02 | **0.905** |
| VAR, FX + macro | 0.97 | 1.06 | 0.99 | 0.93 |
| SVR, FX + macro | 1.10 | 0.99 | 0.96 | 1.03 |

- **Replication:** LSTM RMSE 0.91 INR vs the paper's LSTM 1.69. All three of our model classes (VAR 0.92, SVR 0.87, LSTM 0.91) are far below the paper's numbers, and all are close to the random walk (0.95). On FRED data the paper's ranking (LSTM < SVM < VAR) is not reproduced. The FX-only AR (0.85) is best on this window.
- **The first model to beat drift in the walk-forward test** is LSTM + macro at 6 months (U 0.905 vs drift 0.919). The margin is small and needs a significance test (M8).
- **RQ2, deep-learning family (LSTM FX only vs LSTM FX + macro):** macro data hurts at 1 and 3 months (U 0.996 → 1.000, and 0.97 → 1.02) and helps at 6 months (0.95 → 0.905). This matches the VAR's pattern at long horizons.
- **Seed sensitivity:** the FX-only LSTM is stable across seeds (RMSE std 0.005 to 0.017 INR). The macro LSTM is much less stable, especially at 3 months: per-seed RMSE ranges from 1.95 to 2.57. More inputs on about 250 rows make training fragile; seed averaging matters.
- **Early stopping fires early** (median best epoch 13). The networks stay close to a smooth, drift-like forecast, which is why the LSTM tracks drift closely. Architecture choice: hidden 32 in 49 of 80 refits; lookback 6 and 12 about equally often.
- **Cost:** about 6 minutes for the LSTM stage (80 refits × (4 architecture trials + 4 extra seeds)), vs about 12 minutes for SVR and seconds for VAR.

## M8: significance, robustness, ablation ([results/final/analysis.md](../results/final/analysis.md))

The final run (all 7 models, one command, commit `f03748e`) reproduced every stage result exactly, e.g. LSTM+macro h=6 U = 0.905 in both runs.

### RQ1: which model forecasts best, and does any beat the benchmarks? (Diebold–Mariano)
- **No model significantly beats the random walk at 5% in any setting.** The closest are the FX-only AR (p = 0.06) and FX-only SVR (p = 0.07) on the paper's replication window, and drift at 1 and 6 months (p = 0.09 and 0.10).
- **No model significantly beats drift at 5% in the walk-forward test.** The best point estimate, LSTM+macro at 6 months (U 0.905 vs drift 0.919), has p = 0.47 against drift and p = 0.14 against the RW. Several models are significantly *worse* than drift: SVR at 6 months (FX only p = 0.005; FX+macro p = 0.04), LSTM FX-only at 3 months (p = 0.04) and LSTM+macro at 1 month (p = 0.049).
- **Among the multivariate models** almost no difference is significant. The exception is LSTM+macro beating SVR+macro at 6 months (p = 0.02). The paper's ranking (LSTM > SVM > VAR) is not significantly supported on its own test window: every pairwise p is above 0.07.
- **Answer to RQ1:** macro-based VAR, SVR and LSTM forecasts are about as accurate as simple benchmarks (RMSE within ±10% of the random walk) and never significantly better. ML/DL models are not significantly better than the VAR overall. The LSTM's edge is limited to the 6-month horizon.

### RQ2: does macro information add value over FX history alone? (Clark–West, nested)
- **Replication window: no**, in any family. Macro data raises RMSE by 2.5% (LSTM), 8% (VAR vs AR) and 20% (SVR).
- **Walk-forward 2010–2023: macro data carries statistically significant predictive information in 6 of 9 family/horizon cases.** These are econometric (AR→VAR) at h=1, 3, 6; SVR at h=1, 3; and LSTM at h=6 (one-sided p from 0.007 to 0.035).
- **Statistical vs practical value:** in 1 of those 6 cases (VAR at 1 month, RMSE +3.7% vs AR), RMSE is still *higher* with macro data. Clark–West asks whether the bigger model's forecasts contain useful information once you allow for the noise of estimating extra parameters, not whether they have lower RMSE in this sample. Practical RMSE gains are at most 9% (VAR h=6) and never enough to beat drift significantly.
- **Answer to RQ2:** macro differentials contain real but weak information about future USD/INR, most consistently at 3–6 months. Parameter-estimation noise on a small sample (about 250 months) eats most of it. Incorporating macro data improves accuracy only modestly and inconsistently.

### When do macro models work? (sub-periods, walk-forward Theil's U)
- **2010–2014** (including the 2013 taper tantrum, when India's current-account deficit drove the rupee) is the best period for macro models. VAR, VECM and SVR+macro reach U 0.93–0.94 at 1 month, beating every FX-only model.
- **2020–2023** (COVID and after) is the worst for the econometric macro models: VAR 1.29 and VECM 1.74 at 1 month. COVID base effects in industrial production (EDA) push their forecasts off. LSTM and drift do best here (U 0.80–0.83 at 6 months).
- The relative ranking of models changes by period. A single test window (as in the paper) cannot support a general claim.

### Robustness: publication lags
- Delaying macro inputs by 1–2 months barely changes results. LSTM+macro at 6 months: U 0.905 → 0.911 → 0.919. The main findings do not rely on using macro data before it would have been published.

### Ablation: which macro variables help the VAR? (leave one out)
- **Removing IP growth *improves* the VAR at every horizon** (h=1: 1.055 → 0.980). That is the very variable that Granger-causes FX in-sample (M2), a clear case of in-sample significance not carrying over out-of-sample, likely because of COVID-era extremes.
- **The interest-rate differential is the most useful variable at 1 month:** removing it worsens U from 1.055 to 1.088. Equity returns and reserves help a little at 1 month. Removing any single variable barely matters at 6 months.
- These results are diagnostic only. Choosing a variable subset now, after seeing test results, would be tuning on the test set, so the main results keep all 7 variables.
