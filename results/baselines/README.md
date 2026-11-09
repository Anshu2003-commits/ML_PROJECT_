# Results: baselines

Run `baselines-20261109T161543Z-79713761`. Reproduce with `python -m fxforecast.experiments.run --stage baselines --models random_walk drift ar`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | ar | 29 | 0.8470 | 0.6092 | 0.0090 | 99.0991 | 0.7241 | 0.8945 |
| fx_only | drift | 29 | 0.9405 | 0.7281 | 0.0107 | 98.9256 | 0.5517 | 0.9933 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 156 | 1.0152 | 0.7908 | 0.0128 | 98.7216 | 0.5769 | 0.9818 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_only | ar | 156 | 1.0528 | 0.7929 | 0.0128 | 98.7235 | 0.6026 | 1.0181 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 154 | 1.9837 | 1.4901 | 0.0240 | 97.6036 | 0.6169 | 0.9564 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_only | ar | 154 | 2.1489 | 1.6527 | 0.0265 | 97.3452 | 0.5455 | 1.0361 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 151 | 2.8113 | 2.1604 | 0.0342 | 96.5834 | 0.6755 | 0.9191 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_only | ar | 151 | 3.1322 | 2.3130 | 0.0367 | 96.3350 | 0.6490 | 1.0239 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
