# Results: publag1

Run `publag1-20261109T171343Z-d112bad7`. Reproduce with `python -m fxforecast.experiments.run --stage publag1 --models svr lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | svr | 29 | 0.9093 | 0.6931 | 0.0103 | 98.9726 | 0.5862 | 0.9603 |
| fx_plus_macro | lstm | 29 | 0.9344 | 0.7046 | 0.0104 | 98.9578 | 0.5517 | 0.9868 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 156 | 1.0208 | 0.8004 | 0.0129 | 98.7074 | 0.5641 | 0.9872 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_plus_macro | svr | 156 | 1.0342 | 0.8270 | 0.0133 | 98.6737 | 0.5192 | 1.0001 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_plus_macro | svr | 154 | 2.0840 | 1.5922 | 0.0254 | 97.4633 | 0.5000 | 1.0048 |
| fx_plus_macro | lstm | 154 | 2.0950 | 1.6273 | 0.0261 | 97.3934 | 0.6169 | 1.0101 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.7877 | 2.1499 | 0.0340 | 96.6020 | 0.6689 | 0.9113 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_plus_macro | svr | 151 | 3.2425 | 2.5174 | 0.0388 | 96.1182 | 0.5695 | 1.0600 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9457 | 0.0155 | 0.9206 | 0.9590 | 0.9344 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0245 | 0.0086 | 1.0145 | 1.0373 | 1.0208 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.1315 | 0.1033 | 2.0391 | 2.2885 | 2.0950 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.8318 | 0.0644 | 2.7541 | 2.9154 | 2.7877 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
