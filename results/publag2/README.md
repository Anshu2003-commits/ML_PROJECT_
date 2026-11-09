# Results: publag2

Run `publag2-20261109T172157Z-d112bad7`. Reproduce with `python -m fxforecast.experiments.run --stage publag2 --models svr lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 29 | 0.9361 | 0.6989 | 0.0103 | 98.9667 | 0.5517 | 0.9887 |
| fx_plus_macro | svr | 29 | 0.9391 | 0.7041 | 0.0104 | 98.9556 | 0.5862 | 0.9918 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_plus_macro | svr | 156 | 1.0428 | 0.8177 | 0.0131 | 98.6868 | 0.5449 | 1.0084 |
| fx_plus_macro | lstm | 156 | 1.0504 | 0.8316 | 0.0134 | 98.6594 | 0.5577 | 1.0157 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 154 | 2.0508 | 1.5845 | 0.0254 | 97.4606 | 0.6169 | 0.9888 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_plus_macro | svr | 154 | 2.0864 | 1.5755 | 0.0251 | 97.4929 | 0.5000 | 1.0059 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.8111 | 2.1764 | 0.0343 | 96.5690 | 0.6755 | 0.9190 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_plus_macro | svr | 151 | 3.1248 | 2.4006 | 0.0373 | 96.2750 | 0.5762 | 1.0215 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9402 | 0.0163 | 0.9202 | 0.9551 | 0.9361 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0683 | 0.0743 | 1.0101 | 1.1924 | 1.0504 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.0796 | 0.0862 | 2.0124 | 2.2186 | 2.0508 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.8494 | 0.0813 | 2.7787 | 2.9650 | 2.8111 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
