# Results: publag2_rbi

Run `publag2_rbi-20261109T190941Z-56da7133`. Reproduce with `python -m fxforecast.experiments.run --stage publag2_rbi --models svr lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 29 | 0.9301 | 0.7013 | 0.0104 | 98.9640 | 0.5517 | 0.9823 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |
| fx_plus_macro | svr | 29 | 0.9844 | 0.7638 | 0.0114 | 98.8639 | 0.5172 | 1.0396 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 156 | 1.0122 | 0.7929 | 0.0128 | 98.7187 | 0.5513 | 0.9789 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_plus_macro | svr | 156 | 1.0583 | 0.8282 | 0.0133 | 98.6699 | 0.5000 | 1.0234 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 154 | 2.0027 | 1.5255 | 0.0245 | 97.5484 | 0.6039 | 0.9656 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_plus_macro | svr | 154 | 2.1019 | 1.6152 | 0.0256 | 97.4385 | 0.4805 | 1.0134 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.8029 | 2.1567 | 0.0340 | 96.5987 | 0.6623 | 0.9163 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_plus_macro | svr | 151 | 3.1226 | 2.4206 | 0.0374 | 96.2636 | 0.5629 | 1.0208 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9359 | 0.0222 | 0.9085 | 0.9533 | 0.9301 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0164 | 0.0126 | 1.0035 | 1.0370 | 1.0122 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.0162 | 0.0434 | 1.9731 | 2.0699 | 2.0027 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.8347 | 0.0495 | 2.7885 | 2.9181 | 2.8029 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
