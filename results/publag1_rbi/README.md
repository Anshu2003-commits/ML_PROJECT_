# Results: publag1_rbi

Run `publag1_rbi-20261109T190116Z-56da7133`. Reproduce with `python -m fxforecast.experiments.run --stage publag1_rbi --models svr lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | svr | 29 | 0.8818 | 0.6601 | 0.0098 | 99.0231 | 0.6552 | 0.9313 |
| fx_plus_macro | lstm | 29 | 0.9274 | 0.7027 | 0.0104 | 98.9628 | 0.5517 | 0.9795 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 156 | 1.0202 | 0.7982 | 0.0129 | 98.7101 | 0.5769 | 0.9866 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_plus_macro | svr | 156 | 1.0427 | 0.8210 | 0.0132 | 98.6816 | 0.5321 | 1.0083 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 154 | 1.9978 | 1.5408 | 0.0248 | 97.5208 | 0.6039 | 0.9633 |
| fx_plus_macro | svr | 154 | 2.0681 | 1.5787 | 0.0252 | 97.4818 | 0.5260 | 0.9971 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.7740 | 2.1453 | 0.0339 | 96.6076 | 0.6689 | 0.9068 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_plus_macro | svr | 151 | 3.1233 | 2.4222 | 0.0375 | 96.2534 | 0.6026 | 1.0210 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9309 | 0.0166 | 0.9121 | 0.9451 | 0.9274 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0246 | 0.0111 | 1.0091 | 1.0402 | 1.0202 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.0095 | 0.0266 | 1.9662 | 2.0381 | 1.9978 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.8110 | 0.0565 | 2.7313 | 2.8686 | 2.7740 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
