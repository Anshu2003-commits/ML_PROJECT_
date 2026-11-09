# Results: lstm

Run `lstm-20261109T164320Z-d112bad7`. Reproduce with `python -m fxforecast.experiments.run --stage lstm --models drift ar var lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | ar | 29 | 0.8470 | 0.6092 | 0.0090 | 99.0991 | 0.7241 | 0.8945 |
| fx_only | lstm | 29 | 0.9087 | 0.6849 | 0.0101 | 98.9894 | 0.5862 | 0.9597 |
| fx_plus_macro | var | 29 | 0.9172 | 0.6626 | 0.0098 | 99.0196 | 0.6897 | 0.9687 |
| fx_plus_macro | lstm | 29 | 0.9313 | 0.7126 | 0.0105 | 98.9454 | 0.5517 | 0.9836 |
| fx_only | drift | 29 | 0.9405 | 0.7281 | 0.0107 | 98.9256 | 0.5517 | 0.9933 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 156 | 1.0152 | 0.7908 | 0.0128 | 98.7216 | 0.5769 | 0.9818 |
| fx_only | lstm | 156 | 1.0299 | 0.8014 | 0.0129 | 98.7095 | 0.5705 | 0.9960 |
| fx_plus_macro | lstm | 156 | 1.0336 | 0.8125 | 0.0131 | 98.6875 | 0.5833 | 0.9995 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_only | ar | 156 | 1.0528 | 0.7929 | 0.0128 | 98.7235 | 0.6026 | 1.0181 |
| fx_plus_macro | var | 156 | 1.0914 | 0.8370 | 0.0134 | 98.6634 | 0.5577 | 1.0554 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 154 | 1.9837 | 1.4901 | 0.0240 | 97.6036 | 0.6169 | 0.9564 |
| fx_only | lstm | 154 | 2.0145 | 1.5036 | 0.0242 | 97.5812 | 0.6299 | 0.9713 |
| fx_plus_macro | var | 154 | 2.0460 | 1.5876 | 0.0256 | 97.4432 | 0.5974 | 0.9865 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_plus_macro | lstm | 154 | 2.1140 | 1.6353 | 0.0262 | 97.3766 | 0.6169 | 1.0193 |
| fx_only | ar | 154 | 2.1489 | 1.6527 | 0.0265 | 97.3452 | 0.5455 | 1.0361 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.7694 | 2.1413 | 0.0339 | 96.6071 | 0.6689 | 0.9054 |
| fx_only | drift | 151 | 2.8113 | 2.1604 | 0.0342 | 96.5834 | 0.6755 | 0.9191 |
| fx_plus_macro | var | 151 | 2.8415 | 2.1941 | 0.0347 | 96.5262 | 0.6821 | 0.9289 |
| fx_only | lstm | 151 | 2.8994 | 2.2043 | 0.0348 | 96.5156 | 0.6623 | 0.9478 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_only | ar | 151 | 3.1322 | 2.3130 | 0.0367 | 96.3350 | 0.6490 | 1.0239 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_only | lstm | 5 | 0.9101 | 0.0165 | 0.8932 | 0.9294 | 0.9087 |
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9527 | 0.0270 | 0.9154 | 0.9903 | 0.9313 |
| walk_forward | 1 | fx_only | lstm | 5 | 1.0357 | 0.0066 | 1.0287 | 1.0433 | 1.0299 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0407 | 0.0187 | 1.0251 | 1.0709 | 1.0336 |
| walk_forward | 3 | fx_only | lstm | 5 | 2.0184 | 0.0051 | 2.0115 | 2.0258 | 2.0145 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.1674 | 0.2356 | 1.9543 | 2.5678 | 2.1140 |
| walk_forward | 6 | fx_only | lstm | 5 | 2.9095 | 0.0130 | 2.8916 | 2.9201 | 2.8994 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.8025 | 0.0725 | 2.7239 | 2.9077 | 2.7694 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
