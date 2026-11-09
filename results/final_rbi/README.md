# Results: final_rbi

Run `final_rbi-20261109T184136Z-56da7133`. Reproduce with `python -m fxforecast.experiments.run --stage final_rbi --models drift ar var vecm svr lstm`.

Level metrics are on INR per USD. **theil_u** = RMSE / RMSE(random walk) on the same origins (below 1 = beats "no change").
**hit_rate** = share of correct up/down calls (undefined for the random walk, which never calls a direction).
Settings each model chose on each refit (lag order, cointegration rank, ...) are in `model_choices.csv`.

### Replication, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | ar | 29 | 0.8470 | 0.6092 | 0.0090 | 99.0991 | 0.7241 | 0.8945 |
| fx_only | svr | 29 | 0.8682 | 0.6210 | 0.0092 | 99.0822 | 0.6897 | 0.9169 |
| fx_plus_macro | var | 29 | 0.9023 | 0.6663 | 0.0098 | 99.0153 | 0.7241 | 0.9530 |
| fx_only | lstm | 29 | 0.9087 | 0.6849 | 0.0101 | 98.9894 | 0.5862 | 0.9597 |
| fx_plus_macro | lstm | 29 | 0.9285 | 0.7089 | 0.0105 | 98.9531 | 0.5517 | 0.9806 |
| fx_only | drift | 29 | 0.9405 | 0.7281 | 0.0107 | 98.9256 | 0.5517 | 0.9933 |
| fx_only | random_walk | 29 | 0.9469 | 0.7247 | 0.0107 | 98.9340 |  | 1.0000 |
| fx_plus_macro | svr | 29 | 0.9694 | 0.7414 | 0.0110 | 98.8972 | 0.6207 | 1.0238 |
| fx_plus_macro | vecm | 29 | 1.1756 | 0.9690 | 0.0143 | 98.5715 | 0.4828 | 1.2416 |

### Walk Forward, h = 1 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 156 | 1.0152 | 0.7908 | 0.0128 | 98.7216 | 0.5769 | 0.9818 |
| fx_plus_macro | lstm | 156 | 1.0180 | 0.7975 | 0.0129 | 98.7094 | 0.5897 | 0.9845 |
| fx_plus_macro | svr | 156 | 1.0238 | 0.8075 | 0.0130 | 98.6969 | 0.5449 | 0.9901 |
| fx_only | lstm | 156 | 1.0299 | 0.8014 | 0.0129 | 98.7095 | 0.5705 | 0.9960 |
| fx_only | random_walk | 156 | 1.0341 | 0.7946 | 0.0128 | 98.7160 |  | 1.0000 |
| fx_only | svr | 156 | 1.0404 | 0.7940 | 0.0128 | 98.7230 | 0.6090 | 1.0061 |
| fx_only | ar | 156 | 1.0528 | 0.7929 | 0.0128 | 98.7235 | 0.6026 | 1.0181 |
| fx_plus_macro | var | 156 | 1.1145 | 0.8408 | 0.0134 | 98.6586 | 0.5577 | 1.0778 |
| fx_plus_macro | vecm | 156 | 1.2795 | 0.9296 | 0.0146 | 98.5382 | 0.5769 | 1.2373 |

### Walk Forward, h = 3 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_only | drift | 154 | 1.9837 | 1.4901 | 0.0240 | 97.6036 | 0.6169 | 0.9564 |
| fx_plus_macro | lstm | 154 | 2.0077 | 1.5629 | 0.0252 | 97.4801 | 0.6234 | 0.9680 |
| fx_only | lstm | 154 | 2.0145 | 1.5036 | 0.0242 | 97.5812 | 0.6299 | 0.9713 |
| fx_only | svr | 154 | 2.0482 | 1.5562 | 0.0250 | 97.4988 | 0.5519 | 0.9875 |
| fx_plus_macro | var | 154 | 2.0487 | 1.5742 | 0.0254 | 97.4615 | 0.5649 | 0.9878 |
| fx_plus_macro | svr | 154 | 2.0622 | 1.5952 | 0.0254 | 97.4573 | 0.5519 | 0.9943 |
| fx_only | random_walk | 154 | 2.0741 | 1.5322 | 0.0245 | 97.5498 |  | 1.0000 |
| fx_only | ar | 154 | 2.1489 | 1.6527 | 0.0265 | 97.3452 | 0.5455 | 1.0361 |
| fx_plus_macro | vecm | 154 | 2.6313 | 1.9981 | 0.0312 | 96.8824 | 0.4740 | 1.2687 |

### Walk Forward, h = 6 month(s)

| feature_set | model | n | rmse | mae | mape | accuracy_pct | hit_rate | theil_u |
|---|---|---|---|---|---|---|---|---|
| fx_plus_macro | lstm | 151 | 2.7517 | 2.1234 | 0.0337 | 96.6284 | 0.6556 | 0.8996 |
| fx_only | drift | 151 | 2.8113 | 2.1604 | 0.0342 | 96.5834 | 0.6755 | 0.9191 |
| fx_plus_macro | var | 151 | 2.8755 | 2.2160 | 0.0350 | 96.5006 | 0.6689 | 0.9400 |
| fx_only | lstm | 151 | 2.8994 | 2.2043 | 0.0348 | 96.5156 | 0.6623 | 0.9478 |
| fx_only | svr | 151 | 2.9447 | 2.2719 | 0.0361 | 96.3903 | 0.6424 | 0.9627 |
| fx_plus_macro | svr | 151 | 3.0512 | 2.4520 | 0.0381 | 96.1871 | 0.5563 | 0.9975 |
| fx_only | random_walk | 151 | 3.0589 | 2.3400 | 0.0365 | 96.3511 |  | 1.0000 |
| fx_only | ar | 151 | 3.1322 | 2.3130 | 0.0367 | 96.3350 | 0.6490 | 1.0239 |
| fx_plus_macro | vecm | 151 | 3.9938 | 3.0830 | 0.0470 | 95.3032 | 0.5563 | 1.3056 |

### Seed variability (each seed scored on its own; the model above is the seed average)

| scheme | horizon | feature_set | model | seeds | rmse_mean | rmse_std | rmse_min | rmse_max | rmse_ensemble |
|---|---|---|---|---|---|---|---|---|---|
| replication | 1 | fx_only | lstm | 5 | 0.9101 | 0.0165 | 0.8932 | 0.9294 | 0.9087 |
| replication | 1 | fx_plus_macro | lstm | 5 | 0.9311 | 0.0184 | 0.9107 | 0.9473 | 0.9285 |
| walk_forward | 1 | fx_only | lstm | 5 | 1.0357 | 0.0066 | 1.0287 | 1.0433 | 1.0299 |
| walk_forward | 1 | fx_plus_macro | lstm | 5 | 1.0216 | 0.0068 | 1.0119 | 1.0278 | 1.0180 |
| walk_forward | 3 | fx_only | lstm | 5 | 2.0184 | 0.0051 | 2.0115 | 2.0258 | 2.0145 |
| walk_forward | 3 | fx_plus_macro | lstm | 5 | 2.0212 | 0.0197 | 1.9949 | 2.0482 | 2.0077 |
| walk_forward | 6 | fx_only | lstm | 5 | 2.9095 | 0.0130 | 2.8916 | 2.9201 | 2.8994 |
| walk_forward | 6 | fx_plus_macro | lstm | 5 | 2.7794 | 0.0546 | 2.7104 | 2.8487 | 2.7517 |

### Base paper, for reference (replication period, h = 1, Bloomberg data)

| model | rmse | mape | mpe | accuracy_pct |
|---|---|---|---|---|
| VAR (paper) | 2.9381 | 0.0369 | 0.0286 | 96.3100 |
| SVM (paper) | 2.5585 | 0.0283 | -0.0261 | 97.1700 |
| LSTM (paper) | 1.6872 | 0.0217 | 0.0031 | 97.8300 |
