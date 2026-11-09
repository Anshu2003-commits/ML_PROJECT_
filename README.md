# USD/INR Forecasting with Macroeconomic Indicators: VAR vs SVR vs LSTM

ML coursework project. It extends Giri & Kaushik (2020), [arXiv:2002.10247](https://arxiv.org/abs/2002.10247).

## Research questions
1. **RQ1:** Using multivariate macroeconomic indicators, how accurately can USD/INR be forecast? How do SVR (machine learning) and LSTM (deep learning) compare with the econometric VAR model, and does any of them beat a random-walk benchmark?
2. **RQ2:** For each model family, how much do macroeconomic indicators improve forecasts compared with using the exchange-rate history alone?

## What is different from the base paper
- A random-walk benchmark and Theil's U (a high level-"accuracy" is easy for FX, so the real test is beating the no-change forecast)
- Expanding-window walk-forward validation with leakage-free scaling and tuning
- Theory-based US−India differentials (inflation, growth, interest rates, money, trade)
- Univariate vs multivariate ablation for every model family
- Diebold–Mariano and Clark–West significance tests
- Multi-horizon forecasts (1, 3, 6 months); replication on 1994–2018 plus an extension to the latest data

## Headline results
Full numbers: [results/final](results/final/README.md) · tests: [analysis](results/final/analysis.md) · narrative: [docs/findings.md](docs/findings.md)

- **The base paper's results do not beat a "no change" forecast.** On its own test window (2016-08 to 2018-12), the random walk has RMSE 0.95 INR (1.28 on end-of-month rates). The paper's best model, the LSTM, has 1.69.
- **RQ1:** VAR, SVR and LSTM with macro inputs forecast within about ±10% of the random walk's RMSE. **None is significantly better than the random walk or drift** (Diebold–Mariano, 5%) over 2010–2023 at 1, 3 or 6 months. ML/DL models are not significantly better than the VAR overall. The paper's LSTM > SVM > VAR ranking is not reproduced.
- **RQ2:** macro differentials carry **statistically significant but small** predictive information in 6 of 9 family/horizon cases (Clark–West, walk-forward), mostly at 3–6 months. RMSE gains are at most about 9%, and there are none on the paper's short window.
- Macro models do best in 2010–2014 (taper tantrum) and worst in 2020–2023 (COVID). Results are robust to 1–2 month publication lags.

## Status
See **[docs/PLAN.md](docs/PLAN.md)** for the full plan, architecture and milestones.

| Milestone | Status |
|---|---|
| M0 Plan and setup | ✅ |
| M1 Data layer: 17 FRED series, 1994-04 to 2023-01 ([data dictionary](docs/data_dictionary.md)) | ✅ |
| M2 EDA and diagnostics ([report](results/eda/README.md)) | ✅ |
| M3 Features and walk-forward splitter ([splits](results/splits/README.md)) | ✅ |
| M4 Baselines and metrics ([results](results/baselines/README.md), [findings](docs/findings.md)) | ✅ |
| M5 VAR and VECM ([results](results/var/README.md)) | ✅ |
| M6 SVR ([results](results/svr/README.md)) | ✅ |
| M7 LSTM ([results](results/lstm/README.md)) | ✅ |
| M8 Full evaluation: significance, sub-periods, publication lags, ablation ([analysis](results/final/analysis.md)) | ✅ |
| M9 Report | ⏳ |

## Setup
```bash
python -m venv .venv          # Python 3.11 or 3.12 recommended
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
python -m fxforecast.data              # validate cached FRED data, build data/processed/panel.parquet
python -m fxforecast.experiments.eda   # figures + diagnostics -> results/eda/README.md
python -m fxforecast.experiments.splits  # train/test folds -> results/splits/README.md
python -m fxforecast.experiments.run --stage baselines --models random_walk drift ar
python -m fxforecast.experiments.run --stage var --models drift ar var vecm
python -m fxforecast.experiments.run --stage svr --models drift ar var svr   # ~12 min
python -m fxforecast.experiments.run --stage lstm --models drift ar var lstm # ~6 min

# final evaluation (all models, ~20 min) + publication-lag robustness + tests/ablation
python -m fxforecast.experiments.run --stage final --models drift ar var vecm svr lstm
python -m fxforecast.experiments.run --stage publag1 --models svr lstm --feature-sets fx_plus_macro --publication-lag 1
python -m fxforecast.experiments.run --stage publag2 --models svr lstm --feature-sets fx_plus_macro --publication-lag 2
python -m fxforecast.experiments.analysis   # -> results/final/analysis.md

python -m pytest                       # run the test suite
```

## Data
Monthly series from [FRED](https://fred.stlouisfed.org/), downloaded through the public CSV endpoint (no API key needed). A raw snapshot is committed in `data/raw/`. See the [data dictionary](docs/data_dictionary.md).
