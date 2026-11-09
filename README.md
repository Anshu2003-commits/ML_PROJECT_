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

## Status
Planning stage. See **[docs/PLAN.md](docs/PLAN.md)** for the full plan, architecture and milestones.

| Milestone | Status |
|---|---|
| M0 Plan and setup | ✅ |
| M1 Data layer | ⏳ |
| M2 EDA and diagnostics | ⏳ |
| M3 Features and walk-forward splitter | ⏳ |
| M4 Baselines and metrics | ⏳ |
| M5–M7 VAR, SVR, LSTM | ⏳ |
| M8 Full evaluation | ⏳ |
| M9 Report | ⏳ |

## Setup
```bash
python -m venv .venv          # Python 3.11 or 3.12 recommended
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data
Monthly series from [FRED](https://fred.stlouisfed.org/) (public CSV endpoint, no API key needed). See the data dictionary in [docs/PLAN.md](docs/PLAN.md#21-data-dictionary-fred-monthly).
