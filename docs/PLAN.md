# Project Plan: Forecasting USD/INR with Macroeconomic Indicators
### VAR vs SVR vs LSTM, with univariate vs multivariate comparisons

> Base paper: A. K. Giri & M. Kaushik, *Forecasting Foreign Exchange Rate: A Multivariate Comparative Analysis between Traditional Econometric, Contemporary Machine Learning & Deep Learning Techniques* (arXiv:2002.10247, 2020).
> Repo: https://github.com/Anshu2003-commits/ML_PROJECT_

This plan follows the 8-phase engineering template (Understand → Design → Structure → Implement → Observe/Evaluate → Production-readiness → Optimise → Validate). It is adapted to an ML research project. **There is no LLM or agent here.** Every step is a deterministic, reproducible pipeline, which is the correct design for this problem (see §1.4).

---

## 0. TL;DR: what this project does and why it will stand out

| | Base paper | This project |
|---|---|---|
| Benchmark | None. "97.8 % accuracy" is reported with nothing to compare it to | **Random walk (no-change) baseline**, plus Theil's U relative to it |
| Validation | One 90/10 split | **Expanding-window walk-forward** (rolling origin) |
| Leakage control | Min–max scaling appears to use the full sample | Scalers and tuning are fit **on training windows only** |
| Features | Raw level differences USA − India (units differ: USD vs INR money supply) | **Theory-based differentials** (inflation, growth, interest, money and trade differentials) |
| Univariate vs multivariate | Not tested | **Clean ablation per model family**: this answers RQ2 directly |
| Significance | None | **Diebold–Mariano** (non-nested) and **Clark–West** (nested) tests |
| Horizons | 1-month only | h = 1, 3, 6 months |
| Sample | 1994–2018 | **Replication** (1994–2018) **plus extension** to the latest data (COVID, 2022 Fed hikes) |
| Realism | Uses macro data at t−1 as if released instantly | Robustness check with **publication lags** |
| Engineering | Not given | Config-driven, tested, seeded, logged, versioned on GitHub |

**The key insight to build the report around:** for exchange rates, a level-MAPE "accuracy" of about 97 % is almost free. The no-change forecast (next month = this month) is expected to score about 98–99 % on USD/INR because the rate typically moves only around 1 % per month. Milestone M4 measures this exactly. The real question is whether any model **beats the random walk**. This is the famous **Meese–Rogoff (1983) puzzle**, and it is exactly what RQ2 is asking. If the models do not beat it, that is still a valid and interesting finding, as long as it is measured rigorously.

---

## 1. PHASE 1: Problem understanding

### 1.1 Research questions (suggested sharpening)

Your two RQs currently overlap: both end with "how do SVM and LSTM compare with VAR". To give each one a distinct job:

- **RQ1 (model comparison, multivariate setting):** Using multivariate macroeconomic indicators, how accurately can USD/INR be forecast out-of-sample? How do SVR (ML) and LSTM (DL) compare with the econometric VAR model, and does any of them significantly beat a random-walk benchmark?
- **RQ2 (value of macro information, ablation):** For each model family (econometric, ML, DL), how much does adding macroeconomic differentials improve forecasting accuracy over the same model trained on historical exchange-rate information alone? Does the gain differ between linear (VAR/AR) and nonlinear (SVR/LSTM) models?

"SVM" in this context means **Support Vector Regression (SVR)**. Use that term in the report.

### 1.2 Answers to the template's questions

| Question | Answer |
|---|---|
| Intended user | Course examiners and readers; secondarily, analysts who want to know if macro data helps FX forecasting |
| Problem | Forecast monthly USD/INR and quantify (a) which model class is best and (b) whether macro data adds value |
| Measurable outcome | Out-of-sample RMSE, MAE, MAPE, directional accuracy, Theil's U vs random walk, DM/CW p-values |
| What needs ML vs deterministic logic | Data fetch, cleaning, transforms, splits and metrics are deterministic code. Only the model fitting is "learning". **No LLM or agent component.** |
| Autonomy | None needed. It is a batch pipeline run by one command per experiment |
| Inputs | FRED monthly series for the US and India (§2.1) |
| Constraints | About 370–390 monthly observations (a small dataset, so the LSTM can overfit); CPU laptop; single developer |
| Risks | Data gaps or discontinued series; look-ahead leakage; LSTM instability across seeds; reporting misleading "accuracy" |

### 1.3 Requirements

**Functional**
1. Download and cache all series. Build a single aligned monthly panel.
2. Run stationarity diagnostics (ADF, KPSS), Granger causality and Johansen cointegration.
3. Train and evaluate the full **model grid** (§2.3) under walk-forward validation for h ∈ {1, 3, 6}.
4. Produce comparison tables, significance tests, plots and feature-importance analysis.
5. Replicate the paper's setup (1994-04 to 2018-12, test from 2016-08) as a sanity anchor.

**Non-functional**
- Reproducible: the same config, data snapshot and seed give the same numbers.
- Leakage-free: no information from the test window enters training (enforced by tests).
- Traceable: every result file links back to the config, data hash and code commit that produced it.
- Simple: no unnecessary abstractions. One package, clear modules.

### 1.4 Assumptions (documented, not asked)
- Target = **monthly average** USD/INR (FRED `EXINUS`, INR per 1 USD), modelled as a log-return. Forecasts are converted back to levels for level metrics.
- Macro variables enter as **US − India differentials** (economic theory: PPP → inflation differential; uncovered interest parity → interest differential; monetary model → money and output differentials).
- FRED's current (revised) data vintage is used. Real-time vintages (ALFRED) are out of scope and noted as a limitation.
- Python 3.11 or 3.12 virtual environment (the system Python 3.14 may lack wheels for some ML libraries). **PyTorch** is used for the LSTM.

### 1.5 Success criteria
- The pipeline runs end to end from one command, and all tests pass.
- Every model in the grid has walk-forward results at all horizons, with significance tests vs the random walk and vs each other.
- The replication run lands in the same ballpark as the paper (LSTM < SVR < VAR in RMSE), **or** the report explains why it does not.
- Both RQs are answered with numbers, tests and an honest discussion.

### 1.6 Failure boundaries and human checkpoints
- If a FRED series is missing or too short, the pipeline **fails loudly** with a coverage report. A human then picks a fallback source (RBI DBIE, IMF IFS, MOSPI) and records the decision in `docs/decisions.md`.
- Human review gates: (1) after the data panel is built (sanity-check the plots), (2) after the first baseline results, (3) before writing the final conclusions.

---

## 2. PHASE 2: Architecture design

The template's Know / Act / Remember / Coordinate map onto an ML pipeline like this:

| Template concept | In this project |
|---|---|
| **Know** (context, knowledge sources) | FRED data and the data dictionary; economic theory that guides feature design |
| **Act** (tools, validation, failure handling) | Fetch → validate → transform → train → evaluate. Each step validates its input and output |
| **Remember** (memory) | *Raw data snapshot* (episodic: what we saw on fetch date X), *processed panel* (working), *run artefacts* (metrics, predictions, configs). No other memory needed |
| **Coordinate** (orchestration, termination) | One experiment runner that loops over models × feature sets × horizons × folds. Bounded loops, early stopping for the LSTM, fixed seeds |

### 2.1 Data dictionary (FRED, monthly)

> **Update (M1, 2026-11-09):** the verified list, coverage and substitutions are in [`data_dictionary.md`](data_dictionary.md). The final panel covers 1994-04 to 2023-01 (346 months), limited by India IP. India M3 uses the non-SA series `MABMM301INM189N`. The table below is the original provisional list.

FRED's `SP500` series only covers the last ~10 years, and US M3 was discontinued in 2006. The OECD-harmonised series (`...MEI`, `MABMM301...`, `SPASTT01...`) give **like-for-like definitions across both countries**, which is better for differentials. **All IDs must be confirmed by the fetch step**: some OECD/IMF series on FRED have been discontinued or stopped updating, and the validation step will report actual coverage.

| Variable | India (FRED ID) | USA (FRED ID) | Feature built |
|---|---|---|---|
| USD/INR (target) | `EXINUS` (monthly avg; `DEXINUS` is daily) | — | Δlog(FX) |
| CPI | `INDCPIALLMINMEI` | `CPIAUCSL` | Inflation differential: Δlog CPI_US − Δlog CPI_IN |
| Industrial production | `INDPROINDMISMEI` | `INDPRO` | Growth differential |
| Short-term interest rate | `IRSTCI01INM156N` (call money) | `FEDFUNDS` | i_US − i_IN (level, then Δ if non-stationary) |
| Money supply (M3) | `MABMM301INM189S` | `MABMM301USM189S` | Money-growth differential |
| Total reserves excl. gold | `TRESEGINM052N` | `TRESEGUSM052N` | Δlog reserves, per country or as a differential |
| Share prices | `SPASTT01INM661N` | `SPASTT01USM661N` | Equity-return differential |
| Exports / imports | `XTEXVA01INM667S`, `XTIMVA01INM667S` | `XTEXVA01USM667S`, `XTIMVA01USM667S` | Net exports / trade balance (scaled) |

Download method: FRED's public CSV endpoint (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`), which needs **no API key**. If the `fredapi` library is used later, the key goes in `.env` (git-ignored) and is never committed.

### 2.2 Data flow

```
FRED CSVs ──fetch──▶ data/raw/<id>.csv (+ fetch metadata: date, sha256)
                        │
                     validate (monthly freq, coverage, no future dates, no duplicates)
                        │
                     align ──▶ data/processed/panel.parquet  (one row per month)
                        │
          ┌─────────────┴──────────────┐
   diagnostics                     features
   (ADF/KPSS, Granger,             (log-diffs, differentials, lags,
    Johansen, correlation)          FX-only vs FX+macro sets)
                                        │
                              walk-forward splitter (expanding window)
                                        │
                 per fold: fit scaler on train → fit model → forecast h steps
                                        │
                     results/runs/<run_id>/ {config.yaml, predictions.csv, metrics.json, log.txt}
                                        │
                   evaluation: metrics, Theil's U, DM / Clark–West tests
                                        │
                     reporting: tables + figures for the write-up
```

### 2.3 Model grid (the core of both RQs)

| Family | **FX-only** (univariate, for RQ2) | **FX + macro** (multivariate, for RQ1) |
|---|---|---|
| Naive benchmark | Random walk (ŷ = y_t), RW with drift | — |
| Econometric | AR(p) / ARIMA (p by AIC) | **VAR(p)** on stationary series (p by AIC, the paper used 3); VECM if Johansen finds cointegration |
| Machine learning | SVR on FX lags | **SVR** on FX lags + macro-differential lags |
| Deep learning | LSTM on an FX sequence | **LSTM** on an FX + macro sequence |

- **RQ1** = compare the right-hand column (plus RW) against each other.
- **RQ2** = within each row, right column vs left column (same model, more information). Clark–West is the correct test here because the FX-only model is *nested* in the multivariate one.

Hyperparameters:
- **SVR**: grid over C, γ and ε with `TimeSeriesSplit` *inside the training window* (the paper's best was C = 1000, γ = 0.001, RBF kernel).
- **LSTM**: 1 layer, 16–64 units, dropout, lookback 6–12 months, early stopping on a time-ordered validation tail, **5–10 seeds** (report mean ± std).

### 2.4 Evaluation design
- **Replication run:** train 1994-04 to 2016-07, test 2016-08 to 2018-12 (as in the paper), level metrics MAPE, MPE, RMSE and "Accuracy = 100 − MAPE".
- **Main run:** expanding-window walk-forward. First origin around 2010, re-fit every 12 months (or every month for cheap models), forecast h = 1, 3, 6, test up to the latest month available.
- **Metrics:** RMSE, MAE, MAPE, MPE, directional accuracy (hit rate of up/down moves), Theil's U (model RMSE / RW RMSE; below 1 means it beats the RW).
- **Tests:** Diebold–Mariano with Harvey–Leybourne–Newbold correction (model vs model), Clark–West (nested, FX-only vs FX+macro). *Stretch:* Model Confidence Set.
- **Sub-period analysis:** calm vs stress periods (2013 taper tantrum, 2020 COVID, 2022 tightening).
- **Interpretability:** Granger causality; permutation importance (model-agnostic, computed on the test window); leave-one-variable-group-out ablation (which macro variable matters most). *Stretch:* SHAP.
- **Robustness:** apply 1–2 month publication lags to macro inputs and re-run the best configs.

---

## 3. PHASE 3: Codebase structure

Grouped by **reason for change**. Dependencies flow one way: `config → data → features → models → evaluation → experiments → reporting`. Models never import data-fetching code. Notebooks only *call* the package and never hold logic.

```
ML_PROJECT_/
├── README.md                    # what, how to run, headline results
├── requirements.txt
├── configs/
│   └── experiment.yaml          # sample periods, horizons, model grid, seeds
├── data/
│   ├── raw/                     # FRED snapshots (small CSVs, committed for reproducibility)
│   └── processed/               # aligned panel (regenerable, git-ignored)
├── docs/
│   ├── PLAN.md                  # this file
│   ├── data_dictionary.md       # final, verified series list and coverage
│   └── decisions.md             # short decision log (why X over Y)
├── src/fxforecast/
│   ├── config.py                # load and validate YAML config
│   ├── data/
│   │   ├── fetch.py             # I/O: download + cache + metadata
│   │   ├── validate.py          # schema / coverage / frequency checks
│   │   ├── panel.py             # align series into one monthly panel
│   │   └── __main__.py          # CLI: python -m fxforecast.data [--refresh]
│   ├── features/
│   │   ├── transforms.py        # log-diff, differentials, lags, FX-only vs macro sets
│   │   └── diagnostics.py       # ADF, KPSS, Granger, Johansen, correlation
│   ├── models/                  # every model: fit(train) / predict(h), same interface
│   │   ├── baselines.py         # random walk, drift, AR/ARIMA
│   │   ├── var.py               # VAR / VECM (statsmodels)
│   │   ├── svr.py               # SVR (scikit-learn)
│   │   └── lstm.py              # LSTM (PyTorch)
│   ├── evaluation/
│   │   ├── splits.py            # expanding-window walk-forward splitter
│   │   ├── metrics.py           # RMSE, MAE, MAPE, MPE, hit rate, Theil's U
│   │   └── stat_tests.py        # Diebold–Mariano, Clark–West
│   ├── experiments/
│   │   ├── eda.py               # M2: figures + diagnostics → results/eda/README.md
│   │   └── run.py               # orchestration: grid × horizons × folds → results/
│   └── reporting/
│       ├── tables.py            # DataFrame → Markdown
│       └── plots.py             # matplotlib figures (one style for the whole report)
├── notebooks/                   # EDA + final figures; import from src only
├── tests/                       # mirrors src/
└── results/                     # run outputs (git-ignored except final summary tables)
```

Before adding any new file, check that the responsibility does not already exist in one of these modules.

---

## 4. PHASE 4: Implementation roadmap (milestones = commits / tags on GitHub)

Each milestone is developed on a short-lived branch, merged into `main`, and tagged.

| # | Milestone | Deliverables | Tests | Tag |
|---|---|---|---|---|
| M0 | **Setup** | Repo, plan, venv, requirements, config skeleton | — | `v0.0-plan` |
| M1 | **Data layer** | `fetch.py`, `validate.py`, raw snapshot, `data_dictionary.md` with real coverage | Validation catches bad frequency, gaps, future dates | `v0.1-data` |
| M2 | **EDA + diagnostics** | Plots, ADF/KPSS table, Granger, Johansen, correlation heatmap | Diagnostics on synthetic random walk vs white noise give the expected verdicts | `v0.2-eda` |
| M3 | **Features + splitter** | Differentials, lags, FX-only vs macro feature sets, walk-forward splitter | **No-leakage tests**: train indices always precede test; scaler fit only on train | `v0.3-features` |
| M4 | **Baselines + metrics** | RW, drift, AR; metrics; first results table | Metrics match hand-computed values; RW forecast equals the last value | `v0.4-baselines` |
| M5 | **VAR** | VAR/VECM, lag selection, replication-period result | Forecast shape and horizon; smoke test on synthetic VAR data recovers coefficients | `v0.5-var` |
| M6 | **SVR** | FX-only and multivariate SVR, time-series grid search | Tuning never touches the test fold | `v0.6-svr` |
| M7 | **LSTM** | FX-only and multivariate LSTM, early stopping, multi-seed | Deterministic with fixed seed; overfits a tiny batch (sanity check) | `v0.7-lstm` |
| M8 | **Full evaluation** | Walk-forward grid for h = 1, 3, 6; DM/CW tests; sub-periods; ablations; publication-lag robustness | Stat tests checked against known examples | `v0.8-eval` |
| M9 | **Report** | Final tables and figures, README results, written answers to RQ1 and RQ2 | Full pipeline re-run from clean clone | `v1.0` |

Coding rules: type hints, small single-purpose functions, no hard-coded paths or dates (everything in `configs/`), no secrets in the repo.

---

## 5. PHASE 5: Observability and evaluation (ML equivalent)

The template's "logging" maps to **experiment tracking**:

| Template item | Here |
|---|---|
| Task / trace ID | `run_id` = timestamp + short config hash |
| Model name / version | Model class + hyperparameters saved in `config.yaml` per run |
| Prompt version | → **config version + git commit SHA** |
| Tool calls / results | Per-step log lines: fetch, validate, fit, predict, with row counts |
| Latency per step | Wall-clock time per fit/predict, logged |
| Token cost | → **compute time** per model (report it: LSTM vs SVR vs VAR cost is a real trade-off) |
| Errors / retries | Fetch retries (bounded, with backoff); fits that fail are logged and marked, never silently dropped |
| Termination reason | LSTM: early stop epoch or max epochs |

Plus: data snapshot SHA-256, library versions and random seeds stored with each run.

**Offline evaluation = the test suite + the replication run.** If a code change shifts the replication numbers unexpectedly, that is a regression. Use deterministic assertions throughout.

---

## 6. PHASE 6: "Production readiness", scaled to a coursework project

Only what the risk level justifies:
- **Reliability:** bounded retries with backoff on FRED downloads; cached raw data so experiments never need the network.
- **Freshness:** each raw file records its fetch date; `make`/CLI flag `--refresh-data` to re-download deliberately.
- **Reproducibility:** pinned `requirements.txt`, seeds, committed raw snapshot.
- **Security:** no API keys needed. If any are added later, they go in `.env` (git-ignored).
- **CI (stretch):** GitHub Actions running `pytest` on every push. Cheap, and it shows engineering maturity.
- Not needed: scaling, queues, auth, and so on. Say so explicitly in the report rather than adding them.

---

## 7. PHASE 7: Optimisation (baseline-first)

1. Establish baselines first (RW, AR) before any complex model.
2. Each change (new feature, new hyperparameter range, a bigger LSTM) is one experiment compared against the previous best **on validation folds only**, never by looking at the test set.
3. Watch for the classic failure patterns: LSTM predicting "yesterday's value" (lag-copy), so check whether it beats RW; SVR flat-lining because of a poor ε or scaling; VAR with too many parameters (8 variables × p lags) overfitting, so consider fewer variables or a lower p.
4. Log every tried configuration, not just the winners. This avoids "garden of forking paths" over-optimism.

---

## 8. PHASE 8: Final validation checklist

- [ ] Every module has one clear responsibility; there are no circular imports.
- [ ] No-leakage tests pass; scalers and tuning are confined to training windows.
- [ ] All models are evaluated on identical test points and horizons.
- [ ] RW baseline is included in every table; Theil's U is reported.
- [ ] DM/CW p-values are reported; claims of "better" are backed by tests.
- [ ] LSTM results are reported as mean ± std over seeds.
- [ ] A clean clone plus one command reproduces the headline tables.
- [ ] The final report covers: what was built, data flow, decisions and rationale, test results, limitations.

### Known limitations to state upfront
- The revised data vintage is not real-time (ALFRED vintages are out of scope).
- The sample is small (about 30 years of monthly data), so deep-learning results are noisy.
- Structural breaks (policy regime changes, RBI intervention) violate stationarity assumptions.
- FRED/OECD coverage for India may force substitutions. All substitutions will be documented.

---

## 9. Report outline (for the final write-up)
1. Introduction and RQs
2. Literature: Meese–Rogoff puzzle, PPP/UIP/monetary models, ML/DL for FX (including the base paper)
3. Data and features (data dictionary, differentials, diagnostics)
4. Methodology (models, walk-forward design, metrics, tests)
5. Results: replication → RQ1 → RQ2 → robustness and interpretability
6. Discussion (why do or don't macro variables help? linear vs nonlinear)
7. Limitations and future work
8. Reproducibility appendix (repo link, how to run)
