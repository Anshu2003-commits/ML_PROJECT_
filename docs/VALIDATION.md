# Final validation (Phase 8)

The project template's final review checklist, filled in for v1.0. Each item points to the evidence.

## 1. What was implemented
A deterministic, configuration-driven forecasting pipeline for monthly USD/INR:
- **Data:** 17 FRED series, validated and aligned (1994-04 to 2023-01).
- **Features:** US−India differentials.
- **Models:** 7 (random walk, drift, AR, VAR, VECM, SVR, LSTM); SVR and LSTM run with two information sets (FX only / FX + macro).
- **Evaluation:** a replication split plus an expanding-window walk-forward evaluation at 1, 3 and 6 months.
- **Statistical tests:** Diebold–Mariano and Clark–West.
- **Further analysis:** sub-periods, publication-lag robustness and a VAR ablation.
- **Write-up:** a written report ([report/REPORT.md](../report/REPORT.md)).

## 2. Architecture and data flow
`config → data (fetch, validate, panel) → features (transforms, supervised, scaling, diagnostics) → models → evaluation (splits, backtest, metrics, stat_tests) → experiments (eda, splits, run, analysis, figures, tracking) → reporting (plots, tables)`.
Dependencies point one way only, with no cycles: models never import data code, and orchestration lives only in `experiments/`. Full diagram: [PLAN.md §2.2](PLAN.md).

## 3. Files
- **Source:** `src/fxforecast/`, 2,592 lines.
- **Tests:** `tests/`, 1,251 lines and 130 tests.
- **Config:** `configs/experiment.yaml`.
- **Data:** `data/raw/`, committed FRED snapshot.
- **Results:** `results/` (eda, splits, baselines, var, svr, svr_grid_v1, lstm, final, publag1, publag2).
- **Docs:** PLAN, data dictionary, decisions (35 entries), findings, this file.

## 4. Key decisions
See [decisions.md](decisions.md). Most consequential:
- D2: random walk as a mandatory benchmark.
- D3/D12: unit-free differentials.
- D14: origin indexing with an h-month embargo.
- D18: identical forecast origins for every model.
- D33: DM and CW tests.

## 5. Checklist

### Architecture
- [x] Every module has one responsibility (module docstrings state it).
- [x] Data flow is linear and documented; dependency boundaries respected.
- [x] No unnecessary abstractions. There is one small `Forecaster` protocol; no frameworks, plugin systems or agents (D1).

### Functionality
- [x] Both research questions are answered with tested numbers ([REPORT §5.2–5.3](../report/REPORT.md)).
- [x] Edge cases handled loudly:
  - missing months or future dates (validation);
  - missing FX levels;
  - empty folds;
  - forecasts without a matching benchmark;
  - unknown models;
  - publication lag requested for VAR/VECM.
- [x] Termination conditions are defined:
  - LSTM early stopping (patience 20, max 300 epochs);
  - bounded fetch retries with backoff (3 attempts).

### Leakage and validity (the "agent behaviour / side effects" equivalent for an ML study)
- [x] Train origins satisfy *t* + *h* ≤ refit date on every fold (`tests/evaluation/test_splits.py`).
- [x] Features never use data after the origin (`test_supervised.py`, `test_transforms.py`).
- [x] Scalers, tuning (inner CV with an h−1 gap) and early stopping see training rows only (`test_scaling.py`, `test_svr.py`, `test_lstm.py`).
- [x] VAR/VECM fits and forecasts are unchanged when future data is perturbed (`test_var.py`).
- [x] All models forecast identical origins; this is checked by the runner and the Theil's U computation.
- [x] Specification diagnostics use only the training window (D13). The SVR grid change was decided from tuning choices, not test scores (D29). The ablation is not used to select variables.

### Testing
- [x] 130 automated tests pass (`python -m pytest`):
  - unit tests (metrics checked against hand-computed values);
  - statistical tests on synthetic data with known answers, including the DM test's false-rejection rate;
  - coefficient recovery (VAR, AR);
  - determinism (SVR, LSTM seeds);
  - an end-to-end run on the real data.
- [x] Regression check: the final combined run reproduced every per-stage result exactly.
- [x] Clean-clone reproduction (Section 6).
- **Not tested:** live FRED downloads (network). The fetch logic is tested with injected fake HTTP responses; the real download was exercised manually on 2026-11-09.

### Observability
- [x] Every run writes `results/runs/<run_id>/`. It records:
  - the config;
  - the git commit and a dirty flag;
  - library versions;
  - a SHA-256 hash for each data file;
  - wall-clock time;
  - all forecasts with the fit time per fold;
  - the settings each model chose at each refit.
- [x] Each stage writes a human-readable report (`results/<stage>/README.md`).
- [x] Failures raise with specific messages; nothing is silently dropped.

### Security and reliability
- [x] No secrets: FRED's public CSV endpoint needs no API key; `.env` is git-ignored if one is ever added.
- [x] Raw data is cached and committed, so experiments need no network access.
- [x] Pinned minimum versions in `requirements.txt`; Python 3.11 virtual environment.

## 6. Clean-clone reproduction
On 2026-11-09 the repository was cloned fresh from GitHub (commit `f4935ed`, tag `v0.8-eval`) into an empty folder. The full pipeline was run from the clone's own code, starting from the committed raw FRED snapshot with no processed data:

`data → eda → splits → run final → run publag1 → run publag2 → analysis → pytest`

- **Result:** all **21 committed result tables** (`results/eda/tables`, `results/final`, `results/publag1`, `results/publag2`) were reproduced with a **maximum absolute numerical difference of 0**, and the text columns were identical.
- **Tests:** the clone's test suite passed (130/130).
- **Run time:** 2,266 s (about 38 minutes) on a laptop CPU.

The report figures (`experiments/figures.py`) were added after that commit. They are generated from the same run outputs.

## 7. Remaining limitations
See [REPORT §7](../report/REPORT.md#7-limitations):
- revised (not real-time) data;
- the India policy-rate proxy;
- the panel ends in 2023-01;
- a small sample;
- monthly-average FX;
- no multiple-testing correction or Model Confidence Set;
- modest SVR/LSTM search spaces.
