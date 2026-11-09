# Decision log

Short records of non-obvious choices: what was decided and why.

| # | Decision | Why |
|---|---|---|
| D1 | No LLM or agent components; deterministic batch pipeline | The task is supervised forecasting. Agentic behaviour adds risk and no value |
| D2 | Random walk is a mandatory baseline | Level-MAPE "accuracy" is about 98–99 % even for a no-change forecast; beating the RW is the real test (Meese–Rogoff) |
| D3 | Macro inputs are US − India differentials of growth rates and rates, not raw level differences | Raw levels have different units and currencies (e.g. USD vs INR money supply) |
| D4 | Expanding-window walk-forward; scalers and tuning fit on train only | The paper's single split and full-sample scaling leak information |
| D5 | OECD-harmonised FRED series where possible; `SPASTT01*` instead of `SP500` | Like-for-like definitions across countries; FRED's `SP500` only covers about 10 years |
| D6 | PyTorch for the LSTM; Python 3.11/3.12 venv | Library wheel availability; avoids TensorFlow version friction |
| D7 | Clark–West test for FX-only vs FX+macro | The models are nested, so Diebold–Mariano is biased in that case |
| D8 | FRED-only panel, 1994-04 to 2023-01; test periods: paper replication (2016-08 to 2018-12) and extension (2019-01 to 2023-01) | Several India series on FRED stopped updating (IP 2023-01, CPI 2025-03). 2023-01 is the last month where all 17 series exist. Splicing in RBI/MOSPI data would add a second source with different definitions; this can be revisited if a longer test period is needed |
| D9 | India M3 uses the non-seasonally-adjusted series `MABMM301INM189N` | The SA series `MABMM301INM189S` ends 2018-12. Seasonality is handled at the feature step (12-month growth or seasonal adjustment fit on training data only) |
| D10 | Downloads are sequential with retry and backoff | FRED dropped or timed out parallel requests during testing on 2026-11-09 |
| D11 | Keep `IRSTCI01INM156N` for India, documented as a policy-rate proxy | It matches the RBI central-bank rate in 346/349 months, flat at 6% from 2003 to 2012. FRED has no Indian market short rate before 2011 (the T-bill series starts 2011-11). Possible upgrade: RBI DBIE weighted-average call rate |
| D12 | Macro features are 12-month log changes (CPI, IP, M3, reserves), monthly equity returns, the rate differential and a trade-balance ratio | Unit-free and comparable across countries; 12-month changes remove seasonality from NSA India series with no fitted parameters (so no leakage) |
| D13 | Diagnostics that set model specification use the training period only (to 2016-07) | Prevents the test period influencing differencing, lag order or the VAR-vs-VECM choice |
