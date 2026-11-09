# Data dictionary

All data comes from FRED (Federal Reserve Bank of St. Louis) and is monthly. Raw files are in `data/raw/` as `<ID>.csv`. Each has a `<ID>.meta.json` that records the fetch time and SHA-256 hash.

**Snapshot:** fetched 2026-11-09. Rebuild with `python -m fxforecast.data`, or `--refresh` to re-download.

**Analysis panel:** 1994-04 to 2023-01, **346 months × 17 columns**, no missing values (`data/processed/panel.parquet`).

| Column | FRED ID | Description | Units / adjustment | Raw coverage |
|---|---|---|---|---|
| `fx` | `EXINUS` | India / U.S. foreign exchange rate (**target**) | INR per 1 USD, monthly average | 1973-01 to 2026-10 |
| `cpi_us` | `CPIAUCSL` | US CPI, all urban consumers | Index, SA | 1947-01 to 2026-09 (2025-10 missing\*) |
| `cpi_in` | `INDCPIALLMINMEI` | India CPI, all items (OECD MEI) | Index | 1957-01 to **2025-03** |
| `ip_us` | `INDPRO` | US industrial production | Index, SA | 1919-01 to 2026-09 |
| `ip_in` | `INDPROINDMISMEI` | India industrial production (OECD MEI) | Index | 1994-04 to **2023-01** ← binding end |
| `rate_us` | `FEDFUNDS` | Effective federal funds rate | % p.a. | 1954-07 to 2026-10 |
| `rate_in` | `IRSTCI01INM156N` | India call money / interbank rate (OECD) | % p.a. | 1968-01 to 2026-09 |
| `m3_us` | `MABMM301USM189S` | US broad money M3 (OECD) | USD, SA | 1960-01 to 2023-11 |
| `m3_in` | `MABMM301INM189N` | India broad money M3 (OECD) | INR, **not SA** | 1960-01 to 2023-09 |
| `reserves_us` | `TRESEGUSM052N` | US total reserves excluding gold (IMF) | Millions of USD | 1950-12 to 2026-09 |
| `reserves_in` | `TRESEGINM052N` | India total reserves excluding gold (IMF) | Millions of USD | 1950-12 to 2026-09 |
| `equity_us` | `SPASTT01USM661N` | US share price index (OECD) | Index | 1957-01 to 2026-09 |
| `equity_in` | `SPASTT01INM661N` | India share price index (OECD) | Index | 1957-01 to 2026-09 |
| `exports_us` | `XTEXVA01USM667S` | US exports of goods (OECD) | USD, SA | 1955-01 to 2026-08 |
| `exports_in` | `XTEXVA01INM667S` | India exports of goods (OECD) | USD, SA | 1990-01 to 2026-08 |
| `imports_us` | `XTIMVA01USM667S` | US imports of goods (OECD) | USD, SA | 1955-01 to 2026-08 |
| `imports_in` | `XTIMVA01INM667S` | India imports of goods (OECD) | USD, SA | 1990-01 to 2026-08 |

\* US CPI for October 2025 was never published (BLS data collection was suspended during the 2025 US government shutdown). It falls outside the analysis window. Reserve series also have gaps in the 1950s, which are likewise outside the window.

## Coverage notes and substitutions
- **Why the panel ends in 2023-01:** India industrial production on FRED (OECD MEI) stops there. It is the earliest-ending series. See decision D8 in `decisions.md`.
- **India M3:** the seasonally adjusted series `MABMM301INM189S` ends 2018-12, so the non-adjusted series is used instead (decision D9).
- **US M3:** the Federal Reserve stopped publishing M3 in 2006. The OECD-constructed series used here continues to 2023-11.
- **S&P 500:** FRED's `SP500` only covers about 10 years, so the OECD share price index is used for both countries for like-for-like definitions.
- **Not used:** `DEXINUS` (daily FX rate). `EXINUS` is already its monthly average.

## Test periods this data supports
| Purpose | Train | Test |
|---|---|---|
| Replication of the base paper | 1994-04 to 2016-07 | 2016-08 to 2018-12 (29 months) |
| Extension (walk-forward) | expanding from 1994-04 | 2010-01 to 2023-01, covering the 2013 taper tantrum and COVID |

## Units warning for feature engineering
Raw levels are **not comparable across countries**. For example, M3 is in USD for the US and INR for India, and the index base years differ. The paper's "USA − India" level differences are therefore not meaningful for money supply, trade or reserves. The feature step (M3) uses growth-rate differentials (Δlog) and rate differentials instead. See decision D3.
