"""Turn the raw panel into unit-free model inputs.

Raw levels are not comparable across countries (USD vs INR money supply,
different index base years), so each macro variable becomes a US - India
differential of something unit-free: a growth rate, an interest rate or a
trade-balance ratio. Each choice follows a standard exchange-rate theory:

    infl_diff      PPP: relative inflation                 (12-month log change)
    ip_growth_diff monetary model: relative output growth  (12-month log change)
    m3_growth_diff monetary model: relative money growth   (12-month log change)
    rate_diff      uncovered interest parity               (percentage points)
    reserves_growth_diff  central-bank intervention capacity (12-month log change)
    equity_ret_diff       portfolio flows                  (monthly log return)
    tb_diff        trade balance: (X - M) / (X + M) per country

12-month changes are used for variables that are not seasonally adjusted for
India (CPI, M3). They are a fixed formula with nothing fitted, so they cannot
leak information from a test period. All transforms here are deterministic and
use only current and past values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "dlog_fx"
MACRO_FEATURES = (
    "infl_diff",
    "ip_growth_diff",
    "rate_diff",
    "m3_growth_diff",
    "reserves_growth_diff",
    "equity_ret_diff",
    "tb_diff",
)


def log_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    """log(x_t) - log(x_{t-periods}); approximately the growth rate over that span."""
    if (s <= 0).any():
        raise ValueError(f"{s.name}: log_diff needs strictly positive values")
    return np.log(s).diff(periods)


def trade_balance_ratio(exports: pd.Series, imports: pd.Series) -> pd.Series:
    """(X - M) / (X + M): unit-free, bounded in [-1, 1]."""
    return (exports - imports) / (exports + imports)


def macro_differentials(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the target and the US - India macro differentials from the raw panel.

    The first 12 rows are dropped because 12-month changes need a year of history.
    """
    yoy = lambda col: log_diff(panel[col], 12)  # noqa: E731
    out = pd.DataFrame(index=panel.index)
    out[TARGET] = log_diff(panel["fx"], 1)
    out["infl_diff"] = yoy("cpi_us") - yoy("cpi_in")
    out["ip_growth_diff"] = yoy("ip_us") - yoy("ip_in")
    out["rate_diff"] = panel["rate_us"] - panel["rate_in"]
    out["m3_growth_diff"] = yoy("m3_us") - yoy("m3_in")
    out["reserves_growth_diff"] = yoy("reserves_us") - yoy("reserves_in")
    out["equity_ret_diff"] = log_diff(panel["equity_us"]) - log_diff(panel["equity_in"])
    out["tb_diff"] = (
        trade_balance_ratio(panel["exports_us"], panel["imports_us"])
        - trade_balance_ratio(panel["exports_in"], panel["imports_in"])
    )
    return out.iloc[12:]


def difference(df: pd.DataFrame, orders: dict[str, int]) -> pd.DataFrame:
    """Difference each column d times (d from ``orders``; default 0), then drop leading NaNs.

    The orders should come from stationarity tests on the training period only.
    """
    out = df.copy()
    for col, d in orders.items():
        for _ in range(d):
            out[col] = out[col].diff()
    return out.dropna()


def relative_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """Log-level counterparts used for cointegration tests (monetary model of FX).

    log FX, log relative prices, log relative output, log relative money and the
    interest differential. Constants from different units or base years only
    shift each series, which the tests absorb.
    """
    lr = lambda a, b: np.log(panel[a]) - np.log(panel[b])  # noqa: E731
    return pd.DataFrame(
        {
            "log_fx": np.log(panel["fx"]),
            "rel_price": lr("cpi_us", "cpi_in"),
            "rel_output": lr("ip_us", "ip_in"),
            "rel_money": lr("m3_us", "m3_in"),
            "rate_diff": panel["rate_us"] - panel["rate_in"],
        },
        index=panel.index,
    )
