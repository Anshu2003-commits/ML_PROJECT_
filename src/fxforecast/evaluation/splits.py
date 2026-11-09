"""Time-ordered train/test splits on forecast origins.

At a refit date T the model may only learn from rows whose target is already
observed at T, i.e. origins t with t + h <= T. It then forecasts from origins
T, T+1, ... until the next refit. This embargo of h months is what stops
overlapping multi-step targets from leaking test information into training.

Two schemes:
    replication_folds   one fit at the paper's train_end, one test block (as in the paper)
    walk_forward_folds  expanding window, refit every k months (main evaluation)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    refit_date: pd.Timestamp     # last month of data the model is allowed to see
    train: pd.DatetimeIndex      # origins used for fitting (targets known by refit_date)
    test: pd.DatetimeIndex       # origins forecast with this fitted model


def _months(n: int) -> pd.DateOffset:
    return pd.DateOffset(months=n)


def make_folds(
    origins: pd.DatetimeIndex,
    refit_dates: list[pd.Timestamp],
    horizon: int,
    last_test_origin: pd.Timestamp | None = None,
) -> list[Fold]:
    """Build one fold per refit date; each test block runs until the next refit date."""
    origins = pd.DatetimeIndex(origins).sort_values()
    refit_dates = sorted(pd.Timestamp(d) for d in refit_dates)
    folds = []
    for i, T in enumerate(refit_dates):
        train = origins[origins <= T - _months(horizon)]
        in_block = origins >= T
        if i + 1 < len(refit_dates):
            in_block &= origins < refit_dates[i + 1]
        if last_test_origin is not None:
            in_block &= origins <= pd.Timestamp(last_test_origin)
        test = origins[in_block]
        if len(test) == 0:
            continue
        if len(train) == 0:
            raise ValueError(f"No training rows before refit date {T:%Y-%m} at horizon {horizon}")
        folds.append(Fold(T, train, test))
    if not folds:
        raise ValueError("No folds with test rows; check the dates against the data range")
    return folds


def replication_folds(
    origins: pd.DatetimeIndex, train_end: pd.Timestamp, test_end: pd.Timestamp, horizon: int
) -> list[Fold]:
    """The paper's single split: fit with data to train_end, forecast targets up to test_end."""
    return make_folds(origins, [pd.Timestamp(train_end)], horizon,
                      last_test_origin=pd.Timestamp(test_end) - _months(horizon))


def walk_forward_folds(
    origins: pd.DatetimeIndex, first_origin: pd.Timestamp, refit_every: int, horizon: int
) -> list[Fold]:
    """Expanding window: refit at first_origin and every ``refit_every`` months after."""
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    refits = list(pd.date_range(pd.Timestamp(first_origin), origins.max(), freq=_months(refit_every)))
    return make_folds(origins, refits, horizon)
