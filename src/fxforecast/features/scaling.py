"""Scale features using statistics from the training rows only.

The paper's min-max scaling appears to use the full sample, which leaks the
test period's range into training. Here the scaler is always fit on train
and only applied to test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def scale_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard"
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | MinMaxScaler]:
    """Fit the scaler on X_train, transform both, keep index and column names."""
    if method not in SCALERS:
        raise ValueError(f"Unknown scaling method {method!r}; expected one of {list(SCALERS)}")
    scaler = SCALERS[method]().fit(X_train.to_numpy())
    wrap = lambda X: pd.DataFrame(scaler.transform(X.to_numpy()), index=X.index, columns=X.columns)  # noqa: E731
    return wrap(X_train), wrap(X_test), scaler
