"""Forecasting models. Each one predicts y = log FX(t+h) - log FX(t) with the same interface:

    model.fit(X_train: DataFrame, y_train: Series) -> model
    model.predict(X_test: DataFrame) -> np.ndarray

Models choose the columns they need from X, so every model can receive the
same rows and be compared on identical forecast origins.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
