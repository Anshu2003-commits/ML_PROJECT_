"""Support Vector Regression (RBF kernel) with time-series hyperparameter search.

Pipeline, all fitted on the training window only:
    select the first n_lags lags -> standardise X -> SVR, with y standardised too
    (so epsilon is in standard deviations of the target, comparable across horizons).

Hyperparameters (n_lags, C, gamma, epsilon) are tuned by grid search with an
expanding TimeSeriesSplit *inside* the training rows. The validation blocks
always come after their training rows, with a gap of h-1 rows: the same
embargo the outer walk-forward uses, so overlapping h-month targets cannot
leak. The test fold is never seen during tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from fxforecast.features.supervised import select_lags


class LagSelector(BaseEstimator, TransformerMixin):
    """sklearn wrapper around select_lags so n_lags can be tuned in the grid."""

    def __init__(self, n_lags: int = 1):
        self.n_lags = n_lags

    def fit(self, X: pd.DataFrame, y=None) -> "LagSelector":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return select_lags(X, self.n_lags).to_numpy()


def inner_cv(horizon: int, n_splits: int) -> TimeSeriesSplit:
    """Expanding-window CV inside the training rows, with an (h-1)-row embargo."""
    return TimeSeriesSplit(n_splits=n_splits, gap=horizon - 1)


class SVRForecaster:
    name = "svr"

    def __init__(self, horizon: int, n_lags=(1, 3, 6), C=(1, 10, 100), gamma=(0.01,),
                 epsilon=(0.1,), cv_splits: int = 5, n_jobs: int = -1):
        self.horizon, self.cv_splits, self.n_jobs = horizon, cv_splits, n_jobs
        self.grid = {
            "regressor__select__n_lags": list(n_lags),
            "regressor__svr__C": list(C),
            "regressor__svr__gamma": list(gamma),
            "regressor__svr__epsilon": list(epsilon),
        }

    def _estimator(self) -> TransformedTargetRegressor:
        pipe = Pipeline([("select", LagSelector()), ("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVRForecaster":
        search = GridSearchCV(self._estimator(), self.grid, cv=inner_cv(self.horizon, self.cv_splits),
                              scoring="neg_mean_squared_error", n_jobs=self.n_jobs, refit=True)
        search.fit(X, y.to_numpy())
        self.model_, self.best_params_ = search.best_estimator_, search.best_params_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def info(self) -> dict:
        return {k.rsplit("__", 1)[1]: v for k, v in self.best_params_.items()}
