"""The same models, given the market context and results-date columns as well.

Each one is its technical twin with nothing changed except its inputs and its name:
the same settings, fixed in advance, and the same code. That makes the comparison
between a model and its twin a comparison of the inputs alone.

They have their own names because they are different predictors: in the prediction
log each starts its own record on the day it joins, and the twin keeps running.
"""

from __future__ import annotations

import pandas as pd

from markets_lab.analysis.nvda.context_features import EXTENDED_FEATURE_COLUMNS
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.models.lstm import LSTMModel
from markets_lab.analysis.nvda.models.xgb import XGBoostModel


def require_context_columns(features: pd.DataFrame) -> None:
    """Refuse to fit on anything but the full set of columns.

    Without this, a context model handed only the technical columns would quietly
    become a copy of its twin.
    """
    if tuple(features.columns) != EXTENDED_FEATURE_COLUMNS:
        raise ValueError("a context model needs the technical, context and results columns")


class LogisticContextModel(LogisticRegressionModel):
    """Logistic regression on the technical, market context and results columns."""

    name = "logistic_ctx"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        require_context_columns(features)
        super().fit(features, target_up)


class XGBoostContextModel(XGBoostModel):
    """Gradient-boosted trees on the technical, market context and results columns."""

    name = "xgboost_ctx"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        require_context_columns(features)
        super().fit(features, target_up)


class LSTMContextModel(LSTMModel):
    """The LSTM over the last 20 sessions of technical, context and results columns."""

    name = "lstm_ctx"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        require_context_columns(features)
        super().fit(features, target_up)
