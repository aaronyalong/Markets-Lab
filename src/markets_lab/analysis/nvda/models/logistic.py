"""Logistic regression on the technical features.

The simplest real model: a weighted sum of the standardised features passed through
a logistic curve to give a probability that the next session closes up.

Choices fixed in advance, not tuned on the results:

- Features are standardised with the mean and spread of the training rows only.
  The same fitted scaling is then applied to the rows being predicted, so nothing
  about those rows (or about the batch they arrive in) affects the scaling.
- L2 regularisation at scikit-learn's default strength (C = 1).
- The call is "up" when the probability is above 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from markets_lab.analysis.nvda.models.common import (
    DECISION_THRESHOLD,
    prediction_matrix,
    training_arrays,
)
from markets_lab.analysis.nvda.predictors import prediction_frame

REGULARISATION_C = 1.0


class LogisticRegressionModel:
    """Standardise on the training rows, then fit an L2-regularised logistic regression."""

    name = "logistic"

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        matrix, target = training_arrays(features, target_up)
        scaler = StandardScaler().fit(matrix)
        model = LogisticRegression(C=REGULARISATION_C, max_iter=1_000)
        model.fit(scaler.transform(matrix), target)
        self._scaler, self._model, self._columns = scaler, model, list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or self._model is None:
            raise RuntimeError("fit must be called before predict")
        matrix = prediction_matrix(features, self._columns)
        if len(features) == 0:
            return prediction_frame(features.index, np.zeros(0, dtype=bool), np.zeros(0))
        up_column = int(np.flatnonzero(self._model.classes_ == 1)[0])
        probability = self._model.predict_proba(self._scaler.transform(matrix))[:, up_column]
        return prediction_frame(features.index, probability > DECISION_THRESHOLD, probability)

    def coefficients(self) -> pd.Series:
        """Fitted weight per standardised feature, for the method notes."""
        if self._model is None:
            raise RuntimeError("fit must be called before reading coefficients")
        return pd.Series(self._model.coef_[0], index=self._columns, name="coefficient")
