"""Gradient-boosted trees (XGBoost) on the technical features.

Unlike logistic regression, trees can pick up thresholds and interactions between
features, for example "a drop matters only when volatility is high".

Settings fixed in advance, not tuned on the results. They are deliberately small
and cautious, because daily direction is mostly noise and a large model would
memorise it:

- 200 trees of depth 3, learning rate 0.05.
- Each tree sees a random 80% of the rows and 80% of the features.
- A fixed random seed and a single thread, so the same training data always gives
  the same model.
- The call is "up" when the probability is above 0.5.

Trees do not need the features to be scaled, so nothing is fitted besides the trees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from markets_lab.analysis.nvda.models.common import (
    DECISION_THRESHOLD,
    prediction_matrix,
    training_arrays,
)
from markets_lab.analysis.nvda.predictors import prediction_frame

N_TREES = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.05
ROW_SAMPLE = 0.8
FEATURE_SAMPLE = 0.8
SEED = 0


class XGBoostModel:
    """A small gradient-boosted tree classifier with fixed settings."""

    name = "xgboost"

    def __init__(self) -> None:
        self._model: XGBClassifier | None = None
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        matrix, target = training_arrays(features, target_up)
        model = XGBClassifier(
            n_estimators=N_TREES,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=ROW_SAMPLE,
            colsample_bytree=FEATURE_SAMPLE,
            objective="binary:logistic",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        )
        model.fit(matrix, target)
        self._model, self._columns = model, list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        matrix = prediction_matrix(features, self._columns)
        if len(features) == 0:
            return prediction_frame(features.index, np.zeros(0, dtype=bool), np.zeros(0))
        probability = self._model.predict_proba(matrix)[:, 1].astype("float64")
        return prediction_frame(features.index, probability > DECISION_THRESHOLD, probability)

    def feature_importance(self) -> pd.Series:
        """Share of the total gain each feature contributed, for the method notes."""
        if self._model is None:
            raise RuntimeError("fit must be called before reading importances")
        return pd.Series(self._model.feature_importances_, index=self._columns, name="importance")
