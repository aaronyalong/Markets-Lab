"""The interface every baseline and model implements.

A predictor is fitted on past sessions and then asked, for each session's features,
whether the next session will close up. Row *t* of the features is what is known
after session *t* closes, so a prediction for row *t* is a call on session *t+1*.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

PREDICTED_UP = "predicted_up"
PROBABILITY_UP = "probability_up"
PREDICTION_COLUMNS = (PREDICTED_UP, PROBABILITY_UP)


class Predictor(Protocol):
    """Something that can be trained on the past and make next-session calls."""

    @property
    def name(self) -> str:
        """Short identifier used in reports and in the prediction log."""
        ...

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        """Learn from sessions whose next-session outcome is already known."""
        ...

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """One call per row, on the same index, with the columns in `PREDICTION_COLUMNS`.

        `predicted_up` is a boolean. `probability_up` is the predictor's probability
        that the next session closes up, or NaN for a rule that has no probability.
        """
        ...


def prediction_frame(
    index: pd.Index, predicted_up: np.ndarray, probability_up: np.ndarray | None = None
) -> pd.DataFrame:
    """Build the frame `Predictor.predict` returns, checking its shape and types."""
    up = np.asarray(predicted_up)
    if up.dtype != np.bool_ or up.shape != (len(index),):
        raise ValueError("predicted_up must be one boolean per row")
    if probability_up is None:
        probability = np.full(len(index), np.nan)
    else:
        probability = np.asarray(probability_up, dtype="float64")
        if probability.shape != (len(index),):
            raise ValueError("probability_up must be one number per row")
        known = probability[~np.isnan(probability)]
        if ((known < 0) | (known > 1)).any():
            raise ValueError("probability_up must be between 0 and 1")
    return pd.DataFrame({PREDICTED_UP: up, PROBABILITY_UP: probability}, index=index)
