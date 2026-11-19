"""Input checks shared by every model, so they all refuse the same bad data."""

from __future__ import annotations

import numpy as np
import pandas as pd

DECISION_THRESHOLD = 0.5
"""A model calls "up" when its probability is above this. Fixed, not tuned."""


def training_arrays(features: pd.DataFrame, target_up: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Validated training data as arrays: a float matrix and a 0/1 integer target."""
    target = target_up.to_numpy(dtype="float64")
    if len(target) != len(features):
        raise ValueError("features and target_up must have the same number of rows")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target_up must be 0 or 1 with no missing values")
    if len(np.unique(target)) < 2:
        raise ValueError("training data has only one outcome; cannot fit a classifier")
    matrix = features.to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ValueError("training features have missing or infinite values")
    return matrix, target.astype("int64")


def prediction_matrix(features: pd.DataFrame, fitted_columns: list[str]) -> np.ndarray:
    """Validated features for prediction, in the columns the model was fitted on."""
    if list(features.columns) != fitted_columns:
        raise ValueError("features do not have the columns the model was fitted on")
    matrix = features.to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ValueError("features have missing or infinite values")
    return matrix
