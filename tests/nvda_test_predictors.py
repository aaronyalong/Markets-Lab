"""Predictors used only by tests: one that really learns, and two that watch for leaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.predictors import prediction_frame


class LeastSquaresLearner:
    """A tiny linear model fitted by least squares. Any change in its training data
    changes its calls, which is what the leakage tests need."""

    name = "least_squares"

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        design = np.column_stack([np.ones(len(features)), features.to_numpy(dtype="float64")])
        self._weights = np.linalg.lstsq(design, target_up.to_numpy(dtype="float64"), rcond=None)[0]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self._weights is not None, "fit must be called before predict"
        design = np.column_stack([np.ones(len(features)), features.to_numpy(dtype="float64")])
        score = np.clip(design @ self._weights, 0.0, 1.0)
        return prediction_frame(features.index, score > 0.5, score)


class Spy:
    """Records exactly what each fold showed it. One shared log across instances."""

    name = "spy"

    def __init__(self, log: list[dict[str, object]]) -> None:
        self._log = log
        self._entry: dict[str, object] = {}

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        self._entry = {
            "train_sessions": features.index,
            "train_rows": len(features),
            "train_targets_missing": int(target_up.isna().sum()),
            "train_index_matches_target": features.index.equals(target_up.index),
        }

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        self._log.append({**self._entry, "test_sessions": features.index})
        return prediction_frame(features.index, np.ones(len(features), dtype=bool))


class Memorizer:
    """Cheats if it can: remembers every (session, outcome) it is trained on and
    replays the outcome when asked about a session it has already seen.

    Under a correct walk-forward it is never asked about a remembered session, so
    `recalled` stays at zero. If training ever includes a tested session, it scores
    perfectly there and `recalled` counts it.
    """

    name = "memorizer"
    recalled = 0

    def __init__(self) -> None:
        self._memory: dict[pd.Timestamp, bool] = {}

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        outcomes = (target_up.to_numpy(dtype="float64") == 1.0).tolist()
        self._memory = dict(zip(pd.DatetimeIndex(target_up.index), outcomes, strict=True))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        calls = []
        for session in features.index:
            known = self._memory.get(pd.Timestamp(session))
            if known is not None:
                type(self).recalled += 1
            calls.append(bool(known))
        return prediction_frame(features.index, np.array(calls, dtype=bool))
