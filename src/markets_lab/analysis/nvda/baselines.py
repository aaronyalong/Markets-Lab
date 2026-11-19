"""Naive baselines that every model has to beat.

None of these learns anything: `fit` does nothing, and each call uses only the
features of the session it is made on. They give a direction and no probability.

- Always up: the next session closes up. Its accuracy is the share of up days.
- Persistence: the next session moves the same way the last one did.
- Moving-average rule: up if the price is above its 200-session average. The
  window is the conventional one and was fixed before any results were looked at;
  it is not tuned.

Buy-and-hold is not a separate predictor. Holding the stock every day is exactly
what following "always up" means, so the always-up baseline is the accuracy
benchmark and its simulated return is the buy-and-hold return benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.features import SMA_WINDOWS
from markets_lab.analysis.nvda.predictors import Predictor, prediction_frame

MOVING_AVERAGE_WINDOW = 200


class AlwaysUp:
    """Calls up every session. Following it is buy-and-hold."""

    name = "always_up"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        del features, target_up

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return prediction_frame(features.index, np.ones(len(features), dtype=bool))


class Persistence:
    """Calls up if the session just finished closed up, otherwise down.

    A flat session counts as not up, the same way the label treats a tie.
    """

    name = "persistence"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        del features, target_up

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return prediction_frame(features.index, _column(features, "ret_1") > 0)


class MovingAverageRule:
    """Calls up if the price is above its moving average, otherwise down."""

    def __init__(self, window: int = MOVING_AVERAGE_WINDOW) -> None:
        if window not in SMA_WINDOWS:
            raise ValueError(f"window must be one of {SMA_WINDOWS}, not {window}")
        self._column = f"sma_{window}_gap"
        self.name = f"ma_{window}"

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        del features, target_up

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return prediction_frame(features.index, _column(features, self._column) > 0)


def default_baselines() -> tuple[Predictor, ...]:
    """The baselines every report includes, in a fixed order."""
    return (AlwaysUp(), Persistence(), MovingAverageRule())


def _column(features: pd.DataFrame, name: str) -> np.ndarray:
    if name not in features.columns:
        raise ValueError(f"features are missing the column {name!r}")
    values = features[name].to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError(f"feature {name!r} has missing values")
    return values
