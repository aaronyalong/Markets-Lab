"""Volatility forecasters, and how a forecast becomes a range.

A forecaster says how volatile the next session will be: one number per session,
the standard deviation it expects of the next log return. That number is turned
into a range the same way for every forecaster, fixed in advance:

- On the training rows, each return is divided by the volatility that had been
  forecast for it. The 10th and 90th percentiles of those ratios are kept.
- The range for a new session is its forecast volatility times those two numbers.

So the range is meant to hold the next return 80% of the time, with 10% of outcomes
below it and 10% above. Nothing assumes returns follow a bell curve, the range need
not be centred on zero, and only past sessions decide its shape.

The baselines here need no learning beyond that:

- `constant`: volatility has always been its average over the training rows. Its
  range is simply the middle 80% of all past returns.
- `rolling_21`: the standard deviation of the last 21 daily returns.
- `ewma_94`: an exponentially weighted average of squared returns (see `dataset`).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.volatility.dataset import EWMA_COLUMN

VOLATILITY = "volatility"
LOWER = "lower"
UPPER = "upper"
FORECAST_COLUMNS = (VOLATILITY, LOWER, UPPER)

RANGE_LEVEL = 0.8
"""The share of outcomes the range is meant to hold. Approved 2026-11-20."""

MIN_TRAINING_SESSIONS = 100
"""Fewer rows than this cannot place a 10th and 90th percentile with any care."""


class RangeForecaster(Protocol):
    """Something that forecasts next-session volatility and a range for the return."""

    @property
    def name(self) -> str:
        """Short identifier used in reports and in the prediction log."""
        ...

    def fit(self, features: pd.DataFrame, next_log_return: pd.Series) -> None:
        """Learn from sessions whose next return is already known."""
        ...

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """One forecast per row, on the same index, with `FORECAST_COLUMNS`.

        `volatility` is the forecast standard deviation of the next log return.
        `lower` and `upper` bound the range for that return, as log returns.
        """
        ...


def forecast_frame(
    index: pd.Index, volatility: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> pd.DataFrame:
    """Build the frame `RangeForecaster.predict` returns, checking it makes sense."""
    columns = {VOLATILITY: volatility, LOWER: lower, UPPER: upper}
    for name, values in columns.items():
        array = np.asarray(values, dtype="float64")
        if array.shape != (len(index),) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be one finite number per row")
        columns[name] = array
    if (columns[VOLATILITY] <= 0).any():
        raise ValueError("volatility must be positive")
    if (columns[LOWER] >= columns[UPPER]).any():
        raise ValueError("the lower bound must be below the upper bound")
    return pd.DataFrame(columns, index=index)[list(FORECAST_COLUMNS)]


class ScaledRangeForecaster:
    """Shared machinery: a volatility forecast scaled by percentiles of past ratios.

    A subclass says what volatility it forecasts for each row (`_volatility`) and,
    if it has anything to learn first, learns it in `_learn`.
    """

    name = ""

    def __init__(self) -> None:
        self._low: float | None = None
        self._high: float | None = None
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, next_log_return: pd.Series) -> None:
        outcomes = next_log_return.to_numpy(dtype="float64")
        if len(outcomes) != len(features):
            raise ValueError("features and next_log_return must have the same number of rows")
        if not np.isfinite(outcomes).all():
            raise ValueError("next_log_return has missing or infinite values")
        if len(outcomes) < MIN_TRAINING_SESSIONS:
            raise ValueError(
                f"a range forecaster needs at least {MIN_TRAINING_SESSIONS} training "
                f"sessions, got {len(outcomes)}"
            )
        self._columns = list(features.columns)
        self._learn(features, outcomes)
        ratios = outcomes / self._checked_volatility(features)
        tail = (1 - RANGE_LEVEL) / 2
        low, high = np.quantile(ratios, [tail, 1 - tail])
        if not low < high:
            raise ValueError("training returns do not vary; cannot form a range")
        self._low, self._high = float(low), float(high)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._low is None or self._high is None:
            raise RuntimeError("fit must be called before predict")
        if list(features.columns) != self._columns:
            raise ValueError("features do not have the columns the forecaster was fitted on")
        volatility = self._checked_volatility(features)
        return forecast_frame(
            features.index, volatility, self._low * volatility, self._high * volatility
        )

    def _learn(self, features: pd.DataFrame, outcomes: np.ndarray) -> None:
        """Fit anything the volatility forecast itself needs. Nothing by default."""

    def _volatility(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _checked_volatility(self, features: pd.DataFrame) -> np.ndarray:
        volatility = np.asarray(self._volatility(features), dtype="float64")
        if volatility.shape != (len(features),):
            raise ValueError("a forecaster must give one volatility per row")
        if not np.isfinite(volatility).all() or (volatility <= 0).any():
            raise ValueError("forecast volatility must be finite and positive")
        return volatility


class ConstantVolatility(ScaledRangeForecaster):
    """Volatility is its average over the training rows, the same every day."""

    name = "constant"

    def __init__(self) -> None:
        super().__init__()
        self._level = float("nan")

    def _learn(self, features: pd.DataFrame, outcomes: np.ndarray) -> None:
        self._level = float(np.sqrt(np.mean(outcomes**2)))

    def _volatility(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self._level)


class RollingVolatility(ScaledRangeForecaster):
    """Volatility is the standard deviation of the last 21 daily returns."""

    name = "rolling_21"
    column = "vol_21"

    def _volatility(self, features: pd.DataFrame) -> np.ndarray:
        return _column(features, self.column)


class EwmaVolatility(ScaledRangeForecaster):
    """Volatility is the exponentially weighted average of squared daily returns."""

    name = "ewma_94"

    def _volatility(self, features: pd.DataFrame) -> np.ndarray:
        return _column(features, EWMA_COLUMN)


def _column(features: pd.DataFrame, name: str) -> np.ndarray:
    if name not in features.columns:
        raise ValueError(f"features have no column {name!r}")
    return features[name].to_numpy(dtype="float64")
