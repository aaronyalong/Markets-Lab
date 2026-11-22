"""Volatility models: forecasters that learn more than the two range percentiles.

`har` is the standard simple model for volatility. Volatility has long memory, and
the usual way to capture it is to describe the recent past at three speeds: the
last session, the last week and the last month. Here those are

- the session's high-to-low range as a share of its close (`range_pct`),
- the standard deviation of the last 5 daily returns (`vol_5`), and
- the standard deviation of the last 21 daily returns (`vol_21`).

The model is log-linear: the log of the next session's variance is a constant plus
a weight on the log of each of the three. The four numbers are fitted by minimising
QLIKE, the same loss the forecasts are scored with (this is a gamma regression with
a log link). Being log-linear, it can never forecast a negative variance.

Choices fixed in advance, not tuned on the results: these three inputs, no
regularisation, and the shared rule that turns a volatility into a range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor

from markets_lab.analysis.nvda.volatility.forecasters import ScaledRangeForecaster

HAR_INPUTS = ("range_pct", "vol_5", "vol_21")

SQUARED_RETURN_FLOOR = 1e-12
"""The gamma fit needs a positive target, and a flat session has a squared return
of zero. Such sessions are given this value. It only shifts the loss by a constant
and does not move the fitted weights."""


class HarVolatility(ScaledRangeForecaster):
    """Log-linear regression of next-session variance on volatility at three speeds."""

    name = "har"

    def __init__(self) -> None:
        super().__init__()
        self._model: GammaRegressor | None = None

    def _learn(self, features: pd.DataFrame, outcomes: np.ndarray) -> None:
        target = np.maximum(outcomes**2, SQUARED_RETURN_FLOOR)
        model = GammaRegressor(alpha=0.0, max_iter=1_000, tol=1e-8)
        model.fit(_log_inputs(features), target)
        self._model = model

    def _volatility(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        if len(features) == 0:
            return np.zeros(0)
        return np.sqrt(self._model.predict(_log_inputs(features)))

    def coefficients(self) -> pd.Series:
        """The constant and the weight on the log of each input, for the method notes."""
        if self._model is None:
            raise RuntimeError("fit must be called before reading coefficients")
        values = [float(self._model.intercept_), *self._model.coef_.tolist()]
        return pd.Series(values, index=["constant", *HAR_INPUTS], name="coefficient")


def _log_inputs(features: pd.DataFrame) -> np.ndarray:
    missing = [name for name in HAR_INPUTS if name not in features.columns]
    if missing:
        raise ValueError(f"features have no columns {missing}")
    values = features[list(HAR_INPUTS)].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("the volatility inputs must be finite and positive")
    return np.log(values)
