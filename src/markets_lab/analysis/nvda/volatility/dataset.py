"""The dataset for the volatility question: the same sessions, a different target.

The direction experiment asks which way the next session closes. This one asks how
large the move will be. It uses the same sessions and the same technical columns,
plus one column the simplest forecasts need, and its target is the next session's
log return (the direction label is not used).

Timing is the same as everywhere: row *t* is what is known after session *t* has
closed, and the target on row *t* is about session *t+1*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, build_dataset

EWMA_LAMBDA = 0.94
"""The weight on yesterday's variance. 0.94 is the RiskMetrics value for daily data,
used here because it is the standard one. It was not tuned."""

EWMA_COLUMN = "ewma_vol_94"
EWMA_MIN_SESSIONS = 21
VOLATILITY_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, EWMA_COLUMN)


@dataclass(frozen=True, slots=True)
class VolatilityDataset:
    """Features and the next session's log return, one row per session.

    `next_log_return` is NaN on the final row, whose next session has not happened.
    """

    features: pd.DataFrame
    next_log_return: pd.Series

    def __len__(self) -> int:
        return len(self.features)

    def labelled(self) -> VolatilityDataset:
        """Only the rows whose next session is known."""
        known = self.next_log_return.notna()
        return VolatilityDataset(self.features[known], self.next_log_return[known])

    def latest(self) -> tuple[date, pd.Series]:
        """The last session and its features: the input for the next forecast."""
        last = pd.Timestamp(self.features.index[-1])
        return last.date(), self.features.iloc[-1]


def compute_ewma_volatility(prices: pd.DataFrame) -> pd.Series:
    """Exponentially weighted daily volatility, from each session and earlier ones.

    The variance is updated once per session: `EWMA_LAMBDA` times the previous
    value plus the rest times the session's squared log return. The result is its
    square root. The first `EWMA_MIN_SESSIONS` values are NaN.
    """
    adj = prices["adj_close"].astype("float64")
    daily = pd.Series(np.log(adj.to_numpy()), index=prices.index).diff()
    variance = (
        (daily**2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=EWMA_MIN_SESSIONS).mean()
    )
    return pd.Series(np.sqrt(variance.to_numpy()), index=prices.index, name=EWMA_COLUMN)


def build_volatility_dataset(prices: pd.DataFrame) -> VolatilityDataset:
    """The sessions and technical columns of `build_dataset(prices)`, the extra
    column, and the next session's log return as the target."""
    base = build_dataset(prices)
    ewma = compute_ewma_volatility(prices).loc[base.features.index]
    if not bool(np.isfinite(ewma.to_numpy()).all()):
        raise ValueError("the exponentially weighted volatility is incomplete")
    features = pd.concat([base.features, ewma], axis=1)[list(VOLATILITY_COLUMNS)]
    next_log_return = pd.Series(
        np.log1p(base.next_return.to_numpy(dtype="float64")),
        index=base.next_return.index,
        name="next_log_return",
    )
    return VolatilityDataset(features=features, next_log_return=next_log_return)
