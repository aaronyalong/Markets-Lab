"""Technical features and the next-day label for the NVDA experiment.

Timing convention, used everywhere in this experiment:

- Row *t* of the features describes what is known after session *t* has closed. It
  is computed from sessions up to and including *t* and nothing later.
- The label on row *t* is about session *t+1*: "up" (1) if its adjusted close is
  above session *t*'s adjusted close, otherwise 0. A tie is not up.
- The last row has features but no label yet. It is the input for the next call.

Features and labels are computed by separate functions so that the label can never
be mixed into the features by accident. The tests check the first rule directly:
features for a session must be identical whether or not later sessions exist.

Every feature is a ratio or a return, so none depends on the price level. That
matters because Yahoo rescales past prices after splits and dividends; a
level-free feature computed today is the same when recomputed after a rescale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

RETURN_LAGS = (1, 2, 3, 5, 10, 21)
VOLATILITY_WINDOWS = (5, 21)
SMA_WINDOWS = (10, 50, 200)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_WINDOW = 21
WEEKDAY_NAMES = ("tue", "wed", "thu", "fri")
"""One column per weekday except Monday, which is the all-zeros case."""

FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"ret_{lag}" for lag in RETURN_LAGS),
    *(f"vol_{window}" for window in VOLATILITY_WINDOWS),
    f"rsi_{RSI_PERIOD}",
    "macd",
    "macd_hist",
    *(f"sma_{window}_gap" for window in SMA_WINDOWS),
    f"volume_ratio_{VOLUME_WINDOW}",
    "range_pct",
    *(f"dow_{name}" for name in WEEKDAY_NAMES),
)

REQUIRED_COLUMNS = ("high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True, slots=True)
class Dataset:
    """Aligned features and outcomes, one row per session with complete features.

    `target_up` is 1.0 or 0.0, and `next_return` is the next session's simple
    return; both are NaN on the final row, whose next session has not happened.
    """

    features: pd.DataFrame
    target_up: pd.Series
    next_return: pd.Series

    def __len__(self) -> int:
        return len(self.features)

    def labelled(self) -> Dataset:
        """Only the rows whose next session is known."""
        known = self.target_up.notna()
        return Dataset(self.features[known], self.target_up[known], self.next_return[known])

    def latest(self) -> tuple[date, pd.Series]:
        """The last session and its features: the input for the next call."""
        last = pd.Timestamp(self.features.index[-1])
        return last.date(), self.features.iloc[-1]


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features for every session, using only that session and earlier ones.

    Returns a frame on the same index with the columns in `FEATURE_COLUMNS`. Early
    rows are NaN where a feature's window is not yet full (the longest needs 200
    sessions).
    """
    _check_input(prices)
    adj = prices["adj_close"].astype("float64")
    log_adj = pd.Series(np.log(adj.to_numpy()), index=prices.index)
    daily = log_adj.diff()

    columns: dict[str, pd.Series] = {}
    for lag in RETURN_LAGS:
        columns[f"ret_{lag}"] = log_adj.diff(lag)
    for window in VOLATILITY_WINDOWS:
        columns[f"vol_{window}"] = daily.rolling(window).std()
    columns[f"rsi_{RSI_PERIOD}"] = _rsi(adj, RSI_PERIOD)

    fast = adj.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    slow = adj.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd = fast - slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    columns["macd"] = macd / adj
    columns["macd_hist"] = (macd - signal) / adj

    for window in SMA_WINDOWS:
        columns[f"sma_{window}_gap"] = adj / adj.rolling(window).mean() - 1.0

    volume = prices["volume"].astype("float64")
    ratio = volume / volume.rolling(VOLUME_WINDOW).mean()
    columns[f"volume_ratio_{VOLUME_WINDOW}"] = pd.Series(
        np.log(ratio.to_numpy()), index=prices.index
    )
    columns["range_pct"] = (prices["high"] - prices["low"]) / prices["close"]

    weekday = pd.DatetimeIndex(prices.index).day_of_week
    for number, name in enumerate(WEEKDAY_NAMES, start=1):
        columns[f"dow_{name}"] = pd.Series(
            (weekday == number).astype("float64"), index=prices.index
        )

    return pd.DataFrame(columns, index=prices.index)[list(FEATURE_COLUMNS)]


def compute_outcomes(prices: pd.DataFrame) -> pd.DataFrame:
    """The next session's return and direction for every session.

    Columns: `next_return` (simple return of the adjusted close from this session
    to the next) and `target_up` (1.0 if that return is positive, else 0.0). Both
    are NaN on the last row. This is the only place the future is looked at.
    """
    _check_input(prices)
    adj = prices["adj_close"].astype("float64")
    next_return = adj.shift(-1) / adj - 1.0
    target_up = (next_return > 0).astype("float64").where(next_return.notna())
    return pd.DataFrame({"next_return": next_return, "target_up": target_up}, index=prices.index)


def build_dataset(prices: pd.DataFrame) -> Dataset:
    """Features and outcomes for every session that has complete features.

    The warm-up sessions at the start are left out. Incomplete features anywhere
    after the warm-up mean the input has a problem, and raise.
    """
    features = compute_features(prices)
    outcomes = compute_outcomes(prices)
    complete = features.notna().all(axis=1)
    if not bool(complete.any()):
        raise ValueError(f"not enough history for features: {len(prices)} sessions")
    start = int(complete.to_numpy().argmax())
    if not bool(complete.iloc[start:].all()):
        bad = complete.iloc[start:]
        first_bad = pd.Timestamp(bad[~bad].index[0]).date()
        raise ValueError(f"features are incomplete after the warm-up, first on {first_bad}")
    if not bool(np.isfinite(features.iloc[start:].to_numpy()).all()):
        raise ValueError("features contain infinite values")
    return Dataset(
        features=features.iloc[start:],
        target_up=outcomes["target_up"].iloc[start:],
        next_return=outcomes["next_return"].iloc[start:],
    )


def _rsi(adj: pd.Series, period: int) -> pd.Series:
    """Wilder's relative strength index, scaled to 0-1. A flat stretch gives 0.5."""
    change = adj.diff()
    gain = change.clip(lower=0.0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    total = gain + loss
    rsi = (gain / total).where(total > 0, 0.5)
    return rsi.where(total.notna())


def _check_input(prices: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in prices.columns]
    if missing:
        raise ValueError(f"prices are missing columns {missing}")
    index = prices.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by session date")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("prices must have unique dates in increasing order")
