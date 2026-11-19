"""Synthetic daily price frames for tests. Every value is made up."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")

LAST_SESSION = "2030-03-14"
"""A Thursday. The default synthetic history ends on this session."""


def at(text: str) -> datetime:
    """A New York wall-clock time, e.g. at("2030-03-15 11:00")."""
    return datetime.fromisoformat(text).replace(tzinfo=NEW_YORK)


def make_prices(rows: int = 300, *, end: str = LAST_SESSION, seed: int = 7) -> pd.DataFrame:
    """A valid frame in the project's shape: business days ending on `end`."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=rows, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, rows)))
    open_ = close * (1 + rng.normal(0, 0.005, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, rows))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close * 0.98,
            "volume": rng.integers(1_000_000, 5_000_000, rows),
        },
        index=index,
    )


def as_yahoo(prices: pd.DataFrame) -> pd.DataFrame:
    """The same data shaped like a yfinance download: capitalised, New York midnight."""
    raw = prices.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    raw.index = pd.DatetimeIndex(prices.index).tz_localize(NEW_YORK).rename("Date")
    return raw


def with_extra_row(prices: pd.DataFrame, session: str, **values: float) -> pd.DataFrame:
    """Append one row for `session`, copying the last row except for `values`."""
    row = prices.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([session], name="date")
    for column, value in values.items():
        row[column] = value
    return pd.concat([prices, row])
