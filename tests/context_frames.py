"""Synthetic index closes and results filings for tests. Every value is made up."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from markets_lab.collectors.market_context import SERIES
from markets_lab.collectors.nvda_earnings import ResultsPublication
from price_frames import make_prices

FIRST_FILING = datetime(2003, 8, 7, 20, 30, tzinfo=UTC)


def make_context(rows: int = 300) -> pd.DataFrame:
    """Made-up index closes on the same sessions as `make_prices(rows)`."""
    return pd.DataFrame(
        {name: make_prices(rows, seed=100 + number)["close"] for number, name in enumerate(SERIES)}
    )


def make_publications(count: int = 107) -> list[ResultsPublication]:
    """Made-up quarterly results filings, 91 days apart from August 2003. Each is
    accepted at 16:30 or 15:30 New York time; the default list ends in January 2030."""
    return [
        ResultsPublication(
            FIRST_FILING + timedelta(days=91 * number),
            f"0000000000-{(3 + number // 4) % 100:02d}-{number:06d}",
            ("12",) if number < 4 else ("2.02", "9.01"),
        )
        for number in range(count)
    ]
