"""Synthetic index closes and results filings for tests. Every value is made up."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd

from markets_lab.collectors.market_context import SERIES
from markets_lab.collectors.nvda_earnings import ResultsPublication
from price_frames import NEW_YORK, as_yahoo, make_prices

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


def context_for(prices: pd.DataFrame) -> pd.DataFrame:
    """Made-up index closes on exactly the sessions of `prices`."""
    closes = make_context(len(prices))
    closes.index = prices.index
    return closes


def yahoo_downloader(
    prices: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    failing: Sequence[str] = (),
) -> Callable[[str], pd.DataFrame]:
    """A stand-in for the Yahoo download: NVDA prices, and one frame per index.

    `context` defaults to closes on the same sessions as `prices`. Asking for a
    symbol in `failing` raises, like a download that did not work.
    """
    closes = context_for(prices) if context is None else context
    by_symbol = {spec.symbol: name for name, spec in SERIES.items()}

    def download(symbol: str) -> pd.DataFrame:
        if symbol in failing:
            raise OSError(f"no answer for {symbol}")
        if symbol not in by_symbol:
            return as_yahoo(prices)
        raw = pd.DataFrame({"Close": closes[by_symbol[symbol]].to_numpy()})
        raw.index = pd.DatetimeIndex(closes.index).tz_localize(NEW_YORK).rename("Date")
        return raw

    return download


def sec_getter(
    publications: Sequence[ResultsPublication] | None = None, *, failing: bool = False
) -> Callable[[str], str]:
    """A stand-in for SEC: one page listing these results filings, newest first."""
    chosen = make_publications() if publications is None else list(publications)
    page = {
        "form": ["8-K"] * len(chosen),
        "accessionNumber": [item.accession for item in reversed(chosen)],
        "acceptanceDateTime": [
            item.accepted.strftime("%Y-%m-%dT%H:%M:%S.000Z") for item in reversed(chosen)
        ],
        "items": [",".join(item.items) for item in reversed(chosen)],
    }
    text = json.dumps({"filings": {"recent": page, "files": []}})

    def get(_url: str) -> str:
        if failing:
            raise OSError("no answer from SEC")
        return text

    return get
