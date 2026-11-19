"""Fetch and validate daily NVDA prices from Yahoo Finance (via yfinance).

Yahoo is an unofficial source, so nothing it returns is trusted until checked:

- The latest row is unreliable. While the market is open it is a partial bar, and
  for hours after the close it can come back with a missing close price. Both are
  removed here and reported in `FetchResult.dropped`; only completed sessions are
  returned. This is the first guard against look-ahead: a model must never see a
  bar for a session that has not finished.
- Everything else is validated (`validate`). Any problem raises `PriceDataError`
  listing what is wrong. Nothing is repaired, filled or guessed.

The data is for local use only. It is saved under `data/`, which is gitignored.

Run it from the repository root:

    uv run python -m markets_lab.collectors.nvda_prices
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SYMBOL = "NVDA"
MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_CSV_PATH = Path("data/nvda_daily.csv")

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
COLUMNS = (*PRICE_COLUMNS, "volume")
_YAHOO_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

SESSION_COMPLETE_AT = time(17, 0)
"""New York time from which today's bar may count as complete (close plus an hour)."""

MIN_ROWS = 6_000
"""Fewer rows than this means Yahoo returned a truncated history (NVDA has ~7,000)."""

MAX_SESSION_GAP_DAYS = 7
"""Longest real gap between sessions is 7 calendar days (after 2001-09-11)."""

MAX_ABS_DAILY_RETURN = 0.6
"""Largest real close-to-close move is 42%; more than 60% suggests a bad split."""

MAX_STALENESS_DAYS = 5
"""The last completed session may be at most this many calendar days old."""

_MAX_PROBLEMS_SHOWN = 10

Downloader = Callable[[str], pd.DataFrame]


class PriceDataError(ValueError):
    """Downloaded or stored price data failed a check."""


@dataclass(frozen=True, slots=True)
class DroppedBar:
    """A row that was removed because its session was not complete."""

    session: date
    reason: str


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Validated daily prices, plus any trailing rows that were not complete."""

    prices: pd.DataFrame
    dropped: tuple[DroppedBar, ...]

    @property
    def last_session(self) -> date:
        return self.prices.index[-1].date()


def download_raw(symbol: str = SYMBOL) -> pd.DataFrame:
    """Download the full daily history from Yahoo, unadjusted OHLC plus adjusted close.

    This is the only function here that touches the network.
    """
    import yfinance as yf  # imported here so tests never need it or the network

    return yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False, actions=False)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance frame into the project's shape, without changing any value.

    The result is indexed by session date (timezone-free, named "date") and has the
    columns in `COLUMNS`. Other columns are dropped. Rows keep their order.
    """
    missing = [name for name in _YAHOO_COLUMNS if name not in raw.columns]
    if missing:
        raise PriceDataError(f"missing columns from Yahoo: {missing}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise PriceDataError(f"expected a date index, got {type(raw.index).__name__}")
    index = raw.index
    if index.tz is not None:
        index = index.tz_convert(MARKET_TZ).tz_localize(None)
    if len(index) and not (index == index.normalize()).all():
        raise PriceDataError("timestamps are not at midnight New York time; not daily bars")
    frame = raw[list(_YAHOO_COLUMNS)].rename(columns=_YAHOO_COLUMNS)
    frame.index = pd.DatetimeIndex(index, name="date")
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    volume = pd.to_numeric(frame["volume"], errors="raise")
    if volume.isna().any() or (volume != volume.round()).any():
        raise PriceDataError("volume has missing or non-whole values")
    frame["volume"] = volume.astype("int64")
    return frame


def split_complete(
    prices: pd.DataFrame, *, now: datetime
) -> tuple[pd.DataFrame, tuple[DroppedBar, ...]]:
    """Remove trailing rows whose session is not complete, and say which and why.

    - "in progress": the row is for today (New York) and it is before
      `SESSION_COMPLETE_AT`.
    - "incomplete": the last remaining row has a missing price. Yahoo does this to
      the latest session for some hours after the close. Only one such row is
      tolerated; missing prices anywhere else are left for `validate` to reject.

    A row dated after today raises, because it cannot be real.
    """
    today, clock = _market_date_and_time(now)
    dropped: list[DroppedBar] = []
    if len(prices) and prices.index[-1].date() > today:
        raise PriceDataError(f"row dated {prices.index[-1].date()} is in the future")
    if len(prices) and prices.index[-1].date() == today and clock < SESSION_COMPLETE_AT:
        dropped.append(DroppedBar(today, "in progress"))
        prices = prices.iloc[:-1]
    if len(prices) and bool(prices.iloc[-1][list(PRICE_COLUMNS)].isna().any()):
        dropped.append(DroppedBar(prices.index[-1].date(), "incomplete"))
        prices = prices.iloc[:-1]
    return prices, tuple(dropped)


def validate(prices: pd.DataFrame, *, now: datetime, min_rows: int = MIN_ROWS) -> None:
    """Raise `PriceDataError` listing every check that completed-session data fails."""
    problems = _problems(prices, now=now, min_rows=min_rows)
    if problems:
        shown = problems[:_MAX_PROBLEMS_SHOWN]
        more = len(problems) - len(shown)
        suffix = f"; and {more} more" if more else ""
        raise PriceDataError("price data failed validation: " + "; ".join(shown) + suffix)


def fetch_daily(
    *,
    now: datetime,
    downloader: Downloader = download_raw,
    symbol: str = SYMBOL,
    min_rows: int = MIN_ROWS,
) -> FetchResult:
    """Download, keep completed sessions only, validate, and return the prices."""
    try:
        raw = downloader(symbol)
    except Exception as exc:
        raise PriceDataError(f"download of {symbol} failed: {exc}") from exc
    if raw.empty:
        raise PriceDataError(f"download of {symbol} returned no rows")
    prices, dropped = split_complete(normalize(raw), now=now)
    validate(prices, now=now, min_rows=min_rows)
    return FetchResult(prices=prices, dropped=dropped)


def check_history_kept(previous: pd.DataFrame, new: pd.DataFrame) -> None:
    """Raise if a session that was stored before is absent from the new download.

    Values are not compared: splits and dividends legitimately rewrite past prices.
    """
    lost = previous.index.difference(new.index)
    if len(lost):
        sample = ", ".join(str(ts.date()) for ts in lost[:5])
        raise PriceDataError(f"new download is missing {len(lost)} stored sessions, e.g. {sample}")


def save_csv(prices: pd.DataFrame, path: Path) -> None:
    """Write the prices to CSV, replacing any existing file in one step."""
    temporary = path.with_name(path.name + ".tmp")
    prices.to_csv(temporary, date_format="%Y-%m-%d")
    os.replace(temporary, path)


def load_csv(path: Path) -> pd.DataFrame:
    """Read prices written by `save_csv`. Call `validate` before trusting them."""
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise PriceDataError(f"{path}: missing columns {missing}")
    return frame[list(COLUMNS)]


def _market_date_and_time(now: datetime) -> tuple[date, time]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(MARKET_TZ)
    return local.date(), local.time()


def _problems(prices: pd.DataFrame, *, now: datetime, min_rows: int) -> list[str]:
    today, _ = _market_date_and_time(now)
    if list(prices.columns) != list(COLUMNS):
        return [f"columns are {list(prices.columns)}, expected {list(COLUMNS)}"]
    if len(prices) < min_rows:
        return [f"only {len(prices)} rows, expected at least {min_rows}"]
    index = prices.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        return ["index is not a timezone-free date index"]

    problems: list[str] = []

    def flag(mask: pd.Series, what: str) -> None:
        bad = mask[mask]
        if len(bad):
            first = pd.Timestamp(bad.index[0]).date()
            problems.append(f"{what} on {len(bad)} rows, first {first}")

    if index.has_duplicates:
        problems.append("duplicate dates")
    if not index.is_monotonic_increasing:
        problems.append("dates are not in increasing order")
    if problems:
        return problems  # the row-by-row checks below assume ordered, unique dates

    flag(pd.Series(index.day_of_week >= 5, index=index), "weekend date")
    flag(prices[list(COLUMNS)].isna().any(axis=1), "missing value")
    flag((prices[list(PRICE_COLUMNS)] <= 0).any(axis=1), "non-positive price")
    flag(prices["volume"] <= 0, "non-positive volume")
    flag(prices["high"] < prices[["open", "close", "low"]].max(axis=1), "high below open/close/low")
    flag(prices["low"] > prices[["open", "close", "high"]].min(axis=1), "low above open/close/high")
    flag(prices["adj_close"] > prices["close"] * (1 + 1e-9), "adjusted close above close")

    gap_days = pd.Series(index, index=index).diff().dt.days
    flag(gap_days > MAX_SESSION_GAP_DAYS, f"gap of more than {MAX_SESSION_GAP_DAYS} days")
    for column in ("close", "adj_close"):
        change = prices[column].pct_change().abs()
        flag(
            change > MAX_ABS_DAILY_RETURN,
            f"{column} moved more than {MAX_ABS_DAILY_RETURN:.0%} in a day",
        )

    last = index[-1].date()
    if last > today:
        problems.append(f"last row {last} is in the future")
    elif (today - last).days > MAX_STALENESS_DAYS:
        problems.append(
            f"stale: last completed session is {last}, {(today - last).days} days before {today}"
        )
    return problems


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    downloader: Downloader = download_raw,
    now: Callable[[], datetime] = lambda: datetime.now(tz=MARKET_TZ),
    report: Callable[[str], None] = _print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.collectors.nvda_prices",
        description="Download and validate daily NVDA prices into a local CSV.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_CSV_PATH, help=f"CSV file (default {DEFAULT_CSV_PATH})"
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    if not out.parent.is_dir():
        report(f"error: directory {out.parent} does not exist; run from the repository root")
        return 2
    try:
        result = fetch_daily(now=now(), downloader=downloader)
        if out.exists():
            check_history_kept(load_csv(out), result.prices)
    except PriceDataError as exc:
        report(f"error: {exc}")
        return 1
    save_csv(result.prices, out)
    first = result.prices.index[0].date()
    report(f"saved {len(result.prices)} sessions, {first} to {result.last_session}, to {out}")
    for bar in result.dropped:
        report(f"left out {bar.session}: {bar.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
