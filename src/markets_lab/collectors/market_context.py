"""Fetch and validate daily closes of four market indices (via yfinance).

These are context for the NVDA experiment: what the wider market, technology
stocks, semiconductor stocks and expected volatility did on each session. The
series were chosen because Yahoo has every one of NVDA's sessions for all four
(measured 2026-11-20); the matching funds (QQQ, SOXX, SMH) start too late.

The same rules apply as for NVDA prices, and nothing is trusted until checked:

- Only the closing level is kept. Index volume is mostly zero on Yahoo and is not
  used.
- A session's date is read in the series' own time zone. The VIX is published in
  Chicago time, the others in New York time.
- The latest row is unreliable (a partial bar while the market is open, a missing
  close for hours afterwards). Rows that are not complete are removed and reported
  in `ContextResult.dropped`, and every series is cut back to the last session that
  all four have. A model must never see a session that has not finished.
- The S&P 500's sessions are the calendar. A row on a day the S&P 500 did not trade
  is left out and counted (the VIX has a few, on US holidays). A session the S&P 500
  has and another series lacks is an error.
- Nothing is repaired, filled or guessed. `align_to_sessions` raises when a session
  it is asked for is absent.

The data is for local use only. It is saved under `data/`, which is gitignored.

Run it from the repository root:

    uv run python -m markets_lab.collectors.market_context
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from markets_lab.collectors.nvda_prices import (
    MARKET_TZ,
    MAX_SESSION_GAP_DAYS,
    MAX_STALENESS_DAYS,
    SESSION_COMPLETE_AT,
    Downloader,
    DroppedBar,
    PriceDataError,
    check_history_kept,
    download_raw,
    market_date_and_time,
    save_csv,
)


@dataclass(frozen=True, slots=True)
class ContextSeries:
    """One index: its Yahoo symbol and the largest daily move that is believable."""

    symbol: str
    description: str
    max_abs_daily_move: float


SERIES: dict[str, ContextSeries] = {
    "sp500": ContextSeries("^GSPC", "S&P 500", 0.3),
    "nasdaq100": ContextSeries("^NDX", "Nasdaq 100", 0.3),
    "semis": ContextSeries("^SOX", "PHLX Semiconductor index", 0.3),
    "vix": ContextSeries("^VIX", "Cboe volatility index", 2.0),
}
"""Column name to series. The largest real one-day moves since 1999 are 19% for the
three stock indices and 116% for the VIX (2018-02-05), so each limit is well clear
of real data and still catches a level that is off by a factor."""

CALENDAR_SERIES = "sp500"
"""Its sessions decide which days count. They are identical to NVDA's since 1999."""

HISTORY_START = date(1999, 1, 1)
"""Earlier rows are not needed: NVDA's first session is 1999-01-22."""

DEFAULT_CSV_PATH = Path("data/market_context.csv")

MIN_ROWS = 6_000
"""Fewer rows than this means a truncated history (there are ~7,000 since 1999)."""

IN_PROGRESS = "in progress"
INCOMPLETE = "incomplete"

_MAX_PROBLEMS_SHOWN = 10


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Validated closes, one column per series, plus what was left out and why."""

    closes: pd.DataFrame
    dropped: tuple[DroppedBar, ...]
    off_calendar: Mapping[str, int]
    """Rows left out per series because the calendar series had no session that day."""

    @property
    def last_session(self) -> date:
        return self.closes.index[-1].date()


def normalize_close(raw: pd.DataFrame) -> pd.Series:
    """The closing level from a yfinance frame, indexed by session date.

    No value is changed. The date is the one in the frame's own time zone, so a
    series published in Chicago time keeps its sessions. Dates must be unique and
    in increasing order.
    """
    if "Close" not in raw.columns:
        raise PriceDataError("missing column from Yahoo: Close")
    index = raw.index
    if not isinstance(index, pd.DatetimeIndex):
        raise PriceDataError(f"expected a date index, got {type(index).__name__}")
    if len(index) and not (index == index.normalize()).all():
        raise PriceDataError("timestamps are not at midnight local time; not daily bars")
    if index.tz is not None:
        index = index.tz_localize(None)
    if index.has_duplicates:
        raise PriceDataError("duplicate dates")
    if not index.is_monotonic_increasing:
        raise PriceDataError("dates are not in increasing order")
    close = pd.to_numeric(raw["Close"], errors="raise").astype("float64")
    return pd.Series(close.to_numpy(), index=pd.DatetimeIndex(index, name="date"), name="close")


def split_complete(close: pd.Series, *, now: datetime) -> tuple[pd.Series, tuple[DroppedBar, ...]]:
    """Remove trailing rows whose session is not complete, and say which and why.

    The rules are the ones used for NVDA prices: today's row before
    `SESSION_COMPLETE_AT` is "in progress", and one last row with a missing close is
    "incomplete". A missing close anywhere else is left for `validate` to reject. A
    row dated after today raises.
    """
    today, clock = market_date_and_time(now)
    dropped: list[DroppedBar] = []
    if len(close) and close.index[-1].date() > today:
        raise PriceDataError(f"row dated {close.index[-1].date()} is in the future")
    if len(close) and close.index[-1].date() == today and clock < SESSION_COMPLETE_AT:
        dropped.append(DroppedBar(today, IN_PROGRESS))
        close = close.iloc[:-1]
    if len(close) and bool(pd.isna(close.iloc[-1])):
        dropped.append(DroppedBar(close.index[-1].date(), INCOMPLETE))
        close = close.iloc[:-1]
    return close, tuple(dropped)


def combine(
    series: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, int], dict[date, list[str]]]:
    """Put the series side by side on the calendar series' sessions.

    Returns the closes, the number of off-calendar rows left out per series, and,
    for each session after the last one that every series has, the series that
    lack it. Raises if a series lacks a session the calendar series has.
    """
    if list(series) != list(SERIES):
        raise PriceDataError(f"series are {list(series)}, expected {list(SERIES)}")
    common_last = min(pd.Timestamp(close.index[-1]) for close in series.values())
    later: set[pd.Timestamp] = set()
    for close in series.values():
        later.update(pd.Timestamp(ts) for ts in close.index[close.index > common_last])
    lacking = {
        ts.date(): [name for name, close in series.items() if ts not in close.index]
        for ts in sorted(later)
    }

    trimmed = {name: close[close.index <= common_last] for name, close in series.items()}
    calendar = pd.DatetimeIndex(trimmed[CALENDAR_SERIES].index)
    problems: list[str] = []
    off_calendar: dict[str, int] = {}
    for name, close in trimmed.items():
        missing = calendar.difference(pd.DatetimeIndex(close.index))
        if len(missing):
            sample = ", ".join(str(ts.date()) for ts in missing[:5])
            problems.append(
                f"{name} has no row for {len(missing)} sessions of {CALENDAR_SERIES}, e.g. {sample}"
            )
        extra = len(close.index.difference(calendar))
        if extra:
            off_calendar[name] = extra
    if problems:
        raise PriceDataError("market context is incomplete: " + "; ".join(problems))
    closes = pd.DataFrame(
        {name: close.loc[calendar].to_numpy() for name, close in trimmed.items()},
        index=calendar.rename("date"),
    )
    return closes, off_calendar, lacking


def validate(closes: pd.DataFrame, *, now: datetime, min_rows: int = MIN_ROWS) -> None:
    """Raise `PriceDataError` listing every check that completed-session data fails."""
    problems = _problems(closes, now=now, min_rows=min_rows)
    if problems:
        shown = problems[:_MAX_PROBLEMS_SHOWN]
        more = len(problems) - len(shown)
        suffix = f"; and {more} more" if more else ""
        raise PriceDataError("market context failed validation: " + "; ".join(shown) + suffix)


def fetch_context(
    *,
    now: datetime,
    downloader: Downloader = download_raw,
    min_rows: int = MIN_ROWS,
) -> ContextResult:
    """Download every series, keep completed common sessions, validate, and return."""
    start = pd.Timestamp(HISTORY_START)
    complete: dict[str, pd.Series] = {}
    in_progress: set[date] = set()
    incomplete: dict[date, list[str]] = {}
    for name, spec in SERIES.items():
        try:
            raw = downloader(spec.symbol)
        except Exception as exc:
            raise PriceDataError(f"download of {spec.symbol} ({name}) failed: {exc}") from exc
        if raw.empty:
            raise PriceDataError(f"download of {spec.symbol} ({name}) returned no rows")
        try:
            close = normalize_close(raw)
            close, dropped = split_complete(close[close.index >= start], now=now)
        except PriceDataError as exc:
            raise PriceDataError(f"{name}: {exc}") from exc
        if close.empty:
            raise PriceDataError(f"{name}: no completed sessions since {HISTORY_START}")
        for bar in dropped:
            if bar.reason == IN_PROGRESS:
                in_progress.add(bar.session)
            else:
                incomplete.setdefault(bar.session, []).append(name)
        complete[name] = close

    closes, off_calendar, lacking = combine(complete)
    for session, names in lacking.items():
        known = incomplete.setdefault(session, [])
        known.extend(name for name in names if name not in known)
    validate(closes, now=now, min_rows=min_rows)

    dropped_bars = [
        DroppedBar(session, f"{INCOMPLETE}: no completed row in {', '.join(names)}")
        for session, names in incomplete.items()
    ]
    dropped_bars.extend(DroppedBar(session, IN_PROGRESS) for session in in_progress)
    dropped_bars.sort(key=lambda bar: bar.session)
    return ContextResult(closes=closes, dropped=tuple(dropped_bars), off_calendar=off_calendar)


def align_to_sessions(closes: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The context rows for exactly these sessions, in the same order.

    Raises if any session is absent. Nothing is filled from a neighbouring day,
    because a filled value would let a call use a level it could not have seen or
    hide that one was missing.
    """
    missing = sessions.difference(pd.DatetimeIndex(closes.index))
    if len(missing):
        sample = ", ".join(str(ts.date()) for ts in missing[:5])
        last = closes.index[-1].date() if len(closes) else "none"
        raise PriceDataError(
            f"market context has no row for {len(missing)} sessions, e.g. {sample} "
            f"(last context session: {last})"
        )
    return closes.loc[sessions]


def load_csv(path: Path) -> pd.DataFrame:
    """Read closes written by `save_csv`. Call `validate` before trusting them."""
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    missing = [name for name in SERIES if name not in frame.columns]
    if missing:
        raise PriceDataError(f"{path}: missing columns {missing}")
    return frame[list(SERIES)]


def _problems(closes: pd.DataFrame, *, now: datetime, min_rows: int) -> list[str]:
    today, _ = market_date_and_time(now)
    if list(closes.columns) != list(SERIES):
        return [f"columns are {list(closes.columns)}, expected {list(SERIES)}"]
    if len(closes) < min_rows:
        return [f"only {len(closes)} rows, expected at least {min_rows}"]
    index = closes.index
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
    gap_days = pd.Series(index, index=index).diff().dt.days
    flag(gap_days > MAX_SESSION_GAP_DAYS, f"gap of more than {MAX_SESSION_GAP_DAYS} days")
    for name, spec in SERIES.items():
        close = closes[name]
        flag(close.isna(), f"{name}: missing value")
        flag(pd.Series(np.isinf(close.to_numpy()), index=index), f"{name}: infinite value")
        flag(close <= 0, f"{name}: non-positive level")
        flag(
            close.pct_change().abs() > spec.max_abs_daily_move,
            f"{name}: moved more than {spec.max_abs_daily_move:.0%} in a day",
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
        prog="python -m markets_lab.collectors.market_context",
        description="Download and validate daily closes of four market indices into a local CSV.",
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
        result = fetch_context(now=now(), downloader=downloader)
        if out.exists():
            check_history_kept(load_csv(out), result.closes)
    except PriceDataError as exc:
        report(f"error: {exc}")
        return 1
    save_csv(result.closes, out)
    first = result.closes.index[0].date()
    report(
        f"saved {len(result.closes)} sessions of {', '.join(SERIES)}, "
        f"{first} to {result.last_session}, to {out}"
    )
    for name, count in result.off_calendar.items():
        report(f"left out {count} {name} rows on days with no {CALENDAR_SERIES} session")
    for bar in result.dropped:
        report(f"left out {bar.session}: {bar.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
