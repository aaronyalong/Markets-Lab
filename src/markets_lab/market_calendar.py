"""Which day the next US stock market session falls on, for display only.

The prediction log does not store the session a call is for: a call is scored by
pairing it with the next outcome, so the log never needed a holiday calendar. The
NVIDIA page still wants to say "Calls for Tuesday 24 November" before that session
has happened, so this module works the date out from NYSE's holiday rules.

Nothing here feeds the log, the scoring or the models. A wrong date could only
mislabel the page, and once the session is scored the page uses the date from the
log instead.

Two sources, kept apart on purpose:

- `holidays(year)` applies NYSE's standing rules, so it works for any year,
  including the past (checked against real sessions with the `check` command).
- `PUBLISHED_HOLIDAYS` is NYSE's own list for the years it has published. A next
  session counts as certain only when every day up to it is inside those years.
  A test checks the rules give exactly the published list.

Unscheduled closures (weather, national days of mourning) are announced at short
notice and cannot be known in advance; `PAST_UNSCHEDULED_CLOSURES` lists the ones
found in the history check so that `is_session` is exact for the past.

    uv run python -m markets_lab.market_calendar check data/nvda_daily.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from pathlib import Path

JUNETEENTH_FIRST_YEAR = 2022
"""NYSE first closed for Juneteenth on Monday 20 June 2022."""

# NYSE, "Holidays & Trading Hours", https://www.nyse.com/markets/hours-calendars,
# read 2026-11-24. The page lists 2026, 2027 and 2028. In 2028 New Year's Day falls
# on a Saturday and no holiday is observed for it.
PUBLISHED_HOLIDAYS: dict[int, frozenset[date]] = {
    2026: frozenset(
        {
            date(2026, 1, 1),
            date(2026, 1, 19),
            date(2026, 2, 16),
            date(2026, 4, 3),
            date(2026, 5, 25),
            date(2026, 6, 19),
            date(2026, 7, 3),
            date(2026, 9, 7),
            date(2026, 11, 26),
            date(2026, 12, 25),
        }
    ),
    2027: frozenset(
        {
            date(2027, 1, 1),
            date(2027, 1, 18),
            date(2027, 2, 15),
            date(2027, 3, 26),
            date(2027, 5, 31),
            date(2027, 6, 18),
            date(2027, 7, 5),
            date(2027, 9, 6),
            date(2027, 11, 25),
            date(2027, 12, 24),
        }
    ),
    2028: frozenset(
        {
            date(2028, 1, 17),
            date(2028, 2, 21),
            date(2028, 4, 14),
            date(2028, 5, 29),
            date(2028, 6, 19),
            date(2028, 7, 4),
            date(2028, 9, 4),
            date(2028, 11, 23),
            date(2028, 12, 25),
        }
    ),
}

PAST_UNSCHEDULED_CLOSURES: frozenset[date] = frozenset(
    {
        # The September 11 attacks.
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),  # National day of mourning, President Reagan.
        date(2007, 1, 2),  # National day of mourning, President Ford.
        # Hurricane Sandy.
        date(2012, 10, 29),
        date(2012, 10, 30),
        date(2018, 12, 5),  # National day of mourning, President George H. W. Bush.
        date(2025, 1, 9),  # National day of mourning, President Carter.
    }
)
"""Weekdays the market closed that the holiday rules do not cover.

Found by the history check on 2026-11-24 (7,001 NVDA sessions, 1999-01-22 to
2026-11-19): these ten were the only disagreements, and no session traded on a day
the rules call a holiday."""


@dataclass(frozen=True, slots=True)
class NextSession:
    """The next session after a day, and whether NYSE's published list confirms it."""

    day: date
    certain: bool


def easter(year: int) -> date:
    """Western Easter Sunday (the anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday (Monday is 0) of a month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    after = date(year + month // 12, month % 12 + 1, 1)
    last = after - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """A Saturday holiday moves to the Friday, a Sunday one to the Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@cache
def holidays(year: int) -> frozenset[date]:
    """NYSE's full-day holidays in a year, from its standing rules."""
    days = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King, Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving Day
        _observed(date(year, 12, 25)),  # Christmas Day
    }
    # New Year's Day on a Saturday is not moved back into the old year.
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        days.add(_observed(new_year))
    if year >= JUNETEENTH_FIRST_YEAR:
        days.add(_observed(date(year, 6, 19)))
    return frozenset(days)


def is_session(day: date) -> bool:
    """Whether the market is open on a day: a weekday that is not a holiday."""
    return (
        day.weekday() < 5 and day not in holidays(day.year) and day not in PAST_UNSCHEDULED_CLOSURES
    )


def _published(day: date) -> bool:
    return day.year in PUBLISHED_HOLIDAYS


def next_session(after: date) -> NextSession:
    """The first session after a day.

    `certain` is true only when every day up to and including it falls in a year
    NYSE has published its holidays for. It can still be wrong if the exchange
    closes at short notice; nothing can know that in advance.
    """
    day = after + timedelta(days=1)
    certain = True
    while True:
        certain = certain and _published(day)
        if is_session(day):
            return NextSession(day, certain)
        day += timedelta(days=1)


# ------------------------------------------------------------------ history check


@dataclass(frozen=True, slots=True)
class CheckResult:
    """How the rules compare with the sessions that actually traded."""

    first: date
    last: date
    sessions: int
    missing: list[date]
    """Days the rules call a session that have no trading."""
    unexpected: list[date]
    """Days that traded although the rules call them closed."""


def compare_with_sessions(traded: Iterable[date]) -> CheckResult:
    """Every disagreement between the rules and real sessions, over their span."""
    actual = sorted(set(traded))
    if not actual:
        raise ValueError("no sessions to compare with")
    first, last = actual[0], actual[-1]
    seen = set(actual)
    missing: list[date] = []
    day = first
    while day <= last:
        if is_session(day) and day not in seen:
            missing.append(day)
        day += timedelta(days=1)
    unexpected = [d for d in actual if not is_session(d)]
    return CheckResult(first, last, len(actual), missing, unexpected)


def _read_dates(path: Path) -> list[date]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [date.fromisoformat(row["date"]) for row in csv.DictReader(handle)]


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: Sequence[str] | None = None, *, report: Callable[[str], None] = _print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.market_calendar",
        description="Check NYSE's holiday rules against the sessions in a price file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compare the rules with a CSV that has a date column")
    check.add_argument("prices", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_with_sessions(_read_dates(args.prices))
    except (OSError, KeyError, ValueError) as exc:
        report(f"error: {exc}")
        return 1
    report(f"{result.sessions} sessions from {result.first} to {result.last}")
    report(f"rules expected a session, none traded ({len(result.missing)}):")
    for day in result.missing:
        report(f"  {day} {day:%A}")
    report(f"traded although the rules say closed ({len(result.unexpected)}):")
    for day in result.unexpected:
        report(f"  {day} {day:%A}")
    return 0 if not result.missing and not result.unexpected else 2


if __name__ == "__main__":
    sys.exit(main())
