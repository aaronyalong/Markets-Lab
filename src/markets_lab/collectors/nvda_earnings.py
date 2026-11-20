"""When NVIDIA published financial results, from its SEC filings.

A company that announces results must file a Form 8-K reporting it under Item 2.02
("Results of Operations and Financial Condition"; numbered Item 12 before August
2004). SEC EDGAR records the moment it accepted each filing, to the second. That
moment is used as the time the results became public. It can be slightly late (the
press release usually goes out a few minutes earlier), which errs on the safe side:
a call is never credited with results before they were certainly public.

The rule is fixed and applied without hand-picking: every Form 8-K with Item 2.02
or Item 12 counts, amendments (8-K/A) do not. Measured 2026-11-20:

- 100 such filings, the first on 2003-08-07. NVIDIA filed no results 8-K before
  that, so nothing is known from this source for 1999 to mid 2003.
- 92 were accepted between 16:20 and 16:57 New York time, after the close.
- 6 were accepted before the open of a session (06:11 to 09:29) and 2 during one.
  Some of these are preliminary results or warnings ahead of the quarterly report;
  the rule does not tell them apart and counts them all.
- Consecutive filings are never more than 104 days apart.

Only the acceptance time, the filing's accession number and its item numbers are
kept. `known_at` is the one way to ask what was public at a given moment.

The list is saved under `data/`, which is gitignored. Run it from the repository
root:

    uv run python -m markets_lab.collectors.nvda_earnings
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from markets_lab.collectors.nvda_news import EDGAR_URL, sec_get

SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
DEFAULT_JSON_PATH = Path("data/nvda_results.json")

RESULTS_FORM = "8-K"
RESULTS_ITEMS = frozenset({"2.02", "12"})
"""Item 2.02 and, on filings before August 2004, its predecessor Item 12."""

FIRST_KNOWN_BY = date(2003, 12, 31)
"""The first results filing is from 2003-08-07; a later start means lost history."""

MIN_PUBLICATIONS = 90
"""Fewer than this means SEC returned a truncated list (there are 100)."""

MAX_GAP_DAYS = 130
"""Results come every quarter (largest real gap: 104 days), so more means a hole."""

MAX_STALENESS_DAYS = 130
"""The latest results filing may be at most this many days old."""

_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_OLDER_PAGE = re.compile(r"CIK\d{10}-submissions-\d{3}\.json")
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_COLUMNS = ("form", "accessionNumber", "acceptanceDateTime", "items")
_MAX_PROBLEMS_SHOWN = 10

Getter = Callable[[str], str]


class EarningsDataError(ValueError):
    """Downloaded or stored results-filing data failed a check."""


@dataclass(frozen=True, slots=True, order=True)
class ResultsPublication:
    """One results filing: when SEC accepted it (UTC), and which filing it was."""

    accepted: datetime
    accession: str
    items: tuple[str, ...]


def known_at(
    publications: Sequence[ResultsPublication], moment: datetime
) -> tuple[ResultsPublication, ...]:
    """The results that were public strictly before `moment`.

    Everything that uses results must come through here with the time of the call
    it is preparing, so a call can never use results published at or after its own
    timestamp.
    """
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    return tuple(item for item in publications if item.accepted < moment)


def parse_page(columns: dict[str, Any]) -> list[ResultsPublication]:
    """The results filings in one page of SEC's filing list (a set of columns)."""
    try:
        values = {name: list(columns[name]) for name in _COLUMNS}
    except (KeyError, TypeError) as exc:
        raise EarningsDataError(f"SEC filing list has an unexpected shape: {exc!r}") from exc
    if len({len(column) for column in values.values()}) != 1:
        raise EarningsDataError("SEC filing list has columns of different lengths")

    found: list[ResultsPublication] = []
    for form, accession, accepted, item_codes in zip(*values.values(), strict=True):
        items = tuple(code.strip() for code in str(item_codes or "").split(",") if code.strip())
        if form != RESULTS_FORM or not RESULTS_ITEMS.intersection(items):
            continue
        found.append(ResultsPublication(_parse_accepted(accepted), str(accession), items))
    return found


def parse_submissions(json_text: str) -> tuple[list[ResultsPublication], list[str]]:
    """Results filings on SEC's main page for the company, and its older pages' names."""
    try:
        filings = json.loads(json_text)["filings"]
        recent = filings["recent"]
        older = [str(page["name"]) for page in filings.get("files", [])]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EarningsDataError(f"SEC response has an unexpected shape: {exc!r}") from exc
    for name in older:
        if not _OLDER_PAGE.fullmatch(name):
            raise EarningsDataError(f"SEC response names an unexpected page: {name!r}")
    return parse_page(recent), older


def fetch_publications(*, now: datetime, get: Getter = sec_get) -> list[ResultsPublication]:
    """Download the whole filing list, keep results filings, validate, and return them."""
    try:
        found, older = parse_submissions(get(EDGAR_URL))
        for name in older:
            try:
                page = json.loads(get(SUBMISSIONS_BASE + name))
            except json.JSONDecodeError as exc:
                raise EarningsDataError(f"SEC page {name} is not JSON: {exc!r}") from exc
            found.extend(parse_page(page))
    except EarningsDataError:
        raise
    except Exception as exc:
        raise EarningsDataError(f"download from SEC failed: {exc}") from exc
    publications = sorted(found)
    validate(publications, now=now)
    return publications


def validate(
    publications: Sequence[ResultsPublication],
    *,
    now: datetime,
    min_publications: int = MIN_PUBLICATIONS,
) -> None:
    """Raise `EarningsDataError` listing every check the list fails."""
    problems = _problems(publications, now=now, min_publications=min_publications)
    if problems:
        shown = problems[:_MAX_PROBLEMS_SHOWN]
        more = len(problems) - len(shown)
        suffix = f"; and {more} more" if more else ""
        raise EarningsDataError("results filings failed validation: " + "; ".join(shown) + suffix)


def check_history_kept(
    previous: Sequence[ResultsPublication], new: Sequence[ResultsPublication]
) -> None:
    """Raise if a filing that was stored before is absent or has a different time."""
    current = {item.accession: item for item in new}
    lost = [item.accession for item in previous if current.get(item.accession) != item]
    if lost:
        raise EarningsDataError(
            f"new download changes or drops {len(lost)} stored filings, e.g. {', '.join(lost[:3])}"
        )


def to_json(publications: Sequence[ResultsPublication]) -> str:
    rows = [
        {
            "accepted": item.accepted.strftime(_TIMESTAMP_FORMAT),
            "accession": item.accession,
            "items": list(item.items),
        }
        for item in publications
    ]
    return json.dumps(rows, indent=1) + "\n"


def from_json(text: str) -> list[ResultsPublication]:
    """Read what `to_json` wrote. Call `validate` before trusting it."""
    try:
        rows = json.loads(text)
        return [
            ResultsPublication(
                datetime.strptime(row["accepted"], _TIMESTAMP_FORMAT).replace(tzinfo=UTC),
                str(row["accession"]),
                tuple(str(code) for code in row["items"]),
            )
            for row in rows
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EarningsDataError(f"stored results filings cannot be read: {exc!r}") from exc


def save_json(publications: Sequence[ResultsPublication], path: Path) -> None:
    """Write the list, replacing any existing file in one step."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(to_json(publications), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> list[ResultsPublication]:
    return from_json(path.read_text(encoding="utf-8"))


def _parse_accepted(value: object) -> datetime:
    """SEC writes acceptance times in UTC, e.g. "2026-11-17T21:20:51.000Z"."""
    text = str(value)
    if not text.endswith("Z"):
        raise EarningsDataError(f"acceptance time is not marked as UTC: {text!r}")
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EarningsDataError(f"acceptance time cannot be read: {text!r}") from exc
    return moment.astimezone(UTC).replace(microsecond=0)


def _problems(
    publications: Sequence[ResultsPublication], *, now: datetime, min_publications: int
) -> list[str]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if len(publications) < min_publications:
        return [f"only {len(publications)} filings, expected at least {min_publications}"]

    problems: list[str] = []
    if any(item.accepted.tzinfo is None for item in publications):
        return ["an acceptance time has no time zone"]
    if list(publications) != sorted(publications):
        problems.append("filings are not in order of acceptance")
    accessions = [item.accession for item in publications]
    if len(set(accessions)) != len(accessions):
        problems.append("a filing appears more than once")
    bad = [name for name in accessions if not _ACCESSION.fullmatch(name)]
    if bad:
        problems.append(f"{len(bad)} accession numbers have the wrong form, first {bad[0]!r}")
    wrong = [item.accession for item in publications if not RESULTS_ITEMS.intersection(item.items)]
    if wrong:
        problems.append(f"{len(wrong)} filings are not results filings, first {wrong[0]}")
    if problems:
        return problems  # the checks below assume ordered, unique filings

    first, last = publications[0].accepted, publications[-1].accepted
    if first.date() > FIRST_KNOWN_BY:
        problems.append(f"history starts {first.date()}, expected by {FIRST_KNOWN_BY}")
    for earlier, later in itertools.pairwise(publications):
        gap = (later.accepted - earlier.accepted).days
        if gap > MAX_GAP_DAYS:
            problems.append(
                f"no results filing for {gap} days after {earlier.accepted.date()} "
                f"(more than {MAX_GAP_DAYS})"
            )
    if last > now:
        problems.append(f"last filing {last.date()} is in the future")
    elif (now - last).days > MAX_STALENESS_DAYS:
        problems.append(
            f"stale: last results filing is {last.date()}, {(now - last).days} days before "
            f"{now.astimezone(UTC).date()}"
        )
    return problems


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    get: Getter = sec_get,
    now: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    report: Callable[[str], None] = _print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.collectors.nvda_earnings",
        description="Download and validate the times NVIDIA published results into a local file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"JSON file (default {DEFAULT_JSON_PATH})",
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    if not out.parent.is_dir():
        report(f"error: directory {out.parent} does not exist; run from the repository root")
        return 2
    try:
        publications = fetch_publications(now=now(), get=get)
        if out.exists():
            check_history_kept(load_json(out), publications)
    except EarningsDataError as exc:
        report(f"error: {exc}")
        return 1
    save_json(publications, out)
    first, last = publications[0].accepted, publications[-1].accepted
    span = f"{first.date()} to {last.date()} (UTC)"
    report(f"saved {len(publications)} results filings, {span}, to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
