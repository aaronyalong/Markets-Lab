"""Collect NVIDIA's recent SEC filings (Forms 8-K, 10-Q and 10-K) from SEC EDGAR.

SEC information "may be copied or further distributed" without permission; the
page credits SEC EDGAR. Each title is this site's own plain description of the
form and its item numbers, not text written by NVIDIA or the SEC.

News headlines are not collected. Three sources were tried and dropped, and their
items stay in the old day files, which are still read but never shown:

- NVIDIA's newsroom feed and Yahoo's unofficial news list (through yfinance),
  until 2026-11-24: their terms do not allow it.
- Yahoo Finance's official NVDA RSS feed, 2026-11-24 to 2026-11-26. Its terms do
  allow showing the feed unmodified, but the titles are other publishers' opinion
  headlines ("...a buying opportunity"), which do not belong on a page that gives
  no buy or sell signals, and may not be edited. The page links to Yahoo Finance's
  NVDA news page and to NVIDIA's newsroom instead.

Only the title, source, time and link are kept. Filing text is never stored or
shown: the site links out to it.

Each day's collection is written once to `news/nvda/YYYY-MM-DD.json` (UTC date)
and never changed afterwards, so the archive can later be used as dated model
input without any doubt about what was known when.

Everything fetched here is treated as data: whitespace is normalised, lengths are
capped, and links must be plain https URLs on the expected host. Old day files
hold text written by other people and are re-checked the same way when read.

    uv run python -m markets_lab.collectors.nvda_news            # write today's file
    uv run python -m markets_lab.collectors.nvda_news --dry-run  # show, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

EDGAR_URL = "https://data.sec.gov/submissions/CIK0001045810.json"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/1045810"

# A generic client name. SEC asks automated clients to say who they are; set
# SEC_USER_AGENT (kept as a secret, not in the repository) to add a contact.
DEFAULT_USER_AGENT = "news-collector/0.1"
TIMEOUT_SECONDS = 30

DEFAULT_DIR = Path("news") / "nvda"
WINDOW_HOURS = 24
MAX_TITLE_LENGTH = 200
MAX_PUBLISHER_LENGTH = 60

EDGAR = "sec_edgar"
SOURCES = frozenset({EDGAR})
# No longer collected (see above). Old day files keep their items; the names are
# accepted so those files stay valid, and the items are never shown.
RETIRED_SOURCES = frozenset({"nvidia_newsroom", "yahoo_finance", "yahoo_finance_rss"})

# Filings a reader would recognise as news. Insider-trade forms and the like are
# left out: they are frequent and say little on their own.
FORM_NAMES = {
    "8-K": "current report",
    "10-Q": "quarterly report",
    "10-K": "annual report",
}
# What the item numbers on a Form 8-K mean (SEC's own headings, shortened).
ITEM_NAMES = {
    "1.01": "entry into a material agreement",
    "1.02": "termination of a material agreement",
    "2.01": "completion of an acquisition or disposal",
    "2.02": "results of operations and financial condition",
    "2.03": "creation of a financial obligation",
    "2.05": "costs of exit or disposal activities",
    "3.02": "unregistered sale of shares",
    "5.02": "departure or appointment of directors or officers",
    "5.03": "amendments to articles or bylaws",
    "5.07": "results of a shareholder vote",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
}
ITEMS_NOT_WORTH_NAMING = frozenset({"9.01"})  # "financial statements and exhibits"


class NewsError(RuntimeError):
    """The news could not be collected or stored."""


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One filing (or, in old day files, one headline). `published_at` is always UTC."""

    source: str
    publisher: str
    title: str
    url: str
    published_at: datetime


@dataclass(frozen=True, slots=True)
class Collection:
    """What one run found: items newest first, and the sources that failed."""

    collected_at: datetime
    window_hours: int
    items: tuple[NewsItem, ...]
    failed_sources: tuple[str, ...]


Fetcher = Callable[[], list[NewsItem]]


# ----------------------------------------------------------------------- cleaning


def clean_text(text: object, limit: int) -> str:
    """One line of plain text: whitespace collapsed, control characters dropped."""
    if not isinstance(text, str):
        return ""
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    collapsed = " ".join(printable.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def safe_url(url: object, *, hosts: frozenset[str] | None = None) -> str | None:
    """The URL if it is a plain https link (on an expected host, if given)."""
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if any(ch.isspace() or not ch.isprintable() for ch in candidate):
        return None
    parts = urlsplit(candidate)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return None
    if hosts is not None and parts.hostname not in hosts:
        return None
    return candidate


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("a timestamp without a time zone")
    return moment.astimezone(UTC)


# ------------------------------------------------------------------------ parsing


def filing_title(form: str, item_codes: str) -> str:
    """A plain-English title for a filing, from its form type and 8-K item numbers."""
    title = f"Form {form}: {FORM_NAMES[form]}"
    codes = [code.strip() for code in item_codes.split(",") if code.strip()]
    named = [
        ITEM_NAMES.get(code, f"item {code}") for code in codes if code not in ITEMS_NOT_WORTH_NAMING
    ]
    return f"{title} ({'; '.join(named)})" if named else title


def parse_edgar(json_text: str) -> list[NewsItem]:
    """NVIDIA's recent 8-K, 10-Q and 10-K filings from SEC EDGAR."""
    try:
        recent = json.loads(json_text)["filings"]["recent"]
        columns = {
            name: list(recent[name])
            for name in ("form", "accessionNumber", "acceptanceDateTime", "primaryDocument")
        }
        item_codes = list(recent.get("items") or [""] * len(columns["form"]))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise NewsError(f"SEC EDGAR response has an unexpected shape: {exc!r}") from exc
    lengths = {len(values) for values in columns.values()} | {len(item_codes)}
    if len(lengths) != 1:
        raise NewsError("SEC EDGAR response has columns of different lengths")

    items: list[NewsItem] = []
    for position, form in enumerate(columns["form"]):
        if form not in FORM_NAMES:
            continue
        accession = str(columns["accessionNumber"][position])
        document = str(columns["primaryDocument"][position])
        url = safe_url(
            f"{EDGAR_ARCHIVE}/{accession.replace('-', '')}/{document}",
            hosts=frozenset({"www.sec.gov"}),
        )
        try:
            accepted = str(columns["acceptanceDateTime"][position]).replace("Z", "+00:00")
            published = _utc(datetime.fromisoformat(accepted))
        except ValueError:
            continue
        if url and accession.replace("-", "").isdigit():
            title = filing_title(form, str(item_codes[position] or ""))
            items.append(NewsItem(EDGAR, "SEC EDGAR", title, url, published))
    return items


# ----------------------------------------------------------------------- fetching


def _get(url: str, *, user_agent: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def sec_get(url: str) -> str:
    """Fetch one SEC page, naming the client the way SEC asks automated ones to."""
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
    return _get(url, user_agent=user_agent)


def fetch_edgar() -> list[NewsItem]:
    return parse_edgar(sec_get(EDGAR_URL))


DEFAULT_FETCHERS: dict[str, Fetcher] = {EDGAR: fetch_edgar}


def collect(
    *,
    now: datetime,
    fetchers: Mapping[str, Fetcher] = DEFAULT_FETCHERS,
    window_hours: int = WINDOW_HOURS,
) -> Collection:
    """Fetch every source and keep what was published in the last `window_hours`.

    One source failing is reported and the others are still used. If every source
    fails there is nothing to say, and that is an error.
    """
    now = _utc(now)
    earliest = now - timedelta(hours=window_hours)
    found: list[NewsItem] = []
    failed: list[str] = []
    for name, fetch in fetchers.items():
        try:
            found.extend(fetch())
        except Exception as exc:
            failed.append(name)
            print(f"warning: {name} failed: {exc}", file=sys.stderr, flush=True)
    if fetchers and len(failed) == len(fetchers):
        raise NewsError(f"every news source failed: {', '.join(failed)}")

    recent = [item for item in found if earliest <= item.published_at <= now]
    unique: dict[str, NewsItem] = {}
    for item in sorted(recent, key=lambda i: (i.published_at, i.source, i.url), reverse=True):
        unique.setdefault(item.url, item)
    return Collection(now, window_hours, tuple(unique.values()), tuple(failed))


# ------------------------------------------------------------------------ storage


def day_path(directory: Path, day: date) -> Path:
    return directory / f"{day.isoformat()}.json"


def to_json(collection: Collection) -> str:
    payload = {
        "collected_at": collection.collected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": collection.window_hours,
        "failed_sources": list(collection.failed_sources),
        "items": [
            {**asdict(item), "published_at": item.published_at.strftime("%Y-%m-%dT%H:%M:%SZ")}
            for item in collection.items
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def from_json(text: str) -> Collection:
    """Read a stored day, re-checking every field as if it were new."""
    try:
        payload = json.loads(text)
        items = []
        for raw in payload["items"]:
            url = safe_url(raw["url"])
            title = clean_text(raw["title"], MAX_TITLE_LENGTH)
            publisher = clean_text(raw["publisher"], MAX_PUBLISHER_LENGTH)
            source = str(raw["source"])
            if not url or not title or not publisher or source not in SOURCES | RETIRED_SOURCES:
                raise ValueError(f"bad item: {raw!r}")
            items.append(NewsItem(source, publisher, title, url, _stamp(raw["published_at"])))
        return Collection(
            collected_at=_stamp(payload["collected_at"]),
            window_hours=int(payload["window_hours"]),
            items=tuple(items),
            failed_sources=tuple(str(name) for name in payload["failed_sources"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise NewsError(f"stored news file is not valid: {exc}") from exc


def _stamp(text: object) -> datetime:
    return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def save_day(collection: Collection, directory: Path) -> Path | None:
    """Write the day's file, or return None if that day was already written.

    A day is written once. Running again the same day changes nothing, so a
    retried job cannot rewrite what an earlier run recorded.
    """
    path = day_path(directory, collection.collected_at.date())
    if path.exists():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(collection), encoding="utf-8")
    return path


def load_recent(directory: Path, *, days: int) -> list[Collection]:
    """The most recent stored days, newest first. A missing folder is no news yet."""
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    return [from_json(path.read_text(encoding="utf-8")) for path in reversed(paths[-days:])]


# ------------------------------------------------------------------- command line


def main(
    argv: Sequence[str] | None = None,
    *,
    now: datetime | None = None,
    fetchers: Mapping[str, Fetcher] = DEFAULT_FETCHERS,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.collectors.nvda_news",
        description="Collect the last day's NVIDIA SEC filings into news/nvda/.",
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="show the items, write nothing")
    args = parser.parse_args(argv)
    try:
        collection = collect(now=now or datetime.now(tz=UTC), fetchers=fetchers)
    except NewsError as exc:
        print(f"NEWS COLLECTION FAILED: {exc}", file=sys.stderr)
        return 1

    for item in collection.items:
        print(f"{item.published_at:%Y-%m-%d %H:%M} UTC  {item.publisher}: {item.title}")
    summary = f"{len(collection.items)} item(s) in the last {collection.window_hours} hours"
    if collection.failed_sources:
        summary += f"; failed: {', '.join(collection.failed_sources)}"
    if args.dry_run:
        print(f"{summary}. Dry run: nothing written.")
        return 0
    path = save_day(collection, args.dir)
    if path is None:
        print(f"{summary}. Today's file already exists; nothing written.")
    else:
        print(f"{summary}. Wrote {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
