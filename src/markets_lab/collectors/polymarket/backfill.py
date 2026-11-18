"""Backfill settled 15-minute Up/Down windows from the public Polymarket US API.

The run has three stages:

1. Listing. Each series' events are listed and the windows that have ended and are
   resolved are stored, with their opening and closing index prices.
2. Price histories. One request per window for its one-minute price history. This
   is the data that retention may remove, so it goes first and fast.
3. Settlements. One request per window for the outcome the exchange reports, as a
   cross-check on the outcome implied by the index prices. This endpoint is rate
   limited far more tightly than the others, so this stage is slow.

It is read-only against the API and safe to stop and rerun at any point: finished
work is skipped and nothing is written twice.

What this produces is a pilot dataset: one top-of-book-derived price pair per
minute, no order-book depth, and no observation at exactly 12:30.

Run it from the repository root:

    uv run python -m markets_lab.collectors.polymarket.backfill --limit 50
    uv run python -m markets_lab.collectors.polymarket.backfill
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from markets_lab.collectors.polymarket.client import (
    DEFAULT_REQUESTS_PER_SECOND,
    PolymarketAPIError,
    PolymarketClient,
)
from markets_lab.collectors.polymarket.parse import (
    Market,
    ParseError,
    parse_event_markets,
    parse_price_history,
    parse_series_id,
    parse_settlement,
)
from markets_lab.collectors.polymarket.quality import format_quality
from markets_lab.collectors.polymarket.store import Store, StoreError

DEFAULT_SERIES = ("btc-updown-15m", "eth-updown-15m", "sol-updown-15m", "xrp-updown-15m")
DEFAULT_DB_PATH = Path("data/polymarket.sqlite")

DEFAULT_HISTORY_RPS = DEFAULT_REQUESTS_PER_SECOND
DEFAULT_SETTLEMENT_RPS = 0.25
"""One settlement request every 4 seconds.

Measured on 2026-11-18: one every 2 seconds was answered with 429 after about six
requests; one every 3 seconds ran for a minute without a 429.
"""

HISTORY_LEAD_SECONDS = 120
"""Price history is requested from this long before the window starts..."""
HISTORY_TAIL_SECONDS = 60
"""...until this long after it ends."""

SETTLE_GRACE_SECONDS = 300
"""Windows that ended more recently than this are left for a later run."""

MAX_CONSECUTIVE_ERRORS = 20
"""This many failed windows in a row means something is wrong; the run stops."""

Report = Callable[[str], None]


class BackfillAbortedError(RuntimeError):
    """The run stopped early because too many windows in a row failed."""


@dataclass(slots=True)
class RunStats:
    """What one backfill run did."""

    run_id: int = 0
    events_seen: int = 0
    windows_seen: int = 0
    not_settled_yet: int = 0
    already_complete: int = 0
    selected: int = 0
    histories_fetched: int = 0
    points_added: int = 0
    outcomes_recorded: int = 0
    rate_limited: int = 0
    errors: Counter[str] = field(default_factory=lambda: Counter[str]())
    outcome: str = "running"

    @property
    def error_count(self) -> int:
        return sum(self.errors.values())

    def one_line(self) -> str:
        return (
            f"{self.outcome}: windows seen {self.windows_seen}, already complete "
            f"{self.already_complete}, not settled yet {self.not_settled_yet}, selected "
            f"{self.selected}, histories fetched {self.histories_fetched}, points added "
            f"{self.points_added}, outcomes recorded {self.outcomes_recorded}, "
            f"429s {self.rate_limited}, errors {self.error_count}"
        )


def run_backfill(
    client: PolymarketClient,
    store: Store,
    *,
    series_slugs: Sequence[str] = DEFAULT_SERIES,
    limit: int | None = None,
    fetch_settlements: bool = True,
    history_rps: float = DEFAULT_HISTORY_RPS,
    settlement_rps: float = DEFAULT_SETTLEMENT_RPS,
    clock: Callable[[], float] = time.time,
    report: Report = print,
    progress_seconds: float = 30.0,
    stats: RunStats | None = None,
) -> RunStats:
    """Backfill up to `limit` windows that still need work, and return the counts.

    Per-window problems are recorded in `backfill_errors` and the run moves on. The
    run is always closed in `backfill_runs`, whether it finishes, is interrupted, or
    aborts after `MAX_CONSECUTIVE_ERRORS` failed windows in a row. Pass your own
    `stats` object to keep the counts when the run raises.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if stats is None:
        stats = RunStats()
    stats.run_id = store.start_run(series_slugs, started_at=int(clock()))
    runner = _Runner(client, store, stats, clock, report, progress_seconds)
    rate_limited_before = client.rate_limited_count
    try:
        client.set_requests_per_second(history_rps)
        selected = runner.discover(series_slugs, limit)
        runner.fetch_histories(selected)
        if fetch_settlements:
            client.set_requests_per_second(settlement_rps)
            runner.fetch_settlements(selected)
        else:
            report("settlements skipped (--no-settlements)")
        stats.outcome = "finished"
    except KeyboardInterrupt:
        stats.outcome = "interrupted"
        raise
    except BaseException:
        stats.outcome = "aborted"
        raise
    finally:
        stats.rate_limited = client.rate_limited_count - rate_limited_before
        store.finish_run(stats.run_id, finished_at=int(clock()), summary=stats.one_line())
    return stats


class _Runner:
    def __init__(
        self,
        client: PolymarketClient,
        store: Store,
        stats: RunStats,
        clock: Callable[[], float],
        report: Report,
        progress_seconds: float,
    ) -> None:
        self._client = client
        self._store = store
        self._stats = stats
        self._clock = clock
        self._report = report
        self._progress_seconds = progress_seconds
        self._rate_limited_at_start = client.rate_limited_count

    # -- stage 1: listing --------------------------------------------------------

    def discover(self, series_slugs: Sequence[str], limit: int | None) -> list[Market]:
        """List each series and return the settled windows that still need work.

        A window needs work if it lacks a fetched price history or a recorded
        settlement outcome. Listing stops as soon as `limit` such windows are found.
        Each returned window has been stored in `markets`.
        """
        complete = self._store.complete_market_slugs()
        selected: list[Market] = []
        for series_slug in series_slugs:
            if limit is not None and len(selected) >= limit:
                break
            self._report(f"listing {series_slug} ...")
            before = len(selected)
            try:
                series_id = parse_series_id(self._client.get_series(series_slug), series_slug)
                for event in self._client.iter_events(series_id):
                    self._stats.events_seen += 1
                    self._collect(event, complete, selected)
                    if limit is not None and len(selected) >= limit:
                        break
            except (PolymarketAPIError, ParseError) as exc:
                # Losing a whole series listing is not a per-window problem.
                self._error("listing", f"{series_slug}: {exc}", None)
                raise
            self._report(f"  {series_slug}: {len(selected) - before} windows need work")
        if limit is not None:
            selected = selected[:limit]
        self._stats.selected = len(selected)
        return selected

    def _collect(self, event: object, complete: set[str], selected: list[Market]) -> None:
        try:
            markets = parse_event_markets(event)
        except ParseError as exc:
            self._error("parse-event", str(exc), None)
            return
        now = int(self._clock())
        for market in markets:
            self._stats.windows_seen += 1
            if market.market_slug in complete:
                self._stats.already_complete += 1
            elif not market.is_resolved or market.window_end + SETTLE_GRACE_SECONDS > now:
                self._stats.not_settled_yet += 1
            else:
                try:
                    self._store.upsert_market(market, fetched_at=now)
                except StoreError as exc:
                    self._error("store-market", str(exc), market.market_slug)
                else:
                    selected.append(market)

    # -- stage 2: price histories ------------------------------------------------

    def fetch_histories(self, selected: Sequence[Market]) -> None:
        done = self._store.market_slugs_with_history()
        self._run_stage(
            "histories", [m for m in selected if m.market_slug not in done], self._fetch_history
        )

    def _fetch_history(self, market: Market) -> bool:
        slug = market.market_slug
        try:
            history = parse_price_history(
                self._client.get_price_history(
                    slug,
                    start_ts=market.window_start - HISTORY_LEAD_SECONDS,
                    end_ts=market.window_end + HISTORY_TAIL_SECONDS,
                )
            )
            self._stats.points_added += self._store.save_price_history(
                slug, history, fetched_at=int(self._clock())
            )
        except (PolymarketAPIError, ParseError, StoreError) as exc:
            self._error("price-history", str(exc), slug)
            return False
        self._stats.histories_fetched += 1
        return True

    # -- stage 3: settlements ----------------------------------------------------

    def fetch_settlements(self, selected: Sequence[Market]) -> None:
        done = self._store.market_slugs_with_outcome()
        self._run_stage(
            "settlements",
            [m for m in selected if m.market_slug not in done],
            self._fetch_settlement,
        )

    def _fetch_settlement(self, market: Market) -> bool:
        slug = market.market_slug
        try:
            settlement = self._client.get_settlement(slug)
            if settlement is None:
                self._error("settlement", "resolved market has no settlement", slug)
                return False
            self._store.set_outcome(
                slug, parse_settlement(settlement), fetched_at=int(self._clock())
            )
        except (PolymarketAPIError, ParseError, StoreError) as exc:
            self._error("settlement", str(exc), slug)
            return False
        self._stats.outcomes_recorded += 1
        if self._store.outcome_disagrees_with_prices(slug):
            # Both values stay in the row; the disagreement is a finding, not a failure.
            self._error("outcome-mismatch", "settlement outcome contradicts the index prices", slug)
        return True

    # -- shared ------------------------------------------------------------------

    def _run_stage(
        self, name: str, targets: Sequence[Market], fetch: Callable[[Market], bool]
    ) -> None:
        """Run `fetch` over the targets with progress lines and the abort rule."""
        total = len(targets)
        if total == 0:
            self._report(f"{name}: nothing to fetch")
            return
        rate = self._client.requests_per_second
        self._report(
            f"{name}: fetching {total} at {rate:g} requests/second, "
            f"at least {format_duration(total / rate)}"
        )
        started = self._clock()
        last_report = started
        consecutive_errors = 0
        for done, market in enumerate(targets, start=1):
            if fetch(market):
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise BackfillAbortedError(
                        f"{name}: {consecutive_errors} windows in a row failed; stopping. "
                        "See backfill_errors for the reasons."
                    )
            now = self._clock()
            if now - last_report >= self._progress_seconds or done == total:
                last_report = now
                self._report(self._progress_line(name, done, total, now - started))

    def _progress_line(self, name: str, done: int, total: int, elapsed: float) -> str:
        remaining = (total - done) * elapsed / done
        rate_limited = self._client.rate_limited_count - self._rate_limited_at_start
        return (
            f"{name} {done}/{total} ({100 * done / total:.1f}%) | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)} | "
            f"points {self._stats.points_added} | 429s {rate_limited} | "
            f"errors {self._stats.error_count}"
        )

    def _error(self, stage: str, message: str, market_slug: str | None) -> None:
        self._stats.errors[stage] += 1
        self._store.record_error(
            self._stats.run_id,
            stage=stage,
            message=message,
            occurred_at=int(self._clock()),
            market_slug=market_slug,
        )


def format_duration(seconds: float) -> str:
    """A duration such as 45s, 12m05s or 2h03m."""
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_summary(stats: RunStats, store: Store) -> str:
    """The end-of-run summary: this run's counts, then the state of the database."""
    by_stage = ", ".join(f"{stage} {n}" for stage, n in sorted(stats.errors.items()))
    lines = [
        "",
        f"=== run {stats.run_id} {stats.outcome} ===",
        f"events listed               {stats.events_seen}",
        f"windows seen                {stats.windows_seen}",
        f"  already complete, skipped {stats.already_complete}",
        f"  not settled yet, skipped  {stats.not_settled_yet}",
        f"windows selected this run   {stats.selected}",
        f"price histories fetched     {stats.histories_fetched}",
        f"price points added          {stats.points_added}",
        f"settlement outcomes fetched {stats.outcomes_recorded}",
        f"rate-limit responses (429)  {stats.rate_limited}",
        f"errors this run             {stats.error_count}" + (f" ({by_stage})" if by_stage else ""),
        "",
        "=== database ===",
        format_quality(store.connection),
    ]
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.collectors.polymarket.backfill",
        description="Backfill settled 15-minute Up/Down windows (read-only, resumable).",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help=f"SQLite file (default {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(DEFAULT_SERIES),
        metavar="SLUG",
        help="series to backfill (default: the four 15-minute crypto series)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="work on at most N windows that still need a history or an outcome, then stop",
    )
    parser.add_argument(
        "--no-settlements",
        action="store_true",
        help="skip the slow settlement stage; a later run without this flag fills it in",
    )
    parser.add_argument(
        "--history-rps",
        type=_positive_float,
        default=DEFAULT_HISTORY_RPS,
        help=f"requests per second for listing and price history (default {DEFAULT_HISTORY_RPS})",
    )
    parser.add_argument(
        "--settlement-rps",
        type=_positive_float,
        default=DEFAULT_SETTLEMENT_RPS,
        help=f"requests per second for settlements (default {DEFAULT_SETTLEMENT_RPS})",
    )
    parser.add_argument(
        "--progress-seconds",
        type=_positive_float,
        default=30.0,
        help="seconds between progress lines (default 30)",
    )
    return parser


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[..., PolymarketClient] = PolymarketClient,
    report: Report = _print,
    clock: Callable[[], float] = time.time,
) -> int:
    args = build_parser().parse_args(argv)
    db_path: Path = args.db
    if not db_path.parent.is_dir():
        report(f"error: directory {db_path.parent} does not exist; run from the repository root")
        return 2
    report(f"database: {db_path}")
    with (
        Store(db_path) as store,
        client_factory(requests_per_second=args.history_rps) as client,
    ):
        stats = RunStats()
        exit_code = 0
        try:
            run_backfill(
                client,
                store,
                series_slugs=args.series,
                limit=args.limit,
                fetch_settlements=not args.no_settlements,
                history_rps=args.history_rps,
                settlement_rps=args.settlement_rps,
                clock=clock,
                report=report,
                progress_seconds=args.progress_seconds,
                stats=stats,
            )
        except KeyboardInterrupt:
            report("interrupted; progress so far is saved and a rerun will continue from here")
            exit_code = 130
        except (BackfillAbortedError, PolymarketAPIError, ParseError) as exc:
            report(f"error: {exc}")
            exit_code = 1
        report(format_summary(stats, store))
        if exit_code == 0 and stats.error_count:
            exit_code = 3
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
