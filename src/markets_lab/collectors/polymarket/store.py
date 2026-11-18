"""SQLite storage for Polymarket US market data.

Design rules:

- Raw observations are append-only. `price_points` rows are never updated or
  deleted; database triggers reject both. Each row carries the time the observation
  describes (`snapshot_ts`) and the time it was fetched (`fetched_at`).
- Writing is idempotent. Saving the same observation again adds nothing. If the API
  later returns a different value for the same market and time, it is appended as a
  new row with its own `fetched_at`; the earlier row is kept.
- A backfill can be stopped and rerun. A market is complete once it has a recorded
  outcome and a price history fetched after its window ended; each market's history
  is written in a single transaction.
- Settled facts do not change silently. A different outcome, price or window for a
  market that already has one raises `ConflictError`.
- Every market records the fee schedule in force for its window.

Times are Unix seconds UTC. Prices are integers in units of $0.0001 (see `parse`).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

from markets_lab.collectors.polymarket.fees import FEE_SCHEDULES, fee_schedule_for
from markets_lab.collectors.polymarket.parse import DOWN, PRICE_SCALE, UP, Market, PricePoint

SCHEMA_VERSION = 1

PRICE_HISTORY_SOURCE = "price-history"
"""`price_points.source` for rows from the one-minute price-history endpoint."""

_SCHEMA = f"""
CREATE TABLE fee_schedules (
    schedule_id              TEXT PRIMARY KEY,
    effective_from           INTEGER,
    effective_until          INTEGER,
    is_known                 INTEGER NOT NULL CHECK (is_known IN (0, 1)),
    taker_coefficient_tier1  TEXT,
    source_url               TEXT,
    note                     TEXT NOT NULL
) STRICT;

CREATE TABLE markets (
    market_slug         TEXT PRIMARY KEY,
    market_id           INTEGER NOT NULL,
    event_slug          TEXT NOT NULL,
    series_slug         TEXT NOT NULL,
    coin                TEXT NOT NULL,
    index_symbol        TEXT NOT NULL,
    horizon             TEXT NOT NULL,
    window_start        INTEGER NOT NULL,
    window_end          INTEGER NOT NULL CHECK (window_end > window_start),
    price_to_beat       INTEGER CHECK (price_to_beat > 0),
    settlement_price    INTEGER CHECK (settlement_price > 0),
    status              TEXT NOT NULL,
    fee_schedule_id     TEXT NOT NULL REFERENCES fee_schedules (schedule_id),
    outcome             INTEGER CHECK (outcome IN ({DOWN}, {UP})),
    outcome_fetched_at  INTEGER,
    history_fetched_at  INTEGER,
    first_fetched_at    INTEGER NOT NULL,
    last_fetched_at     INTEGER NOT NULL,
    outcome_from_prices INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN price_to_beat IS NULL OR settlement_price IS NULL THEN NULL
            WHEN settlement_price >= price_to_beat THEN {UP}
            ELSE {DOWN}
        END
    ) VIRTUAL
) STRICT;

CREATE INDEX markets_by_series_and_window ON markets (series_slug, window_start);

CREATE TABLE price_points (
    point_id     INTEGER PRIMARY KEY,
    market_slug  TEXT NOT NULL REFERENCES markets (market_slug),
    snapshot_ts  INTEGER NOT NULL,
    long_price   INTEGER CHECK (long_price BETWEEN 0 AND {PRICE_SCALE}),
    short_price  INTEGER CHECK (short_price BETWEEN 0 AND {PRICE_SCALE}),
    source       TEXT NOT NULL,
    fetched_at   INTEGER NOT NULL
) STRICT;

-- One row per distinct observation. NULL prices are mapped to -1 because SQLite
-- treats NULLs in a unique index as different from each other.
CREATE UNIQUE INDEX price_points_distinct ON price_points (
    market_slug, source, snapshot_ts, COALESCE(long_price, -1), COALESCE(short_price, -1)
);

CREATE TRIGGER price_points_no_update BEFORE UPDATE ON price_points
BEGIN
    SELECT RAISE(ABORT, 'price_points is append-only');
END;

CREATE TRIGGER price_points_no_delete BEFORE DELETE ON price_points
BEGIN
    SELECT RAISE(ABORT, 'price_points is append-only');
END;

CREATE TABLE backfill_runs (
    run_id       INTEGER PRIMARY KEY,
    started_at   INTEGER NOT NULL,
    finished_at  INTEGER,
    series_slugs TEXT NOT NULL,
    summary      TEXT
) STRICT;

CREATE TABLE backfill_errors (
    error_id     INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES backfill_runs (run_id),
    market_slug  TEXT,
    stage        TEXT NOT NULL,
    message      TEXT NOT NULL,
    occurred_at  INTEGER NOT NULL
) STRICT;

CREATE TRIGGER backfill_errors_no_update BEFORE UPDATE ON backfill_errors
BEGIN
    SELECT RAISE(ABORT, 'backfill_errors is append-only');
END;

CREATE TRIGGER backfill_errors_no_delete BEFORE DELETE ON backfill_errors
BEGIN
    SELECT RAISE(ABORT, 'backfill_errors is append-only');
END;

PRAGMA user_version = {SCHEMA_VERSION};
"""

# Columns that identify a market and must never change once stored.
_IDENTITY_FIELDS = (
    "market_id",
    "event_slug",
    "series_slug",
    "coin",
    "index_symbol",
    "horizon",
    "window_start",
    "window_end",
)
# Columns that may start out NULL but must not change once they have a value.
_SETTLED_FIELDS = ("price_to_beat", "settlement_price")


class StoreError(RuntimeError):
    """The database cannot be used as asked."""


class ConflictError(StoreError):
    """New data contradicts a fact that is already stored."""


class Store:
    """The Polymarket SQLite database. Pass ":memory:" for a throwaway database."""

    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._prepare_schema()
        except BaseException:
            self._conn.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for read-only queries by analysis and tests."""
        return self._conn

    # -- markets ---------------------------------------------------------------

    def upsert_market(self, market: Market, *, fetched_at: int) -> bool:
        """Store a market, or refresh one already stored. Returns True if it was new.

        A new market is assigned the fee schedule in force at its window start. For
        a stored market, identity fields and already-known prices must match the
        stored values, otherwise `ConflictError` is raised and nothing changes.
        Prices that were unknown are filled in; known prices are never erased.
        """
        with self._conn:
            stored = self._conn.execute(
                "SELECT * FROM markets WHERE market_slug = ?", (market.market_slug,)
            ).fetchone()
            if stored is None:
                self._conn.execute(
                    """
                    INSERT INTO markets (
                        market_slug, market_id, event_slug, series_slug, coin, index_symbol,
                        horizon, window_start, window_end, price_to_beat, settlement_price,
                        status, fee_schedule_id, first_fetched_at, last_fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market.market_slug,
                        market.market_id,
                        market.event_slug,
                        market.series_slug,
                        market.coin,
                        market.index_symbol,
                        market.horizon,
                        market.window_start,
                        market.window_end,
                        market.price_to_beat,
                        market.settlement_price,
                        market.status,
                        fee_schedule_for(market.window_start).schedule_id,
                        fetched_at,
                        fetched_at,
                    ),
                )
                return True
            for field in _IDENTITY_FIELDS:
                _require_unchanged(market.market_slug, field, stored[field], getattr(market, field))
            for field in _SETTLED_FIELDS:
                new_value = getattr(market, field)
                if stored[field] is not None and new_value is not None:
                    _require_unchanged(market.market_slug, field, stored[field], new_value)
            self._conn.execute(
                """
                UPDATE markets
                SET price_to_beat = COALESCE(price_to_beat, ?),
                    settlement_price = COALESCE(settlement_price, ?),
                    status = ?,
                    last_fetched_at = ?
                WHERE market_slug = ?
                """,
                (
                    market.price_to_beat,
                    market.settlement_price,
                    market.status,
                    fetched_at,
                    market.market_slug,
                ),
            )
            return False

    def set_outcome(self, market_slug: str, outcome: int, *, fetched_at: int) -> None:
        """Record the settlement outcome (1 Up, 0 Down) reported for a market.

        Recording the same outcome again changes nothing. A different outcome raises
        `ConflictError`.
        """
        if outcome not in (DOWN, UP):
            raise ValueError(f"outcome must be {DOWN} or {UP}, not {outcome!r}")
        with self._conn:
            stored = self._market_row(market_slug)
            if stored["outcome"] is None:
                self._conn.execute(
                    "UPDATE markets SET outcome = ?, outcome_fetched_at = ? WHERE market_slug = ?",
                    (outcome, fetched_at, market_slug),
                )
            else:
                _require_unchanged(market_slug, "outcome", stored["outcome"], outcome)

    def outcome_disagrees_with_prices(self, market_slug: str) -> bool:
        """True when the recorded outcome contradicts the open and close index prices."""
        stored = self._market_row(market_slug)
        outcome, from_prices = stored["outcome"], stored["outcome_from_prices"]
        return outcome is not None and from_prices is not None and outcome != from_prices

    # -- price points ----------------------------------------------------------

    def save_price_history(
        self, market_slug: str, points: Sequence[PricePoint], *, fetched_at: int
    ) -> int:
        """Append a market's price-history points. Returns how many rows were added.

        Points already stored with the same time and prices are skipped. If the
        fetch happened at or after the window's end, the market's history is marked
        as fetched, even when there were no points, so a rerun does not ask again.
        A fetch during the window is stored but not marked, because it is partial.
        Everything happens in one transaction: an error stores nothing.

        Only exact duplicates are skipped (`ON CONFLICT DO NOTHING`). `INSERT OR
        IGNORE` is deliberately not used: it would also skip rows that break a
        CHECK constraint, hiding out-of-range prices.
        """
        with self._conn:
            stored = self._market_row(market_slug)
            before = self._conn.total_changes
            self._conn.executemany(
                """
                INSERT INTO price_points
                    (market_slug, snapshot_ts, long_price, short_price, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    (
                        market_slug,
                        p.ts,
                        p.long_price,
                        p.short_price,
                        PRICE_HISTORY_SOURCE,
                        fetched_at,
                    )
                    for p in points
                ],
            )
            added = self._conn.total_changes - before
            if fetched_at >= stored["window_end"] and stored["history_fetched_at"] is None:
                self._conn.execute(
                    "UPDATE markets SET history_fetched_at = ? WHERE market_slug = ?",
                    (fetched_at, market_slug),
                )
            return added

    # -- resuming --------------------------------------------------------------

    def is_complete(self, market_slug: str) -> bool:
        """True once a market has an outcome and a history fetched after its window."""
        stored = self._conn.execute(
            "SELECT outcome, history_fetched_at FROM markets WHERE market_slug = ?",
            (market_slug,),
        ).fetchone()
        return (
            stored is not None
            and stored["outcome"] is not None
            and stored["history_fetched_at"] is not None
        )

    def complete_market_slugs(self) -> set[str]:
        """Slugs of every complete market, so a rerun can skip them."""
        rows = self._conn.execute(
            "SELECT market_slug FROM markets "
            "WHERE outcome IS NOT NULL AND history_fetched_at IS NOT NULL"
        )
        return {row["market_slug"] for row in rows}

    # -- run log ---------------------------------------------------------------

    def start_run(self, series_slugs: Iterable[str], *, started_at: int) -> int:
        """Open a backfill run record and return its id."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO backfill_runs (started_at, series_slugs) VALUES (?, ?)",
                (started_at, ",".join(series_slugs)),
            )
        run_id = cursor.lastrowid
        if run_id is None:
            raise StoreError("could not create a backfill run")
        return run_id

    def finish_run(self, run_id: int, *, finished_at: int, summary: str) -> None:
        """Close a backfill run record with a short summary."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE backfill_runs SET finished_at = ?, summary = ? "
                "WHERE run_id = ? AND finished_at IS NULL",
                (finished_at, summary, run_id),
            )
        if cursor.rowcount != 1:
            raise StoreError(f"backfill run {run_id} does not exist or is already finished")

    def record_error(
        self, run_id: int, *, stage: str, message: str, occurred_at: int, market_slug: str | None
    ) -> None:
        """Append one error from a backfill run. Errors are never edited or removed."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO backfill_errors (run_id, market_slug, stage, message, occurred_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, market_slug, stage, message, occurred_at),
            )

    # -- internals -------------------------------------------------------------

    def _market_row(self, market_slug: str) -> sqlite3.Row:
        stored = self._conn.execute(
            "SELECT * FROM markets WHERE market_slug = ?", (market_slug,)
        ).fetchone()
        if stored is None:
            raise StoreError(f"unknown market: {market_slug}")
        return stored

    def _prepare_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            has_tables = self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
            if has_tables:
                raise StoreError("database has tables but no schema version; refusing to use it")
            self._conn.executescript(f"BEGIN;{_SCHEMA}COMMIT;")
        elif version != SCHEMA_VERSION:
            raise StoreError(
                f"database schema version {version} is not supported (expected {SCHEMA_VERSION})"
            )
        self._sync_fee_schedules()

    def _sync_fee_schedules(self) -> None:
        """Make the fee_schedules table mirror the schedules defined in code."""
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO fee_schedules (
                    schedule_id, effective_from, effective_until, is_known,
                    taker_coefficient_tier1, source_url, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (schedule_id) DO UPDATE SET
                    effective_from = excluded.effective_from,
                    effective_until = excluded.effective_until,
                    is_known = excluded.is_known,
                    taker_coefficient_tier1 = excluded.taker_coefficient_tier1,
                    source_url = excluded.source_url,
                    note = excluded.note
                """,
                [
                    (
                        s.schedule_id,
                        s.effective_from,
                        s.effective_until,
                        int(s.is_known),
                        None
                        if s.taker_coefficient_tier1 is None
                        else str(s.taker_coefficient_tier1),
                        s.source_url,
                        s.note,
                    )
                    for s in FEE_SCHEDULES
                ],
            )


def _require_unchanged(market_slug: str, field: str, stored: object, new: object) -> None:
    if stored != new:
        raise ConflictError(f"{market_slug}: {field} is {stored!r} in the database, got {new!r}")
