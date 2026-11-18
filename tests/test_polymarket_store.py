"""The SQLite store: append-only observations, idempotent writes, resuming, fee labels."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from markets_lab.collectors.polymarket.fees import (
    FEE_SCHEDULES,
    SCHEDULE_2026_11_14,
    SCHEDULE_2026_11_14_START,
    UNKNOWN_BEFORE_2026_11_14,
    fee_schedule_for,
)
from markets_lab.collectors.polymarket.parse import DOWN, UP, Market, PricePoint
from markets_lab.collectors.polymarket.store import (
    PRICE_HISTORY_SOURCE,
    SCHEMA_VERSION,
    ConflictError,
    Store,
    StoreError,
)

WINDOW_START = SCHEDULE_2026_11_14_START + 3_600
WINDOW_END = WINDOW_START + 900
AFTER_WINDOW = WINDOW_END + 60

MARKET = Market(
    market_slug="synthetic-btc-1",
    market_id=1001,
    event_slug="synthetic-event-1",
    series_slug="btc-updown-15m",
    coin="btc",
    index_symbol="BRTI",
    horizon="15m",
    window_start=WINDOW_START,
    window_end=WINDOW_END,
    price_to_beat=500_000_000,
    settlement_price=500_105_000,
    status="MARKET_STATUS_RESOLVED",
)

POINTS = [
    PricePoint(ts=WINDOW_START + 720, long_price=8_500, short_price=1_600),
    PricePoint(ts=WINDOW_START + 780, long_price=9_000, short_price=1_100),
]


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def count(store: Store, table: str) -> int:
    return store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def market_row(store: Store, slug: str = MARKET.market_slug) -> sqlite3.Row:
    return store.connection.execute(
        "SELECT * FROM markets WHERE market_slug = ?", (slug,)
    ).fetchone()


class TestFeeSchedules:
    def test_boundary_is_midnight_eastern_on_2026_11_14(self) -> None:
        midnight = datetime(2026, 11, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert int(midnight.timestamp()) == SCHEDULE_2026_11_14_START

    def test_window_before_the_boundary_is_labelled_unknown(self) -> None:
        last_window_before = SCHEDULE_2026_11_14_START - 900
        schedule = fee_schedule_for(last_window_before)
        assert schedule is UNKNOWN_BEFORE_2026_11_14
        assert not schedule.is_known
        assert schedule.taker_coefficient_tier1 is None

    def test_window_starting_at_the_boundary_uses_the_new_schedule(self) -> None:
        schedule = fee_schedule_for(SCHEDULE_2026_11_14_START)
        assert schedule is SCHEDULE_2026_11_14
        assert str(schedule.taker_coefficient_tier1) == "0.095"

    def test_schedules_cover_all_time_without_overlap(self) -> None:
        for moment in (0, SCHEDULE_2026_11_14_START - 1, SCHEDULE_2026_11_14_START, 4_000_000_000):
            fee_schedule_for(moment)  # raises unless exactly one schedule matches

    def test_schedules_are_mirrored_into_the_database(self, store: Store) -> None:
        rows = store.connection.execute(
            "SELECT schedule_id, is_known, taker_coefficient_tier1 FROM fee_schedules "
            "ORDER BY schedule_id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("2026-11-14", 1, "0.095"),
            ("pre-2026-11-14-unknown", 0, None),
        ]
        assert len(rows) == len(FEE_SCHEDULES)

    def test_each_market_records_its_fee_schedule(self, store: Store) -> None:
        early = replace(
            MARKET,
            market_slug="synthetic-btc-early",
            window_start=SCHEDULE_2026_11_14_START - 900,
            window_end=SCHEDULE_2026_11_14_START,
        )
        store.upsert_market(early, fetched_at=AFTER_WINDOW)
        store.upsert_market(MARKET, fetched_at=AFTER_WINDOW)
        labels = dict(
            store.connection.execute(
                "SELECT m.market_slug, f.is_known FROM markets m "
                "JOIN fee_schedules f ON f.schedule_id = m.fee_schedule_id"
            ).fetchall()
        )
        assert labels == {"synthetic-btc-early": 0, "synthetic-btc-1": 1}
        assert market_row(store, "synthetic-btc-early")["fee_schedule_id"] == (
            "pre-2026-11-14-unknown"
        )


class TestSchema:
    def test_new_database_gets_the_current_schema(self, store: Store) -> None:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_file_database_uses_wal(self, tmp_path: Path) -> None:
        with Store(tmp_path / "pm.sqlite") as store:
            assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_reopening_keeps_data(self, tmp_path: Path) -> None:
        path = tmp_path / "pm.sqlite"
        with Store(path) as store:
            store.upsert_market(MARKET, fetched_at=AFTER_WINDOW)
        with Store(path) as store:
            assert count(store, "markets") == 1
            assert count(store, "fee_schedules") == len(FEE_SCHEDULES)

    def test_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "pm.sqlite"
        Store(path).close()
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.close()
        with pytest.raises(StoreError, match="not supported"):
            Store(path)

    def test_foreign_database_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE something_else (x)")
        conn.commit()
        conn.close()
        with pytest.raises(StoreError, match="refusing"):
            Store(path)


class TestMarkets:
    def test_insert_then_refresh(self, store: Store) -> None:
        assert store.upsert_market(MARKET, fetched_at=100) is True
        assert store.upsert_market(MARKET, fetched_at=200) is False
        assert count(store, "markets") == 1
        row = market_row(store)
        assert (row["first_fetched_at"], row["last_fetched_at"]) == (100, 200)

    def test_prices_are_filled_in_later_and_never_erased(self, store: Store) -> None:
        pending = replace(
            MARKET, price_to_beat=None, settlement_price=None, status="MARKET_STATUS_OPEN"
        )
        store.upsert_market(pending, fetched_at=100)
        assert market_row(store)["price_to_beat"] is None
        store.upsert_market(MARKET, fetched_at=200)
        store.upsert_market(pending, fetched_at=300)
        row = market_row(store)
        assert row["price_to_beat"] == 500_000_000
        assert row["settlement_price"] == 500_105_000
        assert row["status"] == "MARKET_STATUS_OPEN"

    @pytest.mark.parametrize(
        "changed",
        [
            {"price_to_beat": 500_000_100},
            {"settlement_price": 499_000_000},
            {"window_start": WINDOW_START + 900, "window_end": WINDOW_END + 900},
            {"coin": "eth"},
            {"series_slug": "eth-updown-15m"},
            {"market_id": 2002},
        ],
    )
    def test_contradicting_a_stored_fact_raises_and_changes_nothing(
        self, store: Store, changed: dict[str, int | str]
    ) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        before = tuple(market_row(store))
        with pytest.raises(ConflictError):
            store.upsert_market(replace(MARKET, **changed), fetched_at=200)  # type: ignore[arg-type]
        assert tuple(market_row(store)) == before

    @pytest.mark.parametrize(
        ("open_", "close", "expected"),
        [
            (500_000_000, 500_000_100, UP),
            (500_000_000, 499_999_900, DOWN),
            (500_000_000, 500_000_000, UP),
            (None, 500_000_000, None),
            (500_000_000, None, None),
        ],
    )
    def test_outcome_from_prices_column(
        self, store: Store, open_: int | None, close: int | None, expected: int | None
    ) -> None:
        store.upsert_market(
            replace(MARKET, price_to_beat=open_, settlement_price=close), fetched_at=100
        )
        assert market_row(store)["outcome_from_prices"] == expected


class TestOutcome:
    def test_set_once_and_repeat(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        store.set_outcome(MARKET.market_slug, UP, fetched_at=200)
        store.set_outcome(MARKET.market_slug, UP, fetched_at=300)
        row = market_row(store)
        assert (row["outcome"], row["outcome_fetched_at"]) == (UP, 200)

    def test_a_different_outcome_raises(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        store.set_outcome(MARKET.market_slug, UP, fetched_at=200)
        with pytest.raises(ConflictError, match="outcome"):
            store.set_outcome(MARKET.market_slug, DOWN, fetched_at=300)
        assert market_row(store)["outcome"] == UP

    @pytest.mark.parametrize("bad", [2, -1])
    def test_only_zero_or_one(self, store: Store, bad: int) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        with pytest.raises(ValueError, match="outcome must be"):
            store.set_outcome(MARKET.market_slug, bad, fetched_at=200)

    def test_unknown_market_raises(self, store: Store) -> None:
        with pytest.raises(StoreError, match="unknown market"):
            store.set_outcome("nope", UP, fetched_at=200)

    def test_disagreement_with_prices_is_detectable_not_hidden(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)  # prices imply Up
        assert not store.outcome_disagrees_with_prices(MARKET.market_slug)
        store.set_outcome(MARKET.market_slug, DOWN, fetched_at=200)
        assert store.outcome_disagrees_with_prices(MARKET.market_slug)
        row = market_row(store)
        assert (row["outcome"], row["outcome_from_prices"]) == (DOWN, UP)


class TestPricePoints:
    def test_rows_carry_snapshot_time_and_fetch_time(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        assert store.save_price_history(MARKET.market_slug, POINTS, fetched_at=AFTER_WINDOW) == 2
        rows = store.connection.execute(
            "SELECT snapshot_ts, long_price, short_price, source, fetched_at "
            "FROM price_points ORDER BY snapshot_ts"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            (WINDOW_START + 720, 8_500, 1_600, PRICE_HISTORY_SOURCE, AFTER_WINDOW),
            (WINDOW_START + 780, 9_000, 1_100, PRICE_HISTORY_SOURCE, AFTER_WINDOW),
        ]

    def test_saving_the_same_points_again_adds_nothing(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        store.save_price_history(MARKET.market_slug, POINTS, fetched_at=AFTER_WINDOW)
        assert (
            store.save_price_history(MARKET.market_slug, POINTS, fetched_at=AFTER_WINDOW + 5) == 0
        )
        assert count(store, "price_points") == 2
        first_fetch = store.connection.execute("SELECT DISTINCT fetched_at FROM price_points")
        assert [r[0] for r in first_fetch] == [AFTER_WINDOW]

    def test_points_with_a_missing_side_are_not_duplicated(self, store: Store) -> None:
        one_sided = [PricePoint(ts=WINDOW_START + 720, long_price=100, short_price=None)]
        store.upsert_market(MARKET, fetched_at=100)
        store.save_price_history(MARKET.market_slug, one_sided, fetched_at=AFTER_WINDOW)
        store.save_price_history(MARKET.market_slug, one_sided, fetched_at=AFTER_WINDOW + 5)
        assert count(store, "price_points") == 1

    def test_a_changed_value_is_appended_and_the_original_kept(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        store.save_price_history(MARKET.market_slug, POINTS[:1], fetched_at=AFTER_WINDOW)
        revised = [replace(POINTS[0], long_price=8_600)]
        assert (
            store.save_price_history(MARKET.market_slug, revised, fetched_at=AFTER_WINDOW + 5) == 1
        )
        rows = store.connection.execute(
            "SELECT long_price, fetched_at FROM price_points ORDER BY fetched_at"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(8_500, AFTER_WINDOW), (8_600, AFTER_WINDOW + 5)]

    @pytest.mark.parametrize(
        "statement",
        ["UPDATE price_points SET long_price = 1", "DELETE FROM price_points"],
    )
    def test_rows_cannot_be_updated_or_deleted(self, store: Store, statement: str) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        store.save_price_history(MARKET.market_slug, POINTS, fetched_at=AFTER_WINDOW)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute(statement)
        store.connection.rollback()
        assert count(store, "price_points") == 2

    def test_unknown_market_raises(self, store: Store) -> None:
        with pytest.raises(StoreError, match="unknown market"):
            store.save_price_history("nope", POINTS, fetched_at=AFTER_WINDOW)

    def test_a_failed_save_stores_nothing(self, store: Store) -> None:
        store.upsert_market(MARKET, fetched_at=100)
        bad_batch = [*POINTS, PricePoint(ts=WINDOW_START + 840, long_price=20_000, short_price=0)]
        with pytest.raises(sqlite3.IntegrityError):
            store.save_price_history(MARKET.market_slug, bad_batch, fetched_at=AFTER_WINDOW)
        assert count(store, "price_points") == 0
        assert market_row(store)["history_fetched_at"] is None


class TestResuming:
    def test_complete_needs_both_outcome_and_history(self, store: Store) -> None:
        slug = MARKET.market_slug
        assert not store.is_complete(slug)
        store.upsert_market(MARKET, fetched_at=100)
        assert not store.is_complete(slug)
        store.save_price_history(slug, POINTS, fetched_at=AFTER_WINDOW)
        assert not store.is_complete(slug)
        store.set_outcome(slug, UP, fetched_at=AFTER_WINDOW)
        assert store.is_complete(slug)
        assert store.complete_market_slugs() == {slug}

    def test_history_fetched_during_the_window_is_kept_but_not_final(self, store: Store) -> None:
        slug = MARKET.market_slug
        store.upsert_market(MARKET, fetched_at=100)
        store.set_outcome(slug, UP, fetched_at=100)
        store.save_price_history(slug, POINTS[:1], fetched_at=WINDOW_END - 1)
        assert count(store, "price_points") == 1
        assert not store.is_complete(slug)
        store.save_price_history(slug, POINTS, fetched_at=WINDOW_END)
        assert store.is_complete(slug)
        assert count(store, "price_points") == 2

    def test_an_empty_history_after_the_window_still_completes(self, store: Store) -> None:
        slug = MARKET.market_slug
        store.upsert_market(MARKET, fetched_at=100)
        store.set_outcome(slug, DOWN, fetched_at=AFTER_WINDOW)
        assert store.save_price_history(slug, [], fetched_at=AFTER_WINDOW) == 0
        assert store.is_complete(slug)

    def test_stopping_and_rerunning_leaves_no_duplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "pm.sqlite"
        other = replace(MARKET, market_slug="synthetic-btc-2", market_id=1002)

        def run(store: Store, markets: list[Market]) -> None:
            for market in markets:
                if store.is_complete(market.market_slug):
                    continue
                store.upsert_market(market, fetched_at=AFTER_WINDOW)
                store.save_price_history(market.market_slug, POINTS, fetched_at=AFTER_WINDOW)
                store.set_outcome(market.market_slug, UP, fetched_at=AFTER_WINDOW)

        # First run is interrupted after one market and a half-written second one.
        with Store(path) as store:
            run(store, [MARKET])
            store.upsert_market(other, fetched_at=AFTER_WINDOW)
            store.save_price_history(other.market_slug, POINTS, fetched_at=AFTER_WINDOW)
        # Second and third runs go over everything again.
        for _ in range(2):
            with Store(path) as store:
                run(store, [MARKET, other])
        with Store(path) as store:
            assert count(store, "markets") == 2
            assert count(store, "price_points") == 4
            assert store.complete_market_slugs() == {"synthetic-btc-1", "synthetic-btc-2"}


class TestRunLog:
    def test_run_lifecycle(self, store: Store) -> None:
        run_id = store.start_run(["btc-updown-15m", "eth-updown-15m"], started_at=100)
        store.finish_run(run_id, finished_at=200, summary="2 markets")
        row = store.connection.execute("SELECT * FROM backfill_runs").fetchone()
        assert tuple(row) == (run_id, 100, 200, "btc-updown-15m,eth-updown-15m", "2 markets")

    def test_a_run_cannot_be_finished_twice(self, store: Store) -> None:
        run_id = store.start_run(["btc-updown-15m"], started_at=100)
        store.finish_run(run_id, finished_at=200, summary="ok")
        with pytest.raises(StoreError, match="already finished"):
            store.finish_run(run_id, finished_at=300, summary="again")

    def test_errors_are_appended_and_cannot_be_removed(self, store: Store) -> None:
        run_id = store.start_run(["btc-updown-15m"], started_at=100)
        store.record_error(
            run_id, stage="settlement", message="boom", occurred_at=150, market_slug="m"
        )
        store.record_error(
            run_id, stage="events", message="bang", occurred_at=160, market_slug=None
        )
        assert count(store, "backfill_errors") == 2
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("DELETE FROM backfill_errors")
        store.connection.rollback()
        assert count(store, "backfill_errors") == 2

    def test_error_needs_an_existing_run(self, store: Store) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.record_error(999, stage="x", message="y", occurred_at=1, market_slug=None)
