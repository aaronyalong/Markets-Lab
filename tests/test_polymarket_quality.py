"""Data-quality counts: grid gaps and minute-12/13 coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from markets_lab.collectors.polymarket.fees import SCHEDULE_2026_11_14_START
from markets_lab.collectors.polymarket.parse import DOWN, UP, Market, PricePoint
from markets_lab.collectors.polymarket.quality import (
    error_counts,
    format_quality,
    missing_windows,
    series_quality,
)
from markets_lab.collectors.polymarket.store import Store

START = SCHEDULE_2026_11_14_START + 9_000
LATER = START + 100_000

BASE = Market(
    market_slug="m0",
    market_id=1,
    event_slug="e0",
    series_slug="btc-updown-15m",
    coin="btc",
    index_symbol="BRTI",
    horizon="15m",
    window_start=START,
    window_end=START + 900,
    price_to_beat=500_000_000,
    settlement_price=500_010_000,
    status="MARKET_STATUS_RESOLVED",
)


def add(
    store: Store,
    index: int,
    offsets: list[int],
    *,
    series: str = "btc-updown-15m",
    outcome: int | None = UP,
    start: int | None = None,
    short_price: int | None = 1_600,
) -> Market:
    """Store a window with one price point at each offset (seconds after its start)."""
    window_start = START + 900 * index if start is None else start
    market = replace(
        BASE,
        market_slug=f"{series}-{index}",
        market_id=index,
        series_slug=series,
        window_start=window_start,
        window_end=window_start + 900,
    )
    store.upsert_market(market, fetched_at=LATER)
    points = [
        PricePoint(ts=window_start + o, long_price=8_500, short_price=short_price) for o in offsets
    ]
    store.save_price_history(market.market_slug, points, fetched_at=LATER)
    if outcome is not None:
        store.set_outcome(market.market_slug, outcome, fetched_at=LATER)
    return market


class TestMissingWindows:
    def test_no_gaps(self) -> None:
        assert missing_windows([0, 900, 1800]) == []

    def test_reports_each_missing_start_in_order(self) -> None:
        assert missing_windows([3600, 0, 900]) == [1800, 2700]

    def test_empty_and_single(self) -> None:
        assert missing_windows([]) == []
        assert missing_windows([900]) == []

    def test_duplicates_do_not_matter(self) -> None:
        assert missing_windows([0, 0, 1800]) == [900]

    def test_step_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="step"):
            missing_windows([0], step=0)


class TestSeriesQuality:
    def test_minute_12_and_13_coverage(self) -> None:
        with Store(":memory:") as store:
            add(store, 0, [720, 780])  # both
            add(store, 1, [720])  # minute 12 only
            add(store, 2, [780 + 59])  # minute 13 only, late in the minute
            add(store, 3, [660, 840])  # minutes 11 and 14: neither
            add(store, 4, [])  # no points at all
            (q,) = series_quality(store.connection)
        assert q.markets == 5
        assert (q.has_minute_12, q.has_minute_13, q.has_neither_minute) == (2, 2, 2)
        assert q.price_points == 6
        assert q.complete == 5

    def test_a_point_missing_one_side_is_not_usable(self) -> None:
        with Store(":memory:") as store:
            add(store, 0, [720, 780], short_price=None)
            (q,) = series_quality(store.connection)
        assert (q.has_minute_12, q.has_minute_13, q.has_neither_minute) == (0, 0, 1)

    def test_revised_points_do_not_count_a_window_twice(self) -> None:
        with Store(":memory:") as store:
            market = add(store, 0, [720])
            revised = [
                PricePoint(ts=market.window_start + 720, long_price=8_600, short_price=1_500)
            ]
            store.save_price_history(market.market_slug, revised, fetched_at=LATER + 1)
            (q,) = series_quality(store.connection)
        assert q.price_points == 2
        assert q.has_minute_12 == 1

    def test_counts_are_per_series_with_gaps_and_completeness(self) -> None:
        with Store(":memory:") as store:
            add(store, 0, [720])
            add(store, 3, [720], outcome=None)  # windows 1 and 2 are missing; no outcome
            add(store, 0, [720], series="eth-updown-15m")
            btc, eth = series_quality(store.connection)
        assert (btc.series_slug, btc.markets, btc.complete, btc.with_outcome) == (
            "btc-updown-15m",
            2,
            1,
            1,
        )
        assert btc.missing_windows == 2
        assert (btc.first_window_start, btc.last_window_start) == (START, START + 2_700)
        assert (eth.series_slug, eth.markets, eth.missing_windows) == ("eth-updown-15m", 1, 0)

    def test_fee_labels_and_outcome_mismatches_are_counted(self) -> None:
        with Store(":memory:") as store:
            add(store, 0, [720])
            add(store, 1, [720], outcome=DOWN)  # prices in BASE imply Up
            add(store, 2, [720], start=SCHEDULE_2026_11_14_START - 900)
            (q,) = series_quality(store.connection)
        assert (q.fee_schedule_unknown, q.fee_schedule_known) == (1, 2)
        assert q.outcome_mismatches == 1
        assert (q.with_outcome, q.with_outcome_from_prices) == (3, 3)

    def test_empty_database(self) -> None:
        with Store(":memory:") as store:
            assert series_quality(store.connection) == []
            assert format_quality(store.connection) == "errors recorded (all runs)    0"


class TestErrorCounts:
    def test_by_stage_for_one_run_or_all(self) -> None:
        with Store(":memory:") as store:
            first = store.start_run(["btc-updown-15m"], started_at=1)
            second = store.start_run(["btc-updown-15m"], started_at=2)
            for run_id, stage in [(first, "settlement"), (first, "settlement"), (second, "parse")]:
                store.record_error(
                    run_id, stage=stage, message="x", occurred_at=3, market_slug=None
                )
            assert error_counts(store.connection) == {"settlement": 2, "parse": 1}
            assert error_counts(store.connection, first) == {"settlement": 2}
            assert "errors recorded (all runs)    3 (parse 1, settlement 2)" in format_quality(
                store.connection
            )
