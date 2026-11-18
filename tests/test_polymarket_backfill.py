"""The backfill run end to end against a fake gateway. No network access."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import httpx
import pytest

from markets_lab.collectors.polymarket import backfill
from markets_lab.collectors.polymarket.backfill import (
    MAX_CONSECUTIVE_ERRORS,
    SETTLE_GRACE_SECONDS,
    BackfillAbortedError,
    RunStats,
    format_duration,
    format_summary,
    main,
    run_backfill,
)
from markets_lab.collectors.polymarket.client import PolymarketClient
from markets_lab.collectors.polymarket.parse import DOWN, UP
from markets_lab.collectors.polymarket.store import Store
from polymarket_fake_gateway import NOW, T0, FakeGateway, FakeTime

BTC = "btc-updown-15m"
ETH = "eth-updown-15m"


def count(store: Store, table: str, where: str = "1") -> int:
    return store.connection.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()[0]


def run(
    gateway: FakeGateway,
    store: Store,
    *,
    fake_time: FakeTime | None = None,
    lines: list[str] | None = None,
    **kwargs: Any,
) -> RunStats:
    fake_time = fake_time or FakeTime()
    kwargs.setdefault("series_slugs", [BTC])
    sink = lines if lines is not None else []
    with gateway.client(fake_time) as client:
        return run_backfill(client, store, clock=fake_time.clock, report=sink.append, **kwargs)


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


class TestHappyPath:
    def test_backfills_every_settled_window_of_every_series(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index, settled_up=index != 1)
        gateway.add_window(ETH, 0)
        stats = run(gateway, store, series_slugs=[BTC, ETH])

        assert stats.outcome == "finished"
        assert (stats.windows_seen, stats.selected) == (4, 4)
        assert (stats.histories_fetched, stats.outcomes_recorded) == (4, 4)
        assert stats.points_added == 4 * 16
        assert stats.error_count == 0
        assert count(store, "markets") == 4
        assert count(store, "price_points") == 64
        assert len(store.complete_market_slugs()) == 4
        outcomes = store.connection.execute(
            "SELECT outcome FROM markets WHERE series_slug = ? ORDER BY window_start", (BTC,)
        ).fetchall()
        assert [row[0] for row in outcomes] == [UP, DOWN, UP]

    def test_all_histories_are_fetched_before_any_settlement(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
        run(gateway, store)
        kinds = [
            "history" if "price-history" in path else "settlement"
            for path in gateway.paths()
            if "price-history" in path or "settlement" in path
        ]
        assert kinds == ["history"] * 3 + ["settlement"] * 3

    def test_settlements_are_requested_at_their_own_slower_rate(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(4):
            gateway.add_window(BTC, index)
        fake_time = FakeTime()
        arrivals: dict[str, list[float]] = {"history": [], "settlement": []}
        original = gateway.__call__

        def timed(request: httpx.Request) -> httpx.Response:
            for kind in ("history", "settlement"):
                if kind in request.url.path:
                    arrivals[kind].append(fake_time.now)
            return original(request)

        with PolymarketClient(
            transport=httpx.MockTransport(timed), sleep=fake_time.sleep, clock=fake_time.clock
        ) as client:
            run_backfill(
                client,
                store,
                series_slugs=[BTC],
                history_rps=4,
                settlement_rps=0.25,
                clock=fake_time.clock,
                report=lambda _: None,
            )

        def gaps(times: list[float]) -> list[float]:
            return [round(b - a, 6) for a, b in itertools.pairwise(times)]

        assert gaps(arrivals["history"]) == [0.25, 0.25, 0.25]
        assert gaps(arrivals["settlement"]) == [4.0, 4.0, 4.0]

    def test_rows_record_when_they_were_fetched(self, gateway: FakeGateway, store: Store) -> None:
        gateway.add_window(BTC, 0)
        run(gateway, store)
        row = store.connection.execute(
            "SELECT min(snapshot_ts), max(snapshot_ts), min(fetched_at) FROM price_points"
        ).fetchone()
        assert (row[0], row[1]) == (T0, T0 + 900)
        assert row[2] >= NOW

    def test_history_is_requested_around_the_window(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        run(gateway, store)
        (request,) = [r for r in gateway.requests if r.url.path == "/v1/price-history"]
        assert request.url.params["timestamp.startTimestamp"] == str(T0 - 120)
        assert request.url.params["timestamp.endTimestamp"] == str(T0 + 900 + 60)

    def test_run_is_logged(self, gateway: FakeGateway, store: Store) -> None:
        gateway.add_window(BTC, 0)
        stats = run(gateway, store)
        row = store.connection.execute("SELECT * FROM backfill_runs").fetchone()
        assert row["run_id"] == stats.run_id
        assert row["finished_at"] is not None
        assert row["summary"].startswith("finished: windows seen 1")

    def test_only_gets_are_sent_and_only_to_the_gateway(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        run(gateway, store)
        assert {request.method for request in gateway.requests} == {"GET"}
        assert {request.url.host for request in gateway.requests} == {"gateway.polymarket.us"}


class TestWhichWindowsAreTaken:
    def test_unresolved_and_future_windows_are_skipped(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        gateway.add_window(BTC, 1, status="MARKET_STATUS_CLOSED")
        gateway.add_window(BTC, 2, status="MARKET_STATUS_OPEN")
        stats = run(gateway, store)
        assert (stats.windows_seen, stats.not_settled_yet, stats.selected) == (3, 2, 1)
        assert count(store, "markets") == 1
        assert gateway.count("/price-history") == 1

    def test_a_window_that_only_just_ended_waits_for_a_later_run(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        just_ended = NOW - 900 - SETTLE_GRACE_SECONDS + 1
        gateway.add_window(BTC, 0, window_start=just_ended)
        assert run(gateway, store).not_settled_yet == 1
        later = FakeTime(NOW + SETTLE_GRACE_SECONDS)
        assert run(gateway, store, fake_time=later).histories_fetched == 1

    def test_unknown_series_stops_the_run_loudly(self, gateway: FakeGateway, store: Store) -> None:
        with pytest.raises(Exception, match="series not found"):
            run(gateway, store, series_slugs=["doge-updown-15m"])
        row = store.connection.execute("SELECT summary, finished_at FROM backfill_runs").fetchone()
        assert row["summary"].startswith("aborted")
        assert row["finished_at"] is not None
        assert count(store, "backfill_errors", "stage = 'listing'") == 1


class TestResuming:
    def test_second_run_fetches_nothing_and_adds_nothing(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
        run(gateway, store)
        gateway.requests.clear()
        stats = run(gateway, store)
        assert (stats.already_complete, stats.selected, stats.points_added) == (3, 0, 0)
        assert gateway.count("/price-history") == 0
        assert gateway.count("/settlement") == 0
        assert count(store, "price_points") == 48

    def test_no_settlements_then_a_later_run_fills_them_in(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
        first = run(gateway, store, fetch_settlements=False)
        assert (first.histories_fetched, first.outcomes_recorded) == (3, 0)
        assert gateway.count("/settlement") == 0
        assert store.complete_market_slugs() == set()
        # The price-implied outcome is already there from the listing.
        assert count(store, "markets", "outcome_from_prices IS NOT NULL") == 3

        gateway.requests.clear()
        second = run(gateway, store)
        assert (second.histories_fetched, second.outcomes_recorded) == (0, 3)
        assert gateway.count("/price-history") == 0
        assert len(store.complete_market_slugs()) == 3
        assert count(store, "price_points") == 48

    def test_limit_takes_the_first_n_and_the_next_run_continues(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(5):
            gateway.add_window(BTC, index)
        first = run(gateway, store, limit=2)
        assert (first.selected, first.histories_fetched, first.outcomes_recorded) == (2, 2, 2)
        assert count(store, "markets") == 2
        second = run(gateway, store, limit=2)
        assert (second.already_complete, second.histories_fetched) == (2, 2)
        third = run(gateway, store)
        assert (third.already_complete, third.histories_fetched) == (4, 1)
        assert len(store.complete_market_slugs()) == 5
        assert count(store, "price_points") == 5 * 16

    def test_limit_spans_series_and_stops_listing_early(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
            gateway.add_window(ETH, index)
        stats = run(gateway, store, series_slugs=[BTC, ETH], limit=3)
        assert stats.histories_fetched == 3
        assert count(store, "markets", f"series_slug = '{ETH}'") == 0
        assert gateway.count("/v1/series") == 1

    def test_limit_must_be_positive(self, gateway: FakeGateway, store: Store) -> None:
        with pytest.raises(ValueError, match="limit"):
            run(gateway, store, limit=0)

    def test_interrupting_then_rerunning_leaves_no_duplicates(
        self, gateway: FakeGateway, tmp_path: Path
    ) -> None:
        slugs = [gateway.add_window(BTC, index) for index in range(4)]
        path = tmp_path / "pm.sqlite"
        gateway.interrupt_on_history_of = slugs[2]
        with Store(path) as store:
            with pytest.raises(KeyboardInterrupt):
                run(gateway, store)
            assert len(store.market_slugs_with_history()) == 2
            summary = store.connection.execute("SELECT summary FROM backfill_runs").fetchone()[0]
            assert summary.startswith("interrupted")
        gateway.interrupt_on_history_of = None
        gateway.requests.clear()
        with Store(path) as store:
            stats = run(gateway, store)
            assert (stats.histories_fetched, stats.outcomes_recorded) == (2, 4)
            assert gateway.count("/price-history") == 2
            assert count(store, "markets") == 4
            assert count(store, "price_points") == 4 * 16
            assert len(store.complete_market_slugs()) == 4


class TestProblemsAreRecordedNotHidden:
    def test_missing_settlement_is_an_error_and_is_retried_next_run(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        slug = gateway.add_window(BTC, 1, settlement=None)
        stats = run(gateway, store)
        assert (stats.outcomes_recorded, dict(stats.errors)) == (1, {"settlement": 1})
        assert not store.is_complete(slug)
        error = store.connection.execute("SELECT * FROM backfill_errors").fetchone()
        assert (error["market_slug"], error["stage"]) == (slug, "settlement")

        gateway.settlements[slug] = 1
        gateway.requests.clear()
        assert run(gateway, store).outcomes_recorded == 1
        assert store.is_complete(slug)
        assert gateway.count("/price-history") == 0
        assert count(store, "price_points") == 32

    def test_outcome_contradicting_the_prices_is_recorded_with_both_values_kept(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        slug = gateway.add_window(BTC, 0, settled_up=True, settlement=0)
        stats = run(gateway, store)
        assert dict(stats.errors) == {"outcome-mismatch": 1}
        assert stats.outcomes_recorded == 1
        row = store.connection.execute(
            "SELECT outcome, outcome_from_prices FROM markets WHERE market_slug = ?", (slug,)
        ).fetchone()
        assert tuple(row) == (DOWN, UP)

    def test_a_failed_history_fetch_skips_that_window_only(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        slugs = [gateway.add_window(BTC, index) for index in range(3)]
        gateway.broken_history.add(slugs[1])
        stats = run(gateway, store)
        assert (stats.selected, stats.histories_fetched) == (3, 2)
        assert dict(stats.errors) == {"price-history": 1}
        assert store.market_slugs_with_history() == {slugs[0], slugs[2]}

    def test_out_of_range_price_is_an_error_not_a_silent_skip(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        slug = gateway.add_window(BTC, 0, points=[(T0 + 720, 1.5, 0.1)])
        stats = run(gateway, store)
        assert dict(stats.errors) == {"price-history": 1}
        assert not store.is_complete(slug)
        assert count(store, "price_points") == 0

    def test_a_malformed_event_is_recorded_and_the_rest_continue(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        gateway.add_window(BTC, 1)
        series_id = gateway.series_ids[BTC]
        del gateway.events[series_id][0]["markets"][0]["assetPriceTerms"]
        stats = run(gateway, store)
        assert dict(stats.errors) == {"parse-event": 1}
        assert stats.histories_fetched == 1

    def test_many_failures_in_a_row_abort_the_run(self, gateway: FakeGateway, store: Store) -> None:
        for index in range(MAX_CONSECUTIVE_ERRORS + 5):
            gateway.broken_history.add(gateway.add_window(BTC, index))
        with pytest.raises(BackfillAbortedError, match="in a row failed"):
            run(gateway, store)
        assert gateway.count("/price-history") == MAX_CONSECUTIVE_ERRORS
        summary = store.connection.execute("SELECT summary FROM backfill_runs").fetchone()[0]
        assert summary.startswith("aborted")
        assert count(store, "backfill_errors") == MAX_CONSECUTIVE_ERRORS

    def test_scattered_failures_do_not_abort(self, gateway: FakeGateway, store: Store) -> None:
        for index in range(3 * MAX_CONSECUTIVE_ERRORS):
            slug = gateway.add_window(BTC, index)
            if index % 2:
                gateway.broken_history.add(slug)
        stats = run(gateway, store)
        assert stats.outcome == "finished"
        assert stats.errors["price-history"] == 3 * MAX_CONSECUTIVE_ERRORS // 2

    def test_rate_limit_responses_are_counted_and_shown(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
        gateway.settlement_429s = 2
        lines: list[str] = []
        stats = run(gateway, store, lines=lines)
        assert stats.rate_limited == 2
        assert stats.outcomes_recorded == 3
        assert stats.error_count == 0
        assert "429s 2" in lines[-1]
        assert (
            "429s 2" in store.connection.execute("SELECT summary FROM backfill_runs").fetchone()[0]
        )


class TestReporting:
    def test_progress_lines_show_counts_and_eta(self, gateway: FakeGateway, store: Store) -> None:
        for index in range(8):
            gateway.add_window(BTC, index)
        lines: list[str] = []
        # The fake clock advances 0.25 s per history request, so 0.9 s is a few windows.
        run(gateway, store, lines=lines, progress_seconds=0.9, fetch_settlements=False)
        progress = [line for line in lines if "ETA" in line]
        assert len(progress) >= 2
        assert progress[-1].startswith("histories 8/8 (100.0%)")
        assert "ETA 0s" in progress[-1]
        assert all("elapsed" in line and "errors 0" in line for line in progress)
        assert any(line.startswith("histories: fetching 8 at 4 requests/second") for line in lines)

    def test_each_stage_reports_its_own_progress(self, gateway: FakeGateway, store: Store) -> None:
        for index in range(3):
            gateway.add_window(BTC, index)
        lines: list[str] = []
        run(gateway, store, lines=lines)
        assert any(line.startswith("histories 3/3") for line in lines)
        assert any(
            line.startswith("settlements: fetching 3 at 0.25 requests/second") for line in lines
        )
        assert any(line.startswith("settlements 3/3") for line in lines)

    def test_nothing_to_do_says_so(self, gateway: FakeGateway, store: Store) -> None:
        gateway.add_window(BTC, 0)
        run(gateway, store)
        lines: list[str] = []
        run(gateway, store, lines=lines)
        assert "histories: nothing to fetch" in lines
        assert "settlements: nothing to fetch" in lines

    def test_summary_covers_the_run_and_the_database(
        self, gateway: FakeGateway, store: Store
    ) -> None:
        gateway.add_window(BTC, 0)
        gateway.add_window(BTC, 1, settlement=None)
        text = format_summary(run(gateway, store), store)
        assert "finished" in text
        assert "price histories fetched     2" in text
        assert "settlement outcomes fetched 1" in text
        assert "errors this run             1 (settlement 1)" in text
        assert "btc-updown-15m" in text
        assert "has neither                 0" in text
        assert "outcome from index prices   2" in text

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0s"), (45.4, "45s"), (60, "1m00s"), (725, "12m05s"), (7_380, "2h03m"), (-5, "0s")],
    )
    def test_format_duration(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected


class TestCommandLine:
    def run_main(self, gateway: FakeGateway, argv: list[str]) -> tuple[int, list[str]]:
        fake_time = FakeTime()

        def factory(**kwargs: Any) -> PolymarketClient:
            return gateway.client(fake_time, **kwargs)

        lines: list[str] = []
        code = main(argv, client_factory=factory, report=lines.append, clock=fake_time.clock)
        return code, lines

    def test_limit_and_db_options(self, gateway: FakeGateway, tmp_path: Path) -> None:
        for index in range(4):
            gateway.add_window(BTC, index)
        db = tmp_path / "trial.sqlite"
        code, lines = self.run_main(gateway, ["--db", str(db), "--series", BTC, "--limit", "3"])
        assert code == 0
        output = "\n".join(lines)
        assert "price histories fetched     3" in output
        assert "=== database ===" in output
        with Store(db) as store:
            assert len(store.complete_market_slugs()) == 3

    def test_no_settlements_flag(self, gateway: FakeGateway, tmp_path: Path) -> None:
        gateway.add_window(BTC, 0)
        code, lines = self.run_main(
            gateway, ["--db", str(tmp_path / "x.sqlite"), "--series", BTC, "--no-settlements"]
        )
        assert code == 0
        assert gateway.count("/settlement") == 0
        assert "settlements skipped (--no-settlements)" in lines

    def test_recorded_errors_give_a_nonzero_exit_code(
        self, gateway: FakeGateway, tmp_path: Path
    ) -> None:
        gateway.add_window(BTC, 0, settlement=None)
        code, lines = self.run_main(gateway, ["--db", str(tmp_path / "x.sqlite"), "--series", BTC])
        assert code == 3
        assert any("errors this run             1" in line for line in lines)

    def test_interrupt_prints_the_summary_and_exits_130(
        self, gateway: FakeGateway, tmp_path: Path
    ) -> None:
        slugs = [gateway.add_window(BTC, index) for index in range(3)]
        gateway.interrupt_on_history_of = slugs[1]
        code, lines = self.run_main(gateway, ["--db", str(tmp_path / "x.sqlite"), "--series", BTC])
        assert code == 130
        output = "\n".join(lines)
        assert "a rerun will continue from here" in output
        assert "interrupted ===" in output
        assert "price histories fetched     1" in output

    def test_abort_is_reported_with_exit_code_1(self, gateway: FakeGateway, tmp_path: Path) -> None:
        code, lines = self.run_main(
            gateway, ["--db", str(tmp_path / "x.sqlite"), "--series", "nope-15m"]
        )
        assert code == 1
        assert any(line.startswith("error:") and "series not found" in line for line in lines)

    def test_missing_data_directory_is_refused(self, gateway: FakeGateway, tmp_path: Path) -> None:
        code, lines = self.run_main(gateway, ["--db", str(tmp_path / "missing" / "x.sqlite")])
        assert code == 2
        assert "does not exist" in lines[0]

    @pytest.mark.parametrize(
        "argv",
        [["--limit", "0"], ["--limit", "-3"], ["--history-rps", "0"], ["--settlement-rps", "-1"]],
    )
    def test_nonsense_options_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            backfill.build_parser().parse_args(argv)

    def test_defaults(self) -> None:
        args = backfill.build_parser().parse_args([])
        assert args.limit is None
        assert args.no_settlements is False
        assert args.db == Path("data/polymarket.sqlite")
        assert args.series == [BTC, ETH, "sol-updown-15m", "xrp-updown-15m"]
        assert (args.history_rps, args.settlement_rps) == (4.0, 0.25)
