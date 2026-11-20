"""The market context fetch: common completed sessions, validation, storage. No network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markets_lab.collectors.market_context import (
    CALENDAR_SERIES,
    HISTORY_START,
    SERIES,
    align_to_sessions,
    fetch_context,
    load_csv,
    main,
    normalize_close,
    split_complete,
    validate,
)
from markets_lab.collectors.nvda_prices import DroppedBar, PriceDataError, save_csv
from price_frames import LAST_SESSION, as_yahoo, at, make_prices

EVENING = at("2030-03-14 20:00")  # after the last synthetic session has completed
LAST = date(2030, 3, 14)
DAY_BEFORE = date(2030, 3, 13)
NAMES = list(SERIES)


def make_raws(rows: int = 300) -> dict[str, pd.DataFrame]:
    """One made-up Yahoo download per series, on the same sessions. The VIX frame
    is in Chicago time, as Yahoo publishes it."""
    raws: dict[str, pd.DataFrame] = {}
    for seed, name in enumerate(NAMES):
        raw = as_yahoo(make_prices(rows, seed=seed))
        if name == "vix":
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).tz_localize("America/Chicago")
        raws[name] = raw
    return raws


def expected_close(name: str, rows: int = 300) -> pd.Series:
    return make_prices(rows, seed=NAMES.index(name))["close"]


def downloader_for(raws: dict[str, pd.DataFrame]) -> Callable[[str], pd.DataFrame]:
    by_symbol = {SERIES[name].symbol: raw for name, raw in raws.items()}
    return lambda symbol: by_symbol[symbol]


def fetch(raws: dict[str, pd.DataFrame], now: datetime = EVENING, min_rows: int = 250):
    return fetch_context(now=now, downloader=downloader_for(raws), min_rows=min_rows)


def make_closes(rows: int = 300) -> pd.DataFrame:
    """A valid frame in the collector's shape."""
    return pd.DataFrame({name: expected_close(name, rows) for name in NAMES})


def without(raw: pd.DataFrame, position: int) -> pd.DataFrame:
    return raw.drop(raw.index[position])


class TestNormalizeClose:
    def test_keeps_every_value_and_the_session_dates(self) -> None:
        prices = make_prices()
        close = normalize_close(as_yahoo(prices))
        assert close.index.name == "date"
        assert pd.DatetimeIndex(close.index).tz is None
        assert close.index.equals(prices.index)
        np.testing.assert_array_equal(close.to_numpy(), prices["close"].to_numpy())

    def test_a_series_in_chicago_time_keeps_its_own_dates(self) -> None:
        # Converted to New York time, midnight in Chicago is 01:00 and would not
        # look like a daily bar. The date is read where the series is published.
        close = normalize_close(make_raws()["vix"])
        assert close.index[-1].date() == LAST
        assert close.index.equals(make_prices().index)

    def test_missing_close_column_raises(self) -> None:
        with pytest.raises(PriceDataError, match="Close"):
            normalize_close(as_yahoo(make_prices()).drop(columns=["Close"]))

    def test_intraday_timestamps_raise(self) -> None:
        raw = as_yahoo(make_prices())
        raw.index = raw.index + pd.Timedelta(hours=9, minutes=30)
        with pytest.raises(PriceDataError, match="not daily bars"):
            normalize_close(raw)

    def test_non_date_index_raises(self) -> None:
        with pytest.raises(PriceDataError, match="date index"):
            normalize_close(as_yahoo(make_prices()).reset_index(drop=True))

    def test_duplicate_or_unordered_dates_raise(self) -> None:
        raw = as_yahoo(make_prices())
        with pytest.raises(PriceDataError, match="duplicate dates"):
            normalize_close(pd.concat([raw, raw.iloc[[-1]]]))
        with pytest.raises(PriceDataError, match="increasing order"):
            normalize_close(raw.iloc[::-1])


class TestSplitComplete:
    def close(self) -> pd.Series:
        return normalize_close(as_yahoo(make_prices()))

    @pytest.mark.parametrize("clock", ["09:31", "15:59", "16:59"])
    def test_todays_row_is_dropped_until_the_session_is_complete(self, clock: str) -> None:
        kept, dropped = split_complete(self.close(), now=at(f"{LAST_SESSION} {clock}"))
        assert kept.index[-1].date() == DAY_BEFORE
        assert dropped == (DroppedBar(LAST, "in progress"),)

    def test_todays_row_is_kept_once_complete(self) -> None:
        kept, dropped = split_complete(self.close(), now=at(f"{LAST_SESSION} 17:00"))
        assert kept.index[-1].date() == LAST
        assert dropped == ()

    def test_a_missing_latest_close_is_dropped(self) -> None:
        close = self.close()
        close.iloc[-1] = np.nan
        kept, dropped = split_complete(close, now=EVENING)
        assert kept.index[-1].date() == DAY_BEFORE
        assert dropped == (DroppedBar(LAST, "incomplete"),)

    def test_a_missing_close_in_the_middle_is_not_dropped_here(self) -> None:
        close = self.close()
        close.iloc[100] = np.nan
        kept, dropped = split_complete(close, now=EVENING)
        assert len(kept) == len(close)
        assert dropped == ()

    def test_a_future_row_raises(self) -> None:
        with pytest.raises(PriceDataError, match="in the future"):
            split_complete(self.close(), now=at("2030-03-13 20:00"))

    def test_now_must_be_timezone_aware(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            split_complete(self.close(), now=datetime(2030, 3, 14, 20, 0))


class TestFetchContext:
    def test_returns_every_series_side_by_side_unchanged(self) -> None:
        result = fetch(make_raws())
        assert list(result.closes.columns) == NAMES
        assert result.closes.index.name == "date"
        assert result.last_session == LAST
        assert result.dropped == ()
        assert result.off_calendar == {}
        for name in NAMES:
            np.testing.assert_array_equal(
                result.closes[name].to_numpy(), expected_close(name).to_numpy()
            )

    def test_asks_yahoo_for_each_symbol(self) -> None:
        asked: list[str] = []
        by_symbol = {SERIES[name].symbol: raw for name, raw in make_raws().items()}

        def downloader(symbol: str) -> pd.DataFrame:
            asked.append(symbol)
            return by_symbol[symbol]

        fetch_context(now=EVENING, downloader=downloader, min_rows=250)
        assert asked == ["^GSPC", "^NDX", "^SOX", "^VIX"]

    def test_while_the_market_is_open_today_is_left_out_of_every_series(self) -> None:
        result = fetch(make_raws(), now=at(f"{LAST_SESSION} 11:00"))
        assert result.last_session == DAY_BEFORE
        assert result.dropped == (DroppedBar(LAST, "in progress"),)

    @pytest.mark.parametrize("late", NAMES)
    def test_a_session_one_series_has_not_published_is_left_out_of_all(self, late: str) -> None:
        # Whether Yahoo has no row yet or a row with a missing close, no series
        # may run ahead of another: a call would see part of a session.
        for change in ("no row", "missing close"):
            raws = make_raws()
            if change == "no row":
                raws[late] = raws[late].iloc[:-1]
            else:
                raws[late].loc[raws[late].index[-1], "Close"] = np.nan
            result = fetch(raws)
            assert result.last_session == DAY_BEFORE
            assert not result.closes.isna().any().any()
            assert result.dropped == (DroppedBar(LAST, f"incomplete: no completed row in {late}"),)

    def test_a_session_no_series_has_completed_names_them_all(self) -> None:
        raws = make_raws()
        for raw in raws.values():
            raw.loc[raw.index[-1], "Close"] = np.nan
        result = fetch(raws)
        assert result.last_session == DAY_BEFORE
        assert result.dropped == (
            DroppedBar(LAST, "incomplete: no completed row in sp500, nasdaq100, semis, vix"),
        )

    def test_rows_on_days_the_calendar_series_did_not_trade_are_left_out_and_counted(
        self,
    ) -> None:
        # The VIX has rows on some US holidays. The S&P 500's sessions decide.
        raws = make_raws()
        holiday = raws[CALENDAR_SERIES].index[100].date()
        for name in ("sp500", "nasdaq100", "semis"):
            raws[name] = without(raws[name], 100)
        result = fetch(raws)
        assert result.off_calendar == {"vix": 1}
        assert holiday not in {ts.date() for ts in result.closes.index}
        assert len(result.closes) == 299

    def test_a_calendar_session_missing_from_another_series_raises(self) -> None:
        raws = make_raws()
        hole = raws["semis"].index[100].date()
        raws["semis"] = without(raws["semis"], 100)
        with pytest.raises(PriceDataError, match="semis has no row for 1 sessions") as error:
            fetch(raws)
        assert str(hole) in str(error.value)

    def test_rows_before_the_history_start_are_not_kept(self) -> None:
        result = fetch(make_raws(rows=8_400))
        assert result.closes.index[0].date() == HISTORY_START
        assert len(result.closes) < 8_400

    def test_download_failure_names_the_series(self) -> None:
        def failing(_symbol: str) -> pd.DataFrame:
            raise RuntimeError("no network")

        with pytest.raises(PriceDataError, match=r"download of \^GSPC \(sp500\) failed"):
            fetch_context(now=EVENING, downloader=failing)

    def test_empty_download_raises(self) -> None:
        raws = make_raws()
        raws["vix"] = raws["vix"].iloc[:0]
        with pytest.raises(PriceDataError, match=r"\^VIX \(vix\) returned no rows"):
            fetch(raws)

    def test_a_problem_in_one_download_names_the_series(self) -> None:
        raws = make_raws()
        raws["nasdaq100"] = raws["nasdaq100"].drop(columns=["Close"])
        with pytest.raises(PriceDataError, match="nasdaq100: missing column"):
            fetch(raws)

    def test_truncated_history_raises_with_the_default_minimum(self) -> None:
        with pytest.raises(PriceDataError, match="only 300 rows, expected at least 6000"):
            fetch_context(now=EVENING, downloader=downloader_for(make_raws()))


class TestValidate:
    def check(self, closes: pd.DataFrame, now: datetime = EVENING) -> None:
        validate(closes, now=now, min_rows=250)

    def test_accepts_a_clean_history(self) -> None:
        self.check(make_closes())

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (np.nan, "semis: missing value on 1 rows"),
            (np.inf, "semis: infinite value on 1 rows"),
            (0.0, "semis: non-positive level on 1 rows"),
            (-5.0, "semis: non-positive level on 1 rows"),
        ],
    )
    def test_rejects_a_bad_level(self, value: float, message: str) -> None:
        closes = make_closes()
        closes.loc[closes.index[100], "semis"] = value
        with pytest.raises(PriceDataError, match=message):
            self.check(closes)

    def test_the_believable_daily_move_is_set_per_series(self) -> None:
        # The VIX has doubled in a day; a stock index has never moved 40%.
        doubled = make_closes()
        doubled.loc[doubled.index[100] :, "vix"] *= 2.0
        self.check(doubled)

        jumped = make_closes()
        jumped.loc[jumped.index[100] :, "sp500"] *= 1.4
        with pytest.raises(PriceDataError, match="sp500: moved more than 30% in a day"):
            self.check(jumped)

        tripled = make_closes()
        tripled.loc[tripled.index[100] :, "vix"] *= 3.5
        with pytest.raises(PriceDataError, match="vix: moved more than 200% in a day"):
            self.check(tripled)

    def test_rejects_wrong_columns(self) -> None:
        with pytest.raises(PriceDataError, match="columns are"):
            self.check(make_closes()[["vix", "sp500", "nasdaq100", "semis"]])

    def test_rejects_too_few_rows(self) -> None:
        with pytest.raises(PriceDataError, match="only 300 rows"):
            validate(make_closes(), now=EVENING, min_rows=301)

    def test_rejects_weekend_dates_and_long_gaps(self) -> None:
        closes = make_closes()
        weekend = closes.copy()
        weekend.index = weekend.index[:-1].append(pd.DatetimeIndex(["2030-03-16"]))
        with pytest.raises(PriceDataError, match="weekend date"):
            self.check(weekend, now=at("2030-03-16 20:00"))
        gap = pd.concat([closes.iloc[:100], closes.iloc[110:]])
        with pytest.raises(PriceDataError, match="gap of more than 7 days"):
            self.check(gap)

    def test_rejects_duplicates_and_disorder(self) -> None:
        closes = make_closes()
        with pytest.raises(PriceDataError, match="duplicate dates"):
            self.check(pd.concat([closes, closes.iloc[[-1]]]))
        with pytest.raises(PriceDataError, match="increasing order"):
            self.check(closes.iloc[::-1])

    def test_rejects_stale_and_future_data(self) -> None:
        with pytest.raises(PriceDataError, match="stale"):
            self.check(make_closes(), now=at("2030-03-25 20:00"))
        with pytest.raises(PriceDataError, match="in the future"):
            self.check(make_closes(), now=at("2030-03-13 20:00"))

    def test_reports_several_problems_at_once(self) -> None:
        closes = make_closes()
        closes.loc[closes.index[50], "sp500"] = np.nan
        closes.loc[closes.index[60], "vix"] = -1.0
        with pytest.raises(PriceDataError) as error:
            self.check(closes)
        assert "sp500: missing value" in str(error.value)
        assert "vix: non-positive level" in str(error.value)


class TestAlignToSessions:
    def test_returns_exactly_the_sessions_asked_for(self) -> None:
        closes = make_closes()
        sessions = pd.DatetimeIndex(closes.index[50:])
        aligned = align_to_sessions(closes, sessions)
        assert aligned.index.equals(sessions)
        pd.testing.assert_frame_equal(aligned, closes.iloc[50:])

    def test_a_missing_session_raises_instead_of_being_filled(self) -> None:
        closes = make_closes()
        sessions = pd.DatetimeIndex(closes.index)
        holed = closes.drop(closes.index[100])
        with pytest.raises(PriceDataError, match="no row for 1 sessions") as error:
            align_to_sessions(holed, sessions)
        assert str(closes.index[100].date()) in str(error.value)

    def test_a_session_after_the_last_context_row_raises(self) -> None:
        # The case the nightly job must refuse: NVDA's close is in, the context's is not.
        closes = make_closes()
        with pytest.raises(PriceDataError, match="last context session: 2030-03-13"):
            align_to_sessions(closes.iloc[:-1], pd.DatetimeIndex(closes.index))


class TestStorage:
    def test_round_trip_is_exact(self, tmp_path: Path) -> None:
        closes = make_closes()
        path = tmp_path / "context.csv"
        save_csv(closes, path)
        loaded = load_csv(path)
        pd.testing.assert_frame_equal(loaded, closes, check_freq=False)
        validate(loaded, now=EVENING, min_rows=250)

    def test_load_rejects_a_file_without_the_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "context.csv"
        save_csv(make_closes().drop(columns=["vix"]), path)
        with pytest.raises(PriceDataError, match="missing columns"):
            load_csv(path)


class TestCommandLine:
    def run_main(
        self, raws: dict[str, pd.DataFrame], out: Path, now: datetime = EVENING
    ) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = main(
            ["--out", str(out)],
            downloader=downloader_for(raws),
            now=lambda: now,
            report=lines.append,
        )
        return code, lines

    def test_saves_and_reports_what_was_left_out(self, tmp_path: Path) -> None:
        raws = make_raws(rows=6_100)
        for name in ("sp500", "nasdaq100", "semis"):
            raws[name] = without(raws[name], 3_000)
        raws["semis"].loc[raws["semis"].index[-1], "Close"] = np.nan
        out = tmp_path / "context.csv"
        code, lines = self.run_main(raws, out)
        assert code == 0
        assert lines[0].startswith("saved 6098 sessions of sp500, nasdaq100, semis, vix, ")
        assert lines[0].endswith(f"to 2030-03-13, to {out}")
        assert lines[1] == "left out 1 vix rows on days with no sp500 session"
        assert lines[2] == "left out 2030-03-14: incomplete: no completed row in semis"
        assert load_csv(out).index[-1].date() == DAY_BEFORE

    def test_invalid_data_is_not_saved(self, tmp_path: Path) -> None:
        out = tmp_path / "context.csv"
        code, lines = self.run_main(make_raws(), out)
        assert code == 1
        assert lines == [
            "error: market context failed validation: only 300 rows, expected at least 6000"
        ]
        assert not out.exists()

    def test_refuses_to_replace_a_file_with_a_shorter_history(self, tmp_path: Path) -> None:
        out = tmp_path / "context.csv"
        save_csv(make_closes(rows=6_100), out)
        shorter = {name: without(raw, 3_000) for name, raw in make_raws(rows=6_100).items()}
        code, lines = self.run_main(shorter, out)
        assert code == 1
        assert "missing 1 stored sessions" in lines[0]
        assert len(load_csv(out)) == 6_100

    def test_missing_directory_is_refused(self, tmp_path: Path) -> None:
        code, lines = self.run_main(make_raws(), tmp_path / "nope" / "context.csv")
        assert code == 2
        assert "does not exist" in lines[0]
