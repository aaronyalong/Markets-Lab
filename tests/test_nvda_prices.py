"""The NVDA price fetch: completed sessions only, validation, storage. No network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markets_lab.collectors.nvda_prices import (
    COLUMNS,
    DroppedBar,
    PriceDataError,
    check_history_kept,
    fetch_daily,
    load_csv,
    main,
    normalize,
    save_csv,
    split_complete,
    validate,
)
from price_frames import LAST_SESSION, as_yahoo, at, make_prices, with_extra_row

EVENING = at("2030-03-14 20:00")  # after the last synthetic session has completed
NEXT_DAY = "2030-03-15"  # a Friday


def downloader_for(raw: pd.DataFrame) -> Callable[[str], pd.DataFrame]:
    return lambda _symbol: raw


class TestNormalize:
    def test_reshapes_without_changing_values(self) -> None:
        prices = make_prices()
        raw = as_yahoo(prices)
        raw["Dividends"] = 0.0
        result = normalize(raw)
        assert list(result.columns) == list(COLUMNS)
        assert result.index.name == "date"
        assert pd.DatetimeIndex(result.index).tz is None
        assert result.index[-1].date() == date(2030, 3, 14)
        pd.testing.assert_frame_equal(result, prices, check_freq=False)

    def test_dates_are_new_york_sessions_whatever_the_input_zone(self) -> None:
        raw = as_yahoo(make_prices())
        raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
        assert normalize(raw).index[-1].date() == date(2030, 3, 14)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(PriceDataError, match="missing columns"):
            normalize(as_yahoo(make_prices()).drop(columns=["Adj Close"]))

    def test_intraday_timestamps_raise(self) -> None:
        raw = as_yahoo(make_prices())
        raw.index = raw.index + pd.Timedelta(hours=9, minutes=30)
        with pytest.raises(PriceDataError, match="not daily bars"):
            normalize(raw)

    @pytest.mark.parametrize("bad_volume", [np.nan, 1234.5])
    def test_bad_volume_raises(self, bad_volume: float) -> None:
        raw = as_yahoo(make_prices())
        raw["Volume"] = raw["Volume"].astype("float64")
        raw.loc[raw.index[5], "Volume"] = bad_volume
        with pytest.raises(PriceDataError, match="volume"):
            normalize(raw)

    def test_non_date_index_raises(self) -> None:
        raw = as_yahoo(make_prices()).reset_index(drop=True)
        with pytest.raises(PriceDataError, match="date index"):
            normalize(raw)


class TestSplitComplete:
    """The latest row is only kept once its session has finished."""

    @pytest.mark.parametrize("clock", ["09:31", "11:00", "15:59", "16:00", "16:59"])
    def test_todays_bar_is_dropped_until_the_session_is_complete(self, clock: str) -> None:
        history = make_prices()
        partial = with_extra_row(history, NEXT_DAY, close=1.0, high=999.0)
        kept, dropped = split_complete(partial, now=at(f"{NEXT_DAY} {clock}"))
        assert dropped == (DroppedBar(date(2030, 3, 15), "in progress"),)
        # Nothing from the partial bar survives: the result is exactly the history.
        pd.testing.assert_frame_equal(kept, history, check_freq=False)

    @pytest.mark.parametrize("clock", ["17:00", "20:00", "23:59"])
    def test_todays_bar_is_kept_once_complete(self, clock: str) -> None:
        full = with_extra_row(make_prices(), NEXT_DAY)
        kept, dropped = split_complete(full, now=at(f"{NEXT_DAY} {clock}"))
        assert dropped == ()
        assert kept.index[-1].date() == date(2030, 3, 15)

    def test_the_new_york_date_decides_not_the_utc_date(self) -> None:
        # 01:00 UTC on the 16th is 21:00 on the 15th in New York.
        full = with_extra_row(make_prices(), NEXT_DAY)
        now = datetime(2030, 3, 16, 1, 0, tzinfo=UTC)
        kept, dropped = split_complete(full, now=now)
        assert dropped == ()
        assert kept.index[-1].date() == date(2030, 3, 15)
        # 14:00 UTC on the 15th is 10:00 in New York: still in progress.
        _, dropped = split_complete(full, now=datetime(2030, 3, 15, 14, 0, tzinfo=UTC))
        assert dropped == (DroppedBar(date(2030, 3, 15), "in progress"),)

    @pytest.mark.parametrize("column", ["close", "adj_close", "open"])
    def test_latest_bar_with_a_missing_price_is_dropped_after_the_close(self, column: str) -> None:
        # Seen in real data: hours after the close, the latest row has no close price.
        history = make_prices()
        unpublished = with_extra_row(history, NEXT_DAY, **{column: np.nan})
        kept, dropped = split_complete(unpublished, now=at(f"{NEXT_DAY} 20:57"))
        assert dropped == (DroppedBar(date(2030, 3, 15), "incomplete"),)
        pd.testing.assert_frame_equal(kept, history, check_freq=False)

    def test_in_progress_today_and_incomplete_yesterday_are_both_dropped(self) -> None:
        history = make_prices(end="2030-03-13")
        frame = with_extra_row(history, LAST_SESSION, close=np.nan)
        frame = with_extra_row(frame, NEXT_DAY, close=5.0)
        kept, dropped = split_complete(frame, now=at(f"{NEXT_DAY} 10:00"))
        assert [bar.reason for bar in dropped] == ["in progress", "incomplete"]
        assert kept.index[-1].date() == date(2030, 3, 13)

    def test_only_one_incomplete_row_is_tolerated(self) -> None:
        frame = with_extra_row(make_prices(end="2030-03-13"), LAST_SESSION, close=np.nan)
        frame = with_extra_row(frame, NEXT_DAY, close=np.nan)
        kept, dropped = split_complete(frame, now=at(f"{NEXT_DAY} 20:00"))
        assert len(dropped) == 1
        with pytest.raises(PriceDataError, match="missing value"):
            validate(kept, now=at(f"{NEXT_DAY} 20:00"), min_rows=100)

    def test_a_missing_price_in_the_middle_is_not_dropped_here(self) -> None:
        frame = make_prices()
        frame.loc[frame.index[100], "close"] = np.nan
        kept, dropped = split_complete(frame, now=EVENING)
        assert dropped == ()
        assert len(kept) == len(frame)

    def test_a_future_row_raises(self) -> None:
        frame = with_extra_row(make_prices(), "2030-03-18")
        with pytest.raises(PriceDataError, match="in the future"):
            split_complete(frame, now=EVENING)

    def test_now_must_be_timezone_aware(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            split_complete(make_prices(), now=datetime(2030, 3, 14, 20, 0))


def break_frame(change: Callable[[pd.DataFrame], pd.DataFrame | None]) -> pd.DataFrame:
    frame = make_prices()
    changed = change(frame)
    return frame if changed is None else changed


def set_cell(row: int, column: str, value: float) -> Callable[[pd.DataFrame], None]:
    def change(frame: pd.DataFrame) -> None:
        if column == "volume":
            frame[column] = frame[column].astype("int64")
        frame.loc[frame.index[row], column] = value

    return change


def scale_from(row: int, factor: float) -> Callable[[pd.DataFrame], None]:
    def change(frame: pd.DataFrame) -> None:
        columns = ["open", "high", "low", "close", "adj_close"]
        frame.loc[frame.index[row] :, columns] *= factor

    return change


class TestValidate:
    def test_accepts_a_clean_history(self) -> None:
        validate(make_prices(), now=EVENING, min_rows=100)

    def test_real_sized_default_minimum(self) -> None:
        with pytest.raises(PriceDataError, match="only 300 rows"):
            validate(make_prices(), now=EVENING)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (set_cell(50, "close", np.nan), "missing value"),
            (set_cell(50, "low", -1.0), "non-positive price"),
            (set_cell(50, "open", 0.0), "non-positive price"),
            (set_cell(50, "volume", 0), "non-positive volume"),
            (set_cell(50, "high", 0.01), "high below open/close/low"),
            (set_cell(50, "low", 1e6), "low above open/close/high"),
            (set_cell(50, "adj_close", 1e6), "adjusted close above close"),
            (scale_from(150, 0.1), "moved more than 60% in a day"),
            (scale_from(150, 4.0), "moved more than 60% in a day"),
            (lambda f: f.iloc[::-1], "not in increasing order"),
            (lambda f: pd.concat([f.iloc[:10], f.iloc[9:]]), "duplicate dates"),
            (lambda f: pd.concat([f.iloc[:100], f.iloc[110:]]), "gap of more than 7 days"),
            (lambda f: f.rename(columns={"close": "Close"}), "columns are"),
            (lambda f: f.iloc[:, ::-1], "columns are"),
            (lambda f: f.tz_localize("UTC"), "timezone-free"),
        ],
    )
    def test_rejects(
        self, change: Callable[[pd.DataFrame], pd.DataFrame | None], message: str
    ) -> None:
        with pytest.raises(PriceDataError, match=message):
            validate(break_frame(change), now=EVENING, min_rows=100)

    def test_rejects_weekend_dates(self) -> None:
        frame = make_prices()
        shifted = frame.index.to_series().mask(
            frame.index == "2030-03-08", pd.Timestamp("2030-03-09")
        )
        frame.index = pd.DatetimeIndex(shifted, name="date")
        with pytest.raises(PriceDataError, match="weekend date"):
            validate(frame, now=EVENING, min_rows=100)

    def test_error_names_how_many_rows_and_the_first_date(self) -> None:
        frame = break_frame(set_cell(-3, "close", np.nan))
        frame.loc[frame.index[-2], "close"] = np.nan
        with pytest.raises(PriceDataError, match=r"missing value on 2 rows, first 2030-03-12"):
            validate(frame, now=EVENING, min_rows=100)

    def test_reports_several_problems_at_once(self) -> None:
        frame = break_frame(set_cell(50, "volume", 0))
        frame.loc[frame.index[60], "low"] = -1.0
        with pytest.raises(PriceDataError) as caught:
            validate(frame, now=EVENING, min_rows=100)
        assert "non-positive volume" in str(caught.value)
        assert "non-positive price" in str(caught.value)

    def test_stale_data_is_rejected(self) -> None:
        validate(make_prices(), now=at("2030-03-19 09:00"), min_rows=100)  # 5 days: allowed
        with pytest.raises(PriceDataError, match="stale: last completed session is 2030-03-14"):
            validate(make_prices(), now=at("2030-03-20 09:00"), min_rows=100)

    def test_a_last_row_after_today_is_rejected(self) -> None:
        with pytest.raises(PriceDataError, match="in the future"):
            validate(make_prices(), now=at("2030-03-13 20:00"), min_rows=100)


class TestFetchDaily:
    def test_returns_validated_completed_sessions(self) -> None:
        history = make_prices()
        raw = as_yahoo(with_extra_row(history, NEXT_DAY, close=1.0))
        result = fetch_daily(
            now=at(f"{NEXT_DAY} 12:00"), downloader=downloader_for(raw), min_rows=100
        )
        assert result.last_session == date(2030, 3, 14)
        assert result.dropped == (DroppedBar(date(2030, 3, 15), "in progress"),)
        pd.testing.assert_frame_equal(result.prices, history, check_freq=False)

    def test_download_failure_is_reported_as_a_data_error(self) -> None:
        def broken(_symbol: str) -> pd.DataFrame:
            raise ConnectionError("no route")

        with pytest.raises(PriceDataError, match="download of NVDA failed: no route"):
            fetch_daily(now=EVENING, downloader=broken)

    def test_empty_download_raises(self) -> None:
        empty = as_yahoo(make_prices()).iloc[:0]
        with pytest.raises(PriceDataError, match="returned no rows"):
            fetch_daily(now=EVENING, downloader=downloader_for(empty))

    def test_invalid_data_raises(self) -> None:
        raw = as_yahoo(break_frame(set_cell(50, "close", np.nan)))
        with pytest.raises(PriceDataError, match="missing value"):
            fetch_daily(now=EVENING, downloader=downloader_for(raw), min_rows=100)

    def test_truncated_history_raises_with_the_default_minimum(self) -> None:
        with pytest.raises(PriceDataError, match="only 300 rows"):
            fetch_daily(now=EVENING, downloader=downloader_for(as_yahoo(make_prices())))


class TestStorage:
    def test_round_trip_is_exact(self, tmp_path: Path) -> None:
        prices = make_prices()
        path = tmp_path / "nvda.csv"
        save_csv(prices, path)
        loaded = load_csv(path)
        pd.testing.assert_frame_equal(loaded, prices, check_freq=False, check_exact=True)
        validate(loaded, now=EVENING, min_rows=100)
        assert [p.name for p in tmp_path.iterdir()] == ["nvda.csv"]

    def test_load_rejects_a_file_without_the_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "nvda.csv"
        make_prices().drop(columns=["adj_close"]).to_csv(path)
        with pytest.raises(PriceDataError, match="missing columns"):
            load_csv(path)

    def test_history_must_not_shrink(self) -> None:
        previous = make_prices()
        check_history_kept(previous, with_extra_row(previous, NEXT_DAY))
        with pytest.raises(PriceDataError, match=r"missing 2 stored sessions, e.g. 2030-03-13"):
            check_history_kept(previous, previous.iloc[:-2])


class TestCommandLine:
    def run_main(
        self, raw: pd.DataFrame, out: Path, now: datetime = EVENING
    ) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = main(
            ["--out", str(out)],
            downloader=downloader_for(raw),
            now=lambda: now,
            report=lines.append,
        )
        return code, lines

    def test_saves_and_reports(self, tmp_path: Path) -> None:
        history = make_prices(rows=6_100)
        raw = as_yahoo(with_extra_row(history, NEXT_DAY, close=np.nan))
        out = tmp_path / "nvda.csv"
        code, lines = self.run_main(raw, out, now=at(f"{NEXT_DAY} 21:00"))
        assert code == 0
        assert lines[0].startswith("saved 6100 sessions, ")
        assert lines[0].endswith(f"to 2030-03-14, to {out}")
        assert lines[1] == "left out 2030-03-15: incomplete"
        assert load_csv(out).index[-1].date() == date(2030, 3, 14)

    def test_invalid_data_is_not_saved(self, tmp_path: Path) -> None:
        out = tmp_path / "nvda.csv"
        code, lines = self.run_main(as_yahoo(make_prices()), out)
        assert code == 1
        assert lines == [
            "error: price data failed validation: only 300 rows, expected at least 6000"
        ]
        assert not out.exists()

    def test_refuses_to_replace_a_file_with_a_shorter_history(self, tmp_path: Path) -> None:
        history = make_prices(rows=6_100)
        out = tmp_path / "nvda.csv"
        save_csv(history, out)
        shorter = as_yahoo(pd.concat([history.iloc[:3_000], history.iloc[3_001:]]))
        code, lines = self.run_main(shorter, out)
        assert code == 1
        assert "missing 1 stored sessions" in lines[0]
        assert len(load_csv(out)) == 6_100

    def test_missing_directory_is_refused(self, tmp_path: Path) -> None:
        code, lines = self.run_main(as_yahoo(make_prices()), tmp_path / "nope" / "nvda.csv")
        assert code == 2
        assert "does not exist" in lines[0]
