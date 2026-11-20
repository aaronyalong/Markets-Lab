"""Market context and results-date features: values, timing and no look-ahead.

All prices, index levels and filings here are made up.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.context_features import (
    CONTEXT_COLUMNS,
    EXTENDED_FEATURE_COLUMNS,
    MAX_SESSIONS_SINCE_RESULTS,
    RESULTS_AFTER_CLOSE,
    RESULTS_COLUMNS,
    RESULTS_UNKNOWN,
    SESSIONS_SINCE_RESULTS,
    CallTooEarlyError,
    build_context_dataset,
    compute_context_features,
    compute_results_features,
    information_cutoff,
)
from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, build_dataset
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.collectors.market_context import SERIES
from markets_lab.collectors.nvda_earnings import ResultsPublication, known_at
from markets_lab.collectors.nvda_prices import PriceDataError
from price_frames import at, make_prices

ROWS = 300
LATE_EVENING = at("2030-03-14 22:30")  # the job's earliest run after the last session


def make_context(rows: int = ROWS) -> pd.DataFrame:
    """Made-up index closes on the same sessions as `make_prices(rows)`."""
    return pd.DataFrame(
        {name: make_prices(rows, seed=100 + number)["close"] for number, name in enumerate(SERIES)}
    )


def filing(when: str, number: int = 0) -> ResultsPublication:
    """A made-up results filing accepted at this New York time."""
    return ResultsPublication(
        at(when).astimezone(UTC), f"0000000000-30-{number:06d}", ("2.02", "9.01")
    )


def some_filings() -> list[ResultsPublication]:
    """Filings at awkward times, in order: after a close, before an open, during a
    session, late in the evening, and after the last session's close."""
    return [
        filing("2029-11-15 16:21", 1),  # Thursday, after the close
        filing("2029-12-14 06:11", 2),  # Friday, before the open
        filing("2030-01-23 12:50", 3),  # Wednesday, during the session
        filing("2030-02-14 21:05", 4),  # Thursday, after that day's cut-off
        filing("2030-03-14 16:40", 5),  # the last session, after the close
    ]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(ROWS)


@pytest.fixture(scope="module")
def context() -> pd.DataFrame:
    return make_context()


@pytest.fixture(scope="module")
def sessions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(make_prices(ROWS).index)


def row_of(frame: pd.DataFrame, session: str) -> pd.Series:
    position = frame.index.get_indexer(pd.DatetimeIndex([session]))[0]
    assert position >= 0, f"{session} is not a session"
    return frame.iloc[position]


class TestInformationCutoff:
    def test_is_eight_in_the_evening_in_new_york_all_year(self) -> None:
        assert information_cutoff(date(2030, 1, 15)) == datetime(2030, 1, 16, 1, 0, tzinfo=UTC)
        assert information_cutoff(date(2030, 7, 15)) == datetime(2030, 7, 16, 0, 0, tzinfo=UTC)

    def test_is_after_the_close_and_before_the_jobs_earliest_run(self) -> None:
        cutoff = information_cutoff(date(2030, 3, 14))
        assert at("2030-03-14 16:00") < cutoff < at("2030-03-14 22:30")


class TestContextFeatures:
    def test_columns_and_warm_up(self, prices: pd.DataFrame, context: pd.DataFrame) -> None:
        features = compute_context_features(prices, context)
        assert tuple(features.columns) == CONTEXT_COLUMNS
        assert features.index.equals(prices.index)
        assert features.iloc[:21].isna().any(axis=1).all()
        assert np.isfinite(features.iloc[21:].to_numpy()).all()

    def test_values(self, prices: pd.DataFrame, context: pd.DataFrame) -> None:
        features = compute_context_features(prices, context)
        t = 150
        semis, vix, adj = context["semis"], context["vix"], prices["adj_close"]
        assert features["semis_ret_1"].iloc[t] == pytest.approx(
            np.log(semis.iloc[t] / semis.iloc[t - 1])
        )
        assert features["semis_ret_5"].iloc[t] == pytest.approx(
            np.log(semis.iloc[t] / semis.iloc[t - 5])
        )
        assert features["nvda_vs_semis_21"].iloc[t] == pytest.approx(
            np.log(adj.iloc[t] / adj.iloc[t - 21]) - np.log(semis.iloc[t] / semis.iloc[t - 21])
        )
        assert features["vix_log"].iloc[t] == pytest.approx(np.log(vix.iloc[t]))
        assert features["vix_chg_5"].iloc[t] == pytest.approx(np.log(vix.iloc[t] / vix.iloc[t - 5]))

    def test_features_are_the_same_whether_or_not_later_sessions_exist(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        full = compute_context_features(prices, context)
        for end in (60, 150, 299):
            part = compute_context_features(prices.iloc[:end], context.iloc[:end])
            pd.testing.assert_frame_equal(part, full.iloc[:end])

    def test_replacing_the_future_with_nonsense_changes_nothing_in_the_past(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        full = compute_context_features(prices, context)
        poisoned_context = context.copy()
        poisoned_context.iloc[200:] = 1e9
        poisoned_prices = prices.copy()
        poisoned_prices.loc[poisoned_prices.index[200] :, "adj_close"] = 1e9
        poisoned = compute_context_features(poisoned_prices, poisoned_context)
        pd.testing.assert_frame_equal(poisoned.iloc[:200], full.iloc[:200])
        assert not poisoned.iloc[200].equals(full.iloc[200])

    def test_the_next_sessions_index_close_does_not_reach_todays_row(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        full = compute_context_features(prices, context)
        changed = context.copy()
        changed.iloc[151] *= 1.07
        after = compute_context_features(prices, changed)
        pd.testing.assert_series_equal(after.iloc[150], full.iloc[150])
        assert not after.iloc[151].equals(full.iloc[151])

    def test_stock_index_columns_do_not_depend_on_the_index_level(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        scaled = context.copy()
        scaled[["sp500", "nasdaq100", "semis"]] *= 37.0
        pd.testing.assert_frame_equal(
            compute_context_features(prices, scaled), compute_context_features(prices, context)
        )

    def test_a_session_missing_from_the_context_raises_instead_of_being_filled(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        with pytest.raises(PriceDataError, match="no row for 1 sessions"):
            compute_context_features(prices, context.drop(context.index[150]))
        # The case the nightly job must refuse: NVDA's close is in, the indices' are not.
        with pytest.raises(PriceDataError, match="last context session: 2030-03-13"):
            compute_context_features(prices, context.iloc[:-1])

    def test_context_rows_on_other_days_are_ignored(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        saturday = pd.DataFrame(
            {name: [5.0] for name in SERIES}, index=pd.DatetimeIndex(["2030-03-09"])
        )
        with_extra = pd.concat([context, saturday]).sort_index()
        pd.testing.assert_frame_equal(
            compute_context_features(prices, with_extra), compute_context_features(prices, context)
        )


class TestResultsFeatures:
    def test_columns(self, sessions: pd.DatetimeIndex) -> None:
        features = compute_results_features(sessions, some_filings())
        assert tuple(features.columns) == RESULTS_COLUMNS
        assert features.index.equals(sessions)
        assert np.isfinite(features.to_numpy()).all()

    def test_before_any_results_are_known_the_row_says_so(self, sessions: pd.DatetimeIndex) -> None:
        features = compute_results_features(sessions, some_filings())
        before = features.loc[: pd.Timestamp("2029-11-14")]
        assert (before[RESULTS_UNKNOWN] == 1.0).all()
        assert (before[SESSIONS_SINCE_RESULTS] == MAX_SESSIONS_SINCE_RESULTS).all()
        assert (before[RESULTS_AFTER_CLOSE] == 0.0).all()
        assert (features.loc[pd.Timestamp("2029-11-15") :, RESULTS_UNKNOWN] == 0.0).all()
        none_at_all = compute_results_features(sessions, [])
        assert (none_at_all[RESULTS_UNKNOWN] == 1.0).all()

    def test_results_after_the_close_are_known_that_evening_and_flagged(
        self, sessions: pd.DatetimeIndex
    ) -> None:
        # Accepted Thursday 16:21: Thursday's row knows, and Friday is the first
        # session to trade on them.
        features = compute_results_features(sessions, some_filings())
        thursday = row_of(features, "2029-11-15")
        assert thursday[SESSIONS_SINCE_RESULTS] == 0.0
        assert thursday[RESULTS_AFTER_CLOSE] == 1.0
        friday, monday = row_of(features, "2029-11-16"), row_of(features, "2029-11-19")
        assert (friday[SESSIONS_SINCE_RESULTS], friday[RESULTS_AFTER_CLOSE]) == (1.0, 0.0)
        assert (monday[SESSIONS_SINCE_RESULTS], monday[RESULTS_AFTER_CLOSE]) == (2.0, 0.0)

    @pytest.mark.parametrize(
        ("session", "day_before"),
        [("2029-12-14", "2029-12-13"), ("2030-01-23", "2030-01-22")],
    )
    def test_results_before_or_during_a_session_are_not_flagged_as_after_the_close(
        self, sessions: pd.DatetimeIndex, session: str, day_before: str
    ) -> None:
        # The market has already traded on them by the time the session's row exists.
        features = compute_results_features(sessions, some_filings())
        assert row_of(features, day_before)[SESSIONS_SINCE_RESULTS] > 0.0
        row = row_of(features, session)
        assert (row[SESSIONS_SINCE_RESULTS], row[RESULTS_AFTER_CLOSE]) == (0.0, 0.0)

    def test_results_after_the_cut_off_wait_for_the_next_session(
        self, sessions: pd.DatetimeIndex
    ) -> None:
        # Accepted Thursday 21:05, after the 20:00 cut-off: Thursday's call was
        # already final. Friday's row is the first to know.
        features = compute_results_features(sessions, some_filings())
        assert row_of(features, "2030-02-14")[SESSIONS_SINCE_RESULTS] > 0.0
        friday = row_of(features, "2030-02-15")
        assert (friday[SESSIONS_SINCE_RESULTS], friday[RESULTS_AFTER_CLOSE]) == (0.0, 0.0)

    def test_the_cut_off_itself_is_too_late_and_a_second_earlier_is_not(
        self, sessions: pd.DatetimeIndex
    ) -> None:
        on_time = compute_results_features(sessions, [filing("2030-02-14 19:59:59")])
        too_late = compute_results_features(sessions, [filing("2030-02-14 20:00:00")])
        assert row_of(on_time, "2030-02-14")[RESULTS_UNKNOWN] == 0.0
        assert row_of(too_late, "2030-02-14")[RESULTS_UNKNOWN] == 1.0
        assert row_of(too_late, "2030-02-15")[RESULTS_UNKNOWN] == 0.0

    def test_every_row_uses_only_results_published_before_its_own_cut_off(
        self, sessions: pd.DatetimeIndex
    ) -> None:
        # Remove everything a row could not have known and the row must not change.
        publications = some_filings()
        full = compute_results_features(sessions, publications)
        for position, stamp in enumerate(sessions):
            cutoff = information_cutoff(pd.Timestamp(stamp).date())
            known = known_at(publications, cutoff)
            limited = compute_results_features(sessions, known)
            pd.testing.assert_series_equal(limited.iloc[position], full.iloc[position])

    def test_rows_are_the_same_whether_or_not_later_sessions_exist(
        self, sessions: pd.DatetimeIndex
    ) -> None:
        full = compute_results_features(sessions, some_filings())
        for end in (150, 250, 299):
            part = compute_results_features(sessions[:end], some_filings())
            pd.testing.assert_frame_equal(part, full.iloc[:end])

    def test_the_count_stops_at_the_cap(self) -> None:
        sessions = pd.DatetimeIndex(make_prices(ROWS).index)
        features = compute_results_features(sessions, [filing("2029-03-01 16:20")])
        assert features[SESSIONS_SINCE_RESULTS].max() == MAX_SESSIONS_SINCE_RESULTS
        assert features[SESSIONS_SINCE_RESULTS].iloc[-1] == MAX_SESSIONS_SINCE_RESULTS
        assert features[RESULTS_UNKNOWN].iloc[-1] == 0.0

    def test_the_latest_known_filing_restarts_the_count(self, sessions: pd.DatetimeIndex) -> None:
        early, later = filing("2030-01-10 16:20", 1), filing("2030-01-17 16:20", 2)
        features = compute_results_features(sessions, [later, early])  # order does not matter
        assert row_of(features, "2030-01-16")[SESSIONS_SINCE_RESULTS] == 4.0
        assert row_of(features, "2030-01-17")[SESSIONS_SINCE_RESULTS] == 0.0

    def test_sessions_must_be_plain_ordered_dates(self, sessions: pd.DatetimeIndex) -> None:
        repeated = pd.DatetimeIndex([*sessions, sessions[-1]])
        for bad in (sessions[::-1], repeated, sessions.tz_localize("UTC")):
            with pytest.raises(ValueError, match="sessions must be"):
                compute_results_features(bad, some_filings())


class TestBuildContextDataset:
    def build(
        self,
        prices: pd.DataFrame,
        context: pd.DataFrame,
        publications: list[ResultsPublication] | None = None,
        as_of: datetime = LATE_EVENING,
    ):
        chosen = some_filings() if publications is None else publications
        return build_context_dataset(prices, context, chosen, as_of=as_of)

    def test_same_sessions_labels_and_technical_columns_as_the_plain_dataset(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        plain, extended = build_dataset(prices), self.build(prices, context)
        assert tuple(extended.features.columns) == EXTENDED_FEATURE_COLUMNS
        assert extended.features.index.equals(plain.features.index)
        pd.testing.assert_frame_equal(extended.features[list(FEATURE_COLUMNS)], plain.features)
        pd.testing.assert_series_equal(extended.target_up, plain.target_up)
        pd.testing.assert_series_equal(extended.next_return, plain.next_return)
        assert np.isfinite(extended.features.to_numpy()).all()

    def test_a_call_only_uses_results_published_before_its_own_timestamp(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        # Whatever the call's timestamp, taking away every filing published at or
        # after it changes nothing in any row, the call's own row included.
        publications = [
            *some_filings(),
            filing("2030-03-14 23:10", 6),
            filing("2030-03-15 06:30", 7),
        ]
        for as_of in (
            at("2030-03-14 20:00"),
            LATE_EVENING,
            at("2030-03-15 05:30"),
            at("2030-03-15 08:15"),
        ):
            everything = self.build(prices, context, publications, as_of)
            only_earlier = self.build(prices, context, list(known_at(publications, as_of)), as_of)
            pd.testing.assert_frame_equal(everything.features, only_earlier.features)
            used = known_at(publications, information_cutoff(everything.latest()[0]))
            assert all(item.accepted < as_of for item in used)

    def test_the_call_does_not_depend_on_what_time_the_job_ran(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        # A filing between two runs of the job must not change the call: both runs
        # use the same 20:00 cut-off.
        publications = [*some_filings(), filing("2030-03-15 06:30", 7)]
        first_run = self.build(prices, context, publications, at("2030-03-14 22:30"))
        retry = self.build(prices, context, publications, at("2030-03-15 07:15"))
        pd.testing.assert_frame_equal(first_run.features, retry.features)
        assert first_run.latest()[1][RESULTS_AFTER_CLOSE] == 1.0  # the 16:40 filing

    def test_a_call_timed_before_its_cut_off_is_refused(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        with pytest.raises(CallTooEarlyError, match="too early for a call based on 2030-03-14"):
            self.build(prices, context, as_of=at("2030-03-14 19:59"))
        self.build(prices, context, as_of=at("2030-03-14 20:00"))

    def test_the_call_time_must_be_timezone_aware(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            self.build(prices, context, as_of=datetime(2030, 3, 14, 22, 30))

    def test_context_missing_for_the_latest_session_raises(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        with pytest.raises(PriceDataError, match="last context session: 2030-03-13"):
            self.build(prices, context.iloc[:-1])

    def test_a_bad_context_level_raises(self, prices: pd.DataFrame, context: pd.DataFrame) -> None:
        broken = context.copy()
        broken.loc[broken.index[250], "vix"] = np.nan
        with pytest.raises(ValueError, match="context features are incomplete, first on"):
            self.build(prices, broken)

    def test_an_existing_model_can_be_fitted_on_the_extended_columns(
        self, prices: pd.DataFrame, context: pd.DataFrame
    ) -> None:
        dataset = self.build(prices, context)
        known = dataset.labelled()
        model = LogisticRegressionModel()
        model.fit(known.features, known.target_up)
        call = model.predict(dataset.features.iloc[[-1]])
        assert 0.0 <= float(call["probability_up"].iloc[0]) <= 1.0


def test_later_filings_never_reach_an_earlier_session_through_the_dataset() -> None:
    """Adding filings after a session's cut-off leaves that session's row unchanged."""
    prices, context = make_prices(ROWS), make_context()
    base = build_context_dataset(prices, context, some_filings()[:2], as_of=LATE_EVENING)
    more = build_context_dataset(prices, context, some_filings(), as_of=LATE_EVENING)
    unchanged_through = pd.Timestamp("2030-01-22")  # the day before the third filing
    pd.testing.assert_frame_equal(
        more.features.loc[:unchanged_through], base.features.loc[:unchanged_through]
    )
    assert not more.features.loc[pd.Timestamp("2030-01-23") :].equals(
        base.features.loc[pd.Timestamp("2030-01-23") :]
    )
    assert timedelta(0) < LATE_EVENING - information_cutoff(date(2030, 3, 14))
