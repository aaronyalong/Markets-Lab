"""The nightly job: outcomes, next-session calls, and the cases where it must refuse."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from context_frames import context_for, make_publications, sec_getter, yahoo_downloader
from markets_lab.analysis.nvda import nightly as nightly_module
from markets_lab.analysis.nvda import prediction_log
from markets_lab.analysis.nvda.backtest import (
    BASELINE_FACTORIES,
    CONTEXT_MODEL_FACTORIES,
    MODEL_FACTORIES,
    PredictorFactory,
)
from markets_lab.analysis.nvda.context_features import (
    EXTENDED_FEATURE_COLUMNS,
    information_cutoff,
)
from markets_lab.analysis.nvda.features import build_dataset
from markets_lab.analysis.nvda.models.context import LogisticContextModel
from markets_lab.analysis.nvda.nightly import (
    EXIT_CONTEXT_SKIPPED,
    NightlyError,
    NightlyResult,
    default_factories,
    default_range_factories,
    main,
    run_nightly,
)
from markets_lab.analysis.nvda.paper_strategy.live import live_record
from markets_lab.analysis.nvda.paper_strategy.rule import GATE_PERCENTILE, RULE_MODELS
from markets_lab.analysis.nvda.prediction_log import (
    LogError,
    Outcome,
    PaperStrategyEntry,
    Prediction,
    RangeForecast,
    RangeOutcome,
    append_records,
    parse_log,
    prediction_window,
    read_log,
    scored_predictions,
    scored_ranges,
)
from markets_lab.analysis.nvda.predictors import prediction_frame
from markets_lab.analysis.nvda.volatility.dataset import build_volatility_dataset
from markets_lab.collectors.nvda_earnings import ResultsPublication
from nvda_test_predictors import LeastSquaresLearner
from price_frames import as_yahoo, at, make_prices, with_extra_row

THU, FRI, MON, TUE = date(2030, 3, 14), date(2030, 3, 15), date(2030, 3, 18), date(2030, 3, 19)
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
FACTORIES: dict[str, PredictorFactory] = {
    **BASELINE_FACTORIES,
    "least_squares": LeastSquaresLearner,
}
MODELS = ["always_up", "persistence", "ma_200", "least_squares"]


def history() -> pd.DataFrame:
    """Completed sessions through Thursday 2030-03-14."""
    return make_prices(rows=420)


def add_session(prices: pd.DataFrame, session: date, change: float) -> pd.DataFrame:
    """Append a session whose prices are the last session's moved by `change`."""
    row = prices.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([pd.Timestamp(session)], name="date")
    row[PRICE_COLUMNS] = row[PRICE_COLUMNS] * (1.0 + change)
    return pd.concat([prices, row])


def nightly(
    log: Path,
    prices: pd.DataFrame,
    now: str,
    *,
    factories: dict[str, PredictorFactory] | None = None,
    dry_run: bool = False,
) -> tuple[list[Outcome], list[Prediction]]:
    result = run_nightly(
        now=at(now),
        log_path=log,
        downloader=lambda _symbol: as_yahoo(prices),
        factories=FACTORIES if factories is None else factories,
        context_factories={},
        range_factories={},
        dry_run=dry_run,
        min_rows=100,
    )
    return list(result.outcomes), list(result.predictions)


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "predictions" / "nvda.jsonl"


class TestFirstRun:
    def test_writes_one_prediction_per_predictor_and_no_outcome(self, log: Path) -> None:
        outcomes, predictions = nightly(log, history(), "2030-03-15 06:00")
        assert outcomes == []
        assert [p.model for p in predictions] == MODELS
        assert {p.based_on_session for p in predictions} == {THU}
        assert {p.made_at for p in predictions} == {at("2030-03-15 06:00")}
        assert read_log(log) == predictions

    def test_baselines_have_no_probability_and_models_do(self, log: Path) -> None:
        _, predictions = nightly(log, history(), "2030-03-15 06:00")
        by_model = {p.model: p for p in predictions}
        assert by_model["always_up"].predicted_up is True
        assert by_model["always_up"].probability_up is None
        assert by_model["persistence"].probability_up is None
        probability = by_model["least_squares"].probability_up
        assert probability is not None
        assert by_model["least_squares"].predicted_up == (probability > 0.5)

    def test_calls_match_the_baseline_rules(self, log: Path) -> None:
        prices = history()
        _, predictions = nightly(log, prices, "2030-03-15 06:00")
        by_model = {p.model: p.predicted_up for p in predictions}
        last = build_dataset(prices).features.iloc[-1]
        assert by_model["persistence"] == bool(last["ret_1"] > 0)
        assert by_model["ma_200"] == bool(last["sma_200_gap"] > 0)


class TestFollowingRuns:
    def test_next_morning_scores_yesterday_then_predicts(self, log: Path) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        prices = add_session(prices, FRI, +0.02)
        outcomes, predictions = nightly(log, prices, "2030-03-16 06:00")
        assert outcomes == [Outcome(at("2030-03-16 06:00"), FRI, THU, True)]
        assert {p.based_on_session for p in predictions} == {FRI}
        scored = scored_predictions(read_log(log))
        thursday = scored[scored["based_on_session"] == THU].set_index("model")
        assert bool(thursday.loc["always_up", "correct"]) is True
        assert thursday["target_session"].tolist() == [FRI] * 4
        assert scored[scored["based_on_session"] == FRI]["correct"].isna().all()

    def test_a_down_or_flat_session_is_recorded_as_not_up(self, log: Path) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        prices = add_session(prices, FRI, -0.03)
        prices = add_session(prices, MON, 0.0)
        outcomes, _ = nightly(log, prices, "2030-03-19 06:00")
        assert [(o.session, o.previous_session, o.went_up) for o in outcomes] == [
            (FRI, THU, False),
            (MON, FRI, False),
        ]

    def test_running_twice_the_same_morning_adds_nothing(self, log: Path) -> None:
        nightly(log, history(), "2030-03-15 06:00")
        before = log.read_bytes()
        outcomes, predictions = nightly(log, history(), "2030-03-15 07:45")
        assert (outcomes, predictions) == ([], [])
        assert log.read_bytes() == before

    def test_weekend_and_holiday_reruns_add_nothing(self, log: Path) -> None:
        prices = add_session(history(), FRI, +0.01)
        nightly(log, prices, "2030-03-16 06:00")
        before = log.read_bytes()
        for moment in ("2030-03-17 06:00", "2030-03-18 06:00"):
            assert nightly(log, prices, moment) == ([], [])
        assert log.read_bytes() == before

    def test_after_missed_days_every_outcome_is_recorded_but_only_one_call_is_made(
        self, log: Path
    ) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        for session, change in ((FRI, 0.01), (MON, -0.01), (TUE, 0.02)):
            prices = add_session(prices, session, change)
        outcomes, predictions = nightly(log, prices, "2030-03-20 06:00")
        assert [o.session for o in outcomes] == [FRI, MON, TUE]
        assert [o.went_up for o in outcomes] == [True, False, True]
        assert {p.based_on_session for p in predictions} == {TUE}
        # The days with no prediction stay missing; they are never filled in later.
        scored = scored_predictions(read_log(log))
        assert sorted(set(scored["based_on_session"])) == [THU, TUE]

    def test_a_new_model_joins_on_its_own_start_date(self, log: Path) -> None:
        nightly(log, history(), "2030-03-15 06:00", factories=dict(BASELINE_FACTORIES))
        _, predictions = nightly(log, history(), "2030-03-15 07:00")
        assert [p.model for p in predictions] == ["least_squares"]
        assert len(read_log(log)) == 4

    def test_the_log_stays_valid_and_append_only_across_runs(self, log: Path) -> None:
        prices = history()
        versions: list[str] = []
        for session, change, now in (
            (None, 0.0, "2030-03-15 06:00"),
            (FRI, 0.01, "2030-03-16 06:00"),
            (MON, -0.02, "2030-03-19 06:00"),
        ):
            if session is not None:
                prices = add_session(prices, session, change)
            nightly(log, prices, now)
            versions.append(log.read_text())
            parse_log(versions[-1])
        assert versions[1].startswith(versions[0])
        assert versions[2].startswith(versions[1])
        assert len(versions[0]) < len(versions[1]) < len(versions[2])


class TestRefusals:
    """Cases where an honest prediction is impossible: nothing may be written."""

    def test_too_late_for_the_session(self, log: Path) -> None:
        # Friday 09:30 New York: the session after Thursday has opened.
        with pytest.raises(NightlyError, match="too late to predict the session after 2030-03-14"):
            nightly(log, history(), "2030-03-15 09:30")
        assert not log.exists()

    def test_too_late_leaves_an_existing_log_untouched(self, log: Path) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        before = log.read_bytes()
        prices = add_session(prices, FRI, 0.01)
        with pytest.raises(NightlyError, match="too late"):
            nightly(log, prices, "2030-03-18 12:00")
        assert log.read_bytes() == before

    def test_yahoo_has_not_published_the_latest_close(self, log: Path) -> None:
        # Seen in real data: the latest row has no close for hours after the bell.
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        before = log.read_bytes()
        unpublished = with_extra_row(prices, str(FRI), close=np.nan, adj_close=np.nan)
        with pytest.raises(NightlyError, match="session 2030-03-15 has finished but Yahoo"):
            nightly(log, unpublished, "2030-03-16 06:00")
        assert log.read_bytes() == before

    def test_unpublished_close_is_an_error_even_on_the_first_run(self, log: Path) -> None:
        unpublished = with_extra_row(history(), str(FRI), close=np.nan)
        with pytest.raises(NightlyError, match="not published"):
            nightly(log, unpublished, "2030-03-15 20:00")
        assert not log.exists()

    def test_a_log_session_missing_from_the_prices_is_an_error(self, log: Path) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        gone = pd.concat([prices.iloc[:-1], add_session(prices, FRI, 0.01).iloc[[-1]]])
        with pytest.raises(
            NightlyError, match="2030-03-14 is in the log but not in the downloaded"
        ):
            nightly(log, gone, "2030-03-16 06:00")


class TestNoLookAhead:
    def test_a_partial_bar_for_today_never_reaches_a_prediction(
        self, log: Path, tmp_path: Path
    ) -> None:
        # A manual run during Friday's session, before any prediction was written,
        # is too late. But the partial bar must also be invisible to the models: with
        # it present or absent, a run before the open writes identical records.
        prices = history()
        partial = with_extra_row(prices, str(FRI), close=1.0, adj_close=1.0, low=0.5)
        clean_log, dirty_log = tmp_path / "clean.jsonl", tmp_path / "dirty.jsonl"
        nightly(clean_log, prices, "2030-03-15 06:00")
        nightly(dirty_log, partial, "2030-03-15 06:00")
        assert dirty_log.read_text() == clean_log.read_text()
        del log

    def test_models_are_fitted_only_on_sessions_before_the_one_they_predict_from(
        self, log: Path
    ) -> None:
        seen: dict[str, object] = {}

        class Watcher:
            name = "watcher"

            def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
                seen["last_trained"] = pd.Timestamp(features.index[-1]).date()
                seen["missing_outcomes"] = int(target_up.isna().sum())

            def predict(self, features: pd.DataFrame) -> pd.DataFrame:
                seen["predicted_from"] = [pd.Timestamp(ts).date() for ts in features.index]
                return prediction_frame(features.index, np.ones(len(features), dtype=bool))

        nightly(log, history(), "2030-03-15 06:00", factories={"watcher": Watcher})
        assert seen["predicted_from"] == [THU]
        assert seen["last_trained"] == date(2030, 3, 13)
        assert seen["missing_outcomes"] == 0

    def test_the_outcome_is_recorded_only_after_the_prediction_it_scores(self, log: Path) -> None:
        prices = history()
        nightly(log, prices, "2030-03-15 06:00")
        nightly(log, add_session(prices, FRI, 0.02), "2030-03-16 06:00")
        records = read_log(log)
        outcome_at = next(i for i, r in enumerate(records) if isinstance(r, Outcome))
        thursday_calls = [
            i
            for i, r in enumerate(records)
            if isinstance(r, Prediction) and r.based_on_session == THU
        ]
        assert max(thursday_calls) < outcome_at


class TestDryRunAndCommandLine:
    def run_main(
        self,
        prices: pd.DataFrame,
        now: datetime,
        *argv: str,
        downloader: Callable[[str], pd.DataFrame] | None = None,
    ) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = main(
            list(argv),
            now=lambda: now,
            downloader=downloader or (lambda _symbol: as_yahoo(prices)),
            factories=FACTORIES,
            context_factories={},
            range_factories={},
            report=lines.append,
            min_rows=100,
        )
        return code, lines

    def test_dry_run_writes_nothing(self, log: Path) -> None:
        outcomes, predictions = nightly(log, history(), "2030-03-15 06:00", dry_run=True)
        assert len(predictions) == 4
        assert outcomes == []
        assert not log.exists()

    def test_success_reports_what_was_appended(self, log: Path) -> None:
        code, lines = self.run_main(history(), at("2030-03-15 06:00"), "--log", str(log))
        assert code == 0
        assert lines[0] == "last completed session: 2030-03-14"
        assert lines[1].startswith("appended prediction: always_up calls up for the session after")
        assert any("least_squares calls" in line and "(p up 0." in line for line in lines)
        assert len(read_log(log)) == 4

    def test_dry_run_flag_says_would(self, log: Path) -> None:
        code, lines = self.run_main(
            history(), at("2030-03-15 06:00"), "--log", str(log), "--dry-run"
        )
        assert code == 0
        assert all(line.startswith("would append") for line in lines[1:])
        assert not log.exists()

    def test_nothing_new_is_a_success(self, log: Path) -> None:
        self.run_main(history(), at("2030-03-15 06:00"), "--log", str(log))
        code, lines = self.run_main(history(), at("2030-03-15 07:00"), "--log", str(log))
        assert code == 0
        assert lines == ["last completed session: 2030-03-14", "nothing new to record"]

    def test_outcomes_are_reported(self, log: Path) -> None:
        prices = history()
        self.run_main(prices, at("2030-03-15 06:00"), "--log", str(log))
        code, lines = self.run_main(
            add_session(prices, FRI, -0.01), at("2030-03-16 06:00"), "--log", str(log)
        )
        assert code == 0
        assert "appended outcome: 2030-03-15 closed not up" in lines

    def test_a_refusal_fails_loudly_and_writes_nothing(self, log: Path) -> None:
        code, lines = self.run_main(history(), at("2030-03-15 10:00"), "--log", str(log))
        assert code == 1
        assert lines[0].startswith("NIGHTLY RUN FAILED, nothing was written: too late")
        assert not log.exists()

    def test_bad_price_data_fails_loudly(self, log: Path) -> None:
        def broken(_symbol: str) -> pd.DataFrame:
            raise ConnectionError("blocked")

        code, lines = self.run_main(
            history(), at("2030-03-15 06:00"), "--log", str(log), downloader=broken
        )
        assert code == 1
        assert "download of NVDA failed: blocked" in lines[0]

    def test_a_corrupt_log_fails_loudly_and_is_not_extended(self, log: Path) -> None:
        prices = history()
        self.run_main(prices, at("2030-03-15 06:00"), "--log", str(log))
        log.write_text(log.read_text().replace('"predicted_up":true', '"predicted_up":false', 1))
        corrupt = log.read_bytes()
        code, lines = self.run_main(
            add_session(prices, FRI, 0.01), at("2030-03-16 06:00"), "--log", str(log)
        )
        assert code == 1
        assert "content does not match its hash" in lines[0]
        assert log.read_bytes() == corrupt


SIX = list(default_factories())
CONTEXT: dict[str, PredictorFactory] = {"logistic_ctx": LogisticContextModel}
EVENING, EARLY_MORNING = "2030-03-14 22:30", "2030-03-15 05:30"


def filed(when: str, number: int = 900) -> ResultsPublication:
    """A made-up results filing accepted at this New York time."""
    return ResultsPublication(
        at(when).astimezone(UTC), f"0000000000-30-{number:06d}", ("2.02", "9.01")
    )


def full_run(
    log: Path,
    now: str,
    *,
    prices: pd.DataFrame | None = None,
    downloader: Callable[[str], pd.DataFrame] | None = None,
    get: Callable[[str], str] | None = None,
    context_factories: dict[str, PredictorFactory] | None = None,
) -> NightlyResult:
    """A run with the real six predictors and one context model."""
    chosen = history() if prices is None else prices
    return run_nightly(
        now=at(now),
        log_path=log,
        downloader=downloader or yahoo_downloader(chosen),
        get=get or sec_getter(),
        context_factories=CONTEXT if context_factories is None else context_factories,
        range_factories={},
        min_rows=100,
    )


def index_download_fails(prices: pd.DataFrame) -> dict[str, Any]:
    return {"downloader": yahoo_downloader(prices, failing=["^SOX"])}


def sec_download_fails(_prices: pd.DataFrame) -> dict[str, Any]:
    return {"get": sec_getter(failing=True)}


def indices_lack_the_latest_session(prices: pd.DataFrame) -> dict[str, Any]:
    return {"downloader": yahoo_downloader(prices, context_for(prices).iloc[:-1])}


class BrokenContextModel(LogisticContextModel):
    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        raise ZeroDivisionError("a bug in a context model")


def a_context_model_has_a_bug(_prices: pd.DataFrame) -> dict[str, Any]:
    return {"context_factories": {"logistic_ctx": BrokenContextModel}}


CONTEXT_FAILURES = [
    index_download_fails,
    sec_download_fails,
    indices_lack_the_latest_session,
    a_context_model_has_a_bug,
]


class TestContextModels:
    """The context models need extra downloads. When those fail, only they wait."""

    def test_every_predictor_calls_when_the_downloads_work(self, log: Path) -> None:
        result = full_run(log, EVENING)
        assert [p.model for p in result.predictions] == [*SIX, "logistic_ctx"]
        assert result.skipped == () and result.skip_reason is None
        assert len(read_log(log)) == 7

    def test_the_context_model_is_given_all_the_columns(self, log: Path) -> None:
        seen: list[list[str]] = []

        class Recorder(LogisticContextModel):
            def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
                seen.append(list(features.columns))
                super().fit(features, target_up)

        full_run(log, EVENING, context_factories={"logistic_ctx": Recorder})
        assert seen == [list(EXTENDED_FEATURE_COLUMNS)]

    @pytest.mark.parametrize("failure", CONTEXT_FAILURES)
    def test_a_failed_context_download_leaves_the_other_six_calls_identical_to_a_normal_run(
        self, tmp_path: Path, failure: Callable[[pd.DataFrame], dict[str, Any]]
    ) -> None:
        normal_log, failed_log = tmp_path / "normal.jsonl", tmp_path / "failed.jsonl"
        normal = full_run(normal_log, EVENING)
        failed = full_run(failed_log, EVENING, **failure(history()))

        six = [p for p in normal.predictions if p.model in SIX]
        assert len(six) == 6
        assert list(failed.predictions) == six
        assert failed.outcomes == normal.outcomes
        assert failed.skipped == ("logistic_ctx",)
        assert failed.skip_reason
        # The file itself is the normal run's file without the context model's line.
        normal_lines = normal_log.read_text().splitlines()
        assert failed_log.read_text().splitlines() == normal_lines[:6]
        assert len(normal_lines) == 7

    @pytest.mark.parametrize("failure", CONTEXT_FAILURES)
    def test_the_same_holds_the_morning_after_when_an_outcome_is_recorded_too(
        self, tmp_path: Path, failure: Callable[[pd.DataFrame], dict[str, Any]]
    ) -> None:
        friday = add_session(history(), FRI, 0.02)
        logs = tmp_path / "normal.jsonl", tmp_path / "failed.jsonl"
        for path in logs:
            full_run(path, EVENING)
        normal = full_run(logs[0], "2030-03-15 22:30", prices=friday)
        failed = full_run(logs[1], "2030-03-15 22:30", prices=friday, **failure(friday))
        assert [o.session for o in failed.outcomes] == [FRI]
        assert failed.outcomes == normal.outcomes
        assert list(failed.predictions) == [p for p in normal.predictions if p.model in SIX]
        assert failed.skipped == ("logistic_ctx",)

    def test_the_reason_names_what_failed(self, log: Path) -> None:
        cases = {
            index_download_fails: "download of ^SOX (semis) failed",
            sec_download_fails: "download from SEC failed",
            indices_lack_the_latest_session: "last context session: 2030-03-13",
            a_context_model_has_a_bug: "ZeroDivisionError: a bug in a context model",
        }
        for number, (failure, expected) in enumerate(cases.items()):
            result = full_run(log.with_name(f"{number}.jsonl"), EVENING, **failure(history()))
            assert expected in str(result.skip_reason)

    def test_before_the_cut_off_only_the_context_models_wait(self, log: Path) -> None:
        # 18:00 on the day of the session: the bar is complete, so the six may
        # call, but information for a context call counts up to 20:00.
        early = full_run(log, "2030-03-14 18:00")
        assert [p.model for p in early.predictions] == SIX
        assert early.skipped == ("logistic_ctx",)
        assert "too early for a call based on 2030-03-14" in str(early.skip_reason)
        later = full_run(log, "2030-03-14 20:00")
        assert [p.model for p in later.predictions] == ["logistic_ctx"]

    def test_a_later_run_before_the_open_adds_the_missing_call(self, log: Path) -> None:
        first = full_run(log, EVENING, **sec_download_fails(history()))
        assert first.skipped == ("logistic_ctx",)
        second = full_run(log, EARLY_MORNING)
        assert [p.model for p in second.predictions] == ["logistic_ctx"]
        assert second.outcomes == () and second.skipped == ()
        assert second.predictions[0].made_at == at(EARLY_MORNING)
        assert second.predictions[0].based_on_session == THU
        assert [r.model for r in read_log(log) if isinstance(r, Prediction)] == [
            *SIX,
            "logistic_ctx",
        ]

    def test_a_late_fill_makes_the_call_the_first_run_would_have_made(self, tmp_path: Path) -> None:
        # Results came out after Thursday's close, before the 20:00 cut-off. More
        # arrive overnight, after the cut-off and before the retry. The retry must
        # make Thursday evening's call, not a better-informed one.
        known_by_evening = [*make_publications(), filed("2030-03-14 16:25", 901)]
        known_by_morning = [*known_by_evening, filed("2030-03-14 23:10", 902)]

        on_time = full_run(tmp_path / "a.jsonl", EVENING, get=sec_getter(known_by_evening))
        full_run(tmp_path / "b.jsonl", EVENING, **sec_download_fails(history()))
        filled = full_run(tmp_path / "b.jsonl", EARLY_MORNING, get=sec_getter(known_by_morning))

        first, late = on_time.predictions[-1], filled.predictions[0]
        assert first.model == late.model == "logistic_ctx"
        assert (late.predicted_up, late.probability_up) == (
            first.predicted_up,
            first.probability_up,
        )
        assert late.made_at > first.made_at

    @pytest.mark.parametrize("when", ["2030-03-15 09:30", "2030-03-15 10:00", "2030-03-15 15:00"])
    def test_a_late_fill_after_the_open_is_refused(self, log: Path, when: str) -> None:
        full_run(log, EVENING, **index_download_fails(history()))
        before = log.read_bytes()
        # The downloads work now, but the session the call was for has opened.
        with pytest.raises(NightlyError, match="too late to predict the session after 2030-03-14"):
            full_run(log, when)
        assert log.read_bytes() == before

    def test_a_missed_day_is_never_filled_in_afterwards(self, log: Path) -> None:
        full_run(log, EVENING, **index_download_fails(history()))
        friday = add_session(history(), FRI, 0.02)
        monday = add_session(friday, MON, -0.01)
        full_run(log, "2030-03-15 22:30", prices=friday)
        full_run(log, "2030-03-18 22:30", prices=monday)

        calls = [r for r in read_log(log) if isinstance(r, Prediction)]
        based_on = {
            name: [c.based_on_session for c in calls if c.model == name]
            for name in ("logistic", "logistic_ctx")
        }
        assert based_on == {"logistic": [THU, FRI, MON], "logistic_ctx": [FRI, MON]}
        # Two sessions have been scored since. The context model has a call for
        # only one of them: Friday's session was never called and stays that way.
        scored = scored_predictions(read_log(log))
        scored = scored[scored["correct"].notna()]
        counts = scored.groupby("model").size()
        assert (counts["logistic"], counts["logistic_ctx"]) == (2, 1)

    def test_the_log_itself_refuses_a_context_call_written_after_the_open(self, log: Path) -> None:
        # Even if the job's own check were removed, the log would not take it.
        full_run(log, EVENING, **index_download_fails(history()))
        late = Prediction(
            made_at=at("2030-03-15 09:45"),
            based_on_session=THU,
            model="logistic_ctx",
            predicted_up=True,
            probability_up=0.6,
        )
        with pytest.raises(LogError, match="after the next session could have opened"):
            append_records(log, [late])

    def test_a_retry_that_still_fails_adds_nothing_and_says_so_again(self, log: Path) -> None:
        full_run(log, EVENING, **index_download_fails(history()))
        before = log.read_bytes()
        again = full_run(log, EARLY_MORNING, **index_download_fails(history()))
        assert again.predictions == () and again.outcomes == ()
        assert again.skipped == ("logistic_ctx",)
        assert log.read_bytes() == before

    def test_once_every_call_is_written_the_extra_downloads_are_not_made_again(
        self, log: Path
    ) -> None:
        full_run(log, EVENING)
        rerun = full_run(
            log,
            EARLY_MORNING,
            downloader=yahoo_downloader(history(), failing=["^GSPC", "^NDX", "^SOX", "^VIX"]),
            get=sec_getter(failing=True),
        )
        assert rerun.predictions == () and rerun.skipped == ()

    def test_a_name_cannot_be_both_a_model_and_a_context_model(self, log: Path) -> None:
        with pytest.raises(ValueError, match="both a model and a context model"):
            full_run(log, EVENING, context_factories={"logistic": LogisticContextModel})

    def run_main(self, log: Path, now: str, **overrides: Any) -> tuple[int, list[str]]:
        lines: list[str] = []
        arguments: dict[str, Any] = {
            "downloader": yahoo_downloader(history()),
            "get": sec_getter(),
            "context_factories": CONTEXT,
            "range_factories": {},
            **overrides,
        }
        code = main(
            ["--log", str(log)], now=lambda: at(now), report=lines.append, min_rows=100, **arguments
        )
        return code, lines

    def test_the_command_ends_with_its_own_exit_code_after_writing_the_six(self, log: Path) -> None:
        code, lines = self.run_main(log, EVENING, **sec_download_fails(history()))
        assert code == EXIT_CONTEXT_SKIPPED == 3
        assert len(read_log(log)) == 6
        assert sum(line.startswith("appended prediction:") for line in lines) == 6
        assert (
            "NO CALL from logistic_ctx for the session after 2030-03-14: download from SEC failed"
            in lines[-3]
        )
        assert "only before the next session opens" in lines[-2]
        assert lines[-1].startswith("NIGHTLY RUN INCOMPLETE")

    def test_the_command_succeeds_when_a_retry_fills_the_call(self, log: Path) -> None:
        self.run_main(log, EVENING, **sec_download_fails(history()))
        code, lines = self.run_main(log, EARLY_MORNING)
        assert code == 0
        assert [line for line in lines if "prediction" in line] == [
            line for line in lines if "logistic_ctx" in line
        ]

    def test_a_late_fill_fails_loudly_from_the_command_line(self, log: Path) -> None:
        self.run_main(log, EVENING, **sec_download_fails(history()))
        before = log.read_bytes()
        code, lines = self.run_main(log, "2030-03-15 10:00")
        assert code == 1
        assert lines[0].startswith("NIGHTLY RUN FAILED, nothing was written: too late")
        assert log.read_bytes() == before


RANGES = ["constant", "rolling_21", "ewma_94", "har", "xgboost_vol"]


def range_run(
    log: Path,
    prices: pd.DataFrame,
    now: str,
    *,
    range_factories: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> NightlyResult:
    """A run with the usual test predictors and the real volatility forecasters."""
    return run_nightly(
        now=at(now),
        log_path=log,
        downloader=lambda _symbol: as_yahoo(prices),
        factories=FACTORIES,
        context_factories={},
        range_factories=range_factories,
        dry_run=dry_run,
        min_rows=100,
    )


class TestRanges:
    """The size of the move: a range per forecaster, then where the return fell."""

    def test_the_first_run_writes_one_range_per_forecaster(self, log: Path) -> None:
        result = range_run(log, history(), "2030-03-14 22:30")
        assert [f.model for f in result.range_forecasts] == RANGES
        assert result.range_outcomes == ()
        for forecast in result.range_forecasts:
            assert forecast.based_on_session == THU
            assert forecast.level == 0.8
            assert -0.5 < forecast.lower_return < 0 < forecast.upper_return < 0.5
            assert forecast.made_at == at("2030-03-14 22:30")

    def test_the_range_is_the_forecasters_own_output_as_simple_returns(self, log: Path) -> None:
        result = range_run(log, history(), "2030-03-14 22:30")
        dataset = build_volatility_dataset(history())
        known = dataset.labelled()
        forecaster = default_range_factories()["ewma_94"]()
        forecaster.fit(known.features, known.next_log_return)
        expected = forecaster.predict(dataset.features.iloc[[-1]]).iloc[0]
        logged = next(f for f in result.range_forecasts if f.model == "ewma_94")
        assert logged.lower_return == round(float(np.expm1(expected["lower"])), 4)
        assert logged.upper_return == round(float(np.expm1(expected["upper"])), 4)

    @pytest.mark.parametrize(
        ("change", "position"), [(0.30, "above"), (-0.30, "below"), (0.001, "inside")]
    )
    def test_the_next_run_records_where_the_return_fell(
        self, log: Path, change: float, position: str
    ) -> None:
        range_run(log, history(), "2030-03-14 22:30")
        result = range_run(log, add_session(history(), FRI, change), "2030-03-15 22:30")
        assert [(o.model, o.position) for o in result.range_outcomes] == [
            (name, position) for name in RANGES
        ]
        assert {(o.session, o.previous_session) for o in result.range_outcomes} == {(FRI, THU)}
        assert [f.based_on_session for f in result.range_forecasts] == [FRI] * len(RANGES)

    def test_the_position_is_judged_against_the_bounds_as_logged(self, log: Path) -> None:
        first = range_run(log, history(), "2030-03-14 22:30")
        bounds = {f.model: (f.lower_return, f.upper_return) for f in first.range_forecasts}
        # A rise that is above the narrowest logged upper bound and below the widest.
        uppers = sorted(upper for _, upper in bounds.values())
        change = (uppers[0] + uppers[-1]) / 2
        assert uppers[0] < change < uppers[-1]
        result = range_run(log, add_session(history(), FRI, change), "2030-03-15 22:30")
        for scored in result.range_outcomes:
            upper = bounds[scored.model][1]
            assert scored.position == ("above" if change > upper else "inside")
        assert {o.position for o in result.range_outcomes} == {"above", "inside"}

    def test_records_are_written_in_the_order_the_log_needs(self, log: Path) -> None:
        range_run(log, history(), "2030-03-14 22:30")
        range_run(log, add_session(history(), FRI, 0.01), "2030-03-15 22:30")
        kinds = [type(record).__name__ for record in read_log(log)]
        friday = kinds[len(MODELS) + len(RANGES) :]
        assert friday == (
            ["Outcome"]
            + ["RangeOutcome"] * len(RANGES)
            + ["Prediction"] * len(MODELS)
            + ["RangeForecast"] * len(RANGES)
        )

    def test_running_again_adds_nothing(self, log: Path) -> None:
        range_run(log, history(), "2030-03-14 22:30")
        friday = add_session(history(), FRI, 0.01)
        range_run(log, friday, "2030-03-15 22:30")
        before = log.read_bytes()
        again = range_run(log, friday, "2030-03-16 06:00")
        assert again.records == []
        assert log.read_bytes() == before

    def test_a_range_is_not_written_after_the_next_session_could_have_opened(
        self, log: Path
    ) -> None:
        with pytest.raises(NightlyError, match="too late to predict the session after 2030-03-14"):
            range_run(log, history(), "2030-03-15 09:30")
        assert not log.exists()

    def test_a_missed_range_is_never_filled_in_afterwards(self, log: Path) -> None:
        # The direction calls were made on Thursday evening, the ranges were not.
        nightly(log, history(), "2030-03-14 22:30")
        before = log.read_bytes()
        with pytest.raises(NightlyError, match="too late to predict"):
            range_run(log, history(), "2030-03-15 10:00")
        assert log.read_bytes() == before
        range_run(log, add_session(history(), FRI, 0.01), "2030-03-15 22:30")
        ranges = scored_ranges(read_log(log))
        assert set(ranges["based_on_session"]) == {FRI}
        assert not any(isinstance(r, RangeOutcome) for r in read_log(log))

    def test_a_new_forecaster_joins_on_its_own_start_date(self, log: Path) -> None:
        factories = default_range_factories()
        without_har = {name: factory for name, factory in factories.items() if name != "har"}
        range_run(log, history(), "2030-03-14 22:30", range_factories=without_har)
        result = range_run(log, add_session(history(), FRI, 0.01), "2030-03-15 22:30")
        assert "har" not in {o.model for o in result.range_outcomes}
        assert "har" in {f.model for f in result.range_forecasts}

    def test_ranges_are_written_even_when_the_context_models_are_skipped(self, log: Path) -> None:
        result = run_nightly(
            now=at(EVENING),
            log_path=log,
            downloader=yahoo_downloader(history(), failing=["^VIX"]),
            get=sec_getter(),
            factories=FACTORIES,
            context_factories=CONTEXT,
            min_rows=100,
        )
        assert result.skipped == ("logistic_ctx",)
        assert [f.model for f in result.range_forecasts] == RANGES
        assert sum(isinstance(r, RangeForecast) for r in read_log(log)) == len(RANGES)

    def test_a_partial_bar_for_today_never_reaches_a_range(self, tmp_path: Path) -> None:
        prices = history()
        partial = with_extra_row(prices, str(FRI), close=1.0, adj_close=1.0, low=0.5)
        clean_log, dirty_log = tmp_path / "clean.jsonl", tmp_path / "dirty.jsonl"
        range_run(clean_log, prices, "2030-03-15 06:00")
        range_run(dirty_log, partial, "2030-03-15 06:00")
        assert dirty_log.read_text() == clean_log.read_text()

    def test_the_log_stays_valid_and_holds_no_return(self, log: Path) -> None:
        range_run(log, history(), "2030-03-14 22:30")
        change = 0.012345
        range_run(log, add_session(history(), FRI, change), "2030-03-15 22:30")
        text = log.read_text()
        assert parse_log(text) == read_log(log)
        assert "0.012345" not in text and "0.0123" not in text
        for line in text.splitlines():
            if '"type":"range_outcome"' in line:
                assert "return" not in line

    def test_dry_run_writes_nothing_and_the_command_reports_ranges(self, log: Path) -> None:
        result = range_run(log, history(), "2030-03-14 22:30", dry_run=True)
        assert len(result.range_forecasts) == len(RANGES) and not log.exists()
        lines: list[str] = []
        code = main(
            ["--log", str(log)],
            now=lambda: at("2030-03-14 22:30"),
            downloader=lambda _symbol: as_yahoo(history()),
            factories=FACTORIES,
            context_factories={},
            report=lines.append,
            min_rows=100,
        )
        assert code == 0
        ranges = [line for line in lines if line.startswith("appended range: ")]
        assert len(ranges) == len(RANGES)
        assert ranges[2].startswith("appended range: ewma_94 expects -")
        assert "(80% range) for the session after 2030-03-14" in ranges[2]
        main(
            ["--log", str(log)],
            now=lambda: at("2030-03-15 22:30"),
            downloader=lambda _symbol: as_yahoo(add_session(history(), FRI, 0.30)),
            factories=FACTORIES,
            context_factories={},
            report=lines.append,
            min_rows=100,
        )
        assert "appended range outcome: 2030-03-15 closed above ewma_94's range" in lines


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "nightly.yml"


def says(up: bool) -> PredictorFactory:
    """A stand-in model that always makes the same call."""

    class Fixed:
        name = "fixed"

        def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            rows = len(features)
            return prediction_frame(
                features.index, np.full(rows, up), np.full(rows, 0.6 if up else 0.4)
            )

    return Fixed


def paper_run(
    log: Path,
    now: str,
    *,
    prices: pd.DataFrame | None = None,
    up: int = 4,
    downloader: Callable[[str], pd.DataFrame] | None = None,
    get: Callable[[str], str] | None = None,
    range_factories: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> NightlyResult:
    """A run with the baselines and stand-ins for the paper strategy's six models,
    the first `up` of which say up."""
    chosen = calm() if prices is None else prices
    votes = {name: says(position < up) for position, name in enumerate(RULE_MODELS)}
    return run_nightly(
        now=at(now),
        log_path=log,
        downloader=downloader or yahoo_downloader(chosen),
        get=get or sec_getter(),
        factories={**BASELINE_FACTORIES, **{name: votes[name] for name in RULE_MODELS[:3]}},
        context_factories={name: votes[name] for name in RULE_MODELS[3:]},
        range_factories={} if range_factories is None else range_factories,
        dry_run=dry_run,
        min_rows=100,
    )


def calm() -> pd.DataFrame:
    """Sessions through Thursday whose last 40 moves are small, so the gate is open."""
    full = history()
    prices = full.iloc[:-40]
    for position, stamp in enumerate(full.index[-40:]):
        prices = add_session(prices, stamp.date(), 0.002 if position % 2 else -0.001)
    return prices


def gate_is_open(prices: pd.DataFrame) -> bool:
    """The gate for the session after the last one, worked out independently."""
    ewma = build_volatility_dataset(prices).features["ewma_vol_94"].to_numpy()
    return bool(ewma[-1] <= np.percentile(ewma[:-1], GATE_PERCENTILE))


def jolted() -> pd.DataFrame:
    """Sessions through Tuesday, the last three with very large moves."""
    prices = history()
    for session, change in ((FRI, 0.15), (MON, -0.15), (TUE, 0.15)):
        prices = add_session(prices, session, change)
    return prices


def entries_in(log: Path) -> list[PaperStrategyEntry]:
    return [r for r in read_log(log) if isinstance(r, PaperStrategyEntry)]


CONTEXT_THREE = tuple(RULE_MODELS[3:])


class TestPaperStrategy:
    """Holding or In cash for the next session, once all six models have called."""

    @pytest.mark.parametrize(("up", "holding"), [(6, True), (4, True), (3, False), (0, False)])
    def test_the_entry_follows_the_six_calls_made_in_the_same_run(
        self, log: Path, up: int, holding: bool
    ) -> None:
        assert gate_is_open(calm())
        result = paper_run(log, EVENING, up=up)
        assert result.paper_entries == (PaperStrategyEntry(at(EVENING), THU, holding, up, True),)
        assert result.paper_waiting_for == ()
        assert read_log(log)[-1] == result.paper_entries[0]

    def test_a_closed_gate_means_cash_even_when_all_six_say_up(self, log: Path) -> None:
        prices = jolted()
        assert not gate_is_open(prices)
        result = paper_run(log, "2030-03-19 22:30", prices=prices, up=6)
        assert result.paper_entries == (
            PaperStrategyEntry(at("2030-03-19 22:30"), TUE, False, 6, False),
        )

    def test_the_gate_looks_only_at_sessions_whose_next_return_is_known(
        self, log: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[np.ndarray] = []
        real = nightly_module.gate_threshold

        def watching(values: np.ndarray) -> float:
            seen.append(np.array(values))
            return real(values)

        monkeypatch.setattr(nightly_module, "gate_threshold", watching)
        prices = jolted()
        paper_run(log, "2030-03-19 22:30", prices=prices)
        ewma = build_volatility_dataset(prices).features["ewma_vol_94"].to_numpy()
        assert len(seen) == 1
        # Every session but the latest, whose next return has not happened yet.
        assert np.array_equal(seen[0], ewma[:-1])

    def test_a_partial_bar_for_today_never_reaches_the_gate(self, tmp_path: Path) -> None:
        prices = calm()
        partial = with_extra_row(prices, str(FRI), close=1.0, adj_close=1.0, low=0.5)
        clean_log, dirty_log = tmp_path / "clean.jsonl", tmp_path / "dirty.jsonl"
        paper_run(clean_log, EARLY_MORNING, prices=prices)
        paper_run(
            dirty_log,
            EARLY_MORNING,
            downloader=yahoo_downloader(partial, context_for(prices)),
        )
        assert len(entries_in(clean_log)) == 1
        assert dirty_log.read_text() == clean_log.read_text()

    def test_a_night_with_a_call_missing_writes_no_entry(self, log: Path) -> None:
        result = paper_run(log, EVENING, **sec_download_fails(calm()))
        # The baselines and the three technical models are written as usual.
        assert [p.model for p in result.predictions] == [*BASELINE_FACTORIES, *RULE_MODELS[:3]]
        assert result.skipped == CONTEXT_THREE
        assert result.paper_entries == ()
        assert result.paper_waiting_for == CONTEXT_THREE
        assert entries_in(log) == []
        assert not any('"paper_strategy"' in line for line in log.read_text().splitlines())

    @pytest.mark.parametrize("failure", [index_download_fails, sec_download_fails])
    def test_whatever_stopped_the_context_calls_stops_the_entry(
        self, log: Path, failure: Callable[[pd.DataFrame], dict[str, Any]]
    ) -> None:
        result = paper_run(log, EVENING, **failure(calm()))
        assert result.paper_entries == () and entries_in(log) == []

    def test_before_the_cut_off_there_is_no_entry_yet(self, log: Path) -> None:
        early = paper_run(log, "2030-03-14 18:00")
        assert early.paper_entries == () and early.paper_waiting_for == CONTEXT_THREE
        later = paper_run(log, "2030-03-14 20:00")
        assert [p.model for p in later.predictions] == list(CONTEXT_THREE)
        assert [e.made_at for e in later.paper_entries] == [at("2030-03-14 20:00")]

    def test_a_missed_session_is_never_filled_in_afterwards(self, log: Path) -> None:
        paper_run(log, EVENING, **sec_download_fails(calm()))
        prices = add_session(calm(), FRI, +0.02)
        friday = paper_run(log, "2030-03-15 22:30", prices=prices)
        assert [o.session for o in friday.outcomes] == [FRI]
        assert [e.based_on_session for e in friday.paper_entries] == [FRI]
        prices = add_session(prices, MON, -0.01)
        paper_run(log, "2030-03-18 22:30", prices=prices)
        # Thursday's entry (for Friday's session) is missing for good.
        assert [e.based_on_session for e in entries_in(log)] == [FRI, MON]
        record = live_record(read_log(log))
        assert (record.sessions, record.missed) == (1, 0)

    def test_a_session_missed_after_the_first_entry_is_counted_as_missed(self, log: Path) -> None:
        paper_run(log, EVENING)
        prices = add_session(calm(), FRI, +0.02)
        paper_run(log, "2030-03-15 22:30", prices=prices, **sec_download_fails(prices))
        prices = add_session(prices, MON, +0.03)
        paper_run(log, "2030-03-18 22:30", prices=prices)
        prices = add_session(prices, TUE, -0.01)
        paper_run(log, "2030-03-19 22:30", prices=prices)
        # No entry was made on Friday night; Tuesday's is waiting for its outcome.
        assert [e.based_on_session for e in entries_in(log)] == [THU, MON, TUE]
        record = live_record(read_log(log))
        # Friday's and Tuesday's sessions are compared; Monday's has no entry.
        assert (record.sessions, record.missed) == (2, 1)
        assert record.went_up == 1  # Friday up, Tuesday not; Monday's rise is left out

    def test_a_retry_before_the_open_that_completes_the_six_writes_the_entry(
        self, log: Path
    ) -> None:
        paper_run(log, EVENING, **sec_download_fails(calm()))
        retry = paper_run(log, EARLY_MORNING)
        assert [p.model for p in retry.predictions] == list(CONTEXT_THREE)
        assert retry.paper_entries == (PaperStrategyEntry(at(EARLY_MORNING), THU, True, 4, True),)
        assert retry.paper_waiting_for == ()

    @pytest.mark.parametrize("when", ["2030-03-15 09:30", "2030-03-15 10:00", "2030-03-15 17:00"])
    def test_an_entry_is_not_written_once_the_next_session_could_have_opened(
        self, log: Path, when: str
    ) -> None:
        # All six calls are on record but the entry is not (as on the strategy's
        # first day, or after an interrupted run).
        paper_run(log, EVENING)
        without_entry = "".join(log.read_text().splitlines(keepends=True)[:-1])
        log.write_text(without_entry)
        assert entries_in(log) == []
        with pytest.raises(NightlyError, match="too late"):
            paper_run(log, when)
        assert log.read_text() == without_entry

    def test_the_log_refuses_a_late_entry_even_if_the_job_tried(self, log: Path) -> None:
        paper_run(log, EVENING)
        without_entry = "".join(log.read_text().splitlines(keepends=True)[:-1])
        log.write_text(without_entry)
        late = PaperStrategyEntry(at("2030-03-15 10:00"), THU, True, 4, True)
        with pytest.raises(LogError, match="after the next session could have opened"):
            append_records(log, [late])
        assert log.read_text() == without_entry

    def test_with_the_calls_on_record_a_run_before_the_open_adds_just_the_entry(
        self, log: Path
    ) -> None:
        paper_run(log, EVENING)
        log.write_text("".join(log.read_text().splitlines(keepends=True)[:-1]))
        result = paper_run(log, EARLY_MORNING)
        assert result.predictions == ()
        assert result.records == [PaperStrategyEntry(at(EARLY_MORNING), THU, True, 4, True)]

    def test_running_again_adds_nothing(self, log: Path) -> None:
        paper_run(log, EVENING)
        before = log.read_bytes()
        again = paper_run(log, EARLY_MORNING)
        assert again.records == [] and log.read_bytes() == before

    def test_the_entry_is_written_after_everything_it_depends_on(self, log: Path) -> None:
        paper_run(log, EVENING, range_factories=default_range_factories())
        prices = add_session(calm(), FRI, +0.01)
        paper_run(log, "2030-03-15 22:30", prices=prices, range_factories=default_range_factories())
        kinds = [type(r).__name__ for r in read_log(log)]
        second_run = kinds[kinds.index("Outcome") :]
        assert second_run == [
            "Outcome",
            *["RangeOutcome"] * 5,
            *["Prediction"] * 9,
            *["RangeForecast"] * 5,
            "PaperStrategyEntry",
        ]

    def test_without_all_six_models_registered_there_is_no_paper_strategy(self, log: Path) -> None:
        result = full_run(log, EVENING)  # the real six predictors and one context model
        assert result.paper_entries == () and result.paper_waiting_for == ()

    def test_dry_run_writes_nothing(self, log: Path) -> None:
        result = paper_run(log, EVENING, dry_run=True)
        assert len(result.paper_entries) == 1
        assert not log.exists()

    def run_main(self, log: Path, now: str, *arguments: str, **overrides: Any) -> tuple[int, str]:
        votes = {name: says(position < 5) for position, name in enumerate(RULE_MODELS)}
        lines: list[str] = []
        code = main(
            ["--log", str(log), *arguments],
            now=lambda: at(now),
            downloader=overrides.get("downloader", yahoo_downloader(calm())),
            get=overrides.get("get", sec_getter()),
            factories={**BASELINE_FACTORIES, **{name: votes[name] for name in RULE_MODELS[:3]}},
            context_factories={name: votes[name] for name in RULE_MODELS[3:]},
            range_factories={},
            report=lines.append,
            min_rows=100,
        )
        return code, "\n".join(lines)

    def test_the_command_reports_the_entry(self, log: Path) -> None:
        code, output = self.run_main(log, EVENING)
        assert code == 0
        assert (
            "appended paper strategy: Holding for the session after 2030-03-14 "
            "(5 of 6 models up, gate open)"
        ) in output
        code, output = self.run_main(log.with_name("dry.jsonl"), EVENING, "--dry-run")
        assert "would append paper strategy: Holding" in output

    def test_the_command_says_loudly_when_there_is_no_entry(self, log: Path) -> None:
        code, output = self.run_main(log, EVENING, **sec_download_fails(calm()))
        assert code == EXIT_CONTEXT_SKIPPED
        assert (
            "NO PAPER STRATEGY ENTRY for the session after 2030-03-14: no call from "
            "logistic_ctx, xgboost_ctx, lstm_ctx"
        ) in output
        assert "paper strategy: Holding" not in output and "paper strategy: In cash" not in output


class TestWhoMayWriteTheLog:
    """A local run cannot write the repository's own log; only the workflow can."""

    @pytest.fixture
    def tracked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        folder = tmp_path / "repo" / "predictions"
        folder.mkdir(parents=True)
        monkeypatch.setattr(prediction_log, "TRACKED_LOG_DIR", folder)
        monkeypatch.delenv(prediction_log.LOG_WRITER_ENV, raising=False)
        return folder / "nvda.jsonl"

    def test_a_run_is_refused_before_anything_is_downloaded(self, tracked: Path) -> None:
        def no_download(_symbol: str) -> pd.DataFrame:
            raise AssertionError("nothing should be downloaded")

        with pytest.raises(LogError, match="only the nightly workflow may write"):
            run_nightly(
                now=at("2030-03-15 06:00"),
                log_path=tracked,
                downloader=no_download,
                factories=FACTORIES,
                context_factories={},
                range_factories={},
                min_rows=100,
            )
        assert not tracked.exists()

    def test_the_command_fails_loudly_and_writes_nothing(self, tracked: Path) -> None:
        lines: list[str] = []
        code = main(
            ["--log", str(tracked)],
            now=lambda: at("2030-03-15 06:00"),
            downloader=lambda _symbol: as_yahoo(history()),
            factories=FACTORIES,
            context_factories={},
            range_factories={},
            report=lines.append,
            min_rows=100,
        )
        assert code == 1
        assert lines[0].startswith("NIGHTLY RUN FAILED, nothing was written: refusing to write")
        assert not tracked.exists()

    def test_a_dry_run_is_still_allowed(self, tracked: Path) -> None:
        outcomes, predictions = nightly(tracked, history(), "2030-03-15 06:00", dry_run=True)
        assert outcomes == [] and [p.model for p in predictions] == MODELS
        assert not tracked.exists()

    def test_the_workflow_run_writes_it(
        self, tracked: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(prediction_log.LOG_WRITER_ENV, prediction_log.NIGHTLY_WORKFLOW)
        _, predictions = nightly(tracked, history(), "2030-03-15 06:00")
        assert read_log(tracked) == predictions

    def test_only_the_nightly_workflows_own_step_carries_the_mark(self) -> None:
        workflows = sorted(WORKFLOW.parent.glob("*.yml"))
        holders = [w.name for w in workflows if prediction_log.LOG_WRITER_ENV in w.read_text()]
        assert holders == ["nightly.yml"]
        text = WORKFLOW.read_text()
        assert text.count(prediction_log.LOG_WRITER_ENV) == 1
        step = text.split("- name: Record outcomes and append predictions", 1)[1]
        step = step.split("\n      - name:", 1)[0]
        assert f"{prediction_log.LOG_WRITER_ENV}: {prediction_log.NIGHTLY_WORKFLOW}" in step

    def test_the_checks_workflow_compares_the_log_with_the_commit_before(self) -> None:
        checks = (WORKFLOW.parent / "checks.yml").read_text()
        assert "fetch-depth: 2" in checks
        assert "git show HEAD^:predictions/nvda.jsonl" in checks
        assert '--previous "$RUNNER_TEMP/previous.jsonl"' in checks


class TestDefaults:
    def test_default_predictors_are_the_baselines_and_measured_models(self) -> None:
        assert list(default_factories()) == [*BASELINE_FACTORIES, *MODEL_FACTORIES]
        assert list(default_factories())[:4] == ["always_up", "persistence", "ma_200", "logistic"]

    def test_the_real_registry_runs_end_to_end(self, log: Path) -> None:
        result = run_nightly(
            now=at("2030-03-15 06:00"),
            log_path=log,
            downloader=yahoo_downloader(history()),
            get=sec_getter(),
            min_rows=100,
        )
        names = [*default_factories(), *CONTEXT_MODEL_FACTORIES]
        assert [p.model for p in result.predictions] == names
        assert names[-3:] == ["logistic_ctx", "xgboost_ctx", "lstm_ctx"]
        assert result.skipped == ()
        assert [f.model for f in result.range_forecasts] == [
            "constant",
            "rolling_21",
            "ewma_94",
            "har",
            "xgboost_vol",
        ]
        assert list(default_range_factories()) == [f.model for f in result.range_forecasts]
        # The real registry holds all six of the paper strategy's models.
        (entry,) = result.paper_entries
        up = sum(p.predicted_up for p in result.predictions if p.model in RULE_MODELS)
        assert entry.models_up == up and entry.based_on_session == THU
        assert read_log(log)[-1] == entry
        for prediction in result.predictions:
            if prediction.model in {*MODEL_FACTORIES, *CONTEXT_MODEL_FACTORIES}:
                assert prediction.probability_up is not None
                assert 0 < prediction.probability_up < 1
            else:
                assert prediction.probability_up is None


class TestWorkflowFile:
    """Guards on the workflow text, so a later edit cannot quietly weaken it."""

    def test_the_log_is_verified_against_the_committed_copy_before_any_commit(self) -> None:
        text = WORKFLOW.read_text()
        verify = text.index("prediction_log verify")
        assert '--previous "$RUNNER_TEMP/previous.jsonl"' in text
        assert text.index('cp "$LOG" "$RUNNER_TEMP/previous.jsonl"') < text.index("nvda.nightly")
        assert text.index("nvda.nightly") < verify < text.index("git commit")

    def test_only_the_log_file_is_committed(self) -> None:
        text = WORKFLOW.read_text()
        assert 'git add -- "$LOG"' in text
        assert text.count("git add") == 1
        assert "git add -A" not in text and "git add ." not in text
        assert "LOG: predictions/nvda.jsonl" in text
        assert "--force" not in text

    def test_two_runs_land_in_the_window_whatever_githubs_delay(self) -> None:
        # GitHub started scheduled runs 7 to 12 hours late (2026-11-19 to 24), and
        # the delay varies. For any delay up to a whole day, in winter and summer
        # time, at least two runs (a call and a retry) must start after the 20:00
        # cut-off the context models wait for and before the next open.
        text = WORKFLOW.read_text()
        slots = [
            time(int(hour), int(minute))
            for minute, hour in re.findall(r'cron: "(\d+) (\d+) \* \* \*"', text)
        ]
        assert len(slots) == text.count("cron:") == 8
        sessions = [date(2026, 1, 12) + timedelta(days=i) for i in range(5)]
        sessions += [date(2026, 7, 13) + timedelta(days=i) for i in range(5)]
        for session in sessions:
            earliest = information_cutoff(session)
            _, deadline = prediction_window(session)
            for quarter_hours in range(24 * 4 + 1):
                delay = timedelta(minutes=15 * quarter_hours)
                starts = [
                    datetime.combine(session + timedelta(days=shift), slot, tzinfo=UTC) + delay
                    for shift in (-1, 0, 1)
                    for slot in slots
                ]
                inside = [start for start in starts if earliest <= start < deadline]
                assert len(inside) >= 2, (session, delay)

    def test_has_a_clear_bot_message(self) -> None:
        text = WORKFLOW.read_text()
        assert 'git commit -m "bot: append NVDA outcomes and predictions' in text
        assert "continue-on-error" not in text

    def test_a_skipped_context_call_is_committed_first_and_then_fails_the_job(self) -> None:
        text = WORKFLOW.read_text()
        # Only exit code 3 is let through; any other failure stops before the commit.
        assert 'if [ "$code" -ne 0 ] && [ "$code" -ne 3 ]; then' in text
        assert 'echo "code=$code" >> "$GITHUB_OUTPUT"' in text
        failing_step = text.index("- name: Fail if the context models made no call")
        assert text.index("prediction_log verify") < text.index("git push") < failing_step
        last_step = text[failing_step:]
        assert "if: steps.nightly.outputs.code == '3'" in last_step
        assert "exit 1" in last_step
        assert EXIT_CONTEXT_SKIPPED == 3

    def test_sec_is_told_who_is_asking_only_through_a_secret(self) -> None:
        text = WORKFLOW.read_text()
        assert "SEC_USER_AGENT: ${{ secrets.SEC_USER_AGENT }}" in text
        assert "@" not in text.replace("actions/checkout@", "").replace(
            "astral-sh/setup-uv@", ""
        ).replace("github-actions[bot]@users.noreply.github.com", "")
