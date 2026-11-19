"""The nightly job: outcomes, next-session calls, and the cases where it must refuse."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.backtest import BASELINE_FACTORIES, PredictorFactory
from markets_lab.analysis.nvda.features import build_dataset
from markets_lab.analysis.nvda.nightly import (
    NightlyError,
    default_factories,
    main,
    run_nightly,
)
from markets_lab.analysis.nvda.prediction_log import (
    Outcome,
    Prediction,
    parse_log,
    read_log,
    scored_predictions,
)
from markets_lab.analysis.nvda.predictors import prediction_frame
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


class TestDefaults:
    def test_default_predictors_are_the_baselines_and_measured_models(self) -> None:
        assert list(default_factories()) == ["always_up", "persistence", "ma_200", "logistic"]

    def test_the_real_registry_runs_end_to_end(self, log: Path) -> None:
        result = run_nightly(
            now=at("2030-03-15 06:00"),
            log_path=log,
            downloader=lambda _symbol: as_yahoo(history()),
            min_rows=100,
        )
        assert [p.model for p in result.predictions] == [
            "always_up",
            "persistence",
            "ma_200",
            "logistic",
        ]
        logistic = result.predictions[-1]
        assert logistic.probability_up is not None
        assert 0 < logistic.probability_up < 1


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "nightly.yml"


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

    def test_runs_before_the_market_opens_and_has_a_clear_bot_message(self) -> None:
        text = WORKFLOW.read_text()
        assert 'cron: "30 10 * * *"' in text
        assert 'cron: "15 12 * * *"' in text
        assert 'git commit -m "bot: append NVDA outcomes and predictions' in text
        assert "continue-on-error" not in text
