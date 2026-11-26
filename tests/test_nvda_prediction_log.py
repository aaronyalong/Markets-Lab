"""The prediction log: append-only, tamper-evident, direction only."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from markets_lab.analysis.nvda.paper_strategy.rule import RULE_MODELS
from markets_lab.analysis.nvda.prediction_log import (
    GENESIS_HASH,
    LogError,
    Outcome,
    PaperStrategyEntry,
    Prediction,
    RangeForecast,
    RangeOutcome,
    Record,
    append_records,
    check_append_only,
    main,
    parse_log,
    prediction_window,
    range_position,
    read_log,
    scored_paper_strategy,
    scored_predictions,
    scored_ranges,
)
from price_frames import at

THU, FRI, MON, TUE = date(2030, 3, 14), date(2030, 3, 15), date(2030, 3, 18), date(2030, 3, 19)


def prediction(
    based_on: date = THU,
    model: str = "logistic",
    *,
    up: bool = True,
    probability: float | None = 0.53,
    made: str | None = None,
) -> Prediction:
    made_at = at(made) if made else at(f"{based_on} 18:00")
    return Prediction(made_at, based_on, model, up, probability)


def outcome(session: date, previous: date, *, up: bool, recorded: str | None = None) -> Outcome:
    recorded_at = at(recorded) if recorded else at(f"{session} 17:30")
    return Outcome(recorded_at, session, previous, up)


def week() -> list[Record]:
    """Three sessions of two models, in the order the nightly job writes them."""
    return [
        prediction(THU, "always_up", probability=None),
        prediction(THU, "logistic", up=False, probability=0.47),
        outcome(FRI, THU, up=True),
        prediction(FRI, "always_up", probability=None),
        prediction(FRI, "logistic", up=True, probability=0.52),
        outcome(MON, FRI, up=False, recorded="2030-03-18 17:30"),
        prediction(MON, "always_up", probability=None),
        prediction(MON, "logistic", up=False, probability=0.49),
    ]


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "predictions" / "nvda.jsonl"
    append_records(path, week())
    return path


def lines_of(path: Path) -> list[str]:
    return path.read_text().splitlines()


class TestRoundTrip:
    def test_what_is_appended_is_what_is_read(self, log: Path) -> None:
        assert read_log(log) == week()

    def test_a_missing_file_is_an_empty_log(self, tmp_path: Path) -> None:
        assert read_log(tmp_path / "none.jsonl") == []
        assert parse_log("") == []

    def test_lines_form_a_chain_from_the_genesis_hash(self, log: Path) -> None:
        records = [json.loads(line) for line in lines_of(log)]
        assert records[0]["prev_hash"] == GENESIS_HASH
        for before, after in itertools.pairwise(records):
            assert after["prev_hash"] == before["hash"]
        assert len({r["hash"] for r in records}) == len(records)

    def test_appending_in_two_goes_gives_the_same_file_as_in_one(
        self, log: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.jsonl"
        append_records(other, week()[:3])
        append_records(other, week()[3:])
        assert other.read_text() == log.read_text()

    def test_times_are_stored_in_utc_to_the_second(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        made = at("2030-03-14 18:00").replace(microsecond=123_456)
        append_records(path, [Prediction(made, THU, "logistic", True, 0.5)])
        assert json.loads(lines_of(path)[0])["made_at"] == "2030-03-14T22:00:00Z"
        assert read_log(path)[0] == prediction(THU, probability=0.5)

    def test_probability_is_kept_to_four_decimals(self) -> None:
        assert prediction(probability=0.123456789).probability_up == 0.1235


class TestDirectionOnly:
    """The log is published. It must not grow fields that carry prices or returns."""

    def test_records_have_exactly_the_approved_fields(self, log: Path) -> None:
        fields = {
            json.loads(line)["type"]: set(json.loads(line)) - {"v", "type", "prev_hash", "hash"}
            for line in lines_of(log)
        }
        assert fields == {
            "prediction": {
                "made_at",
                "based_on_session",
                "model",
                "predicted_up",
                "probability_up",
            },
            "outcome": {"recorded_at", "session", "previous_session", "went_up"},
        }

    def test_an_extra_field_is_refused(self, log: Path) -> None:
        first = json.loads(lines_of(log)[0])
        first["close"] = 151.67
        tampered = json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n"
        with pytest.raises(LogError, match=r"line 1: unexpected fields \['close'\]"):
            parse_log(tampered)


class TestTamperEvidence:
    def edit(self, log: Path, number: int, **changes: object) -> str:
        lines = lines_of(log)
        record = json.loads(lines[number - 1])
        record.update(changes)
        lines[number - 1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return "\n".join(lines) + "\n"

    def test_changing_a_past_call_is_detected(self, log: Path) -> None:
        with pytest.raises(LogError, match="line 2: content does not match its hash"):
            parse_log(self.edit(log, 2, predicted_up=True))

    def test_changing_a_past_probability_is_detected(self, log: Path) -> None:
        with pytest.raises(LogError, match="line 5: content does not match its hash"):
            parse_log(self.edit(log, 5, probability_up=0.9))

    def test_changing_a_past_outcome_is_detected(self, log: Path) -> None:
        with pytest.raises(LogError, match="line 3: content does not match its hash"):
            parse_log(self.edit(log, 3, went_up=False))

    def test_recomputing_one_lines_hash_breaks_the_next_link(self, log: Path) -> None:
        # Flip the call, then give the line a correct hash for its new content.
        lines = lines_of(log)
        honest = read_log(log)
        forged = Prediction(at("2030-03-14 18:00"), THU, "logistic", True, 0.47)
        scratch = log.with_name("scratch.jsonl")
        append_records(scratch, [honest[0], forged])
        lines[1] = lines_of(scratch)[1]
        with pytest.raises(LogError, match="line 3: chain broken"):
            parse_log("\n".join(lines) + "\n")

    def test_removing_a_line_is_detected(self, log: Path) -> None:
        lines = lines_of(log)
        del lines[3]
        with pytest.raises(LogError, match="line 4: chain broken"):
            parse_log("\n".join(lines) + "\n")

    def test_reordering_lines_is_detected(self, log: Path) -> None:
        lines = lines_of(log)
        lines[0], lines[1] = lines[1], lines[0]
        with pytest.raises(LogError, match="line 1: chain broken"):
            parse_log("\n".join(lines) + "\n")

    def test_reformatting_a_line_is_detected(self, log: Path) -> None:
        lines = lines_of(log)
        lines[0] = json.dumps(json.loads(lines[0]), indent=None, separators=(", ", ": "))
        with pytest.raises(LogError, match=r"line 1: .*canonical form"):
            parse_log("\n".join(lines) + "\n")

    def test_a_cut_off_last_line_is_detected(self, log: Path) -> None:
        with pytest.raises(LogError, match="does not end with a newline"):
            parse_log(log.read_text()[:-20])

    @pytest.mark.parametrize("junk", ["not json\n", "[1, 2]\n", '{"v": 2, "type": "prediction"}\n'])
    def test_junk_is_refused(self, junk: str) -> None:
        with pytest.raises(LogError, match="line 1"):
            parse_log(junk)


class TestAppendOnly:
    def test_adding_lines_extends_the_previous_version(self, log: Path) -> None:
        before = log.read_text()
        append_records(log, [outcome(TUE, MON, up=True, recorded="2030-03-19 17:30")])
        check_append_only(before, log.read_text())
        assert log.read_text().startswith(before)
        assert len(lines_of(log)) == 9

    def test_a_fully_rewritten_chain_is_valid_on_its_own_but_caught_against_the_old_one(
        self, log: Path, tmp_path: Path
    ) -> None:
        # Rewrite history from scratch with one call flipped: every hash is correct.
        rewritten = week()
        rewritten[1] = prediction(THU, "logistic", up=True, probability=0.47)
        forged = tmp_path / "forged.jsonl"
        append_records(forged, rewritten)
        assert len(parse_log(forged.read_text())) == 8
        with pytest.raises(LogError, match="not an extension of its previous version"):
            check_append_only(log.read_text(), forged.read_text())

    def test_truncation_is_caught(self, log: Path) -> None:
        shorter = "\n".join(lines_of(log)[:-1]) + "\n"
        with pytest.raises(LogError, match="not an extension"):
            check_append_only(log.read_text(), shorter)

    def test_a_refused_append_leaves_the_file_untouched(self, log: Path) -> None:
        before = log.read_bytes()
        late = prediction(MON, "xgboost", made="2030-03-19 10:00")
        with pytest.raises(LogError):
            append_records(log, [prediction(TUE, made="2030-03-19 18:00"), late])
        assert log.read_bytes() == before
        assert [p.name for p in log.parent.iterdir()] == ["nvda.jsonl"]

    def test_appending_to_a_corrupt_log_is_refused(self, log: Path) -> None:
        log.write_text(log.read_text().replace('"predicted_up":false', '"predicted_up":true', 1))
        with pytest.raises(LogError, match="content does not match its hash"):
            append_records(log, [outcome(TUE, MON, up=True, recorded="2030-03-19 17:30")])

    def test_appending_nothing_does_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        append_records(path, [])
        assert not path.exists()


class TestTimingRules:
    """A prediction must be written before the session it is about."""

    def test_window_runs_from_the_close_to_the_next_weekday_open(self) -> None:
        earliest, deadline = prediction_window(THU)
        assert earliest == at("2030-03-14 16:00")
        assert deadline == at("2030-03-15 09:30")

    def test_a_friday_session_has_until_monday_morning(self) -> None:
        assert prediction_window(FRI)[1] == at("2030-03-18 09:30")

    def test_window_follows_new_york_clocks_across_daylight_saving(self) -> None:
        winter = prediction_window(date(2030, 1, 15))[1]
        summer = prediction_window(date(2030, 7, 15))[1]
        assert winter.astimezone(UTC).hour == 14
        assert summer.astimezone(UTC).hour == 13

    @pytest.mark.parametrize("made", ["2030-03-14 16:00", "2030-03-14 23:59", "2030-03-15 09:29"])
    def test_inside_the_window_is_accepted(self, tmp_path: Path, made: str) -> None:
        append_records(tmp_path / "log.jsonl", [prediction(THU, made=made)])

    @pytest.mark.parametrize("made", ["2030-03-15 09:30", "2030-03-15 12:00", "2030-03-18 08:00"])
    def test_at_or_after_the_next_open_is_refused(self, tmp_path: Path, made: str) -> None:
        with pytest.raises(LogError, match="a prediction must come before its session"):
            append_records(tmp_path / "log.jsonl", [prediction(THU, made=made)])

    def test_before_the_session_it_is_based_on_closed_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match="had closed"):
            append_records(tmp_path / "log.jsonl", [prediction(THU, made="2030-03-14 15:59")])

    def test_a_prediction_cannot_follow_the_outcome_it_predicts(self, tmp_path: Path) -> None:
        # Monday's outcome is on file. A prediction for Monday (based on Friday) is
        # then offered with a clock that claims Friday evening. The claimed time is
        # inside its window, but the order in the file decides, not the clock.
        path = tmp_path / "log.jsonl"
        append_records(
            path,
            [outcome(FRI, THU, up=True), outcome(MON, FRI, up=True, recorded="2030-03-18 17:30")],
        )
        backdated = prediction(FRI, made="2030-03-15 18:00")
        with pytest.raises(
            LogError,
            match=r"based on 2030-03-15, but the outcome of a later session \(2030-03-18\)",
        ):
            append_records(path, [backdated])

    def test_records_cannot_go_back_in_time(self, log: Path) -> None:
        early = outcome(TUE, MON, up=True, recorded="2030-03-19 17:30")
        append_records(log, [early])
        with pytest.raises(LogError, match="before the record above it"):
            append_records(log, [prediction(TUE, made="2030-03-19 17:00")])

    def test_outcome_cannot_be_recorded_before_the_close(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match="recorded before its close"):
            append_records(
                tmp_path / "log.jsonl", [outcome(FRI, THU, up=True, recorded="2030-03-15 15:59")]
            )


class TestSequenceRules:
    def test_one_prediction_per_model_per_session(self, log: Path) -> None:
        with pytest.raises(LogError, match="logistic already has a prediction based on 2030-03-18"):
            append_records(log, [prediction(MON, "logistic", up=True, made="2030-03-18 19:00")])

    def test_a_new_model_can_join_later(self, log: Path) -> None:
        append_records(log, [prediction(MON, "xgboost", made="2030-03-18 19:00")])
        assert len(read_log(log)) == 9

    def test_outcomes_must_be_consecutive(self, log: Path) -> None:
        skipping = outcome(date(2030, 3, 20), TUE, up=True, recorded="2030-03-20 17:30")
        with pytest.raises(LogError, match="the last recorded session is 2030-03-18"):
            append_records(log, [skipping])

    def test_an_outcome_cannot_be_recorded_twice(self, log: Path) -> None:
        with pytest.raises(LogError, match="the last recorded session is 2030-03-18"):
            append_records(log, [outcome(MON, FRI, up=True, recorded="2030-03-18 18:00")])


class TestRecordValidation:
    @pytest.mark.parametrize("name", ["Logistic", "x y", "", "1abc", "a" * 41, "drop;table"])
    def test_model_names_are_restricted(self, name: str) -> None:
        with pytest.raises(LogError, match="model name"):
            prediction(model=name)

    @pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), "0.5", True])
    def test_probability_must_be_a_number_from_zero_to_one(self, probability: object) -> None:
        with pytest.raises(LogError, match="probability_up"):
            Prediction(at("2030-03-14 18:00"), THU, "logistic", True, probability)  # type: ignore[arg-type]

    def test_direction_must_be_a_boolean(self) -> None:
        with pytest.raises(LogError, match="predicted_up"):
            Prediction(at("2030-03-14 18:00"), THU, "logistic", 1, 0.5)  # type: ignore[arg-type]
        with pytest.raises(LogError, match="went_up"):
            Outcome(at("2030-03-15 17:30"), FRI, THU, "yes")  # type: ignore[arg-type]

    def test_times_must_carry_a_timezone(self) -> None:
        with pytest.raises(LogError, match="timezone-aware"):
            Prediction(datetime(2030, 3, 14, 18, 0), THU, "logistic", True, 0.5)

    def test_sessions_are_weekdays(self) -> None:
        saturday = date(2030, 3, 16)
        with pytest.raises(LogError, match="weekend"):
            Prediction(at("2030-03-16 18:00"), saturday, "logistic", True, 0.5)
        with pytest.raises(LogError, match="weekend"):
            Outcome(at("2030-03-18 17:30"), MON, saturday, True)

    def test_an_outcome_compares_a_session_with_an_earlier_one(self) -> None:
        with pytest.raises(LogError, match="previous_session must be before session"):
            Outcome(at("2030-03-15 17:30"), FRI, FRI, True)


class TestScoring:
    def test_predictions_are_paired_with_the_next_sessions_outcome(self, log: Path) -> None:
        scored = scored_predictions(read_log(log))
        assert len(scored) == 6
        logistic = scored[scored["model"] == "logistic"].reset_index(drop=True)
        assert logistic["based_on_session"].tolist() == [THU, FRI, MON]
        assert logistic["target_session"].tolist()[:2] == [FRI, MON]
        assert logistic["predicted_up"].tolist() == [False, True, False]
        assert logistic["went_up"].tolist()[:2] == [True, False]
        assert logistic["correct"].tolist()[:2] == [False, False]

    def test_always_up_is_right_exactly_when_the_session_went_up(self, log: Path) -> None:
        scored = scored_predictions(read_log(log))
        always = scored[scored["model"] == "always_up"].reset_index(drop=True)
        assert always["correct"].tolist()[:2] == [True, False]
        assert always["probability_up"].isna().all()

    def test_the_latest_predictions_are_not_scored_yet(self, log: Path) -> None:
        scored = scored_predictions(read_log(log))
        pending = scored[scored["based_on_session"] == MON]
        assert len(pending) == 2
        assert pending["correct"].isna().all()
        assert pending["went_up"].isna().all()
        assert pending["target_session"].isna().all()

    def test_scoring_then_fills_them_in(self, log: Path) -> None:
        append_records(log, [outcome(TUE, MON, up=False, recorded="2030-03-19 17:30")])
        scored = scored_predictions(read_log(log))
        monday = scored[scored["based_on_session"] == MON].set_index("model")
        assert bool(monday.loc["logistic", "correct"]) is True
        assert bool(monday.loc["always_up", "correct"]) is False

    def test_an_empty_log_scores_to_an_empty_table(self) -> None:
        scored = scored_predictions([])
        assert len(scored) == 0
        assert "correct" in scored.columns


class TestCommandLine:
    def run(self, *argv: str) -> tuple[int, str]:
        lines: list[str] = []
        return main(list(argv), report=lines.append), "\n".join(lines)

    def test_valid_log(self, log: Path) -> None:
        code, output = self.run("verify", "--log", str(log))
        assert code == 0
        assert output == (
            "log is valid: 8 records (6 predictions, 2 outcomes, 0 range forecasts, "
            "0 range outcomes, 0 paper strategy entries)"
        )

    def test_valid_extension_of_a_previous_copy(self, log: Path, tmp_path: Path) -> None:
        previous = tmp_path / "previous.jsonl"
        previous.write_text(log.read_text())
        append_records(log, [outcome(TUE, MON, up=True, recorded="2030-03-19 17:30")])
        code, output = self.run("verify", "--log", str(log), "--previous", str(previous))
        assert code == 0
        assert "extends the previous version: 9 records" in output

    def test_tampering_fails_loudly(self, log: Path) -> None:
        log.write_text(log.read_text().replace('"went_up":true', '"went_up":false', 1))
        code, output = self.run("verify", "--log", str(log))
        assert code == 1
        assert output.startswith("LOG VERIFICATION FAILED: line 3")

    def test_rewritten_history_fails_against_the_previous_copy(
        self, log: Path, tmp_path: Path
    ) -> None:
        previous = tmp_path / "previous.jsonl"
        previous.write_text(log.read_text())
        rewritten = week()
        rewritten[1] = prediction(THU, "logistic", up=True, probability=0.47)
        log.unlink()
        append_records(log, rewritten)
        code, output = self.run("verify", "--log", str(log), "--previous", str(previous))
        assert code == 1
        assert "not an extension of its previous version" in output

    def test_missing_log_fails(self, tmp_path: Path) -> None:
        code, output = self.run("verify", "--log", str(tmp_path / "none.jsonl"))
        assert code == 1
        assert "does not exist" in output


def test_helper_times_are_new_york() -> None:
    assert at("2030-03-14 18:00") - at("2030-03-14 16:00") == timedelta(hours=2)
    assert at("2030-03-14 18:00").astimezone(UTC).hour == 22


def forecast(
    based_on: date = THU,
    model: str = "ewma_94",
    *,
    lower: float = -0.031,
    upper: float = 0.034,
    made: str | None = None,
) -> RangeForecast:
    made_at = at(made) if made else at(f"{based_on} 18:00")
    return RangeForecast(made_at, based_on, model, 0.8, lower, upper)


def placed(
    session: date,
    previous: date,
    position: str,
    model: str = "ewma_94",
    *,
    recorded: str | None = None,
) -> RangeOutcome:
    recorded_at = at(recorded) if recorded else at(f"{session} 17:30")
    return RangeOutcome(recorded_at, session, previous, model, position)


def range_week() -> list[Record]:
    """Two sessions with direction calls and ranges, in the order the job writes them."""
    return [
        prediction(THU, "always_up", probability=None),
        forecast(THU, "constant", lower=-0.045, upper=0.047),
        forecast(THU, "ewma_94"),
        outcome(FRI, THU, up=True),
        placed(FRI, THU, "inside", "constant"),
        placed(FRI, THU, "above", "ewma_94"),
        prediction(FRI, "always_up", probability=None),
        forecast(FRI, "constant", lower=-0.045, upper=0.047),
        forecast(FRI, "ewma_94", lower=-0.036, upper=0.039),
    ]


@pytest.fixture
def range_log(tmp_path: Path) -> Path:
    path = tmp_path / "predictions" / "ranges.jsonl"
    append_records(path, range_week())
    return path


class TestRangeRecords:
    def test_what_is_appended_is_what_is_read(self, range_log: Path) -> None:
        assert read_log(range_log) == range_week()

    def test_records_have_exactly_the_approved_fields(self, range_log: Path) -> None:
        by_type = {json.loads(line)["type"]: set(json.loads(line)) for line in lines_of(range_log)}
        common = {"v", "type", "prev_hash", "hash"}
        assert by_type["range_forecast"] == common | {
            "made_at",
            "based_on_session",
            "model",
            "level",
            "lower_return",
            "upper_return",
        }
        assert by_type["range_outcome"] == common | {
            "recorded_at",
            "session",
            "previous_session",
            "model",
            "position",
        }

    def test_what_happened_is_stored_only_as_a_position(self, range_log: Path) -> None:
        # No field can hold the session's return: a range outcome has no number
        # in it at all besides the record version.
        for line in lines_of(range_log):
            record = json.loads(line)
            if record["type"] == "range_outcome":
                numbers = [v for v in record.values() if isinstance(v, int | float)]
                assert numbers == [1]
                assert record["position"] in ("below", "inside", "above")

    def test_bounds_are_kept_to_four_decimals(self) -> None:
        record = forecast(lower=-0.031249, upper=0.034051)
        assert (record.lower_return, record.upper_return) == (-0.0312, 0.0341)

    def test_an_extra_field_is_refused(self, range_log: Path) -> None:
        lines = lines_of(range_log)
        record = json.loads(lines[4])
        assert record["type"] == "range_outcome"
        record["realised_return"] = 0.021
        lines[4] = json.dumps(record)
        with pytest.raises(LogError, match="unexpected fields"):
            parse_log("\n".join(lines) + "\n")

    def test_changing_a_past_range_or_position_is_detected(self, range_log: Path) -> None:
        text = range_log.read_text()
        for old, new in (('"upper_return":0.034', '"upper_return":0.064'), ('"above"', '"inside"')):
            assert old in text
            with pytest.raises(LogError, match="does not match its hash"):
                parse_log(text.replace(old, new, 1))

    @pytest.mark.parametrize(
        ("lower", "upper", "message"),
        [
            (0.02, 0.02, "below upper_return"),
            (0.03, 0.01, "below upper_return"),
            (-1.0, 0.01, "above -1"),
            (float("nan"), 0.01, "must be finite"),
            (-0.01, float("inf"), "must be finite"),
            (True, 0.01, "must be a number"),
            ("-0.01", 0.01, "must be a number"),
        ],
    )
    def test_bounds_must_be_an_ordered_pair_of_real_returns(
        self, lower: Any, upper: Any, message: str
    ) -> None:
        with pytest.raises(LogError, match=message):
            RangeForecast(at("2030-03-14 18:00"), THU, "ewma_94", 0.8, lower, upper)

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.2, True, "0.8"])
    def test_level_must_be_a_share_between_zero_and_one(self, level: Any) -> None:
        with pytest.raises(LogError, match="level must be"):
            RangeForecast(at("2030-03-14 18:00"), THU, "ewma_94", level, -0.03, 0.03)

    def test_position_and_names_are_restricted(self) -> None:
        with pytest.raises(LogError, match="position must be one of below, inside, above"):
            placed(FRI, THU, "outside")
        with pytest.raises(LogError, match="model name"):
            forecast(model="EWMA 94")
        with pytest.raises(LogError, match="model name"):
            placed(FRI, THU, "inside", "EWMA 94")
        with pytest.raises(LogError, match="previous_session must be before session"):
            placed(THU, FRI, "inside")

    def test_range_position_counts_the_bounds_as_inside(self) -> None:
        assert range_position(-0.03, 0.04, -0.0301) == "below"
        assert range_position(-0.03, 0.04, -0.03) == "inside"
        assert range_position(-0.03, 0.04, 0.04) == "inside"
        assert range_position(-0.03, 0.04, 0.0401) == "above"


class TestRangeRules:
    @pytest.mark.parametrize("made", ["2030-03-14 16:00", "2030-03-14 22:30", "2030-03-15 09:29"])
    def test_a_range_written_inside_the_window_is_accepted(self, tmp_path: Path, made: str) -> None:
        append_records(tmp_path / "log.jsonl", [forecast(made=made)])

    @pytest.mark.parametrize("made", ["2030-03-15 09:30", "2030-03-15 12:00", "2030-03-18 08:00"])
    def test_a_range_written_at_or_after_the_next_open_is_refused(
        self, tmp_path: Path, made: str
    ) -> None:
        with pytest.raises(LogError, match="after the next session could have opened"):
            append_records(tmp_path / "log.jsonl", [forecast(made=made)])

    def test_a_range_written_before_its_session_closed_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match="had closed"):
            append_records(tmp_path / "log.jsonl", [forecast(made="2030-03-14 15:59")])

    def test_a_range_cannot_follow_the_outcome_it_forecasts(self, range_log: Path) -> None:
        late = forecast(THU, "har", made="2030-03-15 18:10")
        with pytest.raises(LogError, match="outcome of a later session"):
            append_records(range_log, [late])

    def test_one_range_per_forecaster_per_session(self, range_log: Path) -> None:
        with pytest.raises(LogError, match="ewma_94 already has a range forecast based on"):
            append_records(range_log, [forecast(FRI, "ewma_94", made="2030-03-15 19:00")])

    def test_a_forecaster_and_a_direction_model_may_share_a_name_and_session(
        self, tmp_path: Path
    ) -> None:
        # They are different kinds of record, counted separately.
        append_records(
            tmp_path / "log.jsonl", [prediction(THU, "xgboost"), forecast(THU, "xgboost")]
        )

    def test_a_new_forecaster_can_join_later(self, range_log: Path) -> None:
        append_records(range_log, [forecast(FRI, "har", made="2030-03-15 19:00")])
        assert len(read_log(range_log)) == len(range_week()) + 1

    def test_a_range_outcome_needs_its_forecast(self, range_log: Path) -> None:
        orphan = placed(FRI, THU, "inside", "har", recorded="2030-03-15 19:00")
        with pytest.raises(LogError, match="har has no range forecast based on 2030-03-14"):
            append_records(range_log, [orphan])

    def test_a_range_outcome_needs_the_sessions_outcome_first(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        with pytest.raises(LogError, match="has not been recorded"):
            append_records(path, [forecast(THU), placed(FRI, THU, "inside")])
        # Nor may it name a different session from the one the outcome was for.
        records = [forecast(THU), outcome(FRI, THU, up=True)]
        with pytest.raises(LogError, match="has not been recorded"):
            append_records(
                path, [*records, placed(MON, THU, "inside", recorded="2030-03-18 17:30")]
            )

    def test_a_range_is_scored_once(self, range_log: Path) -> None:
        again = placed(FRI, THU, "inside", "ewma_94", recorded="2030-03-15 19:00")
        with pytest.raises(LogError, match="already has an outcome"):
            append_records(range_log, [again])

    @pytest.mark.parametrize(
        ("up", "lower", "upper", "position"),
        [
            (True, -0.03, 0.04, "below"),  # a rise cannot be below a range that starts below zero
            (True, -0.05, -0.01, "inside"),  # nor inside a range that ends below zero
            (True, -0.05, 0.0, "inside"),
            (False, -0.03, 0.04, "above"),  # a fall cannot be above a range that ends above zero
            (False, -0.03, 0.0, "above"),
            (False, 0.01, 0.04, "inside"),  # nor inside a range that starts above zero
        ],
    )
    def test_a_position_that_contradicts_the_direction_is_refused(
        self, tmp_path: Path, up: bool, lower: float, upper: float, position: str
    ) -> None:
        records = [
            forecast(THU, lower=lower, upper=upper),
            outcome(FRI, THU, up=up),
            placed(FRI, THU, position),
        ]
        with pytest.raises(LogError, match="which cannot be"):
            append_records(tmp_path / "log.jsonl", records)

    @pytest.mark.parametrize(
        ("up", "lower", "upper", "position"),
        [
            (True, -0.03, 0.04, "inside"),
            (True, -0.03, 0.04, "above"),
            (True, 0.01, 0.04, "below"),
            (False, -0.03, 0.04, "inside"),
            (False, -0.03, 0.04, "below"),
            (False, -0.05, -0.01, "above"),
            (False, 0.0, 0.04, "inside"),  # a flat session counts as not up
        ],
    )
    def test_a_position_that_agrees_with_the_direction_is_accepted(
        self, tmp_path: Path, up: bool, lower: float, upper: float, position: str
    ) -> None:
        records = [
            forecast(THU, lower=lower, upper=upper),
            outcome(FRI, THU, up=up),
            placed(FRI, THU, position),
        ]
        append_records(tmp_path / "log.jsonl", records)

    def test_a_refused_range_leaves_the_file_untouched(self, range_log: Path) -> None:
        before = range_log.read_bytes()
        with pytest.raises(LogError):
            append_records(range_log, [forecast(FRI, "ewma_94", made="2030-03-15 19:00")])
        assert range_log.read_bytes() == before


class TestRangeScoring:
    def test_ranges_are_paired_with_where_the_outcome_fell(self, range_log: Path) -> None:
        scored = scored_ranges(read_log(range_log))
        first = scored[scored["based_on_session"] == THU].set_index("model")
        assert first.loc["constant", "position"] == "inside"
        assert first.loc["ewma_94", "position"] == "above"
        assert first.loc["ewma_94", "target_session"] == FRI
        assert (first.loc["ewma_94", "lower_return"], first.loc["ewma_94", "upper_return"]) == (
            -0.031,
            0.034,
        )
        assert first.loc["ewma_94", "level"] == 0.8

    def test_the_latest_ranges_are_not_scored_yet(self, range_log: Path) -> None:
        scored = scored_ranges(read_log(range_log))
        latest = scored[scored["based_on_session"] == FRI]
        assert len(latest) == 2
        assert latest["position"].isna().all() and latest["target_session"].isna().all()

    def test_direction_scoring_ignores_range_records(self, range_log: Path) -> None:
        scored = scored_predictions(read_log(range_log))
        assert list(scored["model"]) == ["always_up", "always_up"]
        assert scored_ranges(week()).empty

    def test_the_verifier_counts_each_kind(self, range_log: Path) -> None:
        lines: list[str] = []
        assert main(["verify", "--log", str(range_log)], report=lines.append) == 0
        assert lines == [
            "log is valid: 9 records (2 predictions, 1 outcomes, 4 range forecasts, "
            "2 range outcomes, 0 paper strategy entries)"
        ]


def six_calls(based_on: date = THU, *, up: int = 4, made: str | None = None) -> list[Record]:
    """A call from each of the paper strategy's six models; the first `up` say up."""
    return [
        prediction(based_on, name, up=position < up, made=made)
        for position, name in enumerate(RULE_MODELS)
    ]


def entry(
    based_on: date = THU, *, up: int = 4, gate: bool = True, made: str | None = None
) -> PaperStrategyEntry:
    """The entry the rule gives for `up` models up and this gate."""
    made_at = at(made) if made else at(f"{based_on} 18:00")
    return PaperStrategyEntry(made_at, based_on, up >= 4 and gate, up, gate)


class TestPaperStrategyRecords:
    """The paper strategy's entry: Holding or In cash, and what the rule looked at."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        records = [*six_calls(), entry()]
        append_records(path, records)
        assert read_log(path) == records

    def test_the_line_holds_the_state_and_the_rules_two_inputs_only(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        append_records(path, [*six_calls(up=5), entry(up=5)])
        line = json.loads(lines_of(path)[-1])
        assert line["type"] == "paper_strategy"
        assert set(line) == {
            "v",
            "type",
            "made_at",
            "based_on_session",
            "holding",
            "models_up",
            "gate_open",
            "prev_hash",
            "hash",
        }
        assert (line["holding"], line["models_up"], line["gate_open"]) == (True, 5, True)

    @pytest.mark.parametrize(
        ("up", "gate", "holding"),
        [(6, True, True), (4, True, True), (3, True, False), (6, False, False), (0, False, False)],
    )
    def test_the_state_the_rule_gives_is_accepted(
        self, tmp_path: Path, up: int, gate: bool, holding: bool
    ) -> None:
        record = entry(up=up, gate=gate)
        assert record.holding is holding
        append_records(tmp_path / "log.jsonl", [*six_calls(up=up), record])

    @pytest.mark.parametrize(
        ("holding", "up", "gate"),
        [(True, 3, True), (True, 6, False), (False, 4, True), (False, 6, True)],
    )
    def test_a_state_the_rule_does_not_give_is_refused(
        self, holding: bool, up: int, gate: bool
    ) -> None:
        with pytest.raises(LogError, match="does not follow the rule"):
            PaperStrategyEntry(at(f"{THU} 18:00"), THU, holding, up, gate)

    @pytest.mark.parametrize("bad", [-1, 7, 4.0, True, "4"])
    def test_the_count_must_be_a_whole_number_from_zero_to_six(self, bad: Any) -> None:
        with pytest.raises(LogError, match="models_up"):
            PaperStrategyEntry(at(f"{THU} 18:00"), THU, False, bad, False)

    @pytest.mark.parametrize("missing", RULE_MODELS)
    def test_it_needs_a_call_from_every_one_of_the_six(self, tmp_path: Path, missing: str) -> None:
        calls = [c for c in six_calls(up=6) if isinstance(c, Prediction) and c.model != missing]
        path = tmp_path / "log.jsonl"
        append_records(path, calls)
        before = path.read_bytes()
        with pytest.raises(LogError, match=f"none on record from {missing}"):
            append_records(path, [entry(up=5)])
        assert path.read_bytes() == before

    def test_calls_from_other_predictors_do_not_stand_in(self, tmp_path: Path) -> None:
        five = six_calls(up=6)[:5]
        others = [prediction(THU, "always_up", probability=None), prediction(THU, "ma_200")]
        with pytest.raises(LogError, match="none on record from lstm_ctx"):
            append_records(tmp_path / "log.jsonl", [*five, *others, entry(up=5)])

    def test_the_calls_must_be_for_the_same_session(self, tmp_path: Path) -> None:
        records = [*six_calls(THU), outcome(FRI, THU, up=True), entry(FRI)]
        with pytest.raises(LogError, match="none on record from logistic"):
            append_records(tmp_path / "log.jsonl", records)

    def test_the_calls_must_come_first(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match="needs a call from every one of its models"):
            append_records(tmp_path / "log.jsonl", [entry(), *six_calls()])

    def test_the_count_must_match_the_calls_in_the_log(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match=r"says 5 models called up .* but the log has 4"):
            append_records(tmp_path / "log.jsonl", [*six_calls(up=4), entry(up=5)])

    def test_one_entry_per_session(self, tmp_path: Path) -> None:
        with pytest.raises(LogError, match="already has an entry based on 2030-03-14"):
            append_records(tmp_path / "log.jsonl", [*six_calls(), entry(), entry()])

    def test_not_before_its_session_has_closed(self, tmp_path: Path) -> None:
        # Its six calls must come first and cannot be written before the close, so
        # an entry stamped before the close is always out of order.
        early = "2030-03-14 15:59"
        with pytest.raises(LogError, match="before the record above it"):
            append_records(
                tmp_path / "log.jsonl", [*six_calls(made="2030-03-14 16:00"), entry(made=early)]
            )

    @pytest.mark.parametrize("late", ["2030-03-15 09:30", "2030-03-15 12:00", "2030-03-18 08:00"])
    def test_not_after_the_next_session_could_have_opened(self, tmp_path: Path, late: str) -> None:
        path = tmp_path / "log.jsonl"
        append_records(path, six_calls())
        before = path.read_bytes()
        with pytest.raises(LogError, match="after the next session could have opened"):
            append_records(path, [entry(made=late)])
        assert path.read_bytes() == before

    def test_just_before_the_open_is_still_in_time(self, tmp_path: Path) -> None:
        append_records(tmp_path / "log.jsonl", [*six_calls(), entry(made="2030-03-15 09:29")])

    def test_not_once_a_later_outcome_is_on_record(self, tmp_path: Path) -> None:
        records = [*six_calls(THU), outcome(FRI, THU, up=True)]
        path = tmp_path / "log.jsonl"
        append_records(path, records)
        with pytest.raises(LogError, match="outcome of a later session"):
            append_records(path, [entry(THU, made="2030-03-15 17:31")])

    def test_changing_the_state_afterwards_breaks_the_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        append_records(path, [*six_calls(up=3), entry(up=3)])
        text = path.read_text()
        assert '"holding":false' in text
        path.write_text(text.replace('"holding":false', '"holding":true'))
        with pytest.raises(LogError, match="line 7"):
            read_log(path)

    def test_a_session_without_an_entry_is_simply_absent(self, tmp_path: Path) -> None:
        # Thursday has all six calls and an entry; Friday has five calls and none.
        records: list[Record] = [
            *six_calls(THU),
            entry(THU),
            outcome(FRI, THU, up=True),
            *six_calls(FRI)[:5],
            outcome(MON, FRI, up=False),
            *six_calls(MON, up=2),
            entry(MON, up=2),
        ]
        path = tmp_path / "log.jsonl"
        append_records(path, records)
        scored = scored_paper_strategy(read_log(path))
        assert list(scored["based_on_session"]) == [THU, MON]


class TestPaperStrategyScoring:
    def week(self) -> list[Record]:
        return [
            *six_calls(THU, up=5),
            entry(THU, up=5),
            outcome(FRI, THU, up=True),
            *six_calls(FRI, up=6),
            entry(FRI, up=6, gate=False),
        ]

    def test_entries_are_paired_with_the_next_sessions_outcome(self) -> None:
        scored = scored_paper_strategy(self.week())
        first, second = scored.iloc[0], scored.iloc[1]
        assert (first["based_on_session"], first["target_session"]) == (THU, FRI)
        assert bool(first["holding"]) is True and bool(first["went_up"]) is True
        assert (first["models_up"], bool(first["gate_open"])) == (5, True)
        assert bool(second["holding"]) is False
        assert pd.isna(second["target_session"]) and pd.isna(second["went_up"])

    def test_the_other_scorers_ignore_paper_entries(self) -> None:
        with_entries = scored_predictions(self.week())
        without = scored_predictions(
            [r for r in self.week() if not isinstance(r, PaperStrategyEntry)]
        )
        assert with_entries.equals(without)
        assert scored_ranges(self.week()).empty
        assert scored_paper_strategy(week()).empty

    def test_the_verifier_counts_them(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        append_records(path, self.week())
        lines: list[str] = []
        assert main(["verify", "--log", str(path)], report=lines.append) == 0
        assert lines == [
            "log is valid: 15 records (12 predictions, 1 outcomes, 0 range forecasts, "
            "0 range outcomes, 2 paper strategy entries)"
        ]
