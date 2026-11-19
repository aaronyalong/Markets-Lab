"""The prediction log: append-only, tamper-evident, direction only."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from markets_lab.analysis.nvda.prediction_log import (
    GENESIS_HASH,
    LogError,
    Outcome,
    Prediction,
    Record,
    append_records,
    check_append_only,
    main,
    parse_log,
    prediction_window,
    read_log,
    scored_predictions,
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
        assert output == "log is valid: 8 records (6 predictions, 2 outcomes)"

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
