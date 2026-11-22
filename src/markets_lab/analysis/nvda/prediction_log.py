"""The live prediction log: append-only and tamper-evident.

The log is a text file with one JSON record per line. Lines are only ever added.
There are four kinds of record. Two are about direction:

- A prediction: which model, the session it was based on, the direction it called
  for the next session, its probability if it has one, and when it was written.
- An outcome: for a finished session, whether it closed up from the session before.

Two are about the size of the move:

- A range forecast: which forecaster, the session it was based on, and the range it
  expects the next session's return to fall in (the forecaster's own numbers), with
  the share of outcomes the range is built to hold.
- A range outcome: for one range forecast, whether the session's return fell inside
  the range, below it or above it.

Whether a prediction was right is never written into the prediction itself, since
that would mean editing it later. It is worked out by pairing each prediction with
the outcome of the session after the one it was based on (`scored_predictions`,
and `scored_ranges` for ranges).

The log holds no prices and no realised returns. What happened is recorded only as
"up or not" and as "inside, below or above" each forecast range.

What makes it trustworthy:

- Every line carries the SHA-256 hash of the line before it and of its own content,
  so changing, removing or reordering any earlier line breaks the chain.
- `check_append_only` compares the file with its previous version: the old file
  must be an exact prefix of the new one. This catches a rewrite of the whole chain.
- Timing rules are checked on every read: a prediction or range forecast must be
  written after the session it is based on has closed and before the next weekday's
  market open, and never after the outcome it is predicting has been recorded.
- A range outcome must follow the range forecast it scores and the outcome of the
  same session, and must not contradict that outcome: a session that closed up
  cannot have fallen below a range that starts above zero, and so on.

Check a log file from the repository root:

    uv run python -m markets_lab.analysis.nvda.prediction_log verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_LOG_PATH = Path("predictions/nvda.jsonl")
SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PROBABILITY_DECIMALS = 4
RETURN_DECIMALS = 4
"""Range bounds are stored as returns to this many places: 0.0312 is 3.12%."""

BELOW, INSIDE, ABOVE = "below", "inside", "above"
RANGE_POSITIONS = (BELOW, INSIDE, ABOVE)

_MODEL_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_PREDICTION_KEYS = {
    "v",
    "type",
    "made_at",
    "based_on_session",
    "model",
    "predicted_up",
    "probability_up",
    "prev_hash",
    "hash",
}
_OUTCOME_KEYS = {
    "v",
    "type",
    "recorded_at",
    "session",
    "previous_session",
    "went_up",
    "prev_hash",
    "hash",
}
_RANGE_FORECAST_KEYS = {
    "v",
    "type",
    "made_at",
    "based_on_session",
    "model",
    "level",
    "lower_return",
    "upper_return",
    "prev_hash",
    "hash",
}
_RANGE_OUTCOME_KEYS = {
    "v",
    "type",
    "recorded_at",
    "session",
    "previous_session",
    "model",
    "position",
    "prev_hash",
    "hash",
}


class LogError(ValueError):
    """The log is not valid, or a record may not be added to it."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """A call on the session after `based_on_session`, written at `made_at`."""

    made_at: datetime
    based_on_session: date
    model: str
    predicted_up: bool
    probability_up: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "made_at", _utc_seconds(self.made_at, "made_at"))
        _check_session(self.based_on_session, "based_on_session")
        if not _MODEL_NAME.fullmatch(self.model):
            raise LogError(f"model name {self.model!r} is not lowercase letters, digits, _")
        if not isinstance(self.predicted_up, bool):
            raise LogError("predicted_up must be true or false")
        if self.probability_up is not None:
            probability = self.probability_up
            if isinstance(probability, bool) or not isinstance(probability, int | float):
                raise LogError("probability_up must be a number or null")
            if not 0 <= probability <= 1:
                raise LogError("probability_up must be between 0 and 1")
            object.__setattr__(
                self, "probability_up", round(float(probability), PROBABILITY_DECIMALS)
            )


@dataclass(frozen=True, slots=True)
class Outcome:
    """Whether `session` closed above `previous_session`, recorded at `recorded_at`."""

    recorded_at: datetime
    session: date
    previous_session: date
    went_up: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded_at", _utc_seconds(self.recorded_at, "recorded_at"))
        _check_session(self.session, "session")
        _check_session(self.previous_session, "previous_session")
        if self.previous_session >= self.session:
            raise LogError("previous_session must be before session")
        if not isinstance(self.went_up, bool):
            raise LogError("went_up must be true or false")


@dataclass(frozen=True, slots=True)
class RangeForecast:
    """The range a forecaster expects the return of the session after
    `based_on_session` to fall in, written at `made_at`.

    The bounds are simple returns (0.0312 is +3.12%). `level` is the share of
    outcomes the range is built to hold.
    """

    made_at: datetime
    based_on_session: date
    model: str
    level: float
    lower_return: float
    upper_return: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "made_at", _utc_seconds(self.made_at, "made_at"))
        _check_session(self.based_on_session, "based_on_session")
        if not _MODEL_NAME.fullmatch(self.model):
            raise LogError(f"model name {self.model!r} is not lowercase letters, digits, _")
        level = _number(self.level, "level")
        if not 0 < level < 1:
            raise LogError("level must be between 0 and 1")
        lower = round(_number(self.lower_return, "lower_return"), RETURN_DECIMALS)
        upper = round(_number(self.upper_return, "upper_return"), RETURN_DECIMALS)
        if lower <= -1:
            raise LogError("lower_return must be above -1 (a fall of 100%)")
        if not lower < upper:
            raise LogError("lower_return must be below upper_return")
        object.__setattr__(self, "level", round(level, PROBABILITY_DECIMALS))
        object.__setattr__(self, "lower_return", lower)
        object.__setattr__(self, "upper_return", upper)


@dataclass(frozen=True, slots=True)
class RangeOutcome:
    """Where the return of `session` fell against one forecaster's range for it."""

    recorded_at: datetime
    session: date
    previous_session: date
    model: str
    position: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded_at", _utc_seconds(self.recorded_at, "recorded_at"))
        _check_session(self.session, "session")
        _check_session(self.previous_session, "previous_session")
        if self.previous_session >= self.session:
            raise LogError("previous_session must be before session")
        if not _MODEL_NAME.fullmatch(self.model):
            raise LogError(f"model name {self.model!r} is not lowercase letters, digits, _")
        if self.position not in RANGE_POSITIONS:
            raise LogError(f"position must be one of {', '.join(RANGE_POSITIONS)}")


Record = Prediction | Outcome | RangeForecast | RangeOutcome


def range_position(lower_return: float, upper_return: float, realised_return: float) -> str:
    """Where a return fell against a range. The bounds themselves count as inside."""
    if realised_return < lower_return:
        return BELOW
    if realised_return > upper_return:
        return ABOVE
    return INSIDE


def prediction_window(based_on_session: date) -> tuple[datetime, datetime]:
    """When a prediction based on a session may be written: from that session's
    close until just before the open of the next weekday (both New York time).

    The next weekday may be a holiday, in which case the real next session is later
    and the window is stricter than it needs to be. That is deliberate: no market
    calendar is needed, and a prediction can never be written after its session
    has started.
    """
    earliest = datetime.combine(based_on_session, MARKET_CLOSE, tzinfo=MARKET_TZ)
    following = based_on_session + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    deadline = datetime.combine(following, MARKET_OPEN, tzinfo=MARKET_TZ)
    return earliest.astimezone(UTC), deadline.astimezone(UTC)


def parse_log(text: str) -> list[Record]:
    """Parse and fully verify the text of a log. Raises `LogError` naming the line."""
    records: list[Record] = []
    previous_hash = GENESIS_HASH
    if text and not text.endswith("\n"):
        raise LogError("log does not end with a newline; the last line may be incomplete")
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            record, line_hash = _parse_line(line, previous_hash)
        except LogError as exc:
            raise LogError(f"line {number}: {exc}") from exc
        records.append(record)
        previous_hash = line_hash
    _check_sequence(records)
    return records


def read_log(path: Path) -> list[Record]:
    """Read and verify a log file. A file that does not exist is an empty log."""
    if not path.exists():
        return []
    return parse_log(path.read_text(encoding="utf-8"))


def append_records(path: Path, new_records: Sequence[Record]) -> None:
    """Verify the existing log, then add records to the end of it.

    The combined log is verified before anything is written, so a record that
    breaks a rule is refused and the file is left exactly as it was. Existing bytes
    are never rewritten: the new file is the old file plus the new lines.
    """
    if not new_records:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    records = parse_log(existing)
    previous_hash = _last_hash(existing)
    added: list[str] = []
    for record in new_records:
        line = _seal(_payload(record), previous_hash)
        previous_hash = json.loads(line)["hash"]
        added.append(line + "\n")
    combined = existing + "".join(added)
    if len(parse_log(combined)) != len(records) + len(new_records):
        raise LogError("internal error: appended log does not parse to the expected length")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(combined, encoding="utf-8")
    os.replace(temporary, path)


def check_append_only(previous_text: str, current_text: str) -> None:
    """Raise unless the current log is the previous log with lines added at the end."""
    if not current_text.startswith(previous_text):
        raise LogError(
            "the log is not an extension of its previous version: "
            "an earlier line was changed, removed or reordered"
        )


def scored_predictions(records: Sequence[Record]) -> pd.DataFrame:
    """One row per prediction, paired with its outcome where that is known.

    Columns: `model`, `based_on_session`, `target_session`, `predicted_up`,
    `probability_up`, `made_at`, `went_up`, `correct`. The last two and
    `target_session` are missing until the outcome has been recorded.
    """
    outcomes = {r.previous_session: r for r in records if isinstance(r, Outcome)}
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Prediction):
            continue
        outcome = outcomes.get(record.based_on_session)
        rows.append(
            {
                "model": record.model,
                "based_on_session": record.based_on_session,
                "target_session": outcome.session if outcome else None,
                "predicted_up": record.predicted_up,
                "probability_up": record.probability_up,
                "made_at": record.made_at,
                "went_up": outcome.went_up if outcome else None,
                "correct": (record.predicted_up == outcome.went_up) if outcome else None,
            }
        )
    columns = [
        "model",
        "based_on_session",
        "target_session",
        "predicted_up",
        "probability_up",
        "made_at",
        "went_up",
        "correct",
    ]
    return pd.DataFrame(rows, columns=columns)


def scored_ranges(records: Sequence[Record]) -> pd.DataFrame:
    """One row per range forecast, paired with where the outcome fell once known.

    Columns: `model`, `based_on_session`, `target_session`, `level`,
    `lower_return`, `upper_return`, `made_at`, `position`. `target_session` and
    `position` are missing until the range outcome has been recorded.
    """
    outcomes = {(r.model, r.previous_session): r for r in records if isinstance(r, RangeOutcome)}
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, RangeForecast):
            continue
        outcome = outcomes.get((record.model, record.based_on_session))
        rows.append(
            {
                "model": record.model,
                "based_on_session": record.based_on_session,
                "target_session": outcome.session if outcome else None,
                "level": record.level,
                "lower_return": record.lower_return,
                "upper_return": record.upper_return,
                "made_at": record.made_at,
                "position": outcome.position if outcome else None,
            }
        )
    columns = [
        "model",
        "based_on_session",
        "target_session",
        "level",
        "lower_return",
        "upper_return",
        "made_at",
        "position",
    ]
    return pd.DataFrame(rows, columns=columns)


# -- line format ---------------------------------------------------------------


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _seal(payload: dict[str, Any], previous_hash: str) -> str:
    """The canonical text of a line: the payload, the previous hash, and its own hash."""
    body = {**payload, "prev_hash": previous_hash}
    digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    return _canonical({**body, "hash": digest})


def _payload(record: Record) -> dict[str, Any]:
    if isinstance(record, Prediction):
        return {
            "v": SCHEMA_VERSION,
            "type": "prediction",
            "made_at": record.made_at.strftime(_TIMESTAMP_FORMAT),
            "based_on_session": record.based_on_session.isoformat(),
            "model": record.model,
            "predicted_up": record.predicted_up,
            "probability_up": record.probability_up,
        }
    if isinstance(record, RangeForecast):
        return {
            "v": SCHEMA_VERSION,
            "type": "range_forecast",
            "made_at": record.made_at.strftime(_TIMESTAMP_FORMAT),
            "based_on_session": record.based_on_session.isoformat(),
            "model": record.model,
            "level": record.level,
            "lower_return": record.lower_return,
            "upper_return": record.upper_return,
        }
    if isinstance(record, RangeOutcome):
        return {
            "v": SCHEMA_VERSION,
            "type": "range_outcome",
            "recorded_at": record.recorded_at.strftime(_TIMESTAMP_FORMAT),
            "session": record.session.isoformat(),
            "previous_session": record.previous_session.isoformat(),
            "model": record.model,
            "position": record.position,
        }
    return {
        "v": SCHEMA_VERSION,
        "type": "outcome",
        "recorded_at": record.recorded_at.strftime(_TIMESTAMP_FORMAT),
        "session": record.session.isoformat(),
        "previous_session": record.previous_session.isoformat(),
        "went_up": record.went_up,
    }


def _parse_line(line: str, expected_previous_hash: str) -> tuple[Record, str]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LogError("not valid JSON") from exc
    if not isinstance(raw, dict):
        raise LogError("not a JSON object")
    if raw.get("v") != SCHEMA_VERSION:
        raise LogError(f"unsupported record version {raw.get('v')!r}")
    kind = raw.get("type")
    if kind == "prediction":
        _check_keys(raw, _PREDICTION_KEYS)
        record: Record = Prediction(
            made_at=_parse_timestamp(raw["made_at"]),
            based_on_session=_parse_date(raw["based_on_session"]),
            model=_text(raw["model"]),
            predicted_up=raw["predicted_up"],
            probability_up=raw["probability_up"],
        )
    elif kind == "outcome":
        _check_keys(raw, _OUTCOME_KEYS)
        record = Outcome(
            recorded_at=_parse_timestamp(raw["recorded_at"]),
            session=_parse_date(raw["session"]),
            previous_session=_parse_date(raw["previous_session"]),
            went_up=raw["went_up"],
        )
    elif kind == "range_forecast":
        _check_keys(raw, _RANGE_FORECAST_KEYS)
        record = RangeForecast(
            made_at=_parse_timestamp(raw["made_at"]),
            based_on_session=_parse_date(raw["based_on_session"]),
            model=_text(raw["model"]),
            level=raw["level"],
            lower_return=raw["lower_return"],
            upper_return=raw["upper_return"],
        )
    elif kind == "range_outcome":
        _check_keys(raw, _RANGE_OUTCOME_KEYS)
        record = RangeOutcome(
            recorded_at=_parse_timestamp(raw["recorded_at"]),
            session=_parse_date(raw["session"]),
            previous_session=_parse_date(raw["previous_session"]),
            model=_text(raw["model"]),
            position=_text(raw["position"]),
        )
    else:
        raise LogError(f"unknown record type {kind!r}")
    if raw["prev_hash"] != expected_previous_hash:
        raise LogError("chain broken: this line does not follow the line before it")
    if _seal(_payload(record), expected_previous_hash) != line:
        raise LogError("content does not match its hash, or the line is not in canonical form")
    return record, raw["hash"]


def _last_hash(text: str) -> str:
    lines = text.splitlines()
    return json.loads(lines[-1])["hash"] if lines else GENESIS_HASH


def _check_keys(raw: dict[str, Any], expected: set[str]) -> None:
    if set(raw) != expected:
        extra, missing = sorted(set(raw) - expected), sorted(expected - set(raw))
        raise LogError(f"unexpected fields {extra}, missing fields {missing}")


# -- rules across records ------------------------------------------------------


def _check_sequence(records: Sequence[Record]) -> None:
    """Rules that involve more than one record. Raises `LogError` on the first breach."""
    seen_predictions: set[tuple[str, date]] = set()
    range_forecasts: dict[tuple[str, date], RangeForecast] = {}
    range_outcomes: set[tuple[str, date]] = set()
    outcomes: dict[date, Outcome] = {}
    last_outcome: Outcome | None = None
    last_time: datetime | None = None
    for number, record in enumerate(records, start=1):
        where = f"record {number}"
        # Checked first because it does not rely on any clock: once the outcome of
        # a later session is on file, no prediction for that session can be added.
        if (
            isinstance(record, Prediction | RangeForecast)
            and last_outcome is not None
            and record.based_on_session < last_outcome.session
        ):
            raise LogError(
                f"{where}: based on {record.based_on_session}, but the outcome of a "
                f"later session ({last_outcome.session}) was already recorded"
            )
        moment = (
            record.made_at if isinstance(record, Prediction | RangeForecast) else record.recorded_at
        )
        if last_time is not None and moment < last_time:
            raise LogError(f"{where}: written at {moment}, before the record above it")
        last_time = moment

        if isinstance(record, RangeOutcome):
            _check_range_outcome(record, where, outcomes, range_forecasts, range_outcomes)
            range_outcomes.add((record.model, record.previous_session))
        elif isinstance(record, Prediction | RangeForecast):
            key = (record.model, record.based_on_session)
            what = "prediction" if isinstance(record, Prediction) else "range forecast"
            taken = seen_predictions if isinstance(record, Prediction) else range_forecasts
            if key in taken:
                raise LogError(
                    f"{where}: {record.model} already has a {what} based on "
                    f"{record.based_on_session}"
                )
            if isinstance(record, Prediction):
                seen_predictions.add(key)
            else:
                range_forecasts[key] = record
            earliest, deadline = prediction_window(record.based_on_session)
            if record.made_at < earliest:
                raise LogError(
                    f"{where}: written before the session it is based on "
                    f"({record.based_on_session}) had closed"
                )
            if record.made_at >= deadline:
                raise LogError(
                    f"{where}: written at {record.made_at}, after the next session could "
                    f"have opened ({deadline}); a prediction must come before its session"
                )
        else:
            session_close = datetime.combine(record.session, MARKET_CLOSE, tzinfo=MARKET_TZ)
            if record.recorded_at < session_close:
                raise LogError(f"{where}: outcome of {record.session} recorded before its close")
            if last_outcome is not None and record.previous_session != last_outcome.session:
                raise LogError(
                    f"{where}: outcome for {record.session} follows {record.previous_session}, "
                    f"but the last recorded session is {last_outcome.session}"
                )
            last_outcome = record
            outcomes[record.previous_session] = record


def _check_range_outcome(
    record: RangeOutcome,
    where: str,
    outcomes: dict[date, Outcome],
    range_forecasts: dict[tuple[str, date], RangeForecast],
    already: set[tuple[str, date]],
) -> None:
    """A range outcome needs its forecast and its session's outcome, once, and must
    agree with which way that session closed."""
    key = (record.model, record.previous_session)
    forecast = range_forecasts.get(key)
    if forecast is None:
        raise LogError(
            f"{where}: {record.model} has no range forecast based on {record.previous_session}"
        )
    if key in already:
        raise LogError(
            f"{where}: {record.model}'s range based on {record.previous_session} "
            "already has an outcome"
        )
    outcome = outcomes.get(record.previous_session)
    if outcome is None or outcome.session != record.session:
        raise LogError(
            f"{where}: the outcome of {record.session} (after {record.previous_session}) "
            "has not been recorded"
        )
    # The return is not stored, but its sign is. A positive return cannot be below
    # a range that starts above zero, or inside one that ends at or below zero; a
    # return that is not positive cannot be above a range that ends at or above
    # zero, or inside one that starts above zero.
    lower, upper, position = forecast.lower_return, forecast.upper_return, record.position
    if outcome.went_up:
        impossible = (position == BELOW and lower <= 0) or (position == INSIDE and upper <= 0)
    else:
        impossible = (position == ABOVE and upper >= 0) or (position == INSIDE and lower > 0)
    if impossible:
        direction = "up" if outcome.went_up else "not up"
        raise LogError(
            f"{where}: {record.session} closed {direction}, which cannot be {position} "
            f"a range of {lower} to {upper}"
        )


# -- small parsers -------------------------------------------------------------


def _utc_seconds(moment: object, what: str) -> datetime:
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise LogError(f"{what} must be a timezone-aware time")
    return moment.astimezone(UTC).replace(microsecond=0)


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LogError(f"{what} must be a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise LogError(f"{what} must be finite")
    return float(value)


def _check_session(session: object, what: str) -> None:
    if isinstance(session, datetime) or not isinstance(session, date):
        raise LogError(f"{what} must be a date")
    if session.weekday() >= 5:
        raise LogError(f"{what} {session} is a weekend, not a session")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise LogError(f"expected text, got {value!r}")
    return value


def _parse_timestamp(value: object) -> datetime:
    try:
        return datetime.strptime(_text(value), _TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise LogError(f"not a UTC timestamp: {value!r}") from exc


def _parse_date(value: object) -> date:
    try:
        return date.fromisoformat(_text(value))
    except ValueError as exc:
        raise LogError(f"not a date: {value!r}") from exc


# -- command line --------------------------------------------------------------


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: Sequence[str] | None = None, *, report: Callable[[str], None] = _print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.analysis.nvda.prediction_log",
        description="Verify the prediction log.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check the hash chain and every rule")
    verify.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    verify.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="an earlier copy of the log; the current log must extend it exactly",
    )
    args = parser.parse_args(argv)
    log: Path = args.log
    try:
        if not log.exists():
            raise LogError(f"{log} does not exist")
        text = log.read_text(encoding="utf-8")
        records = parse_log(text)
        if args.previous is not None:
            check_append_only(args.previous.read_text(encoding="utf-8"), text)
    except LogError as exc:
        report(f"LOG VERIFICATION FAILED: {exc}")
        return 1
    counts = [
        (Prediction, "predictions"),
        (Outcome, "outcomes"),
        (RangeForecast, "range forecasts"),
        (RangeOutcome, "range outcomes"),
    ]
    summary = ", ".join(
        f"{sum(isinstance(r, kind) for r in records)} {label}" for kind, label in counts
    )
    extends = " and extends the previous version" if args.previous is not None else ""
    report(f"log is valid{extends}: {len(records)} records ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
