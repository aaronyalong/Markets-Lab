"""The nightly job: record what happened, make the next call, append both to the log.

Each run:

1. Downloads the daily prices and keeps completed sessions only.
2. Appends an outcome (up or not) for every session finished since the last one on
   record.
3. For the latest completed session, fits every baseline and model on the sessions
   whose outcome is already known and appends each one's call on the next session.

The run changes nothing unless everything succeeds. It refuses, loudly, to write a
prediction when:

- Yahoo has a newer session that it has not finished publishing. Predicting from
  the session before would be a call on a session that has already happened.
- The window for the latest session has passed (the next weekday's open). A late
  prediction is not written; that day is simply missing from the log.

Running it again the same morning adds nothing. Prices are held in memory only;
the log it writes contains directions, never prices.

    uv run python -m markets_lab.analysis.nvda.nightly --dry-run
    uv run python -m markets_lab.analysis.nvda.nightly
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from markets_lab.analysis.nvda.backtest import (
    BASELINE_FACTORIES,
    MODEL_FACTORIES,
    PredictorFactory,
)
from markets_lab.analysis.nvda.features import build_dataset, compute_outcomes
from markets_lab.analysis.nvda.prediction_log import (
    DEFAULT_LOG_PATH,
    LogError,
    Outcome,
    Prediction,
    Record,
    append_records,
    prediction_window,
    read_log,
)
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP
from markets_lab.collectors.nvda_prices import (
    MIN_ROWS,
    Downloader,
    DroppedBar,
    PriceDataError,
    download_raw,
    fetch_daily,
)


class NightlyError(RuntimeError):
    """The run cannot honestly write a prediction, so it writes nothing."""


@dataclass(frozen=True, slots=True)
class NightlyResult:
    """What a run added to the log (or would add, on a dry run)."""

    last_session: date
    outcomes: tuple[Outcome, ...]
    predictions: tuple[Prediction, ...]


def default_factories() -> dict[str, PredictorFactory]:
    """Every baseline and every measured model, in report order."""
    return {**BASELINE_FACTORIES, **MODEL_FACTORIES}


def plan_records(
    prices: pd.DataFrame,
    dropped: Sequence[DroppedBar],
    existing: Sequence[Record],
    *,
    now: datetime,
    factories: Mapping[str, PredictorFactory],
) -> NightlyResult:
    """Work out which outcomes and predictions this run should append.

    `prices` holds completed sessions only. Nothing is written here.
    """
    dataset = build_dataset(prices)
    last_session, _ = dataset.latest()

    unpublished = [bar for bar in dropped if bar.reason == "incomplete"]
    if unpublished:
        raise NightlyError(
            f"session {unpublished[0].session} has finished but Yahoo has not published its "
            f"close yet; no prediction can be made until it is available"
        )

    outcomes = _missing_outcomes(prices, existing, now=now)

    already = {(r.model, r.based_on_session) for r in existing if isinstance(r, Prediction)}
    missing = [name for name in factories if (name, last_session) not in already]
    predictions: list[Prediction] = []
    if missing:
        _, deadline = prediction_window(last_session)
        if now >= deadline:
            raise NightlyError(
                f"too late to predict the session after {last_session}: predictions had to be "
                f"written before {deadline.strftime('%Y-%m-%d %H:%M UTC')}, and it is now "
                f"{now.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        known = dataset.labelled()
        row = dataset.features.iloc[[-1]]
        for name in missing:
            predictor = factories[name]()
            predictor.fit(known.features, known.target_up)
            call = predictor.predict(row).iloc[0]
            probability = float(call[PROBABILITY_UP])
            predictions.append(
                Prediction(
                    made_at=now,
                    based_on_session=last_session,
                    model=name,
                    predicted_up=bool(call[PREDICTED_UP]),
                    probability_up=None if pd.isna(probability) else probability,
                )
            )
    return NightlyResult(last_session, tuple(outcomes), tuple(predictions))


def _missing_outcomes(
    prices: pd.DataFrame, existing: Sequence[Record], *, now: datetime
) -> list[Outcome]:
    """Outcomes for every completed session after the last one on record.

    With no outcome on record yet, start after the session the earliest prediction
    was based on. With nothing on record at all, there is nothing to score.
    """
    recorded = [r for r in existing if isinstance(r, Outcome)]
    predictions = [r for r in existing if isinstance(r, Prediction)]
    if recorded:
        after = recorded[-1].session
    elif predictions:
        after = min(p.based_on_session for p in predictions)
    else:
        return []
    sessions = [pd.Timestamp(ts).date() for ts in prices.index]
    if after not in sessions:
        raise NightlyError(f"session {after} is in the log but not in the downloaded prices")
    went_up = compute_outcomes(prices)["target_up"].to_numpy(dtype="float64")
    start = sessions.index(after)
    return [
        Outcome(
            recorded_at=now,
            session=sessions[position + 1],
            previous_session=sessions[position],
            went_up=bool(went_up[position] == 1.0),
        )
        for position in range(start, len(sessions) - 1)
    ]


def run_nightly(
    *,
    now: datetime,
    log_path: Path = DEFAULT_LOG_PATH,
    downloader: Downloader = download_raw,
    factories: Mapping[str, PredictorFactory] | None = None,
    dry_run: bool = False,
    min_rows: int = MIN_ROWS,
) -> NightlyResult:
    """Run the job once. Either every new record is appended, or the log is untouched."""
    fetched = fetch_daily(now=now, downloader=downloader, min_rows=min_rows)
    existing = read_log(log_path)
    result = plan_records(
        fetched.prices,
        fetched.dropped,
        existing,
        now=now,
        factories=default_factories() if factories is None else factories,
    )
    if not dry_run:
        append_records(log_path, [*result.outcomes, *result.predictions])
    return result


def describe(result: NightlyResult, *, dry_run: bool) -> list[str]:
    """Plain-text lines saying what the run did."""
    verb = "would append" if dry_run else "appended"
    lines = [f"last completed session: {result.last_session}"]
    if not result.outcomes and not result.predictions:
        lines.append("nothing new to record")
        return lines
    for outcome in result.outcomes:
        direction = "up" if outcome.went_up else "not up"
        lines.append(f"{verb} outcome: {outcome.session} closed {direction}")
    for prediction in result.predictions:
        direction = "up" if prediction.predicted_up else "down"
        probability = (
            "" if prediction.probability_up is None else f" (p up {prediction.probability_up:.4f})"
        )
        lines.append(
            f"{verb} prediction: {prediction.model} calls {direction}{probability} "
            f"for the session after {prediction.based_on_session}"
        )
    return lines


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    downloader: Downloader = download_raw,
    factories: Mapping[str, PredictorFactory] | None = None,
    report: Callable[[str], None] = _print,
    min_rows: int = MIN_ROWS,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.analysis.nvda.nightly",
        description="Record finished sessions and append the next predictions to the log.",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be appended; write nothing"
    )
    args = parser.parse_args(argv)
    try:
        result = run_nightly(
            now=now(),
            log_path=args.log,
            downloader=downloader,
            factories=factories,
            dry_run=args.dry_run,
            min_rows=min_rows,
        )
    except (NightlyError, PriceDataError, LogError) as exc:
        report(f"NIGHTLY RUN FAILED, nothing was written: {exc}")
        return 1
    for line in describe(result, dry_run=args.dry_run):
        report(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
