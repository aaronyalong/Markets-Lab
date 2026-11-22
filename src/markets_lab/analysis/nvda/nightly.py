"""The nightly job: record what happened, make the next call, append both to the log.

Each run:

1. Downloads the daily prices and keeps completed sessions only.
2. Appends an outcome (up or not) for every session finished since the last one on
   record.
3. For the latest completed session, fits every baseline and model on the sessions
   whose outcome is already known and appends each one's call on the next session.
4. Does the same for the context models, which also need the index closes and the
   times NVIDIA published results.
5. For the size of the move: appends each volatility forecaster's range for the
   next session, and, for every range whose session has now finished, whether the
   return fell inside it, below it or above it.

The run refuses, loudly, to write any prediction when:

- Yahoo has a newer session that it has not finished publishing. Predicting from
  the session before would be a call on a session that has already happened.
- The window for the latest session has passed (the next weekday's open). A late
  prediction is not written; that day is simply missing from the log.

In those cases, and when the prices or the log are bad, nothing is written.

The context models are the one exception to all-or-nothing. If their extra
downloads fail, or it is not yet 20:00 New York on the day of the session, the
baselines and the technical models are written exactly as they would have been,
the context models make no call, and the run still ends as a failure so that it
is noticed (exit code 3). A later run can add the missing context calls only while
the window is still open, and it uses the same fixed 20:00 cut-off, so the call
is the one the earlier run would have made. Once the market has opened, the day is
missed for them and is never filled in.

Running it again the same morning adds nothing. Prices are held in memory only.
The log it writes contains directions, forecast ranges and where each return fell
against its range, never prices or realised returns.

    uv run python -m markets_lab.analysis.nvda.nightly --dry-run
    uv run python -m markets_lab.analysis.nvda.nightly
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from markets_lab.analysis.nvda.backtest import (
    BASELINE_FACTORIES,
    CONTEXT_MODEL_FACTORIES,
    MODEL_FACTORIES,
    PredictorFactory,
)
from markets_lab.analysis.nvda.context_features import CallTooEarlyError, build_context_dataset
from markets_lab.analysis.nvda.features import Dataset, build_dataset, compute_outcomes
from markets_lab.analysis.nvda.prediction_log import (
    DEFAULT_LOG_PATH,
    LogError,
    Outcome,
    Prediction,
    RangeForecast,
    RangeOutcome,
    Record,
    append_records,
    prediction_window,
    range_position,
    read_log,
)
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP
from markets_lab.analysis.nvda.volatility import backtest as volatility_backtest
from markets_lab.analysis.nvda.volatility.backtest import ForecasterFactory
from markets_lab.analysis.nvda.volatility.dataset import build_volatility_dataset
from markets_lab.analysis.nvda.volatility.forecasters import LOWER, RANGE_LEVEL, UPPER
from markets_lab.collectors import market_context, nvda_earnings
from markets_lab.collectors.nvda_earnings import Getter
from markets_lab.collectors.nvda_news import sec_get
from markets_lab.collectors.nvda_prices import (
    MIN_ROWS,
    Downloader,
    DroppedBar,
    PriceDataError,
    download_raw,
    fetch_daily,
)

EXIT_FAILED = 1
EXIT_CONTEXT_SKIPPED = 3
"""The log was written, but the context models made no call in this run."""

ContextLoader = Callable[[], Dataset]


class NightlyError(RuntimeError):
    """The run cannot honestly write a prediction, so it writes nothing."""


_EXPECTED = (NightlyError, PriceDataError, nvda_earnings.EarningsDataError, CallTooEarlyError)
"""Reasons a context call can be skipped that explain themselves. Anything else is
reported with its type, since it points at a bug rather than at missing data."""


@dataclass(frozen=True, slots=True)
class NightlyResult:
    """What a run added to the log (or would add, on a dry run)."""

    last_session: date
    outcomes: tuple[Outcome, ...]
    predictions: tuple[Prediction, ...]
    skipped: tuple[str, ...] = ()
    """Context models that owed a call and did not make one in this run."""
    skip_reason: str | None = None
    range_outcomes: tuple[RangeOutcome, ...] = ()
    range_forecasts: tuple[RangeForecast, ...] = ()

    @property
    def records(self) -> list[Record]:
        """Everything to append, in the order the log's rules need: what happened
        first, then the calls on what happens next."""
        return [*self.outcomes, *self.range_outcomes, *self.predictions, *self.range_forecasts]


def default_factories() -> dict[str, PredictorFactory]:
    """Every baseline and every measured model on the technical columns, in report order."""
    return {**BASELINE_FACTORIES, **MODEL_FACTORIES}


def default_range_factories() -> dict[str, ForecasterFactory]:
    """Every volatility baseline and model, in report order."""
    return {**volatility_backtest.BASELINE_FACTORIES, **volatility_backtest.MODEL_FACTORIES}


def plan_records(
    prices: pd.DataFrame,
    dropped: Sequence[DroppedBar],
    existing: Sequence[Record],
    *,
    now: datetime,
    factories: Mapping[str, PredictorFactory],
    context_factories: Mapping[str, PredictorFactory] | None = None,
    load_context: ContextLoader | None = None,
    range_factories: Mapping[str, ForecasterFactory] | None = None,
) -> NightlyResult:
    """Work out which outcomes and predictions this run should append.

    `prices` holds completed sessions only. Nothing is written here.

    `load_context` returns the dataset with the context and results columns for
    these prices, as of `now`. It is called only if a context model owes a call.
    Whatever goes wrong in it, or in a context model, costs only the context
    models' calls: the others are worked out first and do not depend on it.
    """
    context_factories = context_factories or {}
    range_factories = range_factories or {}
    overlap = set(factories) & set(context_factories)
    if overlap:
        raise ValueError(f"names used for both a model and a context model: {sorted(overlap)}")
    dataset = build_dataset(prices)
    last_session, _ = dataset.latest()

    unpublished = [bar for bar in dropped if bar.reason == "incomplete"]
    if unpublished:
        raise NightlyError(
            f"session {unpublished[0].session} has finished but Yahoo has not published its "
            f"close yet; no prediction can be made until it is available"
        )

    outcomes = _missing_outcomes(prices, existing, now=now)
    range_outcomes = _missing_range_outcomes(prices, existing, outcomes, now=now)

    already = {(r.model, r.based_on_session) for r in existing if isinstance(r, Prediction)}
    missing = [name for name in factories if (name, last_session) not in already]
    context_missing = [name for name in context_factories if (name, last_session) not in already]
    ranged = {(r.model, r.based_on_session) for r in existing if isinstance(r, RangeForecast)}
    ranges_missing = [name for name in range_factories if (name, last_session) not in ranged]
    predictions: list[Prediction] = []
    skipped: tuple[str, ...] = ()
    skip_reason: str | None = None
    if missing or context_missing or ranges_missing:
        # One deadline for every call, context or not: nothing is filled in once
        # the next session could have opened.
        _, deadline = prediction_window(last_session)
        if now >= deadline:
            raise NightlyError(
                f"too late to predict the session after {last_session}: predictions had to be "
                f"written before {deadline.strftime('%Y-%m-%d %H:%M UTC')}, and it is now "
                f"{now.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
            )
    if missing:
        predictions += _calls(missing, factories, dataset, last_session, now)
    range_forecasts = _ranges(ranges_missing, range_factories, prices, last_session, now)
    if context_missing:
        try:
            if load_context is None:
                raise NightlyError("no source for the market context and results dates")
            extended = load_context()
            if not extended.features.index.equals(dataset.features.index):
                raise NightlyError("the context dataset is not on the same sessions as the prices")
            context_calls = _calls(context_missing, context_factories, extended, last_session, now)
        except Exception as exc:
            skipped = tuple(context_missing)
            skip_reason = str(exc) if isinstance(exc, _EXPECTED) else f"{type(exc).__name__}: {exc}"
        else:
            predictions += context_calls
    return NightlyResult(
        last_session,
        tuple(outcomes),
        tuple(predictions),
        skipped,
        skip_reason,
        tuple(range_outcomes),
        tuple(range_forecasts),
    )


def _ranges(
    names: Sequence[str],
    factories: Mapping[str, ForecasterFactory],
    prices: pd.DataFrame,
    last_session: date,
    now: datetime,
) -> list[RangeForecast]:
    """Fit each named forecaster on the sessions with a known next return and take
    its range for the session after the last one, as simple returns."""
    if not names:
        return []
    dataset = build_volatility_dataset(prices)
    if dataset.latest()[0] != last_session:
        raise NightlyError("the volatility dataset does not end on the latest session")
    known = dataset.labelled()
    row = dataset.features.iloc[[-1]]
    forecasts: list[RangeForecast] = []
    for name in names:
        forecaster = factories[name]()
        forecaster.fit(known.features, known.next_log_return)
        forecast = forecaster.predict(row).iloc[0]
        forecasts.append(
            RangeForecast(
                made_at=now,
                based_on_session=last_session,
                model=name,
                level=RANGE_LEVEL,
                lower_return=math.expm1(float(forecast[LOWER])),
                upper_return=math.expm1(float(forecast[UPPER])),
            )
        )
    return forecasts


def _missing_range_outcomes(
    prices: pd.DataFrame,
    existing: Sequence[Record],
    new_outcomes: Sequence[Outcome],
    *,
    now: datetime,
) -> list[RangeOutcome]:
    """A range outcome for every range forecast whose session has an outcome (on
    record or about to be) and has not been scored yet.

    The position is worked out against the bounds as they were written to the log.
    """
    target = {
        o.previous_session: o.session
        for o in [*(r for r in existing if isinstance(r, Outcome)), *new_outcomes]
    }
    scored = {(r.model, r.previous_session) for r in existing if isinstance(r, RangeOutcome)}
    owed = [
        r
        for r in existing
        if isinstance(r, RangeForecast)
        and r.based_on_session in target
        and (r.model, r.based_on_session) not in scored
    ]
    if not owed:
        return []
    next_return = compute_outcomes(prices)["next_return"]
    results: list[RangeOutcome] = []
    for forecast in owed:
        stamp = pd.Timestamp(forecast.based_on_session)
        if stamp not in next_return.index or pd.isna(next_return.loc[stamp]):
            raise NightlyError(
                f"session {forecast.based_on_session} has a range forecast but its next "
                "return is not in the downloaded prices"
            )
        results.append(
            RangeOutcome(
                recorded_at=now,
                session=target[forecast.based_on_session],
                previous_session=forecast.based_on_session,
                model=forecast.model,
                position=range_position(
                    forecast.lower_return, forecast.upper_return, float(next_return.loc[stamp])
                ),
            )
        )
    return results


def _calls(
    names: Sequence[str],
    factories: Mapping[str, PredictorFactory],
    dataset: Dataset,
    last_session: date,
    now: datetime,
) -> list[Prediction]:
    """Fit each named predictor on the sessions with a known outcome and take its
    call from the last session's row."""
    known = dataset.labelled()
    row = dataset.features.iloc[[-1]]
    calls: list[Prediction] = []
    for name in names:
        predictor = factories[name]()
        predictor.fit(known.features, known.target_up)
        call = predictor.predict(row).iloc[0]
        probability = float(call[PROBABILITY_UP])
        calls.append(
            Prediction(
                made_at=now,
                based_on_session=last_session,
                model=name,
                predicted_up=bool(call[PREDICTED_UP]),
                probability_up=None if pd.isna(probability) else probability,
            )
        )
    return calls


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
    get: Getter = sec_get,
    factories: Mapping[str, PredictorFactory] | None = None,
    context_factories: Mapping[str, PredictorFactory] | None = None,
    range_factories: Mapping[str, ForecasterFactory] | None = None,
    dry_run: bool = False,
    min_rows: int = MIN_ROWS,
) -> NightlyResult:
    """Run the job once and append what it produced in one step.

    Nothing is written if the run is refused. If only the context models could
    not make their call, everything else is written and `skipped` names them.
    """
    fetched = fetch_daily(now=now, downloader=downloader, min_rows=min_rows)
    existing = read_log(log_path)

    def load_context() -> Dataset:
        context = market_context.fetch_context(now=now, downloader=downloader, min_rows=min_rows)
        publications = nvda_earnings.fetch_publications(now=now, get=get)
        # `as_of` is this run's own timestamp, the one written into each call.
        # The dataset refuses it if the 20:00 cut-off has not passed, and uses
        # only results published before that cut-off.
        return build_context_dataset(fetched.prices, context.closes, publications, as_of=now)

    result = plan_records(
        fetched.prices,
        fetched.dropped,
        existing,
        now=now,
        factories=default_factories() if factories is None else factories,
        context_factories=(
            CONTEXT_MODEL_FACTORIES if context_factories is None else context_factories
        ),
        load_context=load_context,
        range_factories=default_range_factories() if range_factories is None else range_factories,
    )
    if not dry_run:
        append_records(log_path, result.records)
    return result


def describe(result: NightlyResult, *, dry_run: bool) -> list[str]:
    """Plain-text lines saying what the run did."""
    verb = "would append" if dry_run else "appended"
    lines = [f"last completed session: {result.last_session}"]
    if not result.records:
        lines.append("nothing new to record")
    for outcome in result.outcomes:
        direction = "up" if outcome.went_up else "not up"
        lines.append(f"{verb} outcome: {outcome.session} closed {direction}")
    for scored in result.range_outcomes:
        lines.append(
            f"{verb} range outcome: {scored.session} closed {scored.position} "
            f"{scored.model}'s range"
        )
    for prediction in result.predictions:
        direction = "up" if prediction.predicted_up else "down"
        probability = (
            "" if prediction.probability_up is None else f" (p up {prediction.probability_up:.4f})"
        )
        lines.append(
            f"{verb} prediction: {prediction.model} calls {direction}{probability} "
            f"for the session after {prediction.based_on_session}"
        )
    for forecast in result.range_forecasts:
        lines.append(
            f"{verb} range: {forecast.model} expects {forecast.lower_return:+.2%} to "
            f"{forecast.upper_return:+.2%} ({forecast.level:.0%} range) for the session after "
            f"{forecast.based_on_session}"
        )
    if result.skipped:
        lines.append(
            f"NO CALL from {', '.join(result.skipped)} for the session after "
            f"{result.last_session}: {result.skip_reason}"
        )
        lines.append(
            "a later run can still add these calls, but only before the next session opens"
        )
    return lines


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    downloader: Downloader = download_raw,
    get: Getter = sec_get,
    factories: Mapping[str, PredictorFactory] | None = None,
    context_factories: Mapping[str, PredictorFactory] | None = None,
    range_factories: Mapping[str, ForecasterFactory] | None = None,
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
            get=get,
            factories=factories,
            context_factories=context_factories,
            range_factories=range_factories,
            dry_run=args.dry_run,
            min_rows=min_rows,
        )
    except (NightlyError, PriceDataError, LogError) as exc:
        report(f"NIGHTLY RUN FAILED, nothing was written: {exc}")
        return EXIT_FAILED
    for line in describe(result, dry_run=args.dry_run):
        report(line)
    if result.skipped:
        report("NIGHTLY RUN INCOMPLETE: the context models made no call (see above)")
        return EXIT_CONTEXT_SKIPPED
    return 0


if __name__ == "__main__":
    sys.exit(main())
