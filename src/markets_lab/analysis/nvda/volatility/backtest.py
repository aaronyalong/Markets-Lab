"""Walk-forward backtest of the volatility forecasters; writes the aggregate report.

The layout is the direction backtest's: the same sessions, the same expanding
window and the same folds, so both reports describe the same days. Every forecaster
is refitted on each fold's training rows and then forecasts the fold's test rows.

The report holds aggregates only: losses, coverage, average range width and
intervals. It contains no prices and no per-day series.

Run it from the repository root, after fetching prices:

    uv run python -m markets_lab.collectors.nvda_prices
    uv run python -m markets_lab.analysis.nvda.volatility.backtest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.metrics import BootstrapConfig, bootstrap_interval
from markets_lab.analysis.nvda.validation import (
    FOLD,
    TRAINED_THROUGH,
    Fold,
    LookAheadError,
    WalkForwardConfig,
    make_folds,
)
from markets_lab.analysis.nvda.volatility.dataset import (
    VolatilityDataset,
    build_volatility_dataset,
)
from markets_lab.analysis.nvda.volatility.forecasters import (
    FORECAST_COLUMNS,
    LOWER,
    RANGE_LEVEL,
    UPPER,
    VOLATILITY,
    ConstantVolatility,
    EwmaVolatility,
    RangeForecaster,
    RollingVolatility,
)
from markets_lab.analysis.nvda.volatility.metrics import inside, interval_scores, qlike_losses
from markets_lab.analysis.nvda.volatility.models import HarVolatility, XGBoostVolatility
from markets_lab.collectors.nvda_prices import (
    DEFAULT_CSV_PATH,
    MARKET_TZ,
    MIN_ROWS,
    SYMBOL,
    PriceDataError,
    load_csv,
    validate,
)

DEFAULT_REPORT_PATH = Path("reports/nvda_volatility.json")
REFERENCE = "constant"
"""The bar to beat: assume volatility is always its long-run average."""

ForecasterFactory = Callable[[], RangeForecaster]

BASELINE_FACTORIES: dict[str, ForecasterFactory] = {
    "constant": ConstantVolatility,
    "rolling_21": RollingVolatility,
    "ewma_94": EwmaVolatility,
}

MODEL_FACTORIES: dict[str, ForecasterFactory] = {
    "har": HarVolatility,
    "xgboost_vol": XGBoostVolatility,
}
"""Models are added here one at a time, each after the previous one is measured."""

CAVEATS = (
    "Research only, not financial advice.",
    "The backtest is where forecasters are compared, so it flatters whichever looks "
    "best. The live prediction log is the real test.",
    "A squared daily return is a very noisy measure of that day's volatility. QLIKE "
    "ranks forecasters correctly despite that, but its level has no plain meaning.",
    "Ranges are for the close-to-close return, adjusted for dividends and splits.",
    "A range is built to hold 80% of outcomes on average over many days. On any one "
    "day it either holds the outcome or it does not.",
    "Several forecasters are compared, so some intervals will exclude zero by chance.",
)


def walk_forward_ranges(
    dataset: VolatilityDataset,
    make_forecaster: ForecasterFactory,
    config: WalkForwardConfig | None = None,
    *,
    folds: Sequence[Fold] | None = None,
) -> pd.DataFrame:
    """Out-of-sample forecasts for every session after the initial training window.

    A fresh forecaster is built for each fold. The dataset must be fully labelled.
    Returns one row per tested session with `FORECAST_COLUMNS`, `fold` and
    `trained_through`.
    """
    if bool(dataset.next_log_return.isna().any()):
        raise ValueError("walk_forward_ranges needs a fully labelled dataset; use labelled()")
    index = dataset.features.index
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("dataset sessions must be unique and in increasing order")
    if folds is None:
        folds = make_folds(len(dataset), config or WalkForwardConfig())

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        if fold.train_end < 1 or not fold.test_start < fold.test_end <= len(dataset):
            raise ValueError(f"fold {fold.number}: rows are empty or out of range")
        if fold.train_end > fold.test_start:
            raise LookAheadError(
                f"fold {fold.number}: trains on rows up to {fold.train_end - 1} "
                f"but is tested from row {fold.test_start}"
            )
        test_features = dataset.features.iloc[fold.test_start : fold.test_end]
        forecaster = make_forecaster()
        forecaster.fit(
            dataset.features.iloc[: fold.train_end], dataset.next_log_return.iloc[: fold.train_end]
        )
        forecasts = forecaster.predict(test_features)
        if tuple(forecasts.columns) != FORECAST_COLUMNS or not forecasts.index.equals(
            test_features.index
        ):
            raise ValueError(f"{forecaster.name}: predict returned the wrong columns or sessions")
        forecasts = forecasts.copy()
        forecasts[FOLD] = fold.number
        forecasts[TRAINED_THROUGH] = index[fold.train_end - 1]
        pieces.append(forecasts)

    result = pd.concat(pieces)
    if result.index.has_duplicates:
        raise LookAheadError("folds overlap: a session was tested more than once")
    return result


def run_backtest(
    dataset: VolatilityDataset,
    *,
    baselines: Mapping[str, ForecasterFactory] = BASELINE_FACTORIES,
    models: Mapping[str, ForecasterFactory] = MODEL_FACTORIES,
    config: WalkForwardConfig | None = None,
    bootstrap: BootstrapConfig | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Backtest every baseline and model and return the aggregate report."""
    config = config or WalkForwardConfig()
    bootstrap = bootstrap or BootstrapConfig()
    overlap = set(baselines) & set(models)
    if overlap:
        raise ValueError(f"names used for both a baseline and a model: {sorted(overlap)}")

    labelled = dataset.labelled()
    folds = make_folds(len(labelled), config)
    forecasts = {
        name: walk_forward_ranges(labelled, factory, config, folds=folds)
        for name, factory in {**baselines, **models}.items()
    }
    sessions = next(iter(forecasts.values())).index
    if not all(frame.index.equals(sessions) for frame in forecasts.values()):
        raise ValueError("forecasters were not tested on the same sessions")
    outcome = labelled.next_log_return.loc[sessions].to_numpy(dtype="float64")
    n = len(sessions)

    qlike: dict[str, np.ndarray] = {}
    score: dict[str, np.ndarray] = {}
    forecasters: dict[str, Any] = {}
    for name, frame in forecasts.items():
        volatility = frame[VOLATILITY].to_numpy(dtype="float64")
        lower = frame[LOWER].to_numpy(dtype="float64")
        upper = frame[UPPER].to_numpy(dtype="float64")
        held = inside(lower, upper, outcome)
        qlike[name] = qlike_losses(volatility, outcome)
        score[name] = interval_scores(lower, upper, outcome, RANGE_LEVEL)
        # Widths are reported as simple returns, the way a reader thinks of a move.
        width = np.expm1(upper) - np.expm1(lower)
        forecasters[name] = {
            "kind": "baseline" if name in baselines else "model",
            "sessions": n,
            "qlike": _number(float(qlike[name].mean())),
            "interval_score": _number(float(score[name].mean())),
            "coverage": _number(float(held.mean())),
            "coverage_interval": _pair(
                bootstrap_interval(n, lambda rows, held=held: float(held[rows].mean()), bootstrap)
            ),
            "share_below": _number(float((outcome < lower).mean())),
            "share_above": _number(float((outcome > upper).mean())),
            "mean_width": _number(float(width.mean())),
            "median_width": _number(float(np.median(width))),
            "mean_forecast_volatility": _number(float(volatility.mean())),
        }

    # Every forecaster is compared with each one listed before it.
    comparisons: list[dict[str, Any]] = []
    names = list(forecasts)
    for position, name in enumerate(names):
        comparisons += [
            _compare(name, reference, qlike, score, n, bootstrap) for reference in names[:position]
        ]

    return {
        "generated_at": (generated_at or datetime.now(tz=UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "target": "next session's close-to-close return (adjusted), as a log return",
        "range_level": RANGE_LEVEL,
        "data": {
            "sessions_with_features": len(dataset),
            "first_feature_session": _date(dataset.features.index[0]),
            "last_labelled_session": _date(labelled.features.index[-1]),
            "tested_sessions": n,
            "first_tested_session": _date(sessions[0]),
            "last_tested_session": _date(sessions[-1]),
            "realised_volatility": _number(float(np.sqrt(np.mean(outcome**2)))),
            "mean_absolute_move": _number(float(np.abs(np.expm1(outcome)).mean())),
        },
        "validation": {
            "scheme": "walk-forward, expanding window",
            "initial_train_sessions": config.initial_train_sessions,
            "refit_every_sessions": config.refit_every,
            "folds": len(folds),
        },
        "intervals": {
            "confidence": bootstrap.confidence,
            "method": "paired moving-block bootstrap",
            "block_length_sessions": bootstrap.block_length,
            "resamples": bootstrap.n_resamples,
            "seed": bootstrap.seed,
        },
        "forecasters": forecasters,
        "comparisons": comparisons,
        "caveats": list(CAVEATS),
    }


def _compare(
    name: str,
    reference: str,
    qlike: Mapping[str, np.ndarray],
    score: Mapping[str, np.ndarray],
    n: int,
    bootstrap: BootstrapConfig,
) -> dict[str, Any]:
    qlike_gap = qlike[name] - qlike[reference]
    score_gap = score[name] - score[reference]
    return {
        "forecaster": name,
        "reference": reference,
        "qlike_difference": _number(float(qlike_gap.mean())),
        "qlike_difference_interval": _pair(
            bootstrap_interval(n, lambda rows: float(qlike_gap[rows].mean()), bootstrap)
        ),
        "interval_score_difference": _number(float(score_gap.mean())),
        "interval_score_difference_interval": _pair(
            bootstrap_interval(n, lambda rows: float(score_gap[rows].mean()), bootstrap)
        ),
    }


def format_report(report: Mapping[str, Any]) -> str:
    """A plain-text summary of the report for the terminal."""
    data, validation = report["data"], report["validation"]
    level = report["range_level"]
    lines = [
        f"{report['symbol']} volatility walk-forward backtest",
        f"tested {data['tested_sessions']} sessions, {data['first_tested_session']} to "
        f"{data['last_tested_session']}, {validation['folds']} folds; "
        f"average move {_percent(data['mean_absolute_move'], 2)} a day",
        "",
        f"{'forecaster':<14}{'QLIKE':>9}{'coverage':>10}  {'95% interval':<17}"
        f"{'below':>7}{'above':>7}{'avg width':>11}{'int. score':>12}",
    ]
    for name, row in report["forecasters"].items():
        lines.append(
            f"{name:<14}{row['qlike']:>9.4f}{_percent(row['coverage']):>10}  "
            f"{_range(row['coverage_interval']):<17}{_percent(row['share_below']):>7}"
            f"{_percent(row['share_above']):>7}{_percent(row['mean_width'], 2):>11}"
            f"{row['interval_score']:>12.5f}"
        )
    lines += [
        "",
        f"the range is built to hold {level:.0%} of outcomes; lower QLIKE and interval "
        "score are better",
        "differences, forecaster minus reference:",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"  {row['forecaster']} vs {row['reference']}: QLIKE "
            f"{row['qlike_difference']:+.4f} {_plain_range(row['qlike_difference_interval'], 4)}"
            f", interval score {row['interval_score_difference']:+.5f} "
            f"{_plain_range(row['interval_score_difference_interval'], 5)}"
        )
    return "\n".join(lines)


def _number(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else round(value, 6)


def _pair(interval: tuple[float, float]) -> list[float | None]:
    return [_number(interval[0]), _number(interval[1])]


def _date(timestamp: object) -> str:
    return pd.Timestamp(timestamp).date().isoformat()  # type: ignore[arg-type]


def _percent(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def _range(interval: Sequence[float | None]) -> str:
    low, high = interval
    if low is None or high is None:
        return "[n/a]"
    return f"[{100 * low:.1f}%, {100 * high:.1f}%]"


def _plain_range(interval: Sequence[float | None], digits: int) -> str:
    low, high = interval
    if low is None or high is None:
        return "[n/a]"
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write the report as JSON, replacing any existing file in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(tz=MARKET_TZ),
    report: Callable[[str], None] = _print,
    min_rows: int = MIN_ROWS,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.analysis.nvda.volatility.backtest",
        description="Walk-forward backtest of the volatility forecasters; writes aggregates only.",
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--resamples", type=int, default=BootstrapConfig().n_resamples)
    args = parser.parse_args(argv)
    try:
        prices = load_csv(args.prices)
        validate(prices, now=now(), min_rows=min_rows)
    except FileNotFoundError:
        report(f"error: {args.prices} not found; run the price fetch first")
        return 2
    except PriceDataError as exc:
        report(f"error: {exc}")
        return 1
    result = run_backtest(
        build_volatility_dataset(prices), bootstrap=BootstrapConfig(n_resamples=args.resamples)
    )
    write_report(result, args.out)
    report(format_report(result))
    report(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
