"""Run the walk-forward backtest and write the aggregate report.

Every predictor makes its calls through the same walk-forward folds, so they are
scored on exactly the same sessions. The report holds aggregates only: counts,
accuracies, summary returns and intervals. It contains no prices and no per-day
series, so it is safe to commit and publish.

Models come in two groups. The first sees the technical columns only. The second
(the "context" models) also sees the market context and results-date columns. Both
are tested on the same sessions, so a context model can be compared with its twin.

Run it from the repository root, after fetching the data:

    uv run python -m markets_lab.collectors.nvda_prices
    uv run python -m markets_lab.collectors.market_context
    uv run python -m markets_lab.collectors.nvda_earnings
    uv run python -m markets_lab.analysis.nvda.backtest
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

from markets_lab.analysis.nvda.baselines import AlwaysUp, MovingAverageRule, Persistence
from markets_lab.analysis.nvda.context_features import (
    EXTENDED_FEATURE_COLUMNS,
    CallTooEarlyError,
    build_context_dataset,
)
from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, Dataset
from markets_lab.analysis.nvda.metrics import (
    BootstrapConfig,
    annualised_return,
    annualised_volatility,
    bootstrap_interval,
    brier_score,
    hits,
    sharpe_ratio,
    strategy_net_returns,
    total_return,
    trade_count,
    wilson_interval,
)
from markets_lab.analysis.nvda.models.context import LogisticContextModel
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.models.lstm import LSTMModel
from markets_lab.analysis.nvda.models.xgb import XGBoostModel
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP, Predictor
from markets_lab.analysis.nvda.validation import WalkForwardConfig, make_folds, walk_forward
from markets_lab.collectors import market_context, nvda_earnings
from markets_lab.collectors.nvda_prices import (
    DEFAULT_CSV_PATH,
    MARKET_TZ,
    MIN_ROWS,
    SYMBOL,
    PriceDataError,
    load_csv,
    validate,
)

DEFAULT_REPORT_PATH = Path("reports/nvda_backtest.json")
COST_LEVELS_BPS = (0, 5, 10)
PRIMARY_COST_BPS = 5
BUY_AND_HOLD = "always_up"

PredictorFactory = Callable[[], Predictor]

BASELINE_FACTORIES: dict[str, PredictorFactory] = {
    "always_up": AlwaysUp,
    "persistence": Persistence,
    "ma_200": MovingAverageRule,
}

MODEL_FACTORIES: dict[str, PredictorFactory] = {
    "logistic": LogisticRegressionModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
}
"""Models are added here one at a time, each after the previous one is measured."""

CONTEXT_MODEL_FACTORIES: dict[str, PredictorFactory] = {
    "logistic_ctx": LogisticContextModel,
}
"""The same models on the wider inputs, also added one at a time."""

TECHNICAL_INPUTS = "technical"
CONTEXT_INPUTS = "technical, market context and results dates"

CAVEATS = (
    "Research only, not financial advice.",
    "The backtest is where predictors are compared, so it flatters whichever looks "
    "best. The live prediction log is the real test.",
    "Several predictors are compared, so some intervals will exclude zero by chance.",
    "Accuracy intervals (Wilson) treat sessions as independent.",
    "Simulated returns assume a flat cost per trade, no slippage or taxes, and cash "
    "that earns no interest.",
    "Following always_up is buy-and-hold; it is the return benchmark.",
    "The base-rate Brier score uses the share of up days in the tested period, which "
    "no forecaster knew in advance, so it is a slightly hard reference.",
)


def run_backtest(
    dataset: Dataset,
    *,
    baselines: Mapping[str, PredictorFactory] = BASELINE_FACTORIES,
    models: Mapping[str, PredictorFactory] = MODEL_FACTORIES,
    context_models: Mapping[str, PredictorFactory] | None = None,
    config: WalkForwardConfig | None = None,
    bootstrap: BootstrapConfig | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Backtest every baseline and model and return the aggregate report.

    Baselines and `models` are given the technical columns only, whatever else the
    dataset holds. `context_models` are given every column, and the dataset must
    then carry the context and results columns (see `build_context_dataset`).
    """
    config = config or WalkForwardConfig()
    bootstrap = bootstrap or BootstrapConfig()
    context_models = context_models or {}
    overlap = set(baselines) & set(models)
    if overlap:
        raise ValueError(f"names used for both a baseline and a model: {sorted(overlap)}")
    overlap = (set(baselines) | set(models)) & set(context_models)
    if overlap:
        raise ValueError(f"names used for both a model and a context model: {sorted(overlap)}")

    labelled = dataset.labelled()
    folds = make_folds(len(labelled), config)
    technical = _with_columns(labelled, FEATURE_COLUMNS)
    calls = {
        name: walk_forward(technical, factory, config, folds=folds)
        for name, factory in {**baselines, **models}.items()
    }
    if context_models:
        extended = _with_columns(labelled, EXTENDED_FEATURE_COLUMNS)
        calls.update(
            {
                name: walk_forward(extended, factory, config, folds=folds)
                for name, factory in context_models.items()
            }
        )
    sessions = next(iter(calls.values())).index
    if not all(frame.index.equals(sessions) for frame in calls.values()):
        raise ValueError("predictors were not tested on the same sessions")
    target = labelled.target_up.loc[sessions].to_numpy(dtype="float64")
    next_return = labelled.next_return.loc[sessions].to_numpy(dtype="float64")
    n = len(sessions)

    predicted = {name: frame[PREDICTED_UP].to_numpy(dtype=bool) for name, frame in calls.items()}
    hit = {name: hits(up, target) for name, up in predicted.items()}
    net = {
        name: {
            cost: strategy_net_returns(up, next_return, cost_bps=cost) for cost in COST_LEVELS_BPS
        }
        for name, up in predicted.items()
    }

    predictors: dict[str, Any] = {}
    for name, up in predicted.items():
        hit_count = int(hit[name].sum())
        predictors[name] = {
            "kind": "baseline" if name in baselines else "model",
            "inputs": CONTEXT_INPUTS if name in context_models else TECHNICAL_INPUTS,
            "sessions": n,
            "hits": hit_count,
            "accuracy": _number(hit_count / n),
            "accuracy_interval": _pair(wilson_interval(hit_count, n, bootstrap.confidence)),
            "share_called_up": _number(float(up.mean())),
            "brier_score": _brier(calls[name][PROBABILITY_UP].to_numpy(dtype="float64"), target),
            "trades": trade_count(up),
            "returns_by_cost_bps": {
                str(cost): _return_summary(net[name][cost], n, bootstrap)
                for cost in COST_LEVELS_BPS
            },
        }

    # Every predictor is compared with each one listed before it: baselines with
    # earlier baselines, models with every baseline and every earlier model.
    comparisons: list[dict[str, Any]] = []
    names = list(calls)
    for position, name in enumerate(names):
        comparisons += [
            _compare(name, reference, hit, net, n, bootstrap) for reference in names[:position]
        ]

    return {
        "generated_at": (generated_at or datetime.now(tz=UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "data": {
            "sessions_with_features": len(dataset),
            "first_feature_session": _date(dataset.features.index[0]),
            "last_labelled_session": _date(labelled.features.index[-1]),
            "tested_sessions": n,
            "first_tested_session": _date(sessions[0]),
            "last_tested_session": _date(sessions[-1]),
            "share_of_up_days_tested": _number(float(target.mean())),
            "brier_score_of_base_rate": _number(float(target.mean() * (1 - target.mean()))),
        },
        "validation": {
            "scheme": "walk-forward, expanding window",
            "initial_train_sessions": config.initial_train_sessions,
            "refit_every_sessions": config.refit_every,
            "folds": len(folds),
        },
        "strategy": "hold the stock when the call is up, cash otherwise",
        "cost_levels_bps": list(COST_LEVELS_BPS),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "intervals": {
            "confidence": bootstrap.confidence,
            "accuracy": "Wilson",
            "returns_and_differences": "paired moving-block bootstrap",
            "block_length_sessions": bootstrap.block_length,
            "resamples": bootstrap.n_resamples,
            "seed": bootstrap.seed,
        },
        "predictors": predictors,
        "comparisons": comparisons,
        "caveats": list(CAVEATS),
    }


def _with_columns(dataset: Dataset, columns: Sequence[str]) -> Dataset:
    """The same sessions and outcomes with exactly these feature columns."""
    missing = [name for name in columns if name not in dataset.features.columns]
    if missing:
        raise ValueError(f"the dataset has no columns {missing[:5]}")
    return Dataset(dataset.features[list(columns)], dataset.target_up, dataset.next_return)


def _return_summary(net: np.ndarray, n: int, bootstrap: BootstrapConfig) -> dict[str, Any]:
    interval = bootstrap_interval(n, lambda rows: annualised_return(net[rows]), bootstrap)
    return {
        "total_return": _number(total_return(net)),
        "annualised_return": _number(annualised_return(net)),
        "annualised_return_interval": _pair(interval),
        "annualised_volatility": _number(annualised_volatility(net)),
        "sharpe_ratio": _number(sharpe_ratio(net)),
    }


def _brier(probability: np.ndarray, target: np.ndarray) -> float | None:
    """Brier score, or None for a rule that gives no probabilities."""
    if np.isnan(probability).all():
        return None
    return _number(brier_score(probability, target))


def _compare(
    name: str,
    reference: str,
    hit: Mapping[str, np.ndarray],
    net: Mapping[str, Mapping[int, np.ndarray]],
    n: int,
    bootstrap: BootstrapConfig,
) -> dict[str, Any]:
    ours, theirs = hit[name], hit[reference]
    our_net, their_net = net[name][PRIMARY_COST_BPS], net[reference][PRIMARY_COST_BPS]

    def accuracy_gap(rows: np.ndarray) -> float:
        return float(ours[rows].mean() - theirs[rows].mean())

    def return_gap(rows: np.ndarray) -> float:
        return annualised_return(our_net[rows]) - annualised_return(their_net[rows])

    everything = np.arange(n)
    return {
        "predictor": name,
        "reference": reference,
        "accuracy_difference": _number(accuracy_gap(everything)),
        "accuracy_difference_interval": _pair(bootstrap_interval(n, accuracy_gap, bootstrap)),
        "annualised_return_difference": _number(return_gap(everything)),
        "annualised_return_difference_interval": _pair(
            bootstrap_interval(n, return_gap, bootstrap)
        ),
        "cost_bps": PRIMARY_COST_BPS,
    }


def format_report(report: Mapping[str, Any]) -> str:
    """A plain-text summary of the report for the terminal."""
    data, validation = report["data"], report["validation"]
    cost = str(report["primary_cost_bps"])
    lines = [
        f"{report['symbol']} walk-forward backtest",
        f"tested {data['tested_sessions']} sessions, {data['first_tested_session']} to "
        f"{data['last_tested_session']}, {validation['folds']} folds; "
        f"{_percent(data['share_of_up_days_tested'])} of tested days were up",
        "",
        f"{'predictor':<14}{'accuracy':>9}  {'95% interval':<17}{'called up':>10}"
        f"{'trades':>8}{'ann. return':>13}  {'95% interval':<19}{'Sharpe':>7}",
    ]
    for name, row in report["predictors"].items():
        returns = row["returns_by_cost_bps"][cost]
        lines.append(
            f"{name:<14}{_percent(row['accuracy']):>9}  "
            f"{_range(row['accuracy_interval']):<17}{_percent(row['share_called_up']):>10}"
            f"{row['trades']:>8}{_percent(returns['annualised_return']):>13}  "
            f"{_range(returns['annualised_return_interval']):<19}"
            f"{_plain(returns['sharpe_ratio']):>7}"
        )
    scored = {n: r["brier_score"] for n, r in report["predictors"].items() if r["brier_score"]}
    if scored:
        reference = data["brier_score_of_base_rate"]
        lines += [
            "",
            f"probability quality (Brier score, lower is better; base rate {reference:.4f}):",
        ]
        lines += [f"  {name}: {score:.4f}" for name, score in scored.items()]
    lines += ["", f"differences (returns at {cost} bps per trade), predictor minus reference:"]
    for row in report["comparisons"]:
        accuracy_range = _range(row["accuracy_difference_interval"], points=True)
        return_range = _range(row["annualised_return_difference_interval"], points=True)
        lines.append(
            f"  {row['predictor']} vs {row['reference']}: accuracy "
            f"{_points(row['accuracy_difference'])} {accuracy_range}"
            f", annualised return {_points(row['annualised_return_difference'])} {return_range}"
        )
    return "\n".join(lines)


def _number(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else round(value, 6)


def _pair(interval: tuple[float, float]) -> list[float | None]:
    return [_number(interval[0]), _number(interval[1])]


def _date(timestamp: object) -> str:
    return pd.Timestamp(timestamp).date().isoformat()  # type: ignore[arg-type]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f} pts"


def _plain(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _range(interval: Sequence[float | None], *, points: bool = False) -> str:
    low, high = interval
    if low is None or high is None:
        return "[n/a]"
    if points:
        return f"[{100 * low:+.1f}, {100 * high:+.1f}]"
    return f"[{100 * low:.1f}%, {100 * high:.1f}%]"


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
        prog="python -m markets_lab.analysis.nvda.backtest",
        description="Walk-forward backtest of the baselines and models; writes aggregates only.",
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--context", type=Path, default=market_context.DEFAULT_CSV_PATH)
    parser.add_argument("--results", type=Path, default=nvda_earnings.DEFAULT_JSON_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--resamples", type=int, default=BootstrapConfig().n_resamples)
    args = parser.parse_args(argv)
    moment = now()
    for path, what in (
        (args.prices, "price"),
        (args.context, "market context"),
        (args.results, "results dates"),
    ):
        if not path.exists():
            report(f"error: {path} not found; run the {what} fetch first")
            return 2
    try:
        prices = load_csv(args.prices)
        validate(prices, now=moment, min_rows=min_rows)
        context = market_context.load_csv(args.context)
        market_context.validate(context, now=moment, min_rows=min_rows)
        publications = nvda_earnings.load_json(args.results)
        nvda_earnings.validate(publications, now=moment)
        dataset = build_context_dataset(prices, context, publications, as_of=moment)
    except (PriceDataError, nvda_earnings.EarningsDataError, CallTooEarlyError) as exc:
        report(f"error: {exc}")
        return 1
    result = run_backtest(
        dataset,
        context_models=CONTEXT_MODEL_FACTORIES,
        bootstrap=BootstrapConfig(n_resamples=args.resamples),
    )
    write_report(result, args.out)
    report(format_report(result))
    report(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
