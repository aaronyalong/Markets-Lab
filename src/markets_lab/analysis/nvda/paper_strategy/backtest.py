"""Backtest the paper strategy against buy-and-hold and write the aggregate report.

The six rule models make their calls through the same walk-forward folds as in the
direction backtest, on the same sessions. For each fold the volatility gate's
threshold comes from that fold's training rows only. The scoring was fixed before
any result was seen (see tests/test_nvda_paper_strategy.py):

- primary measure: the Sharpe ratio after 5 basis points per trade, compared with
  buy-and-hold by a paired moving-block bootstrap;
- also reported: total return, annualised return, worst drawdown, time in the
  market and trades;
- the two parts of the rule (the majority alone, the gate alone) are scored too, to
  explain the result. They are not candidates.

The report holds aggregates only. Run it from the repository root after the three
fetches (prices, market context, results dates):

    uv run python -m markets_lab.analysis.nvda.paper_strategy.backtest
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.backtest import (
    CONTEXT_MODEL_FACTORIES,
    COST_LEVELS_BPS,
    INPUT_ERRORS,
    MODEL_FACTORIES,
    PRIMARY_COST_BPS,
    PredictorFactory,
    load_inputs,
    missing_input,
    write_report,
)
from markets_lab.analysis.nvda.context_features import EXTENDED_FEATURE_COLUMNS
from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, Dataset
from markets_lab.analysis.nvda.metrics import (
    BootstrapConfig,
    annualised_return,
    annualised_volatility,
    bootstrap_interval,
    max_drawdown,
    sharpe_ratio,
    strategy_net_returns,
    total_return,
    trade_count,
)
from markets_lab.analysis.nvda.paper_strategy.rule import (
    GATE_FORECASTER,
    GATE_PERCENTILE,
    HOLDING,
    IN_CASH,
    MIN_MODELS_UP,
    RULE_MODELS,
    gate_series,
    gate_threshold,
    majority_series,
)
from markets_lab.analysis.nvda.predictors import PREDICTED_UP
from markets_lab.analysis.nvda.validation import (
    Fold,
    WalkForwardConfig,
    make_folds,
    walk_forward,
)
from markets_lab.analysis.nvda.volatility.dataset import compute_ewma_volatility
from markets_lab.collectors import market_context, nvda_earnings
from markets_lab.collectors.nvda_prices import (
    DEFAULT_CSV_PATH,
    MARKET_TZ,
    MIN_ROWS,
    SYMBOL,
)

DEFAULT_REPORT_PATH = Path("reports/nvda_paper_strategy.json")
PRIMARY_MEASURE = "sharpe_ratio"
STRATEGY = "paper_strategy"
BENCHMARK = "buy_and_hold"
PARTS = ("majority_only", "gate_only")
"""Parts of the rule, scored to explain the result; not candidates."""

ALSO_REPORTED = (
    "total return",
    "annualised return",
    "worst drawdown",
    "time in the market",
    "trades",
)

CAVEATS = (
    "Research only, not financial advice. A paper strategy: simulated, no money involved.",
    "The volatility gate was chosen knowing that volatility is predictable (the one "
    "positive result of the volatility backtest), so this backtest flatters it. The live "
    "log is the real test.",
    "None of the six models beat always calling up in the direction backtest.",
    "Simulated returns assume a flat cost per trade, no slippage or taxes, and cash that "
    "earns no interest; the Sharpe ratio therefore uses a zero risk-free rate.",
    "The two parts of the rule are compared with buy-and-hold as well, so some intervals "
    "may exclude zero by chance.",
    "Worst drawdown and total return depend on the order of the days and have no interval.",
)


def run_paper_backtest(
    dataset: Dataset,
    volatility: pd.Series,
    *,
    models: Mapping[str, PredictorFactory] | None = None,
    context_models: Mapping[str, PredictorFactory] | None = None,
    config: WalkForwardConfig | None = None,
    bootstrap: BootstrapConfig | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Backtest the paper strategy and return the aggregate report.

    `dataset` must carry the context columns (see `build_context_dataset`);
    `models` are given the technical columns and `context_models` all of them.
    `volatility` is the ewma_94 forecast for every session of the dataset.
    """
    models = MODEL_FACTORIES if models is None else models
    context_models = CONTEXT_MODEL_FACTORIES if context_models is None else context_models
    config = config or WalkForwardConfig()
    bootstrap = bootstrap or BootstrapConfig()
    names = [*models, *context_models]
    missing = [name for name in RULE_MODELS if name not in names]
    extra = [name for name in names if name not in RULE_MODELS]
    if missing or extra or len(names) != len(set(names)):
        raise ValueError(
            f"the rule needs exactly {list(RULE_MODELS)}; missing {missing}, extra {extra}"
        )

    labelled = dataset.labelled()
    sessions = labelled.features.index
    history = volatility.reindex(sessions)
    if bool(history.isna().any()):
        raise ValueError("the volatility forecast is missing for some sessions")
    folds = make_folds(len(labelled), config)

    calls: dict[str, pd.Series] = {}
    for group, columns in ((models, FEATURE_COLUMNS), (context_models, EXTENDED_FEATURE_COLUMNS)):
        inputs = _with_columns(labelled, columns)
        for name, factory in group.items():
            calls[name] = walk_forward(inputs, factory, config, folds=folds)[PREDICTED_UP]
    tested = next(iter(calls.values())).index
    if not all(series.index.equals(tested) for series in calls.values()):
        raise ValueError("the models were not tested on the same sessions")

    majority = majority_series({name: s.to_numpy(dtype=bool) for name, s in calls.items()})
    gate = gate_series(history.loc[tested].to_numpy(), gate_thresholds(history, folds))
    positions = {
        STRATEGY: majority & gate,
        BENCHMARK: np.ones(len(tested), dtype=bool),
        PARTS[0]: majority,
        PARTS[1]: gate,
    }
    next_return = labelled.next_return.loc[tested].to_numpy(dtype="float64")
    strategies = {name: summarise(up, next_return, bootstrap) for name, up in positions.items()}
    benchmark_net = strategy_net_returns(
        positions[BENCHMARK], next_return, cost_bps=PRIMARY_COST_BPS
    )
    comparisons = []
    for name in (STRATEGY, *PARTS):
        result = compare(
            strategy_net_returns(positions[name], next_return, cost_bps=PRIMARY_COST_BPS),
            benchmark_net,
            bootstrap,
        )
        comparisons.append(
            {
                "strategy": name,
                "reference": BENCHMARK,
                "cost_bps": PRIMARY_COST_BPS,
                **result,
                "verdict": verdict(result["sharpe_difference_interval"]),
            }
        )

    report: dict[str, Any] = {
        "generated_at": (generated_at or datetime.now(tz=UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "data": {
            "tested_sessions": len(tested),
            "first_tested_session": _date(tested[0]),
            "last_tested_session": _date(tested[-1]),
            "share_majority_up": _number(float(majority.mean())),
            "share_gate_open": _number(float(gate.mean())),
        },
        "validation": {
            "scheme": "walk-forward, expanding window",
            "initial_train_sessions": config.initial_train_sessions,
            "refit_every_sessions": config.refit_every,
            "folds": len(folds),
        },
        "rule": {
            "states": [HOLDING, IN_CASH],
            "models": list(RULE_MODELS),
            "min_models_up": MIN_MODELS_UP,
            "gate_forecaster": GATE_FORECASTER,
            "gate_percentile": GATE_PERCENTILE,
            "gate_history": "the training rows of each fold",
            "gate_chosen_knowing_volatility_is_predictable": True,
        },
        "scoring": {
            "primary_measure": PRIMARY_MEASURE,
            "benchmark": BENCHMARK,
            "cost_bps": PRIMARY_COST_BPS,
            "interval": "paired moving-block bootstrap",
            "confidence": bootstrap.confidence,
            "block_length_sessions": bootstrap.block_length,
            "resamples": bootstrap.n_resamples,
            "seed": bootstrap.seed,
            "also_reported": list(ALSO_REPORTED),
        },
        "cost_levels_bps": list(COST_LEVELS_BPS),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "strategies": strategies,
        "comparisons": comparisons,
        "caveats": list(CAVEATS),
    }
    report["finding"] = finding(report)
    return report


def gate_thresholds(history: pd.Series, folds: Sequence[Fold]) -> np.ndarray:
    """The gate's threshold for every tested session, from its fold's training rows."""
    return np.concatenate(
        [
            np.full(
                fold.test_end - fold.test_start,
                gate_threshold(history.iloc[: fold.train_end].to_numpy(dtype="float64")),
            )
            for fold in folds
        ]
    )


def summarise(
    up: np.ndarray, next_return: np.ndarray, bootstrap: BootstrapConfig
) -> dict[str, Any]:
    """Time in the market, trades, and the return figures at every cost level."""
    n = len(up)
    by_cost: dict[str, Any] = {}
    for cost in COST_LEVELS_BPS:
        net = strategy_net_returns(up, next_return, cost_bps=cost)
        by_cost[str(cost)] = {
            "total_return": _number(total_return(net)),
            "annualised_return": _number(annualised_return(net)),
            "annualised_return_interval": _pair(
                bootstrap_interval(n, lambda rows, net=net: annualised_return(net[rows]), bootstrap)
            ),
            "annualised_volatility": _number(annualised_volatility(net)),
            "sharpe_ratio": _number(sharpe_ratio(net)),
            "sharpe_ratio_interval": _pair(
                bootstrap_interval(n, lambda rows, net=net: sharpe_ratio(net[rows]), bootstrap)
            ),
            "worst_drawdown": _number(max_drawdown(net)),
        }
    return {
        "time_in_market": _number(float(np.mean(up))),
        "trades": trade_count(up),
        "by_cost_bps": by_cost,
    }


def compare(ours: np.ndarray, theirs: np.ndarray, bootstrap: BootstrapConfig) -> dict[str, Any]:
    """Sharpe ratio and annualised return differences, ours minus theirs, paired."""
    n = len(ours)

    def sharpe_gap(rows: np.ndarray) -> float:
        return sharpe_ratio(ours[rows]) - sharpe_ratio(theirs[rows])

    def return_gap(rows: np.ndarray) -> float:
        return annualised_return(ours[rows]) - annualised_return(theirs[rows])

    everything = np.arange(n)
    return {
        "sharpe_difference": _number(sharpe_gap(everything)),
        "sharpe_difference_interval": _pair(bootstrap_interval(n, sharpe_gap, bootstrap)),
        "annualised_return_difference": _number(return_gap(everything)),
        "annualised_return_difference_interval": _pair(
            bootstrap_interval(n, return_gap, bootstrap)
        ),
    }


def verdict(interval: Sequence[float | None]) -> str:
    """'better' or 'worse' only when the whole interval is above or below zero."""
    low, high = interval
    if low is not None and low > 0:
        return "better"
    if high is not None and high < 0:
        return "worse"
    return "no detectable difference"


def finding(report: Mapping[str, Any]) -> str:
    """The primary result in one sentence, worded from the numbers."""
    cost = report["primary_cost_bps"]
    ours = report["strategies"][STRATEGY]["by_cost_bps"][str(cost)]["sharpe_ratio"]
    theirs = report["strategies"][BENCHMARK]["by_cost_bps"][str(cost)]["sharpe_ratio"]
    row = next(r for r in report["comparisons"] if r["strategy"] == STRATEGY)
    low, high = row["sharpe_difference_interval"]
    return (
        f"Risk-adjusted return after {cost} basis points per trade: Sharpe ratio "
        f"{_plain(ours)} for the paper strategy against {_plain(theirs)} for buy-and-hold, "
        f"a difference of {_signed(row['sharpe_difference'])} "
        f"({_signed(low)} to {_signed(high)}): {row['verdict']}."
    )


def format_report(report: Mapping[str, Any]) -> str:
    """A plain-text summary for the terminal."""
    data, rule = report["data"], report["rule"]
    cost = str(report["primary_cost_bps"])
    lines = [
        f"Paper strategy: {rule['states'][0]} when at least {rule['min_models_up']} of "
        f"{len(rule['models'])} models say up and the {rule['gate_forecaster']} forecast is "
        f"not above its {rule['gate_percentile']:.0f}th percentile; otherwise "
        f"{rule['states'][1]}.",
        f"{data['tested_sessions']:,} sessions, {data['first_tested_session']} to "
        f"{data['last_tested_session']}; majority up on {_percent(data['share_majority_up'])}, "
        f"gate open on {_percent(data['share_gate_open'])}. Costs {cost} bps.",
        "",
        f"{'':16}{'in market':>10}{'trades':>8}{'Sharpe':>8}  {'interval':<16}"
        f"{'per year':>9}  {'interval':<18}{'total':>10}{'worst fall':>11}",
    ]
    for name, row in report["strategies"].items():
        money = row["by_cost_bps"][cost]
        lines.append(
            f"{name:16}{_percent(row['time_in_market']):>10}{row['trades']:>8}"
            f"{_plain(money['sharpe_ratio']):>8}  {_pair_text(money['sharpe_ratio_interval']):<16}"
            f"{_percent(money['annualised_return']):>9}  "
            f"{_pair_text(money['annualised_return_interval'], percent=True):<18}"
            f"{_percent(money['total_return']):>10}{_percent(money['worst_drawdown']):>11}"
        )
    lines += ["", "Against buy-and-hold:"]
    for row in report["comparisons"]:
        lines.append(
            f"  {row['strategy']:16} Sharpe {_signed(row['sharpe_difference'])} "
            f"{_pair_text(row['sharpe_difference_interval'])}, per year "
            f"{_points(row['annualised_return_difference'])} "
            f"{_pair_text(row['annualised_return_difference_interval'], points=True)}: "
            f"{row['verdict']}"
        )
    lines += ["", report["finding"]]
    return "\n".join(lines)


def _with_columns(dataset: Dataset, columns: Sequence[str]) -> Dataset:
    missing = [name for name in columns if name not in dataset.features.columns]
    if missing:
        raise ValueError(f"the dataset has no columns {missing[:5]}")
    return Dataset(dataset.features[list(columns)], dataset.target_up, dataset.next_return)


def _number(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else round(value, 6)


def _pair(interval: tuple[float, float]) -> list[float | None]:
    return [_number(interval[0]), _number(interval[1])]


def _date(timestamp: object) -> str:
    return pd.Timestamp(timestamp).date().isoformat()  # type: ignore[arg-type]


def _plain(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f} pts"


def _pair_text(
    interval: Sequence[float | None], *, percent: bool = False, points: bool = False
) -> str:
    low, high = interval
    if low is None or high is None:
        return "[n/a]"
    if percent:
        return f"[{100 * low:.1f}%, {100 * high:.1f}%]"
    if points:
        return f"[{100 * low:+.1f}, {100 * high:+.1f}]"
    return f"[{low:+.2f}, {high:+.2f}]"


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(tz=MARKET_TZ),
    report: Callable[[str], None] = _print,
    min_rows: int = MIN_ROWS,
    config: WalkForwardConfig | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.analysis.nvda.paper_strategy.backtest",
        description="Walk-forward backtest of the paper strategy; writes aggregates only.",
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--context", type=Path, default=market_context.DEFAULT_CSV_PATH)
    parser.add_argument("--results", type=Path, default=nvda_earnings.DEFAULT_JSON_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--resamples", type=int, default=BootstrapConfig().n_resamples)
    args = parser.parse_args(argv)
    moment = now()
    missing = missing_input(args.prices, args.context, args.results)
    if missing:
        report(f"error: {missing}")
        return 2
    try:
        prices, dataset = load_inputs(
            args.prices, args.context, args.results, moment=moment, min_rows=min_rows
        )
    except INPUT_ERRORS as exc:
        report(f"error: {exc}")
        return 1
    result = run_paper_backtest(
        dataset,
        compute_ewma_volatility(prices),
        models=MODEL_FACTORIES,
        context_models=CONTEXT_MODEL_FACTORIES,
        config=config,
        bootstrap=BootstrapConfig(n_resamples=args.resamples),
    )
    write_report(result, args.out)
    report(format_report(result))
    report(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
