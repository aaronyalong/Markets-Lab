"""The paper strategy: its rule and its scoring, written down before any result was seen.

This file is the record of what was fixed in advance (2026-11-25):

THE RULE
- Each session, after the close, the paper strategy is either "Holding" NVDA for
  the next session or "In cash".
- It is Holding only when BOTH hold:
  1. at least 4 of the 6 direction models say up: logistic, xgboost, lstm and
     their "+ context" twins (no baselines; a model added later does not join);
  2. the volatility gate is open: the simple baseline's (ewma_94) volatility
     forecast for the next session is not above the 80th percentile of its values
     on the training rows (the rows the models were fitted on for that call).
- The gate was chosen knowing that volatility is predictable (the one positive
  result so far). The page must say so.
- Nothing here is tuned: 4 of 6 and the 80th percentile are fixed.

THE SCORING
- Positions and costs as in the direction backtest: hold on Holding days, cash
  (earning nothing) otherwise, 5 basis points each time the position changes; also
  shown at 0 and 10.
- Benchmark: buy-and-hold (Holding every session).
- Primary measure: risk-adjusted return, the Sharpe ratio (annualised mean over
  standard deviation of daily net returns, cash earning zero), compared with
  buy-and-hold by a paired moving-block bootstrap, 95% interval.
- "Better" or "worse" only if the whole interval is above or below zero;
  otherwise "no detectable difference".
- Also reported: total return, annualised return (with interval and difference),
  worst drawdown and time in the market. Drawdown and total return are path
  statistics and get no interval.
- Same sessions and walk-forward folds as the direction backtest.
- Two parts of the rule are also scored, to explain the result and not as
  candidates: the majority alone and the gate alone.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from context_frames import make_context, make_publications
from markets_lab.analysis.nvda.backtest import CONTEXT_MODEL_FACTORIES, MODEL_FACTORIES
from markets_lab.analysis.nvda.baselines import AlwaysUp, MovingAverageRule, Persistence
from markets_lab.analysis.nvda.context_features import build_context_dataset
from markets_lab.analysis.nvda.features import Dataset
from markets_lab.analysis.nvda.metrics import (
    BootstrapConfig,
    max_drawdown,
    sharpe_ratio,
    strategy_net_returns,
)
from markets_lab.analysis.nvda.paper_strategy import backtest as paper
from markets_lab.analysis.nvda.paper_strategy.rule import (
    GATE_FORECASTER,
    GATE_PERCENTILE,
    HOLDING,
    IN_CASH,
    MIN_MODELS_UP,
    RULE_MODELS,
    gate_open,
    gate_threshold,
    holding,
    holding_series,
    state_name,
)
from markets_lab.analysis.nvda.validation import WalkForwardConfig, make_folds
from markets_lab.analysis.nvda.volatility.dataset import compute_ewma_volatility
from markets_lab.collectors.nvda_earnings import save_json
from markets_lab.collectors.nvda_prices import save_csv
from nvda_test_predictors import LeastSquaresLearner
from price_frames import at, make_prices

ROWS = 900
CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=50)
QUICK = BootstrapConfig(n_resamples=200)
WHEN = datetime(2030, 3, 15, 12, 0, tzinfo=UTC)

# Quick stand-ins for the six models: different enough to disagree.
STAND_INS = {
    "logistic": LeastSquaresLearner,
    "xgboost": Persistence,
    "lstm": AlwaysUp,
}
CONTEXT_STAND_INS = {
    "logistic_ctx": LeastSquaresLearner,
    "xgboost_ctx": MovingAverageRule,
    "lstm_ctx": Persistence,
}

ALL_UP = dict.fromkeys(RULE_MODELS, True)


def with_up(count: int) -> dict[str, bool]:
    """The first `count` rule models say up, the rest down."""
    return {name: i < count for i, name in enumerate(RULE_MODELS)}


# ---------------------------------------------------------------- the rule


class TestTheRuleAsFixedInAdvance:
    def test_the_six_models_four_of_six_and_the_80th_percentile_gate(self) -> None:
        assert RULE_MODELS == (
            "logistic",
            "xgboost",
            "lstm",
            "logistic_ctx",
            "xgboost_ctx",
            "lstm_ctx",
        )
        assert MIN_MODELS_UP == 4
        assert GATE_FORECASTER == "ewma_94"
        assert GATE_PERCENTILE == 80.0

    def test_the_rule_models_are_the_registered_models(self) -> None:
        assert set(RULE_MODELS) == set(MODEL_FACTORIES) | set(CONTEXT_MODEL_FACTORIES)

    def test_the_two_states_are_named_holding_and_in_cash(self) -> None:
        assert (HOLDING, IN_CASH) == ("Holding", "In cash")
        assert state_name(True) == "Holding" and state_name(False) == "In cash"

    @pytest.mark.parametrize(("up", "expected"), [(6, True), (5, True), (4, True), (3, False)])
    def test_holding_needs_four_of_six_up_when_the_gate_is_open(
        self, up: int, expected: bool
    ) -> None:
        assert holding(with_up(up), forecast=0.02, threshold=0.03) is expected

    def test_a_closed_gate_means_cash_however_many_say_up(self) -> None:
        assert holding(ALL_UP, forecast=0.031, threshold=0.03) is False

    def test_a_forecast_exactly_at_the_threshold_leaves_the_gate_open(self) -> None:
        assert gate_open(0.03, 0.03) is True
        assert gate_open(0.0300001, 0.03) is False
        assert holding(with_up(4), forecast=0.03, threshold=0.03) is True

    def test_every_one_of_the_six_must_have_called(self) -> None:
        five = {name: True for name in RULE_MODELS[:5]}
        with pytest.raises(ValueError, match="lstm_ctx"):
            holding(five, forecast=0.02, threshold=0.03)

    def test_calls_from_anything_else_are_ignored(self) -> None:
        crowd = {**with_up(3), "always_up": True, "ma_200": True, "new_model": True}
        assert holding(crowd, forecast=0.02, threshold=0.03) is False

    def test_the_threshold_is_the_80th_percentile_of_the_history(self) -> None:
        history = np.arange(1.0, 101.0) / 1000  # 0.001 .. 0.100
        assert gate_threshold(history) == pytest.approx(np.percentile(history, 80))
        assert gate_threshold(np.array([0.01, 0.02, 0.03, 0.04, 0.05])) == pytest.approx(0.042)

    @pytest.mark.parametrize("history", [[], [0.01, np.nan], [0.01, -0.02], [0.0, np.inf]])
    def test_the_threshold_refuses_a_bad_history(self, history: list[float]) -> None:
        with pytest.raises(ValueError):
            gate_threshold(np.array(history, dtype="float64"))

    def test_the_vector_form_agrees_with_the_single_call(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        calls = {name: rng.random(n) > 0.5 for name in RULE_MODELS}
        forecast = rng.uniform(0.01, 0.05, n)
        threshold = rng.uniform(0.01, 0.05, n)
        vector = holding_series(calls, forecast, threshold)
        single = [
            holding({k: bool(v[i]) for k, v in calls.items()}, float(forecast[i]), threshold[i])
            for i in range(n)
        ]
        assert vector.dtype == np.bool_ and list(vector) == single


# ---------------------------------------------------------------- scoring


class TestScoringAsFixedInAdvance:
    def test_worst_drawdown_starts_from_the_first_value(self) -> None:
        # Values 1.1, 0.55, 0.66: the worst fall is from 1.1 to 0.55.
        assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
        # A loss on the first day counts: the starting value is a peak.
        assert max_drawdown(np.array([-0.1, 0.05])) == pytest.approx(-0.1)
        assert max_drawdown(np.array([0.01, 0.02, 0.0])) == 0.0

    def test_worst_drawdown_is_the_largest_of_several_falls(self) -> None:
        # 1 -> 0.9 (-10%), up to 1.08, down to 0.864 (-20%), then recovers.
        returns = np.array([-0.1, 0.2, -0.2, 0.5])
        assert max_drawdown(returns) == pytest.approx(-0.2)

    def test_the_primary_measure_and_the_benchmark(self) -> None:
        assert paper.PRIMARY_MEASURE == "sharpe_ratio"
        assert paper.PRIMARY_COST_BPS == 5
        assert paper.COST_LEVELS_BPS == (0, 5, 10)
        assert paper.BENCHMARK == "buy_and_hold"
        assert paper.STRATEGY == "paper_strategy"
        assert paper.PARTS == ("majority_only", "gate_only")

    @pytest.mark.parametrize(
        ("interval", "verdict"),
        [
            ((0.01, 0.4), "better"),
            ((-0.4, -0.01), "worse"),
            ((-0.1, 0.2), "no detectable difference"),
            ((0.0, 0.2), "no detectable difference"),
        ],
    )
    def test_better_or_worse_only_when_the_whole_interval_says_so(
        self, interval: tuple[float, float], verdict: str
    ) -> None:
        assert paper.verdict(interval) == verdict

    def test_a_strategy_against_itself_differs_by_exactly_nothing(self) -> None:
        rng = np.random.default_rng(1)
        net = rng.normal(0.001, 0.03, 500)
        result = paper.compare(net, net, QUICK)
        assert result["sharpe_difference"] == 0.0
        assert result["sharpe_difference_interval"] == [0.0, 0.0]
        assert result["annualised_return_difference_interval"] == [0.0, 0.0]

    def test_the_comparison_is_paired(self) -> None:
        # Doubling every daily return leaves the Sharpe ratio unchanged on any set
        # of days, so a paired comparison finds no difference on every resample. An
        # unpaired one (different days for each side) would spread widely.
        rng = np.random.default_rng(2)
        market = rng.normal(0.001, 0.03, 2000)
        result = paper.compare(2 * market, market, QUICK)
        low, high = result["sharpe_difference_interval"]
        assert low == pytest.approx(0.0, abs=1e-9) and high == pytest.approx(0.0, abs=1e-9)

    def test_summary_of_one_strategy(self) -> None:
        rng = np.random.default_rng(3)
        next_return = rng.normal(0.001, 0.03, 600)
        up = rng.random(600) > 0.3
        summary = paper.summarise(up, next_return, QUICK)
        assert summary["time_in_market"] == pytest.approx(up.mean())
        five = summary["by_cost_bps"]["5"]
        net = strategy_net_returns(up, next_return, cost_bps=5)
        assert five["sharpe_ratio"] == pytest.approx(sharpe_ratio(net))
        assert five["worst_drawdown"] == pytest.approx(max_drawdown(net))
        for key in (
            "total_return",
            "annualised_return",
            "annualised_return_interval",
            "annualised_volatility",
            "sharpe_ratio",
            "sharpe_ratio_interval",
            "worst_drawdown",
        ):
            assert key in five
        assert set(summary["by_cost_bps"]) == {"0", "5", "10"}


# ---------------------------------------------------------------- the backtest


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def extended(prices: pd.DataFrame) -> Dataset:
    return build_context_dataset(
        prices, make_context(ROWS), make_publications(), as_of=at("2030-03-14 22:30")
    )


@pytest.fixture(scope="module")
def ewma(prices: pd.DataFrame) -> pd.Series:
    return compute_ewma_volatility(prices)


@pytest.fixture(scope="module")
def report(extended: Dataset, ewma: pd.Series) -> dict[str, Any]:
    return paper.run_paper_backtest(
        extended,
        ewma,
        models=STAND_INS,
        context_models=CONTEXT_STAND_INS,
        config=CONFIG,
        bootstrap=QUICK,
        generated_at=WHEN,
    )


def walk(value: object, path: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in walk(v, f"{path}/{k}")]
    if isinstance(value, list):
        return [(f"{path}[]", value)] + [leaf for v in value for leaf in walk(v, f"{path}[]")]
    return [(path, value)]


class TestGateThresholds:
    def test_each_fold_uses_only_its_training_rows(
        self, extended: Dataset, ewma: pd.Series
    ) -> None:
        labelled = extended.labelled()
        folds = make_folds(len(labelled), CONFIG)
        history = ewma.loc[labelled.features.index]
        thresholds = paper.gate_thresholds(history, folds)
        assert len(thresholds) == len(labelled) - CONFIG.initial_train_sessions
        position = 0
        for fold in folds:
            expected = np.percentile(history.iloc[: fold.train_end], GATE_PERCENTILE)
            width = fold.test_end - fold.test_start
            assert np.allclose(thresholds[position : position + width], expected)
            position += width

    def test_later_values_cannot_move_an_earlier_threshold(
        self, extended: Dataset, ewma: pd.Series
    ) -> None:
        labelled = extended.labelled()
        folds = make_folds(len(labelled), CONFIG)
        history = ewma.loc[labelled.features.index]
        changed = history.copy()
        cut = folds[3].train_end + 10  # inside fold 3's tested sessions
        changed.iloc[cut:] = changed.iloc[cut:] * 10
        before = paper.gate_thresholds(history, folds)
        after = paper.gate_thresholds(changed, folds)
        # Folds 0 to 3 were trained before the change, so their thresholds stand.
        through_fold_3 = folds[3].test_end - CONFIG.initial_train_sessions
        assert np.array_equal(before[:through_fold_3], after[:through_fold_3])
        assert not np.array_equal(before[through_fold_3:], after[through_fold_3:])


class TestPaperBacktest:
    def test_describes_the_rule_and_the_scoring(self, report: dict[str, Any]) -> None:
        assert report["rule"] == {
            "states": ["Holding", "In cash"],
            "models": list(RULE_MODELS),
            "min_models_up": 4,
            "gate_forecaster": "ewma_94",
            "gate_percentile": 80.0,
            "gate_history": "the training rows of each fold",
            "gate_chosen_knowing_volatility_is_predictable": True,
        }
        assert report["scoring"]["primary_measure"] == "sharpe_ratio"
        assert report["scoring"]["benchmark"] == "buy_and_hold"
        assert report["scoring"]["cost_bps"] == 5
        assert report["scoring"]["interval"] == "paired moving-block bootstrap"

    def test_tested_on_the_direction_backtests_sessions(
        self, report: dict[str, Any], extended: Dataset
    ) -> None:
        tested = len(extended.labelled()) - CONFIG.initial_train_sessions
        assert report["data"]["tested_sessions"] == tested
        assert report["data"]["last_tested_session"] == "2030-03-13"
        assert report["validation"]["folds"] == -(-tested // CONFIG.refit_every)

    def test_scores_the_strategy_the_benchmark_and_the_two_parts(
        self, report: dict[str, Any]
    ) -> None:
        assert list(report["strategies"]) == [
            "paper_strategy",
            "buy_and_hold",
            "majority_only",
            "gate_only",
        ]
        assert report["strategies"]["buy_and_hold"]["time_in_market"] == 1.0
        comparisons = {row["strategy"]: row for row in report["comparisons"]}
        assert set(comparisons) == {"paper_strategy", "majority_only", "gate_only"}
        for row in comparisons.values():
            assert row["reference"] == "buy_and_hold" and row["cost_bps"] == 5

    def test_the_strategy_is_the_majority_and_the_gate_together(
        self, report: dict[str, Any]
    ) -> None:
        shares = report["data"]
        strategy = report["strategies"]["paper_strategy"]["time_in_market"]
        assert strategy <= min(shares["share_majority_up"], shares["share_gate_open"])
        assert report["strategies"]["majority_only"]["time_in_market"] == pytest.approx(
            shares["share_majority_up"]
        )
        assert report["strategies"]["gate_only"]["time_in_market"] == pytest.approx(
            shares["share_gate_open"]
        )

    def test_the_finding_follows_the_primary_interval(self, report: dict[str, Any]) -> None:
        primary = next(r for r in report["comparisons"] if r["strategy"] == "paper_strategy")
        assert primary["verdict"] == paper.verdict(primary["sharpe_difference_interval"])
        assert report["finding"] == paper.finding(report)
        assert primary["verdict"] in report["finding"]

    def test_the_finding_is_worded_from_the_numbers(self) -> None:
        def made_up(interval: list[float]) -> dict[str, Any]:
            return {
                "primary_cost_bps": 5,
                "strategies": {
                    "paper_strategy": {"by_cost_bps": {"5": {"sharpe_ratio": 0.8}}},
                    "buy_and_hold": {"by_cost_bps": {"5": {"sharpe_ratio": 1.0}}},
                },
                "comparisons": [
                    {
                        "strategy": "paper_strategy",
                        "sharpe_difference": -0.2,
                        "sharpe_difference_interval": interval,
                        "verdict": paper.verdict(interval),
                    }
                ],
            }

        assert paper.finding(made_up([-0.5, 0.1])) == (
            "Risk-adjusted return after 5 basis points per trade: Sharpe ratio 0.80 for the "
            "paper strategy against 1.00 for buy-and-hold, a difference of -0.20 "
            "(-0.50 to +0.10): no detectable difference."
        )
        assert paper.finding(made_up([-0.5, -0.1])).endswith("(-0.50 to -0.10): worse.")

    def test_same_inputs_give_the_same_report(
        self, report: dict[str, Any], extended: Dataset, ewma: pd.Series
    ) -> None:
        again = paper.run_paper_backtest(
            extended,
            ewma,
            models=STAND_INS,
            context_models=CONTEXT_STAND_INS,
            config=CONFIG,
            bootstrap=QUICK,
            generated_at=WHEN,
        )
        assert again == report

    def test_needs_exactly_the_six_rule_models(self, extended: Dataset, ewma: pd.Series) -> None:
        with pytest.raises(ValueError, match="lstm_ctx"):
            paper.run_paper_backtest(
                extended,
                ewma,
                models=STAND_INS,
                context_models={k: v for k, v in CONTEXT_STAND_INS.items() if k != "lstm_ctx"},
                config=CONFIG,
                bootstrap=QUICK,
            )

    def test_the_gate_forecast_must_cover_every_session(
        self, extended: Dataset, ewma: pd.Series
    ) -> None:
        with pytest.raises(ValueError, match="volatility"):
            paper.run_paper_backtest(
                extended,
                ewma.iloc[:-5],
                models=STAND_INS,
                context_models=CONTEXT_STAND_INS,
                config=CONFIG,
                bootstrap=QUICK,
            )

    def test_the_technical_models_see_only_the_technical_columns(
        self, extended: Dataset, ewma: pd.Series
    ) -> None:
        seen: dict[str, int] = {}

        class Counting(LeastSquaresLearner):
            def __init__(self, name: str) -> None:
                super().__init__()
                self._who = name

            def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
                seen[self._who] = features.shape[1]
                super().fit(features, target_up)

        paper.run_paper_backtest(
            extended,
            ewma,
            models={name: (lambda n=name: Counting(n)) for name in STAND_INS},
            context_models={name: (lambda n=name: Counting(n)) for name in CONTEXT_STAND_INS},
            config=CONFIG,
            bootstrap=QUICK,
        )
        assert {seen[name] for name in STAND_INS} == {20}
        assert {seen[name] for name in CONTEXT_STAND_INS} == {35}


class TestWhatIsPublished:
    def test_it_is_plain_json(self, report: dict[str, Any]) -> None:
        assert json.loads(json.dumps(report)) == report

    def test_no_per_session_series(self, report: dict[str, Any]) -> None:
        for path, leaf in walk(report):
            if isinstance(leaf, list) and all(isinstance(x, int | float) for x in leaf):
                assert len(leaf) <= 3, f"{path} holds a series of {len(leaf)} numbers"

    def test_no_price_appears_anywhere(self, report: dict[str, Any], prices: pd.DataFrame) -> None:
        raw = {
            round(float(v), 4)
            for column in ("open", "high", "low", "close", "adj_close")
            for v in prices[column]
        }
        numbers = [leaf for _, leaf in walk(report) if isinstance(leaf, float)]
        assert numbers
        assert not [n for n in numbers if round(n, 4) in raw]


class TestCommandLine:
    def test_runs_from_the_data_files_and_writes_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(paper, "MODEL_FACTORIES", STAND_INS)
        monkeypatch.setattr(paper, "CONTEXT_MODEL_FACTORIES", CONTEXT_STAND_INS)
        prices_path = tmp_path / "prices.csv"
        context_path = tmp_path / "context.csv"
        results_path = tmp_path / "results.json"
        out = tmp_path / "out.json"
        save_csv(make_prices(rows=ROWS), prices_path)
        save_csv(make_context(ROWS), context_path)
        save_json(make_publications(), results_path)
        lines: list[str] = []
        code = paper.main(
            [
                "--prices",
                str(prices_path),
                "--context",
                str(context_path),
                "--results",
                str(results_path),
                "--out",
                str(out),
                "--resamples",
                "50",
            ],
            now=lambda: at("2030-03-15 08:00"),
            report=lines.append,
            min_rows=100,
            config=CONFIG,
        )
        assert code == 0, lines
        written = json.loads(out.read_text())
        assert written["finding"] in "\n".join(lines)

    def test_a_missing_data_file_names_the_fetch_to_run(self, tmp_path: Path) -> None:
        lines: list[str] = []
        code = paper.main(
            ["--prices", str(tmp_path / "missing.csv")],
            now=lambda: at("2030-03-15 08:00"),
            report=lines.append,
        )
        assert code == 2
        assert "price fetch" in lines[0]
