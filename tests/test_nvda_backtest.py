"""The backtest report: correct aggregates, and nothing in it but aggregates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.backtest import (
    BASELINE_FACTORIES,
    COST_LEVELS_BPS,
    MODEL_FACTORIES,
    format_report,
    main,
    run_backtest,
    write_report,
)
from markets_lab.analysis.nvda.features import Dataset, build_dataset
from markets_lab.analysis.nvda.metrics import BootstrapConfig
from markets_lab.analysis.nvda.validation import WalkForwardConfig
from markets_lab.collectors.nvda_prices import save_csv
from nvda_test_predictors import LeastSquaresLearner
from price_frames import at, make_prices

ROWS = 900
CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=50)
QUICK = BootstrapConfig(n_resamples=200)
WHEN = datetime(2030, 3, 15, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def dataset(prices: pd.DataFrame) -> Dataset:
    return build_dataset(prices)


@pytest.fixture(scope="module")
def report(dataset: Dataset) -> dict[str, Any]:
    return run_backtest(
        dataset,
        models={"least_squares": LeastSquaresLearner},
        config=CONFIG,
        bootstrap=QUICK,
        generated_at=WHEN,
    )


class TestReportContents:
    def test_describes_the_data_and_the_validation(
        self, report: dict[str, Any], dataset: Dataset
    ) -> None:
        tested = len(dataset.labelled()) - 300
        assert report["generated_at"] == "2030-03-15T12:00:00Z"
        assert report["symbol"] == "NVDA"
        assert report["data"]["tested_sessions"] == tested
        assert report["data"]["last_tested_session"] == "2030-03-13"
        assert report["validation"] == {
            "scheme": "walk-forward, expanding window",
            "initial_train_sessions": 300,
            "refit_every_sessions": 50,
            "folds": -(-tested // 50),
        }
        assert report["cost_levels_bps"] == [0, 5, 10]
        assert report["primary_cost_bps"] == 5

    def test_every_predictor_is_scored_on_the_same_sessions(self, report: dict[str, Any]) -> None:
        names = list(report["predictors"])
        assert names == ["always_up", "persistence", "ma_200", "least_squares"]
        assert {row["sessions"] for row in report["predictors"].values()} == {
            report["data"]["tested_sessions"]
        }
        kinds = [row["kind"] for row in report["predictors"].values()]
        assert kinds == ["baseline", "baseline", "baseline", "model"]

    def test_always_up_accuracy_is_the_share_of_up_days(self, report: dict[str, Any]) -> None:
        always = report["predictors"]["always_up"]
        assert always["accuracy"] == report["data"]["share_of_up_days_tested"]
        assert always["share_called_up"] == 1.0
        assert always["trades"] == 1
        assert always["hits"] == round(always["accuracy"] * always["sessions"])

    def test_always_up_return_is_buy_and_hold(
        self, report: dict[str, Any], dataset: Dataset
    ) -> None:
        labelled = dataset.labelled()
        held = labelled.next_return.iloc[300:].to_numpy()
        buy_and_hold = float(np.prod(1 + held) - 1)
        returns = report["predictors"]["always_up"]["returns_by_cost_bps"]
        assert returns["0"]["total_return"] == pytest.approx(buy_and_hold, abs=1e-6)
        assert returns["5"]["total_return"] == pytest.approx(
            (1 - 0.0005) * (1 + buy_and_hold) - 1, abs=1e-6
        )

    def test_intervals_contain_their_estimates(self, report: dict[str, Any]) -> None:
        for row in report["predictors"].values():
            low, high = row["accuracy_interval"]
            assert low < row["accuracy"] < high
            for cost in COST_LEVELS_BPS:
                returns = row["returns_by_cost_bps"][str(cost)]
                low, high = returns["annualised_return_interval"]
                assert low <= returns["annualised_return"] <= high

    def test_costs_only_reduce_returns(self, report: dict[str, Any]) -> None:
        for row in report["predictors"].values():
            totals = [row["returns_by_cost_bps"][str(c)]["total_return"] for c in COST_LEVELS_BPS]
            assert totals[0] >= totals[1] >= totals[2]

    def test_comparisons_cover_every_model_against_every_baseline(
        self, report: dict[str, Any]
    ) -> None:
        pairs = [(row["predictor"], row["baseline"]) for row in report["comparisons"]]
        assert pairs == [
            ("persistence", "always_up"),
            ("ma_200", "always_up"),
            ("ma_200", "persistence"),
            ("least_squares", "always_up"),
            ("least_squares", "persistence"),
            ("least_squares", "ma_200"),
        ]

    def test_differences_match_the_predictor_rows(self, report: dict[str, Any]) -> None:
        predictors = report["predictors"]
        for row in report["comparisons"]:
            ours, theirs = predictors[row["predictor"]], predictors[row["baseline"]]
            assert row["accuracy_difference"] == pytest.approx(
                ours["accuracy"] - theirs["accuracy"], abs=2e-6
            )
            assert row["annualised_return_difference"] == pytest.approx(
                ours["returns_by_cost_bps"]["5"]["annualised_return"]
                - theirs["returns_by_cost_bps"]["5"]["annualised_return"],
                abs=2e-6,
            )
            low, high = row["accuracy_difference_interval"]
            assert low <= row["accuracy_difference"] <= high

    def test_same_inputs_give_the_same_report(
        self, report: dict[str, Any], dataset: Dataset
    ) -> None:
        again = run_backtest(
            dataset,
            models={"least_squares": LeastSquaresLearner},
            config=CONFIG,
            bootstrap=QUICK,
            generated_at=WHEN,
        )
        assert again == report

    def test_a_name_cannot_be_both_baseline_and_model(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="both a baseline and a model"):
            run_backtest(dataset, models={"always_up": LeastSquaresLearner}, config=CONFIG)

    def test_no_models_are_registered_before_they_are_measured(self) -> None:
        assert list(BASELINE_FACTORIES) == ["always_up", "persistence", "ma_200"]
        assert isinstance(MODEL_FACTORIES, dict)


def walk(value: object, path: str = "") -> list[tuple[str, object]]:
    """Every (path, leaf) in a nested JSON-like structure."""
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in walk(v, f"{path}/{k}")]
    if isinstance(value, list):
        return [(f"{path}[]", value)] + [leaf for v in value for leaf in walk(v, f"{path}[]")]
    return [(path, value)]


class TestReportHoldsAggregatesOnly:
    """The report is committed and published, so it must not carry raw data."""

    def test_it_is_plain_json(self, report: dict[str, Any]) -> None:
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_no_per_session_series(self, report: dict[str, Any]) -> None:
        # The longest list of numbers allowed is an interval or the cost levels.
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

    def test_field_names_do_not_suggest_raw_data(self, report: dict[str, Any]) -> None:
        text = json.dumps(report).lower()
        for word in ('"close"', '"open"', '"price"', '"prices"', '"daily_returns"', '"volume"'):
            assert word not in text


class TestOutput:
    def test_text_summary_names_every_predictor_and_comparison(
        self, report: dict[str, Any]
    ) -> None:
        text = format_report(report)
        for name in report["predictors"]:
            assert f"\n{name}" in text
        assert "least_squares vs always_up: accuracy" in text
        assert "95% interval" in text

    def test_write_report_replaces_in_one_step(
        self, report: dict[str, Any], tmp_path: Path
    ) -> None:
        path = tmp_path / "reports" / "nvda_backtest.json"
        write_report(report, path)
        write_report(report, path)
        assert json.loads(path.read_text()) == report
        assert [p.name for p in path.parent.iterdir()] == ["nvda_backtest.json"]
        assert path.read_text().endswith("}\n")


class TestCommandLine:
    def test_runs_from_a_price_file_and_writes_the_report(self, tmp_path: Path) -> None:
        prices_path, out = tmp_path / "prices.csv", tmp_path / "out" / "report.json"
        save_csv(make_prices(rows=1_700), prices_path)
        lines: list[str] = []
        code = main(
            ["--prices", str(prices_path), "--out", str(out), "--resamples", "50"],
            now=lambda: at("2030-03-14 20:00"),
            report=lines.append,
            min_rows=1_000,
        )
        assert code == 0
        written = json.loads(out.read_text())
        assert list(written["predictors"]) == ["always_up", "persistence", "ma_200"]
        assert written["validation"]["initial_train_sessions"] == 1_260
        assert written["intervals"]["resamples"] == 50
        assert lines[-1] == f"\nwrote {out}"

    def test_missing_price_file(self, tmp_path: Path) -> None:
        lines: list[str] = []
        code = main(["--prices", str(tmp_path / "nope.csv")], report=lines.append)
        assert code == 2
        assert "run the price fetch first" in lines[0]

    def test_stale_or_invalid_prices_are_refused(self, tmp_path: Path) -> None:
        prices_path, out = tmp_path / "prices.csv", tmp_path / "report.json"
        save_csv(make_prices(rows=1_700), prices_path)
        lines: list[str] = []
        code = main(
            ["--prices", str(prices_path), "--out", str(out)],
            now=lambda: at("2030-04-20 20:00"),
            report=lines.append,
            min_rows=1_000,
        )
        assert code == 1
        assert "stale" in lines[0]
        assert not out.exists()
