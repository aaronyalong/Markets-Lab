"""The backtest report: correct aggregates, and nothing in it but aggregates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from context_frames import make_context, make_publications
from markets_lab.analysis.nvda.backtest import (
    BASELINE_FACTORIES,
    CONTEXT_MODEL_FACTORIES,
    COST_LEVELS_BPS,
    MODEL_FACTORIES,
    format_report,
    main,
    run_backtest,
    write_report,
)
from markets_lab.analysis.nvda.context_features import build_context_dataset
from markets_lab.analysis.nvda.features import Dataset, build_dataset
from markets_lab.analysis.nvda.metrics import BootstrapConfig
from markets_lab.analysis.nvda.validation import WalkForwardConfig
from markets_lab.collectors.nvda_earnings import save_json
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


@pytest.fixture(scope="module")
def extended(prices: pd.DataFrame) -> Dataset:
    """The same sessions with the market context and results columns added."""
    return build_context_dataset(
        prices, make_context(ROWS), make_publications(), as_of=at("2030-03-14 22:30")
    )


@pytest.fixture(scope="module")
def both(extended: Dataset) -> dict[str, Any]:
    return run_backtest(
        extended,
        models={"least_squares": LeastSquaresLearner},
        context_models={"least_squares_ctx": LeastSquaresLearner},
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

    def test_every_predictor_is_compared_with_each_one_listed_before_it(
        self, report: dict[str, Any]
    ) -> None:
        pairs = [(row["predictor"], row["reference"]) for row in report["comparisons"]]
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
            ours, theirs = predictors[row["predictor"]], predictors[row["reference"]]
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

    def test_registered_baselines_and_models(self) -> None:
        assert list(BASELINE_FACTORIES) == ["always_up", "persistence", "ma_200"]
        assert list(MODEL_FACTORIES) == ["logistic", "xgboost", "lstm"]
        assert list(CONTEXT_MODEL_FACTORIES) == ["logistic_ctx", "xgboost_ctx"]

    def test_a_later_model_is_also_compared_with_earlier_models(self, dataset: Dataset) -> None:
        result = run_backtest(
            dataset,
            models={"first": LeastSquaresLearner, "second": LeastSquaresLearner},
            config=CONFIG,
            bootstrap=QUICK,
        )
        pairs = [(row["predictor"], row["reference"]) for row in result["comparisons"]]
        assert ("second", "first") in pairs
        assert ("first", "second") not in pairs
        same = next(
            r
            for r in result["comparisons"]
            if r["predictor"] == "second" and r["reference"] == "first"
        )
        assert same["accuracy_difference"] == 0.0
        assert same["accuracy_difference_interval"] == [0.0, 0.0]

    def test_probability_quality_is_reported_only_where_there_are_probabilities(
        self, report: dict[str, Any]
    ) -> None:
        predictors = report["predictors"]
        for name in ("always_up", "persistence", "ma_200"):
            assert predictors[name]["brier_score"] is None
        assert 0 < predictors["least_squares"]["brier_score"] < 1
        share = report["data"]["share_of_up_days_tested"]
        assert report["data"]["brier_score_of_base_rate"] == pytest.approx(
            share * (1 - share), abs=1e-6
        )
        assert "probability quality (Brier score" in format_report(report)


def walk(value: object, path: str = "") -> list[tuple[str, object]]:
    """Every (path, leaf) in a nested JSON-like structure."""
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in walk(v, f"{path}/{k}")]
    if isinstance(value, list):
        return [(f"{path}[]", value)] + [leaf for v in value for leaf in walk(v, f"{path}[]")]
    return [(path, value)]


class TestContextModels:
    """Models on the wider inputs are tested beside the others, on the same sessions."""

    def test_extra_columns_change_nothing_for_the_baselines_and_technical_models(
        self, report: dict[str, Any], both: dict[str, Any]
    ) -> None:
        # The first report was made from the technical dataset alone. Every row
        # in it must come out the same from the dataset with the extra columns.
        for name, row in report["predictors"].items():
            assert both["predictors"][name] == row
        assert both["data"] == report["data"]
        assert both["validation"] == report["validation"]

    def test_a_context_model_is_scored_on_the_same_sessions_and_says_what_it_saw(
        self, both: dict[str, Any]
    ) -> None:
        rows = both["predictors"]
        assert list(rows)[-1] == "least_squares_ctx"
        assert rows["least_squares_ctx"]["sessions"] == both["data"]["tested_sessions"]
        assert rows["least_squares_ctx"]["kind"] == "model"
        assert rows["least_squares_ctx"]["inputs"] == "technical, market context and results dates"
        assert {rows[name]["inputs"] for name in list(rows)[:-1]} == {"technical"}

    def test_the_wider_inputs_reach_the_context_model(self, both: dict[str, Any]) -> None:
        rows = both["predictors"]
        assert rows["least_squares_ctx"]["hits"] != rows["least_squares"]["hits"] or (
            rows["least_squares_ctx"]["brier_score"] != rows["least_squares"]["brier_score"]
        )

    def test_a_context_model_is_compared_with_everything_before_it_including_its_twin(
        self, both: dict[str, Any]
    ) -> None:
        references = [
            row["reference"]
            for row in both["comparisons"]
            if row["predictor"] == "least_squares_ctx"
        ]
        assert references == ["always_up", "persistence", "ma_200", "least_squares"]

    def test_a_context_model_needs_the_context_columns(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="the dataset has no columns"):
            run_backtest(
                dataset, context_models={"least_squares_ctx": LeastSquaresLearner}, config=CONFIG
            )

    def test_a_name_cannot_be_both_a_model_and_a_context_model(self, extended: Dataset) -> None:
        with pytest.raises(ValueError, match="both a model and a context model"):
            run_backtest(
                extended,
                models={"least_squares": LeastSquaresLearner},
                context_models={"least_squares": LeastSquaresLearner},
                config=CONFIG,
            )

    @pytest.mark.parametrize("name", list(CONTEXT_MODEL_FACTORIES))
    def test_a_registered_context_model_refuses_the_technical_columns_alone(
        self, dataset: Dataset, extended: Dataset, name: str
    ) -> None:
        # Otherwise it would quietly be a copy of its twin.
        known = dataset.labelled()
        with pytest.raises(ValueError, match="needs the technical, context and results columns"):
            CONTEXT_MODEL_FACTORIES[name]().fit(known.features, known.target_up)
        wide = extended.labelled()
        model = CONTEXT_MODEL_FACTORIES[name]()
        model.fit(wide.features, wide.target_up)
        assert model.name == name

    @pytest.mark.parametrize("name", list(CONTEXT_MODEL_FACTORIES))
    def test_a_context_model_is_its_twin_with_only_the_inputs_changed(self, name: str) -> None:
        twin_name = name.removesuffix("_ctx")
        assert name == f"{twin_name}_ctx"
        twin, model = MODEL_FACTORIES[twin_name], CONTEXT_MODEL_FACTORIES[name]
        assert isinstance(twin, type) and isinstance(model, type)
        assert issubclass(model, twin)
        # Nothing but the name and the column check is overridden: same settings.
        overridden = {key for key in vars(model) if not key.startswith("__")}
        assert overridden == {"name", "fit"}


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
    def inputs(self, tmp_path: Path, *, context_rows: int = 1_700) -> list[str]:
        """Made-up price, context and results files, as command-line arguments."""
        prices_path = tmp_path / "prices.csv"
        context_path = tmp_path / "context.csv"
        results_path = tmp_path / "results.json"
        save_csv(make_prices(rows=1_700), prices_path)
        save_csv(make_context(1_700).iloc[:context_rows], context_path)
        save_json(make_publications(), results_path)
        return [
            "--prices",
            str(prices_path),
            "--context",
            str(context_path),
            "--results",
            str(results_path),
        ]

    def test_runs_from_the_data_files_and_writes_the_report(self, tmp_path: Path) -> None:
        out = tmp_path / "out" / "report.json"
        lines: list[str] = []
        code = main(
            [*self.inputs(tmp_path), "--out", str(out), "--resamples", "50"],
            now=lambda: at("2030-03-14 20:00"),
            report=lines.append,
            min_rows=1_000,
        )
        assert code == 0
        written = json.loads(out.read_text())
        assert list(written["predictors"]) == [
            "always_up",
            "persistence",
            "ma_200",
            "logistic",
            "xgboost",
            "lstm",
            "logistic_ctx",
            "xgboost_ctx",
        ]
        assert written["predictors"]["logistic"]["inputs"] == "technical"
        assert written["predictors"]["logistic_ctx"]["inputs"].startswith("technical, market")
        assert written["validation"]["initial_train_sessions"] == 1_260
        assert written["intervals"]["resamples"] == 50
        assert lines[-1] == f"\nwrote {out}"

    @pytest.mark.parametrize(
        ("option", "what"),
        [("--prices", "price"), ("--context", "market context"), ("--results", "results dates")],
    )
    def test_a_missing_data_file_names_the_fetch_to_run(
        self, tmp_path: Path, option: str, what: str
    ) -> None:
        arguments = self.inputs(tmp_path)
        arguments[arguments.index(option) + 1] = str(tmp_path / "nope")
        lines: list[str] = []
        code = main(arguments, report=lines.append)
        assert code == 2
        assert f"run the {what} fetch first" in lines[0]

    def run_refused(self, tmp_path: Path, when: str, **kwargs: int) -> list[str]:
        out = tmp_path / "report.json"
        lines: list[str] = []
        code = main(
            [*self.inputs(tmp_path, **kwargs), "--out", str(out)],
            now=lambda: at(when),
            report=lines.append,
            min_rows=1_000,
        )
        assert code == 1
        assert not out.exists()
        return lines

    def test_stale_or_invalid_prices_are_refused(self, tmp_path: Path) -> None:
        assert "stale" in self.run_refused(tmp_path, "2030-04-20 20:00")[0]

    def test_context_that_lacks_the_latest_session_is_refused(self, tmp_path: Path) -> None:
        lines = self.run_refused(tmp_path, "2030-03-14 20:00", context_rows=1_699)
        assert "market context has no row for 1 sessions" in lines[0]

    def test_a_run_before_the_last_sessions_cut_off_is_refused(self, tmp_path: Path) -> None:
        lines = self.run_refused(tmp_path, "2030-03-14 17:30")
        assert "too early for a call based on 2030-03-14" in lines[0]
