"""Volatility forecasts and ranges: the dataset, the baselines, the scores, the report.

All prices here are made up.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, build_dataset
from markets_lab.analysis.nvda.metrics import BootstrapConfig
from markets_lab.analysis.nvda.validation import LookAheadError, WalkForwardConfig, make_folds
from markets_lab.analysis.nvda.volatility.backtest import (
    BASELINE_FACTORIES,
    MODEL_FACTORIES,
    REFERENCE,
    format_report,
    main,
    run_backtest,
    walk_forward_ranges,
    write_report,
)
from markets_lab.analysis.nvda.volatility.dataset import (
    EWMA_COLUMN,
    EWMA_LAMBDA,
    VOLATILITY_COLUMNS,
    VolatilityDataset,
    build_volatility_dataset,
    compute_ewma_volatility,
)
from markets_lab.analysis.nvda.volatility.forecasters import (
    FORECAST_COLUMNS,
    LOWER,
    RANGE_LEVEL,
    UPPER,
    VOLATILITY,
    ConstantVolatility,
    forecast_frame,
)
from markets_lab.analysis.nvda.volatility.metrics import inside, interval_scores, qlike_losses
from markets_lab.analysis.nvda.volatility.models import HAR_INPUTS, HarVolatility
from markets_lab.collectors.nvda_prices import save_csv
from price_frames import at, make_prices

ROWS = 900
TRAIN_ROWS = 500
CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=50)
QUICK = BootstrapConfig(n_resamples=200)
WHEN = datetime(2030, 3, 15, 12, 0, tzinfo=UTC)
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
ALL_FACTORIES = {**BASELINE_FACTORIES, **MODEL_FACTORIES}


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def dataset(prices: pd.DataFrame) -> VolatilityDataset:
    return build_volatility_dataset(prices)


@pytest.fixture(scope="module")
def labelled(dataset: VolatilityDataset) -> VolatilityDataset:
    return dataset.labelled()


@pytest.fixture(scope="module")
def report(dataset: VolatilityDataset) -> dict[str, Any]:
    return run_backtest(dataset, config=CONFIG, bootstrap=QUICK, generated_at=WHEN)


def fitted(name: str, labelled: VolatilityDataset) -> Any:
    forecaster = ALL_FACTORIES[name]()
    forecaster.fit(labelled.features.iloc[:TRAIN_ROWS], labelled.next_log_return.iloc[:TRAIN_ROWS])
    return forecaster


def walk(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every leaf of a JSON-like value, with a readable path to it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, list) and not all(isinstance(x, int | float) for x in node):
        for number, value in enumerate(node):
            yield from walk(value, f"{path}[{number}]")
    else:
        yield path, node


class TestDataset:
    def test_same_sessions_and_technical_columns_as_the_direction_dataset(
        self, prices: pd.DataFrame, dataset: VolatilityDataset
    ) -> None:
        plain = build_dataset(prices)
        assert tuple(dataset.features.columns) == VOLATILITY_COLUMNS
        assert dataset.features.index.equals(plain.features.index)
        pd.testing.assert_frame_equal(dataset.features[list(FEATURE_COLUMNS)], plain.features)
        assert np.isfinite(dataset.features.to_numpy()).all()

    def test_the_target_is_the_next_sessions_log_return(
        self, prices: pd.DataFrame, dataset: VolatilityDataset
    ) -> None:
        adj = prices["adj_close"]
        session = dataset.features.index[100]
        position = int(prices.index.get_indexer(pd.DatetimeIndex([session]))[0])
        expected = np.log(adj.iloc[position + 1] / adj.iloc[position])
        assert dataset.next_log_return.loc[session] == pytest.approx(expected)

    def test_the_last_session_has_no_target_yet(self, dataset: VolatilityDataset) -> None:
        assert np.isnan(dataset.next_log_return.iloc[-1])
        assert dataset.next_log_return.iloc[:-1].notna().all()
        assert len(dataset.labelled()) == len(dataset) - 1
        last, row = dataset.latest()
        assert last.isoformat() == "2030-03-14"
        assert row.equals(dataset.features.iloc[-1])

    def test_the_weighted_volatility_follows_its_update_rule(self, prices: pd.DataFrame) -> None:
        ewma = compute_ewma_volatility(prices)
        daily = pd.Series(np.log(prices["adj_close"].to_numpy()), index=prices.index).diff()
        t = 400
        expected = EWMA_LAMBDA * ewma.iloc[t - 1] ** 2 + (1 - EWMA_LAMBDA) * daily.iloc[t] ** 2
        assert ewma.iloc[t] ** 2 == pytest.approx(expected)
        assert EWMA_LAMBDA == 0.94

    def test_the_weighted_volatility_never_uses_later_sessions(self, prices: pd.DataFrame) -> None:
        full = compute_ewma_volatility(prices)
        for end in (300, 600):
            pd.testing.assert_series_equal(compute_ewma_volatility(prices.iloc[:end]), full[:end])
        poisoned = prices.copy()
        poisoned.loc[poisoned.index[600] :, "adj_close"] = 1e9
        after = compute_ewma_volatility(poisoned)
        pd.testing.assert_series_equal(after.iloc[:600], full.iloc[:600])
        assert after.iloc[600] != full.iloc[600]


@pytest.mark.parametrize("name", list(ALL_FACTORIES))
class TestEveryForecaster:
    def test_registered_under_its_own_name(self, name: str) -> None:
        assert ALL_FACTORIES[name]().name == name

    def test_gives_a_positive_volatility_and_an_ordered_range_per_session(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        rows = labelled.features.iloc[TRAIN_ROWS:]
        forecasts = fitted(name, labelled).predict(rows)
        assert tuple(forecasts.columns) == FORECAST_COLUMNS
        assert forecasts.index.equals(rows.index)
        assert (forecasts[VOLATILITY] > 0).all()
        assert (forecasts[LOWER] < forecasts[UPPER]).all()

    def test_the_range_holds_four_fifths_of_the_returns_it_was_built_from(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        # By construction: the bounds are the 10th and 90th percentiles of the
        # training returns over their forecast volatility.
        train = labelled.features.iloc[:TRAIN_ROWS]
        outcome = labelled.next_log_return.iloc[:TRAIN_ROWS].to_numpy()
        forecasts = fitted(name, labelled).predict(train)
        lower, upper = forecasts[LOWER].to_numpy(), forecasts[UPPER].to_numpy()
        assert inside(lower, upper, outcome).mean() == pytest.approx(RANGE_LEVEL, abs=0.005)
        assert (outcome < lower).mean() == pytest.approx(0.1, abs=0.005)
        assert (outcome > upper).mean() == pytest.approx(0.1, abs=0.005)

    def test_the_range_is_the_volatility_times_two_fixed_numbers(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        forecasts = fitted(name, labelled).predict(labelled.features.iloc[TRAIN_ROWS:])
        assert (forecasts[LOWER] / forecasts[VOLATILITY]).std() == pytest.approx(0, abs=1e-12)
        assert (forecasts[UPPER] / forecasts[VOLATILITY]).std() == pytest.approx(0, abs=1e-12)

    def test_same_training_data_gives_the_same_forecasts(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        rows = labelled.features.iloc[TRAIN_ROWS:]
        pd.testing.assert_frame_equal(
            fitted(name, labelled).predict(rows),
            fitted(name, labelled).predict(rows),
            check_exact=True,
        )

    def test_predicting_does_not_change_the_fitted_forecaster(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        forecaster = fitted(name, labelled)
        rows = labelled.features.iloc[TRAIN_ROWS : TRAIN_ROWS + 20]
        first = forecaster.predict(rows)
        forecaster.predict(labelled.features.iloc[TRAIN_ROWS + 20 :])
        pd.testing.assert_frame_equal(forecaster.predict(rows), first, check_exact=True)

    def test_a_forecast_does_not_depend_on_later_rows_in_the_same_batch(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        forecaster = fitted(name, labelled)
        batch = labelled.features.iloc[TRAIN_ROWS : TRAIN_ROWS + 60].copy()
        alone = forecaster.predict(batch.iloc[:10])
        batch.iloc[30:] = batch.iloc[30:] * 50.0
        pd.testing.assert_frame_equal(forecaster.predict(batch).iloc[:10], alone, check_exact=True)

    def test_corrupting_later_prices_does_not_change_earlier_forecasts(
        self, name: str, prices: pd.DataFrame
    ) -> None:
        cut = 700
        rng = np.random.default_rng(5)
        corrupted = prices.copy()
        later = corrupted.index[cut + 1 :]
        # Nonsense levels, but each bar still has its high above its low, so a
        # forecaster that reads the day's range is given something it accepts.
        close = rng.uniform(1_000, 9_000, len(later))
        open_ = close * rng.uniform(0.9, 1.1, len(later))
        corrupted.loc[later, "close"] = close
        corrupted.loc[later, "adj_close"] = close
        corrupted.loc[later, "open"] = open_
        corrupted.loc[later, "high"] = np.maximum(open_, close) * 1.05
        corrupted.loc[later, "low"] = np.minimum(open_, close) * 0.95
        clean = walk_forward_ranges(
            build_volatility_dataset(prices).labelled(), ALL_FACTORIES[name], CONFIG
        )
        dirty = walk_forward_ranges(
            build_volatility_dataset(corrupted).labelled(), ALL_FACTORIES[name], CONFIG
        )
        # The forecast made after session `cut` is for the first corrupted session,
        # and is itself made only from clean data.
        through = prices.index[cut]
        columns = list(FORECAST_COLUMNS)
        pd.testing.assert_frame_equal(
            dirty.loc[:through, columns], clean.loc[:through, columns], check_exact=True
        )
        assert (
            not dirty.loc[through:, columns].iloc[1:].equals(clean.loc[through:, columns].iloc[1:])
        )

    def test_training_outcomes_matter(self, name: str, labelled: VolatilityDataset) -> None:
        # If the outcomes it is trained on had no effect, the test above would be
        # empty. Doubling them widens every range.
        doubled = VolatilityDataset(labelled.features, labelled.next_log_return * 2.0)
        normal = walk_forward_ranges(labelled, ALL_FACTORIES[name], CONFIG)
        wider = walk_forward_ranges(doubled, ALL_FACTORIES[name], CONFIG)
        assert ((wider[UPPER] - wider[LOWER]) > 1.9 * (normal[UPPER] - normal[LOWER])).all()

    def test_predict_before_fit_raises(self, name: str, labelled: VolatilityDataset) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            ALL_FACTORIES[name]().predict(labelled.features)

    def test_bad_training_data_raises(self, name: str, labelled: VolatilityDataset) -> None:
        features, outcome = labelled.features, labelled.next_log_return
        with pytest.raises(ValueError, match="same number of rows"):
            ALL_FACTORIES[name]().fit(features, outcome.iloc[:-1])
        with pytest.raises(ValueError, match="at least 100 training sessions"):
            ALL_FACTORIES[name]().fit(features.iloc[:50], outcome.iloc[:50])
        broken = outcome.copy()
        broken.iloc[3] = np.nan
        with pytest.raises(ValueError, match="missing or infinite"):
            ALL_FACTORIES[name]().fit(features, broken)

    def test_different_columns_at_predict_time_raise(
        self, name: str, labelled: VolatilityDataset
    ) -> None:
        reordered = labelled.features[list(reversed(VOLATILITY_COLUMNS))]
        with pytest.raises(ValueError, match="columns the forecaster was fitted on"):
            fitted(name, labelled).predict(reordered)


class TestBaselines:
    def test_registered_baselines_and_reference(self) -> None:
        assert list(BASELINE_FACTORIES) == ["constant", "rolling_21", "ewma_94"]
        assert REFERENCE == "constant"
        assert RANGE_LEVEL == 0.8

    def test_constant_is_the_middle_four_fifths_of_past_returns(
        self, labelled: VolatilityDataset
    ) -> None:
        outcome = labelled.next_log_return.iloc[:TRAIN_ROWS].to_numpy()
        forecasts = fitted("constant", labelled).predict(labelled.features.iloc[TRAIN_ROWS:])
        low, high = np.quantile(outcome, [0.1, 0.9])
        assert forecasts[LOWER].to_numpy() == pytest.approx(np.full(len(forecasts), low))
        assert forecasts[UPPER].to_numpy() == pytest.approx(np.full(len(forecasts), high))
        assert forecasts[VOLATILITY].iloc[0] == pytest.approx(np.sqrt(np.mean(outcome**2)))

    @pytest.mark.parametrize(
        ("name", "column"), [("rolling_21", "vol_21"), ("ewma_94", EWMA_COLUMN)]
    )
    def test_the_others_take_their_volatility_from_one_column(
        self, labelled: VolatilityDataset, name: str, column: str
    ) -> None:
        rows = labelled.features.iloc[TRAIN_ROWS:]
        forecasts = fitted(name, labelled).predict(rows)
        np.testing.assert_array_equal(forecasts[VOLATILITY].to_numpy(), rows[column].to_numpy())

    def test_returns_that_never_vary_cannot_make_a_range(self, labelled: VolatilityDataset) -> None:
        flat = labelled.next_log_return * 0.0 + 0.01
        with pytest.raises(ValueError, match="do not vary"):
            ConstantVolatility().fit(labelled.features, flat)

    def test_a_volatility_of_zero_is_refused(self, labelled: VolatilityDataset) -> None:
        broken = labelled.features.copy()
        broken.loc[broken.index[10], "vol_21"] = 0.0
        with pytest.raises(ValueError, match="finite and positive"):
            BASELINE_FACTORIES["rolling_21"]().fit(broken, labelled.next_log_return)

    def test_forecast_frame_refuses_nonsense(self) -> None:
        index = pd.RangeIndex(2)
        ok = np.array([0.02, 0.03])
        forecast_frame(index, ok, -ok, ok)
        with pytest.raises(ValueError, match="volatility must be positive"):
            forecast_frame(index, np.array([0.02, 0.0]), -ok, ok)
        with pytest.raises(ValueError, match="lower bound must be below"):
            forecast_frame(index, ok, ok, ok)
        with pytest.raises(ValueError, match="one finite number per row"):
            forecast_frame(index, ok, np.array([np.nan, -0.1]), ok)

    def test_constant_learns_only_from_its_training_rows(self) -> None:
        assert isinstance(ConstantVolatility(), ConstantVolatility)
        calm = pd.Series(np.tile([0.01, -0.01], 100))
        wild = pd.Series(np.tile([0.05, -0.05], 100))
        features = pd.DataFrame({"x": np.zeros(200)})
        first, second = ConstantVolatility(), ConstantVolatility()
        first.fit(features, calm)
        second.fit(features, wild)
        assert first.predict(features)[VOLATILITY].iloc[0] == pytest.approx(0.01)
        assert second.predict(features)[VOLATILITY].iloc[0] == pytest.approx(0.05)


class TestHar:
    """The regression on volatility at three speeds."""

    def test_registered_models(self) -> None:
        assert list(MODEL_FACTORIES) == ["har"]
        assert HAR_INPUTS == ("range_pct", "vol_5", "vol_21")

    def test_it_reads_only_its_three_inputs(self, labelled: VolatilityDataset) -> None:
        rows = labelled.features.iloc[TRAIN_ROWS:]
        forecaster = fitted("har", labelled)
        changed = rows.copy()
        others = [name for name in rows.columns if name not in HAR_INPUTS]
        changed[others] = changed[others] * 3.0 + 1.0
        pd.testing.assert_frame_equal(forecaster.predict(changed), forecaster.predict(rows))
        for name in HAR_INPUTS:
            moved = rows.copy()
            moved[name] = moved[name] * 2.0
            assert not forecaster.predict(moved)[VOLATILITY].equals(
                forecaster.predict(rows)[VOLATILITY]
            )

    def test_it_recovers_a_known_relationship(self) -> None:
        # Made-up data where the next variance really is exp(-7) * range ** 1.2 *
        # vol_5 ** 0.5, with vol_21 playing no part.
        rng = np.random.default_rng(11)
        n = 40_000
        features = pd.DataFrame(
            {
                "range_pct": np.exp(rng.normal(-3.5, 0.4, n)),
                "vol_5": np.exp(rng.normal(-3.8, 0.4, n)),
                "vol_21": np.exp(rng.normal(-3.8, 0.4, n)),
            }
        )
        variance = np.exp(-7.0) * features["range_pct"] ** 1.2 * features["vol_5"] ** 0.5
        outcome = pd.Series(rng.normal(0.0, 1.0, n) * np.sqrt(variance))
        model = HarVolatility()
        model.fit(features, outcome)
        weights = model.coefficients()
        assert weights["constant"] == pytest.approx(-7.0, abs=0.3)
        assert weights["range_pct"] == pytest.approx(1.2, abs=0.08)
        assert weights["vol_5"] == pytest.approx(0.5, abs=0.08)
        assert weights["vol_21"] == pytest.approx(0.0, abs=0.08)
        forecast = model.predict(features)[VOLATILITY].to_numpy()
        assert np.corrcoef(np.log(forecast), np.log(np.sqrt(variance)))[0, 1] > 0.99

    def test_a_flat_session_in_the_training_data_is_accepted(
        self, labelled: VolatilityDataset
    ) -> None:
        outcome = labelled.next_log_return.copy()
        outcome.iloc[[5, 50, 300]] = 0.0
        model = HarVolatility()
        model.fit(labelled.features, outcome)
        assert np.isfinite(model.predict(labelled.features)[VOLATILITY]).all()

    def test_an_input_that_is_not_positive_is_refused(self, labelled: VolatilityDataset) -> None:
        broken = labelled.features.copy()
        broken.loc[broken.index[10], "range_pct"] = 0.0
        with pytest.raises(ValueError, match="finite and positive"):
            HarVolatility().fit(broken, labelled.next_log_return)
        with pytest.raises(ValueError, match="finite and positive"):
            fitted("har", labelled).predict(broken)

    def test_coefficients_need_a_fit(self) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            HarVolatility().coefficients()


class TestScores:
    def test_qlike_by_hand(self) -> None:
        losses = qlike_losses(np.array([0.02, 0.01]), np.array([0.02, -0.03]))
        assert losses[0] == pytest.approx(np.log(0.0004) + 1.0)
        assert losses[1] == pytest.approx(np.log(0.0001) + 9.0)

    def test_qlike_is_lowest_at_the_true_volatility(self) -> None:
        rng = np.random.default_rng(3)
        outcome = rng.normal(0, 0.03, 20_000)
        loss = {
            scale: qlike_losses(np.full(20_000, 0.03 * scale), outcome).mean()
            for scale in (0.5, 1, 2)
        }
        assert loss[1] < loss[0.5] and loss[1] < loss[2]

    def test_interval_score_by_hand(self) -> None:
        lower, upper = np.full(3, -0.02), np.full(3, 0.03)
        outcome = np.array([0.0, -0.05, 0.04])
        scores = interval_scores(lower, upper, outcome, 0.8)
        # Width 0.05; outside by 0.03 and 0.01, each charged 2 / (1 - 0.8) = 10 times.
        assert scores == pytest.approx([0.05, 0.05 + 0.30, 0.05 + 0.10])

    def test_a_wider_range_only_helps_if_it_catches_more(self) -> None:
        outcome = np.array([0.0, 0.01, -0.01])
        narrow = interval_scores(np.full(3, -0.02), np.full(3, 0.02), outcome, 0.8).mean()
        wide = interval_scores(np.full(3, -0.04), np.full(3, 0.04), outcome, 0.8).mean()
        assert narrow < wide

    def test_inside_includes_the_bounds(self) -> None:
        held = inside(np.full(4, -0.02), np.full(4, 0.03), np.array([-0.02, 0.03, 0.031, -0.021]))
        assert held.tolist() == [True, True, False, False]

    def test_bad_inputs_raise(self) -> None:
        two, three = np.array([0.01, 0.02]), np.array([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="same length"):
            qlike_losses(two, three)
        with pytest.raises(ValueError, match="volatility must be positive"):
            qlike_losses(np.array([0.01, 0.0]), two)
        with pytest.raises(ValueError, match="must be finite"):
            qlike_losses(two, np.array([np.nan, 0.01]))
        with pytest.raises(ValueError, match="lower bound must be below"):
            interval_scores(two, two, two, 0.8)
        with pytest.raises(ValueError, match="level must be between"):
            interval_scores(-two, two, two, 1.0)


class TestWalkForward:
    def test_every_tested_session_is_forecast_once_by_a_forecaster_trained_before_it(
        self, labelled: VolatilityDataset
    ) -> None:
        forecasts = walk_forward_ranges(labelled, ConstantVolatility, CONFIG)
        assert forecasts.index.equals(labelled.features.index[300:])
        assert (forecasts["trained_through"] < forecasts.index).all()
        assert forecasts["fold"].iloc[0] == 0 and forecasts["fold"].is_monotonic_increasing

    def test_an_unlabelled_dataset_is_refused(self, dataset: VolatilityDataset) -> None:
        with pytest.raises(ValueError, match="fully labelled"):
            walk_forward_ranges(dataset, ConstantVolatility, CONFIG)

    def test_a_fold_that_trains_on_its_test_rows_is_refused(
        self, labelled: VolatilityDataset
    ) -> None:
        folds = make_folds(len(labelled), CONFIG)
        leaky = [type(folds[0])(0, folds[0].train_end + 5, folds[0].test_start, folds[0].test_end)]
        with pytest.raises(LookAheadError, match="trains on rows up to"):
            walk_forward_ranges(labelled, ConstantVolatility, CONFIG, folds=leaky)


class TestReportContents:
    def test_describes_the_data_and_the_validation(
        self, report: dict[str, Any], dataset: VolatilityDataset
    ) -> None:
        tested = len(dataset.labelled()) - 300
        assert report["generated_at"] == "2030-03-15T12:00:00Z"
        assert report["symbol"] == "NVDA"
        assert report["range_level"] == 0.8
        assert report["data"]["tested_sessions"] == tested
        assert report["data"]["last_tested_session"] == "2030-03-13"
        assert report["validation"] == {
            "scheme": "walk-forward, expanding window",
            "initial_train_sessions": 300,
            "refit_every_sessions": 50,
            "folds": -(-tested // 50),
        }

    def test_every_forecaster_is_scored_on_the_same_sessions(self, report: dict[str, Any]) -> None:
        rows = report["forecasters"]
        assert list(rows) == [*BASELINE_FACTORIES, *MODEL_FACTORIES]
        assert {row["sessions"] for row in rows.values()} == {report["data"]["tested_sessions"]}
        kinds = [row["kind"] for row in rows.values()]
        assert kinds == ["baseline"] * len(BASELINE_FACTORIES) + ["model"] * len(MODEL_FACTORIES)

    def test_coverage_and_the_two_kinds_of_miss_add_up(self, report: dict[str, Any]) -> None:
        for row in report["forecasters"].values():
            total = row["coverage"] + row["share_below"] + row["share_above"]
            assert total == pytest.approx(1.0, abs=2e-6)
            low, high = row["coverage_interval"]
            assert low <= row["coverage"] <= high
            assert row["median_width"] > 0 and row["mean_width"] > 0

    def test_made_up_prices_with_steady_volatility_are_covered_about_as_built(
        self, report: dict[str, Any]
    ) -> None:
        # The synthetic returns have the same spread throughout, so even the
        # simplest forecaster's range should hold close to 80% out of sample.
        assert report["forecasters"]["constant"]["coverage"] == pytest.approx(0.8, abs=0.06)

    def test_every_forecaster_is_compared_with_each_one_listed_before_it(
        self, report: dict[str, Any]
    ) -> None:
        pairs = [(row["forecaster"], row["reference"]) for row in report["comparisons"]]
        names = [*BASELINE_FACTORIES, *MODEL_FACTORIES]
        assert pairs == [
            (name, earlier) for position, name in enumerate(names) for earlier in names[:position]
        ]
        assert pairs[:3] == [
            ("rolling_21", "constant"),
            ("ewma_94", "constant"),
            ("ewma_94", "rolling_21"),
        ]

    def test_differences_match_the_forecaster_rows(self, report: dict[str, Any]) -> None:
        rows = report["forecasters"]
        for comparison in report["comparisons"]:
            ours, theirs = rows[comparison["forecaster"]], rows[comparison["reference"]]
            assert comparison["qlike_difference"] == pytest.approx(
                ours["qlike"] - theirs["qlike"], abs=2e-6
            )
            assert comparison["interval_score_difference"] == pytest.approx(
                ours["interval_score"] - theirs["interval_score"], abs=2e-6
            )
            for key in ("qlike_difference", "interval_score_difference"):
                low, high = comparison[f"{key}_interval"]
                assert low <= comparison[key] <= high

    def test_same_inputs_give_the_same_report(
        self, report: dict[str, Any], dataset: VolatilityDataset
    ) -> None:
        again = run_backtest(dataset, config=CONFIG, bootstrap=QUICK, generated_at=WHEN)
        assert again == report

    def test_a_model_is_listed_after_the_baselines_and_compared_with_all_of_them(
        self, dataset: VolatilityDataset
    ) -> None:
        class Wider(ConstantVolatility):
            name = "wider"

            def _volatility(self, features: pd.DataFrame) -> np.ndarray:
                return super()._volatility(features) * 1.5

        result = run_backtest(dataset, models={"wider": Wider}, config=CONFIG, bootstrap=QUICK)
        assert list(result["forecasters"])[:3] == ["constant", "rolling_21", "ewma_94"]
        assert list(result["forecasters"])[-1] == "wider"
        assert result["forecasters"]["wider"]["kind"] == "model"
        references = [r["reference"] for r in result["comparisons"] if r["forecaster"] == "wider"]
        assert references == ["constant", "rolling_21", "ewma_94"]
        # The same range in the end: the percentiles absorb the constant factor.
        assert (
            result["forecasters"]["wider"]["coverage"]
            == result["forecasters"]["constant"]["coverage"]
        )

    def test_a_name_cannot_be_both_baseline_and_model(self, dataset: VolatilityDataset) -> None:
        with pytest.raises(ValueError, match="both a baseline and a model"):
            run_backtest(dataset, models={"constant": ConstantVolatility}, config=CONFIG)


class TestReportHoldsAggregatesOnly:
    """The report is committed and published, so it must not carry raw data."""

    def test_it_is_plain_json(self, report: dict[str, Any]) -> None:
        assert json.loads(json.dumps(report, allow_nan=False)) == report

    def test_no_per_session_series(self, report: dict[str, Any]) -> None:
        for path, leaf in walk(report):
            if isinstance(leaf, list):
                assert len(leaf) <= 2, f"{path} holds a series of {len(leaf)} numbers"

    def test_no_price_appears_anywhere(self, report: dict[str, Any], prices: pd.DataFrame) -> None:
        raw = {round(float(v), 4) for column in PRICE_COLUMNS for v in prices[column]}
        numbers = [leaf for _, leaf in walk(report) if isinstance(leaf, float)]
        assert numbers
        assert not [n for n in numbers if round(n, 4) in raw]

    def test_field_names_do_not_suggest_raw_data(self, report: dict[str, Any]) -> None:
        text = json.dumps(report).lower()
        for word in ('"close"', '"open"', '"price"', '"prices"', '"daily_returns"', '"volume"'):
            assert word not in text


class TestOutput:
    def test_text_summary_names_every_forecaster_and_comparison(
        self, report: dict[str, Any]
    ) -> None:
        text = format_report(report)
        for name in report["forecasters"]:
            assert f"\n{name}" in text
        assert "ewma_94 vs constant: QLIKE" in text
        assert "built to hold 80% of outcomes" in text

    def test_write_report_replaces_in_one_step(
        self, report: dict[str, Any], tmp_path: Path
    ) -> None:
        path = tmp_path / "reports" / "nvda_volatility.json"
        write_report(report, path)
        write_report(report, path)
        assert json.loads(path.read_text()) == report
        assert [p.name for p in path.parent.iterdir()] == ["nvda_volatility.json"]


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
        assert list(written["forecasters"]) == ["constant", "rolling_21", "ewma_94", "har"]
        assert written["validation"]["initial_train_sessions"] == 1_260
        assert written["intervals"]["resamples"] == 50
        assert lines[-1] == f"\nwrote {out}"

    def test_missing_price_file(self, tmp_path: Path) -> None:
        lines: list[str] = []
        code = main(["--prices", str(tmp_path / "nope.csv")], report=lines.append)
        assert code == 2
        assert "run the price fetch first" in lines[0]

    def test_stale_prices_are_refused(self, tmp_path: Path) -> None:
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
