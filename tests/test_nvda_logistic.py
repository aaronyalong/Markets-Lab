"""Logistic regression: it learns, it is repeatable, and its scaling cannot leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.backtest import MODEL_FACTORIES
from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, Dataset, build_dataset
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP, Predictor
from markets_lab.analysis.nvda.validation import WalkForwardConfig, walk_forward
from price_frames import make_prices

CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=50)
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


def learnable(rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Random features where the outcome is decided by a mix of the first two."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2030-01-07", periods=rows, name="date")
    features = pd.DataFrame(
        rng.normal(0, 1, (rows, 4)) * [1.0, 50.0, 0.01, 1.0] + [0.0, 100.0, 0.0, 5.0],
        index=index,
        columns=["a", "b", "c", "d"],
    )
    signal = features["a"] + (features["b"] - 100.0) / 50.0
    return features, (signal > 0).astype("float64")


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=900)


@pytest.fixture(scope="module")
def labelled(prices: pd.DataFrame) -> Dataset:
    return build_dataset(prices).labelled()


class TestLearning:
    def test_learns_a_pattern_that_is_there(self) -> None:
        features, target = learnable(1_200)
        model = LogisticRegressionModel()
        model.fit(features.iloc[:800], target.iloc[:800])
        calls = model.predict(features.iloc[800:])
        accuracy = (calls[PREDICTED_UP] == (target.iloc[800:] == 1.0)).mean()
        assert accuracy > 0.95

    def test_finds_nothing_in_noise(self) -> None:
        features, _ = learnable(3_000)
        noise = pd.Series(
            (np.random.default_rng(9).random(3_000) > 0.5).astype("float64"), index=features.index
        )
        model = LogisticRegressionModel()
        model.fit(features.iloc[:2_000], noise.iloc[:2_000])
        calls = model.predict(features.iloc[2_000:])
        accuracy = (calls[PREDICTED_UP] == (noise.iloc[2_000:] == 1.0)).mean()
        assert 0.44 < accuracy < 0.56

    def test_feature_scale_does_not_matter(self) -> None:
        # Column b is 5,000 times larger than column c; standardising handles it.
        features, target = learnable(1_000)
        model = LogisticRegressionModel()
        model.fit(features.iloc[:700], target.iloc[:700])
        weights = model.coefficients()
        assert abs(weights["a"]) > 10 * abs(weights["c"])
        assert abs(weights["b"]) > 10 * abs(weights["c"])
        assert list(weights.index) == ["a", "b", "c", "d"]


class TestOutput:
    def test_probabilities_and_calls_are_consistent(self, labelled: Dataset) -> None:
        model = LogisticRegressionModel()
        model.fit(labelled.features.iloc[:500], labelled.target_up.iloc[:500])
        calls = model.predict(labelled.features.iloc[500:])
        probability = calls[PROBABILITY_UP]
        assert calls.index.equals(labelled.features.index[500:])
        assert ((probability > 0) & (probability < 1)).all()
        assert (calls[PREDICTED_UP] == (probability > 0.5)).all()

    def test_same_training_data_gives_the_same_model(self, labelled: Dataset) -> None:
        results = []
        for _ in range(2):
            model = LogisticRegressionModel()
            model.fit(labelled.features.iloc[:500], labelled.target_up.iloc[:500])
            results.append(model.predict(labelled.features.iloc[500:]))
        pd.testing.assert_frame_equal(results[0], results[1], check_exact=True)

    def test_no_rows_gives_no_calls(self, labelled: Dataset) -> None:
        model = LogisticRegressionModel()
        model.fit(labelled.features.iloc[:500], labelled.target_up.iloc[:500])
        assert len(model.predict(labelled.features.iloc[:0])) == 0


class TestGuards:
    def test_predict_before_fit_raises(self, labelled: Dataset) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            LogisticRegressionModel().predict(labelled.features)
        with pytest.raises(RuntimeError, match="fit must be called"):
            LogisticRegressionModel().coefficients()

    def test_one_sided_training_data_raises(self) -> None:
        features, target = learnable(100)
        with pytest.raises(ValueError, match="only one outcome"):
            LogisticRegressionModel().fit(features, target * 0.0)

    @pytest.mark.parametrize("bad", [np.nan, 0.5, 2.0])
    def test_bad_outcomes_raise(self, bad: float) -> None:
        features, target = learnable(100)
        target.iloc[3] = bad
        with pytest.raises(ValueError, match="must be 0 or 1"):
            LogisticRegressionModel().fit(features, target)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_bad_feature_values_raise(self, bad: float) -> None:
        features, target = learnable(100)
        broken = features.copy()
        broken.loc[broken.index[3], "a"] = bad
        with pytest.raises(ValueError, match="missing or infinite"):
            LogisticRegressionModel().fit(broken, target)
        model = LogisticRegressionModel()
        model.fit(features, target)
        with pytest.raises(ValueError, match="missing or infinite"):
            model.predict(broken)

    def test_different_columns_at_predict_time_raise(self) -> None:
        features, target = learnable(100)
        model = LogisticRegressionModel()
        model.fit(features, target)
        with pytest.raises(ValueError, match="columns the model was fitted on"):
            model.predict(features[["b", "a", "c", "d"]])


class TestNoLookAhead:
    def test_a_call_does_not_depend_on_the_other_rows_it_is_predicted_with(
        self, labelled: Dataset
    ) -> None:
        # If scaling were refitted on the rows being predicted, a row's probability
        # would change with its batch. Here the batch includes absurd outliers.
        model = LogisticRegressionModel()
        model.fit(labelled.features.iloc[:500], labelled.target_up.iloc[:500])
        batch = labelled.features.iloc[500:560].copy()
        alone = [model.predict(batch.iloc[[i]])[PROBABILITY_UP].iloc[0] for i in range(10)]
        batch.iloc[30:] = batch.iloc[30:] * 1_000.0
        together = model.predict(batch)[PROBABILITY_UP].iloc[:10].tolist()
        assert together == pytest.approx(alone, rel=1e-12)

    def test_predicting_does_not_change_the_fitted_model(self, labelled: Dataset) -> None:
        model = LogisticRegressionModel()
        model.fit(labelled.features.iloc[:500], labelled.target_up.iloc[:500])
        before = model.coefficients().copy()
        first = model.predict(labelled.features.iloc[500:520])
        model.predict(labelled.features.iloc[520:] * 1_000.0)
        pd.testing.assert_series_equal(model.coefficients(), before, check_exact=True)
        pd.testing.assert_frame_equal(
            model.predict(labelled.features.iloc[500:520]), first, check_exact=True
        )

    @pytest.mark.parametrize("cut", [500, 800])
    def test_corrupting_later_prices_does_not_change_earlier_calls(
        self, prices: pd.DataFrame, cut: int
    ) -> None:
        rng = np.random.default_rng(5)
        corrupted = prices.copy()
        later = corrupted.index[cut + 1 :]
        for column in PRICE_COLUMNS:
            corrupted.loc[later, column] = rng.uniform(1_000, 9_000, len(later))
        corrupted.loc[later, "volume"] = rng.integers(1, 10, len(later))

        clean = walk_forward(build_dataset(prices).labelled(), LogisticRegressionModel, CONFIG)
        dirty = walk_forward(build_dataset(corrupted).labelled(), LogisticRegressionModel, CONFIG)
        through = prices.index[cut]
        columns = [PREDICTED_UP, PROBABILITY_UP]
        pd.testing.assert_frame_equal(
            dirty.loc[:through, columns], clean.loc[:through, columns], check_exact=True
        )
        assert (
            not dirty.loc[through:, columns].iloc[1:].equals(clean.loc[through:, columns].iloc[1:])
        )

    def test_training_outcomes_matter(self, labelled: Dataset) -> None:
        flipped = Dataset(labelled.features, 1.0 - labelled.target_up, labelled.next_return)
        normal = walk_forward(labelled, LogisticRegressionModel, CONFIG)[PROBABILITY_UP]
        other = walk_forward(flipped, LogisticRegressionModel, CONFIG)[PROBABILITY_UP]
        assert (normal + other).to_numpy() == pytest.approx(1.0, abs=1e-3)


class TestRegistration:
    def test_registered_under_its_name_and_fits_the_interface(self, labelled: Dataset) -> None:
        assert MODEL_FACTORIES["logistic"] is LogisticRegressionModel
        model: Predictor = MODEL_FACTORIES["logistic"]()
        assert model.name == "logistic"
        model.fit(labelled.features.iloc[:400], labelled.target_up.iloc[:400])
        assert list(labelled.features.columns) == list(FEATURE_COLUMNS)
        assert len(model.predict(labelled.features.iloc[400:410])) == 10
