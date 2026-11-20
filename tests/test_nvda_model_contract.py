"""Rules every registered model must obey. A new model gets these tests for free.

Models on the technical columns and models on the wider "context" columns are both
covered; each is given the columns it is registered for.

The important ones are the look-ahead tests: whatever the model is, its calls for a
session must not depend on anything that happened after that session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from context_frames import make_context, make_publications
from markets_lab.analysis.nvda.backtest import (
    CONTEXT_MODEL_FACTORIES,
    MODEL_FACTORIES,
    PredictorFactory,
)
from markets_lab.analysis.nvda.context_features import (
    EXTENDED_FEATURE_COLUMNS,
    build_context_dataset,
)
from markets_lab.analysis.nvda.features import FEATURE_COLUMNS, Dataset, build_dataset
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP, Predictor
from markets_lab.analysis.nvda.validation import WalkForwardConfig, walk_forward
from price_frames import at, make_prices

CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=100)
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
TRAIN_ROWS = 500
ROWS = 900
ALL_FACTORIES: dict[str, PredictorFactory] = {**MODEL_FACTORIES, **CONTEXT_MODEL_FACTORIES}

pytestmark = pytest.mark.parametrize("name", list(ALL_FACTORIES))


def dataset_for(name: str, prices: pd.DataFrame) -> Dataset:
    """The labelled dataset with the columns this model is registered for."""
    if name in CONTEXT_MODEL_FACTORIES:
        return build_context_dataset(
            prices, make_context(ROWS), make_publications(), as_of=at("2030-03-14 22:30")
        ).labelled()
    return build_dataset(prices).labelled()


def columns_for(name: str) -> list[str]:
    return list(EXTENDED_FEATURE_COLUMNS if name in CONTEXT_MODEL_FACTORIES else FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def datasets(prices: pd.DataFrame) -> dict[str, Dataset]:
    return {name: dataset_for(name, prices) for name in ALL_FACTORIES}


@pytest.fixture
def labelled(request: pytest.FixtureRequest, datasets: dict[str, Dataset]) -> Dataset:
    return datasets[request.node.callspec.params["name"]]


def fitted(name: str, labelled: Dataset) -> Predictor:
    model = ALL_FACTORIES[name]()
    model.fit(labelled.features.iloc[:TRAIN_ROWS], labelled.target_up.iloc[:TRAIN_ROWS])
    return model


def factory(name: str) -> PredictorFactory:
    return ALL_FACTORIES[name]


class TestOutput:
    def test_registered_under_its_own_name(self, name: str) -> None:
        assert ALL_FACTORIES[name]().name == name

    def test_probabilities_and_calls_are_consistent(self, name: str, labelled: Dataset) -> None:
        calls = fitted(name, labelled).predict(labelled.features.iloc[TRAIN_ROWS:])
        probability = calls[PROBABILITY_UP]
        assert calls.index.equals(labelled.features.index[TRAIN_ROWS:])
        assert ((probability >= 0) & (probability <= 1)).all()
        assert (calls[PREDICTED_UP] == (probability > 0.5)).all()
        assert list(labelled.features.columns) == columns_for(name)

    def test_same_training_data_gives_the_same_model(self, name: str, labelled: Dataset) -> None:
        first = fitted(name, labelled).predict(labelled.features.iloc[TRAIN_ROWS:])
        second = fitted(name, labelled).predict(labelled.features.iloc[TRAIN_ROWS:])
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    def test_no_rows_gives_no_calls(self, name: str, labelled: Dataset) -> None:
        assert len(fitted(name, labelled).predict(labelled.features.iloc[:0])) == 0


class TestGuards:
    def test_predict_before_fit_raises(self, name: str, labelled: Dataset) -> None:
        with pytest.raises(RuntimeError, match="fit must be called"):
            ALL_FACTORIES[name]().predict(labelled.features)

    def test_one_sided_training_data_raises(self, name: str, labelled: Dataset) -> None:
        with pytest.raises(ValueError, match="only one outcome"):
            ALL_FACTORIES[name]().fit(labelled.features, labelled.target_up * 0.0)

    @pytest.mark.parametrize("bad", [np.nan, 0.5, 2.0])
    def test_bad_outcomes_raise(self, name: str, labelled: Dataset, bad: float) -> None:
        target = labelled.target_up.copy()
        target.iloc[3] = bad
        with pytest.raises(ValueError, match="must be 0 or 1"):
            ALL_FACTORIES[name]().fit(labelled.features, target)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_bad_feature_values_raise(self, name: str, labelled: Dataset, bad: float) -> None:
        broken = labelled.features.copy()
        broken.loc[broken.index[3], "ret_1"] = bad
        with pytest.raises(ValueError, match="missing or infinite"):
            ALL_FACTORIES[name]().fit(broken, labelled.target_up)
        with pytest.raises(ValueError, match="missing or infinite"):
            fitted(name, labelled).predict(broken)

    def test_different_columns_at_predict_time_raise(self, name: str, labelled: Dataset) -> None:
        reordered = labelled.features[list(reversed(columns_for(name)))]
        with pytest.raises(ValueError, match="columns the model was fitted on"):
            fitted(name, labelled).predict(reordered)

    def test_mismatched_lengths_raise(self, name: str, labelled: Dataset) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            ALL_FACTORIES[name]().fit(labelled.features, labelled.target_up.iloc[:-1])


class TestNoLookAhead:
    def test_a_call_does_not_depend_on_later_rows_in_the_same_batch(
        self, name: str, labelled: Dataset
    ) -> None:
        # Predict the first 10 sessions one prefix at a time, then again inside a
        # batch whose later rows are absurd outliers. If anything were refitted or
        # rescaled on the rows being predicted, or a later row could reach an
        # earlier one, the two would differ.
        model = fitted(name, labelled)
        batch = labelled.features.iloc[TRAIN_ROWS : TRAIN_ROWS + 60].copy()
        prefix = [model.predict(batch.iloc[: i + 1])[PROBABILITY_UP].iloc[-1] for i in range(10)]
        batch.iloc[30:] = batch.iloc[30:] * 1_000.0
        together = model.predict(batch)[PROBABILITY_UP].iloc[:10].tolist()
        assert together == pytest.approx(prefix, rel=1e-5)

    def test_predicting_does_not_change_the_fitted_model(
        self, name: str, labelled: Dataset
    ) -> None:
        model = fitted(name, labelled)
        rows = labelled.features.iloc[TRAIN_ROWS : TRAIN_ROWS + 20]
        first = model.predict(rows)
        model.predict(labelled.features.iloc[TRAIN_ROWS + 20 :] * 1_000.0)
        pd.testing.assert_frame_equal(model.predict(rows), first, check_exact=True)

    def test_corrupting_later_prices_does_not_change_earlier_calls(
        self, name: str, prices: pd.DataFrame
    ) -> None:
        cut = 700
        rng = np.random.default_rng(5)
        corrupted = prices.copy()
        later = corrupted.index[cut + 1 :]
        for column in PRICE_COLUMNS:
            corrupted.loc[later, column] = rng.uniform(1_000, 9_000, len(later))
        corrupted.loc[later, "volume"] = rng.integers(1, 10, len(later))

        clean = walk_forward(dataset_for(name, prices), factory(name), CONFIG)
        dirty = walk_forward(dataset_for(name, corrupted), factory(name), CONFIG)
        through = prices.index[cut]
        columns = [PREDICTED_UP, PROBABILITY_UP]
        pd.testing.assert_frame_equal(
            dirty.loc[:through, columns], clean.loc[:through, columns], check_exact=True
        )
        assert (
            not dirty.loc[through:, columns].iloc[1:].equals(clean.loc[through:, columns].iloc[1:])
        )

    def test_training_outcomes_matter(self, name: str, labelled: Dataset) -> None:
        # If the outcomes it is trained on had no effect, the test above would be
        # empty. Flipping them changes its probabilities.
        flipped = Dataset(labelled.features, 1.0 - labelled.target_up, labelled.next_return)
        normal = walk_forward(labelled, factory(name), CONFIG)[PROBABILITY_UP]
        other = walk_forward(flipped, factory(name), CONFIG)[PROBABILITY_UP]
        assert float(np.abs(normal - other).mean()) > 0.01
