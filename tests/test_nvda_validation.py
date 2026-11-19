"""Walk-forward validation, and the tests that fail if a model can see the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.baselines import AlwaysUp
from markets_lab.analysis.nvda.features import Dataset, build_dataset
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PREDICTION_COLUMNS, PROBABILITY_UP
from markets_lab.analysis.nvda.validation import (
    FOLD,
    TRAINED_THROUGH,
    Fold,
    LookAheadError,
    WalkForwardConfig,
    make_folds,
    walk_forward,
)
from nvda_test_predictors import LeastSquaresLearner, Memorizer, Spy
from price_frames import make_prices

ROWS = 900
CONFIG = WalkForwardConfig(initial_train_sessions=300, refit_every=50)
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def labelled(prices: pd.DataFrame) -> Dataset:
    return build_dataset(prices).labelled()


class TestMakeFolds:
    def test_folds_cover_every_later_row_once_in_order(self) -> None:
        folds = make_folds(1_000, WalkForwardConfig(initial_train_sessions=300, refit_every=50))
        assert len(folds) == 14
        assert folds[0] == Fold(number=0, train_end=300, test_start=300, test_end=350)
        tested = [row for fold in folds for row in range(fold.test_start, fold.test_end)]
        assert tested == list(range(300, 1_000))

    def test_every_fold_trains_only_on_rows_before_its_test_rows(self) -> None:
        for fold in make_folds(
            1_234, WalkForwardConfig(initial_train_sessions=100, refit_every=21)
        ):
            assert fold.train_end == fold.test_start
            assert fold.test_start < fold.test_end

    def test_last_fold_may_be_short(self) -> None:
        folds = make_folds(320, WalkForwardConfig(initial_train_sessions=300, refit_every=50))
        assert folds == [Fold(number=0, train_end=300, test_start=300, test_end=320)]

    def test_defaults_are_the_approved_definitions(self) -> None:
        config = WalkForwardConfig()
        assert (config.initial_train_sessions, config.refit_every) == (1_260, 21)

    def test_needs_more_rows_than_the_initial_window(self) -> None:
        with pytest.raises(ValueError, match="not more than the initial training window"):
            make_folds(300, WalkForwardConfig(initial_train_sessions=300, refit_every=50))

    @pytest.mark.parametrize("kwargs", [{"initial_train_sessions": 0}, {"refit_every": 0}])
    def test_rejects_nonsense_settings(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="must be at least 1"):
            WalkForwardConfig(**kwargs)


class TestWalkForwardOutput:
    def test_one_call_per_tested_session(self, labelled: Dataset) -> None:
        calls = walk_forward(labelled, AlwaysUp, CONFIG)
        assert calls.index.equals(labelled.features.index[300:])
        assert list(calls.columns) == [*PREDICTION_COLUMNS, FOLD, TRAINED_THROUGH]
        assert calls[PREDICTED_UP].all()

    def test_records_which_model_made_each_call(self, labelled: Dataset) -> None:
        calls = walk_forward(labelled, AlwaysUp, CONFIG)
        index = labelled.features.index
        assert calls[FOLD].iloc[0] == 0
        assert calls[TRAINED_THROUGH].iloc[0] == index[299]
        assert calls[FOLD].iloc[50] == 1
        assert calls[TRAINED_THROUGH].iloc[50] == index[349]
        # Every call was made by a model trained strictly before that session.
        assert (calls[TRAINED_THROUGH] < calls.index).all()

    def test_a_fresh_predictor_is_built_for_every_fold(self, labelled: Dataset) -> None:
        built: list[AlwaysUp] = []

        def factory() -> AlwaysUp:
            built.append(AlwaysUp())
            return built[-1]

        walk_forward(labelled, factory, CONFIG)
        assert len(built) == len(make_folds(len(labelled), CONFIG))
        assert len({id(p) for p in built}) == len(built)

    def test_unlabelled_rows_are_refused(self, prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fully labelled"):
            walk_forward(build_dataset(prices), AlwaysUp, CONFIG)

    def test_predictor_returning_the_wrong_sessions_is_refused(self, labelled: Dataset) -> None:
        class Shifted(AlwaysUp):
            def predict(self, features: pd.DataFrame) -> pd.DataFrame:
                return super().predict(features.iloc[1:])

        with pytest.raises(ValueError, match="wrong columns or sessions"):
            walk_forward(labelled, Shifted, CONFIG)


class TestNoLookAhead:
    """Each of these fails if a model is given anything from its own future."""

    def test_spy_sees_only_sessions_before_the_ones_it_is_asked_about(
        self, labelled: Dataset
    ) -> None:
        log: list[dict[str, object]] = []
        walk_forward(labelled, lambda: Spy(log), CONFIG)
        index = labelled.features.index
        assert len(log) == len(make_folds(len(labelled), CONFIG))
        previous_rows = 0
        for entry in log:
            train = entry["train_sessions"]
            test = entry["test_sessions"]
            assert isinstance(train, pd.DatetimeIndex)
            assert isinstance(test, pd.DatetimeIndex)
            assert train.max() < test.min()
            assert entry["train_targets_missing"] == 0
            assert entry["train_index_matches_target"] is True
            # Expanding window: all rows before the first tested session, no more.
            assert entry["train_rows"] == index.get_loc(test.min())
            assert train.equals(index[: len(train)])
            assert isinstance(entry["train_rows"], int)
            assert entry["train_rows"] > previous_rows
            previous_rows = entry["train_rows"]

    def test_the_last_training_outcome_is_known_when_the_first_call_is_made(
        self, prices: pd.DataFrame, labelled: Dataset
    ) -> None:
        # The outcome on the last training row is about the session after it. That
        # session must be no later than the first session a call is made on.
        log: list[dict[str, object]] = []
        walk_forward(labelled, lambda: Spy(log), CONFIG)
        all_sessions = prices.index
        for entry in log:
            train, test = entry["train_sessions"], entry["test_sessions"]
            assert isinstance(train, pd.DatetimeIndex)
            assert isinstance(test, pd.DatetimeIndex)
            position = all_sessions.get_loc(train.max())
            assert isinstance(position, int)
            outcome_session = all_sessions[position + 1]
            assert outcome_session <= test.min()

    def test_a_predictor_that_memorises_outcomes_never_gets_to_use_them(
        self, labelled: Dataset
    ) -> None:
        Memorizer.recalled = 0
        calls = walk_forward(labelled, Memorizer, CONFIG)
        assert Memorizer.recalled == 0
        assert not calls[PREDICTED_UP].any()

    def test_the_memoriser_does_catch_a_leaky_layout(self, labelled: Dataset) -> None:
        # Same predictor, but bypassing the harness with a fold that trains on its
        # own test rows: it recalls every outcome and scores perfectly.
        Memorizer.recalled = 0
        cheat = Memorizer()
        cheat.fit(labelled.features, labelled.target_up)
        calls = cheat.predict(labelled.features.iloc[300:350])
        assert Memorizer.recalled == 50
        assert (calls[PREDICTED_UP] == (labelled.target_up.iloc[300:350] == 1.0)).all()

    @pytest.mark.parametrize(
        "fold",
        [
            Fold(number=0, train_end=350, test_start=300, test_end=350),
            Fold(number=0, train_end=301, test_start=300, test_end=350),
        ],
    )
    def test_a_fold_that_trains_on_its_test_rows_is_refused(
        self, labelled: Dataset, fold: Fold
    ) -> None:
        with pytest.raises(LookAheadError, match="trains on rows up to"):
            walk_forward(labelled, AlwaysUp, folds=[fold])

    def test_overlapping_test_blocks_are_refused(self, labelled: Dataset) -> None:
        folds = [
            Fold(number=0, train_end=300, test_start=300, test_end=350),
            Fold(number=1, train_end=340, test_start=340, test_end=400),
        ]
        with pytest.raises(LookAheadError, match="tested more than once"):
            walk_forward(labelled, AlwaysUp, folds=folds)

    @pytest.mark.parametrize("cut", [450, 600, 850])
    def test_corrupting_all_later_prices_does_not_change_earlier_calls(
        self, prices: pd.DataFrame, cut: int
    ) -> None:
        # Replace every session after `cut` with nonsense, rebuild everything from
        # the raw prices, and rerun a model that really learns. Its calls up to and
        # including session `cut` must be identical.
        rng = np.random.default_rng(5)
        corrupted = prices.copy()
        later = corrupted.index[cut + 1 :]
        for column in PRICE_COLUMNS:
            corrupted.loc[later, column] = rng.uniform(1_000, 9_000, len(later))
        corrupted.loc[later, "volume"] = rng.integers(1, 10, len(later))

        clean_calls = walk_forward(build_dataset(prices).labelled(), LeastSquaresLearner, CONFIG)
        dirty_calls = walk_forward(build_dataset(corrupted).labelled(), LeastSquaresLearner, CONFIG)

        through = prices.index[cut]
        columns = [PREDICTED_UP, PROBABILITY_UP]
        pd.testing.assert_frame_equal(
            dirty_calls.loc[:through, columns], clean_calls.loc[:through, columns], check_exact=True
        )
        # The learner is sensitive enough to notice: later calls do change.
        assert (
            not dirty_calls.loc[through:, columns]
            .iloc[1:]
            .equals(clean_calls.loc[through:, columns].iloc[1:])
        )

    def test_the_learner_really_depends_on_its_training_data(self, labelled: Dataset) -> None:
        # If flipping the training outcomes changed nothing, the test above would
        # be empty. It changes the calls.
        flipped = Dataset(labelled.features, 1.0 - labelled.target_up, labelled.next_return)
        normal = walk_forward(labelled, LeastSquaresLearner, CONFIG)[PREDICTED_UP]
        other = walk_forward(flipped, LeastSquaresLearner, CONFIG)[PREDICTED_UP]
        assert (normal != other).mean() > 0.5
