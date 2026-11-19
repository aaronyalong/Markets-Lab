"""The naive baselines and the predictor interface they share."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.baselines import (
    AlwaysUp,
    MovingAverageRule,
    Persistence,
    default_baselines,
)
from markets_lab.analysis.nvda.features import Dataset, build_dataset
from markets_lab.analysis.nvda.predictors import (
    PREDICTED_UP,
    PREDICTION_COLUMNS,
    PROBABILITY_UP,
    Predictor,
    prediction_frame,
)
from price_frames import make_prices


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_dataset(make_prices(rows=420))


def features_with(**columns: list[float]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    index = pd.bdate_range("2030-01-07", periods=length, name="date")
    return pd.DataFrame(columns, index=index)


class TestAlwaysUp:
    def test_calls_up_every_session_with_no_probability(self, dataset: Dataset) -> None:
        calls = AlwaysUp().predict(dataset.features)
        assert calls[PREDICTED_UP].all()
        assert calls[PROBABILITY_UP].isna().all()
        assert calls.index.equals(dataset.features.index)
        assert tuple(calls.columns) == PREDICTION_COLUMNS

    def test_accuracy_is_the_share_of_up_days(self, dataset: Dataset) -> None:
        labelled = dataset.labelled()
        calls = AlwaysUp().predict(labelled.features)
        accuracy = (calls[PREDICTED_UP] == (labelled.target_up == 1.0)).mean()
        assert accuracy == pytest.approx(labelled.target_up.mean())


class TestPersistence:
    def test_repeats_the_last_sessions_direction(self) -> None:
        features = features_with(ret_1=[0.02, -0.01, 0.0, 1e-9, -1e-9])
        calls = Persistence().predict(features)
        assert calls[PREDICTED_UP].tolist() == [True, False, False, True, False]

    def test_matches_the_previous_label_on_real_shaped_data(self, dataset: Dataset) -> None:
        # Today's call equals yesterday's realised label: "same as last time".
        calls = Persistence().predict(dataset.features)[PREDICTED_UP]
        previous_label = dataset.target_up.shift(1) == 1.0
        assert (calls.iloc[1:] == previous_label.iloc[1:]).all()


class TestMovingAverageRule:
    def test_up_only_when_price_is_above_the_average(self) -> None:
        features = features_with(sma_200_gap=[0.10, -0.05, 0.0], sma_50_gap=[-0.10, 0.05, 0.0])
        assert MovingAverageRule().predict(features)[PREDICTED_UP].tolist() == [True, False, False]
        assert MovingAverageRule(50).predict(features)[PREDICTED_UP].tolist() == [
            False,
            True,
            False,
        ]

    def test_default_window_is_200_and_named(self) -> None:
        assert MovingAverageRule().name == "ma_200"
        assert MovingAverageRule(50).name == "ma_50"

    def test_only_windows_that_exist_as_features(self) -> None:
        with pytest.raises(ValueError, match="window must be one of"):
            MovingAverageRule(123)


class TestSharedBehaviour:
    def test_default_baselines_have_distinct_fixed_names(self) -> None:
        assert [b.name for b in default_baselines()] == ["always_up", "persistence", "ma_200"]

    @pytest.mark.parametrize("position", range(3))
    def test_a_call_uses_only_its_own_sessions_features(
        self, dataset: Dataset, position: int
    ) -> None:
        # Calls on the first k sessions are the same with or without later sessions.
        baseline = default_baselines()[position]
        full = baseline.predict(dataset.features)
        for cut in (1, 50, len(dataset) - 1):
            pd.testing.assert_frame_equal(baseline.predict(dataset.features.iloc[:cut]), full[:cut])

    @pytest.mark.parametrize("position", range(3))
    def test_fitting_changes_nothing_so_labels_cannot_leak_in(
        self, dataset: Dataset, position: int
    ) -> None:
        labelled = dataset.labelled()
        baseline = default_baselines()[position]
        before = baseline.predict(dataset.features)
        baseline.fit(labelled.features, labelled.target_up)
        baseline.fit(labelled.features, 1.0 - labelled.target_up)
        pd.testing.assert_frame_equal(baseline.predict(dataset.features), before)

    @pytest.mark.parametrize("position", range(3))
    def test_no_rows_gives_no_calls(self, dataset: Dataset, position: int) -> None:
        calls = default_baselines()[position].predict(dataset.features.iloc[:0])
        assert len(calls) == 0
        assert tuple(calls.columns) == PREDICTION_COLUMNS

    def test_missing_feature_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing the column 'ret_1'"):
            Persistence().predict(features_with(sma_200_gap=[0.1]))

    def test_missing_feature_values_raise(self) -> None:
        with pytest.raises(ValueError, match="has missing values"):
            Persistence().predict(features_with(ret_1=[0.1, np.nan]))

    def test_baselines_satisfy_the_predictor_interface(self, dataset: Dataset) -> None:
        def run(predictor: Predictor) -> pd.DataFrame:
            labelled = dataset.labelled()
            predictor.fit(labelled.features, labelled.target_up)
            return predictor.predict(dataset.features)

        for baseline in (AlwaysUp(), Persistence(), MovingAverageRule()):
            assert run(baseline)[PREDICTED_UP].dtype == bool


class TestPredictionFrame:
    INDEX = pd.bdate_range("2030-01-07", periods=3, name="date")

    def test_direction_only(self) -> None:
        frame = prediction_frame(self.INDEX, np.array([True, False, True]))
        assert frame[PREDICTED_UP].tolist() == [True, False, True]
        assert frame[PROBABILITY_UP].isna().all()

    def test_with_probabilities(self) -> None:
        frame = prediction_frame(
            self.INDEX, np.array([True, False, True]), np.array([0.7, 0.2, np.nan])
        )
        assert frame[PROBABILITY_UP].iloc[:2].tolist() == [0.7, 0.2]

    @pytest.mark.parametrize(
        ("up", "probability"),
        [
            (np.array([1, 0, 1]), None),
            (np.array([True, False]), None),
            (np.array([True, False, True]), np.array([0.5, 0.5])),
            (np.array([True, False, True]), np.array([0.5, 1.2, 0.1])),
            (np.array([True, False, True]), np.array([-0.1, 0.5, 0.1])),
        ],
    )
    def test_rejects_wrong_shapes_types_and_ranges(
        self, up: np.ndarray, probability: np.ndarray | None
    ) -> None:
        with pytest.raises(ValueError, match="must be"):
            prediction_frame(self.INDEX, up, probability)
