"""LSTM specifics. The rules shared by all models are in test_nvda_model_contract.py."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from markets_lab.analysis.nvda.models import lstm
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.models.lstm import SEQUENCE_LENGTH, LSTMModel
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, PROBABILITY_UP, Predictor

SPLIT = 2_000


def yesterdays_sign(rows: int = 3_000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """The outcome is up when feature a was positive one session earlier.

    The current session's features say nothing about it; only the sequence does.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2030-01-07", periods=rows, name="date")
    features = pd.DataFrame(rng.normal(0, 1, (rows, 4)), index=index, columns=["a", "b", "c", "d"])
    target = (features["a"].shift(1) > 0).astype("float64")
    return features, target


def accuracy(model: Predictor, features: pd.DataFrame, target: pd.Series) -> float:
    model.fit(features.iloc[:SPLIT], target.iloc[:SPLIT])
    calls = model.predict(features.iloc[SPLIT:])
    return float((calls[PREDICTED_UP] == (target.iloc[SPLIT:] == 1.0)).mean())


@pytest.fixture(scope="module")
def sequence_data() -> tuple[pd.DataFrame, pd.Series]:
    return yesterdays_sign()


@pytest.fixture(scope="module")
def fitted(sequence_data: tuple[pd.DataFrame, pd.Series]) -> LSTMModel:
    features, target = sequence_data
    model = LSTMModel()
    model.fit(features.iloc[:SPLIT], target.iloc[:SPLIT])
    return model


class TestLearning:
    def test_learns_a_pattern_that_only_exists_across_sessions(
        self, sequence_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = sequence_data
        assert accuracy(LSTMModel(), features, target) > 0.85
        assert accuracy(LogisticRegressionModel(), features, target) < 0.58

    def test_finds_nothing_in_noise(self, sequence_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, _ = sequence_data
        noise = pd.Series(
            (np.random.default_rng(9).random(len(features)) > 0.5).astype("float64"),
            index=features.index,
        )
        assert 0.44 < accuracy(LSTMModel(), features, noise) < 0.56


class TestWindows:
    """Where look-ahead could creep into a sequence model."""

    def test_the_first_call_reaches_back_into_the_end_of_the_training_sessions(
        self, sequence_data: tuple[pd.DataFrame, pd.Series], fitted: LSTMModel
    ) -> None:
        # The first predicted session's outcome depends on feature a of the last
        # training session. Overwrite that one remembered value: the first call must
        # follow it, which shows the window is aligned to the session just before.
        features, _ = sequence_data
        first = features.iloc[SPLIT : SPLIT + 1]
        probabilities = {}
        for value in (-3.0, 3.0):
            model = copy.deepcopy(fitted)
            model._tail[-1, 0] = value
            probabilities[value] = float(model.predict(first)[PROBABILITY_UP].iloc[0])
        assert probabilities[-3.0] < 0.2
        assert probabilities[3.0] > 0.8

    def test_changing_a_later_session_never_changes_an_earlier_call(
        self, sequence_data: tuple[pd.DataFrame, pd.Series], fitted: LSTMModel
    ) -> None:
        features, _ = sequence_data
        batch = features.iloc[SPLIT : SPLIT + 60]
        base = fitted.predict(batch)[PROBABILITY_UP].to_numpy()
        for changed_row in (59, 30, 10, 1):
            altered = batch.copy()
            altered.iloc[changed_row] = 50.0
            after = fitted.predict(altered)[PROBABILITY_UP].to_numpy()
            assert after[:changed_row] == pytest.approx(base[:changed_row], abs=1e-6)
            assert after[changed_row] != pytest.approx(base[changed_row], abs=1e-6)

    def test_changing_an_earlier_session_does_change_later_calls_within_the_window(
        self, sequence_data: tuple[pd.DataFrame, pd.Series], fitted: LSTMModel
    ) -> None:
        features, _ = sequence_data
        batch = features.iloc[SPLIT : SPLIT + 60]
        base = fitted.predict(batch)[PROBABILITY_UP].to_numpy()
        altered = batch.copy()
        altered.iloc[10] = 50.0
        after = fitted.predict(altered)[PROBABILITY_UP].to_numpy()
        inside = slice(10, 10 + SEQUENCE_LENGTH)
        beyond = slice(10 + SEQUENCE_LENGTH, None)
        assert np.abs(after[inside] - base[inside]).max() > 1e-3
        # Sessions more than a window later no longer see the changed session.
        assert after[beyond] == pytest.approx(base[beyond], abs=1e-6)

    def test_sessions_it_was_trained_on_cannot_be_predicted(
        self, sequence_data: tuple[pd.DataFrame, pd.Series], fitted: LSTMModel
    ) -> None:
        features, _ = sequence_data
        with pytest.raises(ValueError, match="only predict sessions after"):
            fitted.predict(features.iloc[SPLIT - 5 : SPLIT + 5])
        with pytest.raises(ValueError, match="only predict sessions after"):
            fitted.predict(features.iloc[:10])

    def test_sessions_must_be_in_order(
        self, sequence_data: tuple[pd.DataFrame, pd.Series], fitted: LSTMModel
    ) -> None:
        features, target = sequence_data
        with pytest.raises(ValueError, match="increasing order"):
            fitted.predict(features.iloc[SPLIT : SPLIT + 10].iloc[::-1])
        with pytest.raises(ValueError, match="increasing order"):
            LSTMModel().fit(features.iloc[:100].iloc[::-1], target.iloc[:100].iloc[::-1])

    def test_too_little_training_data_raises(
        self, sequence_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = sequence_data
        with pytest.raises(ValueError, match="at least 40 training sessions"):
            LSTMModel().fit(features.iloc[:39], target.iloc[:39])
        LSTMModel().fit(features.iloc[:40], target.iloc[:40])


class TestReproducibility:
    def test_fitting_leaves_torchs_global_settings_as_they_were(
        self, sequence_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = sequence_data
        torch.manual_seed(1234)
        threads_before = torch.get_num_threads()
        expected_next = torch.rand(3)
        torch.manual_seed(1234)
        model = LSTMModel()
        model.fit(features.iloc[:200], target.iloc[:200])
        model.predict(features.iloc[200:210])
        assert torch.equal(torch.rand(3), expected_next)
        assert torch.get_num_threads() == threads_before

    def test_settings_are_the_ones_fixed_in_advance(self) -> None:
        # Changing these after seeing results would be tuning on the test data.
        assert (lstm.SEQUENCE_LENGTH, lstm.HIDDEN_SIZE, lstm.EPOCHS) == (20, 32, 10)
        assert (lstm.BATCH_SIZE, lstm.LEARNING_RATE, lstm.WEIGHT_DECAY) == (64, 1e-3, 1e-4)
        assert lstm.SEED == 0
