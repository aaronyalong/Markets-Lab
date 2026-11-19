"""Logistic regression specifics. The rules shared by all models are in
test_nvda_model_contract.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.predictors import PREDICTED_UP


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


def test_learns_a_linear_pattern_that_is_there() -> None:
    features, target = learnable(1_200)
    model = LogisticRegressionModel()
    model.fit(features.iloc[:800], target.iloc[:800])
    calls = model.predict(features.iloc[800:])
    accuracy = (calls[PREDICTED_UP] == (target.iloc[800:] == 1.0)).mean()
    assert accuracy > 0.95


def test_finds_nothing_in_noise() -> None:
    features, _ = learnable(3_000)
    noise = pd.Series(
        (np.random.default_rng(9).random(3_000) > 0.5).astype("float64"), index=features.index
    )
    model = LogisticRegressionModel()
    model.fit(features.iloc[:2_000], noise.iloc[:2_000])
    calls = model.predict(features.iloc[2_000:])
    accuracy = (calls[PREDICTED_UP] == (noise.iloc[2_000:] == 1.0)).mean()
    assert 0.44 < accuracy < 0.56


def test_feature_scale_does_not_matter() -> None:
    # Column b is 5,000 times larger than column c; standardising handles it.
    features, target = learnable(1_000)
    model = LogisticRegressionModel()
    model.fit(features.iloc[:700], target.iloc[:700])
    weights = model.coefficients()
    assert abs(weights["a"]) > 10 * abs(weights["c"])
    assert abs(weights["b"]) > 10 * abs(weights["c"])
    assert list(weights.index) == ["a", "b", "c", "d"]


def test_scaling_is_learned_from_training_rows_and_not_changed_by_predicting() -> None:
    features, target = learnable(600)
    model = LogisticRegressionModel()
    model.fit(features.iloc[:400], target.iloc[:400])
    before = model.coefficients().copy()
    model.predict(features.iloc[400:] * 1_000.0)
    pd.testing.assert_series_equal(model.coefficients(), before, check_exact=True)


def test_coefficients_need_a_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        LogisticRegressionModel().coefficients()
