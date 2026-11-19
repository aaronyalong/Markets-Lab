"""XGBoost specifics. The rules shared by all models are in
test_nvda_model_contract.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.models import xgb
from markets_lab.analysis.nvda.models.logistic import LogisticRegressionModel
from markets_lab.analysis.nvda.models.xgb import XGBoostModel
from markets_lab.analysis.nvda.predictors import PREDICTED_UP, Predictor


def interaction(rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """The outcome is up when a and b have the same sign: no single feature helps."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2030-01-07", periods=rows, name="date")
    features = pd.DataFrame(rng.normal(0, 1, (rows, 4)), index=index, columns=["a", "b", "c", "d"])
    return features, ((features["a"] * features["b"]) > 0).astype("float64")


def accuracy(model: Predictor, features: pd.DataFrame, target: pd.Series, split: int) -> float:
    model.fit(features.iloc[:split], target.iloc[:split])
    calls = model.predict(features.iloc[split:])
    return float((calls[PREDICTED_UP] == (target.iloc[split:] == 1.0)).mean())


def test_learns_an_interaction_that_a_linear_model_cannot() -> None:
    features, target = interaction(3_000)
    assert accuracy(XGBoostModel(), features, target, 2_000) > 0.90
    assert accuracy(LogisticRegressionModel(), features, target, 2_000) < 0.60


def test_finds_nothing_in_noise() -> None:
    features, _ = interaction(3_000)
    noise = pd.Series(
        (np.random.default_rng(9).random(3_000) > 0.5).astype("float64"), index=features.index
    )
    assert 0.44 < accuracy(XGBoostModel(), features, noise, 2_000) < 0.56


def test_feature_scale_does_not_matter() -> None:
    features, target = interaction(1_500)
    rescaled = features * [1.0, 5_000.0, 0.001, 1.0]
    plain = XGBoostModel()
    plain.fit(features.iloc[:1_000], target.iloc[:1_000])
    scaled = XGBoostModel()
    scaled.fit(rescaled.iloc[:1_000], target.iloc[:1_000])
    same = (
        plain.predict(features.iloc[1_000:])[PREDICTED_UP]
        == scaled.predict(rescaled.iloc[1_000:])[PREDICTED_UP]
    ).mean()
    assert same > 0.97


def test_importance_points_at_the_features_that_matter() -> None:
    features, target = interaction(2_000)
    model = XGBoostModel()
    model.fit(features, target)
    importance = model.feature_importance()
    assert list(importance.index) == ["a", "b", "c", "d"]
    assert importance.sum() == pytest.approx(1.0, abs=1e-5)
    assert importance["a"] + importance["b"] > 0.8


def test_importance_needs_a_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        XGBoostModel().feature_importance()


def test_settings_are_the_ones_fixed_in_advance() -> None:
    # Changing these after seeing results would be tuning on the test data.
    assert (xgb.N_TREES, xgb.MAX_DEPTH, xgb.LEARNING_RATE) == (200, 3, 0.05)
    assert (xgb.ROW_SAMPLE, xgb.FEATURE_SAMPLE, xgb.SEED) == (0.8, 0.8, 0)
