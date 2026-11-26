"""The paper strategy's rule, fixed on 2026-11-25 before any result was seen.

After each session's close the strategy is "Holding" NVDA for the next session or
"In cash". It is Holding only when both:

1. at least `MIN_MODELS_UP` of the six direction models in `RULE_MODELS` say up
   (no baselines; a model added later does not join the rule), and
2. the volatility gate is open: the simple baseline's (ewma_94) volatility forecast
   for the next session is not above the `GATE_PERCENTILE`th percentile of its
   values on the training rows, the rows the models were fitted on for that call.

The gate was chosen knowing that volatility is predictable, the one positive result
of the volatility backtest; the page says so. None of these settings is tuned.
tests/test_nvda_paper_strategy.py holds the same rule as tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

RULE_MODELS: tuple[str, ...] = (
    "logistic",
    "xgboost",
    "lstm",
    "logistic_ctx",
    "xgboost_ctx",
    "lstm_ctx",
)
MIN_MODELS_UP = 4
GATE_FORECASTER = "ewma_94"
GATE_PERCENTILE = 80.0

HOLDING = "Holding"
IN_CASH = "In cash"


def state_name(is_holding: bool) -> str:
    return HOLDING if is_holding else IN_CASH


def gate_threshold(history: np.ndarray) -> float:
    """The gate's threshold: the `GATE_PERCENTILE`th percentile of past forecasts."""
    values = np.asarray(history, dtype="float64")
    if values.size == 0:
        raise ValueError("the volatility gate needs a history of forecasts")
    if not bool(np.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("the volatility history must be finite and above zero")
    return float(np.percentile(values, GATE_PERCENTILE))


def gate_open(forecast: float, threshold: float) -> bool:
    """Open unless the forecast is above the threshold."""
    return bool(forecast <= threshold)


def majority_up(calls: Mapping[str, bool]) -> bool:
    """At least `MIN_MODELS_UP` of the rule models say up. Other calls are ignored."""
    _check_complete(calls)
    return sum(bool(calls[name]) for name in RULE_MODELS) >= MIN_MODELS_UP


def holding(calls: Mapping[str, bool], forecast: float, threshold: float) -> bool:
    """The rule for one session: Holding (True) or In cash (False)."""
    return majority_up(calls) and gate_open(forecast, threshold)


def majority_series(calls: Mapping[str, np.ndarray]) -> np.ndarray:
    """`majority_up` for many sessions at once; each array holds one model's calls."""
    _check_complete(calls)
    votes = sum(np.asarray(calls[name], dtype=bool).astype("int64") for name in RULE_MODELS)
    return np.asarray(votes >= MIN_MODELS_UP, dtype=bool)


def gate_series(forecast: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """`gate_open` for many sessions at once."""
    return np.asarray(
        np.asarray(forecast, dtype="float64") <= np.asarray(threshold, dtype="float64"),
        dtype=bool,
    )


def holding_series(
    calls: Mapping[str, np.ndarray], forecast: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """`holding` for many sessions at once."""
    return majority_series(calls) & gate_series(forecast, threshold)


def _check_complete(calls: Mapping[str, object]) -> None:
    missing = [name for name in RULE_MODELS if name not in calls]
    if missing:
        raise ValueError(f"the rule needs a call from every one of its models; missing {missing}")
