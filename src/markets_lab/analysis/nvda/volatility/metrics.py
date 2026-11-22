"""How volatility forecasts and their ranges are scored. Each returns one loss per
session, so two forecasters can be compared on the same sessions.

- QLIKE scores the volatility forecast against the squared return that followed.
  A squared return is a very noisy measure of that day's true volatility, and QLIKE
  is one of the few losses that still ranks forecasters correctly when it is used.
  Lower is better. Its level means little; differences between forecasters do.
- The interval score scores the range. It is the width of the range, plus a
  penalty whenever the outcome falls outside, in proportion to how far outside. A
  narrow range that still holds 80% of outcomes scores best. Lower is better.
- Coverage is the share of outcomes inside the range. It should be close to the
  level the range was built for, with the misses split evenly above and below.
"""

from __future__ import annotations

import numpy as np


def qlike_losses(volatility: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """QLIKE loss per session: log(variance) + squared return / variance."""
    volatility, outcome = _aligned(volatility, outcome)
    if (volatility <= 0).any():
        raise ValueError("volatility must be positive")
    variance = volatility**2
    return np.log(variance) + outcome**2 / variance


def interval_scores(
    lower: np.ndarray, upper: np.ndarray, outcome: np.ndarray, level: float
) -> np.ndarray:
    """Interval (Winkler) score per session for a central range at this level."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    lower, outcome = _aligned(lower, outcome)
    upper, _ = _aligned(upper, outcome)
    if (lower >= upper).any():
        raise ValueError("the lower bound must be below the upper bound")
    penalty = 2 / (1 - level)
    below = np.clip(lower - outcome, 0.0, None)
    above = np.clip(outcome - upper, 0.0, None)
    return (upper - lower) + penalty * (below + above)


def inside(lower: np.ndarray, upper: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """True where the outcome fell inside the range, bounds included."""
    lower, outcome = _aligned(lower, outcome)
    upper, _ = _aligned(upper, outcome)
    return (outcome >= lower) & (outcome <= upper)


def _aligned(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(first, dtype="float64")
    b = np.asarray(second, dtype="float64")
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("inputs must be one-dimensional and the same length")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must be finite")
    return a, b
