"""Accuracy, simulated returns with costs, and confidence intervals.

Simulated strategy (the approved definition): hold the stock for the next session
when the call is up, hold cash when it is down. Cash earns nothing here, which is a
simplification. A cost is charged on the traded value every time the position
changes, including the first purchase; the final position is not sold at the end.

Intervals:

- Accuracy: Wilson score interval. It treats sessions as independent.
- Returns, and every difference between two predictors: a paired moving-block
  bootstrap. Whole blocks of consecutive sessions are resampled, the same blocks
  for both predictors, which keeps day-to-day dependence and the pairing intact.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

SESSIONS_PER_YEAR = 252
BASIS_POINT = 1e-4


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Settings for the block bootstrap. The seed makes every interval repeatable."""

    block_length: int = 21
    n_resamples: int = 2_000
    confidence: float = 0.95
    seed: int = 20_261_119

    def __post_init__(self) -> None:
        if self.block_length < 1 or self.n_resamples < 1:
            raise ValueError("block_length and n_resamples must be at least 1")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion of `successes` out of `n`."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    share = successes / n
    denominator = 1 + z * z / n
    centre = (share + z * z / (2 * n)) / denominator
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / denominator
    # At the extremes the bound is exactly 0 or 1; say so rather than 1e-17.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def hits(predicted_up: np.ndarray, target_up: np.ndarray) -> np.ndarray:
    """True where the call matched what happened."""
    predicted = np.asarray(predicted_up)
    target = np.asarray(target_up, dtype="float64")
    if predicted.dtype != np.bool_ or predicted.shape != target.shape:
        raise ValueError("predicted_up must be boolean and the same length as target_up")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target_up must be 0 or 1 with no missing values")
    return predicted == (target == 1.0)


def strategy_net_returns(
    predicted_up: np.ndarray, next_return: np.ndarray, *, cost_bps: float
) -> np.ndarray:
    """Daily net returns of holding the stock on up calls and cash otherwise.

    Entry *t* is the return over session *t+1*, after a cost of `cost_bps` basis
    points of the traded value whenever the position differs from the one before.
    The position before the first call is cash.
    """
    position = _positions(predicted_up, next_return)
    if cost_bps < 0:
        raise ValueError("cost_bps must not be negative")
    previous = np.concatenate(([0.0], position[:-1]))
    traded = np.abs(position - previous)
    growth = (1.0 - cost_bps * BASIS_POINT * traded) * (1.0 + position * next_return)
    return growth - 1.0


def trade_count(predicted_up: np.ndarray) -> int:
    """How many times the position changes, counting the first purchase."""
    position = np.asarray(predicted_up, dtype="float64")
    previous = np.concatenate(([0.0], position[:-1]))
    return int(np.abs(position - previous).sum())


def total_return(net_returns: np.ndarray) -> float:
    """Compounded return over the whole period."""
    return float(np.expm1(np.log1p(net_returns).sum()))


def annualised_return(net_returns: np.ndarray) -> float:
    """Compounded return per year of `SESSIONS_PER_YEAR` sessions."""
    return float(np.expm1(np.log1p(net_returns).mean() * SESSIONS_PER_YEAR))


def annualised_volatility(net_returns: np.ndarray) -> float:
    """Standard deviation of daily returns, scaled to a year."""
    if len(net_returns) < 2:
        return math.nan
    return float(np.std(net_returns, ddof=1) * math.sqrt(SESSIONS_PER_YEAR))


def sharpe_ratio(net_returns: np.ndarray) -> float:
    """Mean daily return over its standard deviation, annualised. Cash earns zero."""
    if len(net_returns) < 2:
        return math.nan
    spread = float(np.std(net_returns, ddof=1))
    if spread == 0:
        return math.nan
    return float(np.mean(net_returns) / spread * math.sqrt(SESSIONS_PER_YEAR))


def block_bootstrap_indices(
    n: int, *, block_length: int, n_resamples: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield `n_resamples` index arrays of length `n` made of whole blocks.

    Blocks start at random positions and wrap around the end (circular), so every
    session is equally likely to be drawn.
    """
    if n < 1:
        raise ValueError("nothing to resample")
    length = min(block_length, n)
    blocks = math.ceil(n / length)
    offsets = np.arange(length)
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=blocks)
        yield ((starts[:, None] + offsets) % n).ravel()[:n]


def bootstrap_interval(
    n: int, statistic: Callable[[np.ndarray], float], config: BootstrapConfig
) -> tuple[float, float]:
    """Percentile interval for `statistic`, which takes resampled row positions.

    Pass a statistic that indexes every series it needs with the same positions;
    that is what keeps two predictors paired.
    """
    rng = np.random.default_rng(config.seed)
    draws = np.fromiter(
        (
            statistic(positions)
            for positions in block_bootstrap_indices(
                n, block_length=config.block_length, n_resamples=config.n_resamples, rng=rng
            )
        ),
        dtype="float64",
        count=config.n_resamples,
    )
    tail = (1 - config.confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return float(low), float(high)


def _positions(predicted_up: np.ndarray, next_return: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_up)
    returns = np.asarray(next_return, dtype="float64")
    if predicted.dtype != np.bool_ or predicted.shape != returns.shape or predicted.ndim != 1:
        raise ValueError("predicted_up must be boolean and the same length as next_return")
    if np.isnan(returns).any():
        raise ValueError("next_return has missing values")
    if (returns <= -1).any():
        raise ValueError("next_return has a loss of 100% or more")
    return predicted.astype("float64")
