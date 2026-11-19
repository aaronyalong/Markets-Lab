"""Accuracy, simulated returns with costs, and the confidence intervals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from markets_lab.analysis.nvda.metrics import (
    BootstrapConfig,
    annualised_return,
    annualised_volatility,
    block_bootstrap_indices,
    bootstrap_interval,
    hits,
    sharpe_ratio,
    strategy_net_returns,
    total_return,
    trade_count,
    wilson_interval,
)

UP, DOWN = True, False
QUICK = BootstrapConfig(n_resamples=300, block_length=5)


def calls(*values: bool) -> np.ndarray:
    return np.array(values, dtype=bool)


class TestWilsonInterval:
    def test_known_value(self) -> None:
        low, high = wilson_interval(50, 100)
        assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)

    def test_narrows_with_more_data_and_contains_the_estimate(self) -> None:
        small = wilson_interval(55, 100)
        large = wilson_interval(5_500, 10_000)
        assert small[0] < 0.55 < small[1]
        assert large[0] < 0.55 < large[1]
        assert large[1] - large[0] < small[1] - small[0]

    def test_stays_inside_zero_and_one_at_the_extremes(self) -> None:
        assert wilson_interval(0, 20)[0] == 0.0
        assert wilson_interval(20, 20)[1] == 1.0
        assert wilson_interval(0, 20)[1] > 0.0
        assert wilson_interval(20, 20)[0] < 1.0

    def test_higher_confidence_is_wider(self) -> None:
        narrow, wide = wilson_interval(60, 100, 0.90), wilson_interval(60, 100, 0.99)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]

    def test_about_385_sessions_are_needed_to_tell_55_from_50(self) -> None:
        # The claim made on the site's track-record section.
        assert wilson_interval(round(0.55 * 385), 385)[0] > 0.499
        assert wilson_interval(round(0.55 * 300), 300)[0] < 0.50

    @pytest.mark.parametrize(
        ("successes", "n", "confidence"), [(1, 0, 0.95), (5, 4, 0.95), (-1, 4, 0.95), (1, 4, 1.0)]
    )
    def test_rejects_impossible_inputs(self, successes: int, n: int, confidence: float) -> None:
        with pytest.raises(ValueError, match="must be"):
            wilson_interval(successes, n, confidence)


class TestHits:
    def test_a_call_is_right_when_it_matches_the_outcome(self) -> None:
        result = hits(calls(UP, UP, DOWN, DOWN), np.array([1.0, 0.0, 0.0, 1.0]))
        assert result.tolist() == [True, False, True, False]

    @pytest.mark.parametrize(
        ("predicted", "target"),
        [
            (np.array([1, 0]), np.array([1.0, 0.0])),
            (calls(UP), np.array([1.0, 0.0])),
            (calls(UP, DOWN), np.array([1.0, np.nan])),
            (calls(UP, DOWN), np.array([1.0, 0.5])),
        ],
    )
    def test_rejects_bad_inputs(self, predicted: np.ndarray, target: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must be"):
            hits(predicted, target)


class TestStrategyReturns:
    RETURNS = np.array([0.10, -0.05, 0.02, 0.03])

    def test_long_on_up_calls_and_cash_on_down_calls(self) -> None:
        net = strategy_net_returns(calls(UP, DOWN, UP, UP), self.RETURNS, cost_bps=0)
        assert net.tolist() == pytest.approx([0.10, 0.0, 0.02, 0.03])

    def test_cost_is_charged_on_every_position_change_including_the_first(self) -> None:
        cost = 0.0005  # 5 basis points
        net = strategy_net_returns(calls(UP, DOWN, UP, UP), self.RETURNS, cost_bps=5)
        assert net.tolist() == pytest.approx(
            [
                (1 - cost) * 1.10 - 1,  # buy
                (1 - cost) * 1.00 - 1,  # sell, then hold cash
                (1 - cost) * 1.02 - 1,  # buy again
                0.03,  # still holding, no trade
            ]
        )
        assert trade_count(calls(UP, DOWN, UP, UP)) == 3

    def test_always_up_is_buy_and_hold_with_one_purchase(self) -> None:
        always = calls(UP, UP, UP, UP)
        gross = total_return(strategy_net_returns(always, self.RETURNS, cost_bps=0))
        assert gross == pytest.approx(float(np.prod(1 + self.RETURNS) - 1))
        net = total_return(strategy_net_returns(always, self.RETURNS, cost_bps=5))
        assert net == pytest.approx((1 - 0.0005) * (1 + gross) - 1)
        assert trade_count(always) == 1

    def test_never_holding_earns_nothing_and_costs_nothing(self) -> None:
        net = strategy_net_returns(calls(DOWN, DOWN, DOWN, DOWN), self.RETURNS, cost_bps=10)
        assert net.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert trade_count(calls(DOWN, DOWN, DOWN, DOWN)) == 0

    def test_higher_costs_never_help(self) -> None:
        rng = np.random.default_rng(1)
        predicted = rng.random(500) > 0.5
        returns = rng.normal(0.001, 0.02, 500)
        totals = [
            total_return(strategy_net_returns(predicted, returns, cost_bps=c)) for c in (0, 5, 10)
        ]
        assert totals[0] > totals[1] > totals[2]

    def test_a_perfect_caller_beats_buy_and_hold_and_an_inverted_one_loses(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.normal(0.0, 0.02, 500)
        perfect = total_return(strategy_net_returns(returns > 0, returns, cost_bps=5))
        hold = total_return(strategy_net_returns(np.ones(500, dtype=bool), returns, cost_bps=5))
        wrong = total_return(strategy_net_returns(returns <= 0, returns, cost_bps=5))
        assert perfect > hold > wrong

    @pytest.mark.parametrize(
        ("predicted", "returns", "cost"),
        [
            (np.array([1, 0]), np.array([0.1, 0.1]), 5),
            (calls(UP), np.array([0.1, 0.1]), 5),
            (calls(UP, DOWN), np.array([0.1, np.nan]), 5),
            (calls(UP, DOWN), np.array([0.1, -1.0]), 5),
            (calls(UP, DOWN), np.array([0.1, 0.1]), -1),
        ],
    )
    def test_rejects_bad_inputs(
        self, predicted: np.ndarray, returns: np.ndarray, cost: float
    ) -> None:
        with pytest.raises(ValueError, match=r"must|missing|loss"):
            strategy_net_returns(predicted, returns, cost_bps=cost)


class TestReturnSummaries:
    def test_total_and_annualised(self) -> None:
        steady = np.full(252, 0.001)
        assert total_return(steady) == pytest.approx(1.001**252 - 1)
        assert annualised_return(steady) == pytest.approx(1.001**252 - 1)
        assert annualised_return(np.full(504, 0.001)) == pytest.approx(1.001**252 - 1)

    def test_volatility_and_sharpe(self) -> None:
        returns = np.array([0.01, -0.01] * 126)
        assert annualised_volatility(returns) == pytest.approx(
            np.std(returns, ddof=1) * math.sqrt(252)
        )
        assert sharpe_ratio(returns) == pytest.approx(0.0, abs=1e-12)
        assert sharpe_ratio(returns + 0.001) > 0

    def test_sharpe_is_undefined_without_variation(self) -> None:
        assert math.isnan(sharpe_ratio(np.zeros(10)))
        assert math.isnan(sharpe_ratio(np.array([0.01])))
        assert math.isnan(annualised_volatility(np.array([0.01])))


class TestBlockBootstrap:
    def test_indices_are_whole_wrapped_blocks(self) -> None:
        rng = np.random.default_rng(0)
        (positions,) = block_bootstrap_indices(10, block_length=4, n_resamples=1, rng=rng)
        assert positions.shape == (10,)
        assert ((positions >= 0) & (positions < 10)).all()
        for start in (0, 4):
            block = positions[start : start + 4]
            assert ((block - block[0]) % 10 == np.arange(4)).all()

    def test_block_longer_than_the_data_is_capped(self) -> None:
        rng = np.random.default_rng(0)
        (positions,) = block_bootstrap_indices(5, block_length=21, n_resamples=1, rng=rng)
        assert sorted(positions.tolist()) == [0, 1, 2, 3, 4]

    def test_same_seed_gives_the_same_interval(self) -> None:
        data = np.random.default_rng(3).normal(0, 1, 400)
        first = bootstrap_interval(400, lambda rows: float(data[rows].mean()), QUICK)
        second = bootstrap_interval(400, lambda rows: float(data[rows].mean()), QUICK)
        assert first == second
        assert first[0] < float(data.mean()) < first[1]

    def test_a_series_compared_with_itself_has_no_difference(self) -> None:
        data = np.random.default_rng(4).normal(0, 1, 300)
        low, high = bootstrap_interval(
            300, lambda rows: float((data[rows] - data[rows]).mean()), QUICK
        )
        assert (low, high) == (0.0, 0.0)

    def test_pairing_detects_a_small_but_consistent_edge(self) -> None:
        # B beats A by a little on every single day. Paired, the difference is
        # certain; the interval must sit strictly above zero.
        rng = np.random.default_rng(5)
        a = rng.normal(0, 0.02, 500)
        b = a + 0.0005
        low, high = bootstrap_interval(500, lambda rows: float((b[rows] - a[rows]).mean()), QUICK)
        assert low == pytest.approx(0.0005)
        assert high == pytest.approx(0.0005)

    def test_noise_alone_gives_an_interval_that_includes_zero(self) -> None:
        rng = np.random.default_rng(6)
        a, b = rng.normal(0, 0.02, 2_000), rng.normal(0, 0.02, 2_000)
        low, high = bootstrap_interval(2_000, lambda rows: float((b[rows] - a[rows]).mean()), QUICK)
        assert low < 0 < high

    def test_block_resampling_is_wider_than_independent_when_days_are_correlated(self) -> None:
        # A slowly drifting series: treating days as independent understates the
        # uncertainty of its mean. Longer blocks must give a wider interval.
        rng = np.random.default_rng(7)
        drifting = np.cumsum(rng.normal(0, 1, 3_000)) / 50

        def width(block_length: int) -> float:
            config = BootstrapConfig(n_resamples=400, block_length=block_length)
            low, high = bootstrap_interval(3_000, lambda rows: float(drifting[rows].mean()), config)
            return high - low

        assert width(63) > 2 * width(1)

    @pytest.mark.parametrize(
        "kwargs", [{"block_length": 0}, {"n_resamples": 0}, {"confidence": 1.0}, {"confidence": 0}]
    )
    def test_rejects_nonsense_settings(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be"):
            BootstrapConfig(**kwargs)  # type: ignore[arg-type]
