"""Features and labels for the NVDA experiment, including the look-ahead tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from markets_lab.analysis.nvda.features import (
    FEATURE_COLUMNS,
    SMA_WINDOWS,
    build_dataset,
    compute_features,
    compute_outcomes,
)
from price_frames import make_prices

ROWS = 420
WARMUP = max(SMA_WINDOWS) - 1  # first row index with every feature present
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices(rows=ROWS)


@pytest.fixture(scope="module")
def features(prices: pd.DataFrame) -> pd.DataFrame:
    return compute_features(prices)


def poison_after(prices: pd.DataFrame, position: int) -> pd.DataFrame:
    """A copy where every session after `position` is replaced with nonsense."""
    rng = np.random.default_rng(99)
    poisoned = prices.copy()
    later = poisoned.index[position + 1 :]
    for column in PRICE_COLUMNS:
        poisoned.loc[later, column] = rng.uniform(1_000, 9_000, len(later))
    poisoned.loc[later, "volume"] = rng.integers(1, 10, len(later))
    return poisoned


class TestNoLookAhead:
    """Features for a session must not depend on anything after that session."""

    def test_features_are_the_same_whether_or_not_later_sessions_exist(
        self, prices: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        # For every cut point: computing on history up to that session gives exactly
        # the row that computing on the full history gives.
        for position in range(WARMUP, ROWS):
            truncated = compute_features(prices.iloc[: position + 1]).iloc[-1]
            pd.testing.assert_series_equal(
                truncated, features.iloc[position], check_exact=True, check_names=False
            )

    @pytest.mark.parametrize("position", [WARMUP, 250, 300, ROWS - 2])
    def test_replacing_the_future_with_nonsense_changes_nothing_in_the_past(
        self, prices: pd.DataFrame, features: pd.DataFrame, position: int
    ) -> None:
        poisoned = compute_features(poison_after(prices, position))
        pd.testing.assert_frame_equal(
            poisoned.iloc[: position + 1], features.iloc[: position + 1], check_exact=True
        )
        # And the poison is strong enough to be seen: later rows do change.
        assert not poisoned.iloc[position + 1 :].equals(features.iloc[position + 1 :])

    def test_the_next_close_does_not_reach_todays_features(self, prices: pd.DataFrame) -> None:
        position = 300
        changed = prices.copy()
        next_session = changed.index[position + 1]
        changed.loc[next_session, "adj_close"] = changed["adj_close"].iloc[position] * 0.5
        before, after = compute_features(prices), compute_features(changed)
        pd.testing.assert_series_equal(before.iloc[position], after.iloc[position])
        # The outcome for that session does change, so the edit really mattered.
        assert compute_outcomes(changed)["next_return"].iloc[position] == pytest.approx(-0.5)
        assert compute_outcomes(prices)["next_return"].iloc[position] != pytest.approx(-0.5)

    def test_a_feature_that_peeked_would_be_caught(self, prices: pd.DataFrame) -> None:
        # The check above is only worth something if it fails on a leaky feature.
        def leaky(frame: pd.DataFrame) -> pd.Series:
            return frame["adj_close"].shift(-1) / frame["adj_close"] - 1.0

        position = 300
        full = leaky(prices).iloc[position]
        truncated = leaky(prices.iloc[: position + 1]).iloc[-1]
        assert not np.isnan(full)
        assert np.isnan(truncated)


class TestFeatureProperties:
    def test_columns_and_shape(self, prices: pd.DataFrame, features: pd.DataFrame) -> None:
        assert tuple(features.columns) == FEATURE_COLUMNS
        assert features.index.equals(prices.index)
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 20

    def test_warm_up_then_complete_and_finite(self, features: pd.DataFrame) -> None:
        assert features.iloc[:WARMUP].isna().any(axis=1).all()
        after = features.iloc[WARMUP:]
        assert not after.isna().any().any()
        assert np.isfinite(after.to_numpy()).all()

    def test_features_do_not_depend_on_the_price_level(
        self, prices: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        # Yahoo rescales past prices after splits; a 10-for-1 split divides prices
        # by 10 and multiplies volume by 10.
        rescaled = prices.copy()
        rescaled[PRICE_COLUMNS] = rescaled[PRICE_COLUMNS] / 10.0
        rescaled["volume"] = rescaled["volume"] * 10
        pd.testing.assert_frame_equal(
            compute_features(rescaled), features, check_exact=False, rtol=1e-9, atol=1e-12
        )

    def test_features_do_not_depend_on_how_far_back_history_starts_for_windowed_ones(
        self, prices: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        # Rolling-window features only look back a fixed number of sessions.
        windowed = [c for c in FEATURE_COLUMNS if not c.startswith(("rsi", "macd"))]
        later_start = compute_features(prices.iloc[100:])
        pd.testing.assert_frame_equal(
            later_start[windowed].iloc[WARMUP:],
            features[windowed].iloc[100 + WARMUP :],
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )


def simple_prices(closes: list[float], *, start: str = "2030-01-07") -> pd.DataFrame:
    """A frame with the given closes on consecutive business days (start is a Monday)."""
    index = pd.bdate_range(start=start, periods=len(closes), name="date")
    close = np.array(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(closes), 1_000),
        },
        index=index,
    )


class TestFeatureValues:
    def test_returns_are_log_returns_over_the_lag(self) -> None:
        frame = simple_prices([100.0, 110.0, 121.0, 133.1])
        features = compute_features(frame)
        assert features["ret_1"].iloc[1] == pytest.approx(np.log(1.1))
        assert features["ret_2"].iloc[2] == pytest.approx(np.log(1.21))
        assert features["ret_3"].iloc[3] == pytest.approx(np.log(1.331))
        assert np.isnan(features["ret_2"].iloc[1])

    def test_range_is_high_minus_low_over_close(self) -> None:
        features = compute_features(simple_prices([100.0, 110.0]))
        assert features["range_pct"].tolist() == pytest.approx([0.03, 0.03])

    def test_gap_from_moving_average(self) -> None:
        closes = [100.0] * 9 + [110.0]
        features = compute_features(simple_prices(closes))
        assert features["sma_10_gap"].iloc[-1] == pytest.approx(110.0 / 101.0 - 1.0)

    def test_volatility_of_a_steady_climb_is_zero(self) -> None:
        closes = [100.0 * 1.01**i for i in range(30)]
        features = compute_features(simple_prices(closes))
        assert features["vol_5"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
        assert features["vol_21"].iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_volume_ratio_is_zero_when_volume_is_constant(self) -> None:
        features = compute_features(simple_prices([100.0 + i for i in range(30)]))
        assert features["volume_ratio_21"].iloc[-1] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("closes", "expected"),
        [
            ([100.0 + i for i in range(40)], 1.0),
            ([100.0 - i for i in range(40)], 0.0),
            ([100.0] * 40, 0.5),
        ],
    )
    def test_rsi_extremes(self, closes: list[float], expected: float) -> None:
        features = compute_features(simple_prices(closes))
        assert features["rsi_14"].iloc[-1] == pytest.approx(expected)
        assert features["rsi_14"].iloc[:14].isna().all()

    def test_rsi_stays_between_zero_and_one(self, features: pd.DataFrame) -> None:
        rsi = features["rsi_14"].dropna()
        assert ((rsi >= 0) & (rsi <= 1)).all()

    def test_macd_is_positive_in_an_uptrend_and_negative_in_a_downtrend(self) -> None:
        up = compute_features(simple_prices([100.0 * 1.01**i for i in range(60)]))
        down = compute_features(simple_prices([100.0 * 0.99**i for i in range(60)]))
        assert up["macd"].iloc[-1] > 0
        assert down["macd"].iloc[-1] < 0

    def test_weekday_columns(self) -> None:
        features = compute_features(simple_prices([100.0] * 6))  # Monday to next Monday
        weekday_columns = ["dow_tue", "dow_wed", "dow_thu", "dow_fri"]
        assert features[weekday_columns].to_numpy().tolist() == [
            [0, 0, 0, 0],  # Monday
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],  # Monday
        ]


class TestOutcomes:
    def test_label_is_about_the_next_session(self) -> None:
        outcomes = compute_outcomes(simple_prices([100.0, 110.0, 99.0, 99.0, 105.0]))
        assert outcomes["target_up"].iloc[:-1].tolist() == [1.0, 0.0, 0.0, 1.0]
        assert outcomes["next_return"].iloc[0] == pytest.approx(0.10)
        assert outcomes["next_return"].iloc[1] == pytest.approx(-0.10)

    def test_a_tie_is_not_up(self) -> None:
        outcomes = compute_outcomes(simple_prices([100.0, 100.0, 101.0]))
        assert outcomes["target_up"].iloc[0] == 0.0
        assert outcomes["next_return"].iloc[0] == 0.0

    def test_last_session_has_no_outcome_yet(self) -> None:
        outcomes = compute_outcomes(simple_prices([100.0, 110.0, 120.0]))
        assert np.isnan(outcomes["target_up"].iloc[-1])
        assert np.isnan(outcomes["next_return"].iloc[-1])
        assert outcomes.iloc[:-1].notna().all().all()

    def test_label_uses_the_adjusted_close(self) -> None:
        frame = simple_prices([100.0, 100.0])
        frame["adj_close"] = [99.0, 100.0]  # a dividend makes the total return positive
        assert compute_outcomes(frame)["target_up"].iloc[0] == 1.0

    def test_outcome_for_a_session_depends_only_on_it_and_the_next(
        self, prices: pd.DataFrame
    ) -> None:
        outcomes = compute_outcomes(prices)
        truncated = compute_outcomes(prices.iloc[:302])
        pd.testing.assert_frame_equal(truncated.iloc[:301], outcomes.iloc[:301])
        assert np.isnan(truncated["target_up"].iloc[301])


class TestBuildDataset:
    def test_drops_warm_up_and_keeps_the_unlabelled_last_row(self, prices: pd.DataFrame) -> None:
        dataset = build_dataset(prices)
        assert len(dataset) == ROWS - WARMUP
        assert dataset.features.index[0] == prices.index[WARMUP]
        assert dataset.features.index.equals(dataset.target_up.index)
        assert dataset.features.index.equals(dataset.next_return.index)
        assert np.isnan(dataset.target_up.iloc[-1])
        assert dataset.target_up.iloc[:-1].isin([0.0, 1.0]).all()

    def test_labelled_excludes_only_the_last_row(self, prices: pd.DataFrame) -> None:
        dataset = build_dataset(prices)
        labelled = dataset.labelled()
        assert len(labelled) == len(dataset) - 1
        assert labelled.target_up.notna().all()
        assert labelled.features.index[-1] == prices.index[-2]

    def test_latest_is_the_last_completed_session(self, prices: pd.DataFrame) -> None:
        session, row = build_dataset(prices).latest()
        assert session == date(2030, 3, 14)
        assert tuple(row.index) == FEATURE_COLUMNS
        assert row.notna().all()

    def test_features_row_and_label_row_line_up(self, prices: pd.DataFrame) -> None:
        dataset = build_dataset(prices)
        adj = prices["adj_close"]
        for position in (0, 50, len(dataset) - 2):
            session = dataset.features.index[position]
            where = prices.index.get_loc(session)
            assert isinstance(where, int)
            expected_up = float(adj.iloc[where + 1] > adj.iloc[where])
            assert dataset.target_up.iloc[position] == expected_up
            assert dataset.features["ret_1"].iloc[position] == pytest.approx(
                np.log(adj.iloc[where] / adj.iloc[where - 1])
            )

    def test_too_little_history_raises(self) -> None:
        with pytest.raises(ValueError, match="not enough history"):
            build_dataset(make_prices(rows=150))

    def test_a_hole_after_the_warm_up_raises(self) -> None:
        broken = make_prices(rows=ROWS)
        broken.loc[broken.index[300], "volume"] = np.nan
        with pytest.raises(ValueError, match="incomplete after the warm-up"):
            build_dataset(broken)

    @pytest.mark.parametrize(
        "change",
        [
            lambda f: f.drop(columns=["adj_close"]),
            lambda f: f.iloc[::-1],
            lambda f: pd.concat([f, f.iloc[[-1]]]),
            lambda f: f.reset_index(drop=True),
        ],
    )
    def test_malformed_input_raises(self, change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        with pytest.raises(ValueError, match="prices"):
            compute_features(change(make_prices(rows=ROWS)))
