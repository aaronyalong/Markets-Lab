"""Parsing of Polymarket US payloads into typed rows."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from markets_lab.collectors.polymarket.parse import (
    DOWN,
    PRICE_SCALE,
    UP,
    ParseError,
    PricePoint,
    outcome_from_prices,
    parse_event_markets,
    parse_events_page,
    parse_price_history,
    parse_series_id,
    parse_settlement,
    to_price_units,
)
from polymarket_payloads import amount, event, market, price_history


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


class TestToPriceUnits:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0.0100", 100),
            ("0.8", 8_000),
            ("0.80", 8_000),
            (0.34, 3_400),
            (0.07, 700),
            (1, 10_000),
            (0, 0),
            (Decimal("0.99"), 9_900),
            ("86291.74", 862_917_400),
        ],
    )
    def test_converts_exactly(self, value: object, expected: int) -> None:
        assert to_price_units(value) == expected

    def test_every_whole_cent_float_is_exact(self) -> None:
        # Guards the bucket edges: 0.8 as a float must be exactly 80 cents, not 79.99.
        for cents in range(101):
            assert to_price_units(cents / 100) == cents * PRICE_SCALE // 100

    @pytest.mark.parametrize(
        "value",
        ["0.00001", "abc", "", None, True, float("nan"), float("inf"), "Infinity", [0.5]],
    )
    def test_rejects_values_it_would_have_to_guess_or_round(self, value: object) -> None:
        with pytest.raises(ParseError):
            to_price_units(value)


class TestOutcome:
    def test_close_above_open_is_up(self) -> None:
        assert outcome_from_prices(500_000_000, 500_000_100) == UP

    def test_close_below_open_is_down(self) -> None:
        assert outcome_from_prices(500_000_000, 499_999_900) == DOWN

    def test_tie_is_up(self) -> None:
        assert outcome_from_prices(500_000_000, 500_000_000) == UP

    @pytest.mark.parametrize(("open_", "close"), [(None, 1), (1, None), (None, None)])
    def test_missing_price_gives_no_outcome(self, open_: int | None, close: int | None) -> None:
        assert outcome_from_prices(open_, close) is None


class TestParseSettlement:
    @pytest.mark.parametrize(("value", "expected"), [(0, DOWN), (1, UP), (1.0, UP), (0.0, DOWN)])
    def test_accepts_zero_and_one(self, value: object, expected: int) -> None:
        assert parse_settlement({"slug": "m", "settlement": value}) == expected

    @pytest.mark.parametrize("value", [2, -1, 0.5, "1", None, True])
    def test_rejects_anything_else(self, value: object) -> None:
        with pytest.raises(ParseError):
            parse_settlement({"slug": "m", "settlement": value})

    def test_rejects_missing_field(self) -> None:
        with pytest.raises(ParseError):
            parse_settlement({"slug": "m"})


class TestParseSeriesId:
    def test_finds_exact_slug(self) -> None:
        payload = {
            "series": [
                {"id": "7", "slug": "btc-updown-15m-old"},
                {"id": "375", "slug": "btc-updown-15m"},
            ]
        }
        assert parse_series_id(payload, "btc-updown-15m") == 375

    def test_missing_series_raises(self) -> None:
        with pytest.raises(ParseError, match="series not found"):
            parse_series_id({"series": []}, "btc-updown-15m")


class TestParseEventMarkets:
    def test_parses_a_resolved_market(self) -> None:
        (parsed,) = parse_event_markets(event())
        assert parsed.market_slug == "cpc-btc-updown-15m-2030-01-01-1200z"
        assert parsed.market_id == 1001
        assert parsed.event_slug == "btc-updown-15m-2030-01-01-1200z"
        assert parsed.series_slug == "btc-updown-15m"
        assert parsed.coin == "btc"
        assert parsed.index_symbol == "BRTI"
        assert parsed.horizon == "15m"
        assert parsed.window_start == ts("2030-01-01T12:00:00")
        assert parsed.window_end == ts("2030-01-01T12:15:00")
        assert parsed.price_to_beat == 500_000_000
        assert parsed.settlement_price == 500_105_000
        assert parsed.is_resolved

    def test_window_and_coin_come_from_typed_fields_not_the_slug(self) -> None:
        # The slug claims BTC at 09:00; the typed fields say ETH at 12:00.
        misleading = market(slug="cpc-btc-updown-15m-2030-01-01-0900z", coin="eth")
        (parsed,) = parse_event_markets(event(markets=[misleading]))
        assert parsed.coin == "eth"
        assert parsed.window_start == ts("2030-01-01T12:00:00")

    def test_other_slug_styles_parse_the_same_way(self) -> None:
        eth = market(slug="cpceth-eth-updown-15m-2030-01-01-1200z", coin="eth")
        (parsed,) = parse_event_markets(event(markets=[eth]))
        assert parsed.window_end - parsed.window_start == 900

    def test_time_offsets_are_respected(self) -> None:
        shifted = market(
            window_start="2030-01-01T07:00:00-05:00", window_end="2030-01-01T07:15:00-05:00"
        )
        (parsed,) = parse_event_markets(event(markets=[shifted]))
        assert parsed.window_start == ts("2030-01-01T12:00:00")

    def test_unresolved_market_has_no_prices(self) -> None:
        pending = market(price_to_beat=None, settlement_price=None, status="MARKET_STATUS_OPEN")
        (parsed,) = parse_event_markets(event(markets=[pending]))
        assert parsed.price_to_beat is None
        assert parsed.settlement_price is None
        assert not parsed.is_resolved

    def test_event_without_markets_gives_nothing(self) -> None:
        assert parse_event_markets(event(markets=[])) == []

    @pytest.mark.parametrize(
        "broken",
        [
            {**market(), "assetPriceTerms": None},
            {**market(), "id": "12x"},
            {**market(), "status": ""},
            market(window_start="2030-01-01T12:15:00Z", window_end="2030-01-01T12:15:00Z"),
            market(window_start="2030-01-01T12:00:00"),
            market(window_start="not a time"),
            market(window_start="2030-01-01T12:00:00.500Z"),
            market(price_to_beat="-1"),
            market(price_to_beat="0"),
        ],
    )
    def test_rejects_malformed_markets(self, broken: dict[str, Any]) -> None:
        with pytest.raises(ParseError):
            parse_event_markets(event(markets=[broken]))

    def test_rejects_other_market_types(self) -> None:
        other = market()
        other["assetPriceTerms"]["marketType"] = "ASSET_PRICE_MARKET_TYPE_RANGE"
        with pytest.raises(ParseError, match="expected an Up/Down market"):
            parse_event_markets(event(markets=[other]))

    def test_rejects_non_usd_prices(self) -> None:
        other = market()
        other["assetPriceTerms"]["priceToBeat"] = {"value": "50000.00", "currency": "EUR"}
        with pytest.raises(ParseError, match="expected USD"):
            parse_event_markets(event(markets=[other]))

    def test_rejects_event_without_series(self) -> None:
        broken = event()
        del broken["seriesSlug"]
        with pytest.raises(ParseError):
            parse_event_markets(broken)


class TestParseEventsPage:
    def test_returns_event_objects(self) -> None:
        page = parse_events_page({"events": [event(), event(slug="other")]})
        assert [e["slug"] for e in page] == ["btc-updown-15m-2030-01-01-1200z", "other"]

    @pytest.mark.parametrize("payload", [{}, {"events": None}, {"events": ["x"]}, []])
    def test_rejects_unexpected_shapes(self, payload: object) -> None:
        with pytest.raises(ParseError):
            parse_events_page(payload)


class TestParsePriceHistory:
    def test_sorts_points_and_converts_prices(self) -> None:
        points = parse_price_history(price_history((120, 0.9, 0.11), (60, 0.34, 0.67)))
        assert points == [
            PricePoint(ts=60, long_price=3_400, short_price=6_700),
            PricePoint(ts=120, long_price=9_000, short_price=1_100),
        ]

    def test_settled_prices_at_the_bounds_are_allowed(self) -> None:
        assert parse_price_history(price_history((60, 0, 1))) == [
            PricePoint(ts=60, long_price=0, short_price=10_000)
        ]

    def test_missing_side_is_none(self) -> None:
        assert parse_price_history(price_history((60, 0.5, None))) == [
            PricePoint(ts=60, long_price=5_000, short_price=None)
        ]

    def test_empty_history(self) -> None:
        assert parse_price_history({"history": []}) == []

    def test_identical_duplicates_collapse(self) -> None:
        points = parse_price_history(price_history((60, 0.5, 0.51), (60, 0.5, 0.51)))
        assert len(points) == 1

    def test_conflicting_duplicates_raise(self) -> None:
        with pytest.raises(ParseError, match="conflicting"):
            parse_price_history(price_history((60, 0.5, 0.51), (60, 0.6, 0.41)))

    @pytest.mark.parametrize("price", [1.01, -0.01])
    def test_rejects_prices_outside_zero_to_one(self, price: float) -> None:
        with pytest.raises(ParseError, match="outside 0 to 1"):
            parse_price_history(price_history((60, price, 0.5)))

    def test_rejects_missing_history(self) -> None:
        with pytest.raises(ParseError):
            parse_price_history({})


def test_amount_helper_matches_api_shape() -> None:
    assert amount("1.00") == {"value": "1.00", "currency": "USD"}
