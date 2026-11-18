"""Synthetic API payloads in the shape of the Polymarket US gateway responses.

Every value here is made up. No real market data is stored in the repository.
"""

from __future__ import annotations

from typing import Any


def amount(value: str) -> dict[str, str]:
    return {"value": value, "currency": "USD"}


def market(
    *,
    slug: str = "cpc-btc-updown-15m-2030-01-01-1200z",
    market_id: str = "1001",
    coin: str = "btc",
    index_symbol: str = "BRTI",
    window_start: str = "2030-01-01T12:00:00Z",
    window_end: str = "2030-01-01T12:15:00Z",
    price_to_beat: str | None = "50000.00",
    settlement_price: str | None = "50010.50",
    status: str = "MARKET_STATUS_RESOLVED",
) -> dict[str, Any]:
    return {
        "id": market_id,
        "slug": slug,
        "status": status,
        "feeCoefficient": 0.095,
        "assetPriceTerms": {
            "marketType": "ASSET_PRICE_MARKET_TYPE_UP_DOWN",
            "asset": {"assetClass": "ASSET_CLASS_CRYPTO", "symbol": coin},
            "indexSymbol": index_symbol,
            "horizon": "15m",
            "windowStart": window_start,
            "windowEnd": window_end,
            "priceToBeat": None if price_to_beat is None else amount(price_to_beat),
            "settlementPrice": None if settlement_price is None else amount(settlement_price),
        },
    }


def event(
    *,
    slug: str = "btc-updown-15m-2030-01-01-1200z",
    series_slug: str = "btc-updown-15m",
    markets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "501",
        "slug": slug,
        "seriesSlug": series_slug,
        "closed": True,
        "markets": [market()] if markets is None else markets,
    }


def numbered_events(count: int) -> list[dict[str, Any]]:
    """Distinct minimal events named event-0, event-1, ..."""
    return [event(slug=f"event-{i}") for i in range(count)]


def price_history(*points: tuple[int, float | None, float | None]) -> dict[str, Any]:
    return {
        "history": [
            {"timestamp": ts, "longPrice": long_price, "shortPrice": short_price}
            for ts, long_price, short_price in points
        ]
    }
