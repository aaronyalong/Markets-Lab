"""Turn Polymarket US API JSON into typed rows.

Pure functions with no I/O. A payload that does not have the expected shape raises
`ParseError` instead of being guessed at.

Prices are stored as integers in units of $0.0001 (`PRICE_SCALE` units per dollar),
so comparisons such as "priced at 80 cents or more" are exact.

Window times come from the typed `assetPriceTerms` fields. Slugs are never parsed:
their format differs by coin and the API docs say it may change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

PRICE_SCALE = 10_000
"""Price units per dollar: one unit is $0.0001."""

UP = 1
DOWN = 0

UP_DOWN_MARKET_TYPE = "ASSET_PRICE_MARKET_TYPE_UP_DOWN"
RESOLVED_STATUS = "MARKET_STATUS_RESOLVED"


class ParseError(ValueError):
    """An API payload did not have the expected shape or values."""


@dataclass(frozen=True, slots=True)
class Market:
    """One Up/Down window. Prices are in `PRICE_SCALE` units; times are Unix seconds UTC."""

    market_slug: str
    market_id: int
    event_slug: str
    series_slug: str
    coin: str
    index_symbol: str
    horizon: str
    window_start: int
    window_end: int
    price_to_beat: int | None
    settlement_price: int | None
    status: str

    @property
    def is_resolved(self) -> bool:
        return self.status == RESOLVED_STATUS


@dataclass(frozen=True, slots=True)
class PricePoint:
    """One stored price-history observation for a market.

    `long_price` is the Up-side display price and `short_price` the Down-side display
    price, both in `PRICE_SCALE` units. What exactly they are derived from (ask, bid,
    mid or last trade) is not established by this module.
    """

    ts: int
    long_price: int | None
    short_price: int | None


def to_price_units(value: object) -> int:
    """Convert a dollar amount to integer `PRICE_SCALE` units, exactly.

    Accepts decimal strings, ints, floats and Decimals. Raises `ParseError` if the
    value is not a finite number or is finer than one unit, so nothing is rounded.
    """
    if isinstance(value, bool) or not isinstance(value, str | int | float | Decimal):
        raise ParseError(f"not a price: {value!r}")
    try:
        # str() of a float gives its shortest round-trip form, e.g. 0.34 -> "0.34".
        amount = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except InvalidOperation as exc:
        raise ParseError(f"not a price: {value!r}") from exc
    if not amount.is_finite():
        raise ParseError(f"not a finite price: {value!r}")
    units = amount * PRICE_SCALE
    if units != units.to_integral_value():
        raise ParseError(f"price finer than 1/{PRICE_SCALE} of a dollar: {value!r}")
    return int(units)


def outcome_from_prices(price_to_beat: int | None, settlement_price: int | None) -> int | None:
    """Outcome implied by the index prices: Up when close >= open, so a tie is Up.

    Returns None when either price is missing.
    """
    if price_to_beat is None or settlement_price is None:
        return None
    return UP if settlement_price >= price_to_beat else DOWN


def parse_settlement(payload: object) -> int:
    """Parse a settlement response into `UP` (1) or `DOWN` (0)."""
    value = _mapping(payload, "settlement response").get("settlement")
    if isinstance(value, bool) or not isinstance(value, int | float) or value not in (0, 1):
        raise ParseError(f"settlement is not 0 or 1: {value!r}")
    return int(value)


def parse_series_id(payload: object, slug: str) -> int:
    """Find the numeric id of the series with exactly this slug."""
    for item in _list(_mapping(payload, "series response").get("series"), "series"):
        series = _mapping(item, "series")
        if series.get("slug") == slug:
            return _int(series.get("id"), "series id")
    raise ParseError(f"series not found: {slug!r}")


def parse_events_page(payload: object) -> list[Mapping[str, Any]]:
    """Return the raw event objects from one page of an events response."""
    events = _list(_mapping(payload, "events response").get("events"), "events")
    return [_mapping(event, "event") for event in events]


def parse_event_markets(event: object) -> list[Market]:
    """Parse every Up/Down market inside one event object."""
    event_map = _mapping(event, "event")
    event_slug = _str(event_map.get("slug"), "event slug")
    series_slug = _str(event_map.get("seriesSlug"), f"seriesSlug of {event_slug}")
    markets = _list(event_map.get("markets"), f"markets of {event_slug}")
    return [_parse_market(_mapping(m, "market"), event_slug, series_slug) for m in markets]


def parse_price_history(payload: object) -> list[PricePoint]:
    """Parse a price-history response into points sorted by time.

    Exact duplicate points are collapsed. Two different observations at the same
    timestamp raise `ParseError`.
    """
    history = _list(_mapping(payload, "price-history response").get("history"), "history")
    by_ts: dict[int, PricePoint] = {}
    for item in history:
        raw = _mapping(item, "price-history point")
        point = PricePoint(
            ts=_int(raw.get("timestamp"), "price-history timestamp"),
            long_price=_optional_contract_price(raw.get("longPrice")),
            short_price=_optional_contract_price(raw.get("shortPrice")),
        )
        previous = by_ts.setdefault(point.ts, point)
        if previous != point:
            raise ParseError(f"conflicting price-history points at timestamp {point.ts}")
    return [by_ts[ts] for ts in sorted(by_ts)]


def _parse_market(market: Mapping[str, Any], event_slug: str, series_slug: str) -> Market:
    slug = _str(market.get("slug"), f"market slug in {event_slug}")
    terms = _mapping(market.get("assetPriceTerms"), f"assetPriceTerms of {slug}")
    market_type = terms.get("marketType")
    if market_type != UP_DOWN_MARKET_TYPE:
        raise ParseError(f"{slug}: expected an Up/Down market, got {market_type!r}")
    asset = _mapping(terms.get("asset"), f"asset of {slug}")
    window_start = _timestamp(terms.get("windowStart"), f"windowStart of {slug}")
    window_end = _timestamp(terms.get("windowEnd"), f"windowEnd of {slug}")
    if window_end <= window_start:
        raise ParseError(f"{slug}: window ends at or before it starts")
    return Market(
        market_slug=slug,
        market_id=_int(market.get("id"), f"id of {slug}"),
        event_slug=event_slug,
        series_slug=series_slug,
        coin=_str(asset.get("symbol"), f"asset symbol of {slug}").lower(),
        index_symbol=_str(terms.get("indexSymbol"), f"indexSymbol of {slug}"),
        horizon=_str(terms.get("horizon"), f"horizon of {slug}"),
        window_start=window_start,
        window_end=window_end,
        price_to_beat=_optional_amount(terms.get("priceToBeat"), f"priceToBeat of {slug}"),
        settlement_price=_optional_amount(
            terms.get("settlementPrice"), f"settlementPrice of {slug}"
        ),
        status=_str(market.get("status"), f"status of {slug}"),
    )


def _optional_contract_price(value: object) -> int | None:
    """A contract price in units, which must lie between $0 and $1 inclusive."""
    if value is None:
        return None
    units = to_price_units(value)
    if not 0 <= units <= PRICE_SCALE:
        raise ParseError(f"contract price outside 0 to 1 dollar: {value!r}")
    return units


def _optional_amount(value: object, what: str) -> int | None:
    """An `{"value": "...", "currency": "USD"}` object, or None, as positive units."""
    if value is None:
        return None
    amount = _mapping(value, what)
    currency = amount.get("currency")
    if currency != "USD":
        raise ParseError(f"{what}: expected USD, got {currency!r}")
    units = to_price_units(amount.get("value"))
    if units <= 0:
        raise ParseError(f"{what}: not a positive price: {amount.get('value')!r}")
    return units


def _timestamp(value: object, what: str) -> int:
    """An ISO-8601 time with an explicit offset, on a whole second, as Unix seconds."""
    text = _str(value, what)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ParseError(f"{what}: not an ISO-8601 time: {text!r}") from exc
    if moment.tzinfo is None:
        raise ParseError(f"{what}: time has no UTC offset: {text!r}")
    seconds = moment.timestamp()
    if seconds != int(seconds):
        raise ParseError(f"{what}: time is not on a whole second: {text!r}")
    return int(seconds)


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParseError(f"{what}: expected an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ParseError(f"{what}: expected a list, got {type(value).__name__}")
    return cast(list[Any], value)


def _str(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParseError(f"{what}: expected a non-empty string, got {value!r}")
    return value


def _int(value: object, what: str) -> int:
    """An integer given as a JSON number or as a string of digits (ids arrive as strings)."""
    if isinstance(value, bool):
        raise ParseError(f"{what}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ParseError(f"{what}: expected an integer, got {value!r}")
