"""An in-memory stand-in for the Polymarket US gateway, serving synthetic windows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from markets_lab.collectors.polymarket.client import PolymarketClient
from polymarket_payloads import event, market, price_history

T0 = 1_800_000_000
"""Start of the first synthetic window: on the 15-minute grid, after the fee change."""

NOW = T0 + 30 * 86_400
"""The fake current time, long after every synthetic window."""


class FakeTime:
    """One clock for the client and the backfill; it moves only when something sleeps."""

    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_points(window_start: int) -> list[tuple[int, float | None, float | None]]:
    """One point per minute from the window start through minute 15."""
    return [(window_start + 60 * minute, 0.85, 0.16) for minute in range(16)]


class FakeGateway:
    """Routes the four endpoints the backfill uses. Add windows with `add_window`."""

    def __init__(self) -> None:
        self.series_ids: dict[str, int] = {}
        self.events: dict[int, list[dict[str, Any]]] = {}
        self.histories: dict[str, dict[str, Any]] = {}
        self.settlements: dict[str, int] = {}
        self.broken_history: set[str] = set()
        self.interrupt_on_history_of: str | None = None
        self.settlement_429s = 0
        """How many upcoming settlement requests to answer with 429 first."""
        self.requests: list[httpx.Request] = []

    def add_window(
        self,
        series_slug: str,
        index: int,
        *,
        settled_up: bool = True,
        settlement: int | str | None = "from-prices",
        status: str = "MARKET_STATUS_RESOLVED",
        points: list[tuple[int, float | None, float | None]] | None = None,
        window_start: int | None = None,
    ) -> str:
        """Add one window to a series and return its market slug."""
        series_id = self.series_ids.setdefault(series_slug, 100 + len(self.series_ids))
        start = T0 + 900 * index if window_start is None else window_start
        slug = f"synthetic-{series_slug}-{index}"
        resolved = status == "MARKET_STATUS_RESOLVED"
        self.events.setdefault(series_id, []).append(
            event(
                slug=f"event-{series_slug}-{index}",
                series_slug=series_slug,
                markets=[
                    market(
                        slug=slug,
                        market_id=str(10_000 * series_id + index),
                        coin=series_slug.split("-")[0],
                        window_start=iso(start),
                        window_end=iso(start + 900),
                        price_to_beat="50000.00" if resolved else None,
                        settlement_price=("50001.00" if settled_up else "49999.00")
                        if resolved
                        else None,
                        status=status,
                    )
                ],
            )
        )
        self.histories[slug] = price_history(*(default_points(start) if points is None else points))
        if settlement == "from-prices":
            if resolved:
                self.settlements[slug] = 1 if settled_up else 0
        elif isinstance(settlement, int):
            self.settlements[slug] = settlement
        return slug

    def client(self, fake_time: FakeTime, **kwargs: Any) -> PolymarketClient:
        return PolymarketClient(
            transport=httpx.MockTransport(self),
            sleep=fake_time.sleep,
            clock=fake_time.clock,
            **kwargs,
        )

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def count(self, path_part: str) -> int:
        return sum(path_part in path for path in self.paths())

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, params = request.url.path, request.url.params
        if path == "/v1/series":
            slug = params["slug"]
            found = (
                [{"id": str(self.series_ids[slug]), "slug": slug}]
                if slug in self.series_ids
                else []
            )
            return httpx.Response(200, json={"series": found})
        if path == "/v1/events":
            events = self.events.get(int(params["seriesId"]), [])
            offset, limit = int(params["offset"]), int(params["limit"])
            return httpx.Response(200, json={"events": events[offset : offset + limit]})
        if path == "/v1/price-history":
            slug = params["symbol"]
            if slug == self.interrupt_on_history_of:
                raise KeyboardInterrupt
            if slug in self.broken_history:
                return httpx.Response(400, json={"message": "broken on purpose"})
            return httpx.Response(200, json=self.histories.get(slug, {"history": []}))
        if path.startswith("/v1/markets/") and path.endswith("/settlement"):
            slug = path.removeprefix("/v1/markets/").removesuffix("/settlement")
            if self.settlement_429s > 0:
                self.settlement_429s -= 1
                return httpx.Response(429, headers={"Retry-After": "10"})
            if slug not in self.settlements:
                return httpx.Response(404, json={"code": 5, "message": "Settlement not found"})
            return httpx.Response(200, json={"slug": slug, "settlement": self.settlements[slug]})
        return httpx.Response(404, json={"message": f"no route for {path}"})
