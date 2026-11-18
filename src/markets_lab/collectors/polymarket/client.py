"""Read-only HTTP client for the public Polymarket US gateway.

Safety properties, each enforced in code and covered by tests:

- Only GET requests, and only to the public gateway host. Anything else raises
  `ForbiddenRequestError` before it is sent.
- No credentials: no API key and no auth headers. Environment proxy settings and
  `.netrc` are ignored, so nothing on the machine can add credentials either.
- Requests are spaced out to stay well under the documented rate limit, and
  failures are retried with backoff and then raised, never swallowed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping
from types import TracebackType
from typing import Any, Self

import httpx

from markets_lab.collectors.polymarket.parse import parse_events_page

GATEWAY_HOST = "gateway.polymarket.us"
BASE_URL = f"https://{GATEWAY_HOST}/v1"

DEFAULT_REQUESTS_PER_SECOND = 4.0
"""About a sixth of the documented limit of 25 requests per second per IP.

Measured on 2026-11-18: price history accepted 5 requests per second, but the
settlement endpoint answered 429 (Retry-After: 10) at one request every 2 seconds
and accepted one every 3 seconds. Callers set a slower rate for settlements.
"""

DEFAULT_MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 60.0
EVENTS_PAGE_SIZE = 500

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SAFE_SLUG = re.compile(r"[A-Za-z0-9._-]+")


class ForbiddenRequestError(RuntimeError):
    """A request that is not a GET to the public gateway was attempted."""


class PolymarketAPIError(RuntimeError):
    """The API returned an error, an unusable response, or kept failing after retries."""


def check_read_only(request: httpx.Request) -> None:
    """Raise unless the request is an HTTPS GET to the public gateway."""
    if request.method != "GET":
        raise ForbiddenRequestError(f"only GET is allowed, not {request.method}")
    if request.url.scheme != "https" or request.url.host != GATEWAY_HOST:
        raise ForbiddenRequestError(f"only https://{GATEWAY_HOST} is allowed, not {request.url}")


class PolymarketClient:
    """Throttled, retrying, GET-only client for public market data."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._min_interval = 1.0 / requests_per_second
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._rate_limited_count = 0
        self._http = httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            event_hooks={"request": [check_read_only]},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    @property
    def requests_per_second(self) -> float:
        """The highest request rate this client will send at."""
        return 1.0 / self._min_interval

    def set_requests_per_second(self, requests_per_second: float) -> None:
        """Change the request rate, for endpoints with different limits."""
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second

    @property
    def rate_limited_count(self) -> int:
        """How many 429 responses this client has received, including retried ones."""
        return self._rate_limited_count

    def get_series(self, slug: str) -> Any:
        """Raw response of the series lookup for one slug."""
        return self._get_json("/series", {"slug": slug})

    def get_events_page(self, series_id: int, *, limit: int, offset: int) -> Any:
        """Raw response of one page of a series' events, in the API's own stable order."""
        return self._get_json("/events", {"seriesId": series_id, "limit": limit, "offset": offset})

    def iter_events(
        self, series_id: int, *, page_size: int = EVENTS_PAGE_SIZE
    ) -> Iterator[Mapping[str, Any]]:
        """Yield every event of a series once, paging until a short page.

        The series includes future and still-open windows; callers filter on the
        parsed market's status and window end. Events repeated across pages (the
        list can shift while paging) are yielded once. A full page containing
        nothing new means paging is not advancing, which raises.
        """
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        seen: set[str] = set()
        offset = 0
        while True:
            page = parse_events_page(
                self.get_events_page(series_id, limit=page_size, offset=offset)
            )
            new_events = 0
            for event in page:
                slug = event.get("slug")
                if not isinstance(slug, str) or not slug:
                    raise PolymarketAPIError(f"event without a slug in series {series_id}")
                if slug not in seen:
                    seen.add(slug)
                    new_events += 1
                    yield event
            if len(page) < page_size:
                return
            if new_events == 0:
                raise PolymarketAPIError(
                    f"events paging for series {series_id} made no progress at offset {offset}"
                )
            offset += len(page)

    def get_price_history(self, market_slug: str, *, start_ts: int, end_ts: int) -> Any:
        """Raw one-minute price history for a market between two Unix timestamps."""
        _check_slug(market_slug)
        if end_ts <= start_ts:
            raise ValueError("end_ts must be after start_ts")
        return self._get_json(
            "/price-history",
            {
                "symbol": market_slug,
                "timestamp.startTimestamp": start_ts,
                "timestamp.endTimestamp": end_ts,
                "fidelity": 1,
            },
        )

    def get_settlement(self, market_slug: str) -> Any | None:
        """Raw settlement response, or None when the market has no settlement yet."""
        _check_slug(market_slug)
        return self._get_json(f"/markets/{market_slug}/settlement", {}, none_on_404=True)

    def _get_json(
        self, path: str, params: Mapping[str, str | int], *, none_on_404: bool = False
    ) -> Any | None:
        url = f"{BASE_URL}{path}"
        last_problem = "no attempt made"
        for attempt in range(1, self._max_attempts + 1):
            self._throttle()
            retry_after: float | None = None
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return _decode_json(response, url)
                if response.status_code == 404 and none_on_404:
                    return None
                last_problem = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise PolymarketAPIError(f"GET {url} {dict(params)}: {last_problem}")
                if response.status_code == 429:
                    self._rate_limited_count += 1
                retry_after = _retry_after_seconds(response)
            if attempt < self._max_attempts:
                backoff = float(2 ** (attempt - 1))
                self._sleep(min(max(backoff, retry_after or 0.0), MAX_BACKOFF_SECONDS))
        raise PolymarketAPIError(
            f"GET {url} {dict(params)} failed after {self._max_attempts} attempts: {last_problem}"
        )

    def _throttle(self) -> None:
        """Sleep so that consecutive requests start at least the minimum interval apart."""
        if self._last_request_at is not None:
            wait = self._last_request_at + self._min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()


def _check_slug(slug: str) -> None:
    """Reject a slug that could change the request path or query when put in a URL."""
    if not _SAFE_SLUG.fullmatch(slug):
        raise ValueError(f"unexpected characters in market slug: {slug!r}")


def _decode_json(response: httpx.Response, url: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise PolymarketAPIError(f"GET {url}: response is not JSON") from exc


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The Retry-After header as seconds, if it is a plain non-negative number."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
