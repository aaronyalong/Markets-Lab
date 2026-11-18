"""The Polymarket client: read-only guard, retries, throttling and paging.

No test touches the network; every request is answered by an httpx mock transport.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from markets_lab.collectors.polymarket.client import (
    BASE_URL,
    ForbiddenRequestError,
    PolymarketAPIError,
    PolymarketClient,
    check_read_only,
)
from polymarket_payloads import numbered_events

Handler = Callable[[httpx.Request], httpx.Response]


class FakeTime:
    """A clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Recorder:
    """Records requests and answers them from a list of responses or a handler."""

    def __init__(self, *responses: httpx.Response, handler: Handler | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._responses: Iterator[httpx.Response] = iter(responses)
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._handler is not None:
            return self._handler(request)
        return next(self._responses)


def make_client(
    recorder: Recorder, fake: FakeTime | None = None, **kwargs: Any
) -> PolymarketClient:
    fake = fake or FakeTime()
    return PolymarketClient(
        transport=httpx.MockTransport(recorder), sleep=fake.sleep, clock=fake.clock, **kwargs
    )


def ok(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload)


class TestReadOnlyGuard:
    def test_allows_get_to_the_gateway(self) -> None:
        check_read_only(httpx.Request("GET", f"{BASE_URL}/series"))

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_rejects_other_methods(self, method: str) -> None:
        with pytest.raises(ForbiddenRequestError, match="only GET"):
            check_read_only(httpx.Request(method, f"{BASE_URL}/orders"))

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.polymarket.us/v1/orders/open",
            "https://gateway.polymarket.us.example.com/v1/series",
            "http://gateway.polymarket.us/v1/series",
        ],
    )
    def test_rejects_other_hosts_and_plain_http(self, url: str) -> None:
        with pytest.raises(ForbiddenRequestError, match="only https://gateway"):
            check_read_only(httpx.Request("GET", url))

    def test_guard_is_installed_on_the_underlying_http_client(self) -> None:
        recorder = Recorder()
        with make_client(recorder) as client:
            with pytest.raises(ForbiddenRequestError):
                client._http.post(f"{BASE_URL}/orders", json={})
            with pytest.raises(ForbiddenRequestError):
                client._http.get("https://api.polymarket.us/v1/account/balances")
        assert recorder.requests == []

    def test_sends_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYMARKET_US_KEY_ID", "should-never-be-read")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
        recorder = Recorder(ok({"series": []}))
        with make_client(recorder) as client:
            client.get_series("btc-updown-15m")
        (request,) = recorder.requests
        sent = {name.lower() for name in request.headers}
        assert "authorization" not in sent
        assert "cookie" not in sent
        assert not any(name.startswith("x-pm-") for name in sent)

    def test_redirects_are_not_followed(self) -> None:
        redirect = httpx.Response(302, headers={"Location": "https://api.polymarket.us/v1/x"})
        recorder = Recorder(redirect)
        with make_client(recorder) as client, pytest.raises(PolymarketAPIError, match="HTTP 302"):
            client.get_series("btc-updown-15m")
        assert len(recorder.requests) == 1


class TestRequests:
    def test_series_lookup(self) -> None:
        recorder = Recorder(ok({"series": []}))
        with make_client(recorder) as client:
            assert client.get_series("btc-updown-15m") == {"series": []}
        (request,) = recorder.requests
        assert request.method == "GET"
        assert request.url.path == "/v1/series"
        assert dict(request.url.params) == {"slug": "btc-updown-15m"}

    def test_events_page(self) -> None:
        recorder = Recorder(ok({"events": []}))
        with make_client(recorder) as client:
            client.get_events_page(375, limit=500, offset=1000)
        (request,) = recorder.requests
        assert request.url.path == "/v1/events"
        assert dict(request.url.params) == {"seriesId": "375", "limit": "500", "offset": "1000"}

    def test_price_history(self) -> None:
        recorder = Recorder(ok({"history": []}))
        with make_client(recorder) as client:
            client.get_price_history("cpc-btc-x", start_ts=100, end_ts=1_000)
        (request,) = recorder.requests
        assert request.url.path == "/v1/price-history"
        assert dict(request.url.params) == {
            "symbol": "cpc-btc-x",
            "timestamp.startTimestamp": "100",
            "timestamp.endTimestamp": "1000",
            "fidelity": "1",
        }

    def test_price_history_needs_a_forward_range(self) -> None:
        recorder = Recorder()
        with make_client(recorder) as client, pytest.raises(ValueError, match="end_ts"):
            client.get_price_history("cpc-btc-x", start_ts=100, end_ts=100)
        assert recorder.requests == []

    def test_settlement(self) -> None:
        recorder = Recorder(ok({"slug": "cpc-btc-x", "settlement": 1}))
        with make_client(recorder) as client:
            assert client.get_settlement("cpc-btc-x") == {"slug": "cpc-btc-x", "settlement": 1}
        assert recorder.requests[0].url.path == "/v1/markets/cpc-btc-x/settlement"

    def test_settlement_not_found_is_none(self) -> None:
        recorder = Recorder(httpx.Response(404, json={"code": 5}))
        with make_client(recorder) as client:
            assert client.get_settlement("cpc-btc-x") is None
        assert len(recorder.requests) == 1

    @pytest.mark.parametrize("slug", ["a/b", "a?b=c", "../orders", "", "a b"])
    def test_slugs_that_could_alter_the_url_are_rejected(self, slug: str) -> None:
        recorder = Recorder()
        with make_client(recorder) as client:
            with pytest.raises(ValueError, match="market slug"):
                client.get_settlement(slug)
            with pytest.raises(ValueError, match="market slug"):
                client.get_price_history(slug, start_ts=1, end_ts=2)
        assert recorder.requests == []


class TestFailures:
    def test_retries_rate_limits_with_exponential_backoff(self) -> None:
        fake = FakeTime()
        recorder = Recorder(httpx.Response(429), httpx.Response(429), ok({"series": []}))
        with make_client(recorder, fake, requests_per_second=1_000_000) as client:
            assert client.get_series("s") == {"series": []}
        assert len(recorder.requests) == 3
        assert [s for s in fake.sleeps if s >= 1] == [1.0, 2.0]

    def test_honours_a_longer_retry_after(self) -> None:
        fake = FakeTime()
        recorder = Recorder(httpx.Response(429, headers={"Retry-After": "7"}), ok({}))
        with make_client(recorder, fake, requests_per_second=1_000_000) as client:
            client.get_series("s")
        assert [s for s in fake.sleeps if s >= 1] == [7.0]

    def test_backoff_is_capped(self) -> None:
        fake = FakeTime()
        recorder = Recorder(httpx.Response(429, headers={"Retry-After": "86400"}), ok({}))
        with make_client(recorder, fake, requests_per_second=1_000_000) as client:
            client.get_series("s")
        assert max(fake.sleeps) == 60.0

    def test_gives_up_loudly_after_max_attempts(self) -> None:
        recorder = Recorder(handler=lambda _: httpx.Response(429))
        with (
            make_client(recorder, max_attempts=4) as client,
            pytest.raises(PolymarketAPIError, match=r"failed after 4 attempts: HTTP 429"),
        ):
            client.get_series("s")
        assert len(recorder.requests) == 4

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_retries_server_errors(self, status: int) -> None:
        recorder = Recorder(httpx.Response(status), ok({"ok": True}))
        with make_client(recorder) as client:
            assert client.get_series("s") == {"ok": True}

    def test_retries_network_errors_then_raises(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        recorder = Recorder(handler=fail)
        with (
            make_client(recorder, max_attempts=3) as client,
            pytest.raises(PolymarketAPIError, match="ConnectError"),
        ):
            client.get_series("s")
        assert len(recorder.requests) == 3

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_raise_without_retrying(self, status: int) -> None:
        recorder = Recorder(httpx.Response(status))
        with (
            make_client(recorder) as client,
            pytest.raises(PolymarketAPIError, match=f"HTTP {status}"),
        ):
            client.get_series("s")
        assert len(recorder.requests) == 1

    def test_non_json_body_raises(self) -> None:
        recorder = Recorder(httpx.Response(200, text="<html>challenge</html>"))
        with make_client(recorder) as client, pytest.raises(PolymarketAPIError, match="not JSON"):
            client.get_series("s")


class TestThrottle:
    def test_requests_are_spaced_by_the_minimum_interval(self) -> None:
        fake = FakeTime()
        recorder = Recorder(handler=lambda _: ok({}))
        with make_client(recorder, fake, requests_per_second=4) as client:
            for _ in range(3):
                client.get_series("s")
        assert fake.sleeps == [0.25, 0.25]

    def test_no_sleep_when_enough_time_has_passed(self) -> None:
        fake = FakeTime()
        recorder = Recorder(handler=lambda _: ok({}))
        with make_client(recorder, fake, requests_per_second=4) as client:
            client.get_series("s")
            fake.now += 10
            client.get_series("s")
        assert fake.sleeps == []

    @pytest.mark.parametrize("kwargs", [{"requests_per_second": 0}, {"max_attempts": 0}])
    def test_rejects_nonsense_settings(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="must be"):
            PolymarketClient(**kwargs)


def paged(events: list[dict[str, Any]]) -> Handler:
    """Serve `events` through limit/offset paging, like the real endpoint."""

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        return ok({"events": events[offset : offset + limit]})

    return handler


class TestIterEvents:
    def test_pages_until_a_short_page(self) -> None:
        recorder = Recorder(handler=paged(numbered_events(5)))
        with make_client(recorder) as client:
            slugs = [e["slug"] for e in client.iter_events(375, page_size=2)]
        assert slugs == [f"event-{i}" for i in range(5)]
        assert [r.url.params["offset"] for r in recorder.requests] == ["0", "2", "4"]

    def test_exact_multiple_ends_on_an_empty_page(self) -> None:
        recorder = Recorder(handler=paged(numbered_events(4)))
        with make_client(recorder) as client:
            assert len(list(client.iter_events(375, page_size=2))) == 4
        assert [r.url.params["offset"] for r in recorder.requests] == ["0", "2", "4"]

    def test_empty_series(self) -> None:
        recorder = Recorder(handler=paged([]))
        with make_client(recorder) as client:
            assert list(client.iter_events(375)) == []

    def test_events_repeated_across_pages_are_yielded_once(self) -> None:
        a, b, c, d = numbered_events(4)
        pages = iter([[a, b], [b, c], [d]])
        recorder = Recorder(handler=lambda _: ok({"events": next(pages)}))
        with make_client(recorder) as client:
            slugs = [e["slug"] for e in client.iter_events(375, page_size=2)]
        assert slugs == ["event-0", "event-1", "event-2", "event-3"]

    def test_paging_that_does_not_advance_raises(self) -> None:
        same_page = numbered_events(2)
        recorder = Recorder(handler=lambda _: ok({"events": same_page}))
        with make_client(recorder) as client, pytest.raises(PolymarketAPIError, match="progress"):
            list(client.iter_events(375, page_size=2))
        assert len(recorder.requests) == 2

    def test_event_without_slug_raises(self) -> None:
        recorder = Recorder(ok({"events": [{"id": "1"}]}))
        with make_client(recorder) as client, pytest.raises(PolymarketAPIError, match="slug"):
            list(client.iter_events(375))
