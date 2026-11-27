"""Collecting NVIDIA's SEC filings: only title, source, time and link, and all of it
treated as untrusted text. Every payload here is made up; nothing is fetched."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from markets_lab.collectors import nvda_news
from markets_lab.collectors.nvda_news import (
    DEFAULT_FETCHERS,
    EDGAR,
    RETIRED_SOURCES,
    SOURCES,
    Collection,
    NewsError,
    NewsItem,
    clean_text,
    collect,
    day_path,
    filing_title,
    from_json,
    load_recent,
    main,
    parse_edgar,
    safe_url,
    save_day,
    to_json,
)

NOW = datetime(2030, 3, 19, 11, 0, tzinfo=UTC)
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "news.yml"


def item(hours_ago: float, title: str = "NVIDIA does a thing", source: str = EDGAR) -> NewsItem:
    slug = re.sub(r"\W+", "-", title.lower())
    return NewsItem(
        source=source,
        publisher="Example Wire",
        title=title,
        url=f"https://news.example.org/{slug}",
        published_at=NOW - timedelta(hours=hours_ago),
    )


class TestCleaning:
    def test_whitespace_and_control_characters_are_flattened(self) -> None:
        assert clean_text("  NVIDIA\n\tannounces \x00 results\u2028today ", 200) == (
            "NVIDIA announces results today"
        )

    def test_long_text_is_cut_with_an_ellipsis(self) -> None:
        cut = clean_text("word " * 100, 40)
        assert len(cut) <= 40 and cut.endswith("…")

    def test_anything_that_is_not_text_becomes_empty(self) -> None:
        assert clean_text(None, 50) == "" and clean_text(123, 50) == ""

    @pytest.mark.parametrize(
        "url",
        [
            "http://news.example.org/a",  # not https
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "//news.example.org/a",
            "https://user:pass@news.example.org/a",
            "https://news.example.org/a b",
            "https://news.example.org/a\nSet-Cookie: x",
            "https:///no-host",
            "",
            None,
            42,
        ],
    )
    def test_only_plain_https_links_are_accepted(self, url: object) -> None:
        assert safe_url(url) is None

    def test_a_plain_https_link_is_kept_and_a_host_list_is_enforced(self) -> None:
        assert safe_url(" https://news.example.org/a?b=1 ") == "https://news.example.org/a?b=1"
        hosts = frozenset({"www.sec.gov"})
        assert safe_url("https://www.sec.gov/x", hosts=hosts) == "https://www.sec.gov/x"
        assert safe_url("https://www.sec.gov.evil.example/x", hosts=hosts) is None


class TestSources:
    def test_only_sec_filings_are_collected(self) -> None:
        # My decision (2026-11-26): SEC filings only. NVIDIA's terms forbid automated
        # access and public display; Yahoo's feed is allowed but carries other
        # publishers' "buy this stock" headlines, which the page must not show.
        assert set(DEFAULT_FETCHERS) == SOURCES == {EDGAR}
        assert not SOURCES & RETIRED_SOURCES
        assert {"nvidia_newsroom", "yahoo_finance", "yahoo_finance_rss"} == RETIRED_SOURCES

    def test_the_collector_has_no_way_to_reach_a_news_site(self) -> None:
        code = Path(nvda_news.__file__).read_text()
        addresses = set(re.findall(r"https://[^\s\"')]+", code))
        assert addresses
        assert all(
            address.startswith(("https://data.sec.gov/", "https://www.sec.gov/"))
            for address in addresses
        ), addresses
        assert "yfinance" not in code.replace("(through yfinance)", "")
        for gone in ("parse_yahoo_rss", "fetch_yahoo_rss", "YAHOO_RSS", "xml.etree"):
            assert gone not in code, gone


def edgar(**columns: list[str]) -> str:
    base = {
        "form": ["8-K", "4", "10-Q"],
        "accessionNumber": ["0001045810-30-000010", "0001045810-30-000009", "0001045810-30-000008"],
        "acceptanceDateTime": [
            "2030-03-18T21:20:51.000Z",
            "2030-03-17T10:00:00.000Z",
            "2030-03-01T12:00:00.000Z",
        ],
        "primaryDocument": ["nvda-8k.htm", "form4.xml", "nvda-10q.htm"],
        "items": ["2.02,9.01", "", ""],
    }
    return json.dumps({"filings": {"recent": {**base, **columns}}})


class TestEdgar:
    def test_keeps_reports_and_skips_insider_forms(self) -> None:
        filings = parse_edgar(edgar())
        assert [f.title for f in filings] == [
            "Form 8-K: current report (results of operations and financial condition)",
            "Form 10-Q: quarterly report",
        ]
        assert filings[0].url == (
            "https://www.sec.gov/Archives/edgar/data/1045810/000104581030000010/nvda-8k.htm"
        )
        assert filings[0].published_at == datetime(2030, 3, 18, 21, 20, 51, tzinfo=UTC)
        assert all(f.source == EDGAR and f.publisher == "SEC EDGAR" for f in filings)

    def test_filing_titles(self) -> None:
        assert filing_title("10-K", "") == "Form 10-K: annual report"
        assert filing_title("8-K", "9.01") == "Form 8-K: current report"
        assert filing_title("8-K", "5.02, 6.66") == (
            "Form 8-K: current report (departure or appointment of directors or officers; "
            "item 6.66)"
        )

    def test_a_document_name_cannot_point_the_link_somewhere_else(self) -> None:
        hostile = edgar(primaryDocument=["x.htm\nLocation: https://evil.example", "a", "b.htm"])
        assert [f.title for f in parse_edgar(hostile)] == ["Form 10-Q: quarterly report"]

    def test_unexpected_shapes_fail_loudly(self) -> None:
        with pytest.raises(NewsError, match="unexpected shape"):
            parse_edgar("{}")
        with pytest.raises(NewsError, match="unexpected shape"):
            parse_edgar("not json")
        with pytest.raises(NewsError, match="different lengths"):
            parse_edgar(edgar(form=["8-K"]))


class TestCollect:
    def test_keeps_the_window_newest_first_without_duplicates(self) -> None:
        fresh, older, stale, future = item(2, "B"), item(20, "A"), item(30, "old"), item(-1, "soon")
        result = collect(
            now=NOW,
            fetchers={"other": lambda: [older, stale, fresh], EDGAR: lambda: [fresh, future]},
        )
        assert [i.title for i in result.items] == ["B", "A"]
        assert result.failed_sources == ()
        assert (result.collected_at, result.window_hours) == (NOW, 24)

    def test_one_failing_source_is_reported_and_the_rest_are_used(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken() -> list[NewsItem]:
            raise OSError("connection refused")

        result = collect(now=NOW, fetchers={EDGAR: broken, "other": lambda: [item(1)]})
        assert len(result.items) == 1
        assert result.failed_sources == (EDGAR,)
        assert "sec_edgar failed: connection refused" in capsys.readouterr().err

    def test_every_source_failing_is_an_error(self) -> None:
        def broken() -> list[NewsItem]:
            raise NewsError("down")

        with pytest.raises(NewsError, match="every news source failed"):
            collect(now=NOW, fetchers={EDGAR: broken, "other": broken})

    def test_with_sec_as_the_only_source_its_failure_is_the_whole_failure(self) -> None:
        def broken() -> list[NewsItem]:
            raise OSError("connection refused")

        with pytest.raises(NewsError, match="every news source failed: sec_edgar"):
            collect(now=NOW, fetchers={name: broken for name in DEFAULT_FETCHERS})

    def test_a_quiet_day_is_not_an_error(self) -> None:
        assert collect(now=NOW, fetchers={EDGAR: lambda: [item(40)]}).items == ()

    def test_a_time_without_a_zone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="time zone"):
            collect(now=datetime(2030, 3, 19, 11, 0), fetchers={})


class TestStorage:
    def collection(self) -> Collection:
        return Collection(NOW, 24, (item(1, "NVIDIA “quoted” — title"), item(3, "B")), (EDGAR,))

    def test_round_trip(self, tmp_path: Path) -> None:
        path = save_day(self.collection(), tmp_path)
        assert path is not None
        assert path == day_path(tmp_path, NOW.date()) == tmp_path / "2030-03-19.json"
        assert from_json(path.read_text(encoding="utf-8")) == self.collection()

    def test_a_day_is_written_once_and_never_rewritten(self, tmp_path: Path) -> None:
        first = save_day(self.collection(), tmp_path)
        assert first is not None
        before = first.read_text(encoding="utf-8")
        later = Collection(NOW + timedelta(hours=2), 24, (item(0.5, "newer"),), ())
        assert save_day(later, tmp_path) is None
        assert first.read_text(encoding="utf-8") == before

    def test_only_the_four_fields_are_stored_per_item(self, tmp_path: Path) -> None:
        stored = json.loads(to_json(self.collection()))
        assert set(stored) == {"collected_at", "window_hours", "failed_sources", "items"}
        for entry in stored["items"]:
            assert set(entry) == {"source", "publisher", "title", "url", "published_at"}

    def test_load_recent_returns_the_newest_days_first(self, tmp_path: Path) -> None:
        assert load_recent(tmp_path / "missing", days=3) == []
        for days_ago in (3, 2, 1, 0):
            when = NOW - timedelta(days=days_ago)
            save_day(Collection(when, 24, (), ()), tmp_path)
        (tmp_path / "notes.json").write_text("{}")  # ignored: not a day file
        loaded = load_recent(tmp_path, days=3)
        assert [c.collected_at.day for c in loaded] == [19, 18, 17]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("url", "javascript:alert(1)"),
            ("url", "http://news.example.org/a"),
            ("title", ""),
            ("source", "somewhere_else"),
            ("published_at", "2030-03-19 07:00"),
        ],
    )
    def test_a_stored_file_with_a_bad_item_is_refused(self, field: str, value: str) -> None:
        stored = json.loads(to_json(self.collection()))
        stored["items"][0][field] = value
        with pytest.raises(NewsError, match="not valid"):
            from_json(json.dumps(stored))

    def test_day_files_from_retired_sources_still_load(self) -> None:
        old = Collection(NOW, 24, tuple(item(1, f"NVIDIA {s}", s) for s in RETIRED_SOURCES), ())
        assert from_json(to_json(old)) == old

    def test_every_day_file_in_the_repository_still_loads(self) -> None:
        # Day files are never rewritten, so the ones written while headlines were
        # collected must stay readable for good.
        folder = WORKFLOW.parents[2] / "news" / "nvda"
        days = load_recent(folder, days=100_000)
        assert days
        for day in days:
            assert all(i.source in SOURCES | RETIRED_SOURCES for i in day.items)

    def test_a_file_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(NewsError, match="not valid"):
            from_json("{")


class TestCommand:
    def test_writes_the_day_then_does_nothing_on_a_rerun(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetchers = {EDGAR: lambda: [item(1, "NVIDIA ships")]}
        assert main(["--dir", str(tmp_path)], now=NOW, fetchers=fetchers) == 0
        out = capsys.readouterr().out
        assert "Example Wire: NVIDIA ships" in out
        assert "1 item(s) in the last 24 hours. Wrote" in out
        assert (tmp_path / "2030-03-19.json").is_file()

        assert main(["--dir", str(tmp_path)], now=NOW, fetchers=fetchers) == 0
        assert "already exists; nothing written" in capsys.readouterr().out

    def test_dry_run_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetchers = {EDGAR: lambda: [item(1)]}
        assert main(["--dir", str(tmp_path), "--dry-run"], now=NOW, fetchers=fetchers) == 0
        assert "Dry run: nothing written." in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []

    def test_total_failure_exits_nonzero_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken() -> list[NewsItem]:
            raise OSError("offline")

        assert main(["--dir", str(tmp_path)], now=NOW, fetchers={EDGAR: broken}) == 1
        assert "NEWS COLLECTION FAILED" in capsys.readouterr().err
        assert list(tmp_path.iterdir()) == []


class TestWorkflowFile:
    """Guards on the workflow text, so a later edit cannot quietly weaken it."""

    def test_only_new_day_files_can_be_committed(self) -> None:
        text = WORKFLOW.read_text()
        guard = text.index("Refuse anything except new day files")
        assert text.index("markets_lab.collectors.nvda_news") < guard < text.index("git commit")
        assert r'grep -v -E "^\?\? $NEWS_DIR/[0-9]{4}-[0-9]{2}-[0-9]{2}\.json$"' in text
        assert 'git add -- "$NEWS_DIR"' in text
        assert text.count("git add") == 1
        assert "git add -A" not in text and "git add ." not in text
        assert "--force" not in text

    def test_it_runs_before_the_open_with_a_clear_bot_message(self) -> None:
        text = WORKFLOW.read_text()
        assert 'cron: "0 11 * * *"' in text
        assert 'git commit -m "bot: add NVDA headlines' in text
        assert "git pull --rebase origin main" in text

    def test_the_sec_contact_comes_from_a_secret_not_the_repository(self) -> None:
        text = WORKFLOW.read_text()
        assert "SEC_USER_AGENT: ${{ secrets.SEC_USER_AGENT }}" in text
        assert "@" not in text.replace("actions/checkout@v4", "").replace(
            "astral-sh/setup-uv@v6", ""
        ).replace("41898282+github-actions[bot]@users.noreply.github.com", "")

    def test_it_is_separate_from_the_prediction_job(self) -> None:
        text = WORKFLOW.read_text()
        assert "predictions/" not in text and "nvda.nightly" not in text
