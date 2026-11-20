"""When NVIDIA published results: parsing, validation, storage. No network.

Every filing here is made up: the times, the accession numbers and the pages.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from context_frames import FIRST_FILING as FIRST
from context_frames import make_publications
from markets_lab.collectors.nvda_earnings import (
    SUBMISSIONS_BASE,
    EarningsDataError,
    ResultsPublication,
    check_history_kept,
    fetch_publications,
    from_json,
    known_at,
    load_json,
    main,
    parse_page,
    parse_submissions,
    save_json,
    to_json,
    validate,
)
from markets_lab.collectors.nvda_news import EDGAR_URL

NOW = datetime(2030, 3, 15, 12, 0, tzinfo=UTC)
OLDER_PAGE = "CIK0001045810-submissions-001.json"


def sec_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def page_of(publications: list[ResultsPublication], *, noise: bool = True) -> dict[str, Any]:
    """SEC's column layout for these filings, newest first, among other filings."""
    rows = [
        ("8-K", item.accession, sec_time(item.accepted), ",".join(item.items))
        for item in reversed(publications)
    ]
    if noise:
        stamp = sec_time(publications[-1].accepted + timedelta(days=1))
        rows = [
            ("10-Q", "0000000000-99-900001", stamp, ""),
            ("8-K", "0000000000-99-900002", stamp, "5.02,9.01"),
            ("8-K/A", "0000000000-99-900003", stamp, "2.02"),
            ("4", "0000000000-99-900004", stamp, ""),
            *rows,
        ]
    form, accession, accepted, items = (list(column) for column in zip(*rows, strict=True))
    return {
        "form": form,
        "accessionNumber": accession,
        "acceptanceDateTime": accepted,
        "items": items,
        "primaryDocument": ["doc.htm"] * len(rows),
    }


def responses(publications: list[ResultsPublication], *, split: int = 60) -> dict[str, str]:
    """The main page with the newer filings and one older page with the rest."""
    main_page = {
        "cik": "1045810",
        "filings": {
            "recent": page_of(publications[split:]),
            "files": [{"name": OLDER_PAGE, "filingCount": split}],
        },
    }
    return {
        EDGAR_URL: json.dumps(main_page),
        SUBMISSIONS_BASE + OLDER_PAGE: json.dumps(page_of(publications[:split])),
    }


def getter_for(pages: dict[str, str]) -> Callable[[str], str]:
    return lambda url: pages[url]


class TestKnownAt:
    """A call can only use results published before its own timestamp."""

    def test_only_results_before_the_moment_are_returned(self) -> None:
        publications = make_publications()
        moment = publications[50].accepted + timedelta(hours=1)
        assert known_at(publications, moment) == tuple(publications[:51])

    def test_results_published_at_the_very_moment_of_the_call_are_not_known(self) -> None:
        publications = make_publications()
        results = publications[50]
        assert results not in known_at(publications, results.accepted)
        assert results in known_at(publications, results.accepted + timedelta(seconds=1))
        assert results not in known_at(publications, results.accepted - timedelta(seconds=1))

    def test_whatever_the_time_of_the_call_nothing_later_is_returned(self) -> None:
        publications = make_publications()
        moment = FIRST - timedelta(days=30)
        while moment < NOW:
            known = known_at(publications, moment)
            assert all(item.accepted < moment for item in known)
            assert len(known) == sum(item.accepted < moment for item in publications)
            moment += timedelta(days=17, hours=5)

    def test_the_moment_is_compared_as_an_instant_whatever_its_time_zone(self) -> None:
        from zoneinfo import ZoneInfo

        publications = make_publications()
        results = publications[50]  # accepted 20:30 UTC
        in_new_york = (results.accepted + timedelta(minutes=1)).astimezone(
            ZoneInfo("America/New_York")
        )
        assert results in known_at(publications, in_new_york)

    def test_a_moment_without_a_time_zone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            known_at(make_publications(), datetime(2020, 1, 1, 12, 0))


class TestParsePage:
    def test_keeps_results_filings_only(self) -> None:
        publications = make_publications(8)
        found = parse_page(page_of(publications))
        assert sorted(found) == publications
        # Item 12 (the number before August 2004) and Item 2.02 both count.
        assert {item.items for item in found} == {("12",), ("2.02", "9.01")}

    def test_amendments_and_other_items_do_not_count(self) -> None:
        page = page_of(make_publications(1), noise=False)
        page["form"] = ["8-K/A"]
        assert parse_page(page) == []
        page["form"], page["items"] = ["8-K"], ["5.02,9.01"]
        assert parse_page(page) == []

    def test_the_time_is_read_as_utc_and_items_are_trimmed(self) -> None:
        page = {
            "form": ["8-K"],
            "accessionNumber": ["0000000000-26-000001"],
            "acceptanceDateTime": ["2026-11-17T21:20:51.000Z"],
            "items": [" 2.02, 9.01 "],
        }
        assert parse_page(page) == [
            ResultsPublication(
                datetime(2026, 11, 17, 21, 20, 51, tzinfo=UTC),
                "0000000000-26-000001",
                ("2.02", "9.01"),
            )
        ]

    @pytest.mark.parametrize("stamp", ["2026-11-17T21:20:51", "2026-11-17 16:20", "soon", ""])
    def test_a_time_that_is_not_marked_utc_or_cannot_be_read_raises(self, stamp: str) -> None:
        page = page_of(make_publications(1), noise=False)
        page["acceptanceDateTime"] = [stamp]
        with pytest.raises(EarningsDataError, match="acceptance time"):
            parse_page(page)

    def test_a_missing_column_raises(self) -> None:
        page = page_of(make_publications(3))
        del page["items"]
        with pytest.raises(EarningsDataError, match="unexpected shape"):
            parse_page(page)

    def test_columns_of_different_lengths_raise(self) -> None:
        page = page_of(make_publications(3))
        page["items"] = page["items"][:-1]
        with pytest.raises(EarningsDataError, match="different lengths"):
            parse_page(page)


class TestParseSubmissions:
    def test_returns_the_recent_filings_and_the_older_pages(self) -> None:
        publications = make_publications()
        found, older = parse_submissions(responses(publications)[EDGAR_URL])
        assert sorted(found) == publications[60:]
        assert older == [OLDER_PAGE]

    @pytest.mark.parametrize(
        "name", ["../../other.json", "https://example.com/x.json", "CIK1-submissions-1.json"]
    )
    def test_an_unexpected_page_name_is_refused(self, name: str) -> None:
        # The name becomes part of an address that is fetched, so only SEC's own
        # naming pattern is accepted.
        text = json.dumps(
            {"filings": {"recent": page_of(make_publications(3)), "files": [{"name": name}]}}
        )
        with pytest.raises(EarningsDataError, match="unexpected page"):
            parse_submissions(text)

    @pytest.mark.parametrize("text", ["not json", "{}", '{"filings": []}'])
    def test_an_unexpected_response_raises(self, text: str) -> None:
        with pytest.raises(EarningsDataError, match="unexpected shape"):
            parse_submissions(text)


class TestFetchPublications:
    def test_joins_every_page_in_order_of_acceptance(self) -> None:
        publications = make_publications()
        asked: list[str] = []

        def get(url: str) -> str:
            asked.append(url)
            return responses(publications)[url]

        assert fetch_publications(now=NOW, get=get) == publications
        assert asked == [EDGAR_URL, SUBMISSIONS_BASE + OLDER_PAGE]

    def test_a_failed_download_is_reported_as_a_data_error(self) -> None:
        def failing(_url: str) -> str:
            raise OSError("no network")

        with pytest.raises(EarningsDataError, match="download from SEC failed: no network"):
            fetch_publications(now=NOW, get=failing)

    def test_an_older_page_that_is_not_json_raises(self) -> None:
        pages = responses(make_publications())
        pages[SUBMISSIONS_BASE + OLDER_PAGE] = "<html>busy</html>"
        with pytest.raises(EarningsDataError, match="is not JSON"):
            fetch_publications(now=NOW, get=getter_for(pages))

    def test_a_list_missing_its_older_page_fails_validation(self) -> None:
        pages = responses(make_publications())
        main_page = json.loads(pages[EDGAR_URL])
        main_page["filings"]["files"] = []
        pages[EDGAR_URL] = json.dumps(main_page)
        with pytest.raises(EarningsDataError, match="only 47 filings, expected at least 90"):
            fetch_publications(now=NOW, get=getter_for(pages))


class TestValidate:
    def test_accepts_a_clean_history(self) -> None:
        validate(make_publications(), now=NOW)

    def test_rejects_filings_out_of_order(self) -> None:
        publications = make_publications()
        publications[10], publications[11] = publications[11], publications[10]
        with pytest.raises(EarningsDataError, match="not in order"):
            validate(publications, now=NOW)

    def test_rejects_a_repeated_filing(self) -> None:
        publications = make_publications()
        publications[11] = replace(publications[11], accession=publications[10].accession)
        with pytest.raises(EarningsDataError, match="more than once"):
            validate(publications, now=NOW)

    def test_rejects_a_malformed_accession_number(self) -> None:
        publications = make_publications()
        publications[10] = replace(publications[10], accession="../etc")
        with pytest.raises(EarningsDataError, match="wrong form"):
            validate(publications, now=NOW)

    def test_rejects_a_filing_that_is_not_a_results_filing(self) -> None:
        publications = make_publications()
        publications[10] = replace(publications[10], items=("5.02",))
        with pytest.raises(EarningsDataError, match="not results filings"):
            validate(publications, now=NOW)

    def test_rejects_a_history_that_starts_late(self) -> None:
        with pytest.raises(EarningsDataError, match="history starts 2005-"):
            validate(make_publications()[8:], now=NOW)

    def test_rejects_a_missing_quarter(self) -> None:
        publications = make_publications()
        del publications[40]
        with pytest.raises(EarningsDataError, match="no results filing for 182 days"):
            validate(publications, now=NOW)

    def test_rejects_stale_and_future_lists(self) -> None:
        with pytest.raises(EarningsDataError, match="stale"):
            validate(make_publications(), now=NOW + timedelta(days=200))
        with pytest.raises(EarningsDataError, match="in the future"):
            validate(make_publications(), now=NOW - timedelta(days=200))

    def test_rejects_a_short_list(self) -> None:
        with pytest.raises(EarningsDataError, match="only 50 filings"):
            validate(make_publications()[-50:], now=NOW)

    def test_now_must_be_timezone_aware(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            validate(make_publications(), now=datetime(2030, 3, 15, 12, 0))


class TestStorage:
    def test_round_trip_is_exact(self, tmp_path: Path) -> None:
        publications = make_publications()
        path = tmp_path / "results.json"
        save_json(publications, path)
        assert load_json(path) == publications
        assert from_json(to_json(publications)) == publications

    def test_only_time_filing_and_items_are_stored(self) -> None:
        row = json.loads(to_json(make_publications(1)))[0]
        assert row == {
            "accepted": "2003-08-07T20:30:00Z",
            "accession": "0000000000-03-000000",
            "items": ["12"],
        }

    @pytest.mark.parametrize("text", ["not json", '[{"accepted": "yesterday"}]', '{"a": 1}'])
    def test_an_unreadable_file_raises(self, text: str) -> None:
        with pytest.raises(EarningsDataError, match="cannot be read"):
            from_json(text)

    def test_a_stored_filing_must_not_vanish_or_change_time(self) -> None:
        publications = make_publications()
        check_history_kept(publications[:-1], publications)  # a new filing is fine
        with pytest.raises(EarningsDataError, match="changes or drops 1 stored filings"):
            check_history_kept(publications, publications[:40] + publications[41:])
        moved = list(publications)
        moved[40] = replace(moved[40], accepted=moved[40].accepted + timedelta(hours=3))
        with pytest.raises(EarningsDataError, match="changes or drops 1 stored filings"):
            check_history_kept(publications, moved)


class TestCommandLine:
    def run_main(self, pages: dict[str, str], out: Path) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = main(
            ["--out", str(out)], get=getter_for(pages), now=lambda: NOW, report=lines.append
        )
        return code, lines

    def test_saves_and_reports(self, tmp_path: Path) -> None:
        publications = make_publications()
        out = tmp_path / "results.json"
        code, lines = self.run_main(responses(publications), out)
        assert code == 0
        assert lines == [f"saved 107 results filings, 2003-08-07 to 2030-01-03 (UTC), to {out}"]
        assert load_json(out) == publications

    def test_invalid_data_is_not_saved(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        code, lines = self.run_main(responses(make_publications(50), split=20), out)
        assert code == 1
        assert lines == [
            "error: results filings failed validation: only 50 filings, expected at least 90"
        ]
        assert not out.exists()

    def test_refuses_to_replace_a_file_when_a_stored_filing_changed(self, tmp_path: Path) -> None:
        publications = make_publications()
        out = tmp_path / "results.json"
        stored = list(publications)
        stored[40] = replace(stored[40], accepted=stored[40].accepted + timedelta(hours=3))
        save_json(stored, out)
        code, lines = self.run_main(responses(publications), out)
        assert code == 1
        assert "changes or drops 1 stored filings" in lines[0]
        assert load_json(out) == stored

    def test_missing_directory_is_refused(self, tmp_path: Path) -> None:
        code, lines = self.run_main(responses(make_publications()), tmp_path / "nope" / "r.json")
        assert code == 2
        assert "does not exist" in lines[0]
