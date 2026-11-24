"""The page script that says "today" or "tomorrow" beside a session date.

It runs in the visitor's browser, so these tests run it in Node with fixed clocks.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "site"
SCRIPT = SITE / "_includes" / "session-dates.html"
NODE = shutil.which("node")


def relative_days(cases: list[tuple[str, str]], tmp_path: Path) -> list[str | None]:
    """relativeDay(session, now) for each (session date, UTC instant)."""
    source = re.search(r"<script>(.*)</script>", SCRIPT.read_text(), flags=re.DOTALL)
    assert source is not None
    module = tmp_path / "session_dates.js"
    module.write_text(source.group(1), encoding="utf-8")
    runner = (
        f"const {{ relativeDay }} = require({json.dumps(str(module))});\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map(([s, now]) => relativeDay(s, new Date(now)))));\n"
    )
    assert NODE is not None
    result = subprocess.run(
        [NODE, "-e", runner], capture_output=True, text=True, check=True, timeout=30
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_today_and_tomorrow_follow_new_york_time(tmp_path: Path) -> None:
    session = "2026-11-24"  # a Tuesday
    cases = [
        (session, "2026-11-24T04:00:00Z"),  # Monday 23:00 in New York
        (session, "2026-11-24T06:00:00Z"),  # Tuesday 01:00 in New York
        (session, "2026-11-24T21:30:00Z"),  # Tuesday 16:30, after the close
        (session, "2026-11-25T05:30:00Z"),  # Wednesday 00:30
        (session, "2026-11-22T15:00:00Z"),  # Sunday: two days out, no word
    ]
    assert relative_days(cases, tmp_path) == ["tomorrow", "today", "today", "past", None]


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_daylight_saving_time_is_respected(tmp_path: Path) -> None:
    # Clocks went forward on Sunday 8 March 2026. 04:30 UTC on 9 March is 00:30
    # in New York; a fixed five-hour offset would say 23:30 on the 8th.
    cases = [("2026-03-09", "2026-03-09T04:30:00Z"), ("2026-03-09", "2026-03-09T03:30:00Z")]
    assert relative_days(cases, tmp_path) == ["today", "tomorrow"]


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_month_and_year_ends(tmp_path: Path) -> None:
    cases = [("2027-01-04", "2027-01-04T02:00:00Z"), ("2026-12-01", "2026-12-01T03:00:00Z")]
    # 2027-01-04 02:00 UTC is Sunday 3 January 21:00 in New York.
    assert relative_days(cases, tmp_path) == ["tomorrow", "tomorrow"]


def test_only_dates_marked_certain_get_a_word() -> None:
    script = SCRIPT.read_text()
    assert 'time.session-date[data-certain="true"]' in script
    assert '.session-closed[data-certain="true"]' in script
    assert "America/New_York" in script


def test_the_script_is_on_every_page() -> None:
    config = (SITE / "_quarto.yml").read_text()
    assert "      - _includes/session-dates.html\n" in config
