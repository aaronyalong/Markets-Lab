"""Checks on the site's source files that do not need Quarto.

They guard the promises the site makes: the disclaimer is on every page, work
that has not started shows no data, and internal links point at pages that exist. If the site has
been rendered, the built pages are checked for the disclaimer too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / "site"
CONFIG = SITE / "_quarto.yml"
PAGES = sorted(p for p in SITE.rglob("*.qmd") if "_site" not in p.parts)
BUILT = SITE / "_site"

LINK = re.compile(r"\]\(([^)#\s]+\.qmd)(?:#[^)]*)?\)")


def test_expected_pages_exist() -> None:
    names = {str(p.relative_to(SITE)) for p in PAGES}
    assert names == {
        "index.qmd",
        "method.qmd",
        "disclaimer.qmd",
        "experiments/nvidia.qmd",
        "404.qmd",
    }


def test_footer_on_every_page_says_not_financial_advice() -> None:
    config = CONFIG.read_text()
    assert "page-footer:" in config
    footer = config.split("page-footer:", 1)[1].split("\nformat:", 1)[0]
    assert "Not financial advice" in footer
    assert "No buy or sell signals" in footer
    assert "disclaimer.qmd" in footer


def test_every_page_in_the_navigation_exists() -> None:
    for href in re.findall(r"href:\s*(\S+\.qmd)", CONFIG.read_text()):
        assert (SITE / href).is_file(), f"navigation points at missing page {href}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SITE)))
def test_internal_links_point_at_pages_that_exist(page: Path) -> None:
    for target in LINK.findall(page.read_text()):
        resolved = SITE / target.lstrip("/") if target.startswith("/") else page.parent / target
        assert resolved.resolve().is_file(), f"{page.name} links to missing {target}"


def test_theme_uses_the_agreed_palette_and_font() -> None:
    theme = (SITE / "theme.scss").read_text().lower()
    for colour in ("#17171b", "#1f1f25", "#ececf1", "#9aa8ff"):
        assert colour in theme
    assert '"ubuntu"' in theme
    assert "family=Ubuntu" in (SITE / "_includes" / "fonts.html").read_text()


def home_cards() -> list[str]:
    home = (SITE / "index.qmd").read_text()
    grid = home.split("::: {.card-grid}", 1)[1]
    # Each card runs from its opening fence to the first closing fence after it.
    return [card.split("\n:::", 1)[0] for card in grid.split("::: {.lab-card")[1:]]


def test_home_has_one_live_card_and_one_coming_soon() -> None:
    cards = home_cards()
    assert len(cards) == 2
    assert "NVIDIA predictor" in cards[0]
    assert "experiments/nvidia.qmd" in cards[0]
    assert "Coming soon" not in cards[0]
    assert "Earnings explainer" in cards[1]
    assert "Coming soon" in cards[1]
    assert ".coming-soon" in cards[1].splitlines()[0]
    assert ".qmd" not in cards[1], "a coming-soon card must not link anywhere"


def test_work_that_has_not_started_shows_no_numbers() -> None:
    # A coming-soon card describes a plan. It must not look like it has results.
    assert not re.search(r"\d", home_cards()[1])


def test_no_page_uses_the_language_of_advice() -> None:
    banned = ("you should buy", "you should sell", "buy signal:", "sell signal:", "price target")
    for page in PAGES:
        text = page.read_text().lower()
        for phrase in banned:
            assert phrase not in text, f"{page.name} contains {phrase!r}"


def test_home_and_disclaimer_state_it_in_the_body_too() -> None:
    assert "Not financial advice" in (SITE / "index.qmd").read_text()
    assert "Not financial advice" in (SITE / "disclaimer.qmd").read_text()


def test_nvidia_page_has_the_chart_the_notice_and_every_generated_section() -> None:
    page = (SITE / "experiments" / "nvidia.qmd").read_text()
    assert '"symbol": "NASDAQ:NVDA"' in page
    assert "embed-widget-advanced-chart.js" in page
    # TradingView's free widget requires its attribution link to stay.
    assert "tradingview-widget-copyright" in page
    assert "https://www.tradingview.com/" in page
    assert "An experiment, not advice." in page
    for fragment in ("plotly", "tomorrow", "track_record", "backtest", "updated"):
        assert f"{{{{< include /_generated/{fragment}.md >}}}}" in page
    # The chart library must be loaded before the first chart on the page.
    assert page.index("_generated/plotly.md") < page.index("_generated/track_record.md")
    for heading in ("Live chart", "Tomorrow's call", "Track record", "Method"):
        assert f"## {heading}" in page


def test_generated_fragments_are_rebuilt_before_each_render_and_not_committed() -> None:
    assert "pre-render: uv run python -m markets_lab.analysis.nvda.site_page" in CONFIG.read_text()
    ignored = (SITE.parent / ".gitignore").read_text()
    assert "site/_generated/" in ignored


SITE_URL = "https://markets.aaronyalong.com"


def test_the_site_url_is_set_and_public_addresses_are_cleaned_after_each_render() -> None:
    config = CONFIG.read_text()
    assert f'site-url: "{SITE_URL}"' in config
    assert "canonical-url: true" in config
    assert "post-render: uv run python -m markets_lab.site_urls --built-dir _site" in config


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_built_canonical_links_and_sitemap_use_the_addresses_the_host_serves() -> None:
    canonical = re.compile(r'<link rel="canonical" href="([^"]+)"')
    addresses = []
    for page in sorted(BUILT.rglob("*.html")):
        found = canonical.findall(page.read_text())
        assert len(found) == 1, f"{page.name} needs exactly one canonical link"
        addresses.extend(found)
    sitemap = (BUILT / "sitemap.xml").read_text()
    locations = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    assert f"{SITE_URL}/" in locations
    assert f"{SITE_URL}/experiments/nvidia" in locations
    assert not any("404" in location for location in locations)
    for address in addresses + locations:
        assert address.startswith(f"{SITE_URL}/")
        assert not address.endswith(".html"), f"{address} would be redirected by the host"
    assert f"Sitemap: {SITE_URL}/sitemap.xml" in (BUILT / "robots.txt").read_text()


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_every_built_page_carries_the_disclaimer() -> None:
    built_pages = sorted(BUILT.rglob("*.html"))
    assert built_pages
    for page in built_pages:
        assert "Not financial advice" in page.read_text(), f"{page.name} lacks the disclaimer"
