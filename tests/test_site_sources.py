"""Checks on the site's source files that do not need Quarto.

They guard the promises the site makes: the disclaimer is on every page, work
that has not started shows no data, and internal links point at pages that exist. If the site has
been rendered, the built pages are checked for the disclaimer too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from markets_lab import site_brand

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
    assert "Includes one simulated strategy, logged and scored. Not a recommendation." in footer
    assert "disclaimer.qmd" in footer


def test_the_nvidia_page_explains_the_paper_strategy_and_how_its_gate_was_chosen() -> None:
    page = " ".join((SITE / "experiments" / "nvidia.qmd").read_text().split())
    for fragment in ("paper.md", "paper_record.md", "paper_backtest.md"):
        assert f"{{{{< include /_generated/{fragment} >}}}}" in page
    # Stated on the page twice: beside the state, and in the method.
    assert page.count("gate was chosen knowing that volatility is predictable") == 2
    assert "it is not a signal to follow" in page
    assert "**The paper strategy is not a signal.**" in page
    # Its two states are never worded as an instruction.
    without_benchmark = page.lower().replace("buy-and-hold", "").replace("buy or sell", "")
    assert not re.search(r"\bbuy\b|\bsell\b", without_benchmark)


def test_every_page_in_the_navigation_exists() -> None:
    for href in re.findall(r"href:\s*(\S+\.qmd)", CONFIG.read_text()):
        assert (SITE / href).is_file(), f"navigation points at missing page {href}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(SITE)))
def test_internal_links_point_at_pages_that_exist(page: Path) -> None:
    for target in LINK.findall(page.read_text()):
        resolved = SITE / target.lstrip("/") if target.startswith("/") else page.parent / target
        assert resolved.resolve().is_file(), f"{page.name} links to missing {target}"


def palette() -> dict[str, str]:
    """The colours defined as literals at the top of the theme, by variable name."""
    theme = (SITE / "theme.scss").read_text()
    return dict(re.findall(r"^\$lab-([a-z]+): (#[0-9a-fA-F]{6});", theme, re.MULTILINE))


def luminance(colour: str) -> float:
    """Relative luminance of a "#rrggbb" colour, as WCAG 2 defines it."""
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    red, green, blue = (
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_contrast_formula_matches_known_values() -> None:
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


def test_theme_uses_the_agreed_palette_and_fonts() -> None:
    colours = palette()
    assert colours["bg"] == "#06080b"
    assert colours["panel"] == "#0e1217"
    assert colours["green"] == "#1fd68f"
    assert colours["red"] == "#ff5d6c"
    theme = (SITE / "theme.scss").read_text()
    assert '$font-family-sans-serif: "IBM Plex Sans"' in theme
    assert '$font-family-monospace: "IBM Plex Mono"' in theme
    assert '$headings-font-family: "Space Grotesk"' in theme
    fonts = (SITE / "_includes" / "fonts.html").read_text()
    for family in ("IBM+Plex+Sans", "IBM+Plex+Mono", "Space+Grotesk"):
        assert f"family={family}" in fonts


@pytest.mark.parametrize("colour", ["text", "muted", "green", "red", "lime", "mint"])
@pytest.mark.parametrize("background", ["bg", "panel"])
def test_every_text_colour_has_enough_contrast(colour: str, background: str) -> None:
    colours = palette()
    ratio = contrast(colours[colour], colours[background])
    assert ratio >= 4.5, f"{colour} on {background} is only {ratio:.2f}:1"


@pytest.mark.parametrize("stop", ["green", "lime", "mint"])
def test_text_on_the_accent_gradient_has_enough_contrast(stop: str) -> None:
    colours = palette()
    assert contrast(colours["ink"], colours[stop]) >= 4.5


def test_no_text_is_dimmed_with_opacity() -> None:
    # Dimming text with opacity or a translucent colour can take it below the
    # contrast floor without any palette value changing.
    theme = (SITE / "theme.scss").read_text()
    # Keyframes for decorative shapes come after this marker and are not text.
    rules = theme.split(
        "// -------------------------------------------------------------------- animation"
    )[0]
    assert rules != theme
    assert not re.search(r"^\s*opacity: 0\.[1-9]\d*;", rules, re.MULTILINE)
    assert not re.search(r"(color|fill): rgba\(\$lab-(muted|text)", theme)


def test_direction_colours_are_green_and_red_and_never_the_only_cue() -> None:
    theme = (SITE / "theme.scss").read_text()
    call_rules = theme.split("\n.call {", 1)[1].split("\n}\n", 1)[0]
    assert re.search(r"&\.up \{\s*color: \$lab-green;", call_rules)
    assert re.search(r"&\.down \{\s*color: \$lab-red;", call_rules)
    # The arrow and the word come from one helper used for every call.
    from markets_lab.analysis.nvda.site_page import _direction_text

    assert _direction_text(True) == "▲ Up"
    assert _direction_text(False) == "▼ Down"


def test_motion_is_switched_off_for_people_who_ask_for_less() -> None:
    theme = (SITE / "theme.scss").read_text()
    reduced = theme.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "animation: none !important;" in reduced
    assert "transition: none !important;" in reduced

    countup = (SITE / "_includes" / "countup.html").read_text()
    guard = countup.index('matchMedia("(prefers-reduced-motion: reduce)").matches) return;')
    assert guard < countup.index("requestAnimationFrame")
    # A stalled animation must not leave a wrong number on the page.
    assert "setTimeout(finish, duration + 250);" in countup
    assert "      - _includes/countup.html\n" in CONFIG.read_text()

    # The scrolling tape is swapped for a static row of tickers.
    ticker = (SITE / "_includes" / "ticker.md").read_text()
    assert 'matchMedia("(prefers-reduced-motion: reduce)").matches' in ticker
    assert '(still ? "embed-widget-tickers.js" : "embed-widget-ticker-tape.js")' in ticker


def test_the_ticker_is_on_both_main_pages_with_its_attribution() -> None:
    ticker = (SITE / "_includes" / "ticker.md").read_text()
    assert "NASDAQ:NVDA" in ticker
    assert "https://www.tradingview.com/" in ticker and "Quotes by TradingView" in ticker
    for page in ("index.qmd", "experiments/nvidia.qmd"):
        text = (SITE / page).read_text()
        assert "{{< include /_includes/ticker.md >}}" in text


def test_the_site_names_its_author() -> None:
    config = CONFIG.read_text()
    footer = config.split("page-footer:", 1)[1].split("\nformat:", 1)[0]
    assert "Built by [Aaron Yalong](https://aaronyalong.com)" in footer
    # The name sits beside the site title in the top bar of every page.
    navbar = config.split("navbar:", 1)[1].split("page-footer:", 1)[0]
    assert 'title: "Markets Lab [by Aaron Yalong]{.brand-by}"' in navbar
    assert "    right:\n" in navbar and "    left:\n" not in navbar


def home_cards() -> list[str]:
    home = (SITE / "index.qmd").read_text()
    grid = home.split("::: {.card-grid}", 1)[1]
    # Each card runs from its opening fence to the first closing fence after it.
    return [card.split("\n:::", 1)[0] for card in grid.split("::: {.lab-card")[1:]]


def test_the_home_page_never_says_predictor() -> None:
    assert "predictor" not in (SITE / "index.qmd").read_text().lower()


def test_the_home_picture_is_a_crystal_ball_labelled_as_an_illustration() -> None:
    home = (SITE / "index.qmd").read_text()
    svg = home[home.index('<svg class="cone crystal"') : home.index("</svg>")]
    assert 'role="img"' in svg and 'aria-label="Illustration: a crystal ball' in svg
    assert 'class="art-note">Illustration, not data.<' in home
    # Its colours come from the site palette: no blue anywhere.
    colours = set(re.findall(r"#[0-9a-fA-F]{6}", svg))
    assert {"#1fd68f", "#9be564", "#a6f4d0"} <= colours
    # The palette's dark greys lean very slightly blue (#0e1217, #1f2630); a colour
    # whose blue leads both other channels by more than that reads as blue.
    for colour in colours:
        red, green, blue = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
        assert blue - max(red, green) <= 16, f"{colour} is blue"


def test_home_has_one_live_card_and_one_coming_soon() -> None:
    cards = home_cards()
    assert len(cards) == 2
    assert "### [NVIDIA [→]{.arrow}]" in cards[0]
    assert "predictor" not in cards[0].lower()
    assert '<svg class="ball-icon"' in cards[0]  # the crystal ball icon
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
    # Said once, in the notice under the title, and nowhere else in the page's own
    # text (my request, 2026-11-26). The generated sections are checked in
    # tests/test_nvda_site_page.py; the footer's disclaimer is separate.
    assert page.count("An experiment, not advice.") == 1
    assert page.lower().count("advice") == 1
    for fragment in (
        "summary",
        "status",
        "news",
        "plotly",
        "tomorrow",
        "range",
        "track_record",
        "range_record",
        "backtest",
        "range_backtest",
        "updated",
    ):
        assert f"{{{{< include /_generated/{fragment}.md >}}}}" in page
    # The summary sits in the hero, after the notice and before the status strip.
    hero = page[page.index("{.page-hero}") : page.index("_generated/status.md")]
    assert hero.index("An experiment, not advice.") < hero.index("_generated/summary.md")
    # The chart library must be loaded before the first chart on the page.
    assert page.index("_generated/plotly.md") < page.index("_generated/track_record.md")
    # The chart library must also come before the range chart further down.
    assert page.index("_generated/plotly.md") < page.index("_generated/range_backtest.md")
    for heading in (
        "NVIDIA filings",
        "Live chart",
        "Latest calls",
        "Latest range",
        "Track record",
        "Method",
    ):
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
    assert "    - uv run python -m markets_lab.site_urls --built-dir _site\n" in config


def test_only_the_authors_name_is_a_link_in_the_top_bar() -> None:
    config = CONFIG.read_text()
    assert "    - uv run python -m markets_lab.site_brand --built-dir _site\n" in config
    # The brand's link and the footer's go to the same portfolio address.
    assert f"Built by [Aaron Yalong]({site_brand.PORTFOLIO})" in config


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
def test_headline_numbers_are_in_the_built_html_without_javascript() -> None:
    page = (BUILT / "experiments" / "nvidia.html").read_text()
    numbers = re.findall(r'<div class="big-number">([^<]+)</div>', page)
    # Three for the direction backtest and three for the range backtest.
    assert len(numbers) == 6
    assert all(re.search(r"\d", number) for number in numbers)


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_every_built_page_links_only_the_authors_name_in_the_top_bar() -> None:
    for page in sorted(BUILT.rglob("*.html")):
        if "site_libs" in page.parts:
            continue
        built = page.read_text()
        assert site_brand.OUR_BRAND in built, page
        assert '<a class="navbar-brand"' not in built, page


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_every_built_page_carries_the_disclaimer() -> None:
    built_pages = sorted(BUILT.rglob("*.html"))
    assert built_pages
    for page in built_pages:
        assert "Not financial advice" in page.read_text(), f"{page.name} lacks the disclaimer"


NVIDIA_PAGE = SITE / "experiments" / "nvidia.qmd"
JUMP_LINK = re.compile(r'<a href="#([a-z0-9-]+)">([^<]+)</a>')


def _heading_id(heading: str) -> str:
    """The id Pandoc gives a heading (enough for the headings on this page)."""
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-")


def test_the_calls_and_the_range_come_first_on_the_nvidia_page() -> None:
    page = NVIDIA_PAGE.read_text()
    order = [
        "## Latest calls",
        "## Latest range",
        "## Track record",
        "## NVIDIA filings",
        "## Live chart",
        "## What two decades of history say",
        "## Method",
    ]
    positions = [page.index(heading) for heading in order]
    assert positions == sorted(positions)
    # The status strip and the jump bar sit between the hero and the calls.
    assert page.index("_generated/status.md") < page.index('class="jump-bar"')
    assert page.index('class="jump-bar"') < page.index("## Latest calls")


def test_every_jump_link_goes_to_a_section_of_the_page_in_order() -> None:
    page = NVIDIA_PAGE.read_text()
    nav = page[page.index('<nav class="jump-bar"') : page.index("</nav>")]
    assert 'aria-label="On this page"' in nav
    links = JUMP_LINK.findall(nav)
    assert len(links) == 7
    headings = [_heading_id(h) for h in re.findall(r"^## (.+?) \{", page, re.MULTILINE)]
    targets = [target for target, _ in links]
    assert all(target in headings for target in targets)
    assert [headings.index(t) for t in targets] == sorted(headings.index(t) for t in targets)
    # Plain links, so every one is reachable with the keyboard; nothing removes them
    # from the tab order.
    assert "tabindex" not in nav
    assert "{{< include /_includes/jump-bar.md >}}" in page


def test_the_jump_bar_stays_on_one_line_and_leaves_room_above_a_section() -> None:
    theme = (SITE / "theme.scss").read_text()
    bar = theme[theme.index(".jump-inner {") :]
    bar = bar[: bar.index("\n}\n")]
    assert "flex-wrap: nowrap;" in bar and "white-space: nowrap;" in bar
    assert "scroll-margin-top: calc(var(--jump-bar-height" in theme
    assert "&:focus-visible" in bar
    script = (SITE / "_includes" / "jump-bar.md").read_text()
    assert "prefers-reduced-motion: reduce" in script
    assert "bar.offsetHeight + headerHeight()" in script
    assert "heading.focus(" in script


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_every_built_jump_link_has_its_target() -> None:
    built = (BUILT / "experiments" / "nvidia.html").read_text()
    nav = built[built.index('<nav class="jump-bar"') : built.index("</nav>")]
    for target, _ in JUMP_LINK.findall(nav):
        assert f'<section id="{target}"' in built


def test_the_nvidia_page_never_says_predictor_and_has_a_quiet_title() -> None:
    # My request (2026-11-24): no "predictor" on the page, "NVIDIA" in a much
    # softer version of the accent, and no "Experiment · live" bubble above it.
    page = NVIDIA_PAGE.read_text()
    assert "predictor" not in page.lower()
    assert "\n# [NVIDIA]{.grad-soft}\n" in page
    assert ".tag .live" not in page
    # It still says it is an experiment, right under the title.
    hero = page[page.index("{.page-hero}") : page.index("_generated/status.md")]
    assert "**An experiment, not advice.**" in hero
    menu = CONFIG.read_text().split("menu:", 1)[1].split("- href: method.qmd", 1)[0]
    assert "text: NVIDIA\n" in menu
    theme = (SITE / "theme.scss").read_text()
    soft = theme[theme.index(".grad-soft {") :].split("}", 1)[0]
    # Only palette colours that pass the contrast test, light at both ends.
    assert "linear-gradient(100deg, $lab-text 0%, $lab-text 35%, $lab-mint 100%)" in soft


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_the_built_nvidia_page_never_shows_the_word_predictor() -> None:
    built = (BUILT / "experiments" / "nvidia.html").read_text()
    visible = re.sub(r"<script.*?</script>", " ", built, flags=re.DOTALL)
    visible = re.sub(r"<[^>]+>", " ", visible)
    assert "predictor" not in visible.lower()


@pytest.mark.skipif(not BUILT.is_dir(), reason="site has not been rendered")
def test_the_built_nvidia_page_says_experiment_not_advice_once() -> None:
    # My request (2026-11-26): once, under the title. The footer's "Not financial
    # advice" is the site-wide disclaimer and is counted separately.
    built = (BUILT / "experiments" / "nvidia.html").read_text()
    visible = re.sub(r"<script.*?</script>", " ", built, flags=re.DOTALL)
    visible = " ".join(re.sub(r"<[^>]+>", " ", visible).split()).lower()
    assert visible.count("not advice") == 1
    assert visible.count("advice") == 2
    assert visible.index("an experiment, not advice.") < visible.index("calls for")


def test_running_text_has_one_reading_width() -> None:
    # Introductions ran the full 72rem while the Method section was capped.
    theme = (SITE / "theme.scss").read_text()
    assert "$reading-width: 46rem;" in theme
    assert theme.count("max-width: $reading-width;") == 2
    band = theme[theme.index(".band > p,") :].split("}", 1)[0]
    assert ".band section > p {" in band and "max-width: $reading-width;" in band
    prose = theme[theme.index(".prose {") :].split("\n}\n", 1)[0]
    assert "max-width: $reading-width;" in prose
