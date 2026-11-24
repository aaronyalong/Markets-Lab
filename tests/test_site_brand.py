"""The top bar's brand: the site name is plain text, only the author's name links out."""

from __future__ import annotations

from pathlib import Path

import pytest

from markets_lab.site_brand import (
    OUR_BRAND,
    PORTFOLIO,
    SiteBrandError,
    main,
    rewrite_brand,
    rewrite_site,
)

QUARTO = (
    '<div class="navbar-brand-container mx-auto">\n'
    '    <a class="navbar-brand" href="./index.html">\n'
    '    <span class="navbar-title">Markets Lab <span class="brand-by">by Aaron Yalong'
    "</span></span>\n"
    "    </a>\n"
    "  </div>"
)


def page(brand: str = QUARTO) -> str:
    return f"<html><body><header>{brand}</header><main>Text</main></body></html>"


def test_the_site_name_stops_being_a_link_and_the_name_links_to_the_portfolio() -> None:
    rewritten = rewrite_brand(page())
    assert '<a class="navbar-brand"' not in rewritten
    assert '<div class="navbar-brand"><span class="navbar-title">Markets Lab ' in rewritten
    assert f'<a class="brand-author" href="{PORTFOLIO}">Aaron Yalong</a>' in rewritten
    assert PORTFOLIO == "https://aaronyalong.com"
    # Exactly one link in the brand, and it is the name.
    brand = rewritten[rewritten.index("<header>") : rewritten.index("</header>")]
    assert brand.count("<a ") == 1


def test_a_page_in_a_subfolder_is_rewritten_the_same_way() -> None:
    nested = QUARTO.replace("./index.html", "../index.html")
    assert OUR_BRAND in rewrite_brand(page(nested))


def test_rewriting_twice_changes_nothing() -> None:
    once = rewrite_brand(page())
    assert rewrite_brand(once) == once


def test_a_page_without_the_expected_brand_fails_loudly() -> None:
    with pytest.raises(SiteBrandError):
        rewrite_brand(page('<a class="navbar-brand" href="/">Something else</a>'))
    with pytest.raises(SiteBrandError):
        rewrite_brand("<html><body>No header</body></html>")


def test_the_whole_built_site(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "site_libs").mkdir()
    (tmp_path / "index.html").write_text(page(), encoding="utf-8")
    (tmp_path / "experiments" / "nvidia.html").write_text(
        page(QUARTO.replace("./index.html", "../index.html")), encoding="utf-8"
    )
    (tmp_path / "site_libs" / "widget.html").write_text("<p>library file</p>", encoding="utf-8")
    assert rewrite_site(tmp_path) == 2
    assert rewrite_site(tmp_path) == 0
    lines: list[str] = []
    assert main(["--built-dir", str(tmp_path)], report=lines.append) == 0
    (tmp_path / "broken.html").write_text("<html></html>", encoding="utf-8")
    assert main(["--built-dir", str(tmp_path)], report=lines.append) == 1
    assert "broken.html" in lines[-1]
    assert main(["--built-dir", str(tmp_path / "missing")], report=lines.append) == 1
