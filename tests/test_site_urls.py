"""The built site's canonical links and sitemap name the addresses the host serves."""

from __future__ import annotations

from pathlib import Path

import pytest

from markets_lab.site_urls import (
    SiteUrlError,
    clean_built_site,
    clean_canonical_links,
    clean_sitemap,
    clean_url,
    main,
)

BASE = "https://lab.example.org"


@pytest.mark.parametrize(
    ("written", "served"),
    [
        (f"{BASE}/index.html", f"{BASE}/"),
        (f"{BASE}/method.html", f"{BASE}/method"),
        (f"{BASE}/experiments/nvidia.html", f"{BASE}/experiments/nvidia"),
        (f"{BASE}/experiments/index.html", f"{BASE}/experiments/"),
        # Already clean, or not a page: left alone.
        (f"{BASE}/", f"{BASE}/"),
        (f"{BASE}/method", f"{BASE}/method"),
        (f"{BASE}/sitemap.xml", f"{BASE}/sitemap.xml"),
        # Only the ending counts.
        (f"{BASE}/html-notes.html", f"{BASE}/html-notes"),
        (f"{BASE}/reindex.html", f"{BASE}/reindex"),
    ],
)
def test_clean_url(written: str, served: str) -> None:
    assert clean_url(written) == served
    assert clean_url(served) == served


def page(canonical: str | None, body: str = "") -> str:
    link = f'<link rel="canonical" href="{canonical}">' if canonical else ""
    return f"<html><head>{link}</head><body>{body}</body></html>"


def test_only_the_canonical_link_is_rewritten() -> None:
    body = '<a href="./method.html">Method</a> <a href="https://other.example/a.html">x</a>'
    html = page(f"{BASE}/method.html", body)
    cleaned = clean_canonical_links(html)
    assert f'<link rel="canonical" href="{BASE}/method">' in cleaned
    # Links between pages must keep working when the site is opened from disk.
    assert body in cleaned


def test_only_sitemap_locations_are_rewritten() -> None:
    xml = (
        f"<urlset><url><loc>{BASE}/index.html</loc><lastmod>2026-01-02</lastmod></url>"
        f"<url><loc>{BASE}/experiments/nvidia.html</loc></url></urlset>"
    )
    assert clean_sitemap(xml) == (
        f"<urlset><url><loc>{BASE}/</loc><lastmod>2026-01-02</lastmod></url>"
        f"<url><loc>{BASE}/experiments/nvidia</loc></url></urlset>"
    )


def build(tmp_path: Path) -> Path:
    built = tmp_path / "_site"
    (built / "experiments").mkdir(parents=True)
    (built / "index.html").write_text(page(f"{BASE}/"))
    (built / "method.html").write_text(page(f"{BASE}/method.html"))
    (built / "experiments" / "nvidia.html").write_text(page(f"{BASE}/experiments/nvidia.html"))
    (built / "sitemap.xml").write_text(
        f"<urlset><url><loc>{BASE}/index.html</loc></url>"
        f"<url><loc>{BASE}/method.html</loc></url></urlset>"
    )
    return built


def test_a_built_site_is_cleaned_in_place_and_a_second_run_changes_nothing(
    tmp_path: Path,
) -> None:
    built = build(tmp_path)
    # index.html already has a clean canonical link, so three files change.
    assert clean_built_site(built) == 3
    assert f'href="{BASE}/method"' in (built / "method.html").read_text()
    assert f'href="{BASE}/experiments/nvidia"' in (built / "experiments/nvidia.html").read_text()
    sitemap = (built / "sitemap.xml").read_text()
    assert f"<loc>{BASE}/</loc>" in sitemap
    assert f"<loc>{BASE}/method</loc>" in sitemap
    assert ".html" not in sitemap

    assert clean_built_site(built) == 0


def test_a_missing_sitemap_fails_loudly(tmp_path: Path) -> None:
    built = build(tmp_path)
    (built / "sitemap.xml").unlink()
    with pytest.raises(SiteUrlError, match="site-url"):
        clean_built_site(built)


def test_a_page_without_a_canonical_link_fails_loudly(tmp_path: Path) -> None:
    built = build(tmp_path)
    (built / "method.html").write_text(page(None))
    with pytest.raises(SiteUrlError, match="no canonical link"):
        clean_built_site(built)


def test_an_empty_build_fails_loudly(tmp_path: Path) -> None:
    built = tmp_path / "_site"
    built.mkdir()
    (built / "sitemap.xml").write_text("<urlset></urlset>")
    with pytest.raises(SiteUrlError, match="no built pages"):
        clean_built_site(built)


def test_the_command_reports_failure_with_a_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--built-dir", str(tmp_path / "missing")]) == 1
    assert "SITE URL CHECK FAILED" in capsys.readouterr().err

    built = build(tmp_path)
    assert main(["--built-dir", str(built)]) == 0
    assert "cleaned in 3 file(s)" in capsys.readouterr().out
