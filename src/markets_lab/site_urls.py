"""Makes the built site's public addresses match what the host actually serves.

Quarto writes canonical links and sitemap entries that end in ".html". The
Cloudflare Worker serves each page at its clean address and redirects the ".html"
form to it (`html_handling: auto-trailing-slash` in wrangler.jsonc). A canonical
link or sitemap entry that points at a redirect sends search engines a mixed
signal, so after every render this module rewrites both to the clean form:

    https://example.org/index.html              ->  https://example.org/
    https://example.org/method.html             ->  https://example.org/method
    https://example.org/experiments/nvidia.html ->  https://example.org/experiments/nvidia

Links between pages are left alone: they are relative, still work through the
redirect, and keep working when the built site is opened straight from disk.

Quarto runs this as its post-render hook (see site/_quarto.yml). It fails loudly
if the sitemap is missing or a page has no canonical link, because that means the
site URL was dropped from the configuration.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

CANONICAL = re.compile(r'(<link rel="canonical" href=")([^"]+)(")')
SITEMAP_LOC = re.compile(r"(<loc>)([^<]+)(</loc>)")


class SiteUrlError(RuntimeError):
    """The built site is missing something the public addresses depend on."""


def clean_url(url: str) -> str:
    """The address the host serves a page at, given the address Quarto wrote."""
    if url.endswith("/index.html"):
        return url.removesuffix("index.html")
    return url.removesuffix(".html")


def _clean_match(match: re.Match[str]) -> str:
    return f"{match.group(1)}{clean_url(match.group(2))}{match.group(3)}"


def clean_canonical_links(html: str) -> str:
    return CANONICAL.sub(_clean_match, html)


def clean_sitemap(xml: str) -> str:
    return SITEMAP_LOC.sub(_clean_match, xml)


def clean_built_site(built_dir: Path) -> int:
    """Rewrite the canonical links and the sitemap in place.

    Returns the number of files that changed. Running it again changes nothing.
    """
    sitemap = built_dir / "sitemap.xml"
    if not sitemap.is_file():
        raise SiteUrlError(
            f"{sitemap} does not exist. Quarto writes it only when `site-url` is set."
        )
    pages = sorted(built_dir.rglob("*.html"))
    if not pages:
        raise SiteUrlError(f"no built pages under {built_dir}")

    changed = 0
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if not CANONICAL.search(html):
            raise SiteUrlError(
                f"{page} has no canonical link. Check `site-url` and `canonical-url`."
            )
        cleaned = clean_canonical_links(html)
        if cleaned != html:
            page.write_text(cleaned, encoding="utf-8")
            changed += 1

    xml = sitemap.read_text(encoding="utf-8")
    cleaned_xml = clean_sitemap(xml)
    if cleaned_xml != xml:
        sitemap.write_text(cleaned_xml, encoding="utf-8")
        changed += 1
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite the built site's canonical links and sitemap to clean addresses."
    )
    parser.add_argument("--built-dir", type=Path, default=Path("site/_site"))
    args = parser.parse_args(argv)
    try:
        changed = clean_built_site(args.built_dir)
    except SiteUrlError as error:
        print(f"SITE URL CHECK FAILED: {error}", file=sys.stderr)
        return 1
    print(f"Public addresses cleaned in {changed} file(s) under {args.built_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
