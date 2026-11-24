"""Makes the top bar read "Markets Lab by Aaron Yalong" with only the name a link.

Quarto always wraps the navbar title in a link to the home page (its template has
no option to leave it plain), and a link cannot hold another link. So after every
render this module rewrites the brand on each built page: the site name becomes
plain text and the name becomes a link to the author's portfolio.

    <a class="navbar-brand" href="./index.html">
    <span class="navbar-title">Markets Lab <span class="brand-by">by Aaron Yalong</span></span>
    </a>

becomes

    <div class="navbar-brand">
    <span class="navbar-title">Markets Lab <span class="brand-by">by
    <a href="https://aaronyalong.com">Aaron Yalong</a></span></span>
    </div>

The menu's "Home" item still links home. A page that is already rewritten is left
alone, so a partial render followed by this step is safe. A page with neither
form fails loudly, because that means Quarto's markup changed.

Quarto runs this as a post-render hook (see site/_quarto.yml).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PORTFOLIO = "https://aaronyalong.com"
AUTHOR = "Aaron Yalong"
SITE_NAME = "Markets Lab"

QUARTO_BRAND = re.compile(
    r'<a class="navbar-brand" href="[^"]*">\s*'
    r'<span class="navbar-title">'
    + re.escape(SITE_NAME)
    + r' <span class="brand-by">by '
    + re.escape(AUTHOR)
    + r"</span></span>\s*</a>"
)
OUR_BRAND = (
    '<div class="navbar-brand">'
    f'<span class="navbar-title">{SITE_NAME} <span class="brand-by">by '
    f'<a class="brand-author" href="{PORTFOLIO}">{AUTHOR}</a></span></span>'
    "</div>"
)


class SiteBrandError(RuntimeError):
    """A built page does not carry the brand this module knows how to rewrite."""


def rewrite_brand(page: str) -> str:
    """The page with its brand rewritten; unchanged if it already was."""
    if OUR_BRAND in page:
        return page
    rewritten, count = QUARTO_BRAND.subn(OUR_BRAND, page)
    if count != 1:
        raise SiteBrandError(f"expected one navbar brand, found {count}")
    return rewritten


def rewrite_site(built_dir: Path) -> int:
    """Rewrite every built page; returns how many files changed."""
    pages = sorted(built_dir.rglob("*.html"))
    if not pages:
        raise SiteBrandError(f"no pages under {built_dir}")
    changed = 0
    for path in pages:
        if "site_libs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            new = rewrite_brand(text)
        except SiteBrandError as exc:
            raise SiteBrandError(f"{path}: {exc}") from exc
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: Sequence[str] | None = None, *, report: Callable[[str], None] = _print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.site_brand",
        description="Make only the author's name a link in the built site's top bar.",
    )
    parser.add_argument("--built-dir", type=Path, default=Path("site/_site"))
    args = parser.parse_args(argv)
    try:
        changed = rewrite_site(args.built_dir)
    except SiteBrandError as exc:
        report(f"error: {exc}")
        return 1
    report(f"rewrote the brand on {changed} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
