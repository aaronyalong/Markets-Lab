"""Guards on the deploy setup, so a later edit cannot quietly weaken it.

The site is published by .github/workflows/deploy.yml to a Cloudflare Worker
described in wrangler.jsonc. These checks read the two files as text: nothing is
deployed and no network is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"
NIGHTLY = ROOT / ".github" / "workflows" / "nightly.yml"
NEWS = ROOT / ".github" / "workflows" / "news.yml"
WRANGLER = ROOT / "wrangler.jsonc"


def wrangler_config() -> dict[str, object]:
    # The file is JSON with whole-line comments.
    lines = [ln for ln in WRANGLER.read_text().splitlines() if not ln.lstrip().startswith("//")]
    return json.loads("\n".join(lines))


def steps() -> str:
    """The workflow's steps, without the explanatory header."""
    return WORKFLOW.read_text().split("\n    steps:\n", 1)[1]


class TestWranglerConfig:
    def test_it_serves_the_rendered_site_and_nothing_else(self) -> None:
        config = wrangler_config()
        assets = config["assets"]
        assert isinstance(assets, dict)
        assert assets["directory"] == "./site/_site"
        assert assets["not_found_handling"] == "404-page"
        # No Worker script and no bindings: the deployment only serves files.
        assert set(config) == {"name", "compatibility_date", "routes", "assets"}

    def test_it_is_served_on_one_custom_domain(self) -> None:
        config = wrangler_config()
        assert config["routes"] == [{"pattern": "markets.aaronyalong.com", "custom_domain": True}]
        # With a route set, the workers.dev address is off unless turned back on.
        assert "workers_dev" not in config

    def test_the_route_matches_the_address_quarto_builds_for(self) -> None:
        routes = wrangler_config()["routes"]
        assert isinstance(routes, list)
        quarto = (ROOT / "site" / "_quarto.yml").read_text()
        assert f'site-url: "https://{routes[0]["pattern"]}"' in quarto

    def test_clean_addresses_are_what_the_sitemap_is_rewritten_for(self) -> None:
        # markets_lab.site_urls strips ".html" because this setting redirects it.
        assets = wrangler_config()["assets"]
        assert isinstance(assets, dict)
        assert assets["html_handling"] == "auto-trailing-slash"

    def test_the_site_has_a_page_for_missing_addresses(self) -> None:
        assert (ROOT / "site" / "404.qmd").is_file()

    def test_no_account_details_are_written_in_the_repository(self) -> None:
        assert "account_id" not in WRANGLER.read_text()


class TestDeployWorkflow:
    def test_credentials_come_from_secrets(self) -> None:
        text = WORKFLOW.read_text()
        assert "apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}" in text
        assert "accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}" in text
        # A 32-character hex string would be an account id or token pasted in.
        assert not re.search(r"\b[0-9a-f]{32,}\b", text)

    def test_it_cannot_write_to_the_repository(self) -> None:
        text = WORKFLOW.read_text()
        assert "permissions:\n  contents: read" in text
        assert "git push" not in text and "git commit" not in text

    def test_generated_parts_are_written_before_the_render(self) -> None:
        text = steps()
        generate = text.index("nvda.site_page --site-dir site")
        assert generate < text.index("quarto render site")

    def test_nothing_is_published_before_the_checks_pass(self) -> None:
        text = steps()
        render = text.index("quarto render site")
        tests = text.index("uv run pytest")
        verify = text.index("prediction_log verify")
        publish = text.index("cloudflare/wrangler-action")
        # Tests run after the render so the built pages are checked too.
        assert render < tests < publish
        assert verify < publish
        assert "continue-on-error" not in text

    def test_it_redeploys_after_each_job_that_commits_new_data(self) -> None:
        # Those jobs push with the workflow token, which starts no push workflows,
        # so the deploy has to be triggered by name.
        text = WORKFLOW.read_text()
        names = []
        for workflow in (NIGHTLY, NEWS):
            name = re.search(r"^name: (.+)$", workflow.read_text(), re.MULTILINE)
            assert name is not None
            names.append(f'"{name.group(1)}"')
        assert f"workflows: [{', '.join(names)}]" in text
        assert "github.event.workflow_run.conclusion == 'success'" in text
        # A nightly run can commit calls and still end as a failure (only the
        # context models skipped), so the site is rebuilt after every nightly run.
        assert "|| github.event.workflow_run.name == 'Nightly NVDA prediction'" in text
        # A run started by the nightly job must build the commit it just pushed.
        assert "ref: main" in text
