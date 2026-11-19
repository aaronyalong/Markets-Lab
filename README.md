# Markets Lab

A hobby research site about markets. Everything here is measurement and analysis:
each result is compared against a baseline, reported with confidence intervals, and
negative results are published as findings.

> **Not financial advice.** This project is for research and education only. It
> produces no buy or sell signals and no trading recommendations. The code never
> places trades and never holds trading credentials. Nothing here should be used to
> make financial decisions.

## Status

Work in progress.

| Part | State |
|---|---|
| NVIDIA next-day direction experiment | Backtest done; the live prediction log starts with the first nightly run |
| Earnings explainer (cited summaries of SEC filings) | Planned, not started |

## What it will contain

### NVIDIA predictor (experiment)

Can machine learning beat naive baselines at predicting whether NVDA closes up or
down the next day?

- Naive baselines first: always-up, persistence, a moving-average rule, and
  buy-and-hold.
- Then logistic regression, XGBoost on technical features, and an LSTM, each added
  only after the previous one has been measured.
- Walk-forward validation only, tests designed to fail on look-ahead leakage, and
  transaction costs included.
- A nightly job scores yesterday's prediction, makes tomorrow's, and appends both to
  a prediction log that is never edited afterwards. The log records direction only:
  date, model, predicted direction, probability, and whether it was right.
- "No model beats buy-and-hold" is a valid outcome and will be reported as such.

### Earnings explainer (planned)

Plain-English summaries of company 10-Q and 10-K filings from SEC EDGAR, where
every sentence cites its source passage and every extracted figure is checked
against the filing's XBRL data.

## Data and publishing rules

- Raw price and market data stays on the machine that collected it and is
  gitignored. The site shows aggregates and the prediction log only.
- No prices or daily returns are published. Simulated returns appear only as
  summary figures (for example total and annualised return against buy-and-hold,
  with intervals).
- Test fixtures are synthetic. No real market data is stored in this repository.

## Repository layout

```
src/markets_lab/
  collectors/   data fetching (daily NVDA prices)
  analysis/     features, validation, and models
tests/          pytest suite
data/           raw local data (gitignored)
site/           Quarto website (pages, theme); build output is gitignored
predictions/    the live prediction log, appended to by the nightly job
reports/        backtest aggregates shown on the site
```

### The site

The site is a [Quarto](https://quarto.org) website. With Quarto installed:

```bash
uv run python -m markets_lab.analysis.nvda.site_page   # first time only, see below
quarto preview site   # live preview while editing
quarto render site    # build to site/_site/
```

The NVIDIA page's numbers, tables and charts are written by
`markets_lab.analysis.nvda.site_page` from `predictions/nvda.jsonl` and
`reports/nvda_backtest.json` into `site/_generated/` (gitignored). Quarto reruns it
before every render, but the files must exist once before Quarto will start, hence
the first command.

### Deploying

The site is served by a Cloudflare Worker that holds only static files
(`wrangler.jsonc`; there is no Worker script). `.github/workflows/deploy.yml`
renders the site and publishes it on every push to `main`, after each successful
nightly run, and on demand from the Actions tab. It publishes nothing unless the
tests pass against the built pages and the prediction log verifies.

The workflow needs two repository secrets (Settings → Secrets and variables →
Actions):

- `CLOUDFLARE_API_TOKEN`: a Cloudflare API token created with the "Edit Cloudflare
  Workers" permissions, limited to the one account.
- `CLOUDFLARE_ACCOUNT_ID`: the account ID shown in the Cloudflare dashboard.

To publish by hand from a machine with Quarto and Node installed:

```bash
uv run python -m markets_lab.analysis.nvda.site_page
quarto render site
npx wrangler deploy   # asks you to log in to Cloudflare the first time
```

## Development

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for you.

```bash
uv sync              # create .venv and install dependencies
uv run pytest        # tests
uv run ruff check    # lint
uv run ruff format   # format
uv run pyright       # type check
```

Secrets, if any are ever needed, go in a local `.env` file (see `.env.example`).
`.env` is gitignored.

## License

[MIT](LICENSE)
