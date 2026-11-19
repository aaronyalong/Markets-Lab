# Markets Lab

A hobby research site about markets. Everything here is measurement and analysis:
each result is compared against a baseline, reported with confidence intervals, and
negative results are published as findings.

> **Not financial advice.** This project is for research and education only. It
> produces no buy or sell signals and no trading recommendations. The code never
> places trades and never holds trading credentials. Nothing here should be used to
> make financial decisions.

## Status

Work in progress. Nothing is published yet and there are no results.

| Part | State |
|---|---|
| NVIDIA next-day direction experiment | In progress |
| Earnings explainer (cited summaries of SEC filings) | Planned, not started |
| Polymarket favorites study | Parked; data collection code exists, no analysis |

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

### Polymarket favorites study (parked)

In 15-minute crypto Up/Down prediction markets, when one side is priced at 80¢ or
more at 12 minutes 30 seconds into the window, does it win more often than its price
implies once fees and spread are counted? A read-only backfill of public market data
exists under `src/markets_lab/collectors/polymarket/`. No analysis has been done.

## Data and publishing rules

- Raw price and market data stays on the machine that collected it and is
  gitignored. The site shows aggregates and the prediction log only.
- No prices or daily returns are published. Simulated returns appear only as
  summary figures (for example total and annualised return against buy-and-hold,
  with intervals).
- Test fixtures are synthetic. No real market data is stored in this repository.

### Launch blocker for Polymarket results

No Polymarket results are published until this is resolved. It does not block the
rest of the site, which can go live with the Polymarket study shown as "Coming
soon" and no data.

- [ ] **Data terms.** The Polymarket US participant agreement says market data "may
      not be redistributed except as PMUS expressly permits". Ask
      data@polymarket.us whether publishing aggregated statistics (win rates,
      calibration curves, confidence intervals) is permitted, and record the answer
      here. Raw quotes stay local and gitignored regardless of the answer.

## Repository layout

```
src/markets_lab/
  collectors/   data fetching (polymarket/ is the parked backfill)
  analysis/     features, validation, and models
tests/          pytest suite
data/           raw local data (gitignored)
```

A static site will be added under `site/`.

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
