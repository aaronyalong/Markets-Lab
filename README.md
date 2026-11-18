# Markets Lab

A hobby research project with two small experiments about markets. Everything here
is measurement and analysis: each result is compared against a baseline, reported
with confidence intervals, and negative results are published as findings.

> **Not financial advice.** This project is for research and education only. It
> produces no buy or sell signals and no trading recommendations. The code never
> places trades and never holds trading credentials. Nothing here should be used to
> make financial decisions.

## Status

Early scaffold. No data has been collected and there are no results yet.

## Experiments

### 1. Are prediction-market favorites priced right?

In 15-minute crypto Up/Down prediction markets, when one side is priced at 80¢ or
more at 12 minutes 30 seconds into the window, does it win more often than its price
implies once fees and spread are counted?

A read-only recorder will log prices, depth, and outcomes. The analysis will report
win rate by price bucket, calibration curves, Wilson confidence intervals,
fee-inclusive simulated profit and loss, and the sample size each bucket needs
before any conclusion is drawn.

### 2. Can ML beat naive baselines on NVIDIA's next-day direction?

Daily NVDA price data, with naive baselines first (always-up, persistence, a
moving-average rule, buy-and-hold), then logistic regression, XGBoost, and an LSTM.
Evaluation is walk-forward only, with tests designed to fail on lookahead leakage,
and transaction costs included.

## Repository layout

```
src/markets_lab/
  collectors/   read-only market data recorder and price fetchers
  analysis/     calibration, backtests, and models
tests/          pytest suite
data/           raw local databases (gitignored)
```

A static site presenting the results will be added under `site/`. It will publish
aggregated results only, not raw market data.

## Development

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for you.

```bash
uv sync              # create .venv and install dev tools
uv run pytest        # tests
uv run ruff check    # lint
uv run ruff format   # format
uv run pyright       # type check
```

Secrets, if any are ever needed, go in a local `.env` file (see `.env.example`).
`.env` is gitignored.

## License

[MIT](LICENSE)
