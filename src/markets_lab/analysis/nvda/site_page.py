"""Build the data-driven parts of the NVIDIA page.

The page itself is a Quarto document. Everything on it that comes from data is
written here, from two published sources and nothing else:

- the prediction log (calls and outcomes; directions only),
- the backtest report (aggregates only), and
- the collected headlines (headline, publisher, time and link only).

Each fragment is a small file of HTML that the page includes. Sentences that state
a finding are generated from the numbers, so the wording cannot drift from the
results. No prices and no returns series are read or written here.

Quarto runs this before every render (see `site/_quarto.yml`):

    uv run python -m markets_lab.analysis.nvda.site_page --site-dir site
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from markets_lab.analysis.nvda.baselines import default_baselines
from markets_lab.analysis.nvda.metrics import wilson_interval
from markets_lab.analysis.nvda.prediction_log import (
    LogError,
    Record,
    read_log,
    scored_predictions,
)
from markets_lab.collectors.nvda_news import Collection, NewsError, NewsItem, load_recent

GENERATED_DIR = "_generated"
BUY_AND_HOLD = "always_up"
SESSIONS_TO_TELL_55_FROM_50 = 385
NEWS_DAYS_SHOWN = 3
NEWS_ITEMS_SHOWN = 8

DISPLAY_NAMES = {
    "always_up": "Always up",
    "persistence": "Persistence",
    "ma_200": "200-day average rule",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "logistic_ctx": "Logistic regression + context",
    "xgboost_ctx": "XGBoost + context",
    "lstm_ctx": "LSTM + context",
}
DESCRIPTIONS = {
    "always_up": "Calls up every day. Following it is buy-and-hold.",
    "persistence": "Calls whatever the last session did.",
    "ma_200": "Up when the price is above its 200-day average.",
    "logistic": "A weighted sum of the features.",
    "xgboost": "Boosted decision trees on the features.",
    "lstm": "A neural network reading the last 20 sessions.",
    "logistic_ctx": "The same weighted sum, also given market indices, the VIX and results dates.",
    "xgboost_ctx": "The same boosted trees, also given market indices, the VIX and results dates.",
    "lstm_ctx": "The same network, also given market indices, the VIX and results dates.",
}
CONTEXT_SUFFIX = "_ctx"
BASELINE_NAMES = frozenset(b.name for b in default_baselines())

# Site palette (see site/theme.scss). The theme's emerald green and its red mean
# up and down and nothing else, so chart series use lime, amber and white.
TEXT, MUTED, BORDER, PANEL = "#e9edf3", "#94a0b4", "#1f2630", "#0e1217"
LIME, AMBER = "#9be564", "#ffb648"
MODEL_COLOURS = (LIME, AMBER, TEXT, "#ff9ecb", "#d9c27a")
BASELINE_COLOURS = ("#8793a6", "#6c7789", "#566070")
CHART_FONT = "IBM Plex Sans, system-ui, sans-serif"


@dataclass(frozen=True, slots=True)
class TrackRow:
    """One predictor's live record: calls that have been scored so far."""

    name: str
    scored: int
    hits: int
    accuracy: float | None
    interval: tuple[float, float] | None


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def kind_of(name: str) -> str:
    return "baseline" if name in BASELINE_NAMES else "model"


def ordered(names: Sequence[str]) -> list[str]:
    """Known names in report order (baselines, then models), then any others."""
    known = [n for n in DISPLAY_NAMES if n in names]
    return known + sorted(n for n in set(names) if n not in DISPLAY_NAMES)


# ------------------------------------------------------------------- track record


def track_record(scored: pd.DataFrame) -> list[TrackRow]:
    """Live accuracy per predictor, with a Wilson interval, from scored calls only."""
    done = scored[scored["correct"].notna()]
    rows: list[TrackRow] = []
    for name in ordered(list(scored["model"].unique())):
        results = done.loc[done["model"] == name, "correct"].astype(bool)
        count, hit_count = len(results), int(results.sum())
        rows.append(
            TrackRow(
                name=name,
                scored=count,
                hits=hit_count,
                accuracy=hit_count / count if count else None,
                interval=wilson_interval(hit_count, count) if count else None,
            )
        )
    return rows


def cumulative_scores(scored: pd.DataFrame) -> dict[str, tuple[list[date], list[int]]]:
    """Per predictor: the sessions called, and hits minus misses so far after each."""
    done = scored[scored["correct"].notna()]
    series: dict[str, tuple[list[date], list[int]]] = {}
    for name in ordered(list(done["model"].unique())):
        mine = done[done["model"] == name].sort_values("target_session")
        steps = [1 if bool(c) else -1 for c in mine["correct"]]
        running = [sum(steps[: i + 1]) for i in range(len(steps))]
        series[name] = (list(mine["target_session"]), running)
    return series


# ---------------------------------------------------------------------- findings


def findings(report: Mapping[str, Any]) -> list[str]:
    """Plain-English findings from the backtest, worded from the numbers.

    A model "beat" a benchmark only if the whole 95% interval of the difference is
    above zero, and was "measurably worse" only if the whole interval is below it.
    """
    predictors: Mapping[str, Any] = report["predictors"]
    models = [name for name, row in predictors.items() if row["kind"] == "model"]
    if not models:
        return ["No models have been measured yet; only the baselines."]
    versus_hold = {
        row["predictor"]: row
        for row in report["comparisons"]
        if row["reference"] == BUY_AND_HOLD and row["predictor"] in models
    }
    n = len(models)
    lines: list[str] = []
    for label, key, benchmark in (
        ("accuracy", "accuracy_difference_interval", "always calling up"),
        ("simulated return after costs", "annualised_return_difference_interval", "buy-and-hold"),
    ):
        better = sum(1 for m in models if versus_hold[m][key][0] > 0)
        worse = sum(1 for m in models if versus_hold[m][key][1] < 0)
        unclear = n - better - worse
        sentence = (
            f"{_how_many(better, n).capitalize()} of the {n} models beat {benchmark} on {label}."
        )
        details = []
        if worse:
            details.append(f"{_how_many(worse, n)} {_was(worse)} measurably worse")
        if unclear:
            details.append(f"{_how_many(unclear, n)} {_was(unclear)} within the margin of error")
        if details:
            joined = " and ".join(details)
            sentence += f" {joined[0].upper()}{joined[1:]}."
        lines.append(sentence)

    base_rate = report["data"].get("brier_score_of_base_rate")
    scores = [predictors[m].get("brier_score") for m in models]
    if base_rate is not None and all(score is not None for score in scores):
        useful = sum(1 for score in scores if score < base_rate)
        lines.append(
            f"{_how_many(useful, n).capitalize()} of the {n} models gave probabilities more "
            "informative than always quoting the share of up days."
        )
    return lines


def _how_many(k: int, n: int) -> str:
    """'none', 'one', 'two', 'all 3' or a digit."""
    if k == 0:
        return "none"
    if k == n and n > 1:
        return f"all {n}"
    return {1: "one", 2: "two", 3: "three", 4: "four"}.get(k, str(k))


def _was(k: int) -> str:
    return "was" if k == 1 else "were"


# ---------------------------------------------------------------------- fragments


def tomorrow_fragment(records: Sequence[Record]) -> str:
    """The latest call from each predictor, labelled as an experiment."""
    scored = scored_predictions(records)
    if scored.empty:
        return _raw(
            '<div class="panel muted">The prediction log has not started yet. The first '
            "calls will appear here after the nightly job's first run.</div>"
        )
    latest = max(scored["based_on_session"])
    calls = scored[scored["based_on_session"] == latest].set_index("model")
    pending = bool(calls["correct"].isna().all())
    if pending:
        heading = f"Calls for the next session after {_long_date(latest)}"
        note = "Written before that session opened. It has not been scored yet."
    else:
        target = calls["target_session"].iloc[0]
        heading = f"Latest calls were for {_long_date(target)}"
        note = (
            "That session is over and these have been scored. "
            "New calls appear after the next nightly run."
        )

    cards = []
    for name in ordered(list(calls.index)):
        call = calls.loc[name]
        up = bool(call["predicted_up"])
        probability = call["probability_up"]
        if pd.isna(probability):
            detail = "A fixed rule: no probability."
        else:
            detail = f"Model's probability of up: {100 * float(probability):.0f}%"
        result = ""
        if not pending:
            right = bool(call["correct"])
            label = "Right" if right else "Wrong"
            result = f'<span class="call-result {label.lower()}">{label}</span>'
        cards.append(
            f'<div class="call-card {kind_of(name)}">'
            f'<span class="tag">{kind_of(name)}</span>'
            f"<h4>{html.escape(display_name(name))}</h4>"
            f'<div class="call {_direction_class(up)}">{_direction_text(up)}</div>'
            f'<p class="muted">{html.escape(detail)}</p>{result}'
            "</div>"
        )
    return _raw(
        f'<p class="call-heading">{html.escape(heading)}</p>'
        f'<div class="call-grid">{"".join(cards)}</div>'
        f'<p class="muted small">{html.escape(note)} An experiment, not advice.</p>'
    )


def status_fragment(records: Sequence[Record]) -> str:
    """A one-line summary of the experiment: the models' latest calls and live record.

    Everything in it is labelled as an experiment. It states what each model
    called and how often it has been right, and nothing about what to do.
    """
    label = '<span class="status-label">Experiment · not advice</span>'
    scored = scored_predictions(records)
    if scored.empty:
        body = (
            '<span class="status-item"><span class="status-key">Model calls</span>'
            "start after the first nightly run</span>"
        )
        return _status(label + body)

    latest = max(scored["based_on_session"])
    calls = scored[scored["based_on_session"] == latest].set_index("model")
    record = {row.name: row for row in track_record(scored)}
    items = [
        '<span class="status-item"><span class="status-key">Model calls after the close of'
        f"</span>{html.escape(_short_date(latest))}</span>"
    ]
    for name in ordered(list(calls.index)):
        if kind_of(name) != "model":
            continue
        call = calls.loc[name]
        up = bool(call["predicted_up"])
        probability = call["probability_up"]
        chance = "" if pd.isna(probability) else f" {100 * float(probability):.0f}%"
        row = record.get(name)
        hits = f" · {row.hits} of {row.scored} right" if row is not None and row.scored else ""
        items.append(
            '<span class="status-item">'
            f'<span class="status-key">{html.escape(display_name(name))}</span>'
            f'<span class="{_direction_class(up)}">{_direction_text(up)}</span>'
            f"{chance}{hits}</span>"
        )
    sessions = max((row.scored for row in record.values()), default=0)
    items.append(
        '<span class="status-item"><span class="status-key">Sessions scored</span>'
        f"{sessions}</span>"
    )
    return _status(label + "".join(items))


def _status(inner: str) -> str:
    return _raw(
        '<div class="status-band"><div class="status-strip" role="group" '
        f'aria-label="Experiment status">{inner}</div></div>'
    )


def news_fragment(collections: Sequence[Collection]) -> str:
    """The latest collected headlines, each linking to its publisher.

    Headlines are other people's words. They are escaped, shown as plain text and
    labelled as such; nothing here is written or chosen by a person or a model.
    """
    if not collections:
        return _raw(
            '<div class="panel muted">No headlines have been collected yet. They appear '
            "here after the first daily run.</div>"
        )
    seen: dict[str, NewsItem] = {}
    for collection in collections:
        for item in collection.items:
            seen.setdefault(item.url, item)
    latest = sorted(seen.values(), key=lambda item: item.published_at, reverse=True)
    latest = latest[:NEWS_ITEMS_SHOWN]
    collected = max(collection.collected_at for collection in collections)
    if not latest:
        body = '<div class="panel muted">No NVIDIA headlines in the last few days.</div>'
    else:
        rows = "".join(
            "<li>"
            f'<a href="{html.escape(item.url, quote=True)}" target="_blank" '
            f'rel="noopener nofollow">{html.escape(item.title)}</a>'
            f'<span class="news-meta">{html.escape(item.publisher)} · '
            f"{html.escape(_stamp(item.published_at))}</span>"
            "</li>"
            for item in latest
        )
        body = f'<ul class="news-list">{rows}</ul>'
    note = (
        f"Collected automatically on {_stamp(collected)} from NVIDIA's newsroom, SEC filings "
        "and Yahoo Finance. Headlines are their publishers' words, not this site's, and "
        "are not checked or selected by a person. Not advice."
    )
    return _raw(f'{body}<p class="muted small">{html.escape(note)}</p>')


def track_record_fragment(records: Sequence[Record]) -> str:
    """Live accuracy table and the cumulative hits-minus-misses chart."""
    scored = scored_predictions(records)
    rows = track_record(scored) if not scored.empty else []
    sessions_scored = max((row.scored for row in rows), default=0)
    if not rows:
        return _raw('<div class="panel muted">No predictions have been logged yet.</div>')

    started = min(scored["based_on_session"])
    intro = (
        f"The log started with calls based on {_long_date(started)}. "
        f"{sessions_scored} session{'s have' if sessions_scored != 1 else ' has'} been scored "
        f"so far. It takes about {SESSIONS_TO_TELL_55_FROM_50} scored sessions to tell 55% "
        "accuracy from a coin flip, so early numbers mean very little."
    )
    body = "".join(
        "<tr>"
        f"<td>{html.escape(display_name(row.name))}</td>"
        f'<td class="muted">{kind_of(row.name)}</td>'
        f"<td>{row.scored}</td><td>{row.hits}</td>"
        f"<td>{_percent(row.accuracy)}</td><td>{_interval(row.interval)}</td>"
        "</tr>"
        for row in rows
    )
    table = (
        '<table class="lab-table"><thead><tr><th>Predictor</th><th>Type</th>'
        "<th>Scored calls</th><th>Right</th><th>Accuracy</th><th>95% interval</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )
    series = cumulative_scores(scored)
    chart = (
        _figure_html(cumulative_figure(series), "track-record-chart")
        if any(len(points[1]) for points in series.values())
        else '<div class="panel muted">The chart appears once the first calls are scored.</div>'
    )
    return _raw(f'<p class="muted">{html.escape(intro)}</p>{table}{chart}')


def backtest_fragment(report: Mapping[str, Any]) -> str:
    """Headline numbers, findings, the accuracy chart and the results table."""
    data, predictors = report["data"], report["predictors"]
    cost = str(report["primary_cost_bps"])
    models = [n for n, row in predictors.items() if row["kind"] == "model"]
    versus_hold = {
        row["predictor"]: row for row in report["comparisons"] if row["reference"] == BUY_AND_HOLD
    }
    beat = sum(1 for m in models if versus_hold[m]["accuracy_difference_interval"][0] > 0)
    first, last = data["first_tested_session"][:4], data["last_tested_session"][:4]
    tiles = "".join(
        f'<div class="stat"><div class="big-number">{value}</div><p class="muted">{label}</p></div>'
        for value, label in (
            (f"{data['tested_sessions']:,}", f"sessions tested, {first} to {last}"),
            (_percent(data["share_of_up_days_tested"]), "of those days were up: the bar to beat"),
            (f"{beat} of {len(models)}", "models beat always calling up"),
        )
    )
    points = "".join(f"<li>{html.escape(line)}</li>" for line in findings(report))

    rows = []
    for name in ordered(list(predictors)):
        row = predictors[name]
        returns = row["returns_by_cost_bps"][cost]
        gap = versus_hold.get(name)
        difference = (
            '<span class="muted">benchmark</span>'
            if gap is None
            else f"{_points(gap['annualised_return_difference'])} "
            f"{_interval(gap['annualised_return_difference_interval'], points=True)}"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(display_name(name))}<br><span class='muted small'>"
            f"{html.escape(DESCRIPTIONS.get(name, ''))}</span></td>"
            f"<td>{_percent(row['accuracy'])}<br><span class='muted small'>"
            f"{_interval(row['accuracy_interval'])}</span></td>"
            f"<td>{_percent(returns['annualised_return'])}<br><span class='muted small'>"
            f"{_interval(returns['annualised_return_interval'])}</span></td>"
            f"<td>{difference}</td>"
            "</tr>"
        )
    table = (
        '<table class="lab-table"><thead><tr><th>Predictor</th><th>Accuracy</th>'
        f"<th>Simulated return per year<br><span class='muted small'>after {cost} basis "
        "points per trade</span></th><th>Return vs buy-and-hold</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return _raw(
        f'<div class="stat-grid">{tiles}</div>'
        f'<ul class="findings">{points}</ul>'
        f"{_figure_html(accuracy_figure(report), 'backtest-accuracy-chart')}"
        f"{table}"
        '<p class="muted small">Ranges in brackets are 95% intervals. Simulated returns are '
        "summary figures from applying each rule to history; they are not real results.</p>"
    )


def plotly_loader_fragment() -> str:
    """One script tag that loads the charting library before any chart on the page."""
    version = get_plotlyjs_version()
    return _raw(
        f'<script charset="utf-8" src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'
    )


def build_fragments(
    report: Mapping[str, Any],
    records: Sequence[Record],
    news: Sequence[Collection] = (),
) -> dict[str, str]:
    """Every generated file for the NVIDIA page, by file name."""
    return {
        "plotly.md": plotly_loader_fragment(),
        "status.md": status_fragment(records),
        "news.md": news_fragment(news),
        "tomorrow.md": tomorrow_fragment(records),
        "track_record.md": track_record_fragment(records),
        "backtest.md": backtest_fragment(report),
    }


# ------------------------------------------------------------------------ figures


def cumulative_figure(series: Mapping[str, tuple[list[date], list[int]]]) -> go.Figure:
    """Hits minus misses over time, one line per predictor."""
    figure = go.Figure()
    models = [n for n in series if kind_of(n) == "model"]
    baselines = [n for n in series if kind_of(n) == "baseline"]
    for position, name in enumerate(baselines):
        sessions, running = series[name]
        figure.add_trace(
            go.Scatter(
                x=[s.isoformat() for s in sessions],
                y=running,
                name=display_name(name),
                mode="lines",
                line={
                    "color": BASELINE_COLOURS[position % len(BASELINE_COLOURS)],
                    "width": 2,
                    "dash": "dot",
                },
            )
        )
    # A context model is drawn in its twin's colour with a dashed line, so the
    # pair reads as a pair and no line is told apart by colour alone.
    own_colour = [n for n in models if _twin(n) not in models]
    for name in models:
        sessions, running = series[name]
        twin = _twin(name)
        paired = twin is not None and twin in models
        position = own_colour.index(twin if twin is not None and paired else name)
        figure.add_trace(
            go.Scatter(
                x=[s.isoformat() for s in sessions],
                y=running,
                name=display_name(name),
                mode="lines+markers",
                line={
                    "color": MODEL_COLOURS[position % len(MODEL_COLOURS)],
                    "width": 3,
                    "dash": "dash" if paired else "solid",
                },
                marker={"size": 5},
            )
        )
    figure.add_hline(y=0, line={"color": BORDER, "width": 1})
    _style(figure, height=380)
    figure.update_yaxes(title_text="Right calls minus wrong calls")
    return figure


def _twin(name: str) -> str | None:
    """The technical model a context model is paired with, by its name."""
    return name.removesuffix(CONTEXT_SUFFIX) if name.endswith(CONTEXT_SUFFIX) else None


def accuracy_figure(report: Mapping[str, Any]) -> go.Figure:
    """Backtest accuracy per predictor with its 95% interval, against the bar to beat."""
    predictors = report["predictors"]
    names = list(reversed(ordered(list(predictors))))
    bar = report["data"]["share_of_up_days_tested"]
    figure = go.Figure()
    for name in names:
        row = predictors[name]
        low, high = row["accuracy_interval"]
        is_model = row["kind"] == "model"
        figure.add_trace(
            go.Scatter(
                x=[100 * row["accuracy"]],
                y=[display_name(name)],
                mode="markers",
                marker={"size": 13, "color": LIME if is_model else MUTED},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [100 * (high - row["accuracy"])],
                    "arrayminus": [100 * (row["accuracy"] - low)],
                    "color": LIME if is_model else MUTED,
                    "thickness": 2,
                    "width": 6,
                },
                showlegend=False,
                hovertemplate=f"{display_name(name)}: %{{x:.1f}}%<extra></extra>",
            )
        )
    figure.add_vline(
        x=100 * bar,
        line={"color": AMBER, "width": 2, "dash": "dash"},
        annotation_text=f"Always up: {100 * bar:.1f}%",
        annotation_position="top",
        annotation_font={"color": AMBER},
    )
    figure.add_vline(x=50, line={"color": BORDER, "width": 1})
    _style(figure, height=90 + 46 * len(names))
    figure.update_xaxes(
        title_text="Accuracy in the backtest (%), with 95% interval", ticksuffix="%"
    )
    return figure


def _style(figure: go.Figure, *, height: int) -> None:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT, "color": TEXT, "size": 14},
        margin={"l": 10, "r": 20, "t": 40, "b": 10},
        legend={"orientation": "h", "y": -0.2, "font": {"color": MUTED}},
        hoverlabel={"bgcolor": PANEL, "bordercolor": BORDER, "font": {"color": TEXT}},
    )
    figure.update_xaxes(gridcolor=BORDER, zeroline=False, linecolor=BORDER, automargin=True)
    figure.update_yaxes(gridcolor=BORDER, zeroline=False, linecolor=BORDER, automargin=True)


def _figure_html(figure: go.Figure, element_id: str) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=element_id,
        config={"displayModeBar": False, "responsive": True},
    )


# ------------------------------------------------------------------------ helpers


def _raw(markup: str) -> str:
    """Wrap HTML so Quarto passes it through untouched."""
    return f"```{{=html}}\n{markup}\n```\n"


def _direction_class(up: bool) -> str:
    return "up" if up else "down"


def _direction_text(up: bool) -> str:
    """Arrow and word together, so direction never depends on colour alone."""
    return "▲ Up" if up else "▼ Down"


def _long_date(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B %Y}"


def _stamp(moment: datetime) -> str:
    return f"{moment.day} {moment:%b %Y, %H:%M} UTC"


def _short_date(day: date) -> str:
    return f"{day:%a} {day.day} {day:%b %Y}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f} pts"


def _interval(interval: Sequence[float | None] | None, *, points: bool = False) -> str:
    if interval is None or interval[0] is None or interval[1] is None:
        return "n/a"
    low, high = 100 * interval[0], 100 * interval[1]
    return f"({low:+.1f} to {high:+.1f})" if points else f"({low:.1f}% to {high:.1f}%)"


# ------------------------------------------------------------------- command line


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: Sequence[str] | None = None, *, report: Callable[[str], None] = _print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m markets_lab.analysis.nvda.site_page",
        description="Write the data-driven fragments of the NVIDIA page.",
    )
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--log", type=Path, default=None, help="default: predictions/nvda.jsonl")
    parser.add_argument(
        "--report", type=Path, default=None, help="default: reports/nvda_backtest.json"
    )
    parser.add_argument("--news-dir", type=Path, default=None, help="default: news/nvda")
    args = parser.parse_args(argv)
    site_dir: Path = args.site_dir.resolve()
    root = site_dir.parent
    log_path: Path = args.log or root / "predictions" / "nvda.jsonl"
    report_path: Path = args.report or root / "reports" / "nvda_backtest.json"
    news_dir: Path = args.news_dir or root / "news" / "nvda"
    try:
        if not report_path.exists():
            raise FileNotFoundError(f"{report_path} not found; run the backtest first")
        backtest = json.loads(report_path.read_text(encoding="utf-8"))
        records = read_log(log_path)
        news = load_recent(news_dir, days=NEWS_DAYS_SHOWN)
    except (FileNotFoundError, LogError, NewsError, json.JSONDecodeError) as exc:
        report(f"error: {exc}")
        return 1
    out = site_dir / GENERATED_DIR
    out.mkdir(parents=True, exist_ok=True)
    fragments = build_fragments(backtest, records, news)
    for name, content in fragments.items():
        (out / name).write_text(content, encoding="utf-8")
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    (out / "updated.md").write_text(f"Page data last built {stamp}.\n", encoding="utf-8")
    report(f"wrote {len(fragments) + 1} fragments to {out} ({len(records)} log records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
