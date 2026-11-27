"""The data-driven parts of the NVIDIA page: correct numbers, honest wording, no raw data."""

from __future__ import annotations

import copy
import functools
import html
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from context_frames import make_context, make_publications
from markets_lab.analysis.nvda import site_page
from markets_lab.analysis.nvda.backtest import run_backtest
from markets_lab.analysis.nvda.baselines import AlwaysUp, MovingAverageRule, Persistence
from markets_lab.analysis.nvda.context_features import build_context_dataset
from markets_lab.analysis.nvda.features import build_dataset
from markets_lab.analysis.nvda.metrics import BootstrapConfig, wilson_interval
from markets_lab.analysis.nvda.paper_strategy import backtest as paper_backtest
from markets_lab.analysis.nvda.paper_strategy.rule import RULE_MODELS
from markets_lab.analysis.nvda.prediction_log import (
    Outcome,
    PaperStrategyEntry,
    Prediction,
    RangeForecast,
    RangeOutcome,
    Record,
    append_records,
    scored_predictions,
)
from markets_lab.analysis.nvda.site_page import (
    backtest_fragment,
    build_fragments,
    cumulative_scores,
    findings,
    main,
    news_fragment,
    status_fragment,
    tomorrow_fragment,
    track_record,
    track_record_fragment,
)
from markets_lab.analysis.nvda.validation import WalkForwardConfig
from markets_lab.analysis.nvda.volatility import backtest as volatility_backtest
from markets_lab.analysis.nvda.volatility.dataset import (
    build_volatility_dataset,
    compute_ewma_volatility,
)
from markets_lab.collectors.nvda_news import EDGAR, YAHOO_RSS, Collection, NewsItem, save_day
from nvda_test_predictors import LeastSquaresLearner
from price_frames import at, make_prices

THU, FRI, MON, TUE = date(2030, 3, 14), date(2030, 3, 15), date(2030, 3, 18), date(2030, 3, 19)


def call(day: date, model: str, up: bool, probability: float | None = None) -> Prediction:
    return Prediction(at(f"{day} 18:00"), day, model, up, probability)


def went(day: date, previous: date, up: bool) -> Outcome:
    return Outcome(at(f"{day} 17:30"), day, previous, up)


def log_records() -> list[Record]:
    """Three days of calls; Friday and Monday are scored, Tuesday is not yet."""
    return [
        call(THU, "always_up", True),
        call(THU, "logistic", False, 0.47),
        went(FRI, THU, True),  # always_up right, logistic wrong
        call(FRI, "always_up", True),
        call(FRI, "logistic", False, 0.44),
        went(MON, FRI, False),  # always_up wrong, logistic right
        call(MON, "always_up", True),
        call(MON, "logistic", True, 0.5312),
    ]


@functools.cache
def volatility_report() -> dict[str, Any]:
    """The volatility backtest's report on made-up prices, with every forecaster."""
    return volatility_backtest.run_backtest(
        build_volatility_dataset(make_prices(rows=900)),
        config=WalkForwardConfig(initial_train_sessions=300, refit_every=50),
        bootstrap=BootstrapConfig(n_resamples=100),
    )


@functools.cache
def paper_report() -> dict[str, Any]:
    """The paper strategy's backtest report on made-up prices, with stand-in models."""
    rows = 900
    prices = make_prices(rows=rows)
    dataset = build_context_dataset(
        prices, make_context(rows), make_publications(), as_of=at("2030-03-14 22:30")
    )
    return paper_backtest.run_paper_backtest(
        dataset,
        compute_ewma_volatility(prices),
        models={"logistic": LeastSquaresLearner, "xgboost": Persistence, "lstm": AlwaysUp},
        context_models={
            "logistic_ctx": LeastSquaresLearner,
            "xgboost_ctx": MovingAverageRule,
            "lstm_ctx": Persistence,
        },
        config=WalkForwardConfig(initial_train_sessions=300, refit_every=50),
        bootstrap=BootstrapConfig(n_resamples=100),
    )


@pytest.fixture
def volatility() -> dict[str, Any]:
    return copy.deepcopy(volatility_report())


def with_verdicts(report: dict[str, Any], **verdicts: tuple[str, str]) -> dict[str, Any]:
    """The report with each named model's two differences from the simple baseline
    set to "better", "worse" or "same" (volatility score first, then range score)."""
    intervals = {"better": [-0.2, -0.1], "worse": [0.1, 0.2], "same": [-0.1, 0.1]}
    changed = copy.deepcopy(report)
    for row in changed["comparisons"]:
        if row["reference"] == "ewma_94" and row["forecaster"] in verdicts:
            volatility_verdict, range_verdict = verdicts[row["forecaster"]]
            row["qlike_difference_interval"] = intervals[volatility_verdict]
            row["interval_score_difference_interval"] = intervals[range_verdict]
    return changed


def ranged(day: date, model: str, lower: float = -0.03, upper: float = 0.04) -> RangeForecast:
    return RangeForecast(at(f"{day} 18:00"), day, model, 0.8, lower, upper)


def fell(day: date, previous: date, model: str, position: str) -> RangeOutcome:
    return RangeOutcome(at(f"{day} 17:30"), day, previous, model, position)


FORECASTERS = ["constant", "rolling_21", "ewma_94", "har", "xgboost_vol"]


def range_records() -> list[Record]:
    """Two sessions of ranges from every forecaster; the first has been scored."""
    return [
        *(ranged(THU, name) for name in FORECASTERS),
        went(FRI, THU, True),
        *(
            fell(FRI, THU, name, "above" if name == "constant" else "inside")
            for name in FORECASTERS
        ),
        ranged(FRI, "constant", -0.036, 0.04),
        ranged(FRI, "rolling_21", -0.051, 0.056),
        ranged(FRI, "ewma_94", -0.0558, 0.0616),
        ranged(FRI, "har", -0.074, 0.084),
        ranged(FRI, "xgboost_vol", -0.05, 0.055),
    ]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    dataset = build_dataset(make_prices(rows=900))
    return run_backtest(
        dataset,
        models={"logistic": LeastSquaresLearner},
        config=WalkForwardConfig(initial_train_sessions=300, refit_every=50),
        bootstrap=BootstrapConfig(n_resamples=100),
    )


def text_of(fragment: str) -> str:
    """The visible text of a fragment, with tags and scripts removed."""
    without_scripts = re.sub(r"<script.*?</script>", " ", fragment, flags=re.DOTALL)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_scripts)))


def table_rows(fragment: str) -> dict[str, list[str]]:
    """Each body row's cells (inner markup), keyed by the text of its first cell."""
    body = re.search(r"<tbody>(.*?)</tbody>", fragment, flags=re.DOTALL)
    assert body is not None
    rows = {}
    for row in re.findall(r"<tr>(.*?)</tr>", body.group(1), flags=re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)
        rows[text_of(cells[0].split("<br>")[0]).strip()] = cells
    return rows


def table_labels(fragment: str) -> list[tuple[list[str], list[list[str | None]]]]:
    """For every table: its headings' text, and each body row's phone labels."""
    tables = []
    for table in re.findall(r'<table class="lab-table">(.*?)</table>', fragment, re.DOTALL):
        headings = [text_of(h).strip() for h in re.findall(r"<th>(.*?)</th>", table, re.DOTALL)]
        rows = [
            [
                html.unescape(label) if label else None
                for label in re.findall(r'<td(?: data-label="([^"]*)")?[^>]*>', row)
            ]
            for row in re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1], re.DOTALL)
        ]
        tables.append((headings, rows))
    return tables


def chart_lines(figure: Any) -> dict[str, tuple[str, str]]:
    """Each line of a chart by its legend name: (colour, dash style)."""
    traces = figure.to_dict()["data"]
    return {trace["name"]: (trace["line"]["color"], trace["line"]["dash"]) for trace in traces}


class TestTomorrow:
    def test_before_the_log_starts(self) -> None:
        assert "has not started yet" in tomorrow_fragment([])

    def test_shows_the_latest_unscored_calls(self) -> None:
        text = text_of(tomorrow_fragment(log_records()))
        assert "Calls for Tuesday 19 March 2030" in text
        assert "Made after the close on Monday 18 March 2030." in text
        assert "Logistic regression" in text and "Always up" in text
        assert "▲ Up" in text
        assert "Model's probability of up: 53%" in text
        assert "A fixed rule: no probability." in text
        assert "Not scored yet." in text

    def test_the_calls_do_not_repeat_the_notice_under_the_page_title(self) -> None:
        # My request (2026-11-26): the page says "an experiment, not advice" once,
        # under its title, and nowhere else.
        for records in (log_records(), [*log_records(), went(TUE, MON, True)]):
            fragment = tomorrow_fragment(records)
            assert "advice" not in text_of(fragment).lower()
            assert re.search(
                r'<div class="call-head"><p class="call-heading">Calls for .*?</p></div>',
                fragment,
            )
        assert "Written before that session opened." in text_of(tomorrow_fragment(log_records()))

    def test_once_scored_it_says_so_and_shows_the_result(self) -> None:
        text = text_of(tomorrow_fragment([*log_records(), went(TUE, MON, False)]))
        assert "Calls for Tuesday 19 March 2030" in text
        assert "these have been scored" in text
        assert "Wrong" in text  # both called up; the session did not close up
        assert "Not scored yet." not in text

    def test_a_scored_date_comes_from_the_log_and_gets_no_today_or_tomorrow(self) -> None:
        # The log says the session after Monday was Wednesday (say Tuesday closed
        # at short notice); the page must follow the log, not the holiday rules.
        wednesday = date(2030, 3, 20)
        fragment = tomorrow_fragment([*log_records(), went(wednesday, MON, True)])
        assert "Calls for Wednesday 20 March 2030" in text_of(fragment)
        assert "data-certain" not in fragment

    def test_the_date_skips_weekends_and_holidays(self) -> None:
        # Friday 20 November 2026 -> Monday 23; Wednesday 25 -> Friday 27, past
        # Thanksgiving.
        for based_on, expected in (
            (date(2026, 11, 20), "Monday 23 November 2026"),
            (date(2026, 11, 25), "Friday 27 November 2026"),
        ):
            text = text_of(tomorrow_fragment([call(based_on, "logistic", True, 0.6)]))
            assert f"Calls for {expected}" in text

    def test_a_date_inside_nyses_published_calendar_can_say_today_or_tomorrow(self) -> None:
        fragment = tomorrow_fragment([call(date(2026, 11, 23), "logistic", True, 0.6)])
        assert (
            '<time class="session-date" datetime="2026-11-24" data-certain="true">'
            "Tuesday 24 November 2026</time>"
        ) in fragment
        # Shown by the page's script only once the session has closed.
        assert re.search(
            r'<span class="session-closed" data-session="2026-11-24" data-certain="true" '
            r"hidden> That session has closed; the calls are scored on the next nightly run\.",
            fragment,
        )

    def test_a_date_outside_the_published_calendar_stays_plain(self) -> None:
        # 2030 is beyond the years NYSE has published, so the date is shown without
        # "today" or "tomorrow" and without the closed-session line.
        fragment = tomorrow_fragment(log_records())
        assert '<time class="session-date" datetime="2030-03-19">' in fragment
        assert "data-certain" not in fragment
        assert "session-closed" not in fragment

    def test_a_next_session_in_an_unpublished_year_is_not_certain(self) -> None:
        # Friday 29 December 2028 -> Tuesday 2 January 2029, after NYSE's list ends.
        fragment = tomorrow_fragment([call(date(2028, 12, 29), "logistic", True, 0.6)])
        assert "Calls for Tuesday 2 January 2029" in text_of(fragment)
        assert "data-certain" not in fragment

    def test_models_and_baselines_are_two_labelled_groups_models_first(self) -> None:
        fragment = tomorrow_fragment(log_records())
        labels = re.findall(r'<p class="call-group-label">([^<]*)', fragment)
        assert labels == ["Models", "Baselines"]
        models, baselines = fragment.split('<p class="call-group-label">')[1:]
        assert "Logistic regression" in models and "Always up" not in models
        assert "Always up" in baselines and "Logistic regression" not in baselines
        assert "the bar to beat" in text_of(baselines)

    def test_a_group_with_no_members_is_left_out(self) -> None:
        fragment = tomorrow_fragment([call(MON, "logistic", True, 0.6)])
        assert re.findall(r'<p class="call-group-label">([^<]*)', fragment) == ["Models"]

    def test_columns_divide_the_cards_evenly_where_they_can(self) -> None:
        assert [site_page.grid_columns(n) for n in range(1, 10)] == [1, 2, 3, 4, 3, 3, 3, 4, 3]
        fragment = tomorrow_fragment(log_records())
        assert '<div class="call-grid" style="--cols: 1">' in fragment

    def test_down_calls_are_shown_as_down(self) -> None:
        text = text_of(tomorrow_fragment(log_records()[:2]))
        assert "▼ Down" in text
        assert "Model's probability of up: 47%" in text

    def test_direction_never_relies_on_colour_alone(self) -> None:
        # Up and down are coloured green and red by the theme, so every call must
        # also carry an arrow and the word.
        fragment = tomorrow_fragment(log_records()[:2])
        assert 'class="call up">▲ Up<' in fragment
        assert 'class="call down">▼ Down<' in fragment
        calls = re.findall(r'class="call (?:up|down)">([^<]*)<', fragment)
        assert calls and all(re.fullmatch(r"[▲▼] (Up|Down)", text) for text in calls)

    def test_the_calls_use_no_wording_that_reads_as_a_signal(self) -> None:
        text = text_of(tomorrow_fragment(log_records())).lower()
        assert not re.search(r"\b(buy|sell|signal|recommend\w*)\b", text)


def versus_always_up(**intervals: tuple[float, float]) -> dict[str, Any]:
    """A report holding only each model's accuracy difference from always calling up."""
    return {
        "comparisons": [
            {"predictor": name, "reference": "always_up", "accuracy_difference_interval": list(i)}
            for name, i in intervals.items()
        ]
    }


BEAT, SAME, WORSE = (0.01, 0.03), (-0.01, 0.01), (-0.03, -0.01)


def three_models_on_monday() -> list[Record]:
    """Monday's calls: two of three models say up; the baseline is not a model."""
    return [
        *log_records(),
        call(MON, "xgboost", False, 0.44),
        call(MON, "lstm", True, 0.60),
    ]


class TestStancePanel:
    def test_counts_and_averages_the_models_only(self) -> None:
        panel = site_page.stance_panel(three_models_on_monday(), versus_always_up())
        text = text_of(panel)
        assert "2 of 3 models say up" in text  # always_up's call is not counted
        mean = (0.5312 + 0.44 + 0.60) / 3
        assert f"{100 * mean:.0f}% their average probability of up" in text

    def test_says_said_once_the_session_is_scored(self) -> None:
        scored = [*log_records()[:5], call(FRI, "xgboost", True, 0.55), log_records()[5]]
        text = text_of(site_page.stance_panel(scored, versus_always_up()))
        assert "1 of 2 models said up" in text  # Friday's calls, scored by Monday's outcome

    def test_no_panel_without_a_model_call(self) -> None:
        assert site_page.stance_panel([], versus_always_up()) == ""
        only_baseline = [call(THU, "always_up", True)]
        assert site_page.stance_panel(only_baseline, versus_always_up()) == ""

    def test_shows_the_simple_baselines_range_for_the_same_session_only(self) -> None:
        same = [*three_models_on_monday(), ranged(MON, "ewma_94", -0.0558, 0.0616)]
        text = text_of(site_page.stance_panel(same, versus_always_up()))
        assert f"{MINUS}5.6% to +6.2%" in text
        assert "the simple baseline's 80% range for the move, either way" in text
        older = [*three_models_on_monday(), ranged(FRI, "ewma_94")]
        assert "range" not in text_of(site_page.stance_panel(older, versus_always_up()))
        other = [*three_models_on_monday(), ranged(MON, "har")]
        assert "range" not in text_of(site_page.stance_panel(other, versus_always_up()))

    def test_the_backtest_sentence_is_in_the_panel_in_full_text(self) -> None:
        report = versus_always_up(logistic=WORSE, xgboost=SAME, lstm=SAME)
        panel = site_page.stance_panel(three_models_on_monday(), report)
        line = re.search(r'<p class="([^"]*)">([^<]*always calling up[^<]*)</p>', panel)
        assert line is not None
        assert line.group(1) == "stance-backtest"  # not muted, not small
        assert line.group(2) == "In the backtest, none of these 3 models beat always calling up."

    @pytest.mark.parametrize(
        ("intervals", "expected"),
        [
            ({"a": SAME, "b": WORSE}, "In the backtest, none of these 2 models beat"),
            ({"a": BEAT, "b": SAME, "c": SAME}, "In the backtest, 1 of these 3 models beat"),
            ({"a": BEAT, "b": BEAT}, "In the backtest, all 2 of these models beat"),
            ({"a": SAME}, "In the backtest, this model did not beat"),
            ({"a": BEAT}, "In the backtest, this model beat"),
        ],
    )
    def test_the_backtest_sentence_follows_the_intervals(
        self, intervals: dict[str, tuple[float, float]], expected: str
    ) -> None:
        line = site_page.stance_backtest_line(list(intervals), versus_always_up(**intervals))
        assert line == f"{expected} always calling up."

    def test_models_without_a_backtest_are_counted_apart(self) -> None:
        report = versus_always_up(a=SAME, b=SAME)
        assert site_page.stance_backtest_line(["a", "b", "new"], report) == (
            "In the backtest, none of these 2 models beat always calling up. "
            "1 more has no backtest yet."
        )
        assert site_page.stance_backtest_line(["new"], report) == (
            "None of these models has been through the backtest yet."
        )

    def test_nothing_in_it_suggests_an_action(self) -> None:
        records = [*three_models_on_monday(), ranged(MON, "ewma_94")]
        panel = site_page.stance_panel(records, versus_always_up(logistic=SAME))
        assert not re.search(r'class="[^"]*\b(up|down|right|wrong)\b', panel)
        assert "▲" not in panel and "▼" not in panel
        assert "big-number" not in panel  # the count-up script would garble "2 of 3"
        words = text_of(panel).lower()
        for word in ("buy", "sell", "hold", "signal", "recommend", "should", "consider"):
            assert not re.search(rf"\b{word}", words), word

    def test_it_sits_under_the_heading_before_the_calls(self, report: dict[str, Any]) -> None:
        fragment = tomorrow_fragment(three_models_on_monday(), report)
        assert fragment.index("call-heading") < fragment.index("stance-panel")
        assert fragment.index("stance-panel") < fragment.index("call-group")
        assert "stance-panel" not in tomorrow_fragment(three_models_on_monday())
        built = build_fragments(report, three_models_on_monday(), volatility=volatility_report())
        assert "stance-panel" in built["tomorrow.md"]

    def test_the_stylesheet_keeps_it_neutral_and_the_sentence_readable(self) -> None:
        theme = (Path(__file__).parents[1] / "site" / "theme.scss").read_text()

        def rule(selector: str) -> str:
            found = re.search(rf"^{re.escape(selector)} \{{(.*?)^\}}", theme, re.M | re.S)
            assert found is not None, selector
            return found.group(1)

        def size(block: str) -> float:
            found = re.search(r"font-size: ([\d.]+)rem", block)
            assert found is not None
            return float(found.group(1))

        sentence = rule(".stance-backtest")
        assert "color: $lab-text" in sentence
        assert size(sentence) >= size(rule(".stance-tile p")) >= 1.0
        for selector in (".stance-panel", ".stance-number", ".stance-backtest"):
            block = rule(selector)
            for colour in ("$lab-green", "$lab-red", "$lab-accent", "$lab-lime", "$lab-muted"):
                assert colour not in block, (selector, colour)


class TestStatusStrip:
    def test_before_the_log_starts(self) -> None:
        text = text_of(status_fragment([]))
        assert "start after the first nightly run" in text

    def test_states_the_next_session_the_model_calls_and_sessions_scored(self) -> None:
        fragment = status_fragment(log_records())
        text = text_of(fragment)
        assert re.search(r"Next\s+Tue 19 Mar\s+1 model call\s+2 sessions scored", text)
        assert '<time class="session-date" datetime="2030-03-19">' in fragment

    def test_counts_models_only(self) -> None:
        records = [
            call(MON, "always_up", True),
            call(MON, "logistic", True, 0.6),
            call(MON, "xgboost", False, 0.4),
        ]
        assert "2 model calls" in text_of(status_fragment(records))

    def test_once_scored_it_shows_the_session_from_the_log(self) -> None:
        text = text_of(status_fragment([*log_records(), went(TUE, MON, True)]))
        assert re.search(r"Last\s+Tue 19 Mar\s+1 model call\s+3 sessions scored", text)

    def test_a_certain_date_is_marked_for_today_or_tomorrow(self) -> None:
        fragment = status_fragment([call(date(2026, 11, 23), "logistic", True, 0.6)])
        assert (
            '<time class="session-date" datetime="2026-11-24" data-certain="true">Tue 24 Nov</time>'
        ) in fragment

    def test_it_never_says_which_way_the_calls_went(self) -> None:
        # My condition (2026-11-24): no up/down tally in the strip until the
        # stance panel is designed. Direction stays in the calls band.
        for records in (log_records(), log_records()[:2], [*log_records(), went(TUE, MON, True)]):
            fragment = status_fragment(records)
            assert "▲" not in fragment and "▼" not in fragment
            assert not re.search(r"\b(up|down)\b", text_of(fragment).lower())
            assert not re.search(r'class="[^"]*\b(up|down)\b', fragment)
            assert "%" not in text_of(fragment)

    def test_it_never_reads_as_a_signal_and_does_not_repeat_the_notice(self) -> None:
        for records in (
            [],
            log_records()[:2],
            log_records(),
            [*log_records(), went(TUE, MON, True)],
        ):
            fragment = status_fragment(records)
            assert 'aria-label="Experiment status"' in fragment
            text = text_of(fragment).lower()
            assert "advice" not in text
            assert "status-label" not in fragment
            assert not re.search(r"\b(buy|sell|signal|recommend\w*|bullish|bearish)\b", text)


def headline(hour: int, title: str, url: str | None = None, *, source: str = YAHOO_RSS) -> NewsItem:
    return NewsItem(
        source=source,
        publisher="Yahoo Finance" if source == YAHOO_RSS else "Example Wire",
        title=title,
        url=url or f"https://news.example.org/{hour}",
        published_at=at(f"{MON} {hour:02d}:00"),
    )


def news_day(*items: NewsItem) -> Collection:
    return Collection(at(f"{TUE} 11:00"), 24, tuple(items), ())


class TestNews:
    def test_before_anything_is_collected(self) -> None:
        assert "No headlines have been collected yet." in text_of(news_fragment([]))

    def test_a_quiet_spell_says_so(self) -> None:
        assert "No NVIDIA headlines in the last few days." in text_of(news_fragment([news_day()]))

    def test_each_headline_links_out_with_its_publisher_and_time(self) -> None:
        fragment = news_fragment([news_day(headline(9, "NVIDIA ships a thing"))])
        assert (
            '<a href="https://news.example.org/9" target="_blank" rel="noopener nofollow">'
            "NVIDIA ships a thing</a>"
        ) in fragment
        text = text_of(fragment)
        assert "via Yahoo Finance ·" in text and "09:00 UTC" in text

    def test_sec_filings_are_credited_to_sec_edgar_without_via(self) -> None:
        filing = NewsItem(
            EDGAR,
            "SEC EDGAR",
            "Form 8-K: current report",
            "https://www.sec.gov/a",
            at(f"{MON} 09:00"),
        )
        text = text_of(news_fragment([news_day(filing)]))
        assert "SEC EDGAR ·" in text and "via SEC EDGAR" not in text

    def test_headlines_from_retired_sources_are_never_shown(self) -> None:
        # NVIDIA's newsroom and Yahoo's unofficial list stopped on 2026-11-24.
        old = [
            headline(10, "From the newsroom", source="nvidia_newsroom"),
            headline(11, "From the unofficial list", source="yahoo_finance"),
        ]
        fragment = news_fragment([news_day(*old, headline(9, "From the feed"))])
        titles = re.findall(r'rel="noopener nofollow">([^<]+)</a>', fragment)
        assert titles == ["From the feed"]
        only_old = text_of(news_fragment([news_day(*old)]))
        assert "No NVIDIA headlines in the last few days." in only_old

    def test_newest_first_capped_and_without_duplicates_across_days(self) -> None:
        many = [headline(hour, f"Headline {hour}") for hour in range(1, 13)]
        repeat = headline(12, "Headline 12")
        fragment = news_fragment([news_day(*many), news_day(repeat)])
        titles = re.findall(r'rel="noopener nofollow">([^<]+)</a>', fragment)
        assert titles == [f"Headline {hour}" for hour in range(12, 4, -1)]

    def test_markup_in_a_headline_is_shown_as_text_never_run(self) -> None:
        hostile = headline(9, '<script>alert(1)</script> "NVIDIA" <img src=x onerror=y>')
        fragment = news_fragment([news_day(hostile)])
        assert "<script>alert" not in fragment and "<img" not in fragment
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in fragment

    def test_a_link_cannot_break_out_of_its_attribute(self) -> None:
        odd = headline(9, "NVIDIA", url='https://news.example.org/a"onmouseover="x')
        fragment = news_fragment([news_day(odd)])
        assert 'onmouseover="x"' not in fragment
        assert "&quot;onmouseover=&quot;x" in fragment

    def test_it_says_whose_words_these_are(self) -> None:
        text = text_of(news_fragment([news_day(headline(9, "NVIDIA ships a thing"))]))
        assert "Collected automatically on 19 Mar 2030, 11:00 UTC" in text
        assert "their publishers' words, not this site's" in text
        assert "advice" not in text.lower()

    def test_the_note_names_what_is_collected_and_links_to_the_newsroom(self) -> None:
        fragment = news_fragment([news_day(headline(9, "NVIDIA ships a thing"))])
        text = text_of(fragment)
        assert "from NVIDIA's SEC filings (SEC EDGAR) and Yahoo Finance's NVDA news feed" in text
        assert "NVIDIA's own press releases are on its newsroom" in text
        assert ">its newsroom</a>.</p>" in fragment
        assert (
            '<a href="https://nvidianews.nvidia.com/" target="_blank" rel="noopener">' in fragment
        )
        assert "newsroom, SEC filings" not in text


class TestTrackRecord:
    def test_counts_only_scored_calls(self) -> None:
        rows = {row.name: row for row in track_record(scored_predictions(log_records()))}
        assert (rows["always_up"].scored, rows["always_up"].hits) == (2, 1)
        assert (rows["logistic"].scored, rows["logistic"].hits) == (2, 1)
        assert rows["logistic"].accuracy == 0.5
        assert rows["logistic"].interval == wilson_interval(1, 2)

    def test_a_predictor_with_nothing_scored_has_no_accuracy(self) -> None:
        records = [*log_records(), call(MON, "xgboost", True, 0.6)]
        rows = {row.name: row for row in track_record(scored_predictions(records))}
        assert (rows["xgboost"].scored, rows["xgboost"].accuracy, rows["xgboost"].interval) == (
            0,
            None,
            None,
        )

    def test_baselines_are_listed_before_models(self) -> None:
        records = [
            call(THU, "lstm", True, 0.6),
            call(THU, "ma_200", True),
            call(THU, "always_up", True),
        ]
        names = [row.name for row in track_record(scored_predictions(records))]
        assert names == ["always_up", "ma_200", "lstm"]

    def test_cumulative_score_is_hits_minus_misses(self) -> None:
        records = [*log_records(), went(TUE, MON, True)]
        series = cumulative_scores(scored_predictions(records))
        assert series["always_up"] == ([FRI, MON, TUE], [1, 0, 1])
        assert series["logistic"] == ([FRI, MON, TUE], [-1, 0, 1])

    def test_a_context_model_is_drawn_as_a_dashed_line_in_its_twins_colour(self) -> None:
        records = [
            call(THU, "logistic", True, 0.6),
            call(THU, "xgboost", True, 0.6),
            call(THU, "lstm", True, 0.6),
            call(THU, "logistic_ctx", True, 0.6),
            call(THU, "xgboost_ctx", False, 0.4),
            call(THU, "lstm_ctx", False, 0.4),
            went(FRI, THU, True),
        ]
        figure = site_page.cumulative_figure(cumulative_scores(scored_predictions(records)))
        lines = chart_lines(figure)
        for twin, context in (
            ("Logistic regression", "Logistic regression + context"),
            ("XGBoost", "XGBoost + context"),
            ("LSTM", "LSTM + context"),
        ):
            assert lines[twin][1] == "solid" and lines[context][1] == "dash"
            assert lines[twin][0] == lines[context][0]
        # Three pairs, three colours: no two pairs share one.
        assert len({colour for colour, _ in lines.values()}) == 3

    def test_a_context_model_without_its_twin_gets_a_colour_of_its_own(self) -> None:
        records = [
            call(THU, "logistic", True, 0.6),
            call(THU, "xgboost_ctx", True, 0.6),
            went(FRI, THU, True),
        ]
        figure = site_page.cumulative_figure(cumulative_scores(scored_predictions(records)))
        lines = chart_lines(figure)
        assert lines["XGBoost + context"][1] == "solid"
        assert lines["XGBoost + context"][0] != lines["Logistic regression"][0]

    def test_the_chart_is_labelled_by_day_not_by_hour(self) -> None:
        figure = site_page.cumulative_figure(cumulative_scores(scored_predictions(log_records())))
        axis = figure.to_dict()["layout"]["xaxis"]
        assert axis["type"] == "date"
        assert axis["tickformat"] == "%-d %b" and "%H" not in axis["hoverformat"]
        assert axis["rangebreaks"] == [{"bounds": ["sat", "mon"]}]
        assert axis["dtick"] == site_page.DAY_MS  # a short span: one tick a day

    def test_a_long_record_lets_the_chart_choose_its_ticks(self) -> None:
        start = date(2030, 1, 1)
        long = {"logistic": ([start, date(2030, 6, 1)], [1, 2])}
        axis = site_page.cumulative_figure(long).to_dict()["layout"]["xaxis"]
        assert "dtick" not in axis
        assert axis["tickformat"] == "%-d %b"

    def test_fragment_states_the_numbers_and_the_sample_size_warning(self) -> None:
        text = text_of(track_record_fragment(log_records()))
        assert "The log started with calls based on Thursday 14 March 2030." in text
        assert "2 sessions have been scored so far." in text
        assert "about 385 scored sessions" in text
        assert "50.0%" in text
        low, high = wilson_interval(1, 2)
        assert f"({100 * low:.1f}% to {100 * high:.1f}%)" in text

    def test_fragment_has_the_chart_once_something_is_scored(self) -> None:
        assert 'id="track-record-chart"' in track_record_fragment(log_records())
        unscored = log_records()[:2]
        fragment = track_record_fragment(unscored)
        assert 'id="track-record-chart"' not in fragment
        assert "0 sessions have been scored" in text_of(fragment)
        assert "The chart appears once the first calls are scored." in text_of(fragment)

    def test_empty_log(self) -> None:
        assert "No predictions have been logged yet." in track_record_fragment([])

    def test_one_scored_session_reads_naturally(self) -> None:
        text = text_of(track_record_fragment(log_records()[:4]))
        assert "1 session has been scored so far." in text


def with_intervals(
    report: dict[str, Any], accuracy: tuple[float, float], returns: tuple[float, float]
) -> dict[str, Any]:
    """A copy of the report where the model's gap to always_up has these intervals."""
    changed = copy.deepcopy(report)
    for row in changed["comparisons"]:
        if row["predictor"] == "logistic" and row["reference"] == "always_up":
            row["accuracy_difference_interval"] = list(accuracy)
            row["annualised_return_difference_interval"] = list(returns)
    return changed


class TestFindings:
    """The sentences are generated from the numbers, so each case is checked."""

    def test_no_detectable_difference(self, report: dict[str, Any]) -> None:
        lines = findings(with_intervals(report, (-0.02, 0.01), (-0.2, 0.1)))
        assert lines[0] == (
            "None of the 1 models beat always calling up on accuracy. "
            "One was within the margin of error."
        )
        assert lines[1].startswith("None of the 1 models beat buy-and-hold on simulated return")

    def test_measurably_worse(self, report: dict[str, Any]) -> None:
        lines = findings(with_intervals(report, (-0.03, -0.001), (-0.3, -0.01)))
        assert lines[0].endswith("One was measurably worse.")
        assert lines[1].endswith("One was measurably worse.")

    def test_a_real_win_is_reported_as_one(self, report: dict[str, Any]) -> None:
        lines = findings(with_intervals(report, (0.005, 0.04), (0.01, 0.3)))
        assert lines[0] == "One of the 1 models beat always calling up on accuracy."
        assert lines[1] == "One of the 1 models beat buy-and-hold on simulated return after costs."

    def test_an_interval_touching_zero_is_not_a_win(self, report: dict[str, Any]) -> None:
        lines = findings(with_intervals(report, (0.0, 0.04), (0.0, 0.3)))
        assert lines[0].startswith("None of the 1 models beat")

    def test_probability_quality_is_compared_with_the_base_rate(
        self, report: dict[str, Any]
    ) -> None:
        good, bad = copy.deepcopy(report), copy.deepcopy(report)
        good["predictors"]["logistic"]["brier_score"] = 0.20
        bad["predictors"]["logistic"]["brier_score"] = 0.30
        assert findings(good)[-1].startswith("One of the 1 models gave probabilities more")
        assert findings(bad)[-1].startswith("None of the 1 models gave probabilities more")

    def test_real_report_wording(self) -> None:
        real = json.loads(
            (Path(__file__).resolve().parents[1] / "reports" / "nvda_backtest.json").read_text()
        )
        for line in findings(real):
            assert re.match(r"^(None|One|Two|Three|All \d+|\d+) of the \d+ models ", line)

    def test_baselines_only(self, report: dict[str, Any]) -> None:
        only = copy.deepcopy(report)
        del only["predictors"]["logistic"]
        only["comparisons"] = [c for c in only["comparisons"] if c["predictor"] != "logistic"]
        assert findings(only) == ["No models have been measured yet; only the baselines."]


class TestBacktestFragment:
    def test_headline_numbers_come_from_the_report(self, report: dict[str, Any]) -> None:
        text = text_of(backtest_fragment(report))
        assert f"{report['data']['tested_sessions']:,}" in text
        assert f"{100 * report['data']['share_of_up_days_tested']:.1f}%" in text
        assert "of 1 models beat always calling up" in text

    def test_every_predictor_has_a_row_with_its_numbers(self, report: dict[str, Any]) -> None:
        text = text_of(backtest_fragment(report))
        for name, label in (
            ("always_up", "Always up"),
            ("persistence", "Persistence"),
            ("ma_200", "200-day average rule"),
            ("logistic", "Logistic regression"),
        ):
            assert label in text
            assert f"{100 * report['predictors'][name]['accuracy']:.1f}%" in text
        assert "benchmark" in text
        assert "they are not real results" in text

    def test_the_win_count_matches_the_intervals(self, report: dict[str, Any]) -> None:
        won = with_intervals(report, (0.01, 0.04), (0.01, 0.3))
        lost = with_intervals(report, (-0.04, -0.01), (-0.3, -0.01))
        assert "1 of 1" in text_of(backtest_fragment(won))
        assert "0 of 1" in text_of(backtest_fragment(lost))

    def test_has_the_accuracy_chart(self, report: dict[str, Any]) -> None:
        assert 'id="backtest-accuracy-chart"' in backtest_fragment(report)


class TestWhatIsPublished:
    def test_range_records_in_the_log_do_not_disturb_the_direction_fragments(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        # The nightly job writes range forecasts and their outcomes into the same
        # log. They have their own fragments and must change nothing in the others,
        # except the stance panel's one tile showing the simple baseline's range.
        with_ranges = [
            *log_records()[:2],
            RangeForecast(at(f"{THU} 18:00"), THU, "ewma_94", 0.8, -0.03, 0.04),
            log_records()[2],
            RangeOutcome(at(f"{FRI} 17:30"), FRI, THU, "ewma_94", "inside"),
            *log_records()[3:],
            RangeForecast(at(f"{MON} 18:00"), MON, "ewma_94", 0.8, -0.03, 0.04),
        ]
        mixed = build_fragments(report, with_ranges, volatility=volatility)
        plain = build_fragments(report, log_records(), volatility=volatility)
        about_direction = [name for name in plain if not name.startswith("range")]
        assert len(about_direction) == 6
        range_tile = (
            r'<div class="stance-tile"><div class="stance-number">[^<]*</div>'
            r"<p>the simple baseline&#x27;s 80% range[^<]*</p></div>"
        )
        assert len(re.findall(range_tile, mixed["tomorrow.md"])) == 1
        assert re.sub(range_tile, "", mixed["tomorrow.md"]) == plain["tomorrow.md"]
        for name in about_direction:
            if name != "tomorrow.md":
                assert mixed[name] == plain[name]
        assert mixed["range.md"] != plain["range.md"]

    def test_paper_strategy_entries_in_the_log_change_no_fragment_yet(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        # The nightly job now writes the paper strategy's entries into the same
        # log. The page does not show them yet, and must not trip over them.
        entry = PaperStrategyEntry(at(f"{MON} 18:00"), MON, False, 1, True)
        with_entry = [*log_records(), entry]
        assert build_fragments(report, with_entry, volatility=volatility) == build_fragments(
            report, log_records(), volatility=volatility
        )

    def test_fragments_are_raw_html_blocks(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        fragments = build_fragments(report, [*log_records(), *[]], volatility=volatility)
        assert set(fragments) == {
            "plotly.md",
            "status.md",
            "news.md",
            "tomorrow.md",
            "range.md",
            "track_record.md",
            "range_record.md",
            "backtest.md",
            "range_backtest.md",
        }
        for content in fragments.values():
            assert content.startswith("```{=html}\n")
            assert content.endswith("\n```\n")
            assert content.count("```") == 2

    def test_the_chart_library_is_loaded_once_from_a_pinned_version(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        fragments = build_fragments(report, log_records(), volatility=volatility)
        assert re.search(
            r'src="https://cdn\.plot\.ly/plotly-\d+\.\d+\.\d+\.min\.js"', fragments["plotly.md"]
        )
        for name in fragments:
            if name != "plotly.md":
                assert "cdn.plot.ly" not in fragments[name]

    def test_no_price_from_the_input_appears_anywhere(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        prices = make_prices(rows=900)
        raw = {
            f"{float(v):.4f}"
            for column in ("open", "high", "low", "close", "adj_close")
            for v in prices[column]
        }
        records = [*range_records(), *log_records()[3:]]
        everything = "".join(build_fragments(report, records, volatility=volatility).values())
        numbers = set(re.findall(r"\d+\.\d{4}", everything))
        assert not numbers & raw

    def test_names_are_escaped_before_they_reach_the_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The log restricts model names, but the page escapes what it shows anyway.
        monkeypatch.setitem(site_page.DISPLAY_NAMES, "logistic", "<b>bold</b> & co")
        for fragment in (tomorrow_fragment(log_records()), track_record_fragment(log_records())):
            assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in fragment
            assert "<b>bold</b>" not in fragment


MINUS = "\u2212"
THE_FINDING = (
    "None of the models clearly beat the simple baseline in the backtest, and the live "
    "record below will show whether that holds."
)


class TestRangeFinding:
    """The one sentence on the panel is worded from the report, not typed in."""

    def test_no_model_better_on_both_scores_gives_the_agreed_sentence(
        self, volatility: dict[str, Any]
    ) -> None:
        mixed = with_verdicts(volatility, har=("better", "same"), xgboost_vol=("same", "better"))
        assert site_page.clearly_beat_the_baseline(mixed) == []
        assert site_page.range_finding(mixed) == THE_FINDING

    def test_better_on_one_score_only_is_not_clearly_better(
        self, volatility: dict[str, Any]
    ) -> None:
        for verdicts in (
            ("better", "same"),
            ("same", "better"),
            ("better", "worse"),
            ("same", "same"),
        ):
            report = with_verdicts(volatility, har=verdicts, xgboost_vol=("same", "same"))
            assert site_page.range_finding(report) == THE_FINDING

    def test_a_model_better_on_both_scores_changes_the_sentence(
        self, volatility: dict[str, Any]
    ) -> None:
        one = with_verdicts(volatility, har=("better", "better"), xgboost_vol=("same", "same"))
        assert site_page.clearly_beat_the_baseline(one) == ["har"]
        assert site_page.range_finding(one) == (
            "One of the models clearly beat the simple baseline in the backtest, and the live "
            "record below will show whether that holds."
        )
        both = with_verdicts(volatility, har=("better", "better"), xgboost_vol=("better", "better"))
        assert site_page.range_finding(both).startswith("Both models clearly beat")

    def test_a_baseline_never_counts_as_a_model(self, volatility: dict[str, Any]) -> None:
        only = copy.deepcopy(volatility)
        for name in ("har", "xgboost_vol"):
            del only["forecasters"][name]
        only["comparisons"] = [
            row for row in only["comparisons"] if row["forecaster"] not in ("har", "xgboost_vol")
        ]
        assert site_page.range_finding(only).startswith("Only the simple baselines")

    def test_the_backtest_findings_say_how_each_model_did_on_each_score(
        self, volatility: dict[str, Any]
    ) -> None:
        report = with_verdicts(volatility, har=("better", "same"), xgboost_vol=("worse", "better"))
        lines = site_page.range_findings(report)
        assert (
            "Three-speed regression, against the simple baseline: its volatility forecast was "
            "measurably better and its ranges were not measurably different." in lines
        )
        assert (
            "XGBoost (volatility), against the simple baseline: its volatility forecast was "
            "measurably worse and its ranges were measurably better." in lines
        )
        assert lines[0].endswith("beat assuming volatility is always its long-run average.")
        assert lines[0].split(" that look")[0] in (
            "All 4 forecasters",
            "None of the forecasters",
            "One of the forecasters",
            "Two of the forecasters",
            "Three of the forecasters",
        )
        coverage = report["forecasters"]["ewma_94"]["coverage"]
        assert lines[-1] == (
            f"The simple baseline's ranges held {100 * coverage:.1f}% of outcomes, against "
            "the 80% they are built to hold."
        )


class TestRangeFragment:
    def test_before_any_range_is_logged_it_says_so_and_still_states_the_finding(
        self, volatility: dict[str, Any]
    ) -> None:
        text = text_of(site_page.range_fragment(log_records(), volatility))
        assert "No range has been logged yet." in text
        assert "clearly beat the simple baseline" in text

    def test_the_simple_baseline_is_shown_first_with_its_range(
        self, volatility: dict[str, Any]
    ) -> None:
        fragment = site_page.range_fragment(range_records(), volatility)
        text = text_of(fragment)
        assert "Ranges for Monday 18 March 2030" in text
        assert "Made after the close on Friday 15 March 2030." in text
        assert f'<div class="range-number">{MINUS}5.6% to +6.2%</div>' in fragment
        assert "<h4>Weighted recent average</h4>" in fragment
        assert "baseline · 80% range" in text
        assert "about 4 days in 5" in text
        assert "It says nothing about which way." in text
        assert "Not scored yet." in text
        assert "Written before that session opened." in text
        assert "advice" not in text.lower()

    def test_the_range_does_not_repeat_the_notice_in_any_state(
        self, volatility: dict[str, Any]
    ) -> None:
        scored_only = range_records()[: 2 * len(FORECASTERS) + 1]
        for records in (range_records(), scored_only):
            fragment = site_page.range_fragment(records, volatility)
            assert "advice" not in text_of(fragment).lower()
            assert re.search(
                r'<div class="call-head"><p class="call-heading">Ranges for .*?</p></div>',
                fragment,
            )

    def test_the_finding_sits_between_the_headline_and_the_table(
        self, volatility: dict[str, Any]
    ) -> None:
        fragment = site_page.range_fragment(range_records(), volatility)
        assert (
            fragment.index('class="range-panel"')
            < fragment.index('class="range-finding"')
            < fragment.index("<table")
        )

    def test_every_forecaster_is_listed_baselines_first(self, volatility: dict[str, Any]) -> None:
        fragment = site_page.range_fragment(range_records(), volatility)
        table = fragment[fragment.index("<table") :]
        names = [
            "Long-run average",
            "Last 21 sessions",
            "Weighted recent average",
            "Three-speed regression",
            "XGBoost (volatility)",
        ]
        positions = [table.index(name) for name in names]
        assert positions == sorted(positions)
        assert f"{MINUS}7.4% to +8.4%" in table
        assert table.count(">baseline<") == 3 and table.count(">model<") == 2

    def test_a_range_is_never_shown_in_the_colours_or_arrows_of_direction(
        self, volatility: dict[str, Any]
    ) -> None:
        fragments = (
            site_page.range_fragment(range_records(), volatility),
            site_page.range_record_fragment(range_records()),
        )
        for fragment in fragments:
            assert not re.search(r'class="[^"]*\b(up|down)\b', fragment)
            assert "▲" not in fragment and "▼" not in fragment
            assert "big-number" not in fragment

    def test_once_scored_it_says_where_the_move_fell(self, volatility: dict[str, Any]) -> None:
        scored_only = range_records()[: 2 * len(FORECASTERS) + 1]
        fragment = site_page.range_fragment(scored_only, volatility)
        text = text_of(fragment)
        assert "Ranges for Friday 15 March 2030" in text
        assert "Not scored yet." not in text
        assert "The move was inside the range." in text
        assert "<th>The move was</th>" in fragment
        assert "above the range" in text  # the long-run average missed
        assert "That session is over and these have been scored." in text

    def test_without_the_simple_baseline_the_first_forecaster_leads(
        self, volatility: dict[str, Any]
    ) -> None:
        fragment = site_page.range_fragment([ranged(THU, "har", -0.02, 0.05)], volatility)
        assert "<h4>Three-speed regression</h4>" in fragment
        assert "model · 80% range" in text_of(fragment)
        assert f"{MINUS}2.0% to +5.0%" in fragment

    def test_names_are_escaped(
        self, monkeypatch: pytest.MonkeyPatch, volatility: dict[str, Any]
    ) -> None:
        monkeypatch.setitem(site_page.RANGE_NAMES, "ewma_94", "<b>bold</b> & co")
        for fragment in (
            site_page.range_fragment(range_records(), volatility),
            site_page.range_record_fragment(range_records()),
            site_page.range_backtest_fragment(volatility),
        ):
            assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in fragment
            assert "<b>bold</b>" not in fragment


class TestRangeRecord:
    def test_before_any_range_is_logged(self) -> None:
        assert "No ranges have been logged yet." in site_page.range_record_fragment(log_records())

    def test_counts_shares_and_the_sample_size_warning(self) -> None:
        fragment = site_page.range_record_fragment(range_records())
        text = text_of(fragment)
        assert "1 session has been scored so far." in text
        assert "about 250 scored sessions" in text
        assert "built to hold the outcome 80% of the time" in text
        rows = table_rows(fragment)
        held = rows["Weighted recent average"]
        assert held[2:4] == ["1", "1"] and held[4].startswith("100.0%")
        assert held[5:] == ["0 / 0", "7.0 points"]
        missed = rows["Long-run average"]
        assert missed[2:4] == ["1", "0"] and missed[4].startswith("0.0%")
        assert missed[5] == "0 / 1"

    def test_a_forecaster_with_nothing_scored_shows_no_share(self) -> None:
        fragment = site_page.range_record_fragment([ranged(THU, "ewma_94")])
        assert "0 sessions have been scored so far." in text_of(fragment)
        cells = table_rows(fragment)["Weighted recent average"]
        assert cells[2:4] == ["0", "0"] and cells[4].startswith("n/a")
        assert cells[-1] == "n/a"


class TestRangeBacktest:
    def test_headline_numbers_findings_chart_and_table(self, volatility: dict[str, Any]) -> None:
        fragment = site_page.range_backtest_fragment(volatility)
        text = text_of(fragment)
        data = volatility["data"]
        assert f"{data['tested_sessions']:,}" in text
        assert f"{100 * data['mean_absolute_move']:.1f}%" in text
        assert "was NVDA's average move in a day" in text
        assert 'id="range-coverage-chart"' in fragment
        assert fragment.count('class="stat"') == 3
        for name in site_page.RANGE_NAMES.values():
            assert name in fragment
        for line in site_page.range_findings(volatility):
            assert html.escape(line) in fragment

    def test_the_chart_marks_the_share_the_range_is_built_for(
        self, volatility: dict[str, Any]
    ) -> None:
        figure = site_page.coverage_figure(volatility).to_dict()
        assert len(figure["data"]) == 5
        assert any(
            "Built to hold 80%" in str(a.get("text")) for a in figure["layout"]["annotations"]
        )
        shown = {trace["y"][0]: trace["x"][0] for trace in figure["data"]}
        coverage = volatility["forecasters"]["ewma_94"]["coverage"]
        assert shown["Weighted recent average"] == pytest.approx(100 * coverage)


class TestPhoneTables:
    """On phones every table row is a card, so each value must carry its column's name."""

    def fragments(self, report: dict[str, Any], volatility: dict[str, Any]) -> dict[str, str]:
        ranges = range_records()
        return {
            "track record": track_record_fragment(log_records()),
            "backtest": backtest_fragment(report),
            "latest range, pending": site_page.range_fragment(ranges, volatility),
            "latest range, scored": site_page.range_fragment(
                ranges[: 2 * len(FORECASTERS) + 1], volatility
            ),
            "range record": site_page.range_record_fragment(ranges),
            "range backtest": site_page.range_backtest_fragment(volatility),
        }

    def test_every_value_is_labelled_with_its_column(
        self, report: dict[str, Any], volatility: dict[str, Any]
    ) -> None:
        for where, fragment in self.fragments(report, volatility).items():
            tables = table_labels(fragment)
            assert len(tables) == 1, where
            headings, rows = tables[0]
            assert rows, where
            for labels in rows:
                assert len(labels) == len(headings), where
                assert labels[0] is None, f"{where}: the row's title needs no label"
                for heading, label in zip(headings[1:], labels[1:], strict=True):
                    assert label and "<" not in label, where
                    first_word = label.split()[0].lower()
                    assert first_word in heading.lower(), f"{where}: {label!r} for {heading!r}"

    def test_the_stylesheet_turns_rows_into_labelled_cards_on_phones(self) -> None:
        theme = (Path(__file__).parents[1] / "site" / "theme.scss").read_text()
        phone = theme.split("@media (max-width: 40rem)")[1]
        assert "content: attr(data-label)" in phone
        assert "thead {" in phone and "clip-path: inset(50%)" in phone


def six(day: date, up: int, *, without: str | None = None) -> list[Record]:
    """A call from each of the paper strategy's six models; the first `up` say up."""
    return [
        call(day, name, position < up, 0.56 if position < up else 0.44)
        for position, name in enumerate(RULE_MODELS)
        if name != without
    ]


def paper_entry(day: date, up: int, *, gate: bool = True) -> PaperStrategyEntry:
    return PaperStrategyEntry(at(f"{day} 18:00"), day, up >= 4 and gate, up, gate)


def paper_log() -> list[Record]:
    """Thursday: Holding, and Friday closed up. Friday night: one call missing, so
    no entry; Monday closed up. Monday: In cash (gate closed), not scored yet."""
    return [
        *six(THU, 5),
        paper_entry(THU, 5),
        went(FRI, THU, True),
        *six(FRI, 6, without="lstm_ctx"),
        went(MON, FRI, True),
        *six(MON, 6),
        paper_entry(MON, 6, gate=False),
    ]


def with_paper_verdict(low: float, high: float) -> dict[str, Any]:
    """The paper report with the primary interval replaced."""
    changed = copy.deepcopy(paper_report())
    row = next(r for r in changed["comparisons"] if r["strategy"] == "paper_strategy")
    row["sharpe_difference_interval"] = [low, high]
    row["verdict"] = paper_backtest.verdict([low, high])
    return changed


def no_action_words(fragment: str) -> None:
    """Nothing in the fragment reads as an instruction or uses direction colours."""
    assert not re.search(r'class="[^"]*\b(up|down)\b', fragment)
    assert "▲" not in fragment and "▼" not in fragment
    assert "big-number" not in fragment
    words = text_of(fragment).lower().replace("buy-and-hold", "")
    for word in ("buy", "sell", "signal", "recommend", "should", "consider"):
        assert not re.search(rf"\b{word}", words), word


class TestPaperStrategyPage:
    def test_the_names_match_the_backtests(self) -> None:
        assert site_page.PAPER_STRATEGY == paper_backtest.STRATEGY
        assert site_page.PAPER_BENCHMARK == paper_backtest.BENCHMARK
        assert set(site_page.PAPER_NAMES) == {
            paper_backtest.STRATEGY,
            paper_backtest.BENCHMARK,
            *paper_backtest.PARTS,
        }
        assert list(paper_report()["strategies"]) == list(site_page.PAPER_NAMES)

    def test_before_the_first_entry(self) -> None:
        fragment = site_page.paper_fragment(log_records(), paper_report())
        assert "has not made an entry yet" in text_of(fragment)
        assert html.escape(site_page.paper_finding(paper_report())) in fragment

    @pytest.mark.parametrize(
        ("up", "gate", "state", "detail"),
        [
            (5, True, "Holding", "5 of 6 models say up; the volatility gate is open."),
            (3, True, "In cash", "3 of 6 models say up; the volatility gate is open."),
            (6, False, "In cash", "6 of 6 models say up; the volatility gate is closed."),
        ],
    )
    def test_the_state_for_the_next_session(
        self, up: int, gate: bool, state: str, detail: str
    ) -> None:
        records = [*six(MON, up), paper_entry(MON, up, gate=gate)]
        fragment = site_page.paper_fragment(records, paper_report())
        text = text_of(fragment)
        assert f'<div class="paper-state">{state}</div>' in fragment
        assert detail in text
        assert "Paper strategy for Tuesday 19 March 2030" in text
        assert "Not scored yet." in text
        assert "advice" not in text.lower()  # said once, under the page title
        assert "A simulated position, published so it can be scored." in text
        assert "Simulated · no money involved" in text
        no_action_words(fragment)

    def test_a_session_with_a_call_missing_says_no_entry(self) -> None:
        # Through Friday night: five of the six called, so Friday has no entry.
        fragment = site_page.paper_fragment(paper_log()[:13], paper_report())
        text = text_of(fragment)
        assert "Paper strategy for Monday 18 March 2030" in text
        assert '<div class="paper-state">No entry for this session</div>' in fragment
        assert "no call from LSTM + context" in text
        assert "left out of both the paper strategy and the buy-and-hold comparison" in text
        assert "Holding</div>" not in fragment and "In cash</div>" not in fragment
        no_action_words(fragment)

    def test_once_scored_it_says_which_way_the_session_closed(self) -> None:
        fragment = site_page.paper_fragment(paper_log()[:8], paper_report())
        text = text_of(fragment)
        assert "Paper strategy for Friday 15 March 2030" in text
        assert '<div class="paper-state">Holding</div>' in fragment
        assert "5 of 6 models said up; the volatility gate was open." in text
        assert "That session closed up." in text
        assert "Not scored yet." not in text

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (-0.39, 0.21, "was no better than buy-and-hold's"),
            (0.05, 0.4, "had a better risk-adjusted return than buy-and-hold"),
            (-0.4, -0.05, "had a worse risk-adjusted return than buy-and-hold"),
        ],
    )
    def test_the_finding_is_worded_from_the_interval(
        self, low: float, high: float, expected: str
    ) -> None:
        sentence = site_page.paper_finding(with_paper_verdict(low, high))
        assert sentence.startswith("In the backtest the paper strategy")
        assert expected in sentence
        assert ("cannot be told apart from zero" in sentence) == (low < 0 < high)
        assert f"{abs(low):.2f}" in sentence and f"{abs(high):.2f}" in sentence

    def test_the_finding_sits_in_the_panel_in_full_text(self) -> None:
        fragment = site_page.paper_fragment(paper_log(), paper_report())
        finding = html.escape(site_page.paper_finding(paper_report()))
        assert f'<p class="paper-finding">{finding}</p>' in fragment


class TestPaperStrategyRecord:
    def test_before_the_first_entry(self) -> None:
        fragment = site_page.paper_record_fragment(log_records())
        assert "The paper strategy has no entries yet." in fragment

    def test_counts_on_the_same_sessions_with_the_missed_ones_beside_them(self) -> None:
        fragment = site_page.paper_record_fragment(paper_log())
        text = text_of(fragment)
        assert "1 session has both an entry and an outcome so far." in text
        rows = table_rows(fragment)
        # One session compared on both rows: Friday's. Monday's had no entry, so
        # its rise is not counted for buy-and-hold either.
        assert rows["Paper strategy"][1:3] == ["1", "1"]
        assert rows["Buy-and-hold"][1:3] == ["1", "1"]
        assert "Sessions with no entry: 1. It is left out of both rows." in text
        assert "Holding on 1 of the 1 (1 closed up); In cash on 0 (0 did not close up)." in text

    def test_the_missed_count_is_as_readable_as_the_counts(self) -> None:
        fragment = site_page.paper_record_fragment(paper_log())
        line = re.search(r'<p class="([^"]*)">Sessions with no entry', fragment)
        assert line is not None and line.group(1) == "paper-missed"  # not muted, not small

    def test_no_missed_sessions_says_zero(self) -> None:
        records = [*six(THU, 5), paper_entry(THU, 5), went(FRI, THU, False)]
        text = text_of(site_page.paper_record_fragment(records))
        assert "Sessions with no entry: 0." in text and "left out" not in text
        rows = table_rows(site_page.paper_record_fragment(records))
        assert rows["Paper strategy"][2] == "0" and rows["Buy-and-hold"][2] == "0"

    def test_the_two_sentences_about_counts_are_beside_them(self) -> None:
        fragment = site_page.paper_record_fragment(paper_log())
        assert site_page.LIVE_COUNT_NOTES == (
            "These are counts, not returns: a right-side day counts the same whether the "
            "move was large or small, so they can't be compared with the Sharpe ratio in "
            "the backtest.",
            "Risk-adjusted return, total return and worst drawdown are measured in the "
            "backtest only, and this record needs about a year of sessions before it says "
            "much.",
        )
        for sentence in site_page.LIVE_COUNT_NOTES:
            assert f"<p>{html.escape(sentence)}</p>" in fragment

    def test_it_shows_no_return_figures(self) -> None:
        text = text_of(site_page.paper_record_fragment(paper_log())).lower()
        for word in ("sharpe ratio of", "per year", "total return of", "worst fall"):
            assert word not in text
        no_action_words(site_page.paper_record_fragment(paper_log()))

    def test_nothing_scored_yet_shows_no_share(self) -> None:
        records = [*six(THU, 5), paper_entry(THU, 5)]
        fragment = site_page.paper_record_fragment(records)
        assert "0 sessions have both an entry and an outcome so far." in text_of(fragment)
        assert table_rows(fragment)["Paper strategy"][3] == "n/a"
        assert "Holding on" not in fragment


class TestPaperStrategyBacktest:
    def test_findings_table_and_every_strategy(self) -> None:
        report = paper_report()
        fragment = site_page.paper_backtest_fragment(report)
        for line in site_page.paper_findings(report):
            assert html.escape(line) in fragment
        rows = table_rows(fragment)
        assert list(rows) == ["Paper strategy", "Buy-and-hold", "Majority alone", "Gate alone"]
        five = report["strategies"]["paper_strategy"]["by_cost_bps"]["5"]
        assert rows["Paper strategy"][2].startswith(f"{five['sharpe_ratio']:.2f}")
        assert "benchmark" in rows["Buy-and-hold"][3]
        assert rows["Buy-and-hold"][1] == "100.0%"
        assert rows["Paper strategy"][6] == f"{100 * abs(five['worst_drawdown']):.1f}%"

    def test_the_first_finding_is_the_primary_one(self) -> None:
        report = paper_report()
        assert site_page.paper_findings(report)[0] == site_page.paper_finding(report)

    def test_findings_cover_what_was_promised(self) -> None:
        text = " ".join(site_page.paper_findings(paper_report()))
        for phrase in (
            "Sharpe ratio",
            "a year against",
            "worst fall",
            "Holding on",
            "gate was open",
        ):
            assert phrase in text, phrase

    def with_gate(self, **changes: Any) -> dict[str, Any]:
        """The paper report with a gate that plainly was held open by its early history."""
        report = copy.deepcopy(paper_report())
        report["data"]["share_gate_open"] = 0.95
        report["data"]["share_same_position_as_majority"] = 0.983
        report["gate"] = {
            "first_history_session": "1999-11-04",
            "first_tested_session": "2004-11-10",
            "threshold_first_fold": 0.0684,
            "threshold_last_fold": 0.0429,
            "median_forecast_tested": 0.0253,
            "history_share_before_first_tested_session": 0.186,
            "share_of_values_above_final_threshold_from_before_first_tested_session": 0.575,
            "check_warm_up_sessions": 252,
            "check_share_open_with_history_from_first_tested_session": 0.774,
            "check_sessions": 5289,
            **changes,
        }
        return report

    def test_the_gate_finding_gives_the_number_and_says_what_it_means(self) -> None:
        lines = site_page.paper_findings(self.with_gate())
        assert (
            "The volatility gate was open on 95.0% of sessions, so it rarely closed: the "
            "strategy held the same position as the models' majority alone on 98.3% of "
            "sessions. In practice it is close to the majority rule alone."
        ) in lines

    def test_the_cause_is_stated_with_the_checked_numbers(self) -> None:
        assert site_page.gate_cause(self.with_gate()) == (
            "The reason was checked. The gate's threshold comes from every earlier session, "
            "and those begin in 1999: the sessions before November 2004 are 18.6% of that "
            "history but 57.5% of its values above the final threshold. The threshold fell "
            "from 6.8% to 4.3% daily volatility over the test, while the typical forecast was "
            "2.5%. With a history starting in November 2004 instead, the same gate would have "
            "been open on 77.4% of sessions."
        )

    @pytest.mark.parametrize(
        "changes",
        [
            # The check barely differs from what happened.
            {"check_share_open_with_history_from_first_tested_session": 0.93},
            # The early sessions hold no more than their share of the high values.
            {"share_of_values_above_final_threshold_from_before_first_tested_session": 0.15},
        ],
    )
    def test_no_cause_is_claimed_when_the_check_does_not_support_it(
        self, changes: dict[str, float]
    ) -> None:
        sentence = site_page.gate_cause(self.with_gate(**changes))
        assert sentence.startswith("Why it closed that rarely is not explained by the early")
        assert "The reason was checked" not in sentence

    def test_without_a_check_nothing_is_said_about_the_cause(self) -> None:
        report = self.with_gate(check_share_open_with_history_from_first_tested_session=None)
        assert site_page.gate_cause(report) == ""
        assert not any("reason" in line.lower() for line in site_page.paper_findings(report))

    def test_values_are_labelled_for_phones(self) -> None:
        ((headings, rows),) = table_labels(site_page.paper_backtest_fragment(paper_report()))
        assert len(headings) == 7
        for labels in rows:
            assert labels[0] is None and all(labels[1:]) and len(labels) == 7
        ((_, record_rows),) = table_labels(site_page.paper_record_fragment(paper_log()))
        assert all(labels[0] is None and all(labels[1:]) for labels in record_rows)

    def test_nothing_reads_as_an_instruction(self) -> None:
        no_action_words(site_page.paper_backtest_fragment(paper_report()))


class TestCommandLine:
    def write_report(self, root: Path, report: dict[str, Any]) -> None:
        (root / "reports").mkdir()
        (root / "reports" / "nvda_backtest.json").write_text(json.dumps(report))
        (root / "reports" / "nvda_volatility.json").write_text(json.dumps(volatility_report()))
        (root / "reports" / "nvda_paper_strategy.json").write_text(json.dumps(paper_report()))

    def test_writes_every_fragment_without_a_log(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        self.write_report(tmp_path, report)
        (tmp_path / "site").mkdir()
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 0
        written = sorted(p.name for p in (tmp_path / "site" / "_generated").iterdir())
        assert written == [
            "backtest.md",
            "news.md",
            "paper.md",
            "paper_backtest.md",
            "paper_record.md",
            "plotly.md",
            "range.md",
            "range_backtest.md",
            "range_record.md",
            "status.md",
            "tomorrow.md",
            "track_record.md",
            "updated.md",
        ]
        assert (
            "has not started yet" in (tmp_path / "site" / "_generated" / "tomorrow.md").read_text()
        )
        assert "(0 log records)" in lines[0]

    def test_reads_the_log_next_to_the_site(self, tmp_path: Path, report: dict[str, Any]) -> None:
        self.write_report(tmp_path, report)
        (tmp_path / "site").mkdir()
        append_records(tmp_path / "predictions" / "nvda.jsonl", log_records())
        assert main(["--site-dir", str(tmp_path / "site")], report=lambda _line: None) == 0
        tomorrow = (tmp_path / "site" / "_generated" / "tomorrow.md").read_text()
        assert "Monday 18 March 2030" in tomorrow

    def test_reads_the_collected_headlines_next_to_the_site(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        self.write_report(tmp_path, report)
        save_day(news_day(headline(9, "NVIDIA ships a thing")), tmp_path / "news" / "nvda")
        assert main(["--site-dir", str(tmp_path / "site")], report=lambda _line: None) == 0
        generated = (tmp_path / "site" / "_generated" / "news.md").read_text()
        assert "NVIDIA ships a thing" in generated

    def test_a_tampered_news_file_fails_the_build(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        self.write_report(tmp_path, report)
        folder = tmp_path / "news" / "nvda"
        path = save_day(news_day(headline(9, "NVIDIA ships a thing")), folder)
        assert path is not None
        path.write_text(path.read_text().replace("https://", "javascript://"))
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "stored news file is not valid" in lines[0]

    def test_missing_report_fails(self, tmp_path: Path) -> None:
        (tmp_path / "site").mkdir()
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "run the backtest first" in lines[0]

    def test_missing_paper_strategy_report_fails(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        self.write_report(tmp_path, report)
        (tmp_path / "reports" / "nvda_paper_strategy.json").unlink()
        (tmp_path / "site").mkdir()
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "run the paper strategy backtest first" in lines[0]

    def test_missing_volatility_report_fails(self, tmp_path: Path, report: dict[str, Any]) -> None:
        self.write_report(tmp_path, report)
        (tmp_path / "reports" / "nvda_volatility.json").unlink()
        (tmp_path / "site").mkdir()
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "run the volatility backtest first" in lines[0]
        assert not (tmp_path / "site" / "_generated").exists()

    def test_a_tampered_log_fails_the_build(self, tmp_path: Path, report: dict[str, Any]) -> None:
        self.write_report(tmp_path, report)
        (tmp_path / "site").mkdir()
        log = tmp_path / "predictions" / "nvda.jsonl"
        append_records(log, log_records())
        log.write_text(log.read_text().replace('"predicted_up":false', '"predicted_up":true', 1))
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "content does not match its hash" in lines[0]
        assert not (tmp_path / "site" / "_generated").exists()
