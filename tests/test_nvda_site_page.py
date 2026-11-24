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

from markets_lab.analysis.nvda import site_page
from markets_lab.analysis.nvda.backtest import run_backtest
from markets_lab.analysis.nvda.features import build_dataset
from markets_lab.analysis.nvda.metrics import BootstrapConfig, wilson_interval
from markets_lab.analysis.nvda.prediction_log import (
    Outcome,
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
from markets_lab.analysis.nvda.volatility.dataset import build_volatility_dataset
from markets_lab.collectors.nvda_news import Collection, NewsItem, save_day
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

    def test_every_set_of_calls_is_labelled_as_an_experiment(self) -> None:
        assert "An experiment, not advice." in text_of(tomorrow_fragment(log_records()))
        scored_only = [*log_records(), went(TUE, MON, True)]
        assert "An experiment, not advice." in text_of(tomorrow_fragment(scored_only))

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

    def test_the_label_sits_beside_the_heading_in_every_state(self) -> None:
        # My condition (2026-11-24): the "experiment, not advice" label stays
        # directly beside the calls.
        for records in (log_records(), [*log_records(), went(TUE, MON, True)]):
            fragment = tomorrow_fragment(records)
            assert re.search(
                r'<div class="call-head"><p class="call-heading">Calls for .*?</p>'
                r'<span class="tag advice-tag">Experiment · not advice</span></div>',
                fragment,
            )
            assert fragment.index("advice-tag") < fragment.index("call-grid")

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


class TestStatusStrip:
    def test_before_the_log_starts(self) -> None:
        text = text_of(status_fragment([]))
        assert "Experiment · not advice" in text
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

    def test_it_is_always_labelled_an_experiment_and_never_reads_as_a_signal(self) -> None:
        for records in (
            [],
            log_records()[:2],
            log_records(),
            [*log_records(), went(TUE, MON, True)],
        ):
            fragment = status_fragment(records)
            assert 'aria-label="Experiment status"' in fragment
            text = text_of(fragment).lower()
            assert "experiment · not advice" in text
            assert not re.search(r"\b(buy|sell|signal|recommend\w*|bullish|bearish)\b", text)


def headline(hour: int, title: str, url: str | None = None) -> NewsItem:
    return NewsItem(
        source="yahoo_finance",
        publisher="Example Wire",
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
        assert "Example Wire ·" in text and "09:00 UTC" in text

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
        assert "Not advice." in text


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
        # log. They have their own fragments and must change nothing in the others.
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
        for name in about_direction:
            assert mixed[name] == plain[name]
        assert mixed["range.md"] != plain["range.md"]

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
        assert "Written before that session opened. An experiment, not advice." in text

    def test_the_label_sits_beside_the_range_heading_in_every_state(
        self, volatility: dict[str, Any]
    ) -> None:
        scored_only = range_records()[: 2 * len(FORECASTERS) + 1]
        for records in (range_records(), scored_only):
            fragment = site_page.range_fragment(records, volatility)
            assert re.search(
                r'<div class="call-head"><p class="call-heading">Ranges for .*?</p>'
                r'<span class="tag advice-tag">Experiment · not advice</span></div>',
                fragment,
            )
            assert fragment.index("advice-tag") < fragment.index('class="range-panel"')

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
        rows = re.findall(r"<tr><td>(.*?)</tr>", fragment)
        by_name = {row.split("</td>")[0]: row for row in rows}
        held = by_name["Weighted recent average"]
        assert "<td>1</td><td>1</td><td>100.0%" in held
        assert "<td>0 / 0</td>" in held and "<td>7.0 points</td>" in held
        missed = by_name["Long-run average"]
        assert "<td>1</td><td>0</td><td>0.0%" in missed
        assert "<td>0 / 1</td>" in missed

    def test_a_forecaster_with_nothing_scored_shows_no_share(self) -> None:
        fragment = site_page.range_record_fragment([ranged(THU, "ewma_94")])
        assert "0 sessions have been scored so far." in text_of(fragment)
        assert "<td>0</td><td>0</td><td>n/a" in fragment
        assert "<td>n/a</td></tr>" in fragment


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


class TestCommandLine:
    def write_report(self, root: Path, report: dict[str, Any]) -> None:
        (root / "reports").mkdir()
        (root / "reports" / "nvda_backtest.json").write_text(json.dumps(report))
        (root / "reports" / "nvda_volatility.json").write_text(json.dumps(volatility_report()))

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
