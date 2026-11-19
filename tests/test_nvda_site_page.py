"""The data-driven parts of the NVIDIA page: correct numbers, honest wording, no raw data."""

from __future__ import annotations

import copy
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
    status_fragment,
    tomorrow_fragment,
    track_record,
    track_record_fragment,
)
from markets_lab.analysis.nvda.validation import WalkForwardConfig
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


class TestTomorrow:
    def test_before_the_log_starts(self) -> None:
        assert "has not started yet" in tomorrow_fragment([])

    def test_shows_the_latest_unscored_calls(self) -> None:
        text = text_of(tomorrow_fragment(log_records()))
        assert "Calls for the next session after Monday 18 March 2030" in text
        assert "Logistic regression" in text and "Always up" in text
        assert "▲ Up" in text
        assert "Model's probability of up: 53%" in text
        assert "A fixed rule: no probability." in text
        assert "It has not been scored yet." in text

    def test_every_set_of_calls_is_labelled_as_an_experiment(self) -> None:
        assert "An experiment, not advice." in text_of(tomorrow_fragment(log_records()))
        scored_only = [*log_records(), went(TUE, MON, True)]
        assert "An experiment, not advice." in text_of(tomorrow_fragment(scored_only))

    def test_once_scored_it_says_so_and_shows_the_result(self) -> None:
        text = text_of(tomorrow_fragment([*log_records(), went(TUE, MON, False)]))
        assert "Latest calls were for Tuesday 19 March 2030" in text
        assert "these have been scored" in text
        assert "Wrong" in text  # both called up; the session did not close up
        assert "Calls for the next session" not in text

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

    def test_shows_each_models_latest_call_with_arrow_word_and_probability(self) -> None:
        fragment = status_fragment(log_records())
        text = text_of(fragment)
        assert "Model calls after the close of Mon 18 Mar 2030" in text
        assert re.search(r"Logistic regression\s+▲ Up\s+53%", text)
        assert '<span class="up">▲ Up</span>' in fragment

    def test_a_down_call_carries_the_arrow_and_the_word(self) -> None:
        fragment = status_fragment(log_records()[:2])
        assert '<span class="down">▼ Down</span>' in fragment
        assert re.search(r"Logistic regression\s+▼ Down\s+47%", text_of(fragment))

    def test_baselines_are_left_out(self) -> None:
        assert "Always up" not in text_of(status_fragment(log_records()))

    def test_it_reports_the_live_record_once_calls_are_scored(self) -> None:
        text = text_of(status_fragment(log_records()))
        assert re.search(r"Logistic regression\s+▲ Up\s+53% · 1 of 2 right", text)
        assert re.search(r"Sessions scored\s+2", text)

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
    def test_fragments_are_raw_html_blocks(self, report: dict[str, Any]) -> None:
        fragments = build_fragments(report, log_records())
        assert set(fragments) == {
            "plotly.md",
            "status.md",
            "tomorrow.md",
            "track_record.md",
            "backtest.md",
        }
        for content in fragments.values():
            assert content.startswith("```{=html}\n")
            assert content.endswith("\n```\n")
            assert content.count("```") == 2

    def test_the_chart_library_is_loaded_once_from_a_pinned_version(
        self, report: dict[str, Any]
    ) -> None:
        fragments = build_fragments(report, log_records())
        assert re.search(
            r'src="https://cdn\.plot\.ly/plotly-\d+\.\d+\.\d+\.min\.js"', fragments["plotly.md"]
        )
        for name in ("tomorrow.md", "track_record.md", "backtest.md"):
            assert "cdn.plot.ly" not in fragments[name]

    def test_no_price_from_the_input_appears_anywhere(self, report: dict[str, Any]) -> None:
        prices = make_prices(rows=900)
        raw = {
            f"{float(v):.4f}"
            for column in ("open", "high", "low", "close", "adj_close")
            for v in prices[column]
        }
        everything = "".join(build_fragments(report, log_records()).values())
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


class TestCommandLine:
    def write_report(self, root: Path, report: dict[str, Any]) -> None:
        (root / "reports").mkdir()
        (root / "reports" / "nvda_backtest.json").write_text(json.dumps(report))

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
            "plotly.md",
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

    def test_missing_report_fails(self, tmp_path: Path) -> None:
        (tmp_path / "site").mkdir()
        lines: list[str] = []
        assert main(["--site-dir", str(tmp_path / "site")], report=lines.append) == 1
        assert "run the backtest first" in lines[0]

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
