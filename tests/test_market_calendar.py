"""NYSE's holiday rules, used only to put a date on the NVIDIA page's unscored calls."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from markets_lab.market_calendar import (
    PAST_UNSCHEDULED_CLOSURES,
    PUBLISHED_HOLIDAYS,
    compare_with_sessions,
    easter,
    holidays,
    is_session,
    main,
    next_session,
)


class TestRules:
    @pytest.mark.parametrize("year", sorted(PUBLISHED_HOLIDAYS))
    def test_the_rules_give_exactly_nyses_published_list(self, year: int) -> None:
        assert holidays(year) == PUBLISHED_HOLIDAYS[year]

    def test_easter(self) -> None:
        known = {2000: (4, 23), 2008: (3, 23), 2019: (4, 21), 2026: (4, 5), 2038: (4, 25)}
        for year, (month, day) in known.items():
            assert easter(year) == date(year, month, day)

    def test_good_friday_is_closed(self) -> None:
        assert not is_session(date(2030, 4, 19))  # Easter is 21 April 2030
        assert is_session(date(2030, 4, 22))  # Easter Monday trades

    def test_a_saturday_holiday_moves_to_friday_and_a_sunday_one_to_monday(self) -> None:
        # 4 July 2026 is a Saturday; 4 July 2027 is a Sunday.
        assert date(2026, 7, 3) in holidays(2026)
        assert date(2027, 7, 5) in holidays(2027)
        # Christmas 2032 is a Saturday.
        assert date(2032, 12, 24) in holidays(2032)

    def test_new_years_day_on_a_saturday_is_not_moved_into_the_old_year(self) -> None:
        # 1 January 2033 is a Saturday; Friday 31 December 2032 trades.
        assert is_session(date(2032, 12, 31))
        assert not any(day.month == 1 and day.day < 3 for day in holidays(2033))

    def test_juneteenth_only_from_2022(self) -> None:
        assert is_session(date(2021, 6, 18))  # Friday before a Saturday 19 June
        assert not is_session(date(2022, 6, 20))  # first observed, a Monday
        assert not is_session(date(2030, 6, 19))

    def test_weekends_never_trade(self) -> None:
        assert not is_session(date(2030, 3, 16)) and not is_session(date(2030, 3, 17))

    def test_past_unscheduled_closures_are_closed(self) -> None:
        assert PAST_UNSCHEDULED_CLOSURES
        for day in PAST_UNSCHEDULED_CLOSURES:
            assert day.weekday() < 5 and day not in holidays(day.year)
            assert not is_session(day)


class TestNextSession:
    def test_over_a_weekend(self) -> None:
        assert next_session(date(2026, 11, 20)).day == date(2026, 11, 23)

    def test_over_a_holiday(self) -> None:
        assert next_session(date(2026, 11, 25)).day == date(2026, 11, 27)  # Thanksgiving

    def test_over_a_long_weekend(self) -> None:
        # Good Friday 2027 is 26 March.
        assert next_session(date(2027, 3, 25)).day == date(2027, 3, 29)

    def test_certain_only_inside_the_published_years(self) -> None:
        assert next_session(date(2026, 11, 23)).certain
        assert next_session(date(2028, 12, 27)).certain
        assert not next_session(date(2030, 3, 18)).certain
        assert not next_session(date(2025, 12, 30)).certain

    def test_a_next_session_that_crosses_out_of_the_published_years_is_not_certain(
        self,
    ) -> None:
        upcoming = next_session(date(2028, 12, 29))
        assert upcoming.day == date(2029, 1, 2)
        assert not upcoming.certain

    def test_it_always_lands_on_a_session(self) -> None:
        day = date(2026, 1, 1)
        while day < date(2029, 1, 1):
            upcoming = next_session(day)
            assert is_session(upcoming.day) and upcoming.day > day
            gap = upcoming.day - day
            assert all(not is_session(day + timedelta(days=d)) for d in range(1, gap.days))
            day += timedelta(days=1)


class TestHistoryCheck:
    def test_agreement(self) -> None:
        traded = [date(2030, 3, 14), date(2030, 3, 15), date(2030, 3, 18)]
        result = compare_with_sessions(traded)
        assert result.missing == [] and result.unexpected == []
        assert result.sessions == 3

    def test_a_day_with_no_trading_and_a_day_that_should_not_have_traded(self) -> None:
        traded = [date(2030, 4, 17), date(2030, 4, 19), date(2030, 4, 22)]
        result = compare_with_sessions(traded)
        assert result.missing == [date(2030, 4, 18)]
        assert result.unexpected == [date(2030, 4, 19)]  # Good Friday

    def test_nothing_to_compare_with(self) -> None:
        with pytest.raises(ValueError):
            compare_with_sessions([])

    def test_command_line(self, tmp_path: Path) -> None:
        prices = tmp_path / "prices.csv"
        prices.write_text("date,close\n2030-04-17,1\n2030-04-22,1\n", encoding="utf-8")
        lines: list[str] = []
        assert main(["check", str(prices)], report=lines.append) == 2
        assert "  2030-04-18 Thursday" in lines
        clean = tmp_path / "clean.csv"
        clean.write_text("date,close\n2030-04-18,1\n2030-04-22,1\n", encoding="utf-8")
        assert main(["check", str(clean)], report=lines.append) == 0
        assert main(["check", str(tmp_path / "absent.csv")], report=lines.append) == 1
