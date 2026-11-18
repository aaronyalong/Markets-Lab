"""Which Polymarket US fee schedule was in force for a given window.

Only what the published fee page states is recorded here. The schedule before
2026-11-14 was not found, so it is recorded as unknown; analysis of those windows
has to apply a stand-in schedule and label the result as such.

The fee formula under the known schedule is `coefficient * contracts * p * (1 - p)`,
charged to takers. This module records which schedule applies; it does not compute
fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

FEE_PAGE_URL = "https://docs.polymarket.us/fees"

SCHEDULE_2026_11_14_START = 1_794_632_400
"""2026-11-14 00:00 US Eastern (05:00 UTC), when the current schedule took effect."""


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """A period during which one fee schedule applied.

    `effective_from` is inclusive and `effective_until` exclusive, both Unix seconds
    UTC; None means open-ended on that side. `taker_coefficient_tier1` is the taker
    coefficient for the lowest volume tier (under $10,000 of 30-day taker volume),
    or None when the schedule is not known.
    """

    schedule_id: str
    effective_from: int | None
    effective_until: int | None
    is_known: bool
    taker_coefficient_tier1: Decimal | None
    source_url: str | None
    note: str


UNKNOWN_BEFORE_2026_11_14 = FeeSchedule(
    schedule_id="pre-2026-11-14-unknown",
    effective_from=None,
    effective_until=SCHEDULE_2026_11_14_START,
    is_known=False,
    taker_coefficient_tier1=None,
    source_url=None,
    note=(
        "Schedule before 2026-11-14 was not found. The per-market feeCoefficient field "
        "reports today's value for old markets, so it is not evidence of the old schedule."
    ),
)

SCHEDULE_2026_11_14 = FeeSchedule(
    schedule_id="2026-11-14",
    effective_from=SCHEDULE_2026_11_14_START,
    effective_until=None,
    is_known=True,
    taker_coefficient_tier1=Decimal("0.095"),
    source_url=FEE_PAGE_URL,
    note=(
        "Taker fee = coefficient * contracts * p * (1 - p), rounded to the cent with "
        "banker's rounding. Coefficient by 30-day taker volume: 0.095 under $10k, "
        "0.085 under $50k, 0.075 under $250k, 0.0695 above; 0.0695 in an account's "
        "first 7 days."
    ),
)

FEE_SCHEDULES: tuple[FeeSchedule, ...] = (UNKNOWN_BEFORE_2026_11_14, SCHEDULE_2026_11_14)


def fee_schedule_for(window_start: int) -> FeeSchedule:
    """The schedule in force at a window's start time (Unix seconds UTC).

    The window start decides, because entries are made during the window. A window
    that starts exactly at a schedule's effective time belongs to that schedule.
    """
    matches = [
        schedule
        for schedule in FEE_SCHEDULES
        if (schedule.effective_from is None or window_start >= schedule.effective_from)
        and (schedule.effective_until is None or window_start < schedule.effective_until)
    ]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} fee schedules cover window start {window_start}")
    return matches[0]
