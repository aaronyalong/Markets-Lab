"""Data-quality counts for the Polymarket database. Counting only, no analysis."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

WINDOW_STEP_SECONDS = 900
MINUTE_12_OFFSET = 12 * 60
MINUTE_13_OFFSET = 13 * 60


@dataclass(frozen=True, slots=True)
class SeriesQuality:
    """Counts for one series.

    A window "has minute N" when it has at least one price point whose time falls
    within minute N of the window (N:00 to N:59) with both prices present.
    """

    series_slug: str
    markets: int
    complete: int
    with_outcome: int
    with_outcome_from_prices: int
    with_history_fetched: int
    price_points: int
    has_minute_12: int
    has_minute_13: int
    has_neither_minute: int
    outcome_mismatches: int
    fee_schedule_unknown: int
    fee_schedule_known: int
    missing_windows: int
    first_window_start: int | None
    last_window_start: int | None


def missing_windows(window_starts: Iterable[int], step: int = WINDOW_STEP_SECONDS) -> list[int]:
    """Start times absent from the regular grid between the first and last window.

    The grid runs from the earliest start to the latest in steps of `step` seconds.
    Returns the missing start times in order; an empty or single-window input has
    no gaps.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    present = set(window_starts)
    if not present:
        return []
    return [start for start in range(min(present), max(present) + 1, step) if start not in present]


def _has_minute_sql(offset: int) -> str:
    return f"""
        EXISTS (
            SELECT 1 FROM price_points p
            WHERE p.market_slug = m.market_slug
              AND p.long_price IS NOT NULL
              AND p.short_price IS NOT NULL
              AND p.snapshot_ts >= m.window_start + {offset}
              AND p.snapshot_ts < m.window_start + {offset + 60}
        )
    """


_PER_MARKET = f"""
    SELECT
        m.series_slug,
        m.window_start,
        m.outcome IS NOT NULL AS has_outcome,
        m.outcome_from_prices IS NOT NULL AS has_price_outcome,
        m.history_fetched_at IS NOT NULL AS has_history,
        m.outcome IS NOT NULL AND m.outcome_from_prices IS NOT NULL
            AND m.outcome != m.outcome_from_prices AS mismatch,
        f.is_known AS fee_known,
        (SELECT count(*) FROM price_points p WHERE p.market_slug = m.market_slug) AS points,
        {_has_minute_sql(MINUTE_12_OFFSET)} AS has_m12,
        {_has_minute_sql(MINUTE_13_OFFSET)} AS has_m13
    FROM markets m
    JOIN fee_schedules f ON f.schedule_id = m.fee_schedule_id
"""


def series_quality(conn: sqlite3.Connection) -> list[SeriesQuality]:
    """Quality counts for every series in the database, ordered by series slug."""
    rows = conn.execute(
        f"""
        SELECT
            series_slug,
            count(*),
            sum(has_outcome AND has_history),
            sum(has_outcome),
            sum(has_price_outcome),
            sum(has_history),
            sum(points),
            sum(has_m12),
            sum(has_m13),
            sum(NOT has_m12 AND NOT has_m13),
            sum(mismatch),
            sum(NOT fee_known),
            sum(fee_known),
            min(window_start),
            max(window_start)
        FROM ({_PER_MARKET})
        GROUP BY series_slug
        ORDER BY series_slug
        """
    ).fetchall()
    result: list[SeriesQuality] = []
    for row in rows:
        slug = row[0]
        starts = [
            r[0]
            for r in conn.execute("SELECT window_start FROM markets WHERE series_slug = ?", (slug,))
        ]
        result.append(
            SeriesQuality(
                series_slug=slug,
                markets=row[1],
                complete=row[2],
                with_outcome=row[3],
                with_outcome_from_prices=row[4],
                with_history_fetched=row[5],
                price_points=row[6],
                has_minute_12=row[7],
                has_minute_13=row[8],
                has_neither_minute=row[9],
                outcome_mismatches=row[10],
                fee_schedule_unknown=row[11],
                fee_schedule_known=row[12],
                missing_windows=len(missing_windows(starts)),
                first_window_start=row[13],
                last_window_start=row[14],
            )
        )
    return result


def error_counts(conn: sqlite3.Connection, run_id: int | None = None) -> dict[str, int]:
    """Recorded backfill errors by stage, for one run or for all runs."""
    if run_id is None:
        rows = conn.execute("SELECT stage, count(*) FROM backfill_errors GROUP BY stage")
    else:
        rows = conn.execute(
            "SELECT stage, count(*) FROM backfill_errors WHERE run_id = ? GROUP BY stage",
            (run_id,),
        )
    return {stage: n for stage, n in rows}


def format_quality(conn: sqlite3.Connection) -> str:
    """A plain-text table of the quality counts, one block per series."""
    lines: list[str] = []
    for q in series_quality(conn):
        lines += [
            f"{q.series_slug}",
            f"  windows stored              {q.markets}",
            f"  price history fetched       {q.with_history_fetched}",
            f"  price points                {q.price_points}",
            f"  has minute-12 point         {q.has_minute_12}",
            f"  has minute-13 point         {q.has_minute_13}",
            f"  has neither                 {q.has_neither_minute}",
            f"  outcome from index prices   {q.with_outcome_from_prices}",
            f"  outcome from settlement     {q.with_outcome}",
            f"  the two outcomes disagree   {q.outcome_mismatches}",
            f"  complete (history+settled)  {q.complete}",
            f"  fee schedule unknown        {q.fee_schedule_unknown}",
            f"  fee schedule known          {q.fee_schedule_known}",
            f"  gaps in 15-minute grid      {q.missing_windows}",
        ]
    all_errors = error_counts(conn)
    total = sum(all_errors.values())
    detail = ", ".join(f"{stage} {n}" for stage, n in sorted(all_errors.items()))
    lines.append(f"errors recorded (all runs)    {total}" + (f" ({detail})" if detail else ""))
    return "\n".join(lines)
