"""The paper strategy's live record, worked out from the log alone.

The log holds no returns, so the live record counts sessions: how many the
strategy was Holding or In cash for, and which way each of them closed.

Sessions without an entry (a night when one of the six models made no call) are
left out of BOTH sides: the strategy has no state for them, and buy-and-hold is
compared on exactly the sessions the strategy has an entry for. The state from the
session before is never carried forward. Such sessions are counted as missed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from markets_lab.analysis.nvda.prediction_log import (
    Outcome,
    PaperStrategyEntry,
    Record,
    scored_paper_strategy,
)


@dataclass(frozen=True, slots=True)
class LiveRecord:
    """Counts over the sessions that have both an entry and an outcome.

    `sessions` is the number compared; `went_up` is how many of those closed up,
    which is buy-and-hold's side of the comparison on the same sessions.
    """

    sessions: int
    missed: int
    holding: int
    holding_went_up: int
    in_cash: int
    in_cash_not_up: int
    went_up: int

    @property
    def right_side(self) -> int:
        """Sessions the strategy was Holding and closed up, or In cash and did not."""
        return self.holding_went_up + self.in_cash_not_up


def compared_sessions(records: Sequence[Record]) -> pd.DataFrame:
    """The sessions both sides are compared on: one row per entry with an outcome.

    Columns: `target_session`, `holding`, `went_up`.
    """
    scored = scored_paper_strategy(records)
    done = scored[scored["went_up"].notna()]
    return pd.DataFrame(
        {
            "target_session": list(done["target_session"]),
            "holding": [bool(value) for value in done["holding"]],
            "went_up": [bool(value) for value in done["went_up"]],
        }
    )


def missed_sessions(records: Sequence[Record]) -> list[date]:
    """Finished sessions since the strategy's first entry that it has no entry for."""
    entered = {r.based_on_session for r in records if isinstance(r, PaperStrategyEntry)}
    if not entered:
        return []
    start = min(entered)
    return [
        r.session
        for r in records
        if isinstance(r, Outcome)
        and r.previous_session >= start
        and r.previous_session not in entered
    ]


def live_record(records: Sequence[Record]) -> LiveRecord:
    """The strategy's live counts, with missed sessions left out of both sides."""
    compared = compared_sessions(records)
    is_holding = compared["holding"].astype(bool)
    went_up = compared["went_up"].astype(bool)
    return LiveRecord(
        sessions=len(compared),
        missed=len(missed_sessions(records)),
        holding=int(is_holding.sum()),
        holding_went_up=int((is_holding & went_up).sum()),
        in_cash=int((~is_holding).sum()),
        in_cash_not_up=int((~is_holding & ~went_up).sum()),
        went_up=int(went_up.sum()),
    )
