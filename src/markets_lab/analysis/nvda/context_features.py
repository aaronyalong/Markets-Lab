"""Market context and results-date features, added to the technical ones.

The technical features in `features.py` are not changed. This module adds columns
beside them and builds a dataset on exactly the same sessions, so a model given the
extra columns is tested on the same days as the models without them.

The timing rule is the same as everywhere in this experiment, with one addition:

- Row *t* describes what is known once session *t* has closed. Index closes come
  from that same session.
- Results are known on row *t* only if SEC accepted the filing before the cut-off
  for session *t*: 20:00 New York time on the day of the session
  (`information_cutoff`). The backtest and the live job use the same cut-off, so a
  call does not depend on what time the job happened to run. A live call must be
  made after its cut-off (`build_context_dataset` refuses otherwise), which means a
  call can only use results published before its own timestamp.

The columns were fixed before any model saw them and must not be tuned on results:

- For each stock index (S&P 500, Nasdaq 100, semiconductors): its return over 1 and
  5 sessions, and NVDA's return over 21 sessions minus the index's. (NVDA against
  the index over 1 and 5 sessions is already the difference of two columns.)
- VIX: the log of its level and its change over 1 and 5 sessions.
- Results: whether none are known yet, the number of sessions since the latest
  known ones, and whether they came out after this session's close, in which case
  the next session is the first to trade on them.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from markets_lab.analysis.nvda.features import (
    FEATURE_COLUMNS,
    Dataset,
    build_dataset,
)
from markets_lab.collectors.market_context import align_to_sessions
from markets_lab.collectors.nvda_earnings import ResultsPublication, known_at

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
CUTOFF_TIME = time(20, 0)
"""New York time after which a session's call may be made. Results filings accepted
after the close have all arrived by 16:57 so far; the job's earliest run is 22:30."""

STOCK_INDICES = ("sp500", "nasdaq100", "semis")
INDEX_RETURN_LAGS = (1, 5)
RELATIVE_LAG = 21
VIX = "vix"
VIX_CHANGE_LAGS = (1, 5)

RESULTS_UNKNOWN = "results_unknown"
SESSIONS_SINCE_RESULTS = "sessions_since_results"
RESULTS_AFTER_CLOSE = "results_after_close"
MAX_SESSIONS_SINCE_RESULTS = 80
"""The count stops here, and is set to this while no results are known. Real gaps
between results filings are at most 69 sessions, so it only binds for the latter."""

CONTEXT_COLUMNS: tuple[str, ...] = (
    *(f"{name}_ret_{lag}" for name in STOCK_INDICES for lag in INDEX_RETURN_LAGS),
    *(f"nvda_vs_{name}_{RELATIVE_LAG}" for name in STOCK_INDICES),
    "vix_log",
    *(f"vix_chg_{lag}" for lag in VIX_CHANGE_LAGS),
)
RESULTS_COLUMNS: tuple[str, ...] = (RESULTS_UNKNOWN, SESSIONS_SINCE_RESULTS, RESULTS_AFTER_CLOSE)
EXTENDED_FEATURE_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, *CONTEXT_COLUMNS, *RESULTS_COLUMNS)


class CallTooEarlyError(RuntimeError):
    """A call was asked for before the information cut-off of its session."""


def information_cutoff(session: date) -> datetime:
    """The moment up to which information counts for the call based on `session`."""
    return datetime.combine(session, CUTOFF_TIME, tzinfo=MARKET_TZ)


def compute_context_features(prices: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Context features for every session in `prices`, from that session and earlier.

    `context` holds the index closes (see `collectors.market_context`). Every
    session in `prices` must be present in it; nothing is filled. Early rows are NaN
    until a feature's window is full (the longest needs 21 earlier sessions).
    """
    sessions = _sessions(prices)
    closes = align_to_sessions(context, sessions)
    log_adj = _log(prices["adj_close"])

    columns: dict[str, pd.Series] = {}
    for name in STOCK_INDICES:
        log_index = _log(closes[name])
        for lag in INDEX_RETURN_LAGS:
            columns[f"{name}_ret_{lag}"] = log_index.diff(lag)
        columns[f"nvda_vs_{name}_{RELATIVE_LAG}"] = log_adj.diff(RELATIVE_LAG) - log_index.diff(
            RELATIVE_LAG
        )
    log_vix = _log(closes[VIX])
    columns["vix_log"] = log_vix
    for lag in VIX_CHANGE_LAGS:
        columns[f"vix_chg_{lag}"] = log_vix.diff(lag)
    return pd.DataFrame(columns, index=prices.index)[list(CONTEXT_COLUMNS)]


def compute_results_features(
    sessions: pd.DatetimeIndex, publications: Sequence[ResultsPublication]
) -> pd.DataFrame:
    """Results features for every session, using only filings known at its cut-off.

    For each session the known filings are exactly `known_at(publications, cut-off)`.
    The count of sessions is taken on `sessions`, so pass the whole price history,
    not a slice of it.
    """
    if sessions.tz is not None or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("sessions must be unique timezone-free dates in increasing order")
    days = [pd.Timestamp(stamp).date() for stamp in sessions]
    cutoffs = [information_cutoff(day) for day in days]
    closes = [datetime.combine(day, MARKET_CLOSE, tzinfo=MARKET_TZ) for day in days]

    unknown = np.zeros(len(days))
    since = np.zeros(len(days))
    after_close = np.zeros(len(days))
    for row, cutoff in enumerate(cutoffs):
        known = known_at(publications, cutoff)
        if not known:
            unknown[row] = 1.0
            since[row] = MAX_SESSIONS_SINCE_RESULTS
            continue
        latest = max(item.accepted for item in known)
        # The first session whose cut-off is after the filing: the first to know it.
        first_row = bisect.bisect_right(cutoffs, latest)
        since[row] = min(row - first_row, MAX_SESSIONS_SINCE_RESULTS)
        if row == first_row and latest >= closes[row]:
            after_close[row] = 1.0
    return pd.DataFrame(
        {RESULTS_UNKNOWN: unknown, SESSIONS_SINCE_RESULTS: since, RESULTS_AFTER_CLOSE: after_close},
        index=sessions,
    )[list(RESULTS_COLUMNS)]


def build_context_dataset(
    prices: pd.DataFrame,
    context: pd.DataFrame,
    publications: Sequence[ResultsPublication],
    *,
    as_of: datetime,
) -> Dataset:
    """The technical dataset with the context and results columns added.

    The sessions, labels and technical columns are those of `build_dataset(prices)`.
    `as_of` is the timestamp of the call being prepared (or the time a backtest is
    run). It must not be before the cut-off of the last session: each row uses
    results accepted before its own cut-off, and no cut-off is later than the last
    one, so nothing published at or after `as_of` can reach any row.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    base = build_dataset(prices)
    last_session, _ = base.latest()
    cutoff = information_cutoff(last_session)
    if as_of < cutoff:
        raise CallTooEarlyError(
            f"too early for a call based on {last_session}: information counts up to "
            f"{cutoff.strftime('%Y-%m-%d %H:%M %Z')}, and the call is timed "
            f"{as_of.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M %Z')}"
        )

    sessions = _sessions(prices)
    extra = pd.concat(
        [
            compute_context_features(prices, context),
            compute_results_features(sessions, publications),
        ],
        axis=1,
    ).loc[base.features.index]
    if not bool(np.isfinite(extra.to_numpy()).all()):
        bad = extra.index[~np.isfinite(extra.to_numpy()).all(axis=1)]
        raise ValueError(f"context features are incomplete, first on {pd.Timestamp(bad[0]).date()}")
    features = pd.concat([base.features, extra], axis=1)[list(EXTENDED_FEATURE_COLUMNS)]
    return Dataset(features=features, target_up=base.target_up, next_return=base.next_return)


def _sessions(prices: pd.DataFrame) -> pd.DatetimeIndex:
    index = prices.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by session date")
    return index


def _log(values: pd.Series) -> pd.Series:
    return pd.Series(np.log(values.to_numpy(dtype="float64")), index=values.index)
