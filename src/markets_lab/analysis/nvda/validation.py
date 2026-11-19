"""Walk-forward validation: every call is made by a model that only saw the past.

Row *t* of a dataset holds the features known after session *t* closes and the
outcome of session *t+1*. So the outcome on row *t* becomes known when session
*t+1* closes, which is the moment the call on row *t+1* is made.

That gives the rule this module enforces: a model that makes the call on row *s*
may be trained on rows before *s* and on nothing else. Rows are used in blocks: the
model is refitted every `refit_every` sessions on all rows before the block
(an expanding window), then makes the calls for that block without being refitted.

`walk_forward` checks the rule on every fold and raises `LookAheadError` if a fold
would train on a row at or after the first row it is tested on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from markets_lab.analysis.nvda.features import Dataset
from markets_lab.analysis.nvda.predictors import PREDICTION_COLUMNS, Predictor

SESSIONS_PER_YEAR = 252
FOLD = "fold"
TRAINED_THROUGH = "trained_through"


class LookAheadError(RuntimeError):
    """A fold would let a model see data it could not have had at the time."""


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """How the walk-forward is laid out. The defaults are the approved definitions."""

    initial_train_sessions: int = 5 * SESSIONS_PER_YEAR
    refit_every: int = 21

    def __post_init__(self) -> None:
        if self.initial_train_sessions < 1:
            raise ValueError("initial_train_sessions must be at least 1")
        if self.refit_every < 1:
            raise ValueError("refit_every must be at least 1")


@dataclass(frozen=True, slots=True)
class Fold:
    """Train on rows [0, train_end), then make the calls for rows [test_start, test_end)."""

    number: int
    train_end: int
    test_start: int
    test_end: int


def make_folds(n_rows: int, config: WalkForwardConfig) -> list[Fold]:
    """Folds that cover every row after the initial training window exactly once."""
    if n_rows <= config.initial_train_sessions:
        raise ValueError(
            f"{n_rows} sessions is not more than the initial training window of "
            f"{config.initial_train_sessions}"
        )
    folds: list[Fold] = []
    for number, start in enumerate(
        range(config.initial_train_sessions, n_rows, config.refit_every)
    ):
        folds.append(
            Fold(
                number=number,
                train_end=start,
                test_start=start,
                test_end=min(start + config.refit_every, n_rows),
            )
        )
    return folds


def walk_forward(
    dataset: Dataset,
    make_predictor: Callable[[], Predictor],
    config: WalkForwardConfig | None = None,
    *,
    folds: Sequence[Fold] | None = None,
) -> pd.DataFrame:
    """Out-of-sample calls for every session after the initial training window.

    A fresh predictor is built for each fold, so nothing carries over except what
    the predictor learns from that fold's training rows. The dataset must be fully
    labelled (use `Dataset.labelled()`).

    Returns one row per tested session with the predictor's columns plus `fold` and
    `trained_through`, the last session the model for that row was trained on.
    """
    if bool(dataset.target_up.isna().any()):
        raise ValueError("walk_forward needs a fully labelled dataset; use Dataset.labelled()")
    index = dataset.features.index
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("dataset sessions must be unique and in increasing order")
    if folds is None:
        folds = make_folds(len(dataset), config or WalkForwardConfig())

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        _check_fold(fold, len(dataset))
        train_features = dataset.features.iloc[: fold.train_end]
        train_target = dataset.target_up.iloc[: fold.train_end]
        test_features = dataset.features.iloc[fold.test_start : fold.test_end]

        predictor = make_predictor()
        predictor.fit(train_features, train_target)
        calls = predictor.predict(test_features)

        if tuple(calls.columns) != PREDICTION_COLUMNS or not calls.index.equals(
            test_features.index
        ):
            raise ValueError(f"{predictor.name}: predict returned the wrong columns or sessions")
        calls = calls.copy()
        calls[FOLD] = fold.number
        calls[TRAINED_THROUGH] = index[fold.train_end - 1]
        pieces.append(calls)

    result = pd.concat(pieces)
    if result.index.has_duplicates:
        raise LookAheadError("folds overlap: a session was tested more than once")
    return result


def _check_fold(fold: Fold, n_rows: int) -> None:
    if fold.train_end < 1:
        raise ValueError(f"fold {fold.number}: nothing to train on")
    if not fold.test_start < fold.test_end <= n_rows:
        raise ValueError(f"fold {fold.number}: test rows are empty or out of range")
    if fold.train_end > fold.test_start:
        raise LookAheadError(
            f"fold {fold.number}: trains on rows up to {fold.train_end - 1} "
            f"but is tested from row {fold.test_start}"
        )
