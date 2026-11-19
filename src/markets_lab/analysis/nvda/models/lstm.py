"""A small LSTM over the last 20 sessions of technical features.

Logistic regression and XGBoost look at one session's features at a time. The LSTM
reads the features of the last 20 sessions in order, so it can use how they evolved.

How a window is built, which is where look-ahead could creep in:

- The window for a session ends at that session and reaches back 19 sessions. It
  never includes a later session.
- When predicting the sessions that follow the training data, the first windows
  reach back into the end of the training sessions. Those are in the past, so that
  is allowed; the model keeps the last 19 training rows for this purpose.
- Rows passed to `predict` must be the sessions that follow the training sessions,
  in order. Predicting a session the model was trained on is refused.

Settings fixed in advance, not tuned on the results:

- One LSTM layer of 32 units, then a linear layer to a single probability.
- 10 passes over the training windows, batches of 64, Adam with learning rate 0.001
  and weight decay 0.0001.
- Features standardised with the mean and spread of the training rows only.
- Fixed random seed and a single thread, so the same training data always gives
  the same model on the same machine.
- The call is "up" when the probability is above 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from markets_lab.analysis.nvda.models.common import (
    DECISION_THRESHOLD,
    prediction_matrix,
    training_arrays,
)
from markets_lab.analysis.nvda.predictors import prediction_frame

SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 32
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0


class _Network(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(windows)
        return self.head(output[:, -1, :]).squeeze(-1)


class LSTMModel:
    """An LSTM classifier on windows of the last `SEQUENCE_LENGTH` sessions."""

    name = "lstm"

    def __init__(self) -> None:
        self._network: _Network | None = None
        self._columns: list[str] = []
        self._mean = np.zeros(0)
        self._spread = np.ones(0)
        self._tail = np.zeros((0, 0))
        self._trained_through: pd.Timestamp | None = None

    def fit(self, features: pd.DataFrame, target_up: pd.Series) -> None:
        matrix, target = training_arrays(features, target_up)
        if len(matrix) < 2 * SEQUENCE_LENGTH:
            raise ValueError(
                f"the LSTM needs at least {2 * SEQUENCE_LENGTH} training sessions, "
                f"got {len(matrix)}"
            )
        _check_ordered(features.index)
        mean = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
        spread[spread == 0] = 1.0
        scaled = (matrix - mean) / spread

        windows = torch.from_numpy(_windows(scaled))
        outcomes = torch.from_numpy(target[SEQUENCE_LENGTH - 1 :].astype("float32"))
        with _Reproducible():
            network = _Network(matrix.shape[1])
            optimiser = torch.optim.Adam(
                network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            loss_function = nn.BCEWithLogitsLoss()
            shuffler = torch.Generator().manual_seed(SEED)
            network.train()
            for _ in range(EPOCHS):
                order = torch.randperm(len(windows), generator=shuffler)
                for start in range(0, len(windows), BATCH_SIZE):
                    batch = order[start : start + BATCH_SIZE]
                    optimiser.zero_grad()
                    loss = loss_function(network(windows[batch]), outcomes[batch])
                    loss.backward()
                    optimiser.step()
            network.eval()

        self._network = network
        self._columns = list(features.columns)
        self._mean, self._spread = mean, spread
        self._tail = scaled[-(SEQUENCE_LENGTH - 1) :]
        self._trained_through = pd.Timestamp(features.index[-1])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._network is None or self._trained_through is None:
            raise RuntimeError("fit must be called before predict")
        matrix = prediction_matrix(features, self._columns)
        if len(features) == 0:
            return prediction_frame(features.index, np.zeros(0, dtype=bool), np.zeros(0))
        _check_ordered(features.index)
        if pd.Timestamp(features.index[0]) <= self._trained_through:
            raise ValueError(
                "the LSTM can only predict sessions after the ones it was trained on "
                f"(trained through {self._trained_through.date()})"
            )
        scaled = (matrix - self._mean) / self._spread
        # Row i of the batch gets the window ending at row i: the rows before it in
        # the batch, preceded by the end of the training data. Nothing after row i.
        history = np.vstack([self._tail, scaled])
        with _Reproducible(), torch.no_grad():
            logits = self._network(torch.from_numpy(_windows(history)))
            probability = torch.sigmoid(logits).numpy().astype("float64")
        if not np.isfinite(probability).all():
            raise ValueError("the LSTM produced a probability that is not a number")
        return prediction_frame(features.index, probability > DECISION_THRESHOLD, probability)


def _windows(rows: np.ndarray) -> np.ndarray:
    """Every run of `SEQUENCE_LENGTH` consecutive rows, as float32 (window, step, feature)."""
    view = sliding_window_view(rows, SEQUENCE_LENGTH, axis=0)
    return np.ascontiguousarray(view.transpose(0, 2, 1), dtype="float32")


def _check_ordered(index: pd.Index) -> None:
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("the LSTM needs sessions in increasing order with no repeats")


class _Reproducible:
    """Run a block with a fixed seed and one thread, then restore torch's settings."""

    def __enter__(self) -> None:
        self._threads = torch.get_num_threads()
        self._state = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(SEED)

    def __exit__(self, *_: object) -> None:
        torch.set_rng_state(self._state)
        torch.set_num_threads(self._threads)
