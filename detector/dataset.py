"""Reproduce the Lab 1 Task 1 dataset exactly, and expose its 70/30 split.

Lab 1 built its features in `lab1/src/task1_supervised.py::load_dataset`: read the
cleaned parquet, map Label to binary (Benign=0, anything else=1), and drop the
leakage columns. The split is the first-version Lab 1 one: first 70% of rows -> train,
last 30% -> test, in the parquet's order. That order is the raw timestamp, which uses a
12-hour clock without AM/PM, so it is NOT true time order (the afternoon sorts first);
see scripts/split_audit.py and report section 2. Lab 2 Task 1A depends on this being byte-identical
to Lab 1, because membership inference asks "was this flow in the training set".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

WEDNESDAY_PARQUET = ROOT / "data" / "processed" / "cleaned_wednesday_14-02-2018.parquet"
THURSDAY_PARQUET = ROOT / "data" / "processed" / "cleaned_thursday_01-03-2018.parquet"

# Copied from lab1/src/config.py so the two repos cannot drift silently.
LEAKAGE_COLUMNS = [
    "Dst Port",
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
]
LABEL_COL = "Label"
TEMPORAL_TRAIN_FRACTION = 0.70


@dataclass
class Split:
    """One first-version Lab 1 style 70/30 split, with row positions preserved.

    `train_index` / `test_index` are positions in the cleaned parquet, which is what
    makes ground-truth membership available for scoring Task 1A.
    """

    name: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_index: np.ndarray
    test_index: np.ndarray
    timestamps: pd.Series

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def rows(self, positions: np.ndarray) -> np.ndarray:
        """Gather feature rows by absolute position in the cleaned parquet.

        The split is a contiguous prefix/suffix, so a position maps to
        `X_train[pos]` if it is in the train prefix and `X_test[pos - len(train)]`
        otherwise. Membership is then just `pos < len(train)`.
        """
        positions = np.asarray(positions)
        n_train = len(self.X_train)
        out = np.empty((len(positions), self.n_features))
        in_train = positions < n_train
        if in_train.any():
            out[in_train] = self.X_train[positions[in_train]]
        if (~in_train).any():
            out[~in_train] = self.X_test[positions[~in_train] - n_train]
        return out

    def labels(self, positions: np.ndarray) -> np.ndarray:
        """True labels for absolute positions, same mapping as `rows`."""
        positions = np.asarray(positions)
        n_train = len(self.X_train)
        y = np.empty(len(positions), dtype=np.int64)
        in_train = positions < n_train
        y[in_train] = self.y_train[positions[in_train]]
        y[~in_train] = self.y_test[positions[~in_train] - n_train]
        return y

    def is_member(self, positions: np.ndarray) -> np.ndarray:
        """Ground truth for Task 1A: was this flow in the training set."""
        return np.asarray(positions) < len(self.X_train)


def load_dataset(parquet_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Same contract as Lab 1: X (no leakage cols), binary y, Timestamp."""
    df = pd.read_parquet(parquet_path)
    timestamps = df["Timestamp"].copy()
    y = (df[LABEL_COL] != "Benign").astype(int)
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns] + [LABEL_COL]
    X = df.drop(columns=cols_to_drop)
    return X, y, timestamps


def build_split(parquet_path: Path = WEDNESDAY_PARQUET, name: str = "Wednesday") -> Split:
    """Temporal 70/30 split by row order, matching Lab 1's `iloc[:split_idx]` slicing."""
    X, y, timestamps = load_dataset(parquet_path)
    split_idx = int(len(X) * TEMPORAL_TRAIN_FRACTION)
    positions = np.arange(len(X))
    return Split(
        name=name,
        feature_names=list(X.columns),
        X_train=X.iloc[:split_idx].to_numpy(dtype=np.float64),
        y_train=y.iloc[:split_idx].to_numpy(dtype=np.int64),
        X_test=X.iloc[split_idx:].to_numpy(dtype=np.float64),
        y_test=y.iloc[split_idx:].to_numpy(dtype=np.int64),
        train_index=positions[:split_idx],
        test_index=positions[split_idx:],
        timestamps=timestamps,
    )
