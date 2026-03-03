from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class PseudoNext6Fold:
    anchor: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class PseudoNext6CV:
    """Blocked walk-forward CV in observation index space."""

    def __init__(
        self,
        dataset: int | np.ndarray | Iterable[object],
        lookback: int,
        horizon: int,
        n_anchors: int = 10,
        anchor_strategy: str = "tail",
        min_train: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_obs = int(self._infer_n_obs(dataset))
        self.lookback = int(max(1, lookback))
        self.horizon = int(max(1, horizon))
        self.n_anchors = int(max(1, n_anchors))
        self.anchor_strategy = str(anchor_strategy).strip().lower()
        self.min_train = int(max(self.lookback + 1, (self.lookback + 50) if min_train is None else int(min_train)))
        self.seed = int(seed)
        if self.n_obs < (self.min_train + self.horizon):
            raise ValueError(
                f"PseudoNext6CV: not enough observations (n_obs={self.n_obs}, "
                f"required>={self.min_train + self.horizon})."
            )

    @staticmethod
    def _infer_n_obs(dataset: int | np.ndarray | Iterable[object]) -> int:
        if isinstance(dataset, int):
            return int(dataset)
        if isinstance(dataset, np.ndarray):
            return int(dataset.shape[0])
        try:
            return int(len(dataset))  # type: ignore[arg-type]
        except Exception as exc:
            raise TypeError(f"Could not infer dataset length from type {type(dataset)}.") from exc

    def _candidate_anchors(self) -> np.ndarray:
        min_anchor = int(max(self.min_train, self.lookback))
        max_anchor = int(self.n_obs - self.horizon)
        if max_anchor < min_anchor:
            return np.asarray([], dtype=int)
        return np.arange(min_anchor, max_anchor + 1, dtype=int)

    def _tail_anchors(self, candidates: np.ndarray) -> np.ndarray:
        if candidates.size <= self.n_anchors:
            return candidates.copy()

        # Prefer near-tail blocked anchors spaced by horizon.
        max_anchor = int(candidates[-1])
        spaced = [max_anchor - k * self.horizon for k in range(self.n_anchors)]
        spaced = [a for a in reversed(spaced) if a >= int(candidates[0])]
        anchors = np.asarray(spaced, dtype=int)
        anchors = np.unique(anchors)
        if anchors.size >= self.n_anchors:
            return anchors[-self.n_anchors :]

        tail_start = max(0, candidates.size - max(self.n_anchors * max(2, self.horizon), self.n_anchors + 2))
        tail = candidates[tail_start:]
        idx = np.linspace(0, tail.size - 1, num=self.n_anchors, dtype=int)
        return np.unique(tail[idx])

    def _random_tail_anchors(self, candidates: np.ndarray) -> np.ndarray:
        if candidates.size <= self.n_anchors:
            return np.sort(candidates.copy())
        tail_start = max(0, candidates.size - max(self.n_anchors * max(3, self.horizon), self.n_anchors + 4))
        tail = candidates[tail_start:]
        rng = np.random.default_rng(self.seed)
        picked = rng.choice(tail, size=self.n_anchors, replace=False)
        return np.sort(np.asarray(picked, dtype=int))

    def anchors(self) -> np.ndarray:
        candidates = self._candidate_anchors()
        if candidates.size == 0:
            return candidates
        if self.anchor_strategy in {"tail", "tail_blocked"}:
            out = self._tail_anchors(candidates)
        elif self.anchor_strategy in {"tail_random", "random_tail"}:
            out = self._random_tail_anchors(candidates)
        else:
            raise ValueError(f"Unsupported anchor_strategy={self.anchor_strategy}.")
        out = np.asarray(np.sort(np.unique(out)), dtype=int)
        valid: list[int] = []
        for anchor in out.tolist():
            test_idx = np.arange(int(anchor), int(anchor) + self.horizon, dtype=int)
            if test_idx[-1] >= self.n_obs:
                continue
            if int(anchor) < self.min_train:
                continue
            valid.append(int(anchor))
        return np.asarray(valid, dtype=int)

    def split(self) -> list[tuple[np.ndarray, np.ndarray]]:
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for anchor in self.anchors().tolist():
            train_idx = np.arange(0, int(anchor), dtype=int)
            test_idx = np.arange(int(anchor), int(anchor) + self.horizon, dtype=int)
            folds.append((train_idx, test_idx))
        return folds

    def iter_folds(self) -> Iterator[PseudoNext6Fold]:
        for anchor in self.anchors().tolist():
            train_idx = np.arange(0, int(anchor), dtype=int)
            test_idx = np.arange(int(anchor), int(anchor) + self.horizon, dtype=int)
            yield PseudoNext6Fold(anchor=int(anchor), train_idx=train_idx, test_idx=test_idx)


def validate_fold_order(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if train_idx.ndim != 1 or test_idx.ndim != 1:
        raise ValueError("train_idx and test_idx must be 1D arrays.")
    if train_idx.size <= 0 or test_idx.size <= 0:
        raise ValueError("train_idx and test_idx must be non-empty.")
    if not np.all(np.diff(train_idx) == 1):
        raise ValueError("train_idx must be contiguous observations.")
    if not np.all(np.diff(test_idx) == 1):
        raise ValueError("test_idx must be consecutive sampled observations.")
    if int(train_idx[-1]) >= int(test_idx[0]):
        raise ValueError("Leakage: train indices must be strictly before test indices.")
