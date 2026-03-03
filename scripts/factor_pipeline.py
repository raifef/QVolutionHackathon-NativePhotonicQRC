from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from .pipeline import _resolve_surface_pca, sanitize_surface_matrix


def _pca_transform(pca: Any, x2d: np.ndarray) -> np.ndarray:
    if hasattr(pca, "transform"):
        return np.asarray(pca.transform(x2d), dtype=float)
    raise AttributeError("PCA object does not expose transform().")


def _pca_inverse(pca: Any, z2d: np.ndarray) -> np.ndarray:
    if hasattr(pca, "inverse_transform"):
        return np.asarray(pca.inverse_transform(z2d), dtype=float)
    raise AttributeError("PCA object does not expose inverse_transform().")


@dataclass
class FactorPipelineConfig:
    lookback: int = 20
    horizon: int = 6
    pca_factors: int = 8
    seed: int = 0


@dataclass
class FactorWindowData:
    x_train: np.ndarray
    y_train: np.ndarray
    origins: np.ndarray
    base_factors: np.ndarray


@dataclass
class FactorTestWindow:
    anchor: int
    x_test: np.ndarray
    y_true_delta: np.ndarray
    y_true_surface: np.ndarray
    base_factor: np.ndarray


@dataclass
class FactorFutureWindow:
    anchor: int
    x_test: np.ndarray
    base_factor: np.ndarray


class FactorPipeline:
    """Train-only PCA + factor-delta sequence builder."""

    def __init__(self, cfg: FactorPipelineConfig) -> None:
        self.cfg = FactorPipelineConfig(
            lookback=int(max(1, cfg.lookback)),
            horizon=int(max(1, cfg.horizon)),
            pca_factors=int(max(1, cfg.pca_factors)),
            seed=int(cfg.seed),
        )
        self.fit_surface_pca = _resolve_surface_pca()
        self.pca: Any | None = None
        self.price_scale: float = 1.0
        self.clip_cap: float = 1.0
        self._clean_full: np.ndarray | None = None
        self._scaled_full: np.ndarray | None = None
        self._factors_full: np.ndarray | None = None
        self._deltas_full: np.ndarray | None = None
        self._delta_mean: np.ndarray | None = None
        self._delta_std: np.ndarray | None = None
        self._train_end: int = 0

    @property
    def factor_dim(self) -> int:
        return int(self.cfg.pca_factors)

    def fit(self, train_surfaces: np.ndarray, full_surfaces: np.ndarray) -> "FactorPipeline":
        train_arr = np.asarray(train_surfaces, dtype=float)
        full_arr = np.asarray(full_surfaces, dtype=float)
        if train_arr.ndim != 2 or full_arr.ndim != 2:
            raise ValueError("train_surfaces/full_surfaces must be [T, D].")
        if train_arr.shape[1] != full_arr.shape[1]:
            raise ValueError("Feature dimension mismatch between train/full surfaces.")
        if train_arr.shape[0] <= int(self.cfg.lookback + self.cfg.horizon):
            raise ValueError("Insufficient train observations for factor pipeline.")

        clean_train, cap = sanitize_surface_matrix(train_arr)
        clean_full, _ = sanitize_surface_matrix(full_arr, cap_hint=float(cap))
        self.clip_cap = float(cap)
        self._clean_full = np.asarray(clean_full, dtype=float)
        self._train_end = int(clean_train.shape[0])

        scale = max(1.0, float(np.nanpercentile(clean_train, 95)))
        self.price_scale = float(scale)
        scaled_train = np.clip(np.nan_to_num(clean_train / scale, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        scaled_full = np.clip(np.nan_to_num(clean_full / scale, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        self._scaled_full = np.asarray(scaled_full, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            self.pca = self.fit_surface_pca(
                scaled_train,
                d_factors=int(self.cfg.pca_factors),
                seed=int(self.cfg.seed),
            )
            factors_full = _pca_transform(self.pca, scaled_full)
        factors_full = np.clip(np.nan_to_num(factors_full, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        self._factors_full = np.asarray(factors_full, dtype=float)

        deltas = np.zeros_like(factors_full, dtype=float)
        if deltas.shape[0] > 1:
            deltas[1:] = factors_full[1:] - factors_full[:-1]
        self._deltas_full = deltas

        ref = deltas[1 : self._train_end]
        if ref.size <= 0:
            ref = deltas[: self._train_end]
        mean = np.mean(ref, axis=0)
        std = np.std(ref, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        self._delta_mean = np.asarray(mean, dtype=float)
        self._delta_std = np.asarray(std, dtype=float)
        return self

    def _check_ready(self) -> None:
        if self.pca is None or self._factors_full is None or self._deltas_full is None:
            raise RuntimeError("FactorPipeline.fit must be called first.")
        if self._delta_mean is None or self._delta_std is None:
            raise RuntimeError("FactorPipeline normalizer state is missing.")

    def normalize_delta(self, delta_raw: np.ndarray) -> np.ndarray:
        self._check_ready()
        return (np.asarray(delta_raw, dtype=float) - self._delta_mean[None, ...]) / self._delta_std[None, ...]

    def denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        self._check_ready()
        return (np.asarray(delta_norm, dtype=float) * self._delta_std[None, ...]) + self._delta_mean[None, ...]

    def factors_full(self) -> np.ndarray:
        self._check_ready()
        return np.asarray(self._factors_full, dtype=float)

    def deltas_full(self, normalized: bool = False) -> np.ndarray:
        self._check_ready()
        raw = np.asarray(self._deltas_full, dtype=float)
        if not normalized:
            return raw
        out = np.zeros_like(raw, dtype=float)
        out[1:] = (raw[1:] - self._delta_mean[None, :]) / self._delta_std[None, :]
        return out

    def build_train_windows(self, train_end: int) -> FactorWindowData:
        self._check_ready()
        end = int(train_end)
        l = int(self.cfg.lookback)
        h = int(self.cfg.horizon)
        if end <= (l + h):
            raise ValueError(f"build_train_windows: train_end={end} too small for lookback={l}, horizon={h}.")
        deltas_n = self.deltas_full(normalized=True)
        factors = self.factors_full()
        origins = np.arange(l, end - h + 1, dtype=int)
        x = np.stack([deltas_n[t - l : t] for t in origins], axis=0)
        y = np.stack([deltas_n[t : t + h] for t in origins], axis=0)
        base_f = factors[origins - 1]
        return FactorWindowData(
            x_train=np.asarray(x, dtype=float),
            y_train=np.asarray(y, dtype=float),
            origins=origins,
            base_factors=np.asarray(base_f, dtype=float),
        )

    def build_test_window(self, anchor: int) -> FactorTestWindow:
        self._check_ready()
        a = int(anchor)
        l = int(self.cfg.lookback)
        h = int(self.cfg.horizon)
        if a < l:
            raise ValueError(f"build_test_window: anchor={a} < lookback={l}.")
        if (a + h) > int(self._factors_full.shape[0]):
            raise ValueError(f"build_test_window: anchor+h exceeds available range ({a}+{h}).")
        deltas_n = self.deltas_full(normalized=True)
        deltas_r = self.deltas_full(normalized=False)
        clean = np.asarray(self._clean_full, dtype=float)
        factors = self.factors_full()
        return FactorTestWindow(
            anchor=a,
            x_test=np.asarray(deltas_n[a - l : a][None, :, :], dtype=float),
            y_true_delta=np.asarray(deltas_r[a : a + h], dtype=float),
            y_true_surface=np.asarray(clean[a : a + h], dtype=float),
            base_factor=np.asarray(factors[a - 1], dtype=float),
        )

    def build_future_window(self, anchor: int) -> FactorFutureWindow:
        self._check_ready()
        a = int(anchor)
        l = int(self.cfg.lookback)
        n = int(self._factors_full.shape[0])
        if a < l:
            raise ValueError(f"build_future_window: anchor={a} < lookback={l}.")
        if a > n:
            raise ValueError(f"build_future_window: anchor={a} exceeds series length={n}.")
        deltas_n = self.deltas_full(normalized=True)
        factors = self.factors_full()
        return FactorFutureWindow(
            anchor=a,
            x_test=np.asarray(deltas_n[a - l : a][None, :, :], dtype=float),
            base_factor=np.asarray(factors[a - 1], dtype=float),
        )

    def reconstruct_surface_from_delta_norm(self, delta_pred_norm: np.ndarray, base_factor: np.ndarray) -> np.ndarray:
        self._check_ready()
        delta_pred_norm = np.asarray(delta_pred_norm, dtype=float)
        if delta_pred_norm.ndim == 2:
            delta_pred_norm = delta_pred_norm[None, :, :]
        if delta_pred_norm.ndim != 3:
            raise ValueError("delta_pred_norm must be [H, F] or [N, H, F].")
        base = np.asarray(base_factor, dtype=float)
        if base.ndim == 1:
            base = base[None, :]
        if base.ndim != 2:
            raise ValueError("base_factor must be [F] or [N, F].")
        if base.shape[0] == 1 and delta_pred_norm.shape[0] > 1:
            base = np.repeat(base, delta_pred_norm.shape[0], axis=0)
        if base.shape[0] != delta_pred_norm.shape[0]:
            raise ValueError("Batch mismatch between delta_pred_norm and base_factor.")

        delta_raw = self.denormalize_delta(delta_pred_norm)
        cum = np.cumsum(delta_raw, axis=1)
        factors_pred = base[:, None, :] + cum
        factors_pred = np.clip(np.nan_to_num(factors_pred, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

        pca = self.pca
        assert pca is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            if hasattr(pca, "inverse_transform_seq"):
                surf_scaled = np.asarray(pca.inverse_transform_seq(factors_pred), dtype=float)
            else:
                n, h, f = factors_pred.shape
                surf_scaled = _pca_inverse(pca, factors_pred.reshape(n * h, f)).reshape(n, h, -1)
        surf = np.clip(surf_scaled * float(self.price_scale), 0.0, float(self.clip_cap))
        return np.nan_to_num(surf, nan=0.0, posinf=float(self.clip_cap), neginf=0.0)
