from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models_classical import train_classical_forecaster

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _tail_train_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    min_val: int = 8,
    frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x_train.shape[0])
    n_val = int(max(min_val, round(float(frac) * n)))
    n_val = int(max(1, min(n_val, max(1, n // 2))))
    return np.asarray(x_train[-n_val:], dtype=float), np.asarray(y_train[-n_val:], dtype=float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    s1 = float(np.std(aa))
    s2 = float(np.std(bb))
    if (not np.isfinite(s1)) or (not np.isfinite(s2)) or s1 <= 1e-12 or s2 <= 1e-12:
        return 0.0
    c = float(np.corrcoef(aa, bb)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _alpha_from_corr(corr: float, corr_min: float) -> float:
    c = float(corr)
    cmin = float(corr_min)
    if c <= cmin:
        return 0.0
    den = max(1e-12, 1.0 - cmin)
    return float(np.clip((c - cmin) / den, 0.0, 1.0))


@dataclass
class QuantumResidualCorrector:
    base_kind: str
    quantum_kind: str
    gain: float = 1.0
    corr_window: int = 40
    corr_min: float = 0.0
    seed: int = 0
    base_kwargs: dict[str, Any] = field(default_factory=dict)
    quantum_kwargs: dict[str, Any] = field(default_factory=dict)

    base_model: Any | None = None
    quantum_model: Any | None = None
    alpha_by_h: np.ndarray | None = None
    corr_by_h: np.ndarray | None = None
    failed: bool = False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "QuantumResidualCorrector":
        x_t = np.asarray(x_train, dtype=float)
        y_t = np.asarray(y_train, dtype=float)
        x_v = np.asarray(x_val, dtype=float)
        y_v = np.asarray(y_val, dtype=float)
        h = int(y_t.shape[1])
        self.alpha_by_h = np.zeros((h,), dtype=float)
        self.corr_by_h = np.zeros((h,), dtype=float)

        try:
            base_kwargs = dict(self.base_kwargs)
            base_kwargs.setdefault("nn_train_loss", "rmse")
            base_kwargs.setdefault("use_full_training", True)
            self.base_model = train_classical_forecaster(
                x_train=x_t,
                y_train=y_t,
                x_val=x_v,
                y_val=y_v,
                kind=str(self.base_kind),
                seed=int(self.seed),
                **base_kwargs,
            )
            base_pred_train = np.asarray(self.base_model.predict(x_t), dtype=float)
            base_pred_val = np.asarray(self.base_model.predict(x_v), dtype=float)
            resid_train = np.asarray(y_t - base_pred_train, dtype=float)
            resid_val = np.asarray(y_v - base_pred_val, dtype=float)

            q_kwargs = dict(self.quantum_kwargs)
            q_kwargs.setdefault("nn_train_loss", "rmse")
            q_kwargs.setdefault("use_full_training", True)
            self.quantum_model = train_classical_forecaster(
                x_train=x_t,
                y_train=resid_train,
                x_val=x_v,
                y_val=resid_val,
                kind=str(self.quantum_kind),
                seed=int(self.seed),
                **q_kwargs,
            )

            n_cal = int(max(1, min(int(self.corr_window), x_t.shape[0])))
            x_cal = x_t[-n_cal:]
            resid_true_cal = resid_train[-n_cal:]
            resid_pred_cal = np.asarray(self.quantum_model.predict(x_cal), dtype=float)
            corr_arr = np.zeros((h,), dtype=float)
            alpha_arr = np.zeros((h,), dtype=float)
            for k in range(h):
                corr_k = _safe_corr(resid_true_cal[:, k, :], resid_pred_cal[:, k, :])
                corr_arr[k] = float(corr_k)
                alpha_arr[k] = float(_alpha_from_corr(corr_k, float(self.corr_min)))
            self.corr_by_h = corr_arr
            self.alpha_by_h = alpha_arr
        except Exception:
            self.failed = True
            self.corr_by_h = np.zeros((h,), dtype=float)
            self.alpha_by_h = np.zeros((h,), dtype=float)
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise RuntimeError("QuantumResidualCorrector.fit must be called before predict.")
        base_pred = np.asarray(self.base_model.predict(np.asarray(x_seq, dtype=float)), dtype=float)
        if self.quantum_model is None or self.alpha_by_h is None:
            return base_pred
        q_pred = np.asarray(self.quantum_model.predict(np.asarray(x_seq, dtype=float)), dtype=float)
        alpha = np.asarray(self.alpha_by_h, dtype=float)[None, :, None]
        return np.asarray(base_pred + (alpha * float(self.gain) * q_pred), dtype=float)


@dataclass
class EnsembleForecaster:
    member_names: list[str]
    weights: np.ndarray | None = None

    def _fit_weights(self, member_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # member_preds: [N, M, H, F], y_true: [N, H, F]
        p = np.asarray(member_preds, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if p.ndim != 4 or y.ndim != 3:
            raise ValueError("Invalid shapes for ensemble fit.")
        n, m, h, f = p.shape
        x2 = np.transpose(p, (0, 2, 3, 1)).reshape(n * h * f, m)
        y2 = y.reshape(n * h * f)
        x2 = np.clip(np.nan_to_num(x2, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        y2 = np.clip(np.nan_to_num(y2, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        if SKLEARN_AVAILABLE:
            lr = LinearRegression(fit_intercept=False, positive=True)
            lr.fit(x2, y2)
            w = np.asarray(lr.coef_, dtype=float).reshape(-1)
        else:
            # Fallback: unconstrained solve, then clip to nonnegative.
            w, *_ = np.linalg.lstsq(x2, y2, rcond=None)
            w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        if (not np.all(np.isfinite(w))) or float(np.sum(w)) <= 1e-12:
            w = np.zeros((m,), dtype=float)
            w[0] = 1.0
        else:
            w = np.clip(w, 0.0, None)
            s = float(np.sum(w))
            if s <= 1e-12:
                w = np.zeros((m,), dtype=float)
                w[0] = 1.0
            else:
                w = w / s
        return np.asarray(w, dtype=float)

    def fit(self, member_preds: np.ndarray, y_true: np.ndarray) -> "EnsembleForecaster":
        w = self._fit_weights(member_preds=member_preds, y_true=y_true)
        self.weights = np.asarray(w, dtype=float)
        return self

    def predict(self, member_preds: np.ndarray) -> np.ndarray:
        p = np.asarray(member_preds, dtype=float)
        if p.ndim != 4:
            raise ValueError("member_preds must be [N, M, H, F].")
        if self.weights is None:
            raise RuntimeError("EnsembleForecaster.fit must be called before predict.")
        w = np.asarray(self.weights, dtype=float)
        if p.shape[1] != w.shape[0]:
            raise ValueError(f"Member count mismatch: preds={p.shape[1]} weights={w.shape[0]}.")
        p = np.clip(np.nan_to_num(p, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        return np.einsum("nmhf,m->nhf", p, w, optimize=True)
