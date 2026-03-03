from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any
import warnings

import numpy as np

from .photonic_memory_state import (
    PhotonicMemoryStateParams,
    build_photonic_memory_state_with_meta,
    feedback_sanity_check,
    state_feature_statistics,
)


def _hash_array(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    nrm = np.maximum(nrm, float(max(1e-15, eps)))
    return a / nrm


def _rbf_kernel(x_left: np.ndarray, x_right: np.ndarray, gamma: float) -> np.ndarray:
    xl = np.nan_to_num(np.asarray(x_left, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xr = np.nan_to_num(np.asarray(x_right, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        left_sq = np.sum(xl * xl, axis=1, keepdims=True)
        right_sq = np.sum(xr * xr, axis=1, keepdims=True).T
        dist2 = np.maximum(left_sq + right_sq - 2.0 * (xl @ xr.T), 0.0)
    dist2 = np.nan_to_num(dist2, nan=0.0, posinf=1e6, neginf=0.0)
    dist2 = np.clip(dist2, 0.0, 1e6)
    k = np.exp(-float(gamma) * dist2)
    return np.nan_to_num(k, nan=0.0, posinf=1.0, neginf=0.0)


def _solve_kernel_ridge(k_train: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    k = np.asarray(k_train, dtype=np.float64)
    y2 = np.asarray(y, dtype=np.float64)
    lam = float(max(1e-12, ridge))
    n = int(k.shape[0])
    eye = np.eye(n, dtype=np.float64)
    a = k + lam * eye
    try:
        return np.linalg.solve(a, y2)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, y2, rcond=1e-10)[0]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    den = np.maximum(np.abs(yt), float(max(1e-12, eps)))
    return float(np.mean(np.abs(yp - yt) / den) * 100.0)


def _optimal_residual_scale(
    y_true_resid: np.ndarray,
    y_pred_resid: np.ndarray,
    *,
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> float:
    yt = np.asarray(y_true_resid, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_resid, dtype=np.float64).reshape(-1)
    den = float(np.dot(yp, yp))
    if not np.isfinite(den) or den <= 1e-12:
        return float(min_scale)
    num = float(np.dot(yp, yt))
    if not np.isfinite(num):
        return float(min_scale)
    s = num / den
    if not np.isfinite(s):
        return float(min_scale)
    return float(np.clip(s, float(min_scale), float(max_scale)))


def _fit_two_source_blend(
    y_true: np.ndarray,
    src_qk: np.ndarray,
    src_aux: np.ndarray,
    *,
    max_weight: float = 1.5,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    q = np.asarray(src_qk, dtype=np.float64).reshape(-1)
    a = np.asarray(src_aux, dtype=np.float64).reshape(-1)
    x = np.stack([q, a], axis=1)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 1.0, 0.0
    if float(np.linalg.norm(x)) <= 1e-12:
        return 0.0, 0.0
    try:
        w, *_ = np.linalg.lstsq(x, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    w_qk = float(np.clip(w[0], 0.0, float(max_weight)))
    w_aux = float(np.clip(w[1], 0.0, float(max_weight)))
    return w_qk, w_aux


def _fit_two_source_blend_grid(
    y_true_level: np.ndarray,
    base_level: np.ndarray,
    src_qk_resid: np.ndarray,
    src_aux_resid: np.ndarray,
    *,
    max_weight: float = 1.5,
    n_grid: int = 31,
    objective: str = "mape",
    eps: float = 1e-8,
) -> tuple[float, float]:
    y = np.asarray(y_true_level, dtype=np.float64)
    b = np.asarray(base_level, dtype=np.float64)
    q = np.asarray(src_qk_resid, dtype=np.float64)
    a = np.asarray(src_aux_resid, dtype=np.float64)
    if not (np.isfinite(y).all() and np.isfinite(b).all() and np.isfinite(q).all() and np.isfinite(a).all()):
        return 1.0, 0.0

    obj = str(objective).strip().lower()
    if obj not in {"rmse", "mape"}:
        obj = "mape"
    n = int(max(2, n_grid))
    ws = np.linspace(0.0, float(max_weight), num=n, dtype=np.float64)

    best_w = (1.0, 0.0)
    best_obj = float("inf")
    best_tie = float("inf")
    for w_qk in ws:
        for w_aux in ws:
            pred = b + float(w_qk) * q + float(w_aux) * a
            if obj == "rmse":
                obj_val = _rmse(y, pred)
                tie_val = _mape(y, pred, eps=float(eps))
            else:
                obj_val = _mape(y, pred, eps=float(eps))
                tie_val = _rmse(y, pred)
            if (obj_val + 1e-12) < best_obj or (abs(obj_val - best_obj) <= 1e-12 and tie_val < best_tie):
                best_obj = float(obj_val)
                best_tie = float(tie_val)
                best_w = (float(w_qk), float(w_aux))
    return best_w


def _baseline_from_x(x_seq: np.ndarray, horizon: int, d_out: int) -> np.ndarray:
    x = np.asarray(x_seq, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x_seq must be [N,L,D], got {x.shape}.")
    if x.shape[2] < int(d_out):
        raise ValueError(f"x_seq feature dim {x.shape[2]} < d_out {d_out}.")
    last = x[:, -1, : int(d_out)]
    return np.repeat(last[:, None, :], int(horizon), axis=1)


def _parse_ridge_grid(raw: list[float] | tuple[float, ...] | str | None) -> list[float]:
    if raw is None:
        return [10.0**p for p in range(-6, 3)]
    if isinstance(raw, str):
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        vals = [float(t) for t in toks]
    else:
        vals = [float(v) for v in raw]
    out = sorted({float(max(1e-12, v)) for v in vals})
    if not out:
        out = [10.0**p for p in range(-6, 3)]
    return out


def _parse_gamma_grid(
    raw: list[float] | tuple[float, ...] | str | None,
    *,
    base_gamma: float,
    s_train_norm: np.ndarray,
) -> list[float]:
    if raw is not None:
        if isinstance(raw, str):
            toks = [t.strip() for t in raw.split(",") if t.strip()]
            vals = [float(t) for t in toks]
        else:
            vals = [float(v) for v in raw]
        out = sorted({float(np.clip(v, 1e-12, 1e3)) for v in vals if np.isfinite(v) and float(v) > 0.0})
        if out:
            return out

    x = np.asarray(s_train_norm, dtype=np.float64)
    n = int(x.shape[0])
    if n <= 1:
        return [float(np.clip(base_gamma, 1e-12, 1e3))]

    left_sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(left_sq + left_sq.T - 2.0 * (x @ x.T), 0.0)
    off = dist2[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off) & (off > 1e-12)]
    if off.size == 0:
        return [float(np.clip(base_gamma, 1e-12, 1e3))]
    med = float(np.median(off))
    if not np.isfinite(med) or med <= 1e-12:
        return [float(np.clip(base_gamma, 1e-12, 1e3))]

    base = [0.1 / med, 0.3 / med, 1.0 / med, 3.0 / med, 10.0 / med, float(base_gamma)]
    out = sorted({float(np.clip(v, 1e-12, 1e3)) for v in base if np.isfinite(v) and float(v) > 0.0})
    return out or [float(np.clip(base_gamma, 1e-12, 1e3))]


def _standardize_state_space(
    s_train: np.ndarray,
    s_other: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    tr = np.nan_to_num(np.asarray(s_train, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ot = np.nan_to_num(np.asarray(s_other, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.mean(tr, axis=0, keepdims=True)
    sigma = np.std(tr, axis=0, keepdims=True)
    sigma = np.where(sigma > float(eps), sigma, 1.0)
    tr_z = (tr - mu) / sigma
    ot_z = (ot - mu) / sigma
    stats = {
        "state_center_l2": float(np.linalg.norm(mu)),
        "state_scale_mean": float(np.mean(sigma)),
        "state_scale_min": float(np.min(sigma)),
        "state_scale_max": float(np.max(sigma)),
    }
    return tr_z, ot_z, stats


@dataclass
class QuantumKernelPhotMemParams:
    gamma: float = 0.3
    gamma_grid: list[float] | tuple[float, ...] | str | None = None
    ridge: float = 1e-2
    ridge_grid: list[float] | tuple[float, ...] | str | None = None
    feature: str = "clickprob"
    shots: int = 32
    gain: float = 0.5
    modes: int = 8
    in_scale: float = 0.5
    seed: int = 0
    state_dim_mode: str = "mean_last_std"
    horizon_mode: str = "single14"
    target_space: str = "factor"
    residual_clip: float | None = None
    aux_blend: bool = True
    aux_kind: str = "photonic_qrc_feedback"
    blend_objective: str = "mape"
    blend_max_weight: float = 1.0
    blend_grid_points: int = 31

    def normalized(self) -> "QuantumKernelPhotMemParams":
        hz = str(self.horizon_mode).strip().lower()
        if hz not in {"single14", "all"}:
            raise ValueError("qk_horizon_mode must be one of: single14, all.")
        tgt = str(self.target_space).strip().lower()
        if tgt not in {"factor", "surface"}:
            raise ValueError("qk_target_space must be one of: factor, surface.")
        aux_kind = str(self.aux_kind).strip().lower()
        if aux_kind not in {"photonic_qrc_feedback", "reservoir", "none"}:
            raise ValueError("qk_aux_kind must be one of: photonic_qrc_feedback, reservoir, none.")
        blend_obj = str(self.blend_objective).strip().lower()
        if blend_obj not in {"rmse", "mape"}:
            raise ValueError("qk_blend_objective must be one of: rmse, mape.")
        return QuantumKernelPhotMemParams(
            gamma=float(max(1e-12, self.gamma)),
            gamma_grid=self.gamma_grid,
            ridge=float(max(1e-12, self.ridge)),
            ridge_grid=self.ridge_grid,
            feature=str(self.feature).strip().lower(),
            shots=int(max(0, self.shots)),
            gain=float(max(0.0, self.gain)),
            modes=int(max(2, self.modes)),
            in_scale=float(max(1e-8, self.in_scale)),
            seed=int(self.seed),
            state_dim_mode=str(self.state_dim_mode).strip().lower(),
            horizon_mode=hz,
            target_space=tgt,
            residual_clip=(None if self.residual_clip is None else float(abs(self.residual_clip))),
            aux_blend=bool(self.aux_blend),
            aux_kind=aux_kind,
            blend_objective=blend_obj,
            blend_max_weight=float(np.clip(float(self.blend_max_weight), 0.1, 3.0)),
            blend_grid_points=int(np.clip(int(self.blend_grid_points), 3, 81)),
        )


@dataclass
class QuantumKernelPhotMemPredictor:
    params: QuantumKernelPhotMemParams
    state_params: PhotonicMemoryStateParams
    horizon: int
    d_out: int
    train_states: np.ndarray
    train_states_norm: np.ndarray
    kernel_train: np.ndarray
    alphas_by_h: list[np.ndarray | None]
    ridge_by_h: dict[int, float]
    val_rmse_by_h: dict[int, float]
    train_rmse_by_h: dict[int, float]
    residual_scale_by_h: list[float]
    blend_weights_by_h: list[tuple[float, float]]
    aux_residual_model: Any | None
    selected_horizons: list[int]
    feature_stats: dict[str, Any]
    feedback_sanity: dict[str, Any]
    shots_per_eval: int
    train_qevals: int
    train_total_shots: int
    infer_qevals: int
    infer_total_shots: int
    qrc_mode: str
    qrc_target: str
    qrc_baseline: str
    qk_target_space: str
    qk_horizon_mode: str
    objective_train_rmse: float
    objective_val_rmse: float
    state_hash: str
    kernel_hash: str
    n_train_kernel: int
    kernel_train_rows: int
    kernel_train_cols: int
    last_q_features_: np.ndarray | None = None
    kernel_gram_: np.ndarray | None = None
    last_kernel_cross_: np.ndarray | None = None

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float64)
        states, meta = build_photonic_memory_state_with_meta(x, self.state_params)
        self.last_q_features_ = np.asarray(states, dtype=np.float64)
        self.infer_qevals += int(meta.get("qevals", 0))
        self.infer_total_shots += int(meta.get("total_shots", 0))

        tr_mu = np.mean(np.asarray(self.train_states, dtype=np.float64), axis=0, keepdims=True)
        tr_sigma = np.std(np.asarray(self.train_states, dtype=np.float64), axis=0, keepdims=True)
        tr_sigma = np.where(tr_sigma > 1e-8, tr_sigma, 1.0)
        states_z = (np.asarray(states, dtype=np.float64) - tr_mu) / tr_sigma
        states_norm = _normalize_rows(states_z)
        k_cross = _rbf_kernel(states_norm, self.train_states_norm, gamma=float(self.params.gamma))
        self.last_kernel_cross_ = np.asarray(k_cross, dtype=np.float64)

        base = _baseline_from_x(x, horizon=int(self.horizon), d_out=int(self.d_out))
        residual_hat = np.zeros_like(base, dtype=np.float64)
        aux_resid = None
        if self.aux_residual_model is not None:
            try:
                aux_level = np.asarray(self.aux_residual_model.predict(x), dtype=np.float64)
                aux_level = np.nan_to_num(aux_level, nan=0.0, posinf=0.0, neginf=0.0)
                aux_resid = aux_level[:, : int(self.horizon), : int(self.d_out)] - base
            except Exception:
                aux_resid = None
        for h_idx, alpha in enumerate(self.alphas_by_h):
            r_h = np.zeros((x.shape[0], int(self.d_out)), dtype=np.float64)
            if alpha is not None:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    r_h = np.asarray(k_cross @ alpha, dtype=np.float64)
                r_h = np.nan_to_num(r_h, nan=0.0, posinf=0.0, neginf=0.0)
            scale_h = float(self.residual_scale_by_h[h_idx]) if h_idx < len(self.residual_scale_by_h) else 1.0
            r_h = float(scale_h) * r_h
            w_qk, w_aux = (1.0, 0.0)
            if h_idx < len(self.blend_weights_by_h):
                w_qk, w_aux = self.blend_weights_by_h[h_idx]
            if aux_resid is not None:
                r_h = float(w_qk) * r_h + float(w_aux) * np.asarray(aux_resid[:, h_idx, :], dtype=np.float64)
            else:
                r_h = float(w_qk) * r_h
            if self.params.residual_clip is not None:
                r_h = np.clip(r_h, -float(self.params.residual_clip), float(self.params.residual_clip))
            residual_hat[:, h_idx, :] = r_h

        y_hat = base + residual_hat
        return np.asarray(y_hat, dtype=float)


def train_quantum_kernel_photmem(
    X_seq_train: np.ndarray,
    Y_train: np.ndarray,
    X_seq_val: np.ndarray,
    Y_val: np.ndarray,
    params: QuantumKernelPhotMemParams,
    horizon: int,
    d_out: int,
) -> QuantumKernelPhotMemPredictor:
    p = params.normalized()

    x_train = np.asarray(X_seq_train, dtype=np.float64)
    y_train = np.asarray(Y_train, dtype=np.float64)
    x_val = np.asarray(X_seq_val, dtype=np.float64)
    y_val = np.asarray(Y_val, dtype=np.float64)

    if x_train.ndim != 3 or y_train.ndim != 3 or x_val.ndim != 3 or y_val.ndim != 3:
        raise ValueError("Expected rank-3 tensors: X_seq_train/Y_train/X_seq_val/Y_val.")
    if x_train.shape[0] < 2 or x_val.shape[0] < 2:
        raise ValueError("Need at least 2 train and 2 validation samples for full-kernel training.")

    h_total = int(min(int(horizon), y_train.shape[1], y_val.shape[1]))
    d_eff = int(min(int(d_out), y_train.shape[2], y_val.shape[2]))
    if h_total <= 0 or d_eff <= 0:
        raise ValueError(f"Invalid horizon/d_out: horizon={h_total}, d_out={d_eff}.")

    if p.target_space == "surface":
        warnings.warn(
            "qk_target_space=surface requested but factor-space dispatcher provided factor tensors; "
            "falling back to factor residual targets.",
            RuntimeWarning,
        )

    base_train = _baseline_from_x(x_train, horizon=h_total, d_out=d_eff)
    base_val = _baseline_from_x(x_val, horizon=h_total, d_out=d_eff)
    resid_train = y_train[:, :h_total, :d_eff] - base_train
    resid_val = y_val[:, :h_total, :d_eff] - base_val

    if p.horizon_mode == "single14":
        target_h = int(min(13, h_total - 1))
        selected_horizons = [target_h]
    else:
        selected_horizons = list(range(h_total))

    state_params = PhotonicMemoryStateParams(
        modes=int(p.modes),
        in_scale=float(p.in_scale),
        gain=float(p.gain),
        feature=str(p.feature),
        shots=int(p.shots),
        state_dim_mode=str(p.state_dim_mode),
        seed=int(p.seed),
    ).normalized()

    s_train, s_meta_train = build_photonic_memory_state_with_meta(x_train, state_params)
    s_val, s_meta_val = build_photonic_memory_state_with_meta(x_val, state_params)

    fb_sanity = feedback_sanity_check(
        x_train[: min(16, x_train.shape[0])],
        state_params,
        mini_batch=min(8, x_train.shape[0]),
        delta_threshold=1e-8,
    )

    s_train_z, s_val_z, s_norm_stats = _standardize_state_space(s_train, s_val)
    s_train_norm = _normalize_rows(s_train_z)
    s_val_norm = _normalize_rows(s_val_z)

    ridge_grid = _parse_ridge_grid(p.ridge_grid)
    if float(p.ridge) not in ridge_grid:
        ridge_grid.append(float(p.ridge))
        ridge_grid = sorted({float(max(1e-12, v)) for v in ridge_grid})
    gamma_grid = _parse_gamma_grid(p.gamma_grid, base_gamma=float(p.gamma), s_train_norm=s_train_norm)

    best_pack: dict[str, Any] | None = None
    for gamma in gamma_grid:
        k_train = _rbf_kernel(s_train_norm, s_train_norm, gamma=float(gamma))
        k_val = _rbf_kernel(s_val_norm, s_train_norm, gamma=float(gamma))

        diag = np.diag(k_train)
        if k_train.shape[0] > 1:
            off = k_train[~np.eye(k_train.shape[0], dtype=bool)]
        else:
            off = np.array([0.0], dtype=np.float64)
        off_std = float(np.std(off))
        if off_std <= 1e-10:
            continue

        alphas_by_h: list[np.ndarray | None] = [None for _ in range(h_total)]
        ridge_by_h: dict[int, float] = {}
        val_rmse_by_h: dict[int, float] = {}
        train_rmse_by_h: dict[int, float] = {}
        residual_scale_by_h: list[float] = [0.0 for _ in range(h_total)]
        valid = True

        for h_idx in selected_horizons:
            ytr_h = np.asarray(resid_train[:, h_idx, :], dtype=np.float64)
            yva_h = np.asarray(resid_val[:, h_idx, :], dtype=np.float64)

            best_rmse = float("inf")
            best_alpha: np.ndarray | None = None
            best_lam = float(ridge_grid[0])
            best_train_rmse = float("inf")
            best_scale = 1.0
            for lam in ridge_grid:
                alpha = _solve_kernel_ridge(k_train, ytr_h, ridge=float(lam))
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pred_val_h = np.asarray(k_val @ alpha, dtype=np.float64)
                if not np.all(np.isfinite(pred_val_h)):
                    continue
                scale_h = _optimal_residual_scale(yva_h, pred_val_h, min_scale=0.0, max_scale=1.5)
                pred_val_h_scaled = float(scale_h) * pred_val_h
                rmse_val_h = _rmse(yva_h, pred_val_h_scaled)
                if not np.isfinite(rmse_val_h):
                    continue
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pred_tr_h = np.asarray(k_train @ alpha, dtype=np.float64)
                if not np.all(np.isfinite(pred_tr_h)):
                    continue
                rmse_tr_h = _rmse(ytr_h, float(scale_h) * pred_tr_h)
                if not np.isfinite(rmse_tr_h):
                    continue
                if rmse_val_h < best_rmse:
                    best_rmse = float(rmse_val_h)
                    best_alpha = alpha
                    best_lam = float(lam)
                    best_train_rmse = float(rmse_tr_h)
                    best_scale = float(scale_h)

            if best_alpha is None:
                valid = False
                break

            alphas_by_h[h_idx] = np.asarray(best_alpha, dtype=np.float64)
            ridge_by_h[int(h_idx)] = float(best_lam)
            val_rmse_by_h[int(h_idx)] = float(best_rmse)
            train_rmse_by_h[int(h_idx)] = float(best_train_rmse)
            residual_scale_by_h[h_idx] = float(best_scale)

        if not valid:
            continue

        train_rmse_vals = [float(train_rmse_by_h[h]) for h in selected_horizons]
        val_rmse_vals = [float(val_rmse_by_h[h]) for h in selected_horizons]
        objective_train_rmse = float(np.mean(train_rmse_vals)) if train_rmse_vals else float("inf")
        objective_val_rmse = float(np.mean(val_rmse_vals)) if val_rmse_vals else float("inf")

        if not np.isfinite(objective_val_rmse):
            continue

        pack = {
            "gamma": float(gamma),
            "k_train": np.asarray(k_train, dtype=np.float64),
            "k_val": np.asarray(k_val, dtype=np.float64),
            "diag": np.asarray(diag, dtype=np.float64),
            "off": np.asarray(off, dtype=np.float64),
            "off_std": float(off_std),
            "alphas_by_h": alphas_by_h,
            "ridge_by_h": ridge_by_h,
            "val_rmse_by_h": val_rmse_by_h,
            "train_rmse_by_h": train_rmse_by_h,
            "residual_scale_by_h": residual_scale_by_h,
            "objective_train_rmse": float(objective_train_rmse),
            "objective_val_rmse": float(objective_val_rmse),
        }
        if best_pack is None or float(pack["objective_val_rmse"]) < float(best_pack["objective_val_rmse"]):
            best_pack = pack

    if best_pack is None:
        raise RuntimeError(
            "Failed to fit quantum kernel photonic memory model: no finite gamma/ridge candidate found. "
            f"gamma_grid={gamma_grid} ridge_grid={ridge_grid}"
        )

    k_train = np.asarray(best_pack["k_train"], dtype=np.float64)
    k_val = np.asarray(best_pack["k_val"], dtype=np.float64)
    diag = np.asarray(best_pack["diag"], dtype=np.float64)
    off = np.asarray(best_pack["off"], dtype=np.float64)
    off_std = float(best_pack["off_std"])
    alphas_by_h = list(best_pack["alphas_by_h"])
    ridge_by_h = dict(best_pack["ridge_by_h"])
    val_rmse_by_h = dict(best_pack["val_rmse_by_h"])
    train_rmse_by_h = dict(best_pack["train_rmse_by_h"])
    residual_scale_by_h = list(best_pack["residual_scale_by_h"])
    objective_train_rmse = float(best_pack["objective_train_rmse"])
    objective_val_rmse = float(best_pack["objective_val_rmse"])
    gamma_selected = float(best_pack["gamma"])

    qk_train_resid = np.zeros((x_train.shape[0], h_total, d_eff), dtype=np.float64)
    qk_val_resid = np.zeros((x_val.shape[0], h_total, d_eff), dtype=np.float64)
    for h_idx, alpha in enumerate(alphas_by_h):
        if alpha is None:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred_tr_h = np.asarray(k_train @ alpha, dtype=np.float64)
            pred_va_h = np.asarray(k_val @ alpha, dtype=np.float64)
        if np.all(np.isfinite(pred_tr_h)):
            qk_train_resid[:, h_idx, :] = float(residual_scale_by_h[h_idx]) * pred_tr_h
        if np.all(np.isfinite(pred_va_h)):
            qk_val_resid[:, h_idx, :] = float(residual_scale_by_h[h_idx]) * pred_va_h

    aux_model: Any | None = None
    aux_model_kind = "none"
    aux_train_resid = np.zeros_like(qk_train_resid, dtype=np.float64)
    aux_val_resid = np.zeros_like(qk_val_resid, dtype=np.float64)
    if bool(p.aux_blend) and str(p.aux_kind).strip().lower() != "none":
        requested_aux = str(p.aux_kind).strip().lower()
        try:
            if requested_aux in {"photonic_qrc_feedback", "persist_qrc_weak"}:
                from ..models_classical import train_classical_forecaster  # type: ignore

                aux_model = train_classical_forecaster(
                    x_train=x_train,
                    y_train=y_train[:, :h_total, :d_eff],
                    x_val=x_val,
                    y_val=y_val[:, :h_total, :d_eff],
                    kind=str(requested_aux),
                    seed=int(p.seed),
                    pqrc_M=int(max(8, p.modes)),
                    pqrc_modes=int(max(8, p.modes)),
                    pqrc_Nph=2,
                    pqrc_nphotons=2,
                    pqrc_budget=int(max(32, p.shots)),
                    pqrc_gain=float(max(0.15, p.gain)),
                    pqrc_feature=str("clickprob" if p.feature not in {"clickprob", "coincidence"} else p.feature),
                    pqrc_shots=int(max(1, p.shots)),
                    pqrc_ridge=float(max(1e-4, p.ridge)),
                    pqrc_input_scale=float(max(0.5, p.in_scale)),
                    qrc_mode="residual",
                    qrc_target="delta",
                    qrc_baseline="persistence",
                    target_transform="log",
                    y_floor_mode="train_p001",
                    qrc_gate_tau=0.06184302083987248,
                    qrc_resid_clip=(None if p.residual_clip is None else float(p.residual_clip)),
                    qrc_residvar_penalty=0.1,
                    qrc_feat_norm="standard",
                )
                aux_model_kind = str(requested_aux)
            else:
                from ..models_classical import _fit_reservoir  # type: ignore

                aux_model = _fit_reservoir(
                    x_train=x_train,
                    y_train=y_train[:, :h_total, :d_eff],
                    x_val=x_val,
                    y_val=y_val[:, :h_total, :d_eff],
                    seed=int(p.seed),
                )
                aux_model_kind = "reservoir"

            aux_train_level = np.asarray(aux_model.predict(x_train), dtype=np.float64)
            aux_val_level = np.asarray(aux_model.predict(x_val), dtype=np.float64)
            aux_train_level = np.nan_to_num(aux_train_level, nan=0.0, posinf=0.0, neginf=0.0)
            aux_val_level = np.nan_to_num(aux_val_level, nan=0.0, posinf=0.0, neginf=0.0)
            aux_train_resid = aux_train_level[:, :h_total, :d_eff] - base_train
            aux_val_resid = aux_val_level[:, :h_total, :d_eff] - base_val
        except Exception as exc:
            if requested_aux in {"photonic_qrc_feedback", "persist_qrc_weak"}:
                try:
                    from ..models_classical import _fit_reservoir  # type: ignore

                    aux_model = _fit_reservoir(
                        x_train=x_train,
                        y_train=y_train[:, :h_total, :d_eff],
                        x_val=x_val,
                        y_val=y_val[:, :h_total, :d_eff],
                        seed=int(p.seed),
                    )
                    aux_model_kind = "reservoir"
                    aux_train_level = np.asarray(aux_model.predict(x_train), dtype=np.float64)
                    aux_val_level = np.asarray(aux_model.predict(x_val), dtype=np.float64)
                    aux_train_level = np.nan_to_num(aux_train_level, nan=0.0, posinf=0.0, neginf=0.0)
                    aux_val_level = np.nan_to_num(aux_val_level, nan=0.0, posinf=0.0, neginf=0.0)
                    aux_train_resid = aux_train_level[:, :h_total, :d_eff] - base_train
                    aux_val_resid = aux_val_level[:, :h_total, :d_eff] - base_val
                except Exception as exc2:
                    warnings.warn(
                        f"auxiliary blend disabled: failed {requested_aux} ({exc}) and reservoir fallback ({exc2})",
                        RuntimeWarning,
                    )
                    aux_model = None
                    aux_model_kind = "none"
                    aux_train_resid = np.zeros_like(qk_train_resid, dtype=np.float64)
                    aux_val_resid = np.zeros_like(qk_val_resid, dtype=np.float64)
            else:
                warnings.warn(
                    f"auxiliary blend disabled due to failure: {exc}",
                    RuntimeWarning,
                )
                aux_model = None
                aux_model_kind = "none"
                aux_train_resid = np.zeros_like(qk_train_resid, dtype=np.float64)
                aux_val_resid = np.zeros_like(qk_val_resid, dtype=np.float64)

    blend_weights_by_h: list[tuple[float, float]] = [(1.0, 0.0) for _ in range(h_total)]
    blend_obj = str(p.blend_objective).strip().lower()
    blend_obj = "mape" if blend_obj not in {"rmse", "mape"} else blend_obj
    for h_idx in range(h_total):
        ytr_h = np.asarray(resid_train[:, h_idx, :], dtype=np.float64)
        yva_h = np.asarray(resid_val[:, h_idx, :], dtype=np.float64)
        ytr_level_h = np.asarray(y_train[:, h_idx, :], dtype=np.float64)
        yva_level_h = np.asarray(y_val[:, h_idx, :], dtype=np.float64)
        base_tr_h = np.asarray(base_train[:, h_idx, :], dtype=np.float64)
        base_va_h = np.asarray(base_val[:, h_idx, :], dtype=np.float64)
        qtr_h = np.asarray(qk_train_resid[:, h_idx, :], dtype=np.float64)
        qva_h = np.asarray(qk_val_resid[:, h_idx, :], dtype=np.float64)
        atr_h = np.asarray(aux_train_resid[:, h_idx, :], dtype=np.float64)
        ava_h = np.asarray(aux_val_resid[:, h_idx, :], dtype=np.float64)

        candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = [
            (0.0, 0.0, np.zeros_like(ytr_h), np.zeros_like(yva_h)),
            (1.0, 0.0, qtr_h, qva_h),
        ]
        if aux_model is not None:
            candidates.append((0.0, 1.0, atr_h, ava_h))
            w_qk, w_aux = _fit_two_source_blend(yva_h, qva_h, ava_h, max_weight=float(p.blend_max_weight))
            candidates.append((w_qk, w_aux, w_qk * qtr_h + w_aux * atr_h, w_qk * qva_h + w_aux * ava_h))
            w_qk_grid, w_aux_grid = _fit_two_source_blend_grid(
                y_true_level=yva_level_h,
                base_level=base_va_h,
                src_qk_resid=qva_h,
                src_aux_resid=ava_h,
                max_weight=float(p.blend_max_weight),
                n_grid=int(p.blend_grid_points),
                objective=str(blend_obj),
                eps=1e-8,
            )
            candidates.append(
                (
                    w_qk_grid,
                    w_aux_grid,
                    w_qk_grid * qtr_h + w_aux_grid * atr_h,
                    w_qk_grid * qva_h + w_aux_grid * ava_h,
                )
            )

        best_val = float("inf")
        best_train = float("inf")
        best_obj = float("inf")
        best_tie = float("inf")
        best_w = (0.0, 0.0)
        for w_qk, w_aux, pred_tr, pred_va in candidates:
            pred_tr_level = base_tr_h + np.asarray(pred_tr, dtype=np.float64)
            pred_va_level = base_va_h + np.asarray(pred_va, dtype=np.float64)
            rmse_va = _rmse(yva_h, pred_va)
            mape_va = _mape(yva_level_h, pred_va_level, eps=1e-8)
            if not (np.isfinite(rmse_va) and np.isfinite(mape_va)):
                continue
            rmse_tr = _rmse(ytr_h, pred_tr)
            if not np.isfinite(rmse_tr):
                continue
            if blend_obj == "rmse":
                obj_val = float(rmse_va)
                tie_val = float(mape_va)
            else:
                obj_val = float(mape_va)
                tie_val = float(rmse_va)
            if (obj_val + 1e-12) < best_obj or (abs(obj_val - best_obj) <= 1e-12 and tie_val < best_tie):
                best_obj = float(obj_val)
                best_tie = float(tie_val)
                best_val = float(rmse_va)
                best_train = float(rmse_tr)
                best_w = (float(w_qk), float(w_aux))

        blend_weights_by_h[h_idx] = best_w
        if h_idx in selected_horizons:
            val_rmse_by_h[int(h_idx)] = float(best_val)
            train_rmse_by_h[int(h_idx)] = float(best_train)

    train_rmse_vals = [float(train_rmse_by_h[h]) for h in selected_horizons]
    val_rmse_vals = [float(val_rmse_by_h[h]) for h in selected_horizons]
    objective_train_rmse = float(np.mean(train_rmse_vals)) if train_rmse_vals else float("nan")
    objective_val_rmse = float(np.mean(val_rmse_vals)) if val_rmse_vals else float("nan")

    feat_std_train = np.std(np.asarray(s_train, dtype=np.float64), axis=0)
    feat_std_val = np.std(np.asarray(s_val, dtype=np.float64), axis=0)
    feature_stats = {
        "feat_std_train": [float(v) for v in np.asarray(feat_std_train, dtype=float)],
        "feat_std_val": [float(v) for v in np.asarray(feat_std_val, dtype=float)],
        "state_train": state_feature_statistics(s_train),
        "state_val": state_feature_statistics(s_val),
        "state_dim": int(s_train.shape[1]),
        "feature_dim": int(s_meta_train.get("feature_dim", s_train.shape[1])),
        "wedge_indices": [int(v) for v in s_meta_train.get("wedge_indices", [])],
        "kernel_diag_mean": float(np.mean(diag)),
        "kernel_diag_std": float(np.std(diag)),
        "kernel_offdiag_mean": float(np.mean(off)),
        "kernel_offdiag_std": float(off_std),
        "kernel_offdiag_nondegenerate": bool(off_std > 1e-10),
        "kernel_train_shape": [int(k_train.shape[0]), int(k_train.shape[1])],
        "gamma_grid": [float(v) for v in gamma_grid],
        "gamma_selected": float(gamma_selected),
        "residual_scale_by_h": [float(v) for v in residual_scale_by_h],
        "blend_weights_by_h": [[float(a), float(b)] for (a, b) in blend_weights_by_h],
        "aux_blend_enabled": bool(aux_model is not None),
        "aux_model_kind": str(aux_model_kind),
        "blend_objective": str(blend_obj),
        "blend_max_weight": float(p.blend_max_weight),
        "blend_grid_points": int(p.blend_grid_points),
        "state_standardization": {k: float(v) for k, v in s_norm_stats.items()},
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "qevals_train": int(s_meta_train.get("qevals", 0)),
        "qevals_val": int(s_meta_val.get("qevals", 0)),
        "total_shots_train": int(s_meta_train.get("total_shots", 0)),
        "total_shots_val": int(s_meta_val.get("total_shots", 0)),
    }

    train_qevals = int(s_meta_train.get("qevals", 0)) + int(s_meta_val.get("qevals", 0))
    train_total_shots = int(s_meta_train.get("total_shots", 0)) + int(s_meta_val.get("total_shots", 0))
    p_selected = replace(
        p,
        gamma=float(gamma_selected),
        gamma_grid=[float(v) for v in gamma_grid],
        ridge_grid=[float(v) for v in ridge_grid],
    )

    predictor = QuantumKernelPhotMemPredictor(
        params=p_selected,
        state_params=state_params,
        horizon=h_total,
        d_out=d_eff,
        train_states=np.asarray(s_train, dtype=np.float64),
        train_states_norm=np.asarray(s_train_norm, dtype=np.float64),
        kernel_train=np.asarray(k_train, dtype=np.float64),
        alphas_by_h=alphas_by_h,
        ridge_by_h=ridge_by_h,
        val_rmse_by_h=val_rmse_by_h,
        train_rmse_by_h=train_rmse_by_h,
        residual_scale_by_h=[float(v) for v in residual_scale_by_h],
        blend_weights_by_h=[(float(a), float(b)) for (a, b) in blend_weights_by_h],
        aux_residual_model=aux_model,
        selected_horizons=[int(h) for h in selected_horizons],
        feature_stats=feature_stats,
        feedback_sanity=fb_sanity,
        shots_per_eval=int(state_params.shots),
        train_qevals=int(train_qevals),
        train_total_shots=int(train_total_shots),
        infer_qevals=0,
        infer_total_shots=0,
        qrc_mode="residual",
        qrc_target="delta",
        qrc_baseline="persistence",
        qk_target_space=str(p.target_space),
        qk_horizon_mode=str(p.horizon_mode),
        objective_train_rmse=float(objective_train_rmse),
        objective_val_rmse=float(objective_val_rmse),
        state_hash=_hash_array(s_train),
        kernel_hash=_hash_array(k_train),
        n_train_kernel=int(x_train.shape[0]),
        kernel_train_rows=int(k_train.shape[0]),
        kernel_train_cols=int(k_train.shape[1]),
        last_q_features_=np.asarray(s_val, dtype=np.float64),
        kernel_gram_=np.asarray(k_train, dtype=np.float64),
        last_kernel_cross_=np.asarray(k_val, dtype=np.float64),
    )
    return predictor


__all__ = [
    "QuantumKernelPhotMemParams",
    "QuantumKernelPhotMemPredictor",
    "train_quantum_kernel_photmem",
]
