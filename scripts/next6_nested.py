from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cv import PseudoNext6CV, validate_fold_order
from .data_swaptions import prepare_windowed_dataset
from .ensemble import EnsembleForecaster, QuantumResidualCorrector
from .factor_pipeline import FactorPipeline, FactorPipelineConfig
from .models_classical import train_classical_forecaster
from .pipeline import _level_cfg, _load_submission_config, sanitize_surface_matrix
from swaptions.hybrids.gru_residual_quantum import ResidualCorrectedForecaster


ACTIVE_MODELS: list[str] = [
    "persistence_surface_naive",
    "persistence_naive",
    "factor_ar",
    "mlp",
    "gru",
    "lstm",
    "reservoir",
    "hybrid_qrc_fast",
    "qml_classical_reservoir",
    "qml_recursive_classical_reservoir",
    "qml_pca_lstm",
    "qml_convlstm",
    "photonic_qrc_feedback",
    "photonic_qrc_no_feedback",
    "persist_qrc_weak",
    "photonic_memory",
    "photonic_memory_no_feedback",
    "quantum_kernel_photonic_memory",
    "gru_photonic_memory_fb",
    "gru_quantum_kernel",
    "gru_quantum_kernel_103754",
    "gru_photonic_qrc_fb",
    "ensemble_stack_safe",
]

QUANTUM_MODELS: set[str] = {
    "quantum_kernel_photonic_memory",
    "photonic_qrc_feedback",
    "photonic_qrc_no_feedback",
    "persist_qrc_weak",
    "photonic_memory",
    "photonic_memory_no_feedback",
}

HYBRID_GRU_QUANTUM_MODELS: set[str] = {
    "gru_photonic_memory_fb",
    "gru_quantum_kernel",
    "gru_quantum_kernel_103754",
    "gru_photonic_qrc_fb",
}

SLOW_MODELS: set[str] = {
    "photonic_qrc_feedback",
    "photonic_qrc_no_feedback",
    "persist_qrc_weak",
    "photonic_memory",
    "photonic_memory_no_feedback",
    "quantum_kernel_photonic_memory",
    "gru_photonic_memory_fb",
    "gru_quantum_kernel",
    "gru_quantum_kernel_103754",
    "gru_photonic_qrc_fb",
}


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def parse_csv_floats(raw: str | None, default: list[float]) -> list[float]:
    vals = []
    for t in parse_csv_list(raw):
        vals.append(float(t))
    if not vals:
        vals = list(default)
    return vals


def parse_csv_ints(raw: str | None, default: list[int]) -> list[int]:
    vals = []
    for t in parse_csv_list(raw):
        vals.append(int(round(float(t))))
    if not vals:
        vals = list(default)
    return vals


def _coerce_float_list(value: Any, default: list[float]) -> list[float]:
    if isinstance(value, str):
        out = parse_csv_floats(value, default=default)
    elif isinstance(value, (list, tuple, np.ndarray)):
        out = [float(v) for v in np.asarray(value, dtype=float).reshape(-1).tolist()]
    elif value is None:
        out = list(default)
    else:
        out = [float(value)]
    return [float(v) for v in out]


def _coerce_int_list(value: Any, default: list[int]) -> list[int]:
    if isinstance(value, str):
        out = parse_csv_ints(value, default=default)
    elif isinstance(value, (list, tuple, np.ndarray)):
        out = [int(round(float(v))) for v in np.asarray(value, dtype=float).reshape(-1).tolist()]
    elif value is None:
        out = list(default)
    else:
        out = [int(round(float(value)))]
    return [int(v) for v in out]


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *[str(p) for p in parts]])
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return int(int(h, 16) % (2**31 - 1))


@dataclass
class NestedConfig:
    data_dir: str
    level: int = 1
    lookback: int = 20
    horizon: int = 6
    n_anchors: int = 10
    inner_anchors: int = 4
    nested_tune: bool = True
    lambda_std: float = 0.25
    h_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
    factor_mode: bool = True
    pca_factors: int = 8
    tune_budget: int = 24
    tune_mode: str = "grid"
    seed: int = 42
    cv_anchor_strategy: str = "tail"
    min_train: int = 70
    alpha_mode: str = "per_horizon"
    corr_window: int = 40
    corr_min: float = 0.0
    sign_min: float = 0.52
    force_zero_h_if_bad: bool = True
    force_zero_h_list: list[int] = field(default_factory=lambda: [5, 6])
    apply_correction_hmin: int = 3
    cal_anchors: int = 3
    cal_stride: int = 1
    resid_sign_grid: list[int] = field(default_factory=lambda: [1, -1])
    resid_h_weights: list[float] = field(default_factory=lambda: [0.25, 0.25, 1.0, 1.0, 1.5, 1.5])
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.3
    teacher_forcing_schedule: str = "linear"
    ss_passes: int = 4
    skip_slow: bool = False
    gain_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    gate_tau_grid: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0])
    q_modes_grid: list[int] = field(default_factory=lambda: [8, 16, 24])
    q_shots_grid: list[int] = field(default_factory=lambda: [32, 64, 128])
    q_in_scale_grid: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    q_ridge_grid: list[float] = field(default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    q_feature_grid: list[str] = field(default_factory=lambda: ["clickprob", "coincidence"])
    qk_gamma_grid: list[float] = field(default_factory=lambda: [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    qk_factor1_scale_grid: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    gru_hidden_grid: list[int] = field(default_factory=lambda: [48, 64, 96])
    gru_lr_grid: list[float] = field(default_factory=lambda: [3e-4, 1e-3])
    gru_dropout_grid: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    gru_weight_decay_grid: list[float] = field(default_factory=lambda: [1e-6, 1e-5, 1e-4])
    gru_epochs_grid: list[int] = field(default_factory=lambda: [24, 36])
    quantum_base_kind: str = "lstm"
    ensemble: bool = False
    ensemble_members: list[str] = field(default_factory=lambda: ["lstm", "factor_ar", "qml_pca_lstm"])
    ensemble_quantum_kind: str = "gru_quantum_kernel"
    ensemble_quantum_min_improve: float = 0.01
    exclude_ported: bool = True
    imputer: str = "ffill_interp"
    use_cycle_phase: bool = True

    def normalized_h_weights(self) -> np.ndarray:
        w = np.asarray(self.h_weights, dtype=float).reshape(-1)
        if w.size < self.horizon:
            w = np.pad(w, (0, self.horizon - w.size), mode="edge")
        if w.size > self.horizon:
            w = w[: self.horizon]
        w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
        return w


@dataclass
class DatasetBundle:
    ds: Any
    surfaces_input: np.ndarray
    surfaces_truth: np.ndarray
    dates: np.ndarray
    surface_cols: list[str]


@dataclass
class FoldPreparedData:
    fp: FactorPipeline
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    base_factor: np.ndarray
    y_true_surface: np.ndarray
    y_true_delta_norm: np.ndarray


def _tail_train_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    min_val: int = 8,
    frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x_train.shape[0])
    n_val = int(max(min_val, round(frac * n)))
    n_val = int(max(1, min(n_val, max(1, n // 2))))
    return np.asarray(x_train[-n_val:], dtype=float), np.asarray(y_train[-n_val:], dtype=float)


def load_dataset_bundle(cfg: NestedConfig) -> DatasetBundle:
    ds = prepare_windowed_dataset(
        data_dir=str(cfg.data_dir),
        level=int(cfg.level),
        lookback=int(cfg.lookback),
        horizon=int(cfg.horizon),
        imputer=str(cfg.imputer),
        seed=int(cfg.seed),
        use_cycle_phase=bool(cfg.use_cycle_phase),
    )
    inp, cap = sanitize_surface_matrix(np.asarray(ds.filled, dtype=float))
    tgt, _ = sanitize_surface_matrix(np.asarray(ds.target, dtype=float), cap_hint=float(cap))
    return DatasetBundle(
        ds=ds,
        surfaces_input=np.asarray(inp, dtype=float),
        surfaces_truth=np.asarray(tgt, dtype=float),
        dates=np.asarray(ds.dates, dtype=str),
        surface_cols=list(ds.surface_cols),
    )


def resolve_models(requested: list[str] | None, *, exclude_ported: bool, include_ensemble: bool) -> list[str]:
    models = [str(m).strip().lower() for m in (requested or []) if str(m).strip()]
    if not models:
        models = list(ACTIVE_MODELS)
    out: list[str] = []
    seen: set[str] = set()
    for m in models:
        if exclude_ported and m.startswith("ported_"):
            continue
        if m not in ACTIVE_MODELS and m not in {"ensemble", "ensemble_stack_safe"}:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    if include_ensemble and "ensemble_stack_safe" not in seen and "ensemble" not in seen:
        out.append("ensemble_stack_safe")
    return out


def metric_block(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    den = np.maximum(np.abs(yt), float(max(1e-12, eps)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(np.abs(err) / den) * 100.0)
    mae_h = [float(np.mean(np.abs(err[h]))) for h in range(err.shape[0])]
    rmse_h = [float(np.sqrt(np.mean(err[h] * err[h]))) for h in range(err.shape[0])]
    mape_h = [float(np.mean(np.abs(err[h]) / np.maximum(np.abs(yt[h]), float(max(1e-12, eps)))) * 100.0) for h in range(err.shape[0])]
    return {
        "surface_mae": mae,
        "surface_rmse": rmse,
        "surface_mape": mape,
        "horizon_surface_mae": mae_h,
        "horizon_surface_rmse": rmse_h,
        "horizon_surface_mape": mape_h,
    }


def weighted_rmse(horizon_rmse: list[float], h_weights: np.ndarray) -> float:
    rmse = np.asarray(horizon_rmse, dtype=float)
    w = np.asarray(h_weights, dtype=float)
    if rmse.shape[0] != w.shape[0]:
        m = int(min(rmse.shape[0], w.shape[0]))
        rmse = rmse[:m]
        w = w[:m]
    return float(np.sum(w * rmse) / max(1e-12, float(np.sum(w))))


def _prepare_fold_data(
    surfaces_input: np.ndarray,
    surfaces_truth: np.ndarray,
    train_end: int,
    cfg: NestedConfig,
    seed: int,
) -> FoldPreparedData:
    anchor = int(train_end)
    fp = FactorPipeline(
        FactorPipelineConfig(
            lookback=int(cfg.lookback),
            horizon=int(cfg.horizon),
            pca_factors=int(cfg.pca_factors),
            seed=int(seed),
        )
    ).fit(
        train_surfaces=np.asarray(surfaces_input[:anchor], dtype=float),
        full_surfaces=np.asarray(surfaces_input, dtype=float),
    )
    train_data = fp.build_train_windows(train_end=anchor)
    x_val, y_val = _tail_train_val_split(train_data.x_train, train_data.y_train, min_val=8, frac=0.2)
    test_data = fp.build_test_window(anchor=anchor)
    y_true_surface = np.asarray(surfaces_truth[anchor : anchor + int(cfg.horizon)], dtype=float)
    y_true_delta_norm = np.asarray(fp.normalize_delta(test_data.y_true_delta[None, :, :])[0], dtype=float)
    return FoldPreparedData(
        fp=fp,
        x_train=np.asarray(train_data.x_train, dtype=float),
        y_train=np.asarray(train_data.y_train, dtype=float),
        x_val=x_val,
        y_val=y_val,
        x_test=np.asarray(test_data.x_test, dtype=float),
        base_factor=np.asarray(test_data.base_factor, dtype=float),
        y_true_surface=y_true_surface,
        y_true_delta_norm=y_true_delta_norm,
    )


def _is_quantum(model: str) -> bool:
    return str(model).strip().lower() in QUANTUM_MODELS


def _strip_wrapper_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    out = dict(kwargs)
    for k in [
        "use_quantum_corrector",
        "qr_gain",
        "qr_base_kind",
        "qr_corr_window",
        "qr_corr_min",
        "qr_seed",
        "gru_hidden_size",
        "gru_lr",
        "gru_dropout",
        "gru_weight_decay",
        "gru_epochs",
        "gru_batch_size",
        "hybrid_sign_min",
        "hybrid_apply_correction_hmin",
        "hybrid_cal_anchors",
        "hybrid_cal_stride",
        "hybrid_early_min_improve",
        "hybrid_gate_tau",
        "hybrid_gain_early",
        "hybrid_gain_late",
        "hybrid_alpha_mode",
        "hybrid_force_zero_h_if_bad",
        "hybrid_force_zero_h_list",
        "resid_sign",
        "resid_h_weights",
        "teacher_forcing_start",
        "teacher_forcing_end",
        "teacher_forcing_schedule",
        "ss_passes",
        "hybrid_qk_readout_only",
        "hybrid_qk_disable_h_list",
        "qk_resid_clip_k",
        "qk_factor1_scale",
    ]:
        out.pop(k, None)
    return out


def _predict_delta_norm(
    model: str,
    candidate: dict[str, object],
    prep: FoldPreparedData,
    cfg: NestedConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    m = str(model).strip().lower()
    cand = dict(candidate)
    meta: dict[str, Any] = {}
    h = int(cfg.horizon)
    f = int(cfg.pca_factors)
    if m in {"persistence_naive", "persistence_surface_naive"}:
        return np.zeros((h, f), dtype=float), {"persistence_like": True}

    model_kwargs = _strip_wrapper_kwargs(cand)
    model_kwargs.setdefault("nn_train_loss", "rmse")
    model_kwargs.setdefault("use_full_training", True)

    if bool(int(cand.get("use_quantum_corrector", 0))) and _is_quantum(m):
        base_kind = str(cand.get("qr_base_kind", cfg.quantum_base_kind)).strip().lower()
        qr_gain = float(cand.get("qr_gain", 1.0))
        corr_window = int(cand.get("qr_corr_window", cfg.corr_window))
        corr_min = float(cand.get("qr_corr_min", cfg.corr_min))
        q_kwargs = dict(model_kwargs)
        q_seed = int(cand.get("qr_seed", seed))
        q_kwargs.setdefault("qk_target_space", "factor")
        q_kwargs.setdefault("qk_horizon_mode", "all")
        corr = QuantumResidualCorrector(
            base_kind=base_kind,
            quantum_kind=m,
            gain=float(qr_gain),
            corr_window=int(corr_window),
            corr_min=float(corr_min),
            seed=int(q_seed),
            base_kwargs={"nn_train_loss": "rmse", "use_full_training": True},
            quantum_kwargs=q_kwargs,
        ).fit(
            x_train=prep.x_train,
            y_train=prep.y_train,
            x_val=prep.x_val,
            y_val=prep.y_val,
        )
        pred = np.asarray(corr.predict(prep.x_test)[0], dtype=float)
        meta.update(
            {
                "quantum_corrector": True,
                "qr_gain": float(qr_gain),
                "qr_corr_window": int(corr_window),
                "qr_corr_min": float(corr_min),
                "qr_alpha_by_h": (
                    [float(x) for x in np.asarray(corr.alpha_by_h, dtype=float).reshape(-1).tolist()]
                    if corr.alpha_by_h is not None
                    else []
                ),
                "qr_corr_by_h": (
                    [float(x) for x in np.asarray(corr.corr_by_h, dtype=float).reshape(-1).tolist()]
                    if corr.corr_by_h is not None
                    else []
                ),
                "qr_failed": bool(corr.failed),
            }
        )
        return pred, meta

    if m in HYBRID_GRU_QUANTUM_MODELS:
        force_zero_h_list = _coerce_int_list(
            cand.get("hybrid_force_zero_h_list", cfg.force_zero_h_list),
            default=list(cfg.force_zero_h_list),
        )
        qk_readout_default = (1 if m == "gru_quantum_kernel" else 0)
        qk_disable_default = ([5, 6] if m == "gru_quantum_kernel" else [])
        qk_clip_default = (2.5 if m == "gru_quantum_kernel" else 0.0)
        qk_disable_h_list = _coerce_int_list(
            cand.get("hybrid_qk_disable_h_list", qk_disable_default),
            default=list(qk_disable_default),
        )
        hybrid = ResidualCorrectedForecaster(
            kind=str(m),
            lookback=int(cfg.lookback),
            horizon=int(cfg.horizon),
            d_factors=int(cfg.pca_factors),
            seed=int(seed),
            gru_hidden_size=int(cand.get("gru_hidden_size", 64)),
            gru_lr=float(cand.get("gru_lr", 1e-3)),
            gru_dropout=float(cand.get("gru_dropout", 0.05)),
            gru_weight_decay=float(cand.get("gru_weight_decay", 1e-5)),
            gru_epochs=int(cand.get("gru_epochs", 80)),
            gru_batch_size=int(cand.get("gru_batch_size", 32)),
            alpha_mode=str(cand.get("hybrid_alpha_mode", cfg.alpha_mode)),
            corr_window=int(cand.get("qr_corr_window", cfg.corr_window)),
            corr_min=float(cand.get("qr_corr_min", cfg.corr_min)),
            sign_min=float(cand.get("hybrid_sign_min", cfg.sign_min)),
            force_zero_h_if_bad=bool(int(cand.get("hybrid_force_zero_h_if_bad", int(cfg.force_zero_h_if_bad)))),
            force_zero_h_list=tuple(int(v) for v in force_zero_h_list),
            apply_correction_hmin=int(cand.get("hybrid_apply_correction_hmin", cfg.apply_correction_hmin)),
            cal_anchors=int(cand.get("hybrid_cal_anchors", cfg.cal_anchors)),
            cal_stride=int(cand.get("hybrid_cal_stride", cfg.cal_stride)),
            early_min_improve=float(cand.get("hybrid_early_min_improve", 0.01)),
            gate_tau=float(cand.get("hybrid_gate_tau", 1.0)),
            gain_early=float(cand.get("hybrid_gain_early", 0.35)),
            gain_late=float(cand.get("hybrid_gain_late", 0.85)),
            resid_sign=int(round(float(cand.get("resid_sign", 1)))),
            resid_h_weights=_coerce_float_list(cand.get("resid_h_weights", cfg.resid_h_weights), default=list(cfg.resid_h_weights)),
            teacher_forcing_start=float(cand.get("teacher_forcing_start", cfg.teacher_forcing_start)),
            teacher_forcing_end=float(cand.get("teacher_forcing_end", cfg.teacher_forcing_end)),
            teacher_forcing_schedule=str(cand.get("teacher_forcing_schedule", cfg.teacher_forcing_schedule)),
            ss_passes=int(cand.get("ss_passes", cfg.ss_passes)),
            qk_readout_only=bool(int(cand.get("hybrid_qk_readout_only", qk_readout_default))),
            qk_disable_h_list=tuple(int(v) for v in qk_disable_h_list),
            qk_resid_clip_k=float(cand.get("qk_resid_clip_k", qk_clip_default)),
            qk_factor1_scale=float(cand.get("qk_factor1_scale", 1.0)),
            head_kwargs=model_kwargs,
        ).fit_windows(
            x_train=prep.x_train,
            y_train=prep.y_train,
            x_val=prep.x_val,
            y_val=prep.y_val,
        )
        pred, hybrid_meta = hybrid.predict_with_meta(prep.x_test)
        meta.update(dict(hybrid_meta))
        return np.asarray(pred, dtype=float), meta

    model = train_classical_forecaster(
        x_train=prep.x_train,
        y_train=prep.y_train,
        x_val=prep.x_val,
        y_val=prep.y_val,
        kind=m,  # type: ignore[arg-type]
        seed=int(seed),
        **model_kwargs,
    )
    pred = np.asarray(model.predict(prep.x_test)[0], dtype=float)
    return pred, meta


def run_single_fold_model(
    model: str,
    candidate: dict[str, object],
    surfaces_input: np.ndarray,
    surfaces_truth: np.ndarray,
    train_end: int,
    cfg: NestedConfig,
    seed: int,
) -> dict[str, Any]:
    prep = _prepare_fold_data(
        surfaces_input=surfaces_input,
        surfaces_truth=surfaces_truth,
        train_end=int(train_end),
        cfg=cfg,
        seed=int(seed),
    )
    m = str(model).strip().lower()

    if m == "persistence_surface_naive":
        last_surface = np.asarray(surfaces_input[int(train_end) - 1], dtype=float)
        pred_surface = np.repeat(last_surface[None, :], int(cfg.horizon), axis=0)
        pred_delta_norm = np.zeros((int(cfg.horizon), int(cfg.pca_factors)), dtype=float)
        meta: dict[str, Any] = {"surface_persistence": True}
    else:
        pred_delta_norm, meta = _predict_delta_norm(
            model=m,
            candidate=candidate,
            prep=prep,
            cfg=cfg,
            seed=int(seed),
        )
        pred_surface = prep.fp.reconstruct_surface_from_delta_norm(pred_delta_norm, prep.base_factor)[0]

    metrics = metric_block(prep.y_true_surface, pred_surface, eps=1e-8)
    wrmse = weighted_rmse(metrics["horizon_surface_rmse"], cfg.normalized_h_weights())
    extra: dict[str, Any] = {}
    if "hybrid_base_pred_delta_norm" in meta:
        base_delta = np.asarray(meta.get("hybrid_base_pred_delta_norm"), dtype=float)
        if base_delta.ndim == 2:
            base_surface = prep.fp.reconstruct_surface_from_delta_norm(base_delta, prep.base_factor)[0]
            base_metrics = metric_block(prep.y_true_surface, base_surface, eps=1e-8)
            extra = {
                "base_surface_mae": float(base_metrics["surface_mae"]),
                "base_surface_rmse": float(base_metrics["surface_rmse"]),
                "base_surface_mape": float(base_metrics["surface_mape"]),
                "base_horizon_surface_mae": list(base_metrics["horizon_surface_mae"]),
                "base_horizon_surface_rmse": list(base_metrics["horizon_surface_rmse"]),
                "base_horizon_surface_mape": list(base_metrics["horizon_surface_mape"]),
                "corrected_minus_base_rmse": float(metrics["surface_rmse"] - base_metrics["surface_rmse"]),
                "corrected_minus_base_mape": float(metrics["surface_mape"] - base_metrics["surface_mape"]),
            }
            corr_h = np.asarray(metrics["horizon_surface_rmse"], dtype=float).reshape(-1)
            base_h = np.asarray(base_metrics["horizon_surface_rmse"], dtype=float).reshape(-1)
            if corr_h.size >= 3 and base_h.size >= 3:
                corr_h36 = float(np.mean(corr_h[2:]))
                base_h36 = float(np.mean(base_h[2:]))
                extra["h3to6_rmse_corrected_mean"] = corr_h36
                extra["h3to6_rmse_base_mean"] = base_h36
                extra["h3to6_rmse_improve_pct"] = float((base_h36 - corr_h36) / max(1e-12, base_h36) * 100.0)
            meta["hybrid_base_pred_delta_norm"] = base_delta.tolist()
    for k in ("hybrid_resid_pred_delta_norm",):
        if k in meta:
            arr = np.asarray(meta.get(k), dtype=float)
            if arr.ndim == 2:
                meta[k] = arr.tolist()

    row = {
        **metrics,
        **extra,
        "weighted_rmse": float(wrmse),
        "pred_surface": np.asarray(pred_surface, dtype=float),
        "pred_delta_norm": np.asarray(pred_delta_norm, dtype=float),
        "true_delta_norm": np.asarray(prep.y_true_delta_norm, dtype=float),
        "meta": dict(meta),
    }
    return row


def _sample_candidates(
    cands: list[dict[str, object]],
    *,
    budget: int,
    mode: str,
    seed: int,
) -> list[dict[str, object]]:
    if not cands:
        return [{}]
    n = int(max(1, min(int(budget), len(cands))))
    mode_n = str(mode).strip().lower()
    if len(cands) <= n or mode_n == "grid":
        return cands[:n]
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(cands), size=n, replace=False)
    idx = np.sort(np.asarray(idx, dtype=int))
    return [cands[int(i)] for i in idx.tolist()]


def build_model_candidates(model: str, cfg: NestedConfig) -> list[dict[str, object]]:
    m = str(model).strip().lower()
    if m in {"persistence_naive", "persistence_surface_naive", "factor_ar", "mlp", "gru", "lstm", "reservoir", "hybrid_qrc_fast", "qml_classical_reservoir", "qml_recursive_classical_reservoir", "qml_pca_lstm", "qml_convlstm"}:
        return [{}]

    if m == "gru_quantum_kernel_103754":
        # Frozen legacy copy from swaptions_cv_20260303_103754 (best_params_json).
        return [
            {
                "gru_hidden_size": 96,
                "gru_lr": 3e-4,
                "gru_dropout": 0.0,
                "gru_weight_decay": 1e-5,
                "gru_epochs": 14,
                "gru_batch_size": 32,
                "qr_corr_window": 30,
                "qr_corr_min": 0.0,
                "hybrid_alpha_mode": "per_horizon",
                "hybrid_sign_min": 0.52,
                "hybrid_force_zero_h_if_bad": 1,
                "hybrid_force_zero_h_list": "5,6",
                "hybrid_apply_correction_hmin": 3,
                "hybrid_cal_anchors": 1,
                "hybrid_cal_stride": 1,
                "hybrid_early_min_improve": 0.01,
                "hybrid_gate_tau": 4.0,
                "hybrid_gain_early": 0.35,
                "hybrid_gain_late": 1.0,
                "resid_sign": 1,
                "resid_h_weights": "0.25,0.25,1,1,1.5,1.5",
                "teacher_forcing_start": 1.0,
                "teacher_forcing_end": 0.3,
                "teacher_forcing_schedule": "linear",
                "ss_passes": 3,
                "qrc_gate_tau": 4.0,
                "hybrid_qk_readout_only": 0,
                "hybrid_qk_disable_h_list": "",
                "qk_resid_clip_k": 0.0,
                "qk_feature": "coincidence",
                "qk_state_dim_mode": "mean_last_std",
                "qk_modes": 16,
                "qk_shots": 64,
                "qk_in_scale": 1.25,
                "qk_gamma": 0.3,
                "qk_ridge": 0.001,
                "qk_gamma_grid": "0.3",
                "qk_ridge_grid": "0.001",
                "qk_horizon_mode": "all",
                "qk_target_space": "factor",
                "qk_aux_blend": 0,
                "qk_aux_kind": "none",
                "qk_resid_clip": None,
                "qk_blend_objective": "rmse",
            }
        ]

    if m in HYBRID_GRU_QUANTUM_MODELS:
        out: list[dict[str, object]] = []
        is_smoke_fast = bool(int(cfg.tune_budget) <= 12 and int(cfg.n_anchors) <= 2 and int(cfg.inner_anchors) <= 1)
        mode = str(cfg.tune_mode).strip().lower()
        rng = np.random.default_rng(stable_seed(cfg.seed, "hybrid-cands", m))
        target_random = int(max(24, int(cfg.tune_budget) * 8))
        grid_cap = int(max(128, int(cfg.tune_budget) * 250))

        def _build_candidate(
            hs: int,
            lr: float,
            drop: float,
            wd: float,
            ep: int,
            feature: str,
            modes: int,
            shots: int,
            in_scale: float,
            ridge: float,
            gain_late: float,
            gate_tau: float,
            gamma: float | None,
            resid_sign: int,
            qk_factor1_scale: float,
        ) -> dict[str, object]:
            late_gain = float(gain_late)
            if m in {"gru_photonic_memory_fb", "gru_photonic_qrc_fb"}:
                late_gain = float(min(1.0, late_gain))
            if m == "gru_quantum_kernel":
                late_gain = float(min(0.75, late_gain))
            epochs_eff = int(min(int(ep), 14 if is_smoke_fast else int(ep)))
            cal_anchors_eff = int(1 if is_smoke_fast else cfg.cal_anchors)
            base = {
                "gru_hidden_size": int(hs),
                "gru_lr": float(lr),
                "gru_dropout": float(drop),
                "gru_weight_decay": float(wd),
                "gru_epochs": int(epochs_eff),
                "gru_batch_size": 32,
                "qr_corr_window": int(cfg.corr_window),
                "qr_corr_min": float(cfg.corr_min),
                "hybrid_alpha_mode": str(cfg.alpha_mode),
                "hybrid_sign_min": float(cfg.sign_min),
                "hybrid_force_zero_h_if_bad": int(bool(cfg.force_zero_h_if_bad)),
                "hybrid_force_zero_h_list": ",".join(str(int(v)) for v in cfg.force_zero_h_list),
                "hybrid_apply_correction_hmin": int(1 if m == "gru_quantum_kernel" else cfg.apply_correction_hmin),
                "hybrid_cal_anchors": int(cal_anchors_eff),
                "hybrid_cal_stride": int(cfg.cal_stride),
                "hybrid_early_min_improve": float(0.0025 if m == "gru_quantum_kernel" else 0.01),
                "hybrid_gate_tau": float(gate_tau),
                "hybrid_gain_early": float(min(0.35, max(0.0, late_gain))),
                "hybrid_gain_late": float(late_gain),
                "resid_sign": int(1 if int(resid_sign) >= 0 else -1),
                "resid_h_weights": ",".join(f"{float(v):g}" for v in cfg.resid_h_weights),
                "teacher_forcing_start": float(cfg.teacher_forcing_start),
                "teacher_forcing_end": float(cfg.teacher_forcing_end),
                "teacher_forcing_schedule": str(cfg.teacher_forcing_schedule),
                "ss_passes": int(cfg.ss_passes),
                "qrc_gate_tau": float(gate_tau),
                "hybrid_qk_readout_only": int(1 if m == "gru_quantum_kernel" else 0),
                "hybrid_qk_disable_h_list": ("5,6" if m == "gru_quantum_kernel" else ""),
                "qk_resid_clip_k": float(2.5 if m == "gru_quantum_kernel" else 0.0),
                "qk_factor1_scale": float(np.clip(float(qk_factor1_scale), 0.0, 1.0)),
            }
            if m == "gru_photonic_memory_fb":
                base.update(
                    {
                        "pqrc_feature": str(feature),
                        "pqrc_modes": int(modes),
                        "pqrc_shots": int(shots),
                        "pqrc_input_scale": float(in_scale),
                        "pqrc_ridge": float(ridge),
                        "pqrc_gain": float(late_gain),
                        "qrc_mode": "residual",
                        "qrc_target": "delta",
                        "qrc_baseline": "persistence",
                        "target_transform": "log",
                        "y_floor_mode": "train_p001",
                        "qrc_resid_clip": 3.0,
                        "qrc_residvar_penalty": 0.0,
                    }
                )
                return base
            if m == "gru_photonic_qrc_fb":
                base.update(
                    {
                        "pqrc_feature": str(feature),
                        "pqrc_modes": int(modes),
                        "pqrc_shots": int(shots),
                        "pqrc_input_scale": float(in_scale),
                        "pqrc_ridge": float(ridge),
                        "pqrc_gain": float(late_gain),
                        "pqrc_higher_order": int(1 if modes >= 12 else 0),
                        "qrc_mode": "residual",
                        "qrc_target": "delta",
                        "qrc_baseline": "persistence",
                        "target_transform": "log",
                        "y_floor_mode": "train_p001",
                        "qrc_resid_clip": 3.0,
                        "qrc_residvar_penalty": 0.0,
                    }
                )
                return base
            g = float(cfg.qk_gamma_grid[0] if gamma is None else gamma)
            base.update(
                {
                    "qk_feature": str(feature),
                    "qk_state_dim_mode": "mean_last_std",
                    "qk_modes": int(modes),
                    "qk_shots": int(shots),
                    "qk_in_scale": float(in_scale),
                    "qk_gamma": float(g),
                    "qk_ridge": float(ridge),
                    "qk_gamma_grid": str(g),
                    "qk_ridge_grid": str(ridge),
                    "qk_horizon_mode": "all",
                    "qk_target_space": "factor",
                    "qk_aux_blend": 0,
                    "qk_aux_kind": "none",
                    "qk_resid_clip": None,
                    "qk_blend_objective": "rmse",
                }
            )
            return base

        if mode == "random":
            seen: set[str] = set()
            attempts = 0
            max_attempts = int(target_random * 40)
            while len(out) < target_random and attempts < max_attempts:
                attempts += 1
                cand = _build_candidate(
                    hs=int(rng.choice(np.asarray(cfg.gru_hidden_grid, dtype=int))),
                    lr=float(rng.choice(np.asarray(cfg.gru_lr_grid, dtype=float))),
                    drop=float(rng.choice(np.asarray(cfg.gru_dropout_grid, dtype=float))),
                    wd=float(rng.choice(np.asarray(cfg.gru_weight_decay_grid, dtype=float))),
                    ep=int(rng.choice(np.asarray(cfg.gru_epochs_grid, dtype=int))),
                    feature=str(rng.choice(np.asarray(cfg.q_feature_grid, dtype=object))),
                    modes=int(rng.choice(np.asarray(cfg.q_modes_grid, dtype=int))),
                    shots=int(rng.choice(np.asarray(cfg.q_shots_grid, dtype=int))),
                    in_scale=float(rng.choice(np.asarray(cfg.q_in_scale_grid, dtype=float))),
                    ridge=float(rng.choice(np.asarray(cfg.q_ridge_grid, dtype=float))),
                    gain_late=float(rng.choice(np.asarray(cfg.gain_grid, dtype=float))),
                    gate_tau=float(rng.choice(np.asarray(cfg.gate_tau_grid, dtype=float))),
                    gamma=(None if m != "gru_quantum_kernel" else float(rng.choice(np.asarray(cfg.qk_gamma_grid, dtype=float)))),
                    resid_sign=int(rng.choice(np.asarray(cfg.resid_sign_grid, dtype=int))),
                    qk_factor1_scale=(1.0 if m != "gru_quantum_kernel" else float(rng.choice(np.asarray(cfg.qk_factor1_scale_grid, dtype=float)))),
                )
                key = json.dumps(cand, sort_keys=True, default=str, separators=(",", ":"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
            if out:
                return out

        for hs in cfg.gru_hidden_grid:
            for lr in cfg.gru_lr_grid:
                for drop in cfg.gru_dropout_grid:
                    for wd in cfg.gru_weight_decay_grid:
                        for ep in cfg.gru_epochs_grid:
                            for feature in cfg.q_feature_grid:
                                for modes in cfg.q_modes_grid:
                                    for shots in cfg.q_shots_grid:
                                        for in_scale in cfg.q_in_scale_grid:
                                            for ridge in cfg.q_ridge_grid:
                                                for gain_late in cfg.gain_grid:
                                                    for gate_tau in cfg.gate_tau_grid:
                                                        for resid_sign in cfg.resid_sign_grid:
                                                            if m == "gru_quantum_kernel":
                                                                for gamma in cfg.qk_gamma_grid:
                                                                    for f1 in cfg.qk_factor1_scale_grid:
                                                                        out.append(
                                                                            _build_candidate(
                                                                                hs=int(hs),
                                                                                lr=float(lr),
                                                                                drop=float(drop),
                                                                                wd=float(wd),
                                                                                ep=int(ep),
                                                                                feature=str(feature),
                                                                                modes=int(modes),
                                                                                shots=int(shots),
                                                                                in_scale=float(in_scale),
                                                                                ridge=float(ridge),
                                                                                gain_late=float(gain_late),
                                                                                gate_tau=float(gate_tau),
                                                                                gamma=float(gamma),
                                                                                resid_sign=int(resid_sign),
                                                                                qk_factor1_scale=float(f1),
                                                                            )
                                                                        )
                                                                        if len(out) >= grid_cap:
                                                                            return out
                                                                continue
                                                            out.append(
                                                                _build_candidate(
                                                                    hs=int(hs),
                                                                    lr=float(lr),
                                                                    drop=float(drop),
                                                                    wd=float(wd),
                                                                    ep=int(ep),
                                                                    feature=str(feature),
                                                                    modes=int(modes),
                                                                    shots=int(shots),
                                                                    in_scale=float(in_scale),
                                                                    ridge=float(ridge),
                                                                    gain_late=float(gain_late),
                                                                    gate_tau=float(gate_tau),
                                                                    gamma=None,
                                                                    resid_sign=int(resid_sign),
                                                                    qk_factor1_scale=1.0,
                                                                )
                                                            )
                                                            if len(out) >= grid_cap:
                                                                return out
        return out

    if m == "quantum_kernel_photonic_memory":
        out: list[dict[str, object]] = []
        for feature in cfg.q_feature_grid:
            for modes in cfg.q_modes_grid:
                for shots in cfg.q_shots_grid:
                    for in_scale in cfg.q_in_scale_grid:
                        for gamma in cfg.qk_gamma_grid:
                            for ridge in cfg.q_ridge_grid:
                                for gain in cfg.gain_grid:
                                    out.append(
                                        {
                                            "use_quantum_corrector": 1,
                                            "qr_base_kind": str(cfg.quantum_base_kind),
                                            "qr_gain": float(gain),
                                            "qr_corr_window": int(cfg.corr_window),
                                            "qr_corr_min": float(cfg.corr_min),
                                            "qk_feature": str(feature),
                                            "qk_state_dim_mode": "mean_last_std",
                                            "qk_modes": int(modes),
                                            "qk_shots": int(shots),
                                            "qk_in_scale": float(in_scale),
                                            "qk_gamma": float(gamma),
                                            "qk_ridge": float(ridge),
                                            "qk_gamma_grid": str(gamma),
                                            "qk_ridge_grid": str(ridge),
                                            "qk_horizon_mode": "all",
                                            "qk_target_space": "factor",
                                            "qk_aux_blend": 0,
                                            "qk_aux_kind": "none",
                                            "qk_resid_clip": None,
                                            "qk_blend_objective": "rmse",
                                        }
                                    )
        return out

    if m in QUANTUM_MODELS:
        out = []
        no_feedback = m in {"photonic_qrc_no_feedback", "photonic_memory_no_feedback", "persist_qrc_weak"}
        gains_internal = [0.0] if no_feedback else cfg.gain_grid
        for feature in cfg.q_feature_grid:
            for modes in cfg.q_modes_grid:
                for shots in cfg.q_shots_grid:
                    for in_scale in cfg.q_in_scale_grid:
                        for ridge in cfg.q_ridge_grid:
                            for gate_tau in cfg.gate_tau_grid:
                                for gain in cfg.gain_grid:
                                    out.append(
                                        {
                                            "use_quantum_corrector": 1,
                                            "qr_base_kind": str(cfg.quantum_base_kind),
                                            "qr_gain": float(gain),
                                            "qr_corr_window": int(cfg.corr_window),
                                            "qr_corr_min": float(cfg.corr_min),
                                            "pqrc_feature": str(feature),
                                            "pqrc_modes": int(modes),
                                            "pqrc_shots": int(shots),
                                            "pqrc_input_scale": float(in_scale),
                                            "pqrc_ridge": float(ridge),
                                            "pqrc_gain": float(gains_internal[-1] if no_feedback else gain),
                                            "qrc_gate_tau": float(gate_tau),
                                            "qrc_mode": "residual",
                                            "qrc_target": "delta",
                                            "qrc_baseline": "persistence",
                                            "target_transform": "log",
                                            "y_floor_mode": "train_p001",
                                            "qrc_resid_clip": None,
                                            "qrc_residvar_penalty": 0.0,
                                        }
                                    )
        return out
    return [{}]


def evaluate_candidate_on_inner(
    model: str,
    candidate: dict[str, object],
    segment_input: np.ndarray,
    segment_truth: np.ndarray,
    inner_cv: PseudoNext6CV,
    cfg: NestedConfig,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    h_weights = cfg.normalized_h_weights()
    for fold_idx, fold in enumerate(inner_cv.iter_folds(), start=1):
        validate_fold_order(fold.train_idx, fold.test_idx)
        anchor = int(fold.train_idx.shape[0])
        row = run_single_fold_model(
            model=model,
            candidate=candidate,
            surfaces_input=segment_input,
            surfaces_truth=segment_truth,
            train_end=anchor,
            cfg=cfg,
            seed=stable_seed(seed, model, "inner", fold_idx, anchor, json.dumps(candidate, sort_keys=True, default=str)),
        )
        wrmse = float(weighted_rmse(row["horizon_surface_rmse"], h_weights))
        rows.append(
            {
                "fold_idx": int(fold_idx),
                "anchor": int(anchor),
                "weighted_rmse": float(wrmse),
                "surface_rmse": float(row["surface_rmse"]),
                "surface_mae": float(row["surface_mae"]),
                "surface_mape": float(row["surface_mape"]),
            }
        )
    df = pd.DataFrame(rows)
    mean_w = float(df["weighted_rmse"].mean()) if not df.empty else float("inf")
    std_w = float(df["weighted_rmse"].std(ddof=0)) if not df.empty else float("inf")
    obj = float(mean_w + float(cfg.lambda_std) * std_w)
    return {
        "objective": float(obj),
        "mean_weighted_rmse": float(mean_w),
        "std_weighted_rmse": float(std_w),
        "rows": rows,
    }


def tune_model_on_segment(
    model: str,
    segment_input: np.ndarray,
    segment_truth: np.ndarray,
    cfg: NestedConfig,
    seed: int,
    inner_anchors_override: int | None = None,
) -> tuple[dict[str, object], dict[str, Any], pd.DataFrame]:
    n_inner = int(cfg.inner_anchors if inner_anchors_override is None else inner_anchors_override)
    inner_cv = PseudoNext6CV(
        dataset=int(segment_input.shape[0]),
        lookback=int(cfg.lookback),
        horizon=int(cfg.horizon),
        n_anchors=int(max(1, n_inner)),
        anchor_strategy=str(cfg.cv_anchor_strategy),
        min_train=int(cfg.min_train),
        seed=int(seed),
    )
    candidates = build_model_candidates(model, cfg)
    if not bool(cfg.nested_tune):
        candidates = [candidates[0] if candidates else {}]
    sampled = _sample_candidates(
        candidates,
        budget=int(cfg.tune_budget),
        mode=str(cfg.tune_mode),
        seed=stable_seed(seed, model, "sample"),
    )
    if bool(int(cfg.tune_budget) <= 12 and int(cfg.n_anchors) <= 2 and int(cfg.inner_anchors) <= 1):
        sampled = sampled[: int(max(1, min(len(sampled), 1)))]
    rows: list[dict[str, Any]] = []
    best_obj = float("inf")
    best_cand: dict[str, object] = {}
    best_meta: dict[str, Any] = {}
    for idx, cand in enumerate(sampled, start=1):
        eval_out = evaluate_candidate_on_inner(
            model=model,
            candidate=cand,
            segment_input=segment_input,
            segment_truth=segment_truth,
            inner_cv=inner_cv,
            cfg=cfg,
            seed=stable_seed(seed, model, "cand", idx),
        )
        row = {
            "candidate_idx": int(idx),
            "objective": float(eval_out["objective"]),
            "mean_weighted_rmse": float(eval_out["mean_weighted_rmse"]),
            "std_weighted_rmse": float(eval_out["std_weighted_rmse"]),
            "candidate_json": json.dumps(cand, sort_keys=True, default=str),
        }
        rows.append(row)
        if float(row["objective"]) < best_obj:
            best_obj = float(row["objective"])
            best_cand = dict(cand)
            best_meta = {
                "objective": float(eval_out["objective"]),
                "mean_weighted_rmse": float(eval_out["mean_weighted_rmse"]),
                "std_weighted_rmse": float(eval_out["std_weighted_rmse"]),
            }
    return best_cand, best_meta, pd.DataFrame(rows)


def _aggregate_model_rows(rows: list[dict[str, Any]], cfg: NestedConfig) -> dict[str, Any]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "outer_folds": int(len(rows)),
        "weighted_rmse_mean": float(df["weighted_rmse"].mean()),
        "weighted_rmse_std": float(df["weighted_rmse"].std(ddof=0)),
        "surface_rmse_mean": float(df["surface_rmse"].mean()),
        "surface_rmse_std": float(df["surface_rmse"].std(ddof=0)),
        "surface_mae_mean": float(df["surface_mae"].mean()),
        "surface_mae_std": float(df["surface_mae"].std(ddof=0)),
        "surface_mape_mean": float(df["surface_mape"].mean()),
        "surface_mape_std": float(df["surface_mape"].std(ddof=0)),
        "objective": float(df["weighted_rmse"].mean() + float(cfg.lambda_std) * df["weighted_rmse"].std(ddof=0)),
    }
    h = int(cfg.horizon)
    def _arr(key: str) -> np.ndarray:
        vals = []
        for r in rows:
            v = r.get(key, [])
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = []
            vals.append(np.asarray(v, dtype=float).reshape(-1))
        mat = np.asarray(vals, dtype=float)
        if mat.ndim != 2 or mat.shape[1] != h:
            mat2 = np.zeros((len(rows), h), dtype=float)
            for i, v in enumerate(vals):
                k = int(min(h, v.shape[0]))
                if k > 0:
                    mat2[i, :k] = v[:k]
            mat = mat2
        return mat

    rmse_mat = _arr("horizon_surface_rmse")
    mae_mat = _arr("horizon_surface_mae")
    out["horizon_surface_rmse_mean"] = json.dumps(np.mean(rmse_mat, axis=0).tolist())
    out["horizon_surface_rmse_std"] = json.dumps(np.std(rmse_mat, axis=0, ddof=0).tolist())
    out["horizon_surface_mae_mean"] = json.dumps(np.mean(mae_mat, axis=0).tolist())
    out["horizon_surface_mae_std"] = json.dumps(np.std(mae_mat, axis=0, ddof=0).tolist())

    for key in [
        "base_horizon_surface_rmse",
        "base_horizon_surface_mae",
        "hybrid_alpha_h",
        "hybrid_corr_h",
        "hybrid_sign_h",
        "hybrid_sign_agreement_h",
        "hybrid_gain_h",
    ]:
        if key in df.columns:
            mat = _arr(key)
            out[f"{key}_mean"] = json.dumps(np.mean(mat, axis=0).tolist())
            out[f"{key}_std"] = json.dumps(np.std(mat, axis=0, ddof=0).tolist())
    for key in [
        "hybrid_resid_sign",
        "hybrid_teacher_forcing_start",
        "hybrid_teacher_forcing_end",
        "hybrid_ss_passes",
    ]:
        if key in df.columns:
            vv = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
            vv = vv[np.isfinite(vv)]
            if vv.size:
                out[f"{key}_mean"] = float(np.mean(vv))
                out[f"{key}_std"] = float(np.std(vv, ddof=0))
    return out


def _fit_ensemble_inner(
    segment_input: np.ndarray,
    segment_truth: np.ndarray,
    cfg: NestedConfig,
    seed: int,
    base_members: list[str] | None = None,
    q_model_override: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inner_cv = PseudoNext6CV(
        dataset=int(segment_input.shape[0]),
        lookback=int(cfg.lookback),
        horizon=int(cfg.horizon),
        n_anchors=int(max(1, cfg.inner_anchors)),
        anchor_strategy=str(cfg.cv_anchor_strategy),
        min_train=int(cfg.min_train),
        seed=int(seed),
    )

    members_cfg = [str(m).strip().lower() for m in (base_members if base_members is not None else cfg.ensemble_members)]
    if not members_cfg:
        members_cfg = ["gru", "factor_ar", "qml_pca_lstm"]
    member_cfgs: dict[str, dict[str, object]] = {}
    tune_rows: list[dict[str, Any]] = []
    for m in members_cfg:
        best_c, best_meta, tune_df = tune_model_on_segment(
            model=str(m),
            segment_input=segment_input,
            segment_truth=segment_truth,
            cfg=cfg,
            seed=stable_seed(seed, "ens-member", m),
            inner_anchors_override=max(1, cfg.inner_anchors),
        )
        member_cfgs[str(m)] = dict(best_c)
        for _, r in tune_df.iterrows():
            rr = dict(r)
            rr["member_model"] = str(m)
            tune_rows.append(rr)
        tune_rows.append(
            {
                "member_model": str(m),
                "selected": 1,
                "objective": float(best_meta.get("objective", float("nan"))),
                "candidate_json": json.dumps(best_c, sort_keys=True, default=str),
            }
        )

    fold_pack_base: list[dict[str, Any]] = []
    for fidx, fold in enumerate(inner_cv.iter_folds(), start=1):
        validate_fold_order(fold.train_idx, fold.test_idx)
        anchor = int(fold.train_idx.shape[0])
        prep = _prepare_fold_data(
            surfaces_input=segment_input,
            surfaces_truth=segment_truth,
            train_end=anchor,
            cfg=cfg,
            seed=stable_seed(seed, "ens-prep", fidx, anchor),
        )
        member_delta: list[np.ndarray] = []
        member_surface: list[np.ndarray] = []
        for m in members_cfg:
            pred_delta, _meta = _predict_delta_norm(
                model=str(m),
                candidate=member_cfgs[str(m)],
                prep=prep,
                cfg=cfg,
                seed=stable_seed(seed, "ens-member-pred", m, fidx),
            )
            pred_surface = prep.fp.reconstruct_surface_from_delta_norm(pred_delta, prep.base_factor)[0]
            member_delta.append(np.asarray(pred_delta, dtype=float))
            member_surface.append(np.asarray(pred_surface, dtype=float))
        fold_pack_base.append(
            {
                "anchor": int(anchor),
                "prep": prep,
                "member_delta": np.asarray(member_delta, dtype=float),
                "member_surface": np.asarray(member_surface, dtype=float),
            }
        )

    y_true_delta = np.asarray([p["prep"].y_true_delta_norm for p in fold_pack_base], dtype=float)
    member_pred_delta = np.asarray([p["member_delta"] for p in fold_pack_base], dtype=float)
    member_pred_delta = np.transpose(member_pred_delta, (0, 1, 2, 3))
    ens_base = EnsembleForecaster(member_names=list(members_cfg)).fit(member_pred_delta, y_true_delta)

    def _objective_from_weights(weights: np.ndarray, fold_pack: list[dict[str, Any]]) -> float:
        wr: list[float] = []
        h_weights = cfg.normalized_h_weights()
        for pack in fold_pack:
            prep: FoldPreparedData = pack["prep"]
            deltas = np.asarray(pack["member_delta"], dtype=float)
            pred_delta = np.einsum("mhf,m->hf", deltas, weights, optimize=True)
            pred_surface = prep.fp.reconstruct_surface_from_delta_norm(pred_delta, prep.base_factor)[0]
            block = metric_block(prep.y_true_surface, pred_surface, eps=1e-8)
            wr.append(weighted_rmse(block["horizon_surface_rmse"], h_weights))
        arr = np.asarray(wr, dtype=float)
        return float(np.mean(arr) + float(cfg.lambda_std) * np.std(arr, ddof=0))

    objective_base = _objective_from_weights(np.asarray(ens_base.weights, dtype=float), fold_pack_base)

    include_quantum = False
    quantum_cfg: dict[str, object] = {}
    final_weights = np.asarray(ens_base.weights, dtype=float)
    final_members = list(members_cfg)
    objective_with_quantum = float("nan")
    relative_improve = float("nan")
    decision = "dropped"

    q_model = str(q_model_override if q_model_override is not None else cfg.ensemble_quantum_kind).strip().lower()
    if q_model in ACTIVE_MODELS and q_model not in members_cfg:
        best_q, _, _ = tune_model_on_segment(
            model=q_model,
            segment_input=segment_input,
            segment_truth=segment_truth,
            cfg=cfg,
            seed=stable_seed(seed, "ens-q"),
            inner_anchors_override=max(1, cfg.inner_anchors),
        )
        fold_pack_q: list[dict[str, Any]] = []
        for fidx, fold in enumerate(inner_cv.iter_folds(), start=1):
            anchor = int(fold.train_idx.shape[0])
            prep = _prepare_fold_data(
                surfaces_input=segment_input,
                surfaces_truth=segment_truth,
                train_end=anchor,
                cfg=cfg,
                seed=stable_seed(seed, "ens-q-prep", fidx, anchor),
            )
            pred_delta, _ = _predict_delta_norm(
                model=q_model,
                candidate=best_q,
                prep=prep,
                cfg=cfg,
                seed=stable_seed(seed, "ens-q-pred", fidx, anchor),
            )
            fold_pack_q.append({"member_delta_q": np.asarray(pred_delta, dtype=float), "prep": prep})

        member_plus = []
        for i, base_pack in enumerate(fold_pack_base):
            d_base = np.asarray(base_pack["member_delta"], dtype=float)
            d_q = np.asarray(fold_pack_q[i]["member_delta_q"], dtype=float)[None, :, :]
            member_plus.append(np.concatenate([d_base, d_q], axis=0))
        member_plus_arr = np.asarray(member_plus, dtype=float)
        ens_q = EnsembleForecaster(member_names=list(members_cfg) + [q_model]).fit(member_plus_arr, y_true_delta)
        q_obj = _objective_from_weights(np.asarray(ens_q.weights, dtype=float), [{"member_delta": member_plus_arr[i], "prep": fold_pack_base[i]["prep"]} for i in range(len(fold_pack_base))])
        objective_with_quantum = float(q_obj)
        improve = float((objective_base - q_obj) / max(1e-12, objective_base))
        relative_improve = float(improve)
        if improve >= float(cfg.ensemble_quantum_min_improve):
            include_quantum = True
            quantum_cfg = dict(best_q)
            final_weights = np.asarray(ens_q.weights, dtype=float)
            final_members = list(members_cfg) + [q_model]
            decision = "kept"
        else:
            decision = "dropped"

    out = {
        "members": list(final_members),
        "weights": [float(x) for x in np.asarray(final_weights, dtype=float).tolist()],
        "member_cfgs": {k: dict(v) for k, v in member_cfgs.items()},
        "include_quantum": bool(include_quantum),
        "quantum_model": str(q_model),
        "quantum_cfg": dict(quantum_cfg),
        "objective": float(objective_with_quantum if include_quantum and np.isfinite(objective_with_quantum) else objective_base),
        "objective_base": float(objective_base),
        "objective_with_quantum": float(objective_with_quantum),
        "relative_improve": float(relative_improve),
        "decision": str(decision),
    }
    tune_rows.append(
        {
            "member_model": "quantum_candidate",
            "selected": int(include_quantum),
            "objective_base": float(objective_base),
            "objective_with_quantum": float(objective_with_quantum),
            "relative_improve": float(relative_improve),
            "decision": str(decision),
            "candidate_json": json.dumps(quantum_cfg if include_quantum else {}, sort_keys=True, default=str),
        }
    )
    return out, pd.DataFrame(tune_rows)


def run_nested_cv(
    cfg: NestedConfig,
    models: list[str],
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset_bundle(cfg)
    n_obs = int(bundle.surfaces_input.shape[0])

    outer_cv = PseudoNext6CV(
        dataset=n_obs,
        lookback=int(cfg.lookback),
        horizon=int(cfg.horizon),
        n_anchors=int(cfg.n_anchors),
        anchor_strategy=str(cfg.cv_anchor_strategy),
        min_train=int(cfg.min_train),
        seed=int(cfg.seed),
    )
    outer_folds = list(outer_cv.iter_folds())

    all_summary_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    best_params_by_model: dict[str, dict[str, object]] = {}
    selected_by_fold: list[dict[str, Any]] = []

    for model in models:
        m = str(model).strip().lower()
        if bool(cfg.skip_slow) and m in SLOW_MODELS:
            print(f"[nested][{m}] skipped because --skip_slow=1", flush=True)
            continue
        model_rows: list[dict[str, Any]] = []
        model_best_candidates: list[dict[str, object]] = []
        model_inner_objectives: list[float] = []
        model_dir = out_dir / m
        model_dir.mkdir(parents=True, exist_ok=True)

        for outer_idx, fold in enumerate(outer_folds, start=1):
            validate_fold_order(fold.train_idx, fold.test_idx)
            train_end = int(fold.train_idx.shape[0])
            test_start = int(fold.test_idx[0])
            test_end = int(fold.test_idx[-1])
            print(
                f"[nested][{m}] outer_fold={outer_idx}/{len(outer_folds)} "
                f"train_end_idx={train_end - 1} test_idx={test_start}-{test_end}",
                flush=True,
            )

            segment_input = np.asarray(bundle.surfaces_input[:train_end], dtype=float)
            segment_truth = np.asarray(bundle.surfaces_truth[:train_end], dtype=float)

            is_ensemble = m in {"ensemble", "ensemble_stack_safe"}
            if is_ensemble:
                ens_base_members: list[str] | None = None
                ens_q_override: str | None = None
                if m == "ensemble_stack_safe":
                    ens_base_members = ["gru", "factor_ar", "qml_pca_lstm"]
                    q_kind = str(cfg.ensemble_quantum_kind).strip().lower()
                    if q_kind not in HYBRID_GRU_QUANTUM_MODELS:
                        q_kind = "gru_quantum_kernel"
                    ens_q_override = q_kind
                ens_state, ens_tune_df = _fit_ensemble_inner(
                    segment_input=segment_input,
                    segment_truth=segment_truth,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "outer", outer_idx),
                    base_members=ens_base_members,
                    q_model_override=ens_q_override,
                )
                best_cand = {"ensemble_state": ens_state}
                tune_meta = {"objective": float(ens_state.get("objective", np.nan))}
                if not ens_tune_df.empty:
                    ens_tune_df.to_csv(model_dir / f"outer{outer_idx:02d}_inner_tuning.csv", index=False)
            else:
                best_cand, tune_meta, tune_df = tune_model_on_segment(
                    model=m,
                    segment_input=segment_input,
                    segment_truth=segment_truth,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "outer", outer_idx),
                    inner_anchors_override=max(1, cfg.inner_anchors),
                )
                if not tune_df.empty:
                    tune_df.to_csv(model_dir / f"outer{outer_idx:02d}_inner_tuning.csv", index=False)

            selected_by_fold.append(
                {
                    "model": m,
                    "outer_fold": int(outer_idx),
                    "train_end": int(train_end),
                    "test_start": int(test_start),
                    "test_end": int(test_end),
                    "best_candidate_json": json.dumps(best_cand, sort_keys=True, default=str),
                    "inner_objective": float(tune_meta.get("objective", np.nan)),
                }
            )
            model_best_candidates.append(dict(best_cand))
            model_inner_objectives.append(float(tune_meta.get("objective", np.nan)))

            if is_ensemble:
                ens_state = dict(best_cand.get("ensemble_state", {}))
                members = [str(x) for x in ens_state.get("members", [])]
                weights = np.asarray(ens_state.get("weights", []), dtype=float)
                member_cfgs = {str(k): dict(v) for k, v in dict(ens_state.get("member_cfgs", {})).items()}
                include_quantum = bool(ens_state.get("include_quantum", False))
                q_model = str(ens_state.get("quantum_model", ""))
                q_cfg = dict(ens_state.get("quantum_cfg", {}))
                prep = _prepare_fold_data(
                    surfaces_input=bundle.surfaces_input,
                    surfaces_truth=bundle.surfaces_truth,
                    train_end=train_end,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "prep", outer_idx),
                )
                member_deltas: list[np.ndarray] = []
                member_surfaces: list[np.ndarray] = []
                for mm in members:
                    cand_mm = dict(member_cfgs.get(mm, {}))
                    if include_quantum and mm == q_model:
                        cand_mm = dict(q_cfg)
                    pred_delta, _ = _predict_delta_norm(
                        model=mm,
                        candidate=cand_mm,
                        prep=prep,
                        cfg=cfg,
                        seed=stable_seed(cfg.seed, m, "outerpred", outer_idx, mm),
                    )
                    member_deltas.append(np.asarray(pred_delta, dtype=float))
                    member_surfaces.append(np.asarray(prep.fp.reconstruct_surface_from_delta_norm(pred_delta, prep.base_factor)[0], dtype=float))
                if weights.size != len(member_deltas):
                    weights = np.zeros((len(member_deltas),), dtype=float)
                    weights[0] = 1.0
                pred_delta = np.einsum("mhf,m->hf", np.asarray(member_deltas, dtype=float), weights, optimize=True)
                pred_surface = prep.fp.reconstruct_surface_from_delta_norm(pred_delta, prep.base_factor)[0]
                meta = {
                    "ensemble": True,
                    "members": members,
                    "weights": weights.tolist(),
                    "ensemble_objective_base": float(ens_state.get("objective_base", np.nan)),
                    "ensemble_objective_with_quantum": float(ens_state.get("objective_with_quantum", np.nan)),
                    "ensemble_relative_improve": float(ens_state.get("relative_improve", np.nan)),
                    "ensemble_quantum_decision": str(ens_state.get("decision", "")),
                    "ensemble_include_quantum": bool(ens_state.get("include_quantum", False)),
                }
                fold_out: dict[str, Any] = {}
            else:
                fold_out = run_single_fold_model(
                    model=m,
                    candidate=best_cand,
                    surfaces_input=bundle.surfaces_input,
                    surfaces_truth=bundle.surfaces_truth,
                    train_end=train_end,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "outerpred", outer_idx),
                )
                pred_surface = np.asarray(fold_out["pred_surface"], dtype=float)
                pred_delta = np.asarray(fold_out["pred_delta_norm"], dtype=float)
                meta = dict(fold_out.get("meta", {}))

            true_surface = np.asarray(bundle.surfaces_truth[test_start : test_start + int(cfg.horizon)], dtype=float)
            block = metric_block(true_surface, pred_surface, eps=1e-8)
            wrmse = weighted_rmse(block["horizon_surface_rmse"], cfg.normalized_h_weights())
            row = {
                "model": m,
                "outer_fold": int(outer_idx),
                "train_end_idx": int(train_end - 1),
                "test_start_idx": int(test_start),
                "test_end_idx": int(test_end),
                "train_end_date": str(bundle.dates[train_end - 1]),
                "test_dates": json.dumps([str(x) for x in bundle.dates[test_start : test_start + int(cfg.horizon)].tolist()]),
                "weighted_rmse": float(wrmse),
                "surface_rmse": float(block["surface_rmse"]),
                "surface_mae": float(block["surface_mae"]),
                "surface_mape": float(block["surface_mape"]),
                "horizon_surface_rmse": json.dumps(block["horizon_surface_rmse"]),
                "horizon_surface_mae": json.dumps(block["horizon_surface_mae"]),
                "horizon_surface_mape": json.dumps(block["horizon_surface_mape"]),
                "selected_candidate_json": json.dumps(best_cand, sort_keys=True, default=str),
                "meta_json": json.dumps(meta, sort_keys=True, default=str),
            }
            if fold_out:
                for key in [
                    "base_surface_mae",
                    "base_surface_rmse",
                    "base_surface_mape",
                    "corrected_minus_base_rmse",
                    "corrected_minus_base_mape",
                    "h3to6_rmse_corrected_mean",
                    "h3to6_rmse_base_mean",
                    "h3to6_rmse_improve_pct",
                ]:
                    if key in fold_out:
                        row[key] = float(fold_out[key])
                for key in [
                    "base_horizon_surface_mae",
                    "base_horizon_surface_rmse",
                    "base_horizon_surface_mape",
                ]:
                    if key in fold_out:
                        row[key] = json.dumps(fold_out[key])
                if "hybrid_alpha_h" in meta:
                    row["hybrid_alpha_h"] = json.dumps(meta["hybrid_alpha_h"])
                if "hybrid_corr_h" in meta:
                    row["hybrid_corr_h"] = json.dumps(meta["hybrid_corr_h"])
                if "hybrid_sign_h" in meta:
                    row["hybrid_sign_h"] = json.dumps(meta["hybrid_sign_h"])
                if "hybrid_sign_agreement_h" in meta:
                    row["hybrid_sign_agreement_h"] = json.dumps(meta["hybrid_sign_agreement_h"])
                if "hybrid_gain_h" in meta:
                    row["hybrid_gain_h"] = json.dumps(meta["hybrid_gain_h"])
                for key in [
                    "hybrid_resid_sign",
                    "hybrid_teacher_forcing_start",
                    "hybrid_teacher_forcing_end",
                    "hybrid_ss_passes",
                ]:
                    if key in meta:
                        row[key] = float(meta[key]) if key != "hybrid_resid_sign" else int(meta[key])
                for key in [
                    "hybrid_alpha_mode",
                    "hybrid_teacher_forcing_schedule",
                ]:
                    if key in meta:
                        row[key] = str(meta[key])
                if "hybrid_force_zero_h_if_bad" in meta:
                    row["hybrid_force_zero_h_if_bad"] = int(bool(meta["hybrid_force_zero_h_if_bad"]))
                if "hybrid_force_zero_h_list" in meta:
                    row["hybrid_force_zero_h_list"] = json.dumps(meta["hybrid_force_zero_h_list"])
                if "hybrid_resid_h_weights" in meta:
                    row["hybrid_resid_h_weights"] = json.dumps(meta["hybrid_resid_h_weights"])
            model_rows.append(row)
            all_fold_rows.append(dict(row))

            pred_df = pd.DataFrame(pred_surface, columns=bundle.surface_cols)
            pred_df.insert(0, "Date", pd.Series(bundle.dates[test_start : test_start + int(cfg.horizon)]).astype(str).tolist())
            pred_df.to_csv(model_dir / f"outer{outer_idx:02d}_pred.csv", index=False)
            true_df = pd.DataFrame(true_surface, columns=bundle.surface_cols)
            true_df.insert(0, "Date", pd.Series(bundle.dates[test_start : test_start + int(cfg.horizon)]).astype(str).tolist())
            true_df.to_csv(model_dir / f"outer{outer_idx:02d}_true.csv", index=False)

        if not model_rows:
            continue
        fold_df = pd.DataFrame(model_rows)
        fold_df.to_csv(model_dir / "outer_fold_metrics.csv", index=False)
        agg = _aggregate_model_rows(model_rows, cfg)
        summary_row = {"model": m, **agg}
        all_summary_rows.append(summary_row)

        if model_best_candidates:
            valid = [(i, float(v)) for i, v in enumerate(model_inner_objectives) if np.isfinite(float(v))]
            if valid:
                best_i = min(valid, key=lambda t: t[1])[0]
            else:
                best_i = len(model_best_candidates) // 2
            best_params_by_model[m] = dict(model_best_candidates[int(best_i)])

    summary_df = pd.DataFrame(all_summary_rows)
    if not summary_df.empty and "objective" in summary_df.columns:
        summary_df = summary_df.sort_values("objective", ascending=True)
    fold_df_all = pd.DataFrame(all_fold_rows)
    selected_df = pd.DataFrame(selected_by_fold)

    q_help_rows: list[dict[str, Any]] = []
    for qm in ["gru_photonic_memory_fb", "gru_quantum_kernel", "gru_quantum_kernel_103754", "gru_photonic_qrc_fb"]:
        if fold_df_all.empty or "model" not in fold_df_all.columns:
            continue
        mdf = fold_df_all.loc[fold_df_all["model"].astype(str) == str(qm)].copy()
        if mdf.empty or "h3to6_rmse_improve_pct" not in mdf.columns:
            continue
        vals = pd.to_numeric(mdf["h3to6_rmse_improve_pct"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        helped = bool(float(np.mean(vals)) > 0.0)
        q_help_rows.append(
            {
                "model": str(qm),
                "outer_folds": int(vals.size),
                "h3to6_rmse_improve_pct_mean": float(np.mean(vals)),
                "h3to6_rmse_improve_pct_median": float(np.median(vals)),
                "h3to6_rmse_improve_pct_std": float(np.std(vals, ddof=0)),
                "folds_with_positive_improve": int(np.sum(vals > 0.0)),
                "did_quantum_help_gru_h3to6": ("yes" if helped else "no"),
            }
        )
    q_help_df = pd.DataFrame(q_help_rows)

    summary_csv = out_dir / "nested_pseudo_next6_summary.csv"
    folds_csv = out_dir / "nested_pseudo_next6_outer_folds.csv"
    selected_csv = out_dir / "nested_pseudo_next6_selected_params_by_fold.csv"
    q_help_csv = out_dir / "nested_pseudo_next6_quantum_help_summary.csv"
    params_json = out_dir / "nested_pseudo_next6_best_params.json"
    manifest_json = out_dir / "nested_pseudo_next6_manifest.json"
    summary_df.to_csv(summary_csv, index=False)
    fold_df_all.to_csv(folds_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    q_help_df.to_csv(q_help_csv, index=False)
    with params_json.open("w", encoding="utf-8") as f:
        json.dump(best_params_by_model, f, indent=2)

    manifest = {
        "config": asdict(cfg),
        "models": list(models),
        "n_obs": int(n_obs),
        "outer_anchors": [int(x.anchor) for x in outer_folds],
        "summary_csv": str(summary_csv),
        "folds_csv": str(folds_csv),
        "selected_csv": str(selected_csv),
        "quantum_help_csv": str(q_help_csv),
        "best_params_json": str(params_json),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {
        "summary_csv": str(summary_csv),
        "folds_csv": str(folds_csv),
        "selected_csv": str(selected_csv),
        "quantum_help_csv": str(q_help_csv),
        "best_params_json": str(params_json),
        "manifest_json": str(manifest_json),
    }


def run_external_next6_eval(
    cfg: NestedConfig,
    best_params: dict[str, dict[str, object]],
    models: list[str],
    test_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset_bundle(cfg)
    n_obs = int(bundle.surfaces_input.shape[0])
    h = int(cfg.horizon)

    rows: list[dict[str, Any]] = []
    pred_manifest: dict[str, str] = {}
    for model in models:
        m = str(model).strip().lower()
        cand = dict(best_params.get(m, {}))
        if m in {"ensemble", "ensemble_stack_safe"} and "ensemble_state" not in cand:
            continue

        fp_ext = FactorPipeline(
            FactorPipelineConfig(
                lookback=int(cfg.lookback),
                horizon=int(cfg.horizon),
                pca_factors=int(cfg.pca_factors),
                seed=stable_seed(cfg.seed, "external", m),
            )
        ).fit(
            train_surfaces=np.asarray(bundle.surfaces_input, dtype=float),
            full_surfaces=np.asarray(bundle.surfaces_input, dtype=float),
        )
        train_data = fp_ext.build_train_windows(train_end=n_obs)
        x_val, y_val = _tail_train_val_split(train_data.x_train, train_data.y_train, min_val=8, frac=0.2)
        future = fp_ext.build_future_window(anchor=n_obs)
        prep_ext = FoldPreparedData(
            fp=fp_ext,
            x_train=np.asarray(train_data.x_train, dtype=float),
            y_train=np.asarray(train_data.y_train, dtype=float),
            x_val=np.asarray(x_val, dtype=float),
            y_val=np.asarray(y_val, dtype=float),
            x_test=np.asarray(future.x_test, dtype=float),
            base_factor=np.asarray(future.base_factor, dtype=float),
            y_true_surface=np.zeros((h, bundle.surfaces_truth.shape[1]), dtype=float),
            y_true_delta_norm=np.zeros((h, int(cfg.pca_factors)), dtype=float),
        )

        if m in {"ensemble", "ensemble_stack_safe"}:
            ens_state = dict(cand.get("ensemble_state", {}))
            members = [str(x) for x in ens_state.get("members", [])]
            weights = np.asarray(ens_state.get("weights", []), dtype=float)
            member_cfgs = {str(k): dict(v) for k, v in dict(ens_state.get("member_cfgs", {})).items()}
            include_quantum = bool(ens_state.get("include_quantum", False))
            q_model = str(ens_state.get("quantum_model", ""))
            q_cfg = dict(ens_state.get("quantum_cfg", {}))
            member_deltas: list[np.ndarray] = []
            for mm in members:
                cand_mm = dict(member_cfgs.get(mm, {}))
                if include_quantum and mm == q_model:
                    cand_mm = dict(q_cfg)
                pred_delta, _ = _predict_delta_norm(
                    model=mm,
                    candidate=cand_mm,
                    prep=prep_ext,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "external-member", mm),
                )
                member_deltas.append(np.asarray(pred_delta, dtype=float))
            if weights.size != len(member_deltas):
                weights = np.zeros((len(member_deltas),), dtype=float)
                if weights.size > 0:
                    weights[0] = 1.0
            pred_delta = np.einsum("mhf,m->hf", np.asarray(member_deltas, dtype=float), weights, optimize=True)
            pred_surface = prep_ext.fp.reconstruct_surface_from_delta_norm(pred_delta, prep_ext.base_factor)[0]
        else:
            if m == "persistence_surface_naive":
                last = np.asarray(bundle.surfaces_input[n_obs - 1], dtype=float)
                pred_surface = np.repeat(last[None, :], h, axis=0)
            else:
                pred_delta, _ = _predict_delta_norm(
                    model=m,
                    candidate=cand,
                    prep=prep_ext,
                    cfg=cfg,
                    seed=stable_seed(cfg.seed, m, "external-pred"),
                )
                pred_surface = prep_ext.fp.reconstruct_surface_from_delta_norm(pred_delta, prep_ext.base_factor)[0]

        pred_csv = out_dir / f"pred_{m}.csv"
        pred_df = pd.DataFrame(pred_surface, columns=bundle.surface_cols)
        pred_df.insert(0, "Date", pd.Series(test_df["Date"]).astype(str).iloc[:h].tolist())
        pred_df.to_csv(pred_csv, index=False)
        pred_manifest[m] = str(pred_csv)

        cols = [c for c in bundle.surface_cols if c in test_df.columns]
        truth = np.asarray(test_df[cols].iloc[:h].to_numpy(dtype=float), dtype=float)
        pred_use = np.asarray(pred_df[cols].to_numpy(dtype=float), dtype=float)
        block = metric_block(truth, pred_use, eps=1e-8)
        rows.append(
            {
                "model": m,
                "surface_rmse": float(block["surface_rmse"]),
                "surface_mae": float(block["surface_mae"]),
                "surface_mape": float(block["surface_mape"]),
                "horizon_surface_rmse": json.dumps(block["horizon_surface_rmse"]),
                "horizon_surface_mae": json.dumps(block["horizon_surface_mae"]),
                "horizon_surface_mape": json.dumps(block["horizon_surface_mape"]),
                "candidate_json": json.dumps(cand, sort_keys=True, default=str),
            }
        )

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty and "surface_rmse" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("surface_rmse", ascending=True).reset_index(drop=True)
    metrics_csv = out_dir / "external_next6_nested_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    manifest = {
        "config": asdict(cfg),
        "models": list(models),
        "metrics_csv": str(metrics_csv),
        "predictions": pred_manifest,
    }
    manifest_path = out_dir / "external_next6_nested_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"metrics_csv": str(metrics_csv), "manifest_json": str(manifest_path)}
