from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Protocol
import random

import numpy as np
import warnings

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

    class _DummyNN:
        Module = object

    nn = _DummyNN()  # type: ignore[assignment]


Kind = Literal[
    "mlp",
    "gru",
    "lstm",
    "factor_ar",
    "reservoir",
    "hybrid_qrc_fast",
    "qml_classical_reservoir",
    "qml_recursive_classical_reservoir",
    "qml_pca_lstm",
    "qml_convlstm",
    "persistence",
    "persistence_surface_naive",
    "photonic_qrc_feedback",
    "photonic_qrc_no_feedback",
    "persist_qrc_weak",
    "photonic_memory",
    "photonic_memory_no_feedback",
    "quantum_kernel_photonic_memory",
]


class FactorPredictor(Protocol):
    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        ...


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _config_seed(base_seed: int, cfg: dict[str, object]) -> int:
    """Derive deterministic per-config seed without relying on global RNG state."""
    payload = {k: cfg[k] for k in sorted(cfg.keys())}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    salt = int(hashlib.sha1(blob).hexdigest()[:8], 16) % 1_000_003
    return int((int(base_seed) * 1_000_003 + salt) % (2**31 - 1))


def _check_shapes(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
    for name, arr in [("x_train", x_train), ("y_train", y_train), ("x_val", x_val), ("y_val", y_val)]:
        if arr.ndim != 3:
            raise ValueError(f"{name} must be rank-3 [N, T, D], got shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf values.")
    if y_train.shape[2] != y_val.shape[2]:
        raise ValueError("y_train and y_val must have matching factor dimension.")
    if x_train.shape[2] != x_val.shape[2]:
        raise ValueError("Train and validation factor dimensions must match.")
    if x_train.shape[2] < y_train.shape[2]:
        raise ValueError(
            f"x feature dimension ({x_train.shape[2]}) must be >= y factor dimension ({y_train.shape[2]})."
        )


def _internal_train_tune_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    tune_fraction: float = 0.15,
    min_train: int = 8,
    min_tune: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a leakage-safe internal tuning holdout from the training chunk.

    Model fitting uses only the returned train-fit tensors; the tune tensors are
    used for model selection / early stopping. External validation should not be
    used in training.
    """
    n = int(x_train.shape[0])
    n_tune = int(max(min_tune, round(float(tune_fraction) * n)))
    n_tune = int(min(max(1, n_tune), max(1, n - min_train)))
    split = int(n - n_tune)
    if split <= 0 or split >= n:
        raise ValueError(
            "Unable to build internal train/tune split: "
            f"n={n}, split={split}, min_train={min_train}, min_tune={min_tune}."
        )
    return x_train[:split], y_train[:split], x_train[split:], y_train[split:]


@dataclass
class FactorARPredictor:
    model: Ridge
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        if x_seq.shape[2] < self.d_factors:
            raise ValueError(f"Feature dimension mismatch: got {x_seq.shape[2]}, expected >= {self.d_factors}.")
        z_prev = x_seq[:, -1, : self.d_factors].copy()
        out = np.zeros((x_seq.shape[0], self.horizon, self.d_factors), dtype=float)
        for h in range(self.horizon):
            z_prev_s = self.x_scaler.transform(z_prev)
            z_next_s = self.model.predict(z_prev_s)
            z_prev = self.y_scaler.inverse_transform(z_next_s)
            out[:, h, :] = z_prev
        return out


@dataclass
class PersistencePredictor:
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        if x_seq.shape[2] < self.d_factors:
            raise ValueError(f"Feature dimension mismatch: got {x_seq.shape[2]}, expected >= {self.d_factors}.")
        last = x_seq[:, -1, : self.d_factors]
        return np.repeat(last[:, None, :], int(self.horizon), axis=1)


@dataclass
class SurfacePersistencePredictor:
    horizon: int
    d_surface: int

    def fit(self, x_seq: np.ndarray, y_seq: np.ndarray | None = None) -> "SurfacePersistencePredictor":
        _ = np.asarray(x_seq, dtype=float)
        _ = None if y_seq is None else np.asarray(y_seq, dtype=float)
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        if x_seq.shape[2] < self.d_surface:
            raise ValueError(f"Feature dimension mismatch: got {x_seq.shape[2]}, expected >= {self.d_surface}.")
        last = x_seq[:, -1, : self.d_surface]
        return np.repeat(last[:, None, :], int(self.horizon), axis=1)


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GRUNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@dataclass
class MLPResidualPredictor:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        x_flat = x_seq.reshape(x_seq.shape[0], -1)
        x_s = self.x_scaler.transform(x_flat).astype(np.float32)
        with torch.no_grad():
            pred_res_s = self.model(torch.from_numpy(x_s)).cpu().numpy()
        pred_res = self.y_scaler.inverse_transform(pred_res_s).reshape(x_seq.shape[0], self.horizon, self.d_factors)
        return x_seq[:, -1, : self.d_factors][:, None, :] + pred_res


@dataclass
class GRUResidualPredictor:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        n, l, x_dim = x_seq.shape
        x_s = self.x_scaler.transform(x_seq.reshape(-1, x_dim)).reshape(n, l, x_dim).astype(np.float32)
        with torch.no_grad():
            pred_res_s = self.model(torch.from_numpy(x_s)).cpu().numpy()
        pred_res = self.y_scaler.inverse_transform(pred_res_s).reshape(n, self.horizon, self.d_factors)
        return x_seq[:, -1, : self.d_factors][:, None, :] + pred_res


@dataclass
class LSTMResidualPredictor:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        n, l, x_dim = x_seq.shape
        x_s = self.x_scaler.transform(x_seq.reshape(-1, x_dim)).reshape(n, l, x_dim).astype(np.float32)
        with torch.no_grad():
            pred_res_s = self.model(torch.from_numpy(x_s)).cpu().numpy()
        pred_res = self.y_scaler.inverse_transform(pred_res_s).reshape(n, self.horizon, self.d_factors)
        return x_seq[:, -1, : self.d_factors][:, None, :] + pred_res


@dataclass
class QMLClassicalReservoirPredictor:
    model: Ridge
    x_scaler: StandardScaler
    state_scaler: StandardScaler
    w_in: np.ndarray
    w_res: np.ndarray
    b: np.ndarray
    leak: float
    horizon: int
    d_factors: int

    def _states(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        n, l, d = x_seq.shape
        x_s = self.x_scaler.transform(x_seq.reshape(-1, d)).reshape(n, l, d).astype(np.float32)
        phi = evolve_states(
            x_seq=x_s,
            w_in=self.w_in,
            w_res=self.w_res,
            b=self.b,
            leak=float(self.leak),
            washout=0,
            state_feature="last",
            dtype=np.float32,
            input_clip=8.0,
            pre_clip=20.0,
        )
        return np.asarray(phi, dtype=float)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        states = self._states(x_seq)
        states_s = self.state_scaler.transform(states)
        flat = np.asarray(self.model.predict(states_s), dtype=float)
        return flat.reshape(x_seq.shape[0], self.horizon, self.d_factors)


@dataclass
class QMLRecursiveReservoirPredictor:
    model: Ridge
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    w_in: np.ndarray
    w_res: np.ndarray
    b: np.ndarray
    leak: float
    horizon: int
    d_factors: int

    def _run_window_state(self, x_window_scaled: np.ndarray) -> np.ndarray:
        r = int(self.w_res.shape[0])
        state = np.zeros((r,), dtype=np.float32)
        one_minus_leak = np.float32(1.0 - float(self.leak))
        leak_f = np.float32(float(self.leak))
        for t in range(x_window_scaled.shape[0]):
            pre = self.w_in @ x_window_scaled[t] + self.w_res @ state + self.b
            pre = np.tanh(np.clip(pre, -20.0, 20.0)).astype(np.float32, copy=False)
            state = one_minus_leak * state + leak_f * pre
        return state

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        n, l, x_dim = x_seq.shape
        out = np.zeros((n, self.horizon, self.d_factors), dtype=float)
        x_seq = np.asarray(x_seq, dtype=float)
        for i in range(n):
            sample = x_seq[i]
            sample_s = self.x_scaler.transform(sample)
            state = self._run_window_state(sample_s.astype(np.float32))
            extra_last = sample[-1, self.d_factors :] if x_dim > self.d_factors else np.empty((0,), dtype=float)
            for h in range(self.horizon):
                y_step_s = np.asarray(self.model.predict(state[None, :]), dtype=float)
                y_step = np.asarray(self.y_scaler.inverse_transform(y_step_s)[0], dtype=float)
                out[i, h, :] = y_step
                if x_dim > self.d_factors:
                    x_next = np.concatenate([y_step, extra_last], axis=0)
                else:
                    x_next = y_step
                x_next_s = self.x_scaler.transform(x_next[None, :])[0].astype(np.float32)
                pre = self.w_in @ x_next_s + self.w_res @ state + self.b
                pre = np.tanh(np.clip(pre, -20.0, 20.0)).astype(np.float32, copy=False)
                state = (np.float32(1.0 - float(self.leak)) * state) + (np.float32(float(self.leak)) * pre)
        return out


class _QMLPcaLSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, out_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(layers),
            batch_first=True,
            dropout=(0.1 if int(layers) > 1 else 0.0),
        )
        self.fc = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@dataclass
class QMLPcaLSTMPredictor:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        n, l, d = x_seq.shape
        x_s = self.x_scaler.transform(x_seq.reshape(-1, d)).reshape(n, l, d).astype(np.float32)
        with torch.no_grad():
            y_s = self.model(torch.from_numpy(x_s)).cpu().numpy()
        y = self.y_scaler.inverse_transform(y_s).reshape(n, self.horizon, self.d_factors)
        return np.asarray(y, dtype=float)


class _QMLConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_w: int = 3) -> None:
        super().__init__()
        pad_w = int(kernel_w) // 2
        self.hidden_dim = int(hidden_dim)
        self.conv = nn.Conv2d(
            in_channels=int(input_dim) + int(hidden_dim),
            out_channels=4 * int(hidden_dim),
            kernel_size=(1, int(kernel_w)),
            padding=(0, pad_w),
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class _QMLConvLSTMNet(nn.Module):
    def __init__(self, d_factors: int, horizon: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.d_factors = int(d_factors)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.cell = _QMLConvLSTMCell(input_dim=1, hidden_dim=int(hidden_dim), kernel_w=3)
        self.output_conv = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, L, 1, 1, D]
        b, l, d = x.shape
        x5 = x.view(b, l, 1, 1, d)
        h = torch.zeros(b, self.hidden_dim, 1, d, device=x.device)
        c = torch.zeros(b, self.hidden_dim, 1, d, device=x.device)

        for t in range(l):
            h, c = self.cell(x5[:, t], h, c)

        inp = x5[:, -1]
        outs: list[torch.Tensor] = []
        for _ in range(self.horizon):
            h, c = self.cell(inp, h, c)
            out = self.output_conv(h)
            outs.append(out)
            inp = out
        y = torch.stack(outs, dim=1)  # [B, H, 1, 1, D]
        return y.view(b, self.horizon * self.d_factors)


@dataclass
class QMLConvLSTMPredictor:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    horizon: int
    d_factors: int
    input_dim_used: int

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        n, l, _ = x_seq.shape
        d_use = int(self.input_dim_used)
        x_use = np.asarray(x_seq[:, :, :d_use], dtype=float)
        x_s = self.x_scaler.transform(x_use.reshape(-1, d_use)).reshape(n, l, d_use).astype(np.float32)
        with torch.no_grad():
            y_s = self.model(torch.from_numpy(x_s)).cpu().numpy()
        y = self.y_scaler.inverse_transform(y_s).reshape(n, self.horizon, self.d_factors)
        return np.asarray(y, dtype=float)


def _hybrid_qrc_summary_vector(
    x_seq: np.ndarray,
    *,
    feature_mode: str,
    use_cycle_phase: bool,
    use_dt: bool,
    cycle_period: int,
) -> np.ndarray:
    x = np.asarray(x_seq, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x_seq must be [N, L, D], got {x.shape}.")
    n = int(x.shape[0])
    last = x[:, -1, :]
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)

    mode = str(feature_mode).strip().lower()
    parts: list[np.ndarray] = []
    if mode == "last":
        parts.extend([last])
    elif mode == "mean_last":
        parts.extend([last, mean])
    elif mode == "mean_last_std":
        parts.extend([last, mean, std])
    else:
        raise ValueError("hybrid_qrc_fast feature_mode must be one of: last, mean_last, mean_last_std.")

    if bool(use_cycle_phase):
        period = max(1, int(cycle_period))
        phase = np.mod(np.arange(n, dtype=np.int64), period)
        theta = (2.0 * np.pi * (phase.astype(np.float32) / float(period))).astype(np.float32)
        parts.append(np.sin(theta)[:, None])
        parts.append(np.cos(theta)[:, None])

    if bool(use_dt):
        parts.append(np.ones((n, 1), dtype=np.float32))

    out = np.concatenate(parts, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class HybridQRCFastPredictor:
    model: Ridge
    summary_scaler: StandardScaler
    feature_scaler: StandardScaler
    y_scaler: StandardScaler
    w_in: np.ndarray
    u_mix: np.ndarray
    w_photonic: np.ndarray
    a: np.ndarray
    b: np.ndarray
    w_q: np.ndarray
    input_scale: float
    phase_scale: float
    shots: int
    photonic_keep: int
    fast_mode: bool
    residual_scale: float
    residual_clip: float | None
    feature_mode: str
    use_cycle_phase: bool
    use_dt: bool
    cycle_period: int
    horizon: int
    d_factors: int
    modes: int

    def _feature_map(self, summary_s: np.ndarray) -> np.ndarray:
        u = np.asarray(summary_s, dtype=np.float32)
        z = (float(self.input_scale) * u) @ np.asarray(self.w_in, dtype=np.float32)
        phases = float(self.phase_scale) * z
        field = np.exp(1j * phases.astype(np.float32))
        mixed = field @ np.asarray(self.u_mix, dtype=np.float32)
        stacked = np.concatenate([mixed.real.astype(np.float32), mixed.imag.astype(np.float32)], axis=1)

        if int(self.shots) > 0:
            shots = int(max(1, self.shots))
            probs = 0.5 * (np.tanh(stacked) + 1.0)
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            sampled = np.floor(probs * float(shots) + 0.5) / float(shots)
            stacked = (2.0 * sampled - 1.0).astype(np.float32)

        photonic = np.tanh(stacked @ np.asarray(self.w_photonic, dtype=np.float32)).astype(np.float32)
        q1 = photonic @ np.asarray(self.a, dtype=np.float32)
        q2 = photonic @ np.asarray(self.b, dtype=np.float32)
        qmix = (q1 * q2).astype(np.float32)
        quantum = (qmix @ np.asarray(self.w_q, dtype=np.float32)).astype(np.float32)

        if bool(self.fast_mode):
            keep = int(max(0, min(int(self.photonic_keep), photonic.shape[1])))
            if keep > 0:
                feat = np.concatenate([photonic[:, :keep], qmix], axis=1)
            else:
                feat = qmix
        else:
            feat = np.concatenate([photonic, quantum], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def _features(self, x_seq: np.ndarray) -> np.ndarray:
        summary = _hybrid_qrc_summary_vector(
            x_seq,
            feature_mode=str(self.feature_mode),
            use_cycle_phase=bool(self.use_cycle_phase),
            use_dt=bool(self.use_dt),
            cycle_period=int(self.cycle_period),
        )
        summary_s = self.summary_scaler.transform(summary)
        feat = self._feature_map(summary_s)
        return self.feature_scaler.transform(feat)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        n = int(x_seq.shape[0])
        feat_s = self._features(x_seq)
        pred_res_s = np.asarray(self.model.predict(feat_s), dtype=float)
        pred_res = self.y_scaler.inverse_transform(pred_res_s).reshape(n, int(self.horizon), int(self.d_factors))
        if self.residual_clip is not None:
            c = float(abs(self.residual_clip))
            pred_res = np.clip(pred_res, -c, c)
        base = np.asarray(x_seq[:, -1, : self.d_factors], dtype=float)[:, None, :]
        return np.asarray(base + float(self.residual_scale) * pred_res, dtype=float)


def _fit_hybrid_qrc_fast(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    **kwargs: object,
) -> HybridQRCFastPredictor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("hybrid_qrc_fast training requires scikit-learn.")

    n_train, _, x_dim = x_train.shape
    n_val = x_val.shape[0]
    horizon = int(y_train.shape[1])
    d = int(y_train.shape[2])
    if n_train < 8 or n_val < 4:
        raise ValueError(
            f"hybrid_qrc_fast requires at least 8 train and 4 val windows, got n_train={n_train}, n_val={n_val}."
        )

    modes_raw = kwargs.get("hybrid_qrc_modes", kwargs.get("qk_modes", kwargs.get("pqrc_modes", kwargs.get("pqrc_M", 10))))
    modes = int(max(1, int(10 if modes_raw is None else modes_raw)))
    feature_mode = str(kwargs.get("hybrid_qrc_feature_mode", "mean_last_std")).strip().lower()
    use_cycle_phase = bool(kwargs.get("hybrid_qrc_use_cycle_phase", True))
    use_dt = bool(kwargs.get("hybrid_qrc_use_dt", True))
    cycle_period = int(max(2, int(kwargs.get("hybrid_qrc_cycle_period", 19))))
    shots_raw = kwargs.get("hybrid_qrc_shots", kwargs.get("pqrc_shots", 0))
    shots = int(max(0, int(0 if shots_raw is None else shots_raw)))
    in_scale_raw = kwargs.get("hybrid_qrc_input_scale", kwargs.get("pqrc_input_scale", 1.0))
    input_scale = float(1.0 if in_scale_raw is None else in_scale_raw)
    phase_scale = float(kwargs.get("hybrid_qrc_phase_scale", 1.0))
    ridge_raw = kwargs.get("hybrid_qrc_ridge", kwargs.get("pqrc_ridge", 1e-2))
    ridge_default = float(1e-2 if ridge_raw is None else ridge_raw)
    state_dim_raw = kwargs.get("hybrid_qrc_state_dim", max(32, 8 * int(modes)))
    state_dim = int(max(16, int(state_dim_raw)))
    quantum_rank_raw = kwargs.get("hybrid_qrc_quantum_rank", min(16, max(4, state_dim // 2)))
    quantum_rank = int(max(2, min(state_dim, int(quantum_rank_raw))))
    residual_clip_raw = kwargs.get("hybrid_qrc_residual_clip", kwargs.get("qrc_resid_clip", None))
    residual_clip = None if residual_clip_raw is None else float(residual_clip_raw)
    fast_mode = bool(kwargs.get("hybrid_qrc_fast_mode", True))

    summary_train = _hybrid_qrc_summary_vector(
        x_train,
        feature_mode=feature_mode,
        use_cycle_phase=use_cycle_phase,
        use_dt=use_dt,
        cycle_period=cycle_period,
    )
    summary_val = _hybrid_qrc_summary_vector(
        x_val,
        feature_mode=feature_mode,
        use_cycle_phase=use_cycle_phase,
        use_dt=use_dt,
        cycle_period=cycle_period,
    )

    summary_scaler = StandardScaler().fit(summary_train)
    summary_train_s = summary_scaler.transform(summary_train).astype(np.float32)
    summary_val_s = summary_scaler.transform(summary_val).astype(np.float32)

    cfg_seed = _config_seed(
        int(seed),
        {
            "kind": "hybrid_qrc_fast",
            "modes": int(modes),
            "state_dim": int(state_dim),
            "quantum_rank": int(quantum_rank),
            "feature_mode": str(feature_mode),
            "use_cycle_phase": bool(use_cycle_phase),
            "use_dt": bool(use_dt),
            "cycle_period": int(cycle_period),
            "shots": int(shots),
            "input_scale": float(input_scale),
            "phase_scale": float(phase_scale),
        },
    )
    rng = np.random.default_rng(int(cfg_seed))
    w_in = (rng.standard_normal((int(summary_train_s.shape[1]), int(modes))) / np.sqrt(max(1, int(summary_train_s.shape[1])))).astype(np.float32)
    mix_raw = rng.standard_normal((int(modes), int(modes)))
    q, _ = np.linalg.qr(mix_raw)
    u_mix = q.astype(np.float32)
    w_photonic = (rng.standard_normal((2 * int(modes), int(state_dim))) / np.sqrt(max(1, 2 * int(modes)))).astype(np.float32)
    a = (rng.standard_normal((int(state_dim), int(quantum_rank))) / np.sqrt(max(1, int(state_dim)))).astype(np.float32)
    b = (rng.standard_normal((int(state_dim), int(quantum_rank))) / np.sqrt(max(1, int(state_dim)))).astype(np.float32)
    w_q = (rng.standard_normal((int(quantum_rank), int(state_dim))) / np.sqrt(max(1, int(quantum_rank)))).astype(np.float32)
    photonic_keep = int(max(8, min(int(state_dim), int(state_dim) // 4)))

    probe = HybridQRCFastPredictor(
        model=Ridge(alpha=1.0),
        summary_scaler=summary_scaler,
        feature_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        w_in=w_in,
        u_mix=u_mix,
        w_photonic=w_photonic,
        a=a,
        b=b,
        w_q=w_q,
        input_scale=float(input_scale),
        phase_scale=float(phase_scale),
        shots=int(shots),
        photonic_keep=int(photonic_keep),
        fast_mode=bool(fast_mode),
        residual_scale=1.0,
        residual_clip=residual_clip,
        feature_mode=str(feature_mode),
        use_cycle_phase=bool(use_cycle_phase),
        use_dt=bool(use_dt),
        cycle_period=int(cycle_period),
        horizon=int(horizon),
        d_factors=int(d),
        modes=int(modes),
    )
    feat_train = probe._feature_map(summary_train_s)
    feat_val = probe._feature_map(summary_val_s)
    feature_scaler = StandardScaler().fit(feat_train)
    feat_train_s = feature_scaler.transform(feat_train)
    feat_val_s = feature_scaler.transform(feat_val)

    base_train = np.asarray(x_train[:, -1, :d], dtype=float)
    base_val = np.asarray(x_val[:, -1, :d], dtype=float)
    y_train_res = (np.asarray(y_train, dtype=float) - base_train[:, None, :]).reshape(n_train, -1)
    y_val_true = np.asarray(y_val, dtype=float)
    y_scaler = StandardScaler().fit(y_train_res)
    y_train_s = y_scaler.transform(y_train_res)

    alpha_grid = [float(ridge_default), 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]
    alpha_grid = list(dict.fromkeys([float(max(1e-8, a_v)) for a_v in alpha_grid]))

    best_model: Ridge | None = None
    best_val_pred: np.ndarray | None = None
    best_val_mae = float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), random_state=int(seed))
        model.fit(feat_train_s, y_train_s)
        pred_val_res_s = np.asarray(model.predict(feat_val_s), dtype=float)
        pred_val_res = y_scaler.inverse_transform(pred_val_res_s).reshape(n_val, horizon, d)
        if residual_clip is not None:
            c = float(abs(residual_clip))
            pred_val_res = np.clip(pred_val_res, -c, c)
        pred_val = base_val[:, None, :] + pred_val_res
        val_mae = float(np.mean(np.abs(pred_val - y_val_true)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_val_pred = pred_val

    if best_model is None or best_val_pred is None:
        raise RuntimeError("hybrid_qrc_fast selection failed.")

    persistence_val = np.repeat(base_val[:, None, :], horizon, axis=1)
    blend_grid = np.linspace(0.0, 1.0, 11)
    best_alpha = 1.0
    best_blend_mae = float("inf")
    for alpha in blend_grid:
        pred = persistence_val + float(alpha) * (best_val_pred - persistence_val)
        mae = float(np.mean(np.abs(pred - y_val_true)))
        if mae < best_blend_mae:
            best_blend_mae = mae
            best_alpha = float(alpha)

    out = HybridQRCFastPredictor(
        model=best_model,
        summary_scaler=summary_scaler,
        feature_scaler=feature_scaler,
        y_scaler=y_scaler,
        w_in=w_in,
        u_mix=u_mix,
        w_photonic=w_photonic,
        a=a,
        b=b,
        w_q=w_q,
        input_scale=float(input_scale),
        phase_scale=float(phase_scale),
        shots=int(shots),
        photonic_keep=int(photonic_keep),
        fast_mode=bool(fast_mode),
        residual_scale=float(best_alpha),
        residual_clip=residual_clip,
        feature_mode=str(feature_mode),
        use_cycle_phase=bool(use_cycle_phase),
        use_dt=bool(use_dt),
        cycle_period=int(cycle_period),
        horizon=int(horizon),
        d_factors=int(d),
        modes=int(modes),
    )
    setattr(out, "hybrid_qrc_attempt", {"val_mae": float(best_val_mae), "blend_alpha": float(best_alpha)})
    config_hash_payload = {
        "kind": "hybrid_qrc_fast",
        "modes": int(modes),
        "state_dim": int(state_dim),
        "quantum_rank": int(quantum_rank),
        "feature_mode": str(feature_mode),
        "use_cycle_phase": bool(use_cycle_phase),
        "use_dt": bool(use_dt),
        "cycle_period": int(cycle_period),
        "shots": int(shots),
        "input_scale": float(input_scale),
        "phase_scale": float(phase_scale),
        "ridge_alpha": float(ridge_default),
        "residual_clip": None if residual_clip is None else float(residual_clip),
        "qrc_config_hash": str(kwargs.get("qrc_config_hash", "")),
    }
    cfg_hash = hashlib.sha1(
        json.dumps(config_hash_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:12]
    setattr(out, "qrc_config_hash", str(cfg_hash))
    return out


def _sanitize_scaled_sequence(x_seq: np.ndarray, clip: float, dtype: np.dtype = np.float32) -> np.ndarray:
    x = np.asarray(x_seq, dtype=dtype)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=float(clip), neginf=-float(clip), copy=False)
    np.clip(x, -float(clip), float(clip), out=x)
    return x


def _power_iteration_spectral_radius(w_res: np.ndarray, n_iter: int = 25, seed: int = 0) -> float:
    """Estimate spectral radius with power iteration on W (fast, reproducible)."""
    if w_res.ndim != 2 or w_res.shape[0] != w_res.shape[1]:
        raise ValueError(f"w_res must be square, got shape {w_res.shape}.")
    n = int(w_res.shape[0])
    if n == 0:
        return 1.0
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=(n,)).astype(np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return 1.0
    v /= norm
    radius = 1.0
    w = np.asarray(w_res, dtype=np.float64)
    if not np.isfinite(w).all():
        return 1.0
    for _ in range(max(4, int(n_iter))):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            wv = w @ v
        if not np.isfinite(wv).all():
            return 1.0
        radius = float(np.linalg.norm(wv))
        if radius <= 1e-12 or not np.isfinite(radius):
            return 1.0
        v = wv / radius
    return float(max(radius, 1e-12))


def _build_reservoir_weights(
    rng: np.random.Generator,
    input_dim: int,
    n_reservoir: int,
    in_scale: float,
    spectral_radius: float,
    density: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density_v = float(np.clip(density, 1e-3, 1.0))
    w_in = rng.normal(
        0.0,
        float(in_scale) / np.sqrt(max(1, int(input_dim))),
        size=(int(n_reservoir), int(input_dim)),
    ).astype(np.float32)
    if density_v >= 0.999:
        w_res = rng.normal(0.0, 1.0, size=(int(n_reservoir), int(n_reservoir))).astype(np.float32)
    else:
        mask = (rng.random((int(n_reservoir), int(n_reservoir))) < density_v)
        # Keep at least one non-zero outgoing edge per node.
        for i in range(int(n_reservoir)):
            if not bool(mask[i].any()):
                mask[i, int(rng.integers(0, int(n_reservoir)))] = True
        if not bool(mask.any()):
            mask[rng.integers(0, int(n_reservoir)), rng.integers(0, int(n_reservoir))] = True
        w_res = np.zeros((int(n_reservoir), int(n_reservoir)), dtype=np.float32)
        w_res[mask] = rng.normal(0.0, 1.0, size=int(mask.sum())).astype(np.float32)
    w_res *= np.float32(1.0 / np.sqrt(max(1, int(n_reservoir) * density_v)))
    radius_est = _power_iteration_spectral_radius(w_res, n_iter=30, seed=int(seed))
    scale = float(spectral_radius) / max(float(radius_est), 1e-3)
    scale = float(np.clip(scale, 0.05, 10.0))
    w_res *= np.float32(scale)
    np.clip(w_res, -2.0, 2.0, out=w_res)
    b = rng.normal(0.0, 0.05, size=(int(n_reservoir),)).astype(np.float32)
    return w_in, w_res, b


def evolve_states(
    x_seq: np.ndarray,
    w_in: np.ndarray,
    w_res: np.ndarray,
    b: np.ndarray,
    leak: float,
    washout: int,
    state_feature: str = "last_mean",
    activation: str = "tanh",
    dtype: np.dtype = np.float32,
    input_clip: float = 8.0,
    pre_clip: float = 20.0,
) -> np.ndarray:
    """Evolve ESN states.

    Shape contract:
    - x_seq: [N, L, D]
    - w_in: [R, D], w_res: [R, R], b: [R]
    - returns Phi: [N, P] where P depends on `state_feature`:
      * last -> R
      * last_mean -> 2R
      * last_mean_std -> 3R
    """
    x = np.asarray(x_seq, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(f"x_seq must be [N, L, D], got {x.shape}.")
    n, l, d = x.shape
    if n <= 0 or l <= 0 or d <= 0:
        raise ValueError(f"x_seq dimensions must be positive, got {x.shape}.")

    w_in_arr = np.asarray(w_in, dtype=dtype)
    w_res_arr = np.asarray(w_res, dtype=dtype)
    b_arr = np.asarray(b, dtype=dtype)
    if w_in_arr.shape[1] != d:
        raise ValueError(f"w_in shape mismatch: expected second dim {d}, got {w_in_arr.shape}.")
    if w_res_arr.ndim != 2 or w_res_arr.shape[0] != w_res_arr.shape[1]:
        raise ValueError(f"w_res must be square, got {w_res_arr.shape}.")
    if w_in_arr.shape[0] != w_res_arr.shape[0]:
        raise ValueError("w_in and w_res reservoir dimensions must match.")
    if b_arr.shape[0] != w_res_arr.shape[0]:
        raise ValueError(f"b shape mismatch: expected {(w_res_arr.shape[0],)}, got {b_arr.shape}.")
    if activation != "tanh":
        raise ValueError(f"Unsupported activation: {activation}.")
    if state_feature not in {"last", "last_mean", "last_mean_std"}:
        raise ValueError("state_feature must be one of: last, last_mean, last_mean_std.")
    leak_v = float(leak)
    if not (0.0 < leak_v <= 1.0):
        raise ValueError(f"leak must be in (0, 1], got {leak}.")

    x = _sanitize_scaled_sequence(x, clip=float(input_clip), dtype=dtype)
    wash = int(np.clip(int(washout), 0, max(0, l - 1)))
    kept_steps = max(1, l - wash)

    r = int(w_res_arr.shape[0])
    state = np.zeros((n, r), dtype=dtype)
    use_mean = state_feature in {"last_mean", "last_mean_std"}
    use_std = state_feature == "last_mean_std"
    if use_mean:
        state_sum = np.zeros((n, r), dtype=dtype)
    if use_std:
        state_sq_sum = np.zeros((n, r), dtype=dtype)

    w_in_t = w_in_arr.T
    w_res_t = w_res_arr.T
    one_minus_leak = np.float32(1.0 - leak_v)
    leak_f = np.float32(leak_v)
    b_row = b_arr[None, :]

    for t in range(l):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pre = x[:, t, :] @ w_in_t
            pre += state @ w_res_t
        pre += b_row
        if not np.isfinite(pre).all():
            pre = np.nan_to_num(pre, nan=0.0, posinf=float(pre_clip), neginf=-float(pre_clip), copy=False)
        np.clip(pre, -float(pre_clip), float(pre_clip), out=pre)
        np.tanh(pre, out=pre)
        state *= one_minus_leak
        state += leak_f * pre
        if t >= wash and use_mean:
            state_sum += state
            if use_std:
                state_sq_sum += state * state

    if state_feature == "last":
        phi = state
    elif state_feature == "last_mean":
        mean = state_sum / np.float32(kept_steps)
        phi = np.concatenate([state, mean], axis=1)
    else:
        mean = state_sum / np.float32(kept_steps)
        var = np.maximum(0.0, state_sq_sum / np.float32(kept_steps) - mean * mean)
        std = np.sqrt(var, dtype=dtype)
        phi = np.concatenate([state, mean, std], axis=1)
    return np.asarray(phi, dtype=np.float32)


@dataclass
class ReservoirResidualPredictor:
    model: Ridge
    x_scaler: StandardScaler
    phi_scaler: StandardScaler
    y_scaler: StandardScaler
    w_in: np.ndarray
    w_res: np.ndarray
    b: np.ndarray
    leak: float
    washout: int
    state_feature: str
    input_clip: float
    pre_clip: float
    residual_scale: float
    horizon: int
    d_factors: int

    def _features(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        n, l, x_dim = x_seq.shape
        if x_dim != int(self.w_in.shape[1]):
            raise ValueError(
                f"Feature dimension mismatch: x_seq has {x_dim}, reservoir expects {self.w_in.shape[1]}."
            )
        x_s = self.x_scaler.transform(x_seq.reshape(-1, x_dim)).reshape(n, l, x_dim)
        x_s = _sanitize_scaled_sequence(x_s, clip=float(self.input_clip), dtype=np.float32)
        phi = evolve_states(
            x_seq=x_s,
            w_in=self.w_in,
            w_res=self.w_res,
            b=self.b,
            leak=float(self.leak),
            washout=int(self.washout),
            state_feature=str(self.state_feature),
            dtype=np.float32,
            input_clip=float(self.input_clip),
            pre_clip=float(self.pre_clip),
        )
        if not np.isfinite(phi).all():
            raise RuntimeError("Reservoir features contain NaN/Inf.")
        return phi.astype(np.float64, copy=False)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [N, L, D], got {x_seq.shape}.")
        phi = self._features(x_seq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phi_s = self.phi_scaler.transform(phi)
        if not np.isfinite(phi_s).all():
            raise RuntimeError("Reservoir standardized features contain NaN/Inf.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred_res_s = self.model.predict(phi_s)
        if not np.isfinite(pred_res_s).all():
            raise RuntimeError("Reservoir ridge readout produced NaN/Inf residuals.")
        pred_res = self.y_scaler.inverse_transform(pred_res_s).reshape(x_seq.shape[0], self.horizon, self.d_factors)
        return x_seq[:, -1, : self.d_factors][:, None, :] + float(self.residual_scale) * pred_res


def _fit_factor_ar(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> FactorARPredictor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for factor_ar training.")

    _, horizon, d = y_train.shape
    z_prev_hist = x_train[:, :-1, :d].reshape(-1, d)
    z_next_hist = x_train[:, 1:, :d].reshape(-1, d)
    z_prev_bridge = x_train[:, -1, :d]
    z_next_bridge = y_train[:, 0, :]

    z_prev = np.concatenate([z_prev_hist, z_prev_bridge], axis=0)
    z_next = np.concatenate([z_next_hist, z_next_bridge], axis=0)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    z_prev_s = x_scaler.fit_transform(z_prev)
    z_next_s = y_scaler.fit_transform(z_next)

    best: FactorARPredictor | None = None
    best_mae = float("inf")
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0):
        model = Ridge(alpha=float(alpha), random_state=seed)
        model.fit(z_prev_s, z_next_s)
        cand = FactorARPredictor(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            horizon=horizon,
            d_factors=d,
        )
        val_pred = cand.predict(x_val)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        if val_mae < best_mae:
            best_mae = val_mae
            best = cand

    if best is None:
        raise RuntimeError("Failed to fit factor_ar predictor.")
    return best


def _fit_persistence(
    y_train: np.ndarray,
) -> PersistencePredictor:
    _, horizon, d = y_train.shape
    return PersistencePredictor(horizon=int(horizon), d_factors=int(d))


def _fit_surface_persistence(
    y_train: np.ndarray,
) -> SurfacePersistencePredictor:
    _, horizon, d = y_train.shape
    return SurfacePersistencePredictor(horizon=int(horizon), d_surface=int(d))


def _fit_torch_loop(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 80,
    patience: int = 12,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_loss: str = "mae",
) -> tuple[nn.Module, float]:
    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_val_t = torch.from_numpy(x_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_loss_norm = str(train_loss).strip().lower()
    if train_loss_norm not in {"mae", "mse", "rmse"}:
        raise ValueError("train_loss must be one of: mae, mse, rmse.")
    if train_loss_norm == "mae":
        loss_fn: nn.Module = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_raw = float(loss_fn(model(x_val_t), y_val_t).item())
            val = float(np.sqrt(max(val_raw, 0.0))) if train_loss_norm == "rmse" else float(val_raw)
        if val < best_val - 1e-7:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    nn_train_loss: str = "mae",
) -> MLPResidualPredictor:
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        raise RuntimeError("mlp training requires torch and scikit-learn.")

    _set_seed(seed)
    n_train, _, x_dim = x_train.shape
    d = y_train.shape[2]
    n_val = x_val.shape[0]
    horizon = y_train.shape[1]

    base_train = x_train[:, -1, :d]
    base_val = x_val[:, -1, :d]
    y_train_res = (y_train - base_train[:, None, :]).reshape(n_train, -1)
    y_val_res = (y_val - base_val[:, None, :]).reshape(n_val, -1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_flat = x_train.reshape(n_train, -1)
    x_val_flat = x_val.reshape(n_val, -1)
    x_train_s = x_scaler.fit_transform(x_train_flat)
    x_val_s = x_scaler.transform(x_val_flat)
    y_train_s = y_scaler.fit_transform(y_train_res)
    y_val_s = y_scaler.transform(y_val_res)

    model = _MLPNet(input_dim=x_train_s.shape[1], output_dim=y_train_s.shape[1])
    model, _ = _fit_torch_loop(model, x_train_s, y_train_s, x_val_s, y_val_s, train_loss=nn_train_loss)
    model.eval()

    # Track real validation MAE in original factor space for early sanity.
    with torch.no_grad():
        pred_val_s = model(torch.from_numpy(x_val_s.astype(np.float32))).cpu().numpy()
    pred_val_res = y_scaler.inverse_transform(pred_val_s).reshape(n_val, horizon, d)
    val_mae = float(np.mean(np.abs((base_val[:, None, :] + pred_val_res) - y_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("mlp validation produced non-finite MAE.")

    return MLPResidualPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizon=horizon,
        d_factors=d,
    )


def _fit_gru(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    nn_train_loss: str = "mae",
) -> GRUResidualPredictor:
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        raise RuntimeError("gru training requires torch and scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    d = y_train.shape[2]
    n_val = x_val.shape[0]
    horizon = y_train.shape[1]

    base_train = x_train[:, -1, :d]
    base_val = x_val[:, -1, :d]
    y_train_res = (y_train - base_train[:, None, :]).reshape(n_train, -1)
    y_val_res = (y_val - base_val[:, None, :]).reshape(n_val, -1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim)
    y_train_s = y_scaler.fit_transform(y_train_res)
    y_val_s = y_scaler.transform(y_val_res)

    model = _GRUNet(input_dim=x_dim, output_dim=y_train_s.shape[1])
    model, _ = _fit_torch_loop(model, x_train_s, y_train_s, x_val_s, y_val_s, train_loss=nn_train_loss)
    model.eval()

    with torch.no_grad():
        pred_val_s = model(torch.from_numpy(x_val_s.astype(np.float32))).cpu().numpy()
    pred_val_res = y_scaler.inverse_transform(pred_val_s).reshape(n_val, horizon, d)
    val_mae = float(np.mean(np.abs((base_val[:, None, :] + pred_val_res) - y_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("gru validation produced non-finite MAE.")

    return GRUResidualPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizon=horizon,
        d_factors=d,
    )


def _fit_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    nn_train_loss: str = "mae",
) -> LSTMResidualPredictor:
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        raise RuntimeError("lstm training requires torch and scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    d = y_train.shape[2]
    n_val = x_val.shape[0]
    horizon = y_train.shape[1]

    base_train = x_train[:, -1, :d]
    base_val = x_val[:, -1, :d]
    y_train_res = (y_train - base_train[:, None, :]).reshape(n_train, -1)
    y_val_res = (y_val - base_val[:, None, :]).reshape(n_val, -1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim)
    y_train_s = y_scaler.fit_transform(y_train_res)
    y_val_s = y_scaler.transform(y_val_res)

    model = _LSTMNet(input_dim=x_dim, output_dim=y_train_s.shape[1])
    model, _ = _fit_torch_loop(model, x_train_s, y_train_s, x_val_s, y_val_s, train_loss=nn_train_loss)
    model.eval()

    with torch.no_grad():
        pred_val_s = model(torch.from_numpy(x_val_s.astype(np.float32))).cpu().numpy()
    pred_val_res = y_scaler.inverse_transform(pred_val_s).reshape(n_val, horizon, d)
    val_mae = float(np.mean(np.abs((base_val[:, None, :] + pred_val_res) - y_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("lstm validation produced non-finite MAE.")

    return LSTMResidualPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizon=horizon,
        d_factors=d,
    )


def _fit_qml_pca_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    nn_train_loss: str = "mae",
) -> QMLPcaLSTMPredictor:
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        raise RuntimeError("qml_pca_lstm training requires torch and scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    n_val = x_val.shape[0]
    horizon = int(y_train.shape[1])
    d = int(y_train.shape[2])

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim)
    y_train_flat = y_train.reshape(n_train, -1)
    y_val_flat = y_val.reshape(n_val, -1)
    y_train_s = y_scaler.fit_transform(y_train_flat)
    y_val_s = y_scaler.transform(y_val_flat)

    model = _QMLPcaLSTMNet(
        input_dim=int(x_dim),
        hidden_dim=32,
        layers=2,
        out_dim=int(horizon * d),
    )
    model, _ = _fit_torch_loop(
        model,
        x_train_s.astype(np.float32),
        y_train_s.astype(np.float32),
        x_val_s.astype(np.float32),
        y_val_s.astype(np.float32),
        max_epochs=80,
        patience=10,
        batch_size=32,
        lr=1e-3,
        train_loss=nn_train_loss,
    )
    model.eval()

    with torch.no_grad():
        pred_val_s = model(torch.from_numpy(x_val_s.astype(np.float32))).cpu().numpy()
    pred_val = y_scaler.inverse_transform(pred_val_s).reshape(n_val, horizon, d)
    val_mae = float(np.mean(np.abs(pred_val - y_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("qml_pca_lstm validation produced non-finite MAE.")

    return QMLPcaLSTMPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizon=horizon,
        d_factors=d,
    )


def _fit_qml_convlstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    nn_train_loss: str = "mae",
) -> QMLConvLSTMPredictor:
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        raise RuntimeError("qml_convlstm training requires torch and scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    n_val = x_val.shape[0]
    horizon = int(y_train.shape[1])
    d = int(y_train.shape[2])
    if int(d) != int(x_dim):
        # ConvLSTM notebook predicts full surface tensor; here we use factor windows only.
        # Match target dimensionality by feeding factor-only inputs.
        x_train_use = np.asarray(x_train[:, :, :d], dtype=float)
        x_val_use = np.asarray(x_val[:, :, :d], dtype=float)
        x_dim_use = int(d)
    else:
        x_train_use = np.asarray(x_train, dtype=float)
        x_val_use = np.asarray(x_val, dtype=float)
        x_dim_use = int(x_dim)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train_use.reshape(-1, x_dim_use)).reshape(n_train, l, x_dim_use)
    x_val_s = x_scaler.transform(x_val_use.reshape(-1, x_dim_use)).reshape(n_val, l, x_dim_use)
    y_train_flat = y_train.reshape(n_train, -1)
    y_val_flat = y_val.reshape(n_val, -1)
    y_train_s = y_scaler.fit_transform(y_train_flat)
    y_val_s = y_scaler.transform(y_val_flat)

    model = _QMLConvLSTMNet(d_factors=d, horizon=horizon, hidden_dim=24)
    model, _ = _fit_torch_loop(
        model,
        x_train_s.astype(np.float32),
        y_train_s.astype(np.float32),
        x_val_s.astype(np.float32),
        y_val_s.astype(np.float32),
        max_epochs=70,
        patience=10,
        batch_size=16,
        lr=1e-3,
        train_loss=nn_train_loss,
    )
    model.eval()

    with torch.no_grad():
        pred_val_s = model(torch.from_numpy(x_val_s.astype(np.float32))).cpu().numpy()
    pred_val = y_scaler.inverse_transform(pred_val_s).reshape(n_val, horizon, d)
    val_mae = float(np.mean(np.abs(pred_val - y_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("qml_convlstm validation produced non-finite MAE.")

    return QMLConvLSTMPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizon=horizon,
        d_factors=d,
        input_dim_used=int(x_dim_use),
    )


def _fit_qml_classical_reservoir(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> QMLClassicalReservoirPredictor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("qml_classical_reservoir training requires scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    n_val = x_val.shape[0]
    horizon = int(y_train.shape[1])
    d = int(y_train.shape[2])
    if n_train < 8 or n_val < 4:
        raise RuntimeError(
            "qml_classical_reservoir requires at least 8 train and 4 val windows, "
            f"got n_train={n_train}, n_val={n_val}."
        )

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim).astype(np.float32)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim).astype(np.float32)
    y_train_flat = y_train.reshape(n_train, horizon * d)
    y_val_flat = y_val.reshape(n_val, horizon * d)

    attempt_cfgs = [
        {"reservoir_dim": 400, "spectral_radius": 0.6, "sparsity": 0.1, "leak": 0.3, "ridge": 1.0e4},
        {"reservoir_dim": 240, "spectral_radius": 0.7, "sparsity": 0.12, "leak": 0.25, "ridge": 3.0e3},
    ]
    best: dict[str, object] | None = None
    best_val_mae = float("inf")
    for idx, cfg in enumerate(attempt_cfgs):
        cfg_seed = _config_seed(int(seed), {"kind": "qml_classical_reservoir", "idx": int(idx), **cfg})
        rng = np.random.default_rng(int(cfg_seed))
        w_in, w_res, b = _build_reservoir_weights(
            rng=rng,
            input_dim=int(x_dim),
            n_reservoir=int(cfg["reservoir_dim"]),
            in_scale=1.0,
            spectral_radius=float(cfg["spectral_radius"]),
            density=float(cfg["sparsity"]),
            seed=int(cfg_seed),
        )
        st_train = evolve_states(
            x_seq=x_train_s,
            w_in=w_in,
            w_res=w_res,
            b=b,
            leak=float(cfg["leak"]),
            washout=0,
            state_feature="last",
            dtype=np.float32,
            input_clip=8.0,
            pre_clip=20.0,
        ).astype(np.float64)
        st_val = evolve_states(
            x_seq=x_val_s,
            w_in=w_in,
            w_res=w_res,
            b=b,
            leak=float(cfg["leak"]),
            washout=0,
            state_feature="last",
            dtype=np.float32,
            input_clip=8.0,
            pre_clip=20.0,
        ).astype(np.float64)
        st_scaler = StandardScaler()
        st_train_s = st_scaler.fit_transform(st_train)
        st_val_s = st_scaler.transform(st_val)
        model = Ridge(alpha=float(cfg["ridge"]), random_state=int(cfg_seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(st_train_s, y_train_flat)
            pred_val = model.predict(st_val_s)
        val_mae = float(np.mean(np.abs(pred_val - y_val_flat)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best = {
                "model": model,
                "w_in": w_in,
                "w_res": w_res,
                "b": b,
                "state_scaler": st_scaler,
                "leak": float(cfg["leak"]),
            }

    if best is None:
        raise RuntimeError("qml_classical_reservoir training failed.")

    return QMLClassicalReservoirPredictor(
        model=best["model"],  # type: ignore[arg-type]
        x_scaler=x_scaler,
        state_scaler=best["state_scaler"],  # type: ignore[arg-type]
        w_in=best["w_in"],  # type: ignore[arg-type]
        w_res=best["w_res"],  # type: ignore[arg-type]
        b=best["b"],  # type: ignore[arg-type]
        leak=float(best["leak"]),  # type: ignore[arg-type]
        horizon=horizon,
        d_factors=d,
    )


def _fit_qml_recursive_classical_reservoir(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> QMLRecursiveReservoirPredictor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("qml_recursive_classical_reservoir training requires scikit-learn.")

    _set_seed(seed)
    n_train, l, x_dim = x_train.shape
    n_val = x_val.shape[0]
    horizon = int(y_train.shape[1])
    d = int(y_train.shape[2])
    if n_train < 8 or n_val < 4:
        raise RuntimeError(
            "qml_recursive_classical_reservoir requires at least 8 train and 4 val windows, "
            f"got n_train={n_train}, n_val={n_val}."
        )

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim).astype(np.float32)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim).astype(np.float32)
    y_step_train = np.asarray(y_train[:, 0, :], dtype=float)
    y_step_val = np.asarray(y_val[:, 0, :], dtype=float)

    cfg = {"reservoir_dim": 400, "spectral_radius": 0.7, "sparsity": 0.1, "leak": 0.2, "ridge": 1.0e4}
    cfg_seed = _config_seed(int(seed), {"kind": "qml_recursive_classical_reservoir", **cfg})
    rng = np.random.default_rng(int(cfg_seed))
    w_in, w_res, b = _build_reservoir_weights(
        rng=rng,
        input_dim=int(x_dim),
        n_reservoir=int(cfg["reservoir_dim"]),
        in_scale=1.0,
        spectral_radius=float(cfg["spectral_radius"]),
        density=float(cfg["sparsity"]),
        seed=int(cfg_seed),
    )
    st_train = evolve_states(
        x_seq=x_train_s,
        w_in=w_in,
        w_res=w_res,
        b=b,
        leak=float(cfg["leak"]),
        washout=0,
        state_feature="last",
        dtype=np.float32,
        input_clip=8.0,
        pre_clip=20.0,
    ).astype(np.float64)
    st_val = evolve_states(
        x_seq=x_val_s,
        w_in=w_in,
        w_res=w_res,
        b=b,
        leak=float(cfg["leak"]),
        washout=0,
        state_feature="last",
        dtype=np.float32,
        input_clip=8.0,
        pre_clip=20.0,
    ).astype(np.float64)

    y_scaler = StandardScaler()
    y_step_train_s = y_scaler.fit_transform(y_step_train)
    y_step_val_s = y_scaler.transform(y_step_val)
    model = Ridge(alpha=float(cfg["ridge"]), random_state=int(cfg_seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(st_train, y_step_train_s)
        pred_val_s = np.asarray(model.predict(st_val), dtype=float)
    val_mae = float(np.mean(np.abs(y_scaler.inverse_transform(pred_val_s) - y_step_val)))
    if not np.isfinite(val_mae):
        raise RuntimeError("qml_recursive_classical_reservoir validation produced non-finite MAE.")

    return QMLRecursiveReservoirPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        w_in=w_in,
        w_res=w_res,
        b=b,
        leak=float(cfg["leak"]),
        horizon=horizon,
        d_factors=d,
    )


def _fit_reservoir(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_reservoir: int = 160,
    spectral_radius: float = 0.9,
    leak: float = 0.5,
    in_scale: float = 0.8,
    density: float = 0.05,
    state_feature: str = "last_mean",
) -> ReservoirResidualPredictor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("reservoir training requires scikit-learn.")

    n_train, l, x_dim = x_train.shape
    d = y_train.shape[2]
    horizon = y_train.shape[1]
    n_val = x_val.shape[0]
    if n_train < 8 or n_val < 4:
        raise ValueError(
            f"reservoir requires at least 8 train and 4 val windows, got n_train={n_train}, n_val={n_val}."
        )
    if state_feature not in {"last", "last_mean", "last_mean_std"}:
        raise ValueError("state_feature must be one of: last, last_mean, last_mean_std.")

    washout = max(0, int(l // 4))
    input_clip = 8.0
    pre_clip = 20.0

    base_train = x_train[:, -1, :d]
    base_val = x_val[:, -1, :d]
    y_train_res = (y_train - base_train[:, None, :]).reshape(n_train, -1)

    x_scaler = StandardScaler().fit(x_train.reshape(-1, x_dim))
    x_train_s = x_scaler.transform(x_train.reshape(-1, x_dim)).reshape(n_train, l, x_dim)
    x_val_s = x_scaler.transform(x_val.reshape(-1, x_dim)).reshape(n_val, l, x_dim)
    x_train_s = _sanitize_scaled_sequence(x_train_s, clip=input_clip, dtype=np.float32)
    x_val_s = _sanitize_scaled_sequence(x_val_s, clip=input_clip, dtype=np.float32)

    y_scaler = StandardScaler().fit(y_train_res)
    y_train_s = y_scaler.transform(y_train_res)

    rng = np.random.default_rng(int(seed))
    alpha_grid = (1.0, 3.0, 10.0)
    leak_grid = tuple(dict.fromkeys([float(leak), 0.35, 0.65]))
    spectral_grid = tuple(dict.fromkeys([float(spectral_radius), 0.75, 1.05]))
    in_scale_grid = tuple(dict.fromkeys([float(in_scale), 0.55, 1.0]))
    density_grid = tuple(dict.fromkeys([float(density), 0.05, 0.15]))

    best_model: Ridge | None = None
    best_mae = float("inf")
    best_params: dict[str, float] = {}
    best_weights: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_phi_scaler: StandardScaler | None = None
    best_val_residual_pred: np.ndarray | None = None

    for leak_v in leak_grid:
        for sr_v in spectral_grid:
            for in_scale_v in in_scale_grid:
                for density_v in density_grid:
                    w_in, w_res, b = _build_reservoir_weights(
                        rng=rng,
                        input_dim=int(x_dim),
                        n_reservoir=int(n_reservoir),
                        in_scale=float(in_scale_v),
                        spectral_radius=float(sr_v),
                        density=float(density_v),
                        seed=int(seed),
                    )
                    phi_train = evolve_states(
                        x_seq=x_train_s,
                        w_in=w_in,
                        w_res=w_res,
                        b=b,
                        leak=float(leak_v),
                        washout=int(washout),
                        state_feature=str(state_feature),
                        dtype=np.float32,
                        input_clip=input_clip,
                        pre_clip=pre_clip,
                    ).astype(np.float64, copy=False)
                    phi_val = evolve_states(
                        x_seq=x_val_s,
                        w_in=w_in,
                        w_res=w_res,
                        b=b,
                        leak=float(leak_v),
                        washout=int(washout),
                        state_feature=str(state_feature),
                        dtype=np.float32,
                        input_clip=input_clip,
                        pre_clip=pre_clip,
                    ).astype(np.float64, copy=False)
                    if not np.isfinite(phi_train).all() or not np.isfinite(phi_val).all():
                        continue
                    phi_scaler = StandardScaler().fit(phi_train)
                    phi_train_s = phi_scaler.transform(phi_train)
                    phi_val_s = phi_scaler.transform(phi_val)
                    if not np.isfinite(phi_train_s).all() or not np.isfinite(phi_val_s).all():
                        continue

                    candidate_alphas = [1.0]
                    for a in alpha_grid:
                        if a not in candidate_alphas:
                            candidate_alphas.append(float(a))

                    local_best = float("inf")
                    local_passed = False
                    for alpha_v in candidate_alphas:
                        model = Ridge(alpha=float(alpha_v), random_state=seed)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                                model.fit(phi_train_s, y_train_s)
                                pred_val_res_s = model.predict(phi_val_s)
                        if not np.isfinite(pred_val_res_s).all():
                            continue
                        pred_val_res = y_scaler.inverse_transform(pred_val_res_s).reshape(n_val, horizon, d)
                        pred_val = base_val[:, None, :] + pred_val_res
                        val_mae = float(np.mean(np.abs(pred_val - y_val)))
                        if val_mae < local_best:
                            local_best = val_mae
                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_model = model
                            best_params = {
                                "alpha": float(alpha_v),
                                "leak": float(leak_v),
                                "spectral_radius": float(sr_v),
                                "in_scale": float(in_scale_v),
                                "density": float(density_v),
                            }
                            best_weights = (w_in.copy(), w_res.copy(), b.copy())
                            best_phi_scaler = phi_scaler
                            best_val_residual_pred = pred_val_res.copy()
                        if val_mae <= best_mae * 1.02:
                            local_passed = True

                    # Lightweight pruning: skip poor dynamics candidates quickly.
                    if (not local_passed) and np.isfinite(best_mae) and (local_best > best_mae * 1.35):
                        continue

    if best_model is None or best_weights is None or best_phi_scaler is None or best_val_residual_pred is None:
        raise RuntimeError("reservoir selection failed.")

    persistence_val = np.repeat(base_val[:, None, :], int(horizon), axis=1)
    persistence_mae = float(np.mean(np.abs(persistence_val - y_val)))
    residual_scale_grid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=float)
    best_scale = 1.0
    best_blend_mae = float("inf")
    for scale in residual_scale_grid:
        pred = base_val[:, None, :] + float(scale) * best_val_residual_pred
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_blend_mae:
            best_blend_mae = mae
            best_scale = float(scale)
    print(
        "[reservoir] "
        f"val_mae_raw={best_mae:.6f} "
        f"val_mae_scaled={best_blend_mae:.6f} "
        f"persistence_val_mae={persistence_mae:.6f} "
        f"residual_scale={best_scale:.2f} "
        f"washout={washout} state_feature={state_feature} "
        f"alpha={best_params.get('alpha', np.nan):.3g} "
        f"leak={best_params.get('leak', np.nan):.3g} "
        f"radius={best_params.get('spectral_radius', np.nan):.3g} "
        f"in_scale={best_params.get('in_scale', np.nan):.3g} "
        f"density={best_params.get('density', np.nan):.3g}"
    )

    w_in_best, w_res_best, b_best = best_weights
    return ReservoirResidualPredictor(
        model=best_model,
        x_scaler=x_scaler,
        phi_scaler=best_phi_scaler,
        y_scaler=y_scaler,
        w_in=w_in_best,
        w_res=w_res_best,
        b=b_best,
        leak=float(best_params.get("leak", leak)),
        washout=int(washout),
        state_feature=str(state_feature),
        input_clip=float(input_clip),
        pre_clip=float(pre_clip),
        residual_scale=float(best_scale),
        horizon=horizon,
        d_factors=d,
    )


@dataclass
class PersistQRCWeakPredictor:
    """Conservative wrapper: persistence + weak clipped QRC correction."""

    core: FactorPredictor
    alpha: float
    r_clip: float
    floor_eps: float
    d_factors: int
    horizon: int
    qrc_mode: str = "gated_residual"
    qrc_baseline: str = "persistence"

    def __getattr__(self, name: str) -> object:
        return getattr(self.core, name)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        y_q = np.asarray(self.core.predict(x_seq), dtype=np.float64)
        base = np.repeat(np.asarray(x_seq[:, -1, : self.d_factors], dtype=np.float64)[:, None, :], int(self.horizon), axis=1)
        joint_min = float(np.min(np.concatenate([base, y_q], axis=1)))
        shift = float(max(0.0, -joint_min + self.floor_eps))
        base_pos = np.maximum(base + shift, self.floor_eps)
        q_pos = np.maximum(y_q + shift, self.floor_eps)
        log_base = np.log(base_pos)
        log_q = np.log(q_pos)
        r_log = np.clip(log_q - log_base, -abs(float(self.r_clip)), abs(float(self.r_clip)))
        out = np.exp(log_base + float(np.clip(self.alpha, 0.0, 1.0)) * r_log) - shift
        out = np.nan_to_num(out, nan=base, posinf=base + 1.0, neginf=base - 1.0)
        return np.asarray(out, dtype=float)


@dataclass
class PersistenceBlendPredictor:
    """Blend a model prediction toward persistence with a fixed alpha."""

    core: FactorPredictor
    alpha: float
    d_factors: int
    horizon: int

    def __getattr__(self, name: str) -> object:
        return getattr(self.core, name)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        y_hat = np.asarray(self.core.predict(x_seq), dtype=np.float64)
        base = np.repeat(
            np.asarray(x_seq[:, -1, : self.d_factors], dtype=np.float64)[:, None, :],
            int(self.horizon),
            axis=1,
        )
        a = float(np.clip(float(self.alpha), 0.0, 1.0))
        out = base + a * (y_hat - base)
        return np.asarray(np.nan_to_num(out, nan=base, posinf=base + 1.0, neginf=base - 1.0), dtype=float)


def _tune_persistence_blend(
    pred_obj: np.ndarray,
    y_true_obj: np.ndarray,
    persistence_obj: np.ndarray,
    *,
    eps: float = 1e-8,
    shrink: float = 0.85,
    min_gain_abs: float = 0.15,
    min_gain_frac: float = 0.03,
) -> tuple[float, float, float, float]:
    y = np.asarray(y_true_obj, dtype=np.float64)
    pred = np.asarray(pred_obj, dtype=np.float64)
    base = np.asarray(persistence_obj, dtype=np.float64)
    den = np.maximum(np.abs(y), float(max(1e-12, eps)))
    alpha_grid = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)

    base_mae = float(np.mean(np.abs(base - y)))
    base_mape = float(np.mean(np.abs(base - y) / den) * 100.0)
    best_alpha = 0.0
    best_mae = float(base_mae)
    best_mape = float(base_mape)

    for alpha in alpha_grid:
        blend = base + float(alpha) * (pred - base)
        mae = float(np.mean(np.abs(blend - y)))
        mape = float(np.mean(np.abs(blend - y) / den) * 100.0)
        if (mape + 1e-12) < best_mape or (abs(mape - best_mape) <= 1e-12 and mae < best_mae):
            best_alpha = float(alpha)
            best_mae = float(mae)
            best_mape = float(mape)

    min_gain = float(max(float(min_gain_abs), float(min_gain_frac) * base_mape))
    if (base_mape - best_mape) < min_gain:
        best_alpha = 0.0
        best_mae = float(base_mae)
        best_mape = float(base_mape)
    elif best_alpha > 0.0:
        best_alpha = float(np.clip(best_alpha * float(np.clip(shrink, 0.5, 1.0)), 0.0, 1.0))
        blend = base + best_alpha * (pred - base)
        best_mae = float(np.mean(np.abs(blend - y)))
        best_mape = float(np.mean(np.abs(blend - y) / den) * 100.0)

    blend_final = base + float(best_alpha) * (pred - base)
    delta_val = float(np.mean(np.abs(blend_final - base)))
    return float(best_alpha), float(best_mape), float(best_mae), float(delta_val)


def train_classical_forecaster(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    kind: Kind,
    seed: int = 0,
    **kwargs: object,
) -> FactorPredictor:
    """Train a factor-space forecaster and return an object with `.predict(x_seq)`."""
    _check_shapes(x_train, y_train, x_val, y_val)
    use_full_training_raw = kwargs.get("use_full_training", False)
    if isinstance(use_full_training_raw, str):
        use_full_training = str(use_full_training_raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        use_full_training = bool(use_full_training_raw)
    if use_full_training:
        x_fit = x_train
        y_fit = y_train
        x_tune = x_val if int(x_val.shape[0]) > 0 else x_train
        y_tune = y_val if int(y_val.shape[0]) > 0 else y_train
    else:
        x_fit, y_fit, x_tune, y_tune = _internal_train_tune_split(x_train, y_train)

    kind_norm = str(kind).strip().lower()
    if kind_norm not in {
        "mlp",
        "gru",
        "lstm",
        "factor_ar",
        "reservoir",
        "hybrid_qrc_fast",
        "qml_classical_reservoir",
        "qml_recursive_classical_reservoir",
        "qml_pca_lstm",
        "qml_convlstm",
        "persistence",
        "persistence_surface_naive",
        "photonic_qrc_feedback",
        "photonic_qrc_no_feedback",
        "persist_qrc_weak",
        "photonic_memory",
        "photonic_memory_no_feedback",
        "quantum_kernel_photonic_memory",
    }:
        raise ValueError(
            "kind must be one of: mlp, gru, lstm, factor_ar, reservoir, hybrid_qrc_fast, qml_classical_reservoir, "
            "qml_recursive_classical_reservoir, qml_pca_lstm, qml_convlstm, persistence, persistence_surface_naive, "
            "photonic_qrc_feedback, photonic_qrc_no_feedback, persist_qrc_weak, "
            "photonic_memory, photonic_memory_no_feedback, quantum_kernel_photonic_memory."
        )

    nn_train_loss = str(kwargs.get("nn_train_loss", "mae")).strip().lower()
    if nn_train_loss not in {"mae", "mse", "rmse"}:
        raise ValueError("nn_train_loss must be one of: mae, mse, rmse.")

    if kind_norm == "factor_ar":
        return _fit_factor_ar(x_fit, y_fit, x_tune, y_tune, seed=seed)
    if kind_norm == "persistence":
        return _fit_persistence(y_fit)
    if kind_norm == "persistence_surface_naive":
        return _fit_surface_persistence(y_fit)
    if kind_norm == "mlp":
        return _fit_mlp(x_fit, y_fit, x_tune, y_tune, seed=seed, nn_train_loss=nn_train_loss)
    if kind_norm == "gru":
        return _fit_gru(x_fit, y_fit, x_tune, y_tune, seed=seed, nn_train_loss=nn_train_loss)
    if kind_norm == "lstm":
        return _fit_lstm(x_fit, y_fit, x_tune, y_tune, seed=seed, nn_train_loss=nn_train_loss)
    if kind_norm == "reservoir":
        return _fit_reservoir(x_fit, y_fit, x_tune, y_tune, seed=seed)
    if kind_norm == "hybrid_qrc_fast":
        return _fit_hybrid_qrc_fast(x_fit, y_fit, x_tune, y_tune, seed=seed, **kwargs)
    if kind_norm == "qml_classical_reservoir":
        return _fit_qml_classical_reservoir(x_fit, y_fit, x_tune, y_tune, seed=seed)
    if kind_norm == "qml_recursive_classical_reservoir":
        return _fit_qml_recursive_classical_reservoir(x_fit, y_fit, x_tune, y_tune, seed=seed)
    if kind_norm == "qml_pca_lstm":
        return _fit_qml_pca_lstm(x_fit, y_fit, x_tune, y_tune, seed=seed, nn_train_loss=nn_train_loss)
    if kind_norm == "qml_convlstm":
        return _fit_qml_convlstm(x_fit, y_fit, x_tune, y_tune, seed=seed, nn_train_loss=nn_train_loss)
    if kind_norm == "quantum_kernel_photonic_memory":
        from .models.quantum_kernel_photmem import QuantumKernelPhotMemParams, train_quantum_kernel_photmem

        qk_feature = str(kwargs.get("qk_feature", kwargs.get("pqrc_feature", "clickprob"))).strip().lower()
        if qk_feature not in {"clickprob", "coincidence"}:
            raise ValueError("qk_feature must be one of: clickprob, coincidence.")
        qk_state_dim_mode = str(kwargs.get("qk_state_dim_mode", "mean_last_std")).strip().lower()
        if qk_state_dim_mode not in {"mean_last", "mean_last_std"}:
            raise ValueError("qk_state_dim_mode must be one of: mean_last, mean_last_std.")
        qk_horizon_mode = str(kwargs.get("qk_horizon_mode", "single14")).strip().lower()
        if qk_horizon_mode not in {"single14", "all"}:
            raise ValueError("qk_horizon_mode must be one of: single14, all.")
        qk_target_space = str(kwargs.get("qk_target_space", "factor")).strip().lower()
        if qk_target_space not in {"factor", "surface"}:
            raise ValueError("qk_target_space must be one of: factor, surface.")

        qk_ridge_grid_raw = kwargs.get("qk_ridge_grid", None)
        qk_ridge_grid: list[float] | str | None
        if isinstance(qk_ridge_grid_raw, str):
            qk_ridge_grid = str(qk_ridge_grid_raw)
        elif qk_ridge_grid_raw is None:
            qk_ridge_grid = None
        elif isinstance(qk_ridge_grid_raw, (list, tuple, np.ndarray)):
            qk_ridge_grid = [float(v) for v in qk_ridge_grid_raw]
        else:
            qk_ridge_grid = [float(qk_ridge_grid_raw)]
        qk_gamma_grid_raw = kwargs.get("qk_gamma_grid", None)
        qk_gamma_grid: list[float] | str | None
        if isinstance(qk_gamma_grid_raw, str):
            qk_gamma_grid = str(qk_gamma_grid_raw)
        elif qk_gamma_grid_raw is None:
            qk_gamma_grid = None
        elif isinstance(qk_gamma_grid_raw, (list, tuple, np.ndarray)):
            qk_gamma_grid = [float(v) for v in qk_gamma_grid_raw]
        else:
            qk_gamma_grid = [float(qk_gamma_grid_raw)]

        qk_modes_raw = kwargs.get("qk_modes", kwargs.get("pqrc_modes", kwargs.get("pqrc_M", 8)))
        qk_in_scale_raw = kwargs.get("qk_in_scale", kwargs.get("pqrc_input_scale", 0.5))
        qk_shots_raw = kwargs.get("qk_shots", kwargs.get("pqrc_shots", 32))
        qk_gain_raw = kwargs.get("qk_gain", kwargs.get("pqrc_gain", 0.5))
        qk_seed_raw = kwargs.get("qk_seed", kwargs.get("seed", seed))
        qk_gamma_raw = kwargs.get("qk_gamma", 0.3)
        qk_ridge_raw = kwargs.get("qk_ridge", kwargs.get("pqrc_ridge", 1e-2))
        qk_resid_clip = kwargs.get("qk_resid_clip", kwargs.get("qrc_resid_clip", None))
        qk_aux_blend_raw = kwargs.get("qk_aux_blend", 1)
        qk_aux_kind_raw = kwargs.get("qk_aux_kind", "photonic_qrc_feedback")
        qk_blend_objective_raw = kwargs.get("qk_blend_objective", "mape")
        qk_blend_max_weight_raw = kwargs.get("qk_blend_max_weight", 1.0)
        qk_blend_grid_points_raw = kwargs.get("qk_blend_grid_points", 31)
        qk_horizon_mode_eff = str(qk_horizon_mode)
        if int(y_tune.shape[1]) <= 6 and qk_horizon_mode_eff == "single14":
            qk_horizon_mode_eff = "all"
        qk_aux_kind_eff = str(qk_aux_kind_raw).strip().lower()
        if qk_aux_kind_eff not in {"photonic_qrc_feedback", "reservoir", "none"}:
            raise ValueError("qk_aux_kind must be one of: photonic_qrc_feedback, reservoir, none.")
        params = QuantumKernelPhotMemParams(
            gamma=float(0.3 if qk_gamma_raw is None else qk_gamma_raw),
            gamma_grid=qk_gamma_grid,
            ridge=float(1e-2 if qk_ridge_raw is None else qk_ridge_raw),
            ridge_grid=qk_ridge_grid,
            feature=str(qk_feature),
            shots=int(32 if qk_shots_raw is None else qk_shots_raw),
            gain=float(0.5 if qk_gain_raw is None else qk_gain_raw),
            modes=int(8 if qk_modes_raw is None else qk_modes_raw),
            in_scale=float(0.5 if qk_in_scale_raw is None else qk_in_scale_raw),
            seed=int(seed if qk_seed_raw is None else qk_seed_raw),
            state_dim_mode=str(qk_state_dim_mode),
            horizon_mode=str(qk_horizon_mode_eff),
            target_space=str(qk_target_space),
            residual_clip=(None if qk_resid_clip is None else float(qk_resid_clip)),
            aux_blend=bool(int(qk_aux_blend_raw)),
            aux_kind=str(qk_aux_kind_eff),
            blend_objective=str(qk_blend_objective_raw),
            blend_max_weight=float(qk_blend_max_weight_raw),
            blend_grid_points=int(qk_blend_grid_points_raw),
        ).normalized()
        model = train_quantum_kernel_photmem(
            X_seq_train=x_fit,
            Y_train=y_fit,
            X_seq_val=x_tune,
            Y_val=y_tune,
            params=params,
            horizon=int(y_tune.shape[1]),
            d_out=int(y_tune.shape[2]),
        )
        if hasattr(model, "__dict__"):
            setattr(
                model,
                "qk_attempt",
                {
                    "horizon_mode": str(qk_horizon_mode_eff),
                    "aux_kind": str(qk_aux_kind_eff),
                    "objective_horizon": int(max(1, min(6, y_tune.shape[1]))),
                },
            )
        config_hash_payload = {
            "kind": str(kind_norm),
            "qk_gamma": float(params.gamma),
            "qk_gamma_grid": (
                str(params.gamma_grid)
                if isinstance(params.gamma_grid, str)
                else list(params.gamma_grid if isinstance(params.gamma_grid, (list, tuple)) else [])
            ),
            "qk_ridge": float(params.ridge),
            "qk_ridge_grid": (
                str(params.ridge_grid)
                if isinstance(params.ridge_grid, str)
                else list(params.ridge_grid if isinstance(params.ridge_grid, (list, tuple)) else [])
            ),
            "qk_feature": str(params.feature),
            "qk_shots": int(params.shots),
            "qk_gain": float(params.gain),
            "qk_modes": int(params.modes),
            "qk_in_scale": float(params.in_scale),
            "qk_seed": int(params.seed),
            "qk_state_dim_mode": str(params.state_dim_mode),
            "qk_horizon_mode": str(params.horizon_mode),
            "qk_target_space": str(params.target_space),
            "qk_resid_clip": None if params.residual_clip is None else float(params.residual_clip),
            "qk_aux_blend": bool(params.aux_blend),
            "qk_aux_kind": str(params.aux_kind),
            "qk_blend_objective": str(params.blend_objective),
            "qk_blend_max_weight": float(params.blend_max_weight),
            "qk_blend_grid_points": int(params.blend_grid_points),
            "qrc_config_hash": str(kwargs.get("qrc_config_hash", "")),
        }
        cfg_hash = hashlib.sha1(
            json.dumps(config_hash_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:12]
        setattr(model, "qrc_config_hash", str(cfg_hash))
        return model
    if kind_norm in {"photonic_memory", "photonic_memory_no_feedback"}:
        from .models.photonic_memory import PhotonicMemoryParams, train_photonic_memory_forecaster

        feature = str(kwargs.get("pqrc_feature", "coincidence")).strip().lower()
        if feature not in {"coincidence", "clickprob", "pseudo_coincidence"}:
            raise ValueError("pqrc_feature must be one of: coincidence, clickprob, pseudo_coincidence.")
        gain_raw = kwargs.get("pqrc_gain", 0.5)
        gain_val = float(0.5 if gain_raw is None else gain_raw)
        feedback_enabled = bool(kind_norm == "photonic_memory")
        if not feedback_enabled:
            gain_val = 0.0
        qrc_target = str(kwargs.get("qrc_target", "delta")).strip().lower()
        tau_default = 0.5 if qrc_target == "norm_resid" else (0.02 if qrc_target == "log_return" else 0.05)
        tau_val = kwargs.get("qrc_gate_tau", kwargs.get("tau", tau_default))
        if tau_val is None:
            tau_val = tau_default
        if qrc_target == "norm_resid" and float(tau_val) == 0.05:
            tau_val = 0.5
        if qrc_target == "log_return" and float(tau_val) == 0.05:
            tau_val = 0.02
        ridge_raw = kwargs.get("pqrc_ridge", 1e-2)
        input_scale_raw = kwargs.get("pqrc_input_scale", 0.75)
        factor_cap_raw = int(kwargs.get("pqrc_factor_cap", 0))
        base_params = {
            "modes": int(kwargs.get("pqrc_modes", kwargs.get("pqrc_M", 12))),
            "n_photons": int(kwargs.get("pqrc_nphotons", kwargs.get("pqrc_Nph", 2))),
            "budget": int(kwargs.get("pqrc_budget", 32)),
            "gain": float(gain_val),
            "feature": feature,  # type: ignore[dict-item]
            "shots": int(kwargs.get("pqrc_shots", 32)),
            "ridge_alpha": float(1e-2 if ridge_raw is None else ridge_raw),
            "qrc_mode": str(kwargs.get("qrc_mode", "auto")),
            "qrc_target": str(qrc_target),
            "qrc_baseline": str(kwargs.get("qrc_baseline", "persistence")),
            "target_transform": str(kwargs.get("target_transform", "log")),
            "y_floor_mode": str(kwargs.get("y_floor_mode", "train_p001")),
            "y_floor_value": kwargs.get("y_floor_value", None),
            "tau": float(tau_val),
            "resid_clip": kwargs.get("qrc_resid_clip", None),
            "residvar_penalty": float(kwargs.get("qrc_residvar_penalty", kwargs.get("qrc_residual_penalty", 0.1))),
            "feat_norm": str(kwargs.get("qrc_feat_norm", "standard")),
            "in_pca": int(kwargs.get("pqrc_in_pca", 4)),
            "factor_cap": int(factor_cap_raw),
            "input_scale": float(0.75 if input_scale_raw is None else input_scale_raw),
            "min_residual_scale": float(kwargs.get("pqrc_min_residual_scale", 0.05)),
            "higher_order": int(kwargs.get("pqrc_higher_order", 1)),
            "budget_alloc": str(kwargs.get("pqrc_budget_alloc", "uniform")),
            "seed": int(seed),
            "feedback_enabled": bool(feedback_enabled),
        }
        config_hash_payload = dict(base_params)
        config_hash_payload["kind"] = str(kind_norm)
        config_hash_payload["qrc_config_hash"] = str(kwargs.get("qrc_config_hash", ""))
        cfg_hash = hashlib.sha1(
            json.dumps(config_hash_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:12]

        y_dim = int(y_tune.shape[2])
        horizon = int(y_tune.shape[1])
        base_mode = str(base_params.get("qrc_baseline", "persistence")).strip().lower()
        if base_mode == "seasonal13":
            n, l, _ = x_tune.shape
            persistence_val = np.zeros((n, horizon, y_dim), dtype=float)
            for h_idx in range(horizon):
                step = h_idx + 1
                lag = 13 - step
                if lag <= 0:
                    persistence_val[:, h_idx, :] = x_tune[:, -1, :y_dim]
                elif lag < l:
                    persistence_val[:, h_idx, :] = x_tune[:, -(lag + 1), :y_dim]
                else:
                    persistence_val[:, h_idx, :] = x_tune[:, -1, :y_dim]
        else:
            persistence_val = np.repeat(x_tune[:, -1, :y_dim][:, None, :], horizon, axis=1)
        h_obj = int(max(1, min(6, horizon)))
        y_tune_obj = np.asarray(y_tune[:, :h_obj, :], dtype=float)
        persistence_obj = np.asarray(persistence_val[:, :h_obj, :], dtype=float)
        persistence_val_mae = float(np.mean(np.abs(persistence_obj - y_tune_obj)))
        denom = np.maximum(np.abs(y_tune_obj), 1e-8)
        persistence_val_mape = float(np.mean(np.abs(persistence_obj - y_tune_obj) / denom) * 100.0)

        attempt_grid: list[dict[str, float | int | str]] = [
            {
                "input_scale": float(base_params["input_scale"]),
                "min_residual_scale": float(base_params["min_residual_scale"]),
                "gain": float(base_params["gain"]),
                "feature": str(base_params["feature"]),
                "shots": int(base_params["shots"]),
                "ridge_alpha": float(base_params["ridge_alpha"]),
                "factor_cap": int(base_params["factor_cap"]),
            }
        ]
        if feedback_enabled:
            base_gain = float(base_params["gain"])
            base_input = float(base_params["input_scale"])
            base_resid = float(base_params["min_residual_scale"])
            base_feature = str(base_params["feature"])
            base_shots = int(base_params["shots"])
            base_ridge = float(base_params["ridge_alpha"])
            attempt_grid.extend(
                [
                    {
                        "input_scale": max(0.75, base_input),
                        "min_residual_scale": max(0.05, base_resid),
                        "gain": max(0.15, min(base_gain, 0.35)),
                        "feature": base_feature,
                        "shots": base_shots,
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(1.0, base_input),
                        "min_residual_scale": max(0.10, base_resid),
                        "gain": max(base_gain, 0.35),
                        "feature": base_feature,
                        "shots": base_shots,
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(1.5, base_input),
                        "min_residual_scale": max(0.15, base_resid),
                        "gain": max(base_gain, 0.75),
                        "feature": base_feature,
                        "shots": base_shots,
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(2.0, base_input),
                        "min_residual_scale": max(0.20, base_resid),
                        "gain": max(base_gain, 1.0),
                        "feature": base_feature,
                        "shots": base_shots,
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(1.0, base_input),
                        "min_residual_scale": max(0.10, base_resid),
                        "gain": max(base_gain, 0.35),
                        "feature": base_feature,
                        "shots": max(0, min(64, base_shots)),
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(1.25, base_input),
                        "min_residual_scale": max(0.10, base_resid),
                        "gain": max(base_gain, 0.50),
                        "feature": base_feature,
                        "shots": 0,
                        "ridge_alpha": base_ridge,
                    },
                    {
                        "input_scale": max(1.0, base_input),
                        "min_residual_scale": max(0.10, base_resid),
                        "gain": 0.30,
                        "feature": ("clickprob" if base_feature == "coincidence" else "coincidence"),
                        "shots": base_shots,
                        "ridge_alpha": max(1e-4, min(1e-1, base_ridge)),
                    },
                    {
                        "input_scale": max(1.0, base_input),
                        "min_residual_scale": max(0.10, base_resid),
                        "gain": 0.40,
                        "feature": ("clickprob" if base_feature == "coincidence" else "coincidence"),
                        "shots": base_shots,
                        "ridge_alpha": max(1e-4, min(1e-1, base_ridge)),
                    },
                ]
            )
        factor_cap_req = int(base_params["factor_cap"])
        if factor_cap_req <= 0:
            cap_candidates = [0]
        else:
            cap_candidates = [int(max(1, min(factor_cap_req, y_dim)))]
        expanded_attempts: list[dict[str, float | int | str]] = []
        for attempt in attempt_grid:
            for cap in cap_candidates:
                attempt_i = dict(attempt)
                attempt_i["factor_cap"] = int(cap)
                expanded_attempts.append(attempt_i)
        attempt_grid = expanded_attempts
        dedup_attempts: list[dict[str, float | int | str]] = []
        seen_attempt_keys: set[tuple[float, float, float, str, int, float, int]] = set()
        for attempt in attempt_grid:
            key = (
                float(attempt["input_scale"]),
                float(attempt["min_residual_scale"]),
                float(attempt["gain"]),
                str(attempt["feature"]),
                int(attempt["shots"]),
                float(attempt["ridge_alpha"]),
                int(attempt["factor_cap"]),
            )
            if key in seen_attempt_keys:
                continue
            seen_attempt_keys.add(key)
            dedup_attempts.append(attempt)
        attempt_grid = dedup_attempts

        candidate_rows: list[dict[str, float | int | str | FactorPredictor]] = []
        for idx, attempt in enumerate(attempt_grid):
            attempt_seed = _config_seed(
                int(seed),
                {
                    "kind": str(kind_norm),
                    "modes": int(base_params["modes"]),
                    "n_photons": int(base_params["n_photons"]),
                    "budget": int(base_params["budget"]),
                    "qrc_mode": str(base_params["qrc_mode"]),
                    "qrc_target": str(base_params["qrc_target"]),
                    "qrc_baseline": str(base_params["qrc_baseline"]),
                    "input_scale": float(attempt["input_scale"]),
                    "min_residual_scale": float(attempt["min_residual_scale"]),
                    "gain": float(attempt["gain"]),
                    "feature": str(attempt["feature"]),
                    "shots": int(attempt["shots"]),
                    "ridge_alpha": float(attempt["ridge_alpha"]),
                    "factor_cap": int(attempt["factor_cap"]),
                },
            )
            params_i = PhotonicMemoryParams(
                **{
                    **base_params,
                    "input_scale": float(attempt["input_scale"]),
                    "min_residual_scale": float(attempt["min_residual_scale"]),
                    "gain": float(attempt["gain"]),
                    "feature": str(attempt["feature"]),
                    "shots": int(attempt["shots"]),
                    "ridge_alpha": float(attempt["ridge_alpha"]),
                    "factor_cap": int(attempt["factor_cap"]),
                    "seed": int(attempt_seed),
                }
            )
            model_i = train_photonic_memory_forecaster(
                x_train=x_fit,
                y_train=y_fit,
                x_val=x_tune,
                y_val=y_tune,
                params=params_i,
                d_factors=int(y_tune.shape[2]),
                horizon=int(y_tune.shape[1]),
            )
            setattr(model_i, "qrc_config_hash", str(cfg_hash))
            pred_val = np.asarray(model_i.predict(x_tune), dtype=float)
            pred_obj = np.asarray(pred_val[:, :h_obj, :], dtype=float)
            val_mae = float(np.mean(np.abs(pred_obj - y_tune_obj)))
            val_mape = float(np.mean(np.abs(pred_obj - y_tune_obj) / denom) * 100.0)
            delta_val = float(np.mean(np.abs(pred_obj - persistence_obj)))
            candidate_rows.append(
                {
                    "model": model_i,
                    "val_mae": float(val_mae),
                    "val_mape": float(val_mape),
                    "delta_val": float(delta_val),
                    "idx": int(idx),
                    "input_scale": float(attempt["input_scale"]),
                    "min_residual_scale": float(attempt["min_residual_scale"]),
                    "gain": float(attempt["gain"]),
                    "feature": str(attempt["feature"]),
                    "shots": int(attempt["shots"]),
                    "ridge_alpha": float(attempt["ridge_alpha"]),
                    "factor_cap": int(attempt["factor_cap"]),
                }
            )
        best_model: FactorPredictor | None = None
        best_val_mae = float("inf")
        best_val_mape = float("inf")
        best_delta = 0.0
        selected_attempt: dict[str, float | int | str | FactorPredictor] | None = None
        if feedback_enabled and candidate_rows:
            selected_attempt = min(
                candidate_rows,
                key=lambda r: (float(r["val_mape"]), float(r["val_mae"]), -float(r["delta_val"])),
            )
        elif candidate_rows:
            selected_attempt = min(
                candidate_rows,
                key=lambda r: (float(r["val_mae"]), float(r["val_mape"])),
            )
        if selected_attempt is not None:
            best_model = selected_attempt["model"]  # type: ignore[assignment]
            best_val_mae = float(selected_attempt["val_mae"])
            best_val_mape = float(selected_attempt["val_mape"])
            best_delta = float(selected_attempt["delta_val"])
            if hasattr(best_model, "__dict__"):
                setattr(
                    best_model,
                    "photonic_memory_attempt",
                    {
                        "idx": int(selected_attempt["idx"]),
                        "input_scale": float(selected_attempt["input_scale"]),
                        "min_residual_scale": float(selected_attempt["min_residual_scale"]),
                        "gain": float(selected_attempt["gain"]),
                        "feature": str(selected_attempt["feature"]),
                        "shots": int(selected_attempt["shots"]),
                        "ridge_alpha": float(selected_attempt["ridge_alpha"]),
                        "factor_cap": int(selected_attempt["factor_cap"]),
                        "val_mape": float(selected_attempt["val_mape"]),
                        "val_mae": float(selected_attempt["val_mae"]),
                        "delta_vs_persistence": float(selected_attempt["delta_val"]),
                        "objective_horizon": int(h_obj),
                    },
                )
        if best_model is None:
            raise RuntimeError(f"{kind_norm}: failed to train photonic_memory.")
        if feedback_enabled and (best_val_mape > persistence_val_mape + 1e-10):
            warnings.warn(
                f"{kind_norm}: best tuned feedback model did not beat persistence on validation "
                f"(best_mape={best_val_mape:.6f}, persistence_mape={persistence_val_mape:.6f}, "
                f"delta_vs_persistence={best_delta:.3e}).",
                RuntimeWarning,
            )
        if feedback_enabled and best_delta <= 1e-8:
            warnings.warn(
                f"{kind_norm}: tuned feedback model remains near persistence on validation "
                f"(delta_vs_persistence={best_delta:.3e}).",
                RuntimeWarning,
            )
        if feedback_enabled and hasattr(best_model, "photonic_memory_attempt"):
            att = getattr(best_model, "photonic_memory_attempt", {})
            try:
                print(
                    f"[{kind_norm}] selected_attempt idx={int(att.get('idx', -1))} "
                    f"feature={att.get('feature', base_params['feature'])} gain={float(att.get('gain', 0.0)):.3g} "
                    f"input_scale={float(att.get('input_scale', 0.0)):.3g} shots={int(att.get('shots', 0))} "
                    f"factor_cap={int(att.get('factor_cap', base_params['factor_cap']))} "
                    f"ridge={float(att.get('ridge_alpha', 0.0)):.3g} "
                    f"min_residual_scale={float(att.get('min_residual_scale', 0.0)):.3g} "
                    f"val_mape={float(att.get('val_mape', float('nan'))):.6f} "
                    f"delta_vs_persistence={float(att.get('delta_vs_persistence', 0.0)):.3e}"
                )
            except Exception:
                pass
        return best_model
    if kind_norm in {"photonic_qrc_feedback", "photonic_qrc_no_feedback", "persist_qrc_weak"}:
        from .models.photonic_qrc import PhotonicQRCParams, train_photonic_qrc_forecaster

        feature = str(kwargs.get("pqrc_feature", "coincidence")).strip().lower()
        if feature not in {"coincidence", "clickprob", "pseudo_coincidence"}:
            raise ValueError("pqrc_feature must be one of: coincidence, clickprob, pseudo_coincidence.")
        gain_raw = kwargs.get("pqrc_gain", 0.5)
        gain_val = float(0.5 if gain_raw is None else gain_raw)
        feedback_enabled = bool(kind_norm in {"photonic_qrc_feedback", "persist_qrc_weak"})
        if not feedback_enabled:
            gain_val = 0.0
        qrc_target = str(kwargs.get("qrc_target", "delta")).strip().lower()
        tau_default = 0.5 if qrc_target == "norm_resid" else (0.02 if qrc_target == "log_return" else 0.05)
        tau_val = kwargs.get("qrc_gate_tau", kwargs.get("tau", tau_default))
        if tau_val is None:
            tau_val = tau_default
        if qrc_target == "norm_resid" and float(tau_val) == 0.05:
            tau_val = 0.5
        if qrc_target == "log_return" and float(tau_val) == 0.05:
            tau_val = 0.02
        ridge_raw = kwargs.get("pqrc_ridge", 1e-2)
        input_scale_raw = kwargs.get("pqrc_input_scale", 0.75)
        base_params = {
            "modes": int(kwargs.get("pqrc_modes", kwargs.get("pqrc_M", 12))),
            "n_photons": int(kwargs.get("pqrc_nphotons", kwargs.get("pqrc_Nph", 2))),
            "budget": int(kwargs.get("pqrc_budget", 32)),
            "gain": float(gain_val),
            "feature": feature,  # type: ignore[dict-item]
            "shots": int(max(1, int(kwargs.get("pqrc_shots", 32)))),
            "pseudocount": float(kwargs.get("pqrc_pseudocount", 0.25)),
            "ridge_alpha": float(1e-2 if ridge_raw is None else ridge_raw),
            "qrc_mode": str(kwargs.get("qrc_mode", "auto")),
            "qrc_target": str(qrc_target),
            "qrc_baseline": str(kwargs.get("qrc_baseline", "persistence")),
            "target_transform": str(kwargs.get("target_transform", "log")),
            "y_floor_mode": str(kwargs.get("y_floor_mode", "train_p001")),
            "y_floor_value": kwargs.get("y_floor_value", None),
            "tau": float(tau_val),
            "resid_clip": kwargs.get("qrc_resid_clip", None),
            "residvar_penalty": float(kwargs.get("qrc_residvar_penalty", kwargs.get("qrc_residual_penalty", 0.1))),
            "feat_norm": str(kwargs.get("qrc_feat_norm", "standard")),
            "in_pca": int(kwargs.get("pqrc_in_pca", 4)),
            "factor_cap": int(kwargs.get("pqrc_factor_cap", 0)),
            "input_scale": float(0.75 if input_scale_raw is None else input_scale_raw),
            "min_residual_scale": float(kwargs.get("pqrc_min_residual_scale", 0.05)),
            "higher_order": int(kwargs.get("pqrc_higher_order", 1)),
            "budget_alloc": str(kwargs.get("pqrc_budget_alloc", "uniform")),
            "seed": int(seed),
            "feedback_enabled": bool(feedback_enabled),
        }
        if kind_norm == "persist_qrc_weak":
            base_params.update(
                {
                    "qrc_mode": "gated_residual",
                    "qrc_baseline": "persistence",
                    "gain": float(min(0.25, abs(float(kwargs.get("pqrc_gain", 0.25))))),
                    "input_scale": float(max(1e-4, min(1.0, float(kwargs.get("persist_qrc_drive_scale", 0.25))))),
                    "resid_clip": float(max(1e-6, abs(float(kwargs.get("persist_qrc_rclip", 0.05))))),
                    "min_residual_scale": float(min(float(base_params.get("min_residual_scale", 0.05)), 0.25)),
                }
            )
        config_hash_payload = dict(base_params)
        config_hash_payload["kind"] = str(kind_norm)
        config_hash_payload["qrc_config_hash"] = str(kwargs.get("qrc_config_hash", ""))
        cfg_hash = hashlib.sha1(
            json.dumps(config_hash_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:12]
        base_input_scale = float(base_params["input_scale"])
        base_residual_scale = float(base_params["min_residual_scale"])
        base_gain = float(base_params["gain"])
        attempt_grid = [
            {
                "input_scale": base_input_scale,
                "min_residual_scale": base_residual_scale,
                "gain": base_gain if feedback_enabled else 0.0,
            },
            {
                "input_scale": min(0.9, max(0.5, base_input_scale)),
                "min_residual_scale": max(0.05, min(0.12, base_residual_scale)),
                "gain": max(0.08, min(0.25, base_gain)) if feedback_enabled else 0.0,
            },
            {
                "input_scale": max(1.0, base_input_scale),
                "min_residual_scale": max(0.10, base_residual_scale),
                "gain": max(0.20, min(0.50, base_gain)) if feedback_enabled else 0.0,
            },
            {
                "input_scale": max(1.25, base_input_scale),
                "min_residual_scale": max(0.15, base_residual_scale),
                "gain": max(base_gain, 0.65) if feedback_enabled else 0.0,
            },
            {
                "input_scale": max(1.5, base_input_scale),
                "min_residual_scale": max(0.20, base_residual_scale),
                "gain": max(base_gain, 0.90) if feedback_enabled else 0.0,
            },
        ]
        # Deduplicate candidate tuples to avoid redundant retrains when base knobs already match.
        dedup_attempts: list[dict[str, float]] = []
        seen_attempts: set[tuple[float, float, float]] = set()
        for att in attempt_grid:
            key = (
                round(float(att["input_scale"]), 8),
                round(float(att["min_residual_scale"]), 8),
                round(float(att["gain"]), 8),
            )
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            dedup_attempts.append(att)
        attempt_grid = dedup_attempts
        if kind_norm == "persist_qrc_weak":
            attempt_grid = [
                {
                    "input_scale": float(base_params["input_scale"]),
                    "min_residual_scale": float(base_params["min_residual_scale"]),
                    "gain": float(base_params["gain"]),
                }
            ]
        y_dim = int(y_tune.shape[2])
        horizon = int(y_tune.shape[1])
        base_mode = str(base_params.get("qrc_baseline", "persistence")).strip().lower()
        if base_mode == "seasonal13":
            n, l, _ = x_tune.shape
            persistence_val = np.zeros((n, horizon, y_dim), dtype=float)
            for h_idx in range(horizon):
                step = h_idx + 1
                lag = 13 - step
                if lag <= 0:
                    persistence_val[:, h_idx, :] = x_tune[:, -1, :y_dim]
                elif lag < l:
                    persistence_val[:, h_idx, :] = x_tune[:, -(lag + 1), :y_dim]
                else:
                    persistence_val[:, h_idx, :] = x_tune[:, -1, :y_dim]
        else:
            persistence_val = np.repeat(x_tune[:, -1, :y_dim][:, None, :], horizon, axis=1)
        h_obj = int(max(1, min(6, horizon)))
        y_tune_obj = np.asarray(y_tune[:, :h_obj, :], dtype=float)
        persistence_obj = np.asarray(persistence_val[:, :h_obj, :], dtype=float)
        persistence_val_mae = float(np.mean(np.abs(persistence_obj - y_tune_obj)))
        denom = np.maximum(np.abs(y_tune_obj), 1e-8)
        persistence_val_mape = float(np.mean(np.abs(persistence_obj - y_tune_obj) / denom) * 100.0)
        best_model: FactorPredictor | None = None
        best_improving_model: FactorPredictor | None = None
        best_improving_tuple = (float("inf"), float("inf"), float("inf"), float("inf"))
        best_blend_alpha = 1.0
        best_improving_alpha = 1.0
        best_val_mae = float("inf")
        best_delta = 0.0
        for idx, attempt in enumerate(attempt_grid):
            attempt_seed = _config_seed(
                int(seed),
                {
                    **base_params,
                    "kind": str(kind_norm),
                    "attempt": int(idx),
                    "input_scale": float(attempt["input_scale"]),
                    "min_residual_scale": float(attempt["min_residual_scale"]),
                    "gain": float(attempt["gain"]),
                },
            )
            params_i = PhotonicQRCParams(
                **{
                    **base_params,
                    "input_scale": float(attempt["input_scale"]),
                    "min_residual_scale": float(attempt["min_residual_scale"]),
                    "gain": float(attempt["gain"]),
                    "seed": int(attempt_seed),
                }
            )
            model_i = train_photonic_qrc_forecaster(
                x_train=x_fit,
                y_train=y_fit,
                x_val=x_tune,
                y_val=y_tune,
                params=params_i,
            )
            setattr(model_i, "qrc_config_hash", str(cfg_hash))
            pred_val = model_i.predict(x_tune)
            pred_obj = np.asarray(pred_val[:, :h_obj, :], dtype=float)
            blend_alpha, val_mape, val_mae, delta_val = _tune_persistence_blend(
                pred_obj=pred_obj,
                y_true_obj=y_tune_obj,
                persistence_obj=persistence_obj,
                eps=1e-8,
                shrink=(0.80 if feedback_enabled else 0.90),
                min_gain_abs=0.10,
                min_gain_frac=0.02,
            )
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = model_i
                best_delta = delta_val
                best_blend_alpha = float(blend_alpha)
            # Prefer candidates that diverge from persistence and improve at least one core validation metric.
            improved_vs_persistence = bool(
                (delta_val > 1e-6)
                and ((val_mae < persistence_val_mae - 1e-10) or (val_mape < persistence_val_mape - 1e-10))
            )
            if improved_vs_persistence:
                # Score improving attempts by MAPE first, MAE second, then prefer more conservative blend.
                cand_tuple = (float(val_mape), float(val_mae), float(blend_alpha), float(-delta_val))
                if cand_tuple < best_improving_tuple:
                    best_improving_tuple = cand_tuple
                    best_improving_model = model_i
                    best_improving_alpha = float(blend_alpha)
            elif kind_norm != "persist_qrc_weak":
                warnings.warn(
                    f"{kind_norm}: attempt {idx + 1}/{len(attempt_grid)} collapsed to persistence on validation; "
                    "retrying with stronger photonic drive settings.",
                    RuntimeWarning,
                )
        if best_model is None:
            raise RuntimeError(f"{kind_norm}: failed to train a valid photonic model.")
        if best_improving_model is not None:
            if kind_norm == "persist_qrc_weak":
                alpha = float(np.clip(float(kwargs.get("persist_qrc_alpha", 0.25)), 0.0, 1.0))
                rclip = float(max(1e-6, abs(float(kwargs.get("persist_qrc_rclip", 0.05)))))
                wrapped = PersistQRCWeakPredictor(
                    core=best_improving_model,
                    alpha=alpha,
                    r_clip=rclip,
                    floor_eps=1e-8,
                    d_factors=int(y_tune.shape[2]),
                    horizon=int(y_tune.shape[1]),
                )
                setattr(wrapped, "qrc_mode", "gated_residual")
                setattr(wrapped, "qrc_baseline", "persistence")
                return wrapped
            if float(best_improving_alpha) < 0.999999:
                wrapped_blend = PersistenceBlendPredictor(
                    core=best_improving_model,
                    alpha=float(best_improving_alpha),
                    d_factors=int(y_tune.shape[2]),
                    horizon=int(y_tune.shape[1]),
                )
                setattr(wrapped_blend, "qrc_config_hash", str(cfg_hash))
                return wrapped_blend
            return best_improving_model
        warnings.warn(
            f"{kind_norm} remained near persistence after {len(attempt_grid)} attempts; "
            f"returning best available model (best_delta_vs_persistence={best_delta:.3e}) and relying on "
            "collapse diagnostics to mark failure.",
            RuntimeWarning,
        )
        if kind_norm == "persist_qrc_weak":
            alpha = float(np.clip(float(kwargs.get("persist_qrc_alpha", 0.25)), 0.0, 1.0))
            rclip = float(max(1e-6, abs(float(kwargs.get("persist_qrc_rclip", 0.05)))))
            wrapped = PersistQRCWeakPredictor(
                core=best_model,
                alpha=alpha,
                r_clip=rclip,
                floor_eps=1e-8,
                d_factors=int(y_tune.shape[2]),
                horizon=int(y_tune.shape[1]),
            )
            setattr(wrapped, "qrc_mode", "gated_residual")
            setattr(wrapped, "qrc_baseline", "persistence")
            return wrapped
        if float(best_blend_alpha) < 0.999999:
            wrapped_blend = PersistenceBlendPredictor(
                core=best_model,
                alpha=float(best_blend_alpha),
                d_factors=int(y_tune.shape[2]),
                horizon=int(y_tune.shape[1]),
            )
            setattr(wrapped_blend, "qrc_config_hash", str(cfg_hash))
            return wrapped_blend
        return best_model
    return _fit_gru(x_fit, y_fit, x_tune, y_tune, seed=seed)


def _make_synthetic_reservoir_data(
    seed: int = 0,
    n_train: int = 256,
    n_val: int = 96,
    lookback: int = 18,
    horizon: int = 3,
    d_factors: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    aux_dim = 1 + 19 + 2 + 3  # dt_days + phase onehot + sin/cos + missing summary
    x_train = rng.normal(0.0, 1.0, size=(n_train, lookback, d_factors + aux_dim)).astype(np.float32)
    x_val = rng.normal(0.0, 1.0, size=(n_val, lookback, d_factors + aux_dim)).astype(np.float32)

    # Build realistic cadence/missing channels in the tail features.
    for arr in (x_train, x_val):
        phase = rng.integers(0, 19, size=(arr.shape[0], lookback))
        onehot = np.eye(19, dtype=np.float32)[phase]
        dt = (1.0 + (phase % 3 == 0).astype(np.float32) * 0.5).astype(np.float32)
        sin = np.sin((2.0 * np.pi * phase) / 19.0).astype(np.float32)
        cos = np.cos((2.0 * np.pi * phase) / 19.0).astype(np.float32)
        miss = np.clip(rng.normal(0.2, 0.1, size=(arr.shape[0], lookback, 3)), 0.0, 1.0).astype(np.float32)
        arr[:, :, d_factors :] = np.concatenate([dt[:, :, None], onehot, sin[:, :, None], cos[:, :, None], miss], axis=2)

    def _target_from_x(x_seq: np.ndarray) -> np.ndarray:
        n = x_seq.shape[0]
        y = np.zeros((n, horizon, d_factors), dtype=np.float32)
        last = x_seq[:, -1, :d_factors]
        prev = x_seq[:, -2, :d_factors]
        dt = x_seq[:, -1, d_factors]
        phase_sin = x_seq[:, -1, d_factors + 1 + 19]
        phase_cos = x_seq[:, -1, d_factors + 1 + 19 + 1]
        miss = np.mean(x_seq[:, -1, -3:], axis=1)
        z = 0.62 * last + 0.22 * prev
        z += 0.10 * dt[:, None] + 0.05 * phase_sin[:, None] - 0.04 * phase_cos[:, None] - 0.06 * miss[:, None]
        z += rng.normal(0.0, 0.01, size=z.shape).astype(np.float32)
        y[:, 0, :] = z
        for h in range(1, horizon):
            z = 0.72 * z + 0.18 * last + 0.04 * (h + 1)
            y[:, h, :] = z
        return y

    y_train = _target_from_x(x_train)
    y_val = _target_from_x(x_val)
    return x_train, y_train, x_val, y_val


def _self_test_reservoir() -> None:
    x_train, y_train, x_val, y_val = _make_synthetic_reservoir_data(seed=0)
    reservoir = train_classical_forecaster(x_train, y_train, x_val, y_val, kind="reservoir", seed=0)
    persistence = train_classical_forecaster(x_train, y_train, x_val, y_val, kind="persistence", seed=0)

    pred_res = reservoir.predict(x_val)
    pred_per = persistence.predict(x_val)
    if pred_res.shape != y_val.shape:
        raise RuntimeError(f"Reservoir prediction shape mismatch: got {pred_res.shape}, expected {y_val.shape}.")
    if not np.isfinite(pred_res).all():
        raise RuntimeError("Reservoir prediction contains NaN/Inf values.")

    mae_res = float(np.mean(np.abs(pred_res - y_val)))
    mae_per = float(np.mean(np.abs(pred_per - y_val)))
    print(f"[self-test] reservoir_mae={mae_res:.6f} persistence_mae={mae_per:.6f}")
    if not (mae_res < mae_per):
        raise RuntimeError("Reservoir self-test failed: reservoir did not beat persistence.")


if __name__ == "__main__":
    _self_test_reservoir()
