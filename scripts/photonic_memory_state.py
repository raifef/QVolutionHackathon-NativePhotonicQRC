from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhotonicMemoryStateParams:
    """Configuration for feedback-driven photonic-memory state extraction."""

    modes: int = 8
    in_scale: float = 1.0
    gain: float = 0.5
    feature: str = "clickprob"
    shots: int = 32
    state_dim_mode: str = "mean_last"
    seed: int = 0
    wedge_size: int = 0

    def normalized(self) -> "PhotonicMemoryStateParams":
        m = int(max(2, self.modes))
        feat = str(self.feature).strip().lower()
        if feat not in {"clickprob", "coincidence"}:
            raise ValueError("feature must be one of: clickprob, coincidence.")
        mode = str(self.state_dim_mode).strip().lower()
        if mode not in {"mean_last", "mean_last_std"}:
            raise ValueError("state_dim_mode must be one of: mean_last, mean_last_std.")
        return replace(
            self,
            modes=m,
            in_scale=float(max(1e-8, self.in_scale)),
            gain=float(max(0.0, self.gain)),
            feature=feat,
            shots=int(max(0, self.shots)),
            state_dim_mode=mode,
            seed=int(self.seed),
            wedge_size=int(max(0, self.wedge_size)),
        )


def _wrap_phase(phi: np.ndarray) -> np.ndarray:
    out = np.asarray(phi, dtype=np.float64)
    return (out + np.pi) % (2.0 * np.pi) - np.pi


def _pair_indices(m: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(m), k=1)


def _haar_like_unitary(m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    a = rng.normal(size=(int(m), int(m))) + 1j * rng.normal(size=(int(m), int(m)))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    d = np.where(np.abs(d) > 1e-12, d / np.abs(d), 1.0 + 0.0j)
    q = q * d[None, :]
    return np.asarray(q, dtype=np.complex128)


def _state_hash(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def _feature_dim(params: PhotonicMemoryStateParams) -> int:
    p = params.normalized()
    m = int(p.modes)
    if p.feature == "clickprob":
        return int(m)
    return int((m * (m - 1)) // 2)


def _state_dim(params: PhotonicMemoryStateParams) -> int:
    p = params.normalized()
    cdim = int(_feature_dim(p))
    if p.state_dim_mode == "mean_last_std":
        return int(3 * cdim)
    return int(2 * cdim)


def _build_components(d_in: int, params: PhotonicMemoryStateParams) -> dict[str, Any]:
    p = params.normalized()
    m = int(p.modes)
    cdim = int(_feature_dim(p))
    rng = np.random.default_rng(int(p.seed) + 97 * int(d_in) + 7 * m)

    phi0 = rng.uniform(-np.pi, np.pi, size=(m,)).astype(np.float64)
    w_in = rng.normal(0.0, float(p.in_scale) / np.sqrt(max(1, int(d_in))), size=(m, int(d_in))).astype(np.float64)

    wedge_size = int(p.wedge_size) if int(p.wedge_size) > 0 else int(max(1, m // 2))
    wedge_size = int(min(max(1, wedge_size), m))
    wedge_indices = np.sort(rng.choice(m, size=wedge_size, replace=False).astype(int))

    v_fb = rng.normal(0.0, 1.0 / np.sqrt(max(1, cdim)), size=(wedge_size, cdim)).astype(np.float64)
    pair_i, pair_j = _pair_indices(m)
    unitary = _haar_like_unitary(m, seed=int(p.seed) + 1009)

    return {
        "unitary": unitary,
        "phi0": phi0,
        "w_in": w_in,
        "v_fb": v_fb,
        "wedge_indices": wedge_indices,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "feature_dim": cdim,
        "state_dim": _state_dim(p),
        "params": p,
    }


def compute_measurement_features(
    phi_in: np.ndarray,
    *,
    unitary: np.ndarray,
    feature: str,
    shots: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return measurement features from phase-encoded amplitudes.

    This function is intentionally isolated so a future MerLin backend can replace
    the internals without changing state-assembly code.
    """

    a = np.asarray(unitary, dtype=np.complex128) @ np.exp(1j * np.asarray(phi_in, dtype=np.float64))
    p = np.abs(a) ** 2
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    p = p / max(1e-12, float(np.sum(p)))

    shots_i = int(max(0, shots))
    if shots_i > 0:
        counts = rng.multinomial(shots_i, p)
        p_meas = counts.astype(np.float64) / max(1.0, float(np.sum(counts)))
    else:
        p_meas = p

    feat = str(feature).strip().lower()
    if feat == "clickprob":
        return np.asarray(p_meas, dtype=np.float64)
    if feat == "coincidence":
        return np.asarray(p_meas[pair_i] * p_meas[pair_j], dtype=np.float64)
    raise ValueError(f"Unsupported measurement feature mode: {feature}")


def _aggregate_state(c_hist: np.ndarray, state_dim_mode: str) -> np.ndarray:
    c = np.asarray(c_hist, dtype=np.float64)
    mean = np.mean(c, axis=0)
    last = c[-1]
    mode = str(state_dim_mode).strip().lower()
    if mode == "mean_last_std":
        std = np.std(c, axis=0)
        return np.concatenate([mean, last, std], axis=0)
    if mode == "mean_last":
        return np.concatenate([mean, last], axis=0)
    raise ValueError(f"Unsupported state_dim_mode: {state_dim_mode}")


def build_photonic_memory_state_with_meta(
    X_seq: np.ndarray,
    params: PhotonicMemoryStateParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(X_seq, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"X_seq must be [N,L,D_in], got {x.shape}.")
    n, l, d_in = x.shape
    if n <= 0 or l <= 0 or d_in <= 0:
        raise ValueError(f"Invalid X_seq shape: {x.shape}.")

    comp = _build_components(int(d_in), params)
    p = comp["params"]
    feature_dim = int(comp["feature_dim"])
    state_dim = int(comp["state_dim"])

    states = np.zeros((n, state_dim), dtype=np.float64)
    rng = np.random.default_rng(int(p.seed) + 7919)

    for idx in range(n):
        phi = np.asarray(comp["phi0"], dtype=np.float64).copy()
        c_hist = np.zeros((l, feature_dim), dtype=np.float64)
        for t in range(l):
            u = np.asarray(x[idx, t], dtype=np.float64)
            phi_in = _wrap_phase(phi + np.asarray(comp["w_in"], dtype=np.float64) @ u)
            c_t = compute_measurement_features(
                phi_in,
                unitary=comp["unitary"],
                feature=str(p.feature),
                shots=int(p.shots),
                pair_i=comp["pair_i"],
                pair_j=comp["pair_j"],
                rng=rng,
            )
            c_hist[t] = c_t
            wedge = np.asarray(comp["wedge_indices"], dtype=int)
            if wedge.size > 0:
                fb = np.asarray(comp["v_fb"], dtype=np.float64) @ c_t
                phi[wedge] = _wrap_phase(np.asarray(comp["phi0"], dtype=np.float64)[wedge] + float(p.gain) * np.tanh(fb))
        states[idx] = _aggregate_state(c_hist, state_dim_mode=str(p.state_dim_mode))

    qevals = int(n * l)
    total_shots = int(qevals * int(p.shots))
    meta = {
        "params": p,
        "feature_dim": int(feature_dim),
        "state_dim": int(state_dim),
        "wedge_indices": [int(i) for i in np.asarray(comp["wedge_indices"], dtype=int).tolist()],
        "qevals": int(qevals),
        "total_shots": int(total_shots),
        "state_stats": state_feature_statistics(states),
        "state_hash": _state_hash(states),
    }
    return states, meta


def build_photonic_memory_state(X_seq: np.ndarray, params: PhotonicMemoryStateParams) -> np.ndarray:
    """Build fixed-size photonic-memory state vectors from [N,L,D_in] sequences."""

    states, _ = build_photonic_memory_state_with_meta(X_seq, params)
    return states


def state_feature_statistics(states: np.ndarray) -> dict[str, float]:
    s = np.asarray(states, dtype=np.float64)
    s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "l2norm": float(np.linalg.norm(s)),
    }


def feedback_sanity_check(
    X_seq: np.ndarray,
    params: PhotonicMemoryStateParams,
    *,
    mini_batch: int = 8,
    delta_threshold: float = 1e-8,
) -> dict[str, Any]:
    x = np.asarray(X_seq, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0:
        return {
            "feedback_enabled": bool(float(getattr(params, "gain", 0.0)) > 0.0),
            "feedback_inert": True,
            "pqrc_gain": float(getattr(params, "gain", 0.0)),
            "checksum_gain0": "",
            "checksum_gain1": "",
            "phi_diff_gain0_vs_gain1": float("nan"),
            "phi_gain0_mean": float("nan"),
            "phi_gain0_std": float("nan"),
            "phi_gain1_mean": float("nan"),
            "phi_gain1_std": float("nan"),
            "wedge_indices": [],
        }

    n = int(min(max(2, int(mini_batch)), x.shape[0]))
    x_small = x[:n]
    p = params.normalized()
    p0 = replace(p, gain=0.0)

    s0, m0 = build_photonic_memory_state_with_meta(x_small, p0)
    s1, m1 = build_photonic_memory_state_with_meta(x_small, p)
    delta = float(np.std(np.asarray(s1, dtype=np.float64) - np.asarray(s0, dtype=np.float64)))

    gain_pos = bool(float(p.gain) > 0.0)
    if gain_pos and delta <= float(delta_threshold):
        raise AssertionError(
            "Photonic-memory feedback sanity failed: states(gain>0) do not differ from gain=0 "
            f"(std_delta={delta:.3e}, threshold={float(delta_threshold):.3e})."
        )

    checksum0 = str(m0.get("state_hash", _state_hash(s0)))
    checksum1 = str(m1.get("state_hash", _state_hash(s1)))
    return {
        "feedback_enabled": gain_pos,
        "feedback_inert": bool(gain_pos and delta <= float(delta_threshold)),
        "pqrc_gain": float(p.gain),
        "checksum_gain0": checksum0,
        "checksum_gain1": checksum1,
        "phi_diff_gain0_vs_gain1": float(delta),
        "phi_gain0_mean": float(np.mean(s0)),
        "phi_gain0_std": float(np.std(s0)),
        "phi_gain1_mean": float(np.mean(s1)),
        "phi_gain1_std": float(np.std(s1)),
        "wedge_indices": [int(i) for i in m1.get("wedge_indices", [])],
    }


__all__ = [
    "PhotonicMemoryStateParams",
    "build_photonic_memory_state",
    "build_photonic_memory_state_with_meta",
    "compute_measurement_features",
    "state_feature_statistics",
    "feedback_sanity_check",
]
