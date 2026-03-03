from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from submission_swaptions.next6_nested import (
    NestedConfig,
    load_dataset_bundle,
    run_single_fold_model,
    stable_seed,
)


def _load_cfg_from_manifest(manifest_path: Path) -> NestedConfig:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg_raw = dict(obj.get("config", {}))
    valid = {f.name for f in fields(NestedConfig)}
    cfg_kwargs = {k: v for k, v in cfg_raw.items() if k in valid}
    return NestedConfig(**cfg_kwargs)


def _decode_candidate(raw: str) -> dict[str, Any]:
    c = json.loads(str(raw))
    if not isinstance(c, dict):
        return {}
    return dict(c)


def _gru_candidate_from_qk(cand_qk: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("gru_hidden_size", "gru_lr", "gru_dropout", "gru_weight_decay", "gru_epochs", "gru_batch_size"):
        if k in cand_qk:
            out[k] = cand_qk[k]
    return out


def _arr2(x: Any, h: int, f: int) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 2 and a.shape == (h, f):
        return a
    return a.reshape(h, f)


def run_diagnostics(cv_run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = cv_run_dir / "nested_pseudo_next6_manifest.json"
    selected_path = cv_run_dir / "nested_pseudo_next6_selected_params_by_fold.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected params CSV: {selected_path}")

    cfg = _load_cfg_from_manifest(manifest_path)
    bundle = load_dataset_bundle(cfg)
    selected = pd.read_csv(selected_path)
    qk_rows = selected[selected["model"].astype(str).str.lower() == "gru_quantum_kernel"].copy()
    if qk_rows.empty:
        raise ValueError(f"No gru_quantum_kernel rows in {selected_path}")

    recs: list[dict[str, Any]] = []
    for _, row in qk_rows.sort_values("outer_fold").iterrows():
        outer_fold = int(row["outer_fold"])
        train_end = int(row["train_end"])
        cand_qk = _decode_candidate(row["best_candidate_json"])
        qk_seed = stable_seed(cfg.seed, "gru_quantum_kernel", "outerpred", outer_fold)

        out_qk = run_single_fold_model(
            model="gru_quantum_kernel",
            candidate=cand_qk,
            surfaces_input=np.asarray(bundle.surfaces_input, dtype=float),
            surfaces_truth=np.asarray(bundle.surfaces_truth, dtype=float),
            train_end=train_end,
            cfg=cfg,
            seed=qk_seed,
        )

        pred_qk = np.asarray(out_qk["pred_delta_norm"], dtype=float)
        h, f = int(pred_qk.shape[0]), int(pred_qk.shape[1])
        base_qk = _arr2(out_qk["meta"].get("hybrid_base_pred_delta_norm"), h=h, f=f)

        # Force correction off end-to-end.
        cand_zero = dict(cand_qk)
        cand_zero["hybrid_gain_early"] = 0.0
        cand_zero["hybrid_gain_late"] = 0.0
        out_zero = run_single_fold_model(
            model="gru_quantum_kernel",
            candidate=cand_zero,
            surfaces_input=np.asarray(bundle.surfaces_input, dtype=float),
            surfaces_truth=np.asarray(bundle.surfaces_truth, dtype=float),
            train_end=train_end,
            cfg=cfg,
            seed=qk_seed,
        )
        pred_zero = np.asarray(out_zero["pred_delta_norm"], dtype=float)
        base_zero = _arr2(out_zero["meta"].get("hybrid_base_pred_delta_norm"), h=h, f=f)

        # Standalone GRU with matching GRU hyperparams (for confound check).
        cand_gru = _gru_candidate_from_qk(cand_qk)
        out_gru = run_single_fold_model(
            model="gru",
            candidate=cand_gru,
            surfaces_input=np.asarray(bundle.surfaces_input, dtype=float),
            surfaces_truth=np.asarray(bundle.surfaces_truth, dtype=float),
            train_end=train_end,
            cfg=cfg,
            seed=qk_seed,
        )
        pred_gru = np.asarray(out_gru["pred_delta_norm"], dtype=float)

        diff_corr = np.asarray(pred_qk - base_qk, dtype=float)
        energy_by_factor = np.sum(np.square(diff_corr), axis=0)
        total_energy = float(np.sum(energy_by_factor))
        pc1_share = float(energy_by_factor[0] / max(total_energy, 1e-12))
        diff_qk_gru = np.asarray(pred_qk - pred_gru, dtype=float)
        energy_qk_gru_by_factor = np.sum(np.square(diff_qk_gru), axis=0)
        total_energy_qk_gru = float(np.sum(energy_qk_gru_by_factor))
        pc1_share_qk_gru = float(energy_qk_gru_by_factor[0] / max(total_energy_qk_gru, 1e-12))

        recs.append(
            {
                "outer_fold": outer_fold,
                "train_end": train_end,
                "gain0_vs_internal_base_max_abs": float(np.max(np.abs(pred_zero - base_zero))),
                "gain0_vs_internal_base_array_equal": bool(np.array_equal(pred_zero, base_zero)),
                "gain0_vs_qk_base_from_enabled_run_max_abs": float(np.max(np.abs(pred_zero - base_qk))),
                "gain0_vs_standalone_gru_max_abs": float(np.max(np.abs(pred_zero - pred_gru))),
                "qk_correction_energy_total": total_energy,
                "qk_correction_energy_pc1": float(energy_by_factor[0]),
                "qk_correction_energy_pc1_share": pc1_share,
                "qk_correction_energy_factors_json": json.dumps([float(x) for x in energy_by_factor.tolist()]),
                "qk_minus_gru_energy_total": total_energy_qk_gru,
                "qk_minus_gru_energy_pc1": float(energy_qk_gru_by_factor[0]),
                "qk_minus_gru_energy_pc1_share": pc1_share_qk_gru,
                "qk_minus_gru_energy_factors_json": json.dumps([float(x) for x in energy_qk_gru_by_factor.tolist()]),
            }
        )

    out_df = pd.DataFrame(recs).sort_values("outer_fold").reset_index(drop=True)
    summary = {
        "cv_run_dir": str(cv_run_dir),
        "n_folds": int(out_df.shape[0]),
        "gain0_internal_base_max_abs_max": float(out_df["gain0_vs_internal_base_max_abs"].max()),
        "gain0_internal_base_all_array_equal": bool(out_df["gain0_vs_internal_base_array_equal"].all()),
        "gain0_vs_standalone_gru_max_abs_mean": float(out_df["gain0_vs_standalone_gru_max_abs"].mean()),
        "qk_correction_pc1_share_mean": float(out_df["qk_correction_energy_pc1_share"].mean()),
        "qk_correction_pc1_share_median": float(out_df["qk_correction_energy_pc1_share"].median()),
        "qk_minus_gru_pc1_share_mean": float(out_df["qk_minus_gru_energy_pc1_share"].mean()),
        "qk_minus_gru_pc1_share_median": float(out_df["qk_minus_gru_energy_pc1_share"].median()),
    }
    return out_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics for GRU+quantum-kernel pipeline behavior.")
    parser.add_argument(
        "--cv_run_dir",
        type=str,
        required=True,
        help="Directory containing nested_pseudo_next6_manifest.json and selected params CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output dir. Default: <cv_run_dir>/analysis_qk_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv_run_dir = Path(args.cv_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (cv_run_dir / "analysis_qk_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    details_df, summary = run_diagnostics(cv_run_dir=cv_run_dir)
    details_csv = out_dir / "qk_diagnostics_by_fold.csv"
    summary_json = out_dir / "qk_diagnostics_summary.json"
    details_df.to_csv(details_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[qk_diagnostics] details_csv={details_csv}", flush=True)
    print(f"[qk_diagnostics] summary_json={summary_json}", flush=True)
    print(f"[qk_diagnostics] summary={json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
