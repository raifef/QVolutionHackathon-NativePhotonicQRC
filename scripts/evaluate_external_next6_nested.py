from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


def _ensure_repo_root_on_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)
    return repo_root


def _maybe_reexec_with_submission_env(repo_root: Path) -> None:
    if os.environ.get("SUBMISSION_REEXEC") == "1":
        return
    candidates = [
        repo_root / ".venv310_arm" / "bin" / "python",
        repo_root.parent / ".venv310_arm" / "bin" / "python",
    ]
    preferred = next((p for p in candidates if p.exists()), None)
    if preferred is None:
        return
    if Path(sys.executable).resolve() != preferred.resolve():
        env = dict(os.environ)
        env["SUBMISSION_REEXEC"] = "1"
        env.setdefault("MPLCONFIGDIR", str(repo_root / ".mplconfig"))
        os.execve(str(preferred), [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_repo_root = _ensure_repo_root_on_path()
_maybe_reexec_with_submission_env(_repo_root)

from submission_swaptions.next6_nested import (  # noqa: E402
    NestedConfig,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_list,
    resolve_models,
    run_external_next6_eval,
)
from submission_swaptions.pipeline import _level_cfg, _load_submission_config  # noqa: E402


def _excel_col_to_idx(col_ref: str) -> int:
    idx = 0
    for ch in col_ref:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        elif "a" <= ch <= "z":
            idx = idx * 26 + (ord(ch) - ord("a") + 1)
    return max(0, idx - 1)


def _read_xlsx_fallback(path: Path) -> pd.DataFrame:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path, "r") as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("x:si", ns):
                txt = "".join((t.text or "") for t in si.findall(".//x:t", ns))
                shared.append(txt)

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            cands = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
            if not cands:
                raise FileNotFoundError(f"No worksheet XML found in {path}")
            sheet_name = cands[0]

        root = ET.fromstring(zf.read(sheet_name))
        rows_out: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", ns):
            vals: dict[int, str] = {}
            max_col = -1
            for cell in row.findall("x:c", ns):
                ref = str(cell.attrib.get("r", "")).strip()
                letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = _excel_col_to_idx(letters) if letters else (max_col + 1)
                t = str(cell.attrib.get("t", "")).strip()
                if t == "inlineStr":
                    val = "".join((t_node.text or "") for t_node in cell.findall(".//x:t", ns))
                else:
                    v_node = cell.find("x:v", ns)
                    raw = "" if v_node is None or v_node.text is None else str(v_node.text)
                    if t == "s":
                        try:
                            sidx = int(raw)
                            val = shared[sidx] if 0 <= sidx < len(shared) else ""
                        except Exception:
                            val = ""
                    else:
                        val = raw
                vals[col_idx] = val
                max_col = max(max_col, int(col_idx))

            if max_col < 0:
                continue
            row_vals = [""] * (max_col + 1)
            for idx, val in vals.items():
                if 0 <= idx < len(row_vals):
                    row_vals[idx] = val
            rows_out.append(row_vals)

    if not rows_out:
        raise ValueError(f"No rows parsed from worksheet in {path}")
    header = [str(h).strip() for h in rows_out[0]]
    recs = []
    for row in rows_out[1:]:
        padded = list(row) + [""] * max(0, len(header) - len(row))
        recs.append({header[i]: padded[i] for i in range(len(header))})
    return pd.DataFrame.from_records(recs)


def _read_external_test(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        df = _read_xlsx_fallback(path)
    if "Date" not in df.columns:
        raise ValueError(f"External test file missing Date column: {path}")
    out = df.copy()
    out["Date"] = out["Date"].astype(str)
    for c in [x for x in out.columns if x != "Date"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_best_params(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: dict[str, dict[str, object]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out[str(k).strip().lower()] = dict(v)
    return out


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser(description="External next6 evaluation using nested pseudo-next6 factor pipeline.")
    parser.add_argument("--data_dir", type=str, default=str(workspace_root / "Quandela" / "Challenge_Swaptions_from_zip"))
    parser.add_argument("--test_xlsx", type=str, default="/Users/raifefoulkes/Downloads/test.xlsx")
    parser.add_argument("--best_params_json", type=str, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="gru_photonic_memory_fb,gru_quantum_kernel,gru_photonic_qrc_fb,ensemble_stack_safe",
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor_mode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pca_factors", type=int, default=8)
    parser.add_argument("--corr_window", type=int, default=40)
    parser.add_argument("--corr_min", type=float, default=0.0)
    parser.add_argument("--sign_min", type=float, default=0.52)
    parser.add_argument("--apply_correction_hmin", type=int, default=3)
    parser.add_argument("--gain_grid", type=str, default="0,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--gate_tau_grid", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--q_modes", type=str, default="8,16,24")
    parser.add_argument("--q_shots", type=str, default="32,64,128")
    parser.add_argument("--q_in_scales", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--q_ridges", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--q_features", type=str, default="clickprob,coincidence")
    parser.add_argument("--qk_gammas", type=str, default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1")
    parser.add_argument("--qk_factor1_scales", type=str, default="0,0.05,0.1,0.25,0.5,1")
    parser.add_argument("--gru_hidden_sizes", type=str, default="48,64,96")
    parser.add_argument("--gru_lrs", type=str, default="3e-4,1e-3")
    parser.add_argument("--gru_dropouts", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--gru_weight_decays", type=str, default="1e-6,1e-5,1e-4")
    parser.add_argument("--gru_epochs_grid", type=str, default="24,36")
    parser.add_argument("--quantum_base_kind", type=str, default="gru")
    parser.add_argument("--ensemble", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ensemble_members", type=str, default="gru,factor_ar,qml_pca_lstm")
    parser.add_argument("--ensemble_quantum_kind", type=str, default="gru_quantum_kernel")
    parser.add_argument("--ensemble_quantum_min_improve", type=float, default=0.01)
    parser.add_argument("--exclude_ported", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(repo_root / "results" / f"external_next6_nested_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg_yaml = _load_submission_config(repo_root / "config.yaml")
    lvl = _level_cfg(cfg_yaml, int(args.level))
    imputer = str(lvl.get("imputer", "ffill_interp" if int(args.level) == 1 else "svd_iterative"))
    use_cycle_phase = bool(lvl.get("use_cycle_phase", True))

    cfg = NestedConfig(
        data_dir=str(args.data_dir),
        level=int(args.level),
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        factor_mode=bool(int(args.factor_mode)),
        pca_factors=int(args.pca_factors),
        seed=int(args.seed),
        corr_window=int(args.corr_window),
        corr_min=float(args.corr_min),
        sign_min=float(args.sign_min),
        apply_correction_hmin=int(args.apply_correction_hmin),
        gain_grid=parse_csv_floats(args.gain_grid, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]),
        gate_tau_grid=parse_csv_floats(args.gate_tau_grid, default=[0.25, 0.5, 1.0, 2.0, 4.0]),
        q_modes_grid=parse_csv_ints(args.q_modes, default=[8, 16, 24]),
        q_shots_grid=parse_csv_ints(args.q_shots, default=[32, 64, 128]),
        q_in_scale_grid=parse_csv_floats(args.q_in_scales, default=[0.75, 1.0, 1.25]),
        q_ridge_grid=parse_csv_floats(args.q_ridges, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        q_feature_grid=[str(x).strip().lower() for x in parse_csv_list(args.q_features)] or ["clickprob", "coincidence"],
        qk_gamma_grid=parse_csv_floats(args.qk_gammas, default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]),
        qk_factor1_scale_grid=parse_csv_floats(args.qk_factor1_scales, default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0]),
        gru_hidden_grid=parse_csv_ints(args.gru_hidden_sizes, default=[48, 64, 96]),
        gru_lr_grid=parse_csv_floats(args.gru_lrs, default=[3e-4, 1e-3]),
        gru_dropout_grid=parse_csv_floats(args.gru_dropouts, default=[0.0, 0.05, 0.1]),
        gru_weight_decay_grid=parse_csv_floats(args.gru_weight_decays, default=[1e-6, 1e-5, 1e-4]),
        gru_epochs_grid=parse_csv_ints(args.gru_epochs_grid, default=[24, 36]),
        quantum_base_kind=str(args.quantum_base_kind).strip().lower(),
        ensemble=bool(int(args.ensemble)),
        ensemble_members=[str(x).strip().lower() for x in parse_csv_list(args.ensemble_members)] or ["gru", "factor_ar", "qml_pca_lstm"],
        ensemble_quantum_kind=str(args.ensemble_quantum_kind).strip().lower(),
        ensemble_quantum_min_improve=float(args.ensemble_quantum_min_improve),
        exclude_ported=bool(int(args.exclude_ported)),
        imputer=str(imputer),
        use_cycle_phase=bool(use_cycle_phase),
    )

    best_params = _load_best_params(Path(args.best_params_json).resolve())
    requested = [str(x).strip().lower() for x in parse_csv_list(args.models)]
    models = resolve_models(
        requested=requested if requested else list(best_params.keys()),
        exclude_ported=bool(int(args.exclude_ported)),
        include_ensemble=bool(int(args.ensemble)),
    )
    test_df = _read_external_test(Path(args.test_xlsx).resolve()).head(int(args.horizon)).copy()
    out_dir = Path(args.out_dir).resolve()
    out = run_external_next6_eval(
        cfg=cfg,
        best_params=best_params,
        models=models,
        test_df=test_df,
        out_dir=out_dir,
    )
    print(f"[external_nested] out_dir={out_dir}", flush=True)
    print(f"[external_nested] metrics_csv={out['metrics_csv']}", flush=True)
    print(f"[external_nested] manifest_json={out['manifest_json']}", flush=True)


if __name__ == "__main__":
    main()
