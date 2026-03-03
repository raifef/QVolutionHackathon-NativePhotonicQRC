from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANTUM_MODELS = {
    "gru_photonic_memory_fb",
    "gru_photonic_qrc_fb",
    "gru_quantum_kernel",
    "gru_quantum_kernel_103754",
}


def _latest_apples_dir(repo_root: Path) -> Path:
    root = repo_root / "results"
    cands = sorted(
        [p for p in root.glob("external_apples_to_apples_l1_zip_*") if p.is_dir() and (p / "leaderboard_all_seed_rows.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No external_apples_to_apples_l1_zip_* run with leaderboard_all_seed_rows.csv under {root}")
    return cands[-1]


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.sort_values(metric, ascending=True).iloc[0]


def _parse_horizon_list(raw: object) -> np.ndarray:
    if isinstance(raw, str):
        return np.asarray(json.loads(raw), dtype=float).reshape(-1)
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=float).reshape(-1)
    return np.asarray([], dtype=float)


def build_seed_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for seed in sorted(rows["seed"].astype(int).unique().tolist()):
        d = rows[rows["seed"].astype(int) == int(seed)].copy()
        dq = d[d["model"].isin(QUANTUM_MODELS)].copy()
        dc = d[~d["model"].isin(QUANTUM_MODELS)].copy()
        if dq.empty or dc.empty:
            continue
        q_rmse = _best_row(dq, "surface_rmse")
        c_rmse = _best_row(dc, "surface_rmse")
        q_mae = _best_row(dq, "surface_mae")
        c_mae = _best_row(dc, "surface_mae")
        out.append(
            {
                "seed": int(seed),
                "best_quantum_model_rmse": str(q_rmse["model"]),
                "best_classical_model_rmse": str(c_rmse["model"]),
                "best_quantum_rmse": float(q_rmse["surface_rmse"]),
                "best_classical_rmse": float(c_rmse["surface_rmse"]),
                "delta_rmse_classical_minus_quantum": float(c_rmse["surface_rmse"] - q_rmse["surface_rmse"]),
                "best_quantum_model_mae": str(q_mae["model"]),
                "best_classical_model_mae": str(c_mae["model"]),
                "best_quantum_mae": float(q_mae["surface_mae"]),
                "best_classical_mae": float(c_mae["surface_mae"]),
                "delta_mae_classical_minus_quantum": float(c_mae["surface_mae"] - q_mae["surface_mae"]),
                "q_rmse_h": _parse_horizon_list(q_rmse["horizon_surface_rmse"]).tolist(),
                "c_rmse_h": _parse_horizon_list(c_rmse["horizon_surface_rmse"]).tolist(),
            }
        )
    return pd.DataFrame(out).sort_values("seed").reset_index(drop=True)


def make_plot(seed_df: pd.DataFrame, out_png: Path) -> None:
    if seed_df.empty:
        raise ValueError("No seed summary rows to plot.")

    rmse_delta = seed_df["delta_rmse_classical_minus_quantum"].to_numpy(dtype=float)
    mae_delta = seed_df["delta_mae_classical_minus_quantum"].to_numpy(dtype=float)
    seeds = seed_df["seed"].astype(int).tolist()

    idx = np.arange(len(seeds) + 1)
    labels = [f"seed {s}" for s in seeds] + ["mean"]
    rmse_bar = np.concatenate([rmse_delta, [float(np.mean(rmse_delta))]])
    mae_bar = np.concatenate([mae_delta, [float(np.mean(mae_delta))]])
    colors_rmse = ["#1b9e77" if v > 0 else "#d95f02" for v in rmse_bar]
    colors_mae = ["#1b9e77" if v > 0 else "#d95f02" for v in mae_bar]

    # Horizon advantage on seeds where quantum wins RMSE.
    wins = seed_df[seed_df["delta_rmse_classical_minus_quantum"] > 0.0]
    h_adv = None
    if not wins.empty:
        mats = []
        for _, r in wins.iterrows():
            qh = np.asarray(r["q_rmse_h"], dtype=float).reshape(-1)
            ch = np.asarray(r["c_rmse_h"], dtype=float).reshape(-1)
            mats.append(ch - qh)
        h_adv = np.mean(np.asarray(mats, dtype=float), axis=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 8), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.bar(idx, rmse_bar, color=colors_rmse, alpha=0.9)
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xticks(idx, labels, rotation=0)
    ax1.set_ylabel("RMSE Delta (Classical - Quantum)")
    ax1.set_title("External Test RMSE Advantage (Positive = Quantum Better)")
    for i, v in enumerate(rmse_bar):
        ax1.text(i, v + (0.00015 if v >= 0 else -0.00025), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    ax2.bar(idx, mae_bar, color=colors_mae, alpha=0.9)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xticks(idx, labels, rotation=0)
    ax2.set_ylabel("MAE Delta (Classical - Quantum)")
    ax2.set_title("External Test MAE Advantage (Positive = Quantum Better)")
    for i, v in enumerate(mae_bar):
        ax2.text(i, v + (0.00012 if v >= 0 else -0.0002), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    if h_adv is not None and h_adv.size > 0:
        h = np.arange(1, h_adv.size + 1)
        ax3.plot(h, h_adv, marker="o", linewidth=2.0, color="#1b9e77", label="Avg over seeds where quantum wins")
        ax3.fill_between(h, 0.0, h_adv, where=h_adv >= 0, alpha=0.2, color="#1b9e77")
        ax3.fill_between(h, 0.0, h_adv, where=h_adv < 0, alpha=0.2, color="#d95f02")
        ax3.axhline(0.0, color="black", linewidth=1.0)
        ax3.set_xticks(h)
        ax3.set_xlabel("Horizon")
        ax3.set_ylabel("RMSE Delta by Horizon (Classical - Quantum)")
        ax3.set_title("Where Quantum Wins: Horizon-Level RMSE Edge")
    else:
        ax3.text(0.5, 0.5, "No quantum-winning seeds to compute horizon edge.", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    wins_rmse = int(np.sum(rmse_delta > 0.0))
    wins_mae = int(np.sum(mae_delta > 0.0))
    fig.suptitle(
        f"Apples-to-Apples External Test (Real Test Data): Quantum vs Classical\n"
        f"Quantum wins by seed: RMSE {wins_rmse}/{len(rmse_delta)}, MAE {wins_mae}/{len(mae_delta)}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.01,
        0.01,
        "Source: external_apples_to_apples_l1_zip run, level-1 zip dataset + test.xlsx. "
        "Best-in-family comparison per seed.",
        fontsize=9,
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready quantum-vs-classical comparison from apples-to-apples external test run.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--run_dir", type=str, default="", help="Run directory with leaderboard_all_seed_rows.csv.")
    parser.add_argument("--out_png", type=str, default="", help="Output PNG path.")
    parser.add_argument("--out_csv", type=str, default="", help="Optional CSV for per-seed summary.")
    parser.add_argument("--repo_root", type=str, default=str(repo_root))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_apples_dir(repo_root)
    in_csv = run_dir / "leaderboard_all_seed_rows.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}")
    rows = pd.read_csv(in_csv)
    seed_df = build_seed_summary(rows)

    out_png = Path(args.out_png).resolve() if str(args.out_png).strip() else (run_dir / "report_quantum_edge_external_test.png")
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else (run_dir / "report_quantum_edge_external_test_seed_summary.csv")
    make_plot(seed_df, out_png)
    seed_df.to_csv(out_csv, index=False)

    print(f"[report_plot] run_dir={run_dir}", flush=True)
    print(f"[report_plot] out_png={out_png}", flush=True)
    print(f"[report_plot] out_csv={out_csv}", flush=True)


if __name__ == "__main__":
    main()
