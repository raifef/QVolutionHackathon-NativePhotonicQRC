from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _latest_qk_run_dir(repo_root: Path) -> Path:
    root = repo_root / "results"
    cands = sorted(
        [
            p
            for p in root.glob("gru_vs_qk_external_10seeds_*")
            if p.is_dir() and (p / "external_10seed_gru_vs_qk_all_rows.csv").exists()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No gru_vs_qk_external_10seeds_* run with required CSV under {root}")
    return cands[-1]


def _parse_num_list(raw: object) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw.astype(float).reshape(-1)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=float).reshape(-1)
    if raw is None:
        return np.asarray([], dtype=float)
    s = str(raw).strip()
    if not s:
        return np.asarray([], dtype=float)
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
            if isinstance(obj, (list, tuple, np.ndarray)):
                return np.asarray(obj, dtype=float).reshape(-1)
        except Exception:
            continue
    return np.asarray([], dtype=float)


def _build_seed_horizon_tables(all_rows: pd.DataFrame, by_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched_rows: list[dict[str, Any]] = []
    h_rows: list[dict[str, Any]] = []
    for seed in sorted(by_seed["seed"].astype(int).unique().tolist()):
        d = all_rows[all_rows["seed"].astype(int) == int(seed)].copy()
        d_gru = d[d["model"].astype(str) == "gru"]
        d_qk = d[d["model"].astype(str) == "gru_quantum_kernel"]
        if d_gru.empty or d_qk.empty:
            continue
        r_gru = d_gru.sort_values("surface_rmse", ascending=True).iloc[0]
        r_qk = d_qk.sort_values("surface_rmse", ascending=True).iloc[0]

        rmse_g = _parse_num_list(r_gru.get("horizon_surface_rmse"))
        rmse_q = _parse_num_list(r_qk.get("horizon_surface_rmse"))
        mae_g = _parse_num_list(r_gru.get("horizon_surface_mae"))
        mae_q = _parse_num_list(r_qk.get("horizon_surface_mae"))
        n = int(min(rmse_g.size, rmse_q.size, mae_g.size, mae_q.size))
        if n <= 0:
            continue
        rmse_g = rmse_g[:n]
        rmse_q = rmse_q[:n]
        mae_g = mae_g[:n]
        mae_q = mae_q[:n]
        dr = rmse_g - rmse_q
        dm = mae_g - mae_q

        row_seed = by_seed[by_seed["seed"].astype(int) == int(seed)].iloc[0]
        enriched_rows.append(
            {
                "seed": int(seed),
                "surface_rmse_gru": float(row_seed["surface_rmse_gru"]),
                "surface_rmse_qk": float(row_seed["surface_rmse_gru_quantum_kernel"]),
                "rmse_delta_gru_minus_qk": float(row_seed["rmse_delta_gru_minus_qk"]),
                "surface_mae_gru": float(row_seed["surface_mae_gru"]),
                "surface_mae_qk": float(row_seed["surface_mae_gru_quantum_kernel"]),
                "mae_delta_gru_minus_qk": float(row_seed["mae_delta_gru_minus_qk"]),
                "surface_mape_gru": float(r_gru["surface_mape"]),
                "surface_mape_qk": float(r_qk["surface_mape"]),
                "mape_delta_gru_minus_qk": float(float(r_gru["surface_mape"]) - float(r_qk["surface_mape"])),
                "qk_wins_rmse": bool(float(row_seed["rmse_delta_gru_minus_qk"]) > 0.0),
                "qk_wins_mae": bool(float(row_seed["mae_delta_gru_minus_qk"]) > 0.0),
                "horizon_count": int(n),
                "horizon_surface_rmse_gru": json.dumps(rmse_g.tolist()),
                "horizon_surface_rmse_qk": json.dumps(rmse_q.tolist()),
                "horizon_rmse_delta_gru_minus_qk": json.dumps(dr.tolist()),
                "horizon_surface_mae_gru": json.dumps(mae_g.tolist()),
                "horizon_surface_mae_qk": json.dumps(mae_q.tolist()),
                "horizon_mae_delta_gru_minus_qk": json.dumps(dm.tolist()),
            }
        )
        for h in range(n):
            h_rows.append(
                {
                    "seed": int(seed),
                    "horizon": int(h + 1),
                    "surface_rmse_gru_h": float(rmse_g[h]),
                    "surface_rmse_qk_h": float(rmse_q[h]),
                    "rmse_delta_gru_minus_qk_h": float(dr[h]),
                    "surface_mae_gru_h": float(mae_g[h]),
                    "surface_mae_qk_h": float(mae_q[h]),
                    "mae_delta_gru_minus_qk_h": float(dm[h]),
                }
            )

    enriched = pd.DataFrame.from_records(enriched_rows).sort_values("seed").reset_index(drop=True)
    long_h = pd.DataFrame.from_records(h_rows).sort_values(["seed", "horizon"]).reset_index(drop=True)
    return enriched, long_h


def _aggregate_stats(all_rows: pd.DataFrame) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    for model, grp in all_rows.groupby("model", sort=True):
        recs.append(
            {
                "model": str(model),
                "n_seeds": int(grp.shape[0]),
                "rmse_mean": float(grp["surface_rmse"].mean()),
                "rmse_std": float(grp["surface_rmse"].std(ddof=0)),
                "mae_mean": float(grp["surface_mae"].mean()),
                "mae_std": float(grp["surface_mae"].std(ddof=0)),
                "mape_mean": float(grp["surface_mape"].mean()),
                "mape_std": float(grp["surface_mape"].std(ddof=0)),
            }
        )
    return pd.DataFrame.from_records(recs).sort_values("rmse_mean", ascending=True).reset_index(drop=True)


def _plot_single_seed_metrics(row: pd.Series, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    seed = int(row["seed"])
    metrics = [
        ("surface_rmse", float(row["surface_rmse_gru"]), float(row["surface_rmse_qk"]), float(row["rmse_delta_gru_minus_qk"])),
        ("surface_mae", float(row["surface_mae_gru"]), float(row["surface_mae_qk"]), float(row["mae_delta_gru_minus_qk"])),
        ("surface_mape", float(row["surface_mape_gru"]), float(row["surface_mape_qk"]), float(row["mape_delta_gru_minus_qk"])),
    ]
    for ax, (name, g, q, d) in zip(axes, metrics):
        vals = np.asarray([g, q], dtype=float)
        x = np.arange(2)
        ax.bar(x, vals, color=["#4C78A8", "#F58518"], width=0.66)
        ax.set_xticks(x, ["GRU", "QK"])
        ax.set_ylabel(name)
        ax.set_title(name.upper())
        ax.grid(True, axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            0.5,
            0.95,
            f"delta (GRU-QK): {d:+.4f}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            color=("#2CA02C" if d > 0 else "#D62728"),
        )
    fig.suptitle(f"Single-Seed Focus (seed_{seed}): GRU vs Quantum Kernel", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_single_seed_horizon(row: pd.Series, out_path: Path, dpi: int) -> None:
    seed = int(row["seed"])
    rmse_g = _parse_num_list(row["horizon_surface_rmse_gru"])
    rmse_q = _parse_num_list(row["horizon_surface_rmse_qk"])
    n = int(min(rmse_g.size, rmse_q.size))
    rmse_g = rmse_g[:n]
    rmse_q = rmse_q[:n]
    delta = rmse_g - rmse_q
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    ax1.plot(x, rmse_g, marker="o", linewidth=2.0, color="#4C78A8", label="GRU")
    ax1.plot(x, rmse_q, marker="s", linewidth=2.0, color="#F58518", label="QK")
    ax1.set_ylabel("surface_rmse")
    ax1.set_title(f"Horizon RMSE (seed_{seed})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    colors = ["#2CA02C" if v > 0 else "#D62728" for v in delta]
    ax2.bar(x, delta, color=colors, alpha=0.9)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("delta rmse\n(GRU-QK)")
    ax2.set_title("Horizon Edge (Positive = QK Better)")
    ax2.set_xticks(x)
    ax2.grid(True, axis="y", alpha=0.25)
    for xi, v in zip(x, delta):
        ax2.text(xi, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_seed_deltas(seed_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x = np.arange(seed_df.shape[0], dtype=float)
    w = 0.38
    rmse = seed_df["rmse_delta_gru_minus_qk"].to_numpy(dtype=float)
    mae = seed_df["mae_delta_gru_minus_qk"].to_numpy(dtype=float)
    labels = [f"seed_{int(s)}" for s in seed_df["seed"].to_numpy(dtype=int)]

    fig, ax = plt.subplots(figsize=(12.4, 6.5))
    ax.bar(x - w / 2.0, rmse, width=w, color="#2CA02C", label="RMSE delta (GRU-QK)")
    ax.bar(x + w / 2.0, mae, width=w, color="#1F77B4", label="MAE delta (GRU-QK)")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Delta")
    ax.set_title("Seed-Level Delta vs Quantum Kernel (Positive = QK Better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    for i, v in enumerate(rmse):
        ax.text(i - w / 2.0, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for i, v in enumerate(mae):
        ax.text(i + w / 2.0, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    if not summary_df.empty:
        s = summary_df.iloc[0]
        ax.text(
            0.01,
            0.98,
            f"mean RMSE delta: {float(s['rmse_delta_mean_gru_minus_qk']):+.4f} | "
            f"QK RMSE wins: {int(s['qk_wins_rmse_seeds'])}/{int(s['n_seeds'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_horizon_deltas(long_h: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    for seed, grp in long_h.groupby("seed", sort=True):
        x = grp["horizon"].to_numpy(dtype=int)
        y = grp["rmse_delta_gru_minus_qk_h"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.4, alpha=0.6, label=f"seed_{int(seed)}")

    mean_h = long_h.groupby("horizon", sort=True)["rmse_delta_gru_minus_qk_h"].mean().reset_index(drop=False)
    std_h = long_h.groupby("horizon", sort=True)["rmse_delta_gru_minus_qk_h"].std(ddof=0).reset_index(drop=True)
    x = mean_h["horizon"].to_numpy(dtype=int)
    y = mean_h["rmse_delta_gru_minus_qk_h"].to_numpy(dtype=float)
    ystd = std_h.to_numpy(dtype=float)
    ax.plot(x, y, color="black", linewidth=2.8, marker="D", label="Mean across seeds")
    ax.fill_between(x, y - ystd, y + ystd, color="gray", alpha=0.2, label="+/-1 std")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("delta rmse (GRU-QK)")
    ax.set_title("Horizon Delta by Seed (Positive = QK Better)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_aggregate_models(agg: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x = np.arange(agg.shape[0], dtype=float)
    labels = agg["model"].astype(str).tolist()
    colors = ["#4C78A8" if m == "gru" else "#F58518" for m in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.8, 7.8), sharex=True)
    ax1.bar(x, agg["rmse_mean"].to_numpy(dtype=float), yerr=agg["rmse_std"].to_numpy(dtype=float), color=colors, capsize=4)
    ax1.set_ylabel("surface_rmse mean +/- std")
    ax1.set_title("Aggregate Across Seeds")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(x, agg["mae_mean"].to_numpy(dtype=float), yerr=agg["mae_std"].to_numpy(dtype=float), color=colors, capsize=4)
    ax2.set_ylabel("surface_mae mean +/- std")
    ax2.set_xlabel("Model")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_xticks(x, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready GRU vs QK external 10-seed plots.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--run_dir", type=str, default="", help="Directory with external_10seed_gru_vs_qk_*.csv files.")
    parser.add_argument("--all_rows_csv", type=str, default="", help="Optional explicit path to all_rows CSV.")
    parser.add_argument("--by_seed_csv", type=str, default="", help="Optional explicit path to by_seed CSV.")
    parser.add_argument("--summary_csv", type=str, default="", help="Optional explicit path to summary CSV.")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory. Default: <run_dir>/report_plots")
    parser.add_argument("--seed_focus", type=int, default=-1, help="Seed for single-seed plots. Default: best QK RMSE seed.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--repo_root", type=str, default=str(repo_root))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_qk_run_dir(repo_root)

    all_rows_csv = Path(args.all_rows_csv).resolve() if str(args.all_rows_csv).strip() else (run_dir / "external_10seed_gru_vs_qk_all_rows.csv")
    by_seed_csv = Path(args.by_seed_csv).resolve() if str(args.by_seed_csv).strip() else (run_dir / "external_10seed_gru_vs_qk_by_seed.csv")
    summary_csv = Path(args.summary_csv).resolve() if str(args.summary_csv).strip() else (run_dir / "external_10seed_gru_vs_qk_summary.csv")
    for p in (all_rows_csv, by_seed_csv, summary_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input CSV: {p}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (run_dir / "report_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = pd.read_csv(all_rows_csv)
    by_seed = pd.read_csv(by_seed_csv)
    summary = pd.read_csv(summary_csv)

    for c in ("seed",):
        all_rows[c] = pd.to_numeric(all_rows[c], errors="coerce").astype("Int64")
        by_seed[c] = pd.to_numeric(by_seed[c], errors="coerce").astype("Int64")
    all_rows = all_rows.dropna(subset=["seed", "model"]).copy()
    all_rows["seed"] = all_rows["seed"].astype(int)
    by_seed = by_seed.dropna(subset=["seed"]).copy()
    by_seed["seed"] = by_seed["seed"].astype(int)

    numeric_all = ["surface_rmse", "surface_mae", "surface_mape"]
    for c in numeric_all:
        all_rows[c] = pd.to_numeric(all_rows[c], errors="coerce")
    all_rows = all_rows.dropna(subset=numeric_all).copy()

    numeric_seed = [
        "surface_rmse_gru",
        "surface_rmse_gru_quantum_kernel",
        "rmse_delta_gru_minus_qk",
        "surface_mae_gru",
        "surface_mae_gru_quantum_kernel",
        "mae_delta_gru_minus_qk",
    ]
    for c in numeric_seed:
        by_seed[c] = pd.to_numeric(by_seed[c], errors="coerce")
    by_seed = by_seed.dropna(subset=numeric_seed).copy().sort_values("seed").reset_index(drop=True)
    summary = summary.copy()

    enriched, long_h = _build_seed_horizon_tables(all_rows, by_seed)
    if enriched.empty or long_h.empty:
        raise RuntimeError("Could not build enriched seed/horizon table from input CSVs.")
    agg = _aggregate_stats(all_rows)

    if int(args.seed_focus) >= 0:
        focus_seed = int(args.seed_focus)
        d = enriched[enriched["seed"].astype(int) == int(focus_seed)]
        if d.empty:
            raise ValueError(f"--seed_focus={focus_seed} not present in enriched table.")
        focus_row = d.iloc[0]
    else:
        focus_row = enriched.sort_values("rmse_delta_gru_minus_qk", ascending=False).iloc[0]
        focus_seed = int(focus_row["seed"])

    p1 = out_dir / f"01_single_seed_metrics_seed_{focus_seed}.png"
    p2 = out_dir / f"02_single_seed_horizon_rmse_seed_{focus_seed}.png"
    p3 = out_dir / "03_seed_level_delta_bars.png"
    p4 = out_dir / "04_horizon_rmse_delta_by_seed.png"
    p5 = out_dir / "05_model_aggregate_errorbars.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_single_seed_metrics(focus_row, p1, dpi=int(args.dpi))
    _plot_single_seed_horizon(focus_row, p2, dpi=int(args.dpi))
    _plot_seed_deltas(enriched, summary, p3, dpi=int(args.dpi))
    _plot_horizon_deltas(long_h, p4, dpi=int(args.dpi))
    _plot_aggregate_models(agg, p5, dpi=int(args.dpi))

    enriched_csv = out_dir / "gru_vs_qk_seed_enriched.csv"
    long_csv = out_dir / "gru_vs_qk_horizon_long.csv"
    agg_csv = out_dir / "gru_vs_qk_aggregate_stats.csv"
    enriched.to_csv(enriched_csv, index=False)
    long_h.to_csv(long_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    manifest = {
        "run_dir": str(run_dir),
        "inputs": {
            "all_rows_csv": str(all_rows_csv),
            "by_seed_csv": str(by_seed_csv),
            "summary_csv": str(summary_csv),
        },
        "out_dir": str(out_dir),
        "seed_focus": int(focus_seed),
        "n_seeds": int(enriched.shape[0]),
        "plots": [str(p1), str(p2), str(p3), str(p4), str(p5)],
        "tables": [str(enriched_csv), str(long_csv), str(agg_csv)],
    }
    manifest_path = out_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"run_dir={run_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"seed_focus={focus_seed}", flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
