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


def _latest_apples_dir(repo_root: Path) -> Path:
    root = repo_root / "results"
    cands = sorted(
        [p for p in root.glob("external_apples_to_apples_l1_zip_*") if p.is_dir() and (p / "leaderboard_all_seed_rows.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No external_apples_to_apples_l1_zip_* run with leaderboard_all_seed_rows.csv under {root}")
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


def _format_seed(seed: int) -> str:
    return f"seed_{int(seed)}"


def _best_hybrid_per_seed(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    h_records: list[dict[str, Any]] = []

    model_series = rows["model"].astype(str)
    hybrid_mask = model_series.str.startswith("gru_") & model_series.ne("gru")
    rows_hybrid = rows[hybrid_mask].copy()

    for seed in sorted(rows["seed"].astype(int).unique().tolist()):
        d_seed = rows[rows["seed"].astype(int) == int(seed)].copy()
        d_gru = d_seed[d_seed["model"].astype(str) == "gru"].copy()
        d_h = rows_hybrid[rows_hybrid["seed"].astype(int) == int(seed)].copy()
        if d_gru.empty or d_h.empty:
            continue

        gru = d_gru.sort_values("surface_rmse", ascending=True).iloc[0]
        hybrid = d_h.sort_values("surface_rmse", ascending=True).iloc[0]

        gru_h = _parse_num_list(gru.get("horizon_surface_rmse"))
        hyb_h = _parse_num_list(hybrid.get("horizon_surface_rmse"))
        n_h = int(min(gru_h.size, hyb_h.size))
        if n_h <= 0:
            continue
        gru_h = gru_h[:n_h]
        hyb_h = hyb_h[:n_h]
        delta_h = gru_h - hyb_h

        records.append(
            {
                "seed": int(seed),
                "gru_model": "gru",
                "hybrid_model": str(hybrid["model"]),
                "gru_surface_rmse": float(gru["surface_rmse"]),
                "hybrid_surface_rmse": float(hybrid["surface_rmse"]),
                "delta_rmse_gru_minus_hybrid": float(gru["surface_rmse"] - hybrid["surface_rmse"]),
                "gru_surface_mae": float(gru["surface_mae"]),
                "hybrid_surface_mae": float(hybrid["surface_mae"]),
                "delta_mae_gru_minus_hybrid": float(gru["surface_mae"] - hybrid["surface_mae"]),
                "gru_surface_mape": float(gru["surface_mape"]),
                "hybrid_surface_mape": float(hybrid["surface_mape"]),
                "delta_mape_gru_minus_hybrid": float(gru["surface_mape"] - hybrid["surface_mape"]),
                "horizon_count": int(n_h),
                "gru_horizon_surface_rmse": json.dumps(gru_h.tolist()),
                "hybrid_horizon_surface_rmse": json.dumps(hyb_h.tolist()),
                "delta_horizon_surface_rmse": json.dumps(delta_h.tolist()),
            }
        )

        for h_idx in range(n_h):
            h_records.append(
                {
                    "seed": int(seed),
                    "hybrid_model": str(hybrid["model"]),
                    "horizon": int(h_idx + 1),
                    "gru_surface_rmse_h": float(gru_h[h_idx]),
                    "hybrid_surface_rmse_h": float(hyb_h[h_idx]),
                    "delta_rmse_gru_minus_hybrid_h": float(delta_h[h_idx]),
                }
            )

    out = pd.DataFrame.from_records(records).sort_values("seed").reset_index(drop=True)
    out_h = pd.DataFrame.from_records(h_records).sort_values(["seed", "horizon"]).reset_index(drop=True)
    return out, out_h


def _aggregate_model_seed_stats(rows: pd.DataFrame) -> pd.DataFrame:
    model_series = rows["model"].astype(str)
    keep_mask = model_series.eq("gru") | (model_series.str.startswith("gru_") & model_series.ne("gru"))
    d = rows[keep_mask].copy()
    out_rows: list[dict[str, Any]] = []
    for model, grp in d.groupby("model", sort=True):
        out_rows.append(
            {
                "model": str(model),
                "is_hybrid_gru": bool(str(model) != "gru"),
                "n_seeds": int(grp.shape[0]),
                "rmse_mean": float(grp["surface_rmse"].mean()),
                "rmse_std": float(grp["surface_rmse"].std(ddof=0)),
                "mae_mean": float(grp["surface_mae"].mean()),
                "mae_std": float(grp["surface_mae"].std(ddof=0)),
                "mape_mean": float(grp["surface_mape"].mean()),
                "mape_std": float(grp["surface_mape"].std(ddof=0)),
            }
        )
    return pd.DataFrame.from_records(out_rows).sort_values("rmse_mean", ascending=True).reset_index(drop=True)


def _plot_single_seed_metrics(row: pd.Series, out_path: Path, dpi: int) -> None:
    metrics = [
        ("surface_rmse", float(row["gru_surface_rmse"]), float(row["hybrid_surface_rmse"]), float(row["delta_rmse_gru_minus_hybrid"])),
        ("surface_mae", float(row["gru_surface_mae"]), float(row["hybrid_surface_mae"]), float(row["delta_mae_gru_minus_hybrid"])),
        ("surface_mape", float(row["gru_surface_mape"]), float(row["hybrid_surface_mape"]), float(row["delta_mape_gru_minus_hybrid"])),
    ]
    seed = int(row["seed"])
    hybrid_model = str(row["hybrid_model"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for ax, (name, gru_v, hyb_v, delta_v) in zip(axes, metrics):
        x = np.arange(2)
        vals = np.asarray([gru_v, hyb_v], dtype=float)
        ax.bar(x, vals, color=["#4C78A8", "#F58518"], width=0.65)
        ax.set_xticks(x, ["GRU", "Best Hybrid"])
        ax.set_ylabel(name)
        ax.set_title(name.upper())
        ax.grid(True, axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            0.5,
            0.95,
            f"delta (GRU-Hybrid): {delta_v:+.4f}",
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            color=("#2CA02C" if delta_v > 0 else "#D62728"),
        )
    fig.suptitle(
        f"Single-Seed Focus ({_format_seed(seed)}): GRU vs {hybrid_model}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_single_seed_horizon(row: pd.Series, out_path: Path, dpi: int) -> None:
    seed = int(row["seed"])
    hybrid_model = str(row["hybrid_model"])
    gru_h = _parse_num_list(row["gru_horizon_surface_rmse"])
    hyb_h = _parse_num_list(row["hybrid_horizon_surface_rmse"])
    n_h = int(min(gru_h.size, hyb_h.size))
    gru_h = gru_h[:n_h]
    hyb_h = hyb_h[:n_h]
    delta_h = gru_h - hyb_h
    x = np.arange(1, n_h + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    ax1.plot(x, gru_h, marker="o", linewidth=2.0, color="#4C78A8", label="GRU")
    ax1.plot(x, hyb_h, marker="s", linewidth=2.0, color="#F58518", label=hybrid_model)
    ax1.set_ylabel("surface_rmse")
    ax1.set_title(f"Horizon RMSE ({_format_seed(seed)})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    bar_colors = ["#2CA02C" if v > 0 else "#D62728" for v in delta_h]
    ax2.bar(x, delta_h, color=bar_colors, alpha=0.9)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("delta rmse\n(GRU - Hybrid)")
    ax2.set_title("Horizon Edge (Positive = Hybrid Better)")
    ax2.set_xticks(x)
    ax2.grid(True, axis="y", alpha=0.25)
    for xi, dv in zip(x, delta_h):
        ax2.text(xi, dv, f"{dv:+.4f}", ha="center", va="bottom" if dv >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_seed_level_delta_bars(seed_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x = np.arange(seed_df.shape[0], dtype=float)
    w = 0.36
    rmse_delta = seed_df["delta_rmse_gru_minus_hybrid"].to_numpy(dtype=float)
    mae_delta = seed_df["delta_mae_gru_minus_hybrid"].to_numpy(dtype=float)
    labels = [f"{_format_seed(int(s))}\n{m}" for s, m in zip(seed_df["seed"].tolist(), seed_df["hybrid_model"].astype(str).tolist())]

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.bar(x - w / 2.0, rmse_delta, width=w, color="#2CA02C", label="RMSE delta (GRU - Hybrid)")
    ax.bar(x + w / 2.0, mae_delta, width=w, color="#1F77B4", label="MAE delta (GRU - Hybrid)")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Delta metric")
    ax.set_title("Best Hybrid GRU Edge by Seed (Positive = Hybrid Better)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    for i, v in enumerate(rmse_delta):
        ax.text(i - w / 2.0, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    for i, v in enumerate(mae_delta):
        ax.text(i + w / 2.0, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_horizon_delta_by_seed(seed_h_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    for seed, grp in seed_h_df.groupby("seed", sort=True):
        x = grp["horizon"].to_numpy(dtype=int)
        y = grp["delta_rmse_gru_minus_hybrid_h"].to_numpy(dtype=float)
        model = str(grp["hybrid_model"].iloc[0])
        ax.plot(x, y, marker="o", linewidth=1.8, label=f"{_format_seed(int(seed))} ({model})", alpha=0.85)

    mean_curve = (
        seed_h_df.groupby("horizon", sort=True)["delta_rmse_gru_minus_hybrid_h"]
        .mean()
        .reset_index(drop=False)
    )
    ax.plot(
        mean_curve["horizon"].to_numpy(dtype=int),
        mean_curve["delta_rmse_gru_minus_hybrid_h"].to_numpy(dtype=float),
        color="black",
        linewidth=2.8,
        marker="D",
        label="Mean across seeds",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("delta rmse (GRU - Hybrid)")
    ax.set_title("Horizon-Level Hybrid GRU Edge by Seed")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_aggregate_bars(agg_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x = np.arange(agg_df.shape[0], dtype=float)
    labels = agg_df["model"].astype(str).tolist()
    colors = ["#4C78A8" if m == "gru" else "#F58518" for m in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
    ax1.bar(
        x,
        agg_df["rmse_mean"].to_numpy(dtype=float),
        yerr=agg_df["rmse_std"].to_numpy(dtype=float),
        color=colors,
        capsize=4,
        alpha=0.9,
    )
    ax1.set_ylabel("surface_rmse mean +/- std")
    ax1.set_title("Aggregate Across Seeds: GRU and Hybrid GRU Variants")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(
        x,
        agg_df["mae_mean"].to_numpy(dtype=float),
        yerr=agg_df["mae_std"].to_numpy(dtype=float),
        color=colors,
        capsize=4,
        alpha=0.9,
    )
    ax2.set_ylabel("surface_mae mean +/- std")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_xticks(x, labels, rotation=25, ha="right")
    ax2.set_xlabel("Model")

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create scientific report-ready hybrid GRU edge plots for apples-to-apples external test runs.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--run_dir", type=str, default="", help="Run directory containing leaderboard_all_seed_rows.csv.")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory. Default: <run_dir>/hybrid_gru_report_plots")
    parser.add_argument("--seed_focus", type=int, default=-1, help="Seed to use for single-seed plots. Default: best RMSE-edge seed.")
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI.")
    parser.add_argument("--repo_root", type=str, default=str(repo_root))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else _latest_apples_dir(repo_root)
    in_csv = run_dir / "leaderboard_all_seed_rows.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (run_dir / "hybrid_gru_report_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(in_csv)
    for c in ("surface_rmse", "surface_mae", "surface_mape"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows["seed"] = pd.to_numeric(rows["seed"], errors="coerce").astype("Int64")
    rows = rows.dropna(subset=["seed", "model", "surface_rmse", "surface_mae", "surface_mape"]).copy()
    rows["seed"] = rows["seed"].astype(int)

    seed_df, seed_h_df = _best_hybrid_per_seed(rows)
    if seed_df.empty:
        raise RuntimeError("Could not compute best hybrid-vs-GRU seed summary from leaderboard_all_seed_rows.csv.")
    agg_df = _aggregate_model_seed_stats(rows)

    if int(args.seed_focus) >= 0:
        focus_seed = int(args.seed_focus)
        if focus_seed not in set(seed_df["seed"].astype(int).tolist()):
            raise ValueError(f"--seed_focus={focus_seed} is not available in computed seed summary.")
        focus_row = seed_df[seed_df["seed"].astype(int) == int(focus_seed)].iloc[0]
    else:
        focus_row = seed_df.sort_values("delta_rmse_gru_minus_hybrid", ascending=False).iloc[0]
        focus_seed = int(focus_row["seed"])

    seed_csv = out_dir / "best_hybrid_vs_gru_by_seed.csv"
    seed_h_csv = out_dir / "best_hybrid_vs_gru_horizon_delta_long.csv"
    agg_csv = out_dir / "gru_hybrid_aggregate_seed_stats.csv"
    seed_df.to_csv(seed_csv, index=False)
    seed_h_df.to_csv(seed_h_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)

    p1 = out_dir / f"01_single_seed_metrics_{_format_seed(focus_seed)}.png"
    p2 = out_dir / f"02_single_seed_horizon_rmse_{_format_seed(focus_seed)}.png"
    p3 = out_dir / "03_seed_level_delta_bars_best_hybrid_vs_gru.png"
    p4 = out_dir / "04_horizon_delta_lines_best_hybrid_vs_gru.png"
    p5 = out_dir / "05_aggregate_model_bars_gru_hybrids.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_single_seed_metrics(focus_row, p1, dpi=int(args.dpi))
    _plot_single_seed_horizon(focus_row, p2, dpi=int(args.dpi))
    _plot_seed_level_delta_bars(seed_df, p3, dpi=int(args.dpi))
    _plot_horizon_delta_by_seed(seed_h_df, p4, dpi=int(args.dpi))
    _plot_aggregate_bars(agg_df, p5, dpi=int(args.dpi))

    manifest = {
        "run_dir": str(run_dir),
        "source_csv": str(in_csv),
        "out_dir": str(out_dir),
        "seed_focus": int(focus_seed),
        "focus_hybrid_model": str(focus_row["hybrid_model"]),
        "n_seeds": int(seed_df.shape[0]),
        "plots": [str(p1), str(p2), str(p3), str(p4), str(p5)],
        "tables": [str(seed_csv), str(seed_h_csv), str(agg_csv)],
    }
    manifest_path = out_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"run_dir={run_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"seed_focus={focus_seed}", flush=True)
    print(f"focus_hybrid_model={focus_row['hybrid_model']}", flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
