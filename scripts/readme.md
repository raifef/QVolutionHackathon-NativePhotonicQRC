
Use the following CLI for the full comparison on external test data:

MODELS="gru_photonic_memory_fb,gru_photonic_qrc_fb,gru_quantum_kernel,gru_quantum_kernel_103754,gru,mlp,qml_pca_lstm"

for SEED in 0 42 2024; do
  CV_DIR="$OUT_ROOT/cv_seed_${SEED}"
  EXT_DIR="$OUT_ROOT/seed_${SEED}"

  python -m submission_swaptions.scripts.run_pseudo_next6_nested \
    --data_dir "$DATA_DIR" \
    --level 1 --lookback 20 --horizon 6 \
    --models "$MODELS" \
    --factor_mode 1 --pca_factors 8 \
    --seed "$SEED" \
    --exclude_ported 1 \
    --out_dir "$CV_DIR"

  python -m submission_swaptions.scripts.evaluate_external_next6_nested \
    --data_dir "$DATA_DIR" \
    --test_xlsx "$TEST_XLSX" \
    --best_params_json "$CV_DIR/nested_pseudo_next6_best_params.json" \
    --models "$MODELS" \
    --level 1 --lookback 20 --horizon 6 \
    --factor_mode 1 --pca_factors 8 \
    --seed "$SEED" \
    --exclude_ported 1 \
    --out_dir "$EXT_DIR"
done

python - <<'PY'
import os
from pathlib import Path
import pandas as pd

run_dir = Path(os.environ["OUT_ROOT"])
parts = []
for d in sorted(run_dir.glob("seed_*")):
    p = d / "external_next6_nested_metrics.csv"
    if not p.exists():
        continue
    seed = int(d.name.split("_", 1)[1])
    df = pd.read_csv(p)
    df["seed"] = seed
    parts.append(df)

all_df = pd.concat(parts, ignore_index=True)
all_df.to_csv(run_dir / "leaderboard_all_seed_rows.csv", index=False)

agg = (
    all_df.groupby("model", as_index=False)
    .agg(
        rmse_mean=("surface_rmse", "mean"),
        rmse_std=("surface_rmse", "std"),
        mae_mean=("surface_mae", "mean"),
        mae_std=("surface_mae", "std"),
        mape_mean=("surface_mape", "mean"),
        mape_std=("surface_mape", "std"),
        n_seeds=("seed", "nunique"),
    )
    .sort_values("rmse_mean", ascending=True)
)
agg.to_csv(run_dir / "leaderboard_seed_aggregate.csv", index=False)
print(run_dir)
PY

python -m submission_swaptions.scripts.plot_apples_to_apples_quantum_edge --run_dir "$OUT_ROOT"
python -m submission_swaptions.scripts.plot_hybrid_gru_edge_report --run_dir "$OUT_ROOT"
