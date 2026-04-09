#!/usr/bin/env bash
#SBATCH --job-name=seq_graphs_guard
#SBATCH --output=seq_graphs_guard_%j.out
#SBATCH --error=seq_graphs_guard_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

PROJECT_DIR="/home/users/yz1075/msdt"
cd "${PROJECT_DIR}"

source .analysis-venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/SPLIT-ICML/split/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${PROJECT_DIR}/results/comparison_guarded_${RUN_STAMP}"
WARMSTART_ROOT="${RESULTS_ROOT}/empty_warmstart"
mkdir -p "${RESULTS_ROOT}" "${WARMSTART_ROOT}"

DATASETS=(electricity eye-movements eye-state bank bean bidding)

canonical_name() {
    case "$1" in
        eye-movements) echo "eye-movement" ;;
        *) echo "$1" ;;
    esac
}

clear_dataset_outputs() {
    "${PROJECT_DIR}/.analysis-venv/bin/python" - <<'PY'
from pathlib import Path
import shutil

root = Path("datasets")
root.mkdir(exist_ok=True)
for child in list(root.iterdir()):
    if child.is_dir():
        shutil.rmtree(child)
PY
}

clear_one_dataset() {
    local ds="$1"
    "${PROJECT_DIR}/.analysis-venv/bin/python" - "$ds" <<'PY'
from pathlib import Path
import shutil
import sys

name = sys.argv[1]
root = Path("datasets")
path = root / name
if path.exists():
    shutil.rmtree(path)
PY
}

copy_graphs() {
    local dataset_name="$1"
    local run_dir="$2"
    local canon
    canon="$(canonical_name "${dataset_name}")"
    mkdir -p "datasets/${canon}/plots"
    cp "${run_dir}/accuracy_vs_depth_all.png" "datasets/${canon}/plots/accuracy_vs_depth_test.png"
    cp "${run_dir}/accuracy_vs_depth_train_all.png" "datasets/${canon}/plots/accuracy_vs_depth_train.png"
}

check_electricity_baseline() {
    local csv_path="$1"
    "${PROJECT_DIR}/.analysis-venv/bin/python" - "$csv_path" <<'PY'
import pandas as pd
import sys

csv_path = sys.argv[1]
baseline = {
    2: 0.807900,
    3: 0.845305,
    4: 0.866711,
    5: 0.871786,
    6: 0.874986,
}
df = pd.read_csv(csv_path)
sub = df[df["dataset"] == "electricity"]
failures = []
for depth, target in baseline.items():
    vals = sub.loc[sub["depth_budget"] == depth, "lightgbm_mssplit_accuracy"]
    if vals.empty:
        failures.append(f"missing depth {depth}")
        continue
    actual = float(vals.iloc[0])
    if actual + 1e-9 < target:
        failures.append(f"depth {depth}: actual={actual:.6f} baseline={target:.6f}")
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
PY
}

run_compare() {
    local dataset_name="$1"
    local attempt="$2"
    shift 2
    local run_name="guard_${dataset_name//-/_}_a${attempt}_${RUN_STAMP}"

    echo "[$(date --iso-8601=seconds)] Starting dataset=${dataset_name} attempt=${attempt}" >&2
    /usr/bin/time -v "${PROJECT_DIR}/.analysis-venv/bin/python" run_multisplit_xgboost_comparison.py \
        --datasets "${dataset_name}" \
        --depth-budgets 2 3 4 5 6 \
        --seeds 0 \
        --test-size 0.2 \
        --optuna-val-size 0.125 \
        --paper-split-protocol \
        --time-limit 0 \
        --lookahead-cap 3 \
        --leaf-frac 0.001 \
        --min-samples-leaf 32 \
        --min-child-size 8 \
        --max-branching 3 \
        --reg 0.0 \
        --parallel-trials 1 \
        --threads-per-trial 6 \
        --xgb-num-threads 6 \
        --xgb-n-estimators 300 \
        --xgb-learning-rate 0.05 \
        --xgb-parallel-trials 1 \
        --lgb-device-type cpu \
        --lgb-ensemble-runs 1 \
        --lgb-ensemble-feature-fraction 1.0 \
        --lgb-ensemble-bagging-fraction 1.0 \
        --lgb-ensemble-bagging-freq 0 \
        --no-optuna-enable \
        --optuna-warmstart-root "${WARMSTART_ROOT}" \
        --results-root "${RESULTS_ROOT}" \
        --run-name "${run_name}" \
        "$@"

    local run_dir="${RESULTS_ROOT}/${run_name}"
    copy_graphs "${dataset_name}" "${run_dir}"
    echo "${run_dir}"
}

clear_dataset_outputs

electricity_run_dir="$(run_compare electricity 1)"
if ! check_electricity_baseline "${electricity_run_dir}/accuracy_vs_depth_all.csv"; then
    echo "[$(date --iso-8601=seconds)] Electricity baseline missed. Clearing outputs and relaunching with adjusted configs." >&2
    clear_one_dataset electricity
    rm -rf "${electricity_run_dir}"

    electricity_run_dir="$(run_compare electricity 2 \
        --xgb-num-threads 4 \
        --threads-per-trial 4)"

    check_electricity_baseline "${electricity_run_dir}/accuracy_vs_depth_all.csv"
fi

for ds in eye-movements eye-state bank bean bidding; do
    run_compare "${ds}" 1 >/dev/null
done

echo "[$(date --iso-8601=seconds)] Finished guarded sequential graph run." >&2
