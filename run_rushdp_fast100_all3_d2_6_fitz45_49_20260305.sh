#!/bin/bash
#SBATCH --job-name=rushdp_all3_d2_6
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=24:00:00
#SBATCH --output=results/runtime_compare_jobs/rushdp_all3_d2_6_%j.out
#SBATCH --error=results/runtime_compare_jobs/rushdp_all3_d2_6_%j.err

set -euo pipefail

cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs

RUN_NAME="benchmark_rushdp_fast100_all3_d2_6_leaf001_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/runs_lightgbm/${RUN_NAME}"
TREE_DIR="${RUN_DIR}/tree_artifacts"
VIZ_DIR="${RUN_DIR}/tree_visualizations"
PROFILE_PREFIX="results/runtime_compare_jobs/${RUN_NAME}_${SLURM_JOB_ID}"

CMD=(
  .venv/bin/python run_multisplit_experiments_lightgbm.py
  --run-name "${RUN_NAME}"
  --datasets electricity eye-movements eye-state
  --depth-budgets 2 3 4 5 6
  --seeds 0
  --time-limit 10800
  --lookahead-cap 3
  --max-bins 255
  --lgb-num-leaves 1024
  --min-samples-leaf 8
  --min-child-size 8
  --leaf-frac 0.01
  --max-branching 3
  --reg 1e-05
  --msplit-variant rush_dp
  --approx-mode
  --patch-budget-per-feature 0
  --exactify-top-m 0
  --test-size 0.2
  --parallel-trials 32
  --threads-per-trial 1
  --cpu-utilization-target 1.0
  --no-optuna-enable
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759
  --optuna-warmstart-max-per-study 0
  --lgb-device-type cpu
  --lgb-num-threads 1
  --lgb-learning-rate 0.04
  --lgb-feature-fraction 0.90
  --lgb-bagging-fraction 0.90
  --lgb-bagging-freq 1
  --lgb-min-data-in-bin 1
  --lgb-lambda-l2 0.1
  --results-root results/runs_lightgbm
  --stable-results-dir results/lightgbm
  --tree-artifacts-dir "${TREE_DIR}"
  --package-artifacts
)

/usr/bin/time -v -o "${PROFILE_PREFIX}.time.log" "${CMD[@]}"

mkdir -p "${VIZ_DIR}"
printf '%q ' "${CMD[@]}" > "${PROFILE_PREFIX}.cmd.txt"
printf '\n' >> "${PROFILE_PREFIX}.cmd.txt"
mv "${PROFILE_PREFIX}.time.log" "${RUN_DIR}/run_profile.time.log"
mv "${PROFILE_PREFIX}.cmd.txt" "${RUN_DIR}/run_profile.cmd.txt"

# Render tree visualizations from saved artifacts (no retraining).
find "${TREE_DIR}" -type f -name '*.json' | sort | while read -r artifact; do
  ds="$(basename "$(dirname "$(dirname "${artifact}")")")"
  depth_dir="$(basename "$(dirname "${artifact}")")"
  depth="${depth_dir#depth_}"
  seed_file="$(basename "${artifact}")"
  seed="${seed_file#seed_}"
  seed="${seed%.json}"
  out_png="${VIZ_DIR}/${ds}_depth${depth}_seed${seed}.png"
  .venv/bin/python visualize_multisplit_tree.py \
    --dataset "${ds}" \
    --pipeline lightgbm \
    --artifact-in "${artifact}" \
    --out "${out_png}"
done

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=$(realpath "${RUN_DIR}")"
