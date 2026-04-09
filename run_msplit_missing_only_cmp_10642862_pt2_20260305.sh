#!/bin/bash
#SBATCH --job-name=ms_missing_pt2
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=24:00:00
#SBATCH --output=results/runtime_compare_jobs/ms_missing_pt2_%j.out
#SBATCH --error=results/runtime_compare_jobs/ms_missing_pt2_%j.err

set -euo pipefail
cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs

BASE_RUN="comparison_fast100_all3_d2_6_pb0_em0_bins255_pt8_20260305_083914"
CHILD_RUN="${BASE_RUN}_lightgbm"
RES_ROOT="results/comparison/${BASE_RUN}/_tmp/runs_lightgbm/${CHILD_RUN}"
STABLE_DIR="results/comparison/${BASE_RUN}/_tmp/stable_lightgbm"
ART_DIR="results/comparison/${BASE_RUN}/tree_artifacts/lightgbm_mssplit_all"
PROFILE_PREFIX="results/runtime_compare_jobs/${BASE_RUN}_missingonly_${SLURM_JOB_ID}"

COMMON=(
  .venv/bin/python run_multisplit_experiments_lightgbm.py
  --run-name "${CHILD_RUN}"
  --seeds 0
  --test-size 0.2
  --time-limit 10800
  --lookahead-cap 3
  --max-bins 255
  --leaf-frac 0.01
  --min-samples-leaf 8
  --min-child-size 8
  --max-branching 3
  --reg 1e-05
  --msplit-variant rush_dp
  --approx-mode
  --patch-budget-per-feature 0
  --exactify-top-m 0
  --parallel-trials 2
  --threads-per-trial 1
  --cpu-utilization-target 1.0
  --no-optuna-enable
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759
  --optuna-warmstart-max-per-study 0
  --lgb-device-type cpu
  --lgb-ensemble-runs 1
  --lgb-ensemble-feature-fraction 0.90
  --lgb-ensemble-bagging-fraction 0.90
  --lgb-ensemble-bagging-freq 1
  --results-root "${RES_ROOT}"
  --stable-results-dir "${STABLE_DIR}"
  --tree-artifacts-dir "${ART_DIR}"
  --no-package-artifacts
  --resume
)

/usr/bin/time -v -o "${PROFILE_PREFIX}.part1.time.log" "${COMMON[@]}" \
  --datasets electricity eye-state \
  --depth-budgets 5 6

/usr/bin/time -v -o "${PROFILE_PREFIX}.part2.time.log" "${COMMON[@]}" \
  --datasets eye-movements \
  --depth-budgets 4 5 6

echo "BASE_RUN=${BASE_RUN}"
echo "CHILD_RUN=${CHILD_RUN}"
echo "RUN_DIR=$(realpath "${RES_ROOT}/${CHILD_RUN}")"
