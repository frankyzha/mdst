#!/bin/bash
#SBATCH --job-name=cmp_fast100_d2_6
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=24:00:00
#SBATCH --output=results/runtime_compare_jobs/cmp_fast100_d2_6_%j.out
#SBATCH --error=results/runtime_compare_jobs/cmp_fast100_d2_6_%j.err

set -euo pipefail

cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs

RUN_NAME="comparison_fast100_all3_d2_6_pb0_em0_bins255_pt8_$(date +%Y%m%d_%H%M%S)"
PROFILE_PREFIX="results/runtime_compare_jobs/${RUN_NAME}_${SLURM_JOB_ID}"

CMD=(
  .venv/bin/python run_multisplit_xgboost_comparison.py
  --run-name "${RUN_NAME}"
  --datasets electricity eye-movements eye-state
  --depth-budgets 2 3 4 5 6
  --seeds 0
  --test-size 0.2
  --time-limit 10800
  --lookahead-cap 3
  --max-bins 255
  --min-samples-leaf 8
  --min-child-size 8
  --leaf-frac 0.01
  --max-branching 3
  --reg 1e-05
  --msplit-variant rush_dp
  --approx-mode
  --patch-budget-per-feature 0
  --exactify-top-m 0
  --parallel-trials 8
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
  --xgb-num-threads 4
  --xgb-parallel-trials 8
  --results-root results/comparison
)

/usr/bin/time -v -o "${PROFILE_PREFIX}.time.log" "${CMD[@]}"
printf '%q ' "${CMD[@]}" > "${PROFILE_PREFIX}.cmd.txt"
printf '\n' >> "${PROFILE_PREFIX}.cmd.txt"

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=$(realpath "results/comparison/${RUN_NAME}")"
echo "TIME_LOG=$(realpath "${PROFILE_PREFIX}.time.log")"
echo "CMD_LOG=$(realpath "${PROFILE_PREFIX}.cmd.txt")"
