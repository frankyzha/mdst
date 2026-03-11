#!/bin/bash
#SBATCH --job-name=rushdp_apx_rconc
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=08:00:00
#SBATCH --output=results/runtime_compare_jobs/benchmark_rushdp_approx_refv2_rootconc_fitz49_%j.out
#SBATCH --error=results/runtime_compare_jobs/benchmark_rushdp_approx_refv2_rootconc_fitz49_%j.err

set -euo pipefail

cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs

RUN_NAME="benchmark_rushdp_3h_s1_20260228_095759_refv2_rootconc_$(date +%Y%m%d_%H%M%S)"
echo "[$(date --iso-8601=seconds)] run_name=${RUN_NAME}"

/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name "${RUN_NAME}" \
  --datasets eye-state \
  --depth-budgets 4 \
  --seeds 0 \
  --time-limit 10800 \
  --lookahead-cap 3 \
  --max-bins 96 \
  --min-samples-leaf 8 \
  --min-child-size 8 \
  --leaf-frac 0.05 \
  --max-branching 3 \
  --reg 1e-05 \
  --msplit-variant rush_dp \
  --test-size 0.2 \
  --parallel-trials 1 \
  --threads-per-trial 1 \
  --cpu-utilization-target 1.0 \
  --no-optuna-enable \
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759 \
  --optuna-warmstart-max-per-study 0 \
  --lgb-device-type cpu \
  --lgb-num-threads 1 \
  --lgb-num-leaves 48 \
  --lgb-learning-rate 0.04 \
  --lgb-feature-fraction 0.90 \
  --lgb-bagging-fraction 0.90 \
  --lgb-bagging-freq 1 \
  --lgb-min-data-in-bin 1 \
  --lgb-lambda-l2 0.1 \
  --results-root results/runs_lightgbm \
  --stable-results-dir results/lightgbm \
  --no-package-artifacts \
  --approx-mode
