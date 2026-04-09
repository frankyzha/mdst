#!/usr/bin/env bash
#SBATCH --job-name=msdt_rc_prefix_d4
#SBATCH --output=msdt_rc_prefix_d4_%j.out
#SBATCH --error=msdt_rc_prefix_d4_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=1100G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
set -euo pipefail
cd /home/users/yz1075/msdt
source .venv/bin/activate
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}
export MSPLIT_RUNCOLOR_SCOPE=prefix
.venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name benchmark_distilled_runcolor_prefix_electricity_d4_20260309 \
  --datasets electricity \
  --depth-budgets 4 \
  --seeds 0 \
  --time-limit 10800 \
  --lookahead-cap 3 \
  --max-bins 255 \
  --min-samples-leaf 8 \
  --min-child-size 8 \
  --leaf-frac 0.005 \
  --max-branching 3 \
  --reg 3e-4 \
  --msplit-variant rush_dp \
  --test-size 0.2 \
  --parallel-trials 1 \
  --threads-per-trial 16 \
  --cpu-utilization-target 1.0 \
  --no-optuna-enable \
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759 \
  --optuna-warmstart-max-per-study 0 \
  --lgb-device-type cpu \
  --lgb-num-threads 16 \
  --lgb-num-leaves 255 \
  --lgb-learning-rate 0.05 \
  --lgb-feature-fraction 1.0 \
  --lgb-bagging-fraction 1.0 \
  --lgb-bagging-freq 0 \
  --lgb-min-data-in-bin 1 \
  --lgb-lambda-l2 0.0 \
  --results-root results/runs_lightgbm \
  --stable-results-dir results/lightgbm \
  --no-package-artifacts \
  --approx-mode \
  --patch-budget-per-feature 0 \
  --exactify-top-m 0 \
  --approx-ref-shortlist \
  --approx-ref-widen-max 1 \
  --no-approx-challenger-sweep-enabled \
  --approx-distilled-mode \
  --approx-distilled-alpha 0.5 \
  --approx-distilled-max-depth 1 \
  --approx-distilled-geometry-mode teacher_runcolor
