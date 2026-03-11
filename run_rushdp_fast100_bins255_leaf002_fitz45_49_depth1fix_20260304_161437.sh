#!/bin/bash
#SBATCH --job-name=rush255_d1fix
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --output=results/runtime_compare_jobs/benchmark_rushdp_fast100_bins255_leaf002_depth1fix_20260304_161437_%j.out
#SBATCH --error=results/runtime_compare_jobs/benchmark_rushdp_fast100_bins255_leaf002_depth1fix_20260304_161437_%j.err
set -euo pipefail
cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs
/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py   --run-name benchmark_rushdp_fast100_bins255_leaf002_depth1fix_20260304_161437   --datasets eye-state   --depth-budgets 4   --seeds 0   --time-limit 10800   --lookahead-cap 3   --max-bins 255   --lgb-num-leaves 255   --min-samples-leaf 8   --min-child-size 8   --leaf-frac 0.02   --max-branching 3   --reg 1e-05   --msplit-variant rush_dp   --test-size 0.2   --parallel-trials 1   --threads-per-trial 1   --cpu-utilization-target 1.0   --no-optuna-enable   --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759   --optuna-warmstart-max-per-study 0   --lgb-device-type cpu   --lgb-num-threads 1   --lgb-learning-rate 0.04   --lgb-feature-fraction 0.90   --lgb-bagging-fraction 0.90   --lgb-bagging-freq 1   --lgb-min-data-in-bin 1   --lgb-lambda-l2 0.1   --approx-mode   --patch-budget-per-feature 0   --exactify-top-m 0   --results-root results/runs_lightgbm   --stable-results-dir results/lightgbm   --no-package-artifacts
