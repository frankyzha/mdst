#!/bin/bash
set -euo pipefail

cd /home/users/yz1075/msdt

export MSPLIT_FORCE_RUSH_LEGACY=1

/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name benchmark_rushdp_4h_s1_legacy_20260228_164939 \
  --datasets eye-state \
  --depth-budgets 4 \
  --seeds 0 \
  --time-limit 14400 \
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
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_164939 \
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
  --no-package-artifacts
