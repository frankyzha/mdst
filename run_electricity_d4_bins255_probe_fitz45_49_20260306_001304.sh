#!/bin/bash
#SBATCH --job-name=el_d4_b255_probe
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=12:00:00
#SBATCH --output=results/runtime_compare_jobs/electricity_d4_bins255_probe_%j.out
#SBATCH --error=results/runtime_compare_jobs/electricity_d4_bins255_probe_%j.err
set -euo pipefail
cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
COMMON=(
  --datasets electricity
  --depth-budgets 4
  --seeds 0
  --time-limit 1200
  --lookahead-cap 2
  --max-branching 3
  --reg 3e-4
  --msplit-variant rush_dp
  --test-size 0.2
  --parallel-trials 1
  --threads-per-trial 16
  --cpu-utilization-target 1.0
  --no-optuna-enable
  --optuna-warmstart-root results/runtime_compare_jobs/no_warmstart_20260228_095759
  --optuna-warmstart-max-per-study 0
  --lgb-device-type cpu
  --lgb-num-threads 16
  --lgb-learning-rate 0.05
  --lgb-feature-fraction 1.0
  --lgb-bagging-fraction 1.0
  --lgb-bagging-freq 0
  --lgb-min-data-in-bin 1
  --lgb-lambda-l2 0.0
  --approx-mode
  --patch-budget-per-feature 0
  --exactify-top-m 0
  --approx-ref-shortlist
  --approx-ref-widen-max 1
  --no-approx-challenger-sweep-enabled
  --results-root results/runs_lightgbm
  --stable-results-dir results/lightgbm
  --no-package-artifacts
  --leaf-frac 0.0003152883311788631
  --min-samples-leaf 10
  --min-child-size 10
)
/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name electricity_d4_fast100_bins255_probe_l31_20260306_001304 \
  --max-bins 255 \
  --lgb-num-leaves 31 \
  "${COMMON[@]}"
/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name electricity_d4_fast100_bins255_probe_l255_20260306_001304 \
  --max-bins 255 \
  --lgb-num-leaves 255 \
  "${COMMON[@]}"
