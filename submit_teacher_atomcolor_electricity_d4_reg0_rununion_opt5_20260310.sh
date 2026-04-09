#!/usr/bin/env bash
#SBATCH --job-name=msdt_atom_elec_d4_opt5
#SBATCH --output=msdt_atom_elec_d4_opt5_%j.out
#SBATCH --error=msdt_atom_elec_d4_opt5_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

cd /home/users/yz1075/msdt
source .venv/bin/activate
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}

/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name benchmark_teacher_atomcolor_electricity_d4_reg0_rununion_opt_20260310_v6 \
  --datasets electricity \
  --depth-budgets 4 \
  --seeds 0 \
  --time-limit 10800 \
  --lookahead-cap 3 \
  --max-bins 255 \
  --min-samples-leaf 8 \
  --min-child-size 8 \
  --proposal-atom-cap 32 \
  --max-branching 3 \
  --reg 0.0 \
  --msplit-variant rush_dp \
  --test-size 0.2 \
  --parallel-trials 1 \
  --threads-per-trial 6 \
  --cpu-utilization-target 1.0 \
  --no-optuna-enable \
  --no-optuna-warmstart-enable \
  --paper-split-protocol \
  --lgb-device-type cpu \
  --lgb-num-threads 6 \
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
  --approx-distilled-geometry-mode teacher_atomcolor
