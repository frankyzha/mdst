#!/bin/bash
#SBATCH --job-name=msdt_rc_prefx5_d4
#SBATCH --partition=compsci-gpu
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --output=/home/users/yz1075/msdt/msdt_rc_prefix_exactopt5_d4_%j.out
#SBATCH --error=/home/users/yz1075/msdt/msdt_rc_prefix_exactopt5_d4_%j.err

set -euo pipefail
cd /home/users/yz1075/msdt
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export MSPLIT_RUNCOLOR_SCOPE=prefix
/home/users/yz1075/msdt/.venv/bin/python /home/users/yz1075/msdt/multisplit_experiment_runner.py \
  --run-name benchmark_distilled_runcolor_prefix_electricity_d4_exactopt5_20260309 \
  --datasets electricity \
  --depth-budgets 4 \
  --lookahead-cap 3 \
  --seeds 0 \
  --methods rush_dp \
  --max-branching 3 \
  --max-bins 255 \
  --reg 0.005 \
  --parallel-trials 1 \
  --threads-per-trial 12 \
  --core-budget 12 \
  --lgb-device gpu \
  --lgb-max-gpu-jobs 1 \
  --lgb-ensemble-runs 1 \
  --approx-mode \
  --patch-budget-per-feature 0 \
  --exactify-top-m 0 \
  --approx-distilled-mode \
  --approx-distilled-alpha 0.5 \
  --approx-distilled-max-depth 1 \
  --approx-distilled-geometry-mode teacher_runcolor \
  --no-optuna-warmstart-enable
