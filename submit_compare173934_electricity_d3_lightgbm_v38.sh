#!/usr/bin/env bash
#SBATCH --job-name=cmp173934_elec_d3_lgb_v38
#SBATCH --output=cmp173934_elec_d3_lgb_v38_%j.out
#SBATCH --error=cmp173934_elec_d3_lgb_v38_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

cd /home/users/yz1075/msdt
source .venv/bin/activate
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
  --run-name compare_20260216_173934_electricity_d3_lightgbm_v38 \
  --datasets electricity \
  --depth-budgets 3 \
  --seeds 0 1 2 3 4 \
  --test-size 0.2 \
  --time-limit 1200.0 \
  --lookahead-cap 3 \
  --max-bins 6 \
  --min-samples-leaf 8 \
  --min-child-size 8 \
  --max-branching 0 \
  --reg 0.01 \
  --branch-penalty 0.002 \
  --parallel-trials 16 \
  --threads-per-trial 1 \
  --optuna-trials 10 \
  --optuna-val-size 0.2 \
  --optuna-seed 0 \
  --optuna-timeout-sec 0.0 \
  --cpu-utilization-target 0.94 \
  --optuna-max-active-studies 0 \
  --results-root results/comparison/compare_20260216_173934_v38/_tmp/runs_lightgbm \
  --stable-results-dir results/comparison/compare_20260216_173934_v38/_tmp/stable_lightgbm \
  --no-package-artifacts \
  --optuna-enable \
  --tree-artifacts-dir results/comparison/compare_20260216_173934_v38/tree_artifacts/lightgbm_mssplit_all \
  --lgb-device-type gpu \
  --lgb-gpu-platform-id 0 \
  --lgb-gpu-device-id 0 \
  --no-lgb-gpu-fallback
