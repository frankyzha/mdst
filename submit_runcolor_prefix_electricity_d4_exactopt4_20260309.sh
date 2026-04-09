#!/usr/bin/env bash
#SBATCH --job-name=msdt_rc_prefx4_d4
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --output=/home/users/yz1075/msdt/msdt_rc_prefix_exactopt4_d4_%j.out
#SBATCH --error=/home/users/yz1075/msdt/msdt_rc_prefix_exactopt4_d4_%j.err

set -euo pipefail
cd /home/users/yz1075/msdt
export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export BLIS_NUM_THREADS=12
export MSPLIT_RUNCOLOR_SCOPE=prefix
/home/users/yz1075/msdt/.venv/bin/python /home/users/yz1075/msdt/multisplit_experiment_runner.py   --datasets electricity   --depth-budgets 4   --seeds 0   --time-limit 21600   --threads-per-trial 12   --parallel-trials 1   --run-name benchmark_distilled_runcolor_prefix_electricity_d4_exactopt4_20260309   --approx-mode   --msplit-variant rush_dp   --lookahead-cap 3   --max-bins 255   --max-branching 3   --reg 0.005   --patch-budget-per-feature 0   --exactify-top-m 0   --approx-distilled-mode   --approx-distilled-alpha 0.5   --approx-distilled-max-depth 1   --approx-distilled-geometry-mode teacher_runcolor   --no-optuna-warmstart-enable
