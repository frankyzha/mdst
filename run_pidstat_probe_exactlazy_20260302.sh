#!/bin/bash
#SBATCH --job-name=pidstat_d4
#SBATCH --partition=compsci-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --nodelist=linux52
#SBATCH --output=results/runtime_compare_jobs/pidstat_probe_exactlazy_20260302_%j.out
#SBATCH --error=results/runtime_compare_jobs/pidstat_probe_exactlazy_20260302_%j.err
set -euo pipefail
cd /home/users/yz1075/msdt
mkdir -p results/runtime_compare_jobs
export MSPLIT_FORCE_RUSH_LEGACY=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PIDSTAT_LOG="results/runtime_compare_jobs/pidstat_probe_exactlazy_20260302_${SLURM_JOB_ID}.pidstat.log"
CMD_LOG="results/runtime_compare_jobs/pidstat_probe_exactlazy_20260302_${SLURM_JOB_ID}.cmd.log"

(
  /usr/bin/time -v .venv/bin/python run_multisplit_experiments_lightgbm.py \
    --run-name pidstat_probe_exactlazy_d4_20260302 \
    --datasets eye-state \
    --depth-budgets 4 \
    --seeds 0 \
    --time-limit 900 \
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
    --no-package-artifacts
) > "$CMD_LOG" 2>&1 &
TARGET_PID=$!

echo "target_pid=$TARGET_PID" >> "$CMD_LOG"
pidstat -h -r -u -w -p "$TARGET_PID" 5 > "$PIDSTAT_LOG" 2>&1 &
PIDSTAT_PID=$!

wait "$TARGET_PID"
TARGET_RC=$?
kill "$PIDSTAT_PID" >/dev/null 2>&1 || true
wait "$PIDSTAT_PID" >/dev/null 2>&1 || true

echo "target_rc=$TARGET_RC" >> "$CMD_LOG"
exit "$TARGET_RC"
