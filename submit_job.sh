#!/bin/bash
#SBATCH --job-name=msdt             # Name in the queue
#SBATCH --output=msdt_%j.out        # Save output print statements to this file (%j = job ID)
#SBATCH --error=msdt_%j.err         # Save errors to this file
#SBATCH --mail-type=ALL             # Email you when job Starts, Ends, or Fails
# SBATCH --mail-user=yz1075@duke.edu

# --- PARTITION/NODE CONFIGURATION ---
# CPU-first defaults (fast queueing for test_strat and CPU pipelines).
# Override at submit time for GPU runs, e.g.:
#   sbatch -p compsci-gpu --gres=gpu:1 --mem=1100G submit_job.sh
#SBATCH -p compsci
#SBATCH --ntasks=1

# --- MEMORY CONFIGURATION ---
# CPU default memory; override with --mem if needed.
#SBATCH --mem=256G

# --- CPU CONFIGURATION ---
# Request 16 CPU cores by default.
#SBATCH --cpus-per-task=16

# --- TIME LIMIT ---
#SBATCH --time=24:00:00

# --- The Actual Work ---
set -euo pipefail

echo "Job started on $(hostname) at $(date)"

PROJECT_DIR="/home/users/yz1075/msdt"
cd "${PROJECT_DIR}"

# CPU utilization plan:
# - Run multiple (dataset, depth, seed) trials in parallel processes.
# - Keep each process capped to THREADS_PER_TRIAL threads to avoid oversubscription.
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-16}"
TARGET_CPUS="${TARGET_CPUS:-16}"
THREADS_PER_TRIAL="${THREADS_PER_TRIAL:-1}"
if [ "${THREADS_PER_TRIAL}" -lt 1 ]; then
    THREADS_PER_TRIAL=1
fi

if [ -z "${PARALLEL_TRIALS:-}" ]; then
    EFFECTIVE_CPUS="${TOTAL_CPUS}"
    if [ "${TARGET_CPUS}" -gt 0 ] && [ "${TARGET_CPUS}" -lt "${EFFECTIVE_CPUS}" ]; then
        EFFECTIVE_CPUS="${TARGET_CPUS}"
    fi
    PARALLEL_TRIALS=$(( EFFECTIVE_CPUS / THREADS_PER_TRIAL ))
    if [ "${PARALLEL_TRIALS}" -lt 1 ]; then
        PARALLEL_TRIALS=1
    fi
fi

# Optional cap if you need to constrain memory pressure.
MAX_PARALLEL_TRIALS="${MAX_PARALLEL_TRIALS:-0}"
if [ "${MAX_PARALLEL_TRIALS}" -gt 0 ] && [ "${PARALLEL_TRIALS}" -gt "${MAX_PARALLEL_TRIALS}" ]; then
    PARALLEL_TRIALS="${MAX_PARALLEL_TRIALS}"
fi

# Dataset/depth profiles.
DEFAULT_DATASETS=(electricity eye-movements eye-state)
DEFAULT_DEPTH_BUDGETS=(2 3 4 5)
FOCUS_DATASETS=(electricity)
FOCUS_DEPTH_BUDGETS=(2 3 4 5)
DATASETS=("${DEFAULT_DATASETS[@]}")
DEPTH_BUDGETS=("${DEFAULT_DEPTH_BUDGETS[@]}")

# Experiment configuration
TIME_LIMIT="${TIME_LIMIT:-3000}"
LOOKAHEAD_CAP=3
MAX_BINS=255
MIN_SAMPLES_LEAF=8
MIN_CHILD_SIZE=8
LEAF_FRAC="${LEAF_FRAC:-}"
OPTUNA_LEAF_FRAC_GRID="${OPTUNA_LEAF_FRAC_GRID:-}"
MAX_BRANCHES=5
REG=0.01
MSPLIT_VARIANT="${MSPLIT_VARIANT:-rush_dp}"  # optimal_dp | rush_dp
TEST_SIZE=0.2
OPTUNA_ENABLE="${OPTUNA_ENABLE:-1}"
USER_SET_OPTUNA_TRIALS=0
if [ -n "${OPTUNA_TRIALS+x}" ]; then
    USER_SET_OPTUNA_TRIALS=1
fi
OPTUNA_TRIALS="${OPTUNA_TRIALS:-5}"
OPTUNA_VAL_SIZE="${OPTUNA_VAL_SIZE:-0.2}"
OPTUNA_SEED="${OPTUNA_SEED:-0}"
OPTUNA_TIMEOUT_SEC="${OPTUNA_TIMEOUT_SEC:-0}"
OPTUNA_WARMSTART_ENABLE="${OPTUNA_WARMSTART_ENABLE:-1}"
OPTUNA_WARMSTART_ROOT="${OPTUNA_WARMSTART_ROOT:-results}"
OPTUNA_WARMSTART_MAX_PER_STUDY="${OPTUNA_WARMSTART_MAX_PER_STUDY:-8}"
OPTUNA_SEED_CANDIDATES="${OPTUNA_SEED_CANDIDATES:-4}"
CPU_UTILIZATION_TARGET="${CPU_UTILIZATION_TARGET:-1.0}"
PAPER_SPLIT_PROTOCOL="${PAPER_SPLIT_PROTOCOL:-1}"
OPTUNA_MAX_ACTIVE_STUDIES="${OPTUNA_MAX_ACTIVE_STUDIES:-0}"
USER_SET_OPTUNA_MAX_CONCURRENT_TRIALS=0
if [ -n "${OPTUNA_MAX_CONCURRENT_TRIALS+x}" ]; then
    USER_SET_OPTUNA_MAX_CONCURRENT_TRIALS=1
fi
OPTUNA_MAX_CONCURRENT_TRIALS="${OPTUNA_MAX_CONCURRENT_TRIALS:-0}"
LGB_DEVICE_TYPE="${LGB_DEVICE_TYPE:-gpu}"
LGB_GPU_PLATFORM_ID="${LGB_GPU_PLATFORM_ID:-0}"
LGB_GPU_DEVICE_ID="${LGB_GPU_DEVICE_ID:-0}"
LGB_GPU_FALLBACK="${LGB_GPU_FALLBACK:-1}"
LGB_MAX_GPU_JOBS="${LGB_MAX_GPU_JOBS:-1}"
LGB_GPU_LOCK_DIR="${LGB_GPU_LOCK_DIR:-/tmp/msdt_lgb_gpu_lock}"
LGB_ENSEMBLE_RUNS="${LGB_ENSEMBLE_RUNS:-3}"
LGB_ENSEMBLE_FEATURE_FRACTION="${LGB_ENSEMBLE_FEATURE_FRACTION:-0.8}"
LGB_ENSEMBLE_BAGGING_FRACTION="${LGB_ENSEMBLE_BAGGING_FRACTION:-0.8}"
LGB_ENSEMBLE_BAGGING_FREQ="${LGB_ENSEMBLE_BAGGING_FREQ:-1}"
LGB_THRESHOLD_DEDUP_EPS="${LGB_THRESHOLD_DEDUP_EPS:-1e-9}"

# Choose preprocessing mode:
#   PIPELINE=cart              -> CART threshold binning + SPLIT C++ solver
#   PIPELINE=lightgbm          -> LightGBM threshold binning + SPLIT C++ solver (multiway)
#   PIPELINE=lightgbm_split    -> LightGBM threshold binning + binary SPLIT (GOSDT)
#   PIPELINE=lightgbm_minimal  -> LightGBM-only mode with higher default Optuna budget
#   PIPELINE=lightgbm_focus    -> LightGBM-only mode on electricity, depths 2-5
#   PIPELINE=compare           -> LightGBM-binned MSPLIT, XGBoost, and ShapeCART reference
#   PIPELINE=test_strat        -> partition-dp benchmark (all-feature mode)
PIPELINE="${PIPELINE:-compare}"

# Focused LightGBM profile: electricity only, depths 2-5.
if [ "${PIPELINE}" = "lightgbm_focus" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_4" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_5" ]; then
    DATASETS=("${FOCUS_DATASETS[@]}")
    DEPTH_BUDGETS=("${FOCUS_DEPTH_BUDGETS[@]}")
fi

# Minimal mode defaults: keep compare defaults unchanged, but use 30 Optuna trials
# for LightGBM-only runs unless user explicitly overrode OPTUNA_TRIALS.
if [ "${PIPELINE}" = "lightgbm_minimal" ] && [ "${USER_SET_OPTUNA_TRIALS}" -eq 0 ]; then
    OPTUNA_TRIALS=30
fi
if [ "${PIPELINE}" = "lightgbm_focus" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_4" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_5" ]; then
    if [ "${USER_SET_OPTUNA_TRIALS}" -eq 0 ]; then
        OPTUNA_TRIALS=50
    fi
fi
if [ "${PIPELINE}" = "lightgbm_minimal" ] && [ "${USER_SET_OPTUNA_MAX_CONCURRENT_TRIALS}" -eq 0 ]; then
    # Keep high throughput while reducing peak memory pressure from 15 studies x 4 jobs.
    OPTUNA_MAX_CONCURRENT_TRIALS=45
fi

# For deep compare runs (depth >= 5), very high process fan-out can exceed job memory.
# Keep a safe default cap; override by setting ENFORCE_SAFE_COMPARE_CAP=0.
if [ "${PIPELINE:-compare}" = "compare" ] && [ "${ENFORCE_SAFE_COMPARE_CAP:-0}" = "1" ]; then
    SAFE_COMPARE_PARALLEL_CAP="${SAFE_COMPARE_PARALLEL_CAP:-20}"
    LAST_DEPTH_INDEX=$(( ${#DEPTH_BUDGETS[@]} - 1 ))
    MAX_DEPTH="${DEPTH_BUDGETS[${LAST_DEPTH_INDEX}]}"
    if [ "${MAX_DEPTH}" -ge 5 ] && [ "${PARALLEL_TRIALS}" -gt "${SAFE_COMPARE_PARALLEL_CAP}" ]; then
        echo "Capping PARALLEL_TRIALS from ${PARALLEL_TRIALS} to ${SAFE_COMPARE_PARALLEL_CAP} for deep compare run (max_depth=${MAX_DEPTH})"
        PARALLEL_TRIALS="${SAFE_COMPARE_PARALLEL_CAP}"
    fi
fi

# Compare pipeline Optuna tuning can OOM at larger trial counts when many studies run in parallel.
# Keep a hard cap of 5 trials for compare jobs.
if [ "${PIPELINE:-compare}" = "compare" ] && [ "${OPTUNA_TRIALS}" -gt 5 ]; then
    echo "Capping OPTUNA_TRIALS from ${OPTUNA_TRIALS} to 5 for compare pipeline"
    OPTUNA_TRIALS=5
fi

# run_vcm_* scripts use SPLIT_THREADS for BLAS/OpenMP caps.
export SPLIT_THREADS="${SPLIT_THREADS:-${THREADS_PER_TRIAL}}"
export LIGHTGBM_NUM_THREADS="${LIGHTGBM_NUM_THREADS:-${THREADS_PER_TRIAL}}"
# Throughput-oriented default for many independent XGBoost trials.
# Override with XGB_NUM_THREADS if you want fatter single-trial models.
export XGB_NUM_THREADS="${XGB_NUM_THREADS:-1}"

echo "CPU settings: total_cpus=${TOTAL_CPUS}, target_cpus=${TARGET_CPUS}, parallel_trials=${PARALLEL_TRIALS}, threads_per_trial=${THREADS_PER_TRIAL}, split_threads=${SPLIT_THREADS}, lgb_threads=${LIGHTGBM_NUM_THREADS}, xgb_threads=${XGB_NUM_THREADS}"
echo "Optuna settings: enable=${OPTUNA_ENABLE}, trials=${OPTUNA_TRIALS}, val_size=${OPTUNA_VAL_SIZE}, seed=${OPTUNA_SEED}, timeout_sec=${OPTUNA_TIMEOUT_SEC}, warmstart_enable=${OPTUNA_WARMSTART_ENABLE}, warmstart_root=${OPTUNA_WARMSTART_ROOT}, warmstart_max=${OPTUNA_WARMSTART_MAX_PER_STUDY}, seed_candidates=${OPTUNA_SEED_CANDIDATES}, cpu_target=${CPU_UTILIZATION_TARGET}, max_active_studies=${OPTUNA_MAX_ACTIVE_STUDIES}, max_concurrent_trials=${OPTUNA_MAX_CONCURRENT_TRIALS}"
echo "Split protocol: paper_split_protocol=${PAPER_SPLIT_PROTOCOL}"
echo "MSPLIT variant: ${MSPLIT_VARIANT}"
echo "LightGBM binning backend: device=${LGB_DEVICE_TYPE}, gpu_platform=${LGB_GPU_PLATFORM_ID}, gpu_device=${LGB_GPU_DEVICE_ID}, gpu_fallback=${LGB_GPU_FALLBACK}, max_gpu_jobs=${LGB_MAX_GPU_JOBS}, gpu_lock_dir=${LGB_GPU_LOCK_DIR}, ensemble_runs=${LGB_ENSEMBLE_RUNS}, ensemble_feature_fraction=${LGB_ENSEMBLE_FEATURE_FRACTION}, ensemble_bagging_fraction=${LGB_ENSEMBLE_BAGGING_FRACTION}, ensemble_bagging_freq=${LGB_ENSEMBLE_BAGGING_FREQ}, threshold_dedup_eps=${LGB_THRESHOLD_DEDUP_EPS}"

COMMON_ARGS=(
    --datasets "${DATASETS[@]}"
    --depth-budgets "${DEPTH_BUDGETS[@]}"
    --time-limit "${TIME_LIMIT}"
    --lookahead-cap "${LOOKAHEAD_CAP}"
    --max-bins "${MAX_BINS}"
    --min-samples-leaf "${MIN_SAMPLES_LEAF}"
    --min-child-size "${MIN_CHILD_SIZE}"
    --max-branching "${MAX_BRANCHES}"
    --reg "${REG}"
    --msplit-variant "${MSPLIT_VARIANT}"
    --test-size "${TEST_SIZE}"
    --parallel-trials "${PARALLEL_TRIALS}"
    --threads-per-trial "${THREADS_PER_TRIAL}"
    --optuna-trials "${OPTUNA_TRIALS}"
    --optuna-val-size "${OPTUNA_VAL_SIZE}"
    --optuna-seed "${OPTUNA_SEED}"
    --optuna-timeout-sec "${OPTUNA_TIMEOUT_SEC}"
    --optuna-warmstart-root "${OPTUNA_WARMSTART_ROOT}"
    --optuna-warmstart-max-per-study "${OPTUNA_WARMSTART_MAX_PER_STUDY}"
    --optuna-seed-candidates "${OPTUNA_SEED_CANDIDATES}"
    --cpu-utilization-target "${CPU_UTILIZATION_TARGET}"
    --optuna-max-active-studies "${OPTUNA_MAX_ACTIVE_STUDIES}"
    --optuna-max-concurrent-trials "${OPTUNA_MAX_CONCURRENT_TRIALS}"
)
if [ -n "${LEAF_FRAC}" ]; then
    COMMON_ARGS+=(--leaf-frac "${LEAF_FRAC}")
fi
if [ -n "${OPTUNA_LEAF_FRAC_GRID}" ]; then
    read -r -a OPTUNA_LEAF_FRAC_GRID_ARR <<< "${OPTUNA_LEAF_FRAC_GRID}"
    if [ "${#OPTUNA_LEAF_FRAC_GRID_ARR[@]}" -gt 0 ]; then
        COMMON_ARGS+=(--optuna-leaf-frac-grid "${OPTUNA_LEAF_FRAC_GRID_ARR[@]}")
    fi
fi
if [ "${OPTUNA_ENABLE}" = "1" ]; then
    COMMON_ARGS+=(--optuna-enable)
else
    COMMON_ARGS+=(--no-optuna-enable)
fi
if [ "${OPTUNA_WARMSTART_ENABLE}" = "1" ]; then
    COMMON_ARGS+=(--optuna-warmstart-enable)
else
    COMMON_ARGS+=(--no-optuna-warmstart-enable)
fi
if [ "${PAPER_SPLIT_PROTOCOL}" = "1" ]; then
    COMMON_ARGS+=(--paper-split-protocol)
else
    COMMON_ARGS+=(--no-paper-split-protocol)
fi

LIGHTGBM_ARGS=(
    --lgb-device-type "${LGB_DEVICE_TYPE}"
    --lgb-ensemble-runs "${LGB_ENSEMBLE_RUNS}"
    --lgb-ensemble-feature-fraction "${LGB_ENSEMBLE_FEATURE_FRACTION}"
    --lgb-ensemble-bagging-fraction "${LGB_ENSEMBLE_BAGGING_FRACTION}"
    --lgb-ensemble-bagging-freq "${LGB_ENSEMBLE_BAGGING_FREQ}"
    --lgb-threshold-dedup-eps "${LGB_THRESHOLD_DEDUP_EPS}"
    --lgb-gpu-platform-id "${LGB_GPU_PLATFORM_ID}"
    --lgb-gpu-device-id "${LGB_GPU_DEVICE_ID}"
    --lgb-max-gpu-jobs "${LGB_MAX_GPU_JOBS}"
    --lgb-gpu-lock-dir "${LGB_GPU_LOCK_DIR}"
)
if [ "${LGB_GPU_FALLBACK}" = "1" ]; then
    LIGHTGBM_ARGS+=(--lgb-gpu-fallback)
else
    LIGHTGBM_ARGS+=(--no-lgb-gpu-fallback)
fi

# Prefer the repo-local venv interpreter to avoid PATH/shebang issues after folder moves.
if [ -x ".venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
else
    VENV_PYTHON="python"
fi

# Preflight: confirm the runtime can import SPLIT and its native extension.
if [ "${PIPELINE}" != "test_strat" ]; then
    "${VENV_PYTHON}" - <<'PY'
import split
import split._libgosdt
print("SPLIT native backend import OK")
PY
fi


if [ -z "${RUN_NAME:-}" ]; then
    RUN_NAME="${PIPELINE}_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_NAME

PIPELINE_CMD=()
if [ "${PIPELINE}" = "lightgbm" ] || [ "${PIPELINE}" = "lightgbm_minimal" ] || [ "${PIPELINE}" = "lightgbm_focus" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_4" ] || [ "${PIPELINE}" = "lightgbm_electricity_d2_5" ]; then
    RUN_DIR="${PROJECT_DIR}/results/runs_lightgbm/${RUN_NAME}"
    PIPELINE_CMD=(./run_vcm_multisplit_lightgbm.sh "${COMMON_ARGS[@]}" "${LIGHTGBM_ARGS[@]}")
elif [ "${PIPELINE}" = "lightgbm_split" ]; then
    PIPELINE_CMD=(
        "${VENV_PYTHON}" run_binarysplit_experiments_lightgbm.py
        "${COMMON_ARGS[@]}"
        "${LIGHTGBM_ARGS[@]}"
    )
elif [ "${PIPELINE}" = "cart" ]; then
    RUN_DIR="${PROJECT_DIR}/results/runs/${RUN_NAME}"
    PIPELINE_CMD=(./run_vcm_multisplit.sh "${COMMON_ARGS[@]}")
elif [ "${PIPELINE}" = "compare" ]; then
    RUN_DIR="${PROJECT_DIR}/results/comparison/${RUN_NAME}"
    PIPELINE_CMD=(
        "${VENV_PYTHON}" run_multisplit_xgboost_comparison.py "${COMMON_ARGS[@]}" "${LIGHTGBM_ARGS[@]}"
        --run-name "${RUN_NAME}"
        --xgb-num-threads "${XGB_NUM_THREADS}"
        --xgb-parallel-trials "${XGB_PARALLEL_TRIALS:-0}"
        --viz-select "${VIZ_SELECT:-best}"
        --viz-min-depth "${VIZ_MIN_DEPTH:-4}"
        --viz-depth "${VIZ_DEPTH:-6}"
        --viz-seed "${VIZ_SEED:-0}"
    )
elif [ "${PIPELINE}" = "test_strat" ]; then
    RUN_DIR="${PROJECT_DIR}/results/test_strat/${RUN_NAME}"
    read -r -a TEST_STRAT_DATASETS_ARR <<< "${TEST_STRAT_DATASETS:-electricity eye-movements eye-state}"
    read -r -a TEST_STRAT_B_SWEEP_ARR <<< "${TEST_STRAT_B_SWEEP:-32 64 128 256 512 1024}"
    TEST_STRAT_TRIALS="${TEST_STRAT_TRIALS:-50}"
    TEST_STRAT_MIN_SUPPORT="${TEST_STRAT_MIN_SUPPORT:-5}"
    TEST_STRAT_LAMBDA="${TEST_STRAT_LAMBDA:-0.01}"
    TEST_STRAT_MAX_BRANCH_N="${TEST_STRAT_MAX_BRANCH_N:-6}"
    TEST_STRAT_BINNING="${TEST_STRAT_BINNING:-lightgbm}"
    TEST_STRAT_ENABLE_LINEAR_FALLBACK="${TEST_STRAT_ENABLE_LINEAR_FALLBACK:-1}"
    TEST_STRAT_ENABLE_DL85="${TEST_STRAT_ENABLE_DL85:-1}"
    TEST_STRAT_ENABLE_BINOCT="${TEST_STRAT_ENABLE_BINOCT:-1}"
    TEST_STRAT_ENABLE_SNIP="${TEST_STRAT_ENABLE_SNIP:-1}"
    TEST_STRAT_ENABLE_GOSDT="${TEST_STRAT_ENABLE_GOSDT:-1}"
    TEST_STRAT_DYNAMIC_UB="${TEST_STRAT_DYNAMIC_UB:-1}"
    if [ -z "${TEST_STRAT_PLOT_OUT:-}" ]; then
        if [ "${TEST_STRAT_DYNAMIC_UB}" = "1" ]; then
            TEST_STRAT_PLOT_OUT="partition-dp/results/runtime_full_comparison_dynamic_ub.pdf"
        else
            TEST_STRAT_PLOT_OUT="partition-dp/results/runtime_full_comparison_static_ub.pdf"
        fi
    fi
    TEST_STRAT_PLOT_DPI="${TEST_STRAT_PLOT_DPI:-220}"
    TEST_STRAT_LGBM_MIN_DATA_IN_BIN="${TEST_STRAT_LGBM_MIN_DATA_IN_BIN:-1}"

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    PIPELINE_CMD=(
        "${VENV_PYTHON}" partition-dp/benchmark.py
        --datasets "${TEST_STRAT_DATASETS_ARR[@]}"
        --b-sweep "${TEST_STRAT_B_SWEEP_ARR[@]}"
        --dataset-trials "${TEST_STRAT_TRIALS}"
        --dataset-binning "${TEST_STRAT_BINNING}"
        --lgbm-min-data-in-bin "${TEST_STRAT_LGBM_MIN_DATA_IN_BIN}"
        --min-support "${TEST_STRAT_MIN_SUPPORT}"
        --lambda-val "${TEST_STRAT_LAMBDA}"
        --max-branch-n "${TEST_STRAT_MAX_BRANCH_N}"
        --plot-out "${TEST_STRAT_PLOT_OUT}"
        --plot-dpi "${TEST_STRAT_PLOT_DPI}"
    )
    if [ "${TEST_STRAT_ENABLE_LINEAR_FALLBACK}" = "1" ]; then
        PIPELINE_CMD+=(--linear-fallback)
    else
        PIPELINE_CMD+=(--no-linear-fallback)
    fi
    if [ "${TEST_STRAT_ENABLE_DL85}" = "1" ]; then
        PIPELINE_CMD+=(--dl85)
    else
        PIPELINE_CMD+=(--no-dl85)
    fi
    if [ "${TEST_STRAT_ENABLE_BINOCT}" = "1" ]; then
        PIPELINE_CMD+=(--binoct)
    else
        PIPELINE_CMD+=(--no-binoct)
    fi
    if [ "${TEST_STRAT_ENABLE_SNIP}" = "1" ]; then
        PIPELINE_CMD+=(--snip)
    else
        PIPELINE_CMD+=(--no-snip)
    fi
    if [ "${TEST_STRAT_ENABLE_GOSDT}" = "1" ]; then
        PIPELINE_CMD+=(--gosdt)
    else
        PIPELINE_CMD+=(--no-gosdt)
    fi
    if [ "${TEST_STRAT_DYNAMIC_UB}" = "1" ]; then
        PIPELINE_CMD+=(--dynamic-ub)
    else
        PIPELINE_CMD+=(--no-dynamic-ub)
    fi
else
    echo "Invalid PIPELINE='${PIPELINE}'. Use 'cart', 'lightgbm', 'lightgbm_split', 'lightgbm_minimal', 'lightgbm_focus', 'compare', or 'test_strat'." >&2
    exit 1
fi

run_with_profile() {
    local run_dir="$1"
    shift

    local run_parent
    local profile_dir
    local profile_prefix
    local final_profile_prefix
    local start_iso
    local end_iso
    local start_epoch
    local end_epoch
    local elapsed_sec
    local exit_code
    local final_dir_exists=0

    run_parent="$(dirname "${run_dir}")"
    mkdir -p "${run_parent}"
    profile_dir="${run_parent}/.profile_${RUN_NAME}_${SLURM_JOB_ID:-$$}"
    mkdir -p "${profile_dir}"
    profile_prefix="${profile_dir}/run_profile"

    start_iso="$(date -Iseconds)"
    start_epoch="$(date +%s)"
    printf '%q ' "$@" > "${profile_prefix}.cmd.txt"
    printf '\n' >> "${profile_prefix}.cmd.txt"

    if command -v /usr/bin/time >/dev/null 2>&1; then
        set +e
        /usr/bin/time -v -o "${profile_prefix}.time.log" "$@"
        exit_code=$?
        set -e
    else
        set +e
        "$@"
        exit_code=$?
        set -e
        end_epoch="$(date +%s)"
        echo "elapsed_sec=$(( end_epoch - start_epoch ))" > "${profile_prefix}.time.log"
    fi

    end_iso="$(date -Iseconds)"
    end_epoch="$(date +%s)"
    elapsed_sec="$(( end_epoch - start_epoch ))"

    cat > "${profile_prefix}.json" <<EOT
{
  "pipeline": "${PIPELINE}",
  "run_name": "${RUN_NAME}",
  "run_dir": "${run_dir}",
  "host": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_cpus_per_task": "${SLURM_CPUS_PER_TASK:-}",
  "parallel_trials": ${PARALLEL_TRIALS},
  "threads_per_trial": ${THREADS_PER_TRIAL},
  "xgb_num_threads": ${XGB_NUM_THREADS},
  "start_time": "${start_iso}",
  "end_time": "${end_iso}",
  "elapsed_sec": ${elapsed_sec},
  "exit_code": ${exit_code}
}
EOT

    if [ -d "${run_dir}" ]; then
        final_dir_exists=1
        mv "${profile_prefix}.cmd.txt" "${run_dir}/run_profile.cmd.txt"
        mv "${profile_prefix}.time.log" "${run_dir}/run_profile.time.log"
        mv "${profile_prefix}.json" "${run_dir}/run_profile.json"
        final_profile_prefix="${run_dir}/run_profile"
        rmdir "${profile_dir}" 2>/dev/null || true
    else
        final_profile_prefix="${profile_prefix}"
    fi

    PROFILE_PREFIX="${final_profile_prefix}"
    echo "Saved run profile: ${PROFILE_PREFIX}.json"
    echo "Saved time profile: ${PROFILE_PREFIX}.time.log"
    echo "Saved command log: ${PROFILE_PREFIX}.cmd.txt"
    if [ "${final_dir_exists}" -eq 0 ]; then
        echo "Run dir not created by pipeline; profile files kept under ${profile_dir}" >&2
    fi
    return "${exit_code}"
}

print_profile_summary() {
    local profile_prefix="$1"
    local time_log="${profile_prefix}.time.log"
    if [ ! -f "${time_log}" ]; then
        return
    fi

    local wall
    local user
    local sys
    local cpu
    local max_rss_kb
    local max_rss_gb

    wall="$(awk -F': ' '/Elapsed \(wall clock\) time/{print $2; exit}' "${time_log}")"
    user="$(awk -F': ' '/User time \(seconds\)/{print $2; exit}' "${time_log}")"
    sys="$(awk -F': ' '/System time \(seconds\)/{print $2; exit}' "${time_log}")"
    cpu="$(awk -F': ' '/Percent of CPU this job got/{print $2; exit}' "${time_log}")"
    max_rss_kb="$(awk -F': ' '/Maximum resident set size \(kbytes\)/{print $2; exit}' "${time_log}")"

    if [ -z "${wall}" ]; then
        wall="$(awk -F'=' '/^elapsed_sec=/{print $2 "s"; exit}' "${time_log}")"
    fi
    if [ -z "${wall}" ]; then wall="N/A"; fi
    if [ -z "${user}" ]; then user="N/A"; fi
    if [ -z "${sys}" ]; then sys="N/A"; fi
    if [ -z "${cpu}" ]; then cpu="N/A"; fi
    if [ -n "${max_rss_kb}" ]; then
        max_rss_gb="$(awk -v kb="${max_rss_kb}" 'BEGIN{printf "%.2f", kb/1024/1024}')"
    else
        max_rss_kb="N/A"
        max_rss_gb="N/A"
    fi

    echo "=== Run Profile Summary ==="
    echo "wall_time=${wall}"
    echo "user_time_sec=${user}"
    echo "sys_time_sec=${sys}"
    echo "cpu_utilization=${cpu}"
    echo "max_rss_kb=${max_rss_kb}"
    echo "max_rss_gb=${max_rss_gb}"
}

PROFILE_PREFIX=""
mkdir -p "${RUN_DIR}"
if ! run_with_profile "${RUN_DIR}" "${PIPELINE_CMD[@]}"; then
    print_profile_summary "${PROFILE_PREFIX}"
    echo "Pipeline execution failed. Check ${PROFILE_PREFIX}.time.log and ${PROFILE_PREFIX}.json" >&2
    exit 1
fi
print_profile_summary "${PROFILE_PREFIX}"


# Optional tree visualization pass (one seed/depth per dataset).
# Enable with: VISUALIZE_TREES=1 sbatch ...
# Note: compare mode already generates XGBoost + ShapeCART + LightGBM/MSPLIT tree images.
if [ "${VISUALIZE_TREES:-0}" = "1" ]; then
    if [ "${PIPELINE}" = "compare" ]; then
        echo "VISUALIZE_TREES is ignored for PIPELINE=compare (already generated by comparison script)."
        echo "Job ended at $(date)"
        exit 0
    fi

    VIZ_SEED="${VIZ_SEED:-0}"
    VIZ_DEPTH="${VIZ_DEPTH:-3}"
    VIZ_ARGS=(
        --test-size "${TEST_SIZE}"
        --time-limit "${TIME_LIMIT}"
        --lookahead-cap "${LOOKAHEAD_CAP}"
        --max-bins "${MAX_BINS}"
        --min-samples-leaf "${MIN_SAMPLES_LEAF}"
        --min-child-size "${MIN_CHILD_SIZE}"
        --max-branching "${MAX_BRANCHES}"
        --reg "${REG}"
        --xgb-num-threads "${XGB_NUM_THREADS}"
    )

    for DS in "${DATASETS[@]}"; do
        VIZ_PIPELINE="${PIPELINE}"
        if [ "${VIZ_PIPELINE}" = "lightgbm_minimal" ] || [ "${VIZ_PIPELINE}" = "lightgbm_focus" ] || [ "${VIZ_PIPELINE}" = "lightgbm_electricity_d2_4" ]; then
            VIZ_PIPELINE="lightgbm"
        fi
        python visualize_multisplit_tree.py \
            --dataset "${DS}" \
            --pipeline "${VIZ_PIPELINE}" \
            --seed "${VIZ_SEED}" \
            --depth-budget "${VIZ_DEPTH}" \
            "${VIZ_ARGS[@]}"
    done
fi

echo "Job ended at $(date)"
