#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
THREADS="${SPLIT_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS}}"

# Keep LightGBM bounded in shared compute environments.
export LIGHTGBM_NUM_THREADS="${LIGHTGBM_NUM_THREADS:-1}"

RUN_NAME="${RUN_NAME:-vcm_lightgbm_$(date +%Y%m%d_%H%M%S)}"
echo "[run_vcm_multisplit_lightgbm] run_name=${RUN_NAME}"

echo "[run_vcm_multisplit_lightgbm] threads=${THREADS}, lgb_threads=${LIGHTGBM_NUM_THREADS}, MPLBACKEND=${MPLBACKEND}"
python run_multisplit_experiments_lightgbm.py --run-name "${RUN_NAME}" --lgb-num-threads "${LIGHTGBM_NUM_THREADS}" "$@"

RUN_DIR="${ROOT_DIR}/results/runs_lightgbm/${RUN_NAME}"
BUNDLE="${RUN_DIR}/${RUN_NAME}_artifacts.tar.gz"

echo "[run_vcm_multisplit_lightgbm] run_dir=${RUN_DIR}"
if [[ -f "${BUNDLE}" ]]; then
  echo "[run_vcm_multisplit_lightgbm] bundle=${BUNDLE}"
fi
