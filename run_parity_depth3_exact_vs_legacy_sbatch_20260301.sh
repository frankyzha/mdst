#!/bin/bash
#SBATCH --job-name=parity_d3_exact_legacy
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=12:00:00
#SBATCH --output=msdt_parity_d3_%j.out
#SBATCH --error=msdt_parity_d3_%j.err

set -euo pipefail

cd /home/users/yz1075/msdt

OUT_DIR="/home/users/yz1075/msdt/results/debug"
mkdir -p "${OUT_DIR}"

TAG="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-na}"
EXACT_JSON="${OUT_DIR}/parity_depth3_exact_${TAG}.json"
LEGACY_JSON="${OUT_DIR}/parity_depth3_legacy_${TAG}.json"
COMBINED_JSON="${OUT_DIR}/parity_depth3_exact_vs_legacy_${TAG}.json"
export EXACT_JSON LEGACY_JSON COMBINED_JSON

echo "[INFO] TAG=${TAG}"
echo "[INFO] EXACT_JSON=${EXACT_JSON}"
echo "[INFO] LEGACY_JSON=${LEGACY_JSON}"
echo "[INFO] COMBINED_JSON=${COMBINED_JSON}"

# Keep OpenML cache inside the project workspace.
export PYTHONUNBUFFERED=1
export OPENML_CACHE_DIR="/home/users/yz1075/msdt/results/openml_cache"

echo "[INFO] Running exact-lazy side with 64 threads"
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export VECLIB_MAXIMUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
unset MSPLIT_FORCE_RUSH_LEGACY
/usr/bin/time -v .venv/bin/python parity_depth3_exact_legacy_single.py \
  --mode exact \
  --dataset eye-state \
  --seed 0 \
  --depth-budget 3 \
  --time-limit 10800 \
  --lgb-num-threads 1 \
  --out "${EXACT_JSON}"

echo "[INFO] Running forced-legacy side with 32 threads"
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export VECLIB_MAXIMUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export MSPLIT_FORCE_RUSH_LEGACY=1
/usr/bin/time -v .venv/bin/python parity_depth3_exact_legacy_single.py \
  --mode legacy \
  --dataset eye-state \
  --seed 0 \
  --depth-budget 3 \
  --time-limit 10800 \
  --lgb-num-threads 1 \
  --out "${LEGACY_JSON}"

echo "[INFO] Merging exact/legacy results"
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

exact_path = Path(os.environ["EXACT_JSON"])
legacy_path = Path(os.environ["LEGACY_JSON"])
combined_path = Path(os.environ["COMBINED_JSON"])

exact = json.loads(exact_path.read_text(encoding="utf-8"))
legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

e = exact["metrics"]
l = legacy["metrics"]

combined = {
    "trial_params": exact["trial_params"],
    "exact": e,
    "legacy": l,
    "comparison": {
        "obj_diff_legacy_minus_exact": float(l["objective"] - e["objective"]),
        "lb_diff_legacy_minus_exact": float(l["lower_bound"] - e["lower_bound"]),
        "ub_diff_legacy_minus_exact": float(l["upper_bound"] - e["upper_bound"]),
        "train_diff_legacy_minus_exact": float(l["train_accuracy"] - e["train_accuracy"]),
        "test_diff_legacy_minus_exact": float(l["test_accuracy"] - e["test_accuracy"]),
        "fit_sec_diff_legacy_minus_exact": float(l["fit_sec"] - e["fit_sec"]),
        "root_feature_exact": e["root_feature"],
        "root_feature_legacy": l["root_feature"],
        "root_group_exact": e["root_group_count"],
        "root_group_legacy": l["root_group_count"],
    },
    "meta": {
        "dataset": exact["dataset"],
        "seed": exact["seed"],
        "depth_budget": exact["depth_budget"],
        "n_fit": exact["n_fit"],
        "n_val": exact["n_val"],
        "n_test": exact["n_test"],
        "exact_json": str(exact_path),
        "legacy_json": str(legacy_path),
    },
}

combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(json.dumps(combined, indent=2))
print(f"WROTE {combined_path}")
PY

echo "[INFO] Done."
