#!/usr/bin/env bash
#SBATCH --job-name=msdt_tga_elec_localk3_v3
#SBATCH --output=msdt_tga_elec_localk3_v3_%j.out
#SBATCH --error=msdt_tga_elec_localk3_v3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodelist=compsci-cluster-fitz-[45-49]

set -euo pipefail

cd /home/users/yz1075/msdt
source .venv/bin/activate

export PYTHONPATH=/home/users/yz1075/msdt/SPLIT-ICML/split/src:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MSPLIT_DEBUG_REDUCED_K3_BUCKETS=1
export MSPLIT_DEBUG_REDUCED_K3_BUCKET_LIMIT=150000
export MSPLIT_DEBUG_REDUCED_K3_BUCKET_INTERVAL_SEC=60
export MSPLIT_DEBUG_ROOT_ATOMS_ONLY=1

/usr/bin/time -v .venv/bin/python - <<'PY'
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import load_electricity
from experiment_utils import encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner
from split import _libgosdt


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> None:
    root_fn = getattr(_libgosdt, "msplit_debug_teacher_guided_atomcolor_root_feature", None)
    if root_fn is None:
        root_fn = getattr(_libgosdt, "msplit_debug_teacher_atomcolor_root_feature")
    local_fn = getattr(_libgosdt, "msplit_debug_teacher_guided_atomcolor_exact_bnb_synthetic")

    log("loading electricity")
    X, y = load_electricity()
    X = pd.DataFrame(X).copy()
    y_arr = encode_binary_target(y, "electricity")

    log("splitting train/val")
    X_train, _, y_train, _ = train_test_split(
        X, y_arr, test_size=0.2, random_state=0, stratify=y_arr
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )

    log("preprocessing fit")
    preprocessor = make_preprocessor(X_fit)
    X_fit_proc = np.ascontiguousarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    log("preprocessing val")
    X_val_proc = np.ascontiguousarray(preprocessor.transform(X_val), dtype=np.float32)

    log("binner fit start")
    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=255,
        min_samples_leaf=8,
        random_state=0,
        n_estimators=2000,
        num_leaves=255,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        max_depth=-1,
        min_data_in_bin=1,
        min_data_in_leaf=2,
        lambda_l2=0.0,
        early_stopping_rounds=50,
        num_threads=6,
        device_type="cpu",
        ensemble_runs=1,
        ensemble_feature_fraction=1.0,
        ensemble_bagging_fraction=1.0,
        ensemble_bagging_freq=0,
        threshold_dedup_eps=1e-9,
        collect_teacher_logit=True,
    )
    log("binner fit done")

    Z_fit = np.ascontiguousarray(binner.transform(X_fit_proc), dtype=np.int32)

    out = {}
    for feature in [0, 1, 4]:
        log(f"feature {feature} atoms-only debug start")
        raw = root_fn(
            z=Z_fit,
            y=y_fit,
            sample_weight=None,
            feature=feature,
            depth_remaining=3,
            full_depth_budget=3,
            lookahead_depth_budget=2,
            regularization=0.0,
            branch_penalty=0.0,
            min_child_size=8,
            min_atom_size=32,
            time_limit_seconds=120.0,
            max_branching=3,
            partition_strategy=1,
            approx_mode=True,
            approx_feature_scan_limit=1,
            approx_ref_shortlist_enabled=False,
            approx_ref_widen_max=0,
            approx_challenger_sweep_enabled=False,
            approx_challenger_sweep_max_features=0,
            approx_challenger_sweep_max_patch_calls_per_node=0,
            approx_distilled_mode=True,
            approx_distilled_alpha=0.0,
            approx_distilled_max_depth=2,
            approx_distilled_geometry_mode=7,
            approx_score_order_enabled=False,
            teacher_logit=getattr(binner, "teacher_train_logit", None),
            teacher_boundary_gain=getattr(binner, "boundary_gain_per_feature", None),
            teacher_boundary_cover=getattr(binner, "boundary_cover_per_feature", None),
            teacher_boundary_value_jump=getattr(binner, "boundary_value_jump_per_feature", None),
            teacher_boundary_left_delta=None,
            teacher_boundary_right_delta=None,
            teacher_boundary_left_conf=None,
            teacher_boundary_right_conf=None,
        )
        log(f"feature {feature} atoms-only debug done")
        payload = json.loads(raw)
        atoms = payload["atoms"]
        row_counts = np.ascontiguousarray([int(a["row_count"]) for a in atoms], dtype=np.int32)
        pos_w = np.ascontiguousarray([float(a["pos_w"]) for a in atoms], dtype=np.float64)
        neg_w = np.ascontiguousarray([float(a["neg_w"]) for a in atoms], dtype=np.float64)

        log(f"feature {feature} direct local g=3 start")
        local_raw = local_fn(
            row_counts=row_counts,
            pos_w=pos_w,
            neg_w=neg_w,
            groups=3,
        )
        log(f"feature {feature} direct local g=3 done")
        out[str(feature)] = {
            "feature": feature,
            "atom_count": len(atoms),
            "local_g3": json.loads(local_raw),
        }

    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
PY
