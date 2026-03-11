#!/usr/bin/env python3
"""Run one parity side (exact-lazy or forced legacy) on eye-state depth-3 config."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import openml
from sklearn.model_selection import train_test_split

from experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner
from split import MSPLIT


FIXED_TEST_FRACTION = 0.20
FIXED_VAL_WITHIN_TRAIN = 0.10 / (1.0 - FIXED_TEST_FRACTION)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one parity side (exact-lazy or forced legacy).")
    p.add_argument("--mode", choices=["exact", "legacy"], required=True)
    p.add_argument("--dataset", type=str, default="eye-state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-budget", type=int, default=3)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--time-limit", type=float, default=10800.0)
    p.add_argument("--openml-cache-dir", type=str, default="results/openml_cache")
    p.add_argument("--lgb-num-threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.openml_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(openml, "config") and hasattr(openml.config, "set_root_cache_directory"):
        openml.config.set_root_cache_directory(str(cache_dir))

    trial_params = {
        "lookahead_depth_budget": 2,
        "max_bins": 96,
        "min_samples_leaf": 525,
        "min_child_size": 525,
        "leaf_frac": 0.05,
        "max_branching": 3,
        "reg": 1e-05,
        "lgb_n_estimators": 10000,
        "lgb_num_leaves": 48,
        "lgb_learning_rate": 0.04,
        "lgb_feature_fraction": 0.9,
        "lgb_bagging_fraction": 0.9,
        "lgb_bagging_freq": 1,
        "lgb_max_depth": -1,
        "lgb_min_data_in_bin": 1,
        "lgb_min_data_in_leaf": 262,
        "lgb_lambda_l2": 0.1,
    }

    X, y = DATASET_LOADERS[args.dataset]()
    y_bin = encode_binary_target(y, args.dataset)

    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    idx_train_all, idx_test = train_test_split(
        all_idx,
        test_size=FIXED_TEST_FRACTION,
        random_state=args.seed,
        stratify=y_bin,
    )
    y_train_all = y_bin[idx_train_all]
    idx_fit, idx_val = train_test_split(
        idx_train_all,
        test_size=FIXED_VAL_WITHIN_TRAIN,
        random_state=args.seed,
        stratify=y_train_all,
    )

    X_fit = X.iloc[idx_fit] if hasattr(X, "iloc") else X[idx_fit]
    X_val = X.iloc[idx_val] if hasattr(X, "iloc") else X[idx_val]
    X_test = X.iloc[idx_test] if hasattr(X, "iloc") else X[idx_test]
    y_fit = np.ascontiguousarray(y_bin[idx_fit], dtype=np.int32)
    y_val = np.ascontiguousarray(y_bin[idx_val], dtype=np.int32)
    y_test = np.ascontiguousarray(y_bin[idx_test], dtype=np.int32)

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.ascontiguousarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.ascontiguousarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.ascontiguousarray(pre.transform(X_test), dtype=np.float32)

    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=int(trial_params["max_bins"]),
        min_samples_leaf=int(trial_params["min_samples_leaf"]),
        random_state=int(args.seed),
        n_estimators=int(trial_params["lgb_n_estimators"]),
        num_leaves=int(trial_params["lgb_num_leaves"]),
        learning_rate=float(trial_params["lgb_learning_rate"]),
        feature_fraction=float(trial_params["lgb_feature_fraction"]),
        bagging_fraction=float(trial_params["lgb_bagging_fraction"]),
        bagging_freq=int(trial_params["lgb_bagging_freq"]),
        max_depth=int(trial_params["lgb_max_depth"]),
        min_data_in_bin=int(trial_params["lgb_min_data_in_bin"]),
        min_data_in_leaf=int(trial_params["lgb_min_data_in_leaf"]),
        lambda_l2=float(trial_params["lgb_lambda_l2"]),
        early_stopping_rounds=100,
        num_threads=max(1, int(args.lgb_num_threads)),
        device_type="cpu",
    )

    Z_fit = binner.transform(X_fit_proc)
    Z_test = binner.transform(X_test_proc)

    if args.mode == "legacy":
        os.environ["MSPLIT_FORCE_RUSH_LEGACY"] = "1"
    else:
        os.environ.pop("MSPLIT_FORCE_RUSH_LEGACY", None)

    model = MSPLIT(
        lookahead_depth_budget=int(trial_params["lookahead_depth_budget"]),
        full_depth_budget=int(args.depth_budget),
        reg=float(trial_params["reg"]),
        branch_penalty=0.0,
        max_bins=int(trial_params["max_bins"]),
        min_samples_leaf=int(trial_params["min_samples_leaf"]),
        min_child_size=int(trial_params["min_child_size"]),
        max_branching=int(trial_params["max_branching"]),
        time_limit=float(args.time_limit),
        verbose=False,
        random_state=int(args.seed),
        input_is_binned=True,
        use_cpp_solver=True,
        interval_partition_solver="rush_dp",
    )

    fit_t0 = time.time()
    model.fit(Z_fit, y_fit)
    fit_sec = time.time() - fit_t0

    yhat_train = model.predict(Z_fit).astype(np.int32)
    yhat_test = model.predict(Z_test).astype(np.int32)

    tree = model.tree_
    root_feature = int(tree.feature) if hasattr(tree, "feature") else None
    root_group_count = int(tree.group_count) if hasattr(tree, "group_count") else None
    root_child_groups = len(tree.children) if hasattr(tree, "children") else 0

    payload = {
        "mode": args.mode,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "depth_budget": int(args.depth_budget),
        "n_fit": int(y_fit.shape[0]),
        "n_val": int(y_val.shape[0]),
        "n_test": int(y_test.shape[0]),
        "trial_params": trial_params,
        "metrics": {
            "fit_sec": float(fit_sec),
            "objective": float(model.objective_),
            "lower_bound": float(model.lower_bound_),
            "upper_bound": float(model.upper_bound_),
            "train_accuracy": float(np.mean(yhat_train == y_fit)),
            "test_accuracy": float(np.mean(yhat_test == y_test)),
            "root_feature": root_feature,
            "root_group_count": root_group_count,
            # Historical field name retained for backward compatibility.
            # In grouped representation this is the number of logical child groups.
            "root_child_bins": int(root_child_groups),
            "root_child_groups": int(root_child_groups),
            "dp_interval_evals": int(getattr(model, "dp_interval_evals_", 0)),
            "dp_subproblem_calls": int(getattr(model, "dp_subproblem_calls_", 0)),
            "dp_unique_states": int(getattr(model, "dp_unique_states_", 0)),
            "rush_refinement_child_calls": int(getattr(model, "rush_refinement_child_calls_", 0)),
            "rush_refinement_recursive_calls": int(getattr(model, "rush_refinement_recursive_calls_", 0)),
            "rush_refinement_recursive_unique_states": int(
                getattr(model, "rush_refinement_recursive_unique_states_", 0)
            ),
            "rush_ub_rescue_picks": int(getattr(model, "rush_ub_rescue_picks_", 0)),
            "rush_global_fallback_picks": int(getattr(model, "rush_global_fallback_picks_", 0)),
            "rush_incumbent_feature_aborts": int(getattr(model, "rush_incumbent_feature_aborts_", 0)),
            "rush_total_time_sec": float(getattr(model, "rush_total_time_sec_", 0.0)),
            "rush_profile_enabled": int(getattr(model, "rush_profile_enabled_", 0)),
            "rush_profile_ub0_ordering_sec": float(getattr(model, "rush_profile_ub0_ordering_sec_", 0.0)),
            "rush_profile_exact_lazy_eval_sec": float(getattr(model, "rush_profile_exact_lazy_eval_sec_", 0.0)),
            "rush_profile_exact_lazy_eval_exclusive_sec": float(
                getattr(model, "rush_profile_exact_lazy_eval_exclusive_sec_", 0.0)
            ),
            "rush_profile_exact_lazy_eval_sec_depth0": float(
                getattr(model, "rush_profile_exact_lazy_eval_sec_depth0_", 0.0)
            ),
            "rush_profile_exact_lazy_eval_exclusive_sec_depth0": float(
                getattr(model, "rush_profile_exact_lazy_eval_exclusive_sec_depth0_", 0.0)
            ),
            "rush_profile_exact_lazy_table_init_sec": float(
                getattr(model, "rush_profile_exact_lazy_table_init_sec_", 0.0)
            ),
            "rush_profile_exact_lazy_dp_recompute_sec": float(
                getattr(model, "rush_profile_exact_lazy_dp_recompute_sec_", 0.0)
            ),
            "rush_profile_exact_lazy_child_solve_sec": float(
                getattr(model, "rush_profile_exact_lazy_child_solve_sec_", 0.0)
            ),
            "rush_profile_exact_lazy_child_solve_sec_depth0": float(
                getattr(model, "rush_profile_exact_lazy_child_solve_sec_depth0_", 0.0)
            ),
            "rush_profile_exact_lazy_closure_sec": float(getattr(model, "rush_profile_exact_lazy_closure_sec_", 0.0)),
            "rush_profile_exact_lazy_dp_recompute_calls": int(
                getattr(model, "rush_profile_exact_lazy_dp_recompute_calls_", 0)
            ),
            "rush_profile_exact_lazy_closure_passes": int(
                getattr(model, "rush_profile_exact_lazy_closure_passes_", 0)
            ),
        },
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
