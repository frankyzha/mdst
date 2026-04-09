"""Compare LightGBM-binned MSPLIT against XGBoost and ShapeCART."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor
from tree_artifact_utils import build_xgb_artifact, write_artifact_json

try:
    import optuna
except Exception:  # optional unless --optuna-enable
    optuna = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LightGBM-binned MSPLIT vs XGBoost and ShapeCART on OpenML datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS.keys()),
        default=["electricity", "eye-movements", "eye-state"],
    )
    parser.add_argument("--depth-budgets", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=3000.0)
    parser.add_argument("--lookahead-cap", type=int, default=3)
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument(
        "--leaf-frac",
        type=float,
        default=0.0,
        help="Optional leaf-fraction constraint passed to LightGBM+MSPLIT branch.",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=8)
    parser.add_argument("--min-split-size", type=int, default=0)
    parser.add_argument("--max-branching", type=int, default=0)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument(
        "--msplit-variant",
        choices=["optimal_dp", "rush_dp"],
        default="rush_dp",
        help="MSPLIT interval partitioning variant for the LightGBM+MSPLIT branch.",
    )
    parser.add_argument(
        "--approx-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable MSPLIT approx controller in the LightGBM+MSPLIT branch.",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help="Parallel trial processes for LightGBM-binned MSPLIT experiment script.",
    )
    parser.add_argument(
        "--threads-per-trial",
        type=int,
        default=1,
        help="Thread cap for each LightGBM-binned MSPLIT trial process.",
    )
    parser.add_argument(
        "--optuna-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Optuna HPO inside LightGBM-binned MSPLIT experiment script.",
    )
    parser.add_argument("--optuna-trials", type=int, default=5)
    parser.add_argument("--optuna-val-size", type=float, default=0.2)
    parser.add_argument(
        "--paper-split-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict split parity across algorithms: "
            "train/test split, then train->fit/val split. "
            "Final test fit uses fit split."
        ),
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=0,
    )
    parser.add_argument("--optuna-timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--optuna-warmstart-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start Optuna studies from historical best_params.csv in child SPLIT runners.",
    )
    parser.add_argument(
        "--optuna-warmstart-root",
        type=str,
        default="results",
        help="Root directory child SPLIT runners scan for warm-start best_params.csv.",
    )
    parser.add_argument(
        "--optuna-warmstart-max-per-study",
        type=int,
        default=8,
        help="Max warm-start trials enqueued per study in child SPLIT runners.",
    )
    parser.add_argument(
        "--optuna-seed-candidates",
        type=int,
        default=4,
        help="Extra hand-crafted candidate trials enqueued per study in child SPLIT runners.",
    )
    parser.add_argument(
        "--cpu-utilization-target",
        type=float,
        default=0.9,
        help="Fraction of requested CPU budget to use in SPLIT runners.",
    )
    parser.add_argument(
        "--optuna-max-active-studies",
        type=int,
        default=0,
        help="Max concurrent Optuna studies in SPLIT runners (0=auto).",
    )
    parser.add_argument(
        "--optuna-max-concurrent-trials",
        type=int,
        default=0,
        help="Max total concurrent Optuna trial evaluations in SPLIT runners (0=auto).",
    )
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-num-threads", type=int, default=4)
    parser.add_argument(
        "--shapecart-k",
        type=int,
        default=3,
        help="Max ShapeCART branching factor k (2=ShapeCART, 3=ShapeCART3).",
    )
    parser.add_argument(
        "--lgb-device-type",
        choices=["cpu", "gpu", "cuda"],
        default="gpu",
        help="LightGBM backend for preprocessing binning in the LightGBM+MSPLIT pipeline.",
    )
    parser.add_argument(
        "--lgb-ensemble-runs",
        type=int,
        default=1,
        help="Number of stochastic LightGBM preprocessing fits to union thresholds across.",
    )
    parser.add_argument("--lgb-ensemble-feature-fraction", type=float, default=0.8)
    parser.add_argument("--lgb-ensemble-bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lgb-ensemble-bagging-freq", type=int, default=1)
    parser.add_argument("--lgb-threshold-dedup-eps", type=float, default=1e-9)
    parser.add_argument("--lgb-gpu-platform-id", type=int, default=0)
    parser.add_argument("--lgb-gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--lgb-gpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If GPU binning fails, retry LightGBM binning on CPU.",
    )
    parser.add_argument(
        "--lgb-max-gpu-jobs",
        type=int,
        default=1,
        help="Max concurrent LightGBM GPU preprocessing jobs across worker processes.",
    )
    parser.add_argument(
        "--lgb-gpu-lock-dir",
        type=str,
        default="/tmp/msdt_lgb_gpu_lock",
        help="Shared lock directory used to enforce --lgb-max-gpu-jobs.",
    )
    parser.add_argument(
        "--xgb-parallel-trials",
        type=int,
        default=0,
        help="Parallel XGBoost trial workers (0=auto from parallel-trials and xgb-num-threads).",
    )
    parser.add_argument(
        "--viz-select",
        choices=["best", "fixed"],
        default="best",
        help="Tree visualization selection mode. 'best' picks best (depth, seed) per dataset/pipeline.",
    )
    parser.add_argument(
        "--viz-min-depth",
        type=int,
        default=4,
        help="Minimum depth considered when --viz-select=best; falls back if no feasible run.",
    )
    parser.add_argument(
        "--viz-depth",
        type=int,
        default=None,
        help="Fallback depth budget for visualization when no run can be selected (or when --viz-select=fixed).",
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=None,
        help="Fallback seed for visualization when no run can be selected (or when --viz-select=fixed).",
    )
    parser.add_argument("--results-root", type=str, default="results/comparison")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-split-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume existing LightGBM+MSPLIT child run directory if present.",
    )

    args = parser.parse_args()
    args.depth_budgets = sorted(set(args.depth_budgets))
    args.seeds = sorted(set(args.seeds))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.parallel_trials = max(1, int(args.parallel_trials))
    args.threads_per_trial = max(1, int(args.threads_per_trial))
    args.xgb_num_threads = max(1, int(args.xgb_num_threads))
    args.shapecart_k = max(2, int(args.shapecart_k))
    args.xgb_parallel_trials = max(0, int(args.xgb_parallel_trials))
    args.optuna_trials = max(1, int(args.optuna_trials))
    args.optuna_timeout_sec = max(0.0, float(args.optuna_timeout_sec))
    args.optuna_warmstart_max_per_study = max(0, int(args.optuna_warmstart_max_per_study))
    args.optuna_seed_candidates = max(0, int(args.optuna_seed_candidates))
    args.optuna_max_active_studies = max(0, int(args.optuna_max_active_studies))
    args.optuna_max_concurrent_trials = max(0, int(args.optuna_max_concurrent_trials))
    args.cpu_utilization_target = min(1.0, max(0.05, float(args.cpu_utilization_target)))
    args.lgb_max_gpu_jobs = max(1, int(args.lgb_max_gpu_jobs))
    args.lgb_ensemble_runs = max(1, int(args.lgb_ensemble_runs))
    args.lgb_ensemble_feature_fraction = min(1.0, max(1e-6, float(args.lgb_ensemble_feature_fraction)))
    args.lgb_ensemble_bagging_fraction = min(1.0, max(1e-6, float(args.lgb_ensemble_bagging_fraction)))
    args.lgb_ensemble_bagging_freq = max(0, int(args.lgb_ensemble_bagging_freq))
    args.lgb_threshold_dedup_eps = max(0.0, float(args.lgb_threshold_dedup_eps))
    if not 0.0 < float(args.optuna_val_size) < 1.0:
        raise ValueError(f"--optuna-val-size must be in (0, 1), got {args.optuna_val_size}")
    args.leaf_frac = max(0.0, float(args.leaf_frac))
    args.viz_min_depth = max(1, int(args.viz_min_depth))
    if args.viz_depth is None:
        args.viz_depth = int(max(args.depth_budgets))
    else:
        args.viz_depth = max(1, int(args.viz_depth))
    if args.viz_seed is None:
        args.viz_seed = int(args.seeds[0])
    else:
        args.viz_seed = int(args.viz_seed)
    return args


def _run_split_pipeline(
    script_name: str,
    run_name: str,
    args: argparse.Namespace,
    tmp_root: Path,
    stable_dir: Path,
    tree_artifacts_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    results_root = tmp_root / run_name
    run_dir = results_root / run_name

    cmd = [
        sys.executable,
        script_name,
        "--run-name",
        run_name,
        "--datasets",
        *args.datasets,
        "--depth-budgets",
        *[str(v) for v in args.depth_budgets],
        "--seeds",
        *[str(v) for v in args.seeds],
        "--test-size",
        str(args.test_size),
        "--time-limit",
        str(args.time_limit),
        "--lookahead-cap",
        str(args.lookahead_cap),
        "--max-bins",
        str(args.max_bins),
        "--leaf-frac",
        str(args.leaf_frac),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--min-child-size",
        str(args.min_child_size),
        "--min-split-size",
        str(args.min_split_size),
        "--max-branching",
        str(args.max_branching),
        "--reg",
        str(args.reg),
        "--msplit-variant",
        str(args.msplit_variant),
        "--parallel-trials",
        str(args.parallel_trials),
        "--threads-per-trial",
        str(args.threads_per_trial),
        "--optuna-trials",
        str(args.optuna_trials),
        "--optuna-val-size",
        str(args.optuna_val_size),
        "--optuna-seed",
        str(args.optuna_seed),
        "--optuna-timeout-sec",
        str(args.optuna_timeout_sec),
        "--optuna-warmstart-root",
        str(args.optuna_warmstart_root),
        "--optuna-warmstart-max-per-study",
        str(args.optuna_warmstart_max_per_study),
        "--optuna-seed-candidates",
        str(args.optuna_seed_candidates),
        "--cpu-utilization-target",
        str(args.cpu_utilization_target),
        "--optuna-max-active-studies",
        str(args.optuna_max_active_studies),
        "--optuna-max-concurrent-trials",
        str(args.optuna_max_concurrent_trials),
        "--results-root",
        str(results_root),
        "--stable-results-dir",
        str(stable_dir),
        "--no-package-artifacts",
    ]
    cmd.append("--optuna-enable" if args.optuna_enable else "--no-optuna-enable")
    cmd.append("--optuna-warmstart-enable" if args.optuna_warmstart_enable else "--no-optuna-warmstart-enable")
    cmd.append("--paper-split-protocol" if args.paper_split_protocol else "--no-paper-split-protocol")
    if bool(getattr(args, "resume_split_run", False)):
        cmd.append("--resume")
    if args.max_trials > 0:
        cmd += ["--max-trials", str(args.max_trials)]
    if tree_artifacts_dir is not None:
        tree_artifacts_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--tree-artifacts-dir", str(tree_artifacts_dir)]
    if extra_args:
        cmd += list(extra_args)

    print(f"[compare] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    summary_df = pd.read_csv(run_dir / "summary_results.csv")
    seed_df = pd.read_csv(run_dir / "seed_results.csv")
    return summary_df, seed_df, run_dir


def _slice_rows(x, idx: np.ndarray):
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _persist_optuna_outputs(src_run_dir: Path, out_dir: Path, pipeline: str) -> Path | None:
    src_optuna = src_run_dir / "optuna"
    if not src_optuna.exists():
        return None
    dst_optuna = out_dir / "optuna_history" / pipeline
    if dst_optuna.exists():
        shutil.rmtree(dst_optuna, ignore_errors=True)
    dst_optuna.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_optuna, dst_optuna)
    return dst_optuna


def _study_seed(dataset_name: str, depth_budget: int, base_seed: int) -> int:
    dataset_bias = sum(ord(ch) for ch in str(dataset_name))
    return int(base_seed) + int(depth_budget) * 1000 + dataset_bias


def _protocol_split_indices(
    y_bin: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, np.ndarray]:
    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y_bin,
    )
    y_train = y_bin[idx_train]
    idx_fit, idx_val = train_test_split(
        idx_train,
        test_size=float(val_size),
        random_state=int(seed),
        stratify=y_train,
    )
    return {
        "idx_train": np.asarray(idx_train, dtype=np.int32),
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
    }


def _resolve_worker_count(
    *,
    num_tasks: int,
    base_parallel_trials: int,
    threads_per_worker: int,
    explicit_workers: int = 0,
) -> int:
    if num_tasks <= 0:
        return 1
    if explicit_workers > 0:
        workers = int(explicit_workers)
    else:
        workers = max(1, int(base_parallel_trials) // max(1, int(threads_per_worker)))
    return max(1, min(int(num_tasks), workers))


def _run_xgboost_reference(args: argparse.Namespace, tree_artifacts_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def _fit_xgb_trial(
        dataset_name: str,
        payload: dict[str, object],
        depth: int,
        seed: int,
    ) -> dict[str, object]:
        X = payload["X"]
        y_bin = np.asarray(payload["y_bin"], dtype=np.int32)
        class_labels = np.asarray(payload["class_labels"], dtype=object)
        target_name = str(payload.get("target_name", "target"))

        split_idx = split_idx_cache[(str(dataset_name), int(seed))]
        idx_train = split_idx["idx_train"]
        idx_fit = split_idx["idx_fit"]
        idx_test = split_idx["idx_test"]
        idx_train_used = idx_fit if bool(args.paper_split_protocol) else idx_train

        X_train = _slice_rows(X, idx_train_used)
        X_test = _slice_rows(X, idx_test)
        y_train = y_bin[idx_train_used]
        y_test = y_bin[idx_test]
        pre = make_preprocessor(X_train)
        X_train_proc = np.asarray(pre.fit_transform(X_train), dtype=np.float32)
        X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)
        try:
            feature_names = pre.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"x{i}" for i in range(X_train_proc.shape[1])]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=int(depth),
            random_state=int(seed),
            n_jobs=int(args.xgb_num_threads),
        )
        model.fit(X_train_proc, y_train)
        pred_train = model.predict(X_train_proc).astype(np.int32)
        train_accuracy = float(np.mean(pred_train == y_train))
        pred = model.predict(X_test_proc).astype(np.int32)
        accuracy = float(np.mean(pred == y_test))

        artifact_path_text = ""
        if tree_artifacts_dir is not None:
            artifact_path = tree_artifacts_dir / dataset_name / f"depth_{int(depth)}" / f"seed_{int(seed)}.json"
            artifact = build_xgb_artifact(
                dataset=dataset_name,
                target_name=target_name,
                class_labels=class_labels,
                feature_names=feature_names,
                accuracy=accuracy,
                seed=int(seed),
                test_size=float(args.test_size),
                depth_budget=int(depth),
                n_estimators=int(args.xgb_n_estimators),
                learning_rate=float(args.xgb_learning_rate),
                num_threads=int(args.xgb_num_threads),
                model=model,
                x_train=X_train_proc,
                y_train=y_train,
                train_indices=idx_train_used,
                test_indices=idx_test,
            )
            write_artifact_json(artifact_path, artifact)
            artifact_path_text = str(artifact_path)
        return {
            "dataset": dataset_name,
            "depth_budget": int(depth),
            "seed": int(seed),
            "xgboost_train_accuracy": train_accuracy,
            "xgboost_accuracy": accuracy,
            "tree_artifact_path": artifact_path_text,
        }

    dataset_payload: dict[str, dict[str, object]] = {}
    rows = []
    for dataset_name in args.datasets:
        print(f"[compare] XGBoost dataset={dataset_name}", flush=True)
        X, y = DATASET_LOADERS[dataset_name]()
        y_bin = encode_binary_target(y, dataset_name)
        class_labels = np.array(sorted(np.unique(np.asarray(y)).tolist(), key=lambda v: str(v)), dtype=object)
        target_name = getattr(y, "name", None) or "target"
        dataset_payload[dataset_name] = {
            "X": X,
            "y_bin": y_bin,
            "class_labels": class_labels,
            "target_name": str(target_name),
        }

    tasks: list[tuple[str, int, int]] = []
    for dataset_name in args.datasets:
        for depth in args.depth_budgets:
            for seed in args.seeds:
                tasks.append((dataset_name, int(depth), int(seed)))

    for dataset_name in args.datasets:
        y_bin = np.asarray(dataset_payload[dataset_name]["y_bin"], dtype=np.int32)
        for seed in args.seeds:
            split_idx_cache[(str(dataset_name), int(seed))] = _protocol_split_indices(
                y_bin=y_bin,
                seed=int(seed),
                test_size=float(args.test_size),
                val_size=float(args.optuna_val_size),
            )

    xgb_workers = _resolve_worker_count(
        num_tasks=len(tasks),
        base_parallel_trials=int(args.parallel_trials),
        threads_per_worker=int(args.xgb_num_threads),
        explicit_workers=int(args.xgb_parallel_trials),
    )
    print(
        (
            f"[compare] XGBoost parallel workers={xgb_workers}, "
            f"threads_per_trial={args.xgb_num_threads}, trials={len(tasks)}"
        ),
        flush=True,
    )

    if xgb_workers == 1:
        for dataset_name, depth, seed in tasks:
            rows.append(_fit_xgb_trial(dataset_name, dataset_payload[dataset_name], depth, seed))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=xgb_workers) as executor:
            future_map = {}
            for dataset_name, depth, seed in tasks:
                future = executor.submit(_fit_xgb_trial, dataset_name, dataset_payload[dataset_name], depth, seed)
                future_map[future] = (dataset_name, depth, seed)
            for future in concurrent.futures.as_completed(future_map):
                rows.append(future.result())

    seed_df = pd.DataFrame(rows)
    summary_df = (
        seed_df.groupby(["dataset", "depth_budget"], as_index=False)
        .agg(
            xgboost_mean_train_accuracy=("xgboost_train_accuracy", "mean"),
            xgboost_mean_accuracy=("xgboost_accuracy", "mean"),
            xgboost_std_accuracy=("xgboost_accuracy", "std"),
            xgboost_n_success=("xgboost_accuracy", "count"),
        )
        .sort_values(["dataset", "depth_budget"])
        .reset_index(drop=True)
    )
    summary_df["xgboost_std_accuracy"] = summary_df["xgboost_std_accuracy"].fillna(0.0)
    return seed_df, summary_df


def _run_shapecart_reference(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_root = (Path(__file__).resolve().parent / "Empowering-DTs-via-Shape-Functions").resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"ShapeCART repo not found at {repo_root}. Expected cloned repo under msdt/Empowering-DTs-via-Shape-Functions."
        )
    if args.optuna_enable and optuna is None:
        raise RuntimeError("Optuna is not installed but --optuna-enable was set.")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore

    def _default_params(depth: int, seed: int) -> dict[str, object]:
        return {
            "max_depth": int(depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "min_samples_split": max(2, int(args.min_child_size)),
            "inner_max_depth": 6,
            "inner_max_leaf_nodes": 32,
            "max_iter": 20,
            "branching_penalty": 0.0,
            "random_state": int(seed),
            "k": int(args.shapecart_k),
            "verbose": False,
        }

    def _sample_optuna_params(trial, depth: int, random_state: int) -> dict[str, object]:
        leaf_hi = max(2, int(args.min_samples_leaf) * 2)
        split_hi = max(4, int(args.min_child_size) * 2)
        return {
            "max_depth": int(depth),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, leaf_hi),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, split_hi),
            "inner_max_depth": trial.suggest_int("inner_max_depth", 3, 10),
            "inner_max_leaf_nodes": trial.suggest_categorical("inner_max_leaf_nodes", [16, 32, 64, 128]),
            "max_iter": trial.suggest_int("max_iter", 8, 30),
            "branching_penalty": 0.0,
            "random_state": int(random_state),
            "k": int(args.shapecart_k),
            "verbose": False,
        }

    dataset_payload: dict[str, dict[str, object]] = {}
    split_idx_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    hpo_by_dataset_depth: dict[tuple[str, int], dict[str, object]] = {}
    rows = []

    for dataset_name in args.datasets:
        print(f"[compare] ShapeCART dataset={dataset_name}", flush=True)
        X, y = DATASET_LOADERS[dataset_name]()
        y_bin = encode_binary_target(y, dataset_name)
        dataset_payload[dataset_name] = {
            "X": X,
            "y_bin": y_bin,
        }
        for seed in args.seeds:
            split_idx_cache[(str(dataset_name), int(seed))] = _protocol_split_indices(
                y_bin=np.asarray(y_bin, dtype=np.int32),
                seed=int(seed),
                test_size=float(args.test_size),
                val_size=float(args.optuna_val_size),
            )

    for dataset_name in args.datasets:
        payload = dataset_payload[dataset_name]
        X = payload["X"]
        y_bin = np.asarray(payload["y_bin"], dtype=np.int32)
        for depth in args.depth_budgets:
            depth_i = int(depth)
            best_params = _default_params(depth=depth_i, seed=int(args.optuna_seed))
            best_val_accuracy = np.nan
            optuna_trials_used = 0

            if args.optuna_enable:
                study_seed = _study_seed(dataset_name, depth_i, int(args.optuna_seed))
                split_idx = _protocol_split_indices(
                    y_bin=y_bin,
                    seed=study_seed,
                    test_size=float(args.test_size),
                    val_size=float(args.optuna_val_size),
                )
                idx_fit = split_idx["idx_fit"]
                idx_val = split_idx["idx_val"]
                X_fit = _slice_rows(X, idx_fit)
                X_val = _slice_rows(X, idx_val)
                y_fit = y_bin[idx_fit]
                y_val = y_bin[idx_val]

                pre_val = make_preprocessor(X_fit)
                X_fit_proc = np.asarray(pre_val.fit_transform(X_fit), dtype=np.float32)
                X_val_proc = np.asarray(pre_val.transform(X_val), dtype=np.float32)
                sampler = optuna.samplers.TPESampler(seed=study_seed)
                study = optuna.create_study(direction="maximize", sampler=sampler)

                def _objective(trial) -> float:
                    trial_params = _sample_optuna_params(trial, depth=depth_i, random_state=study_seed)
                    model_inner = ShapeCARTClassifier(**trial_params)
                    try:
                        model_inner.fit(X_fit_proc, y_fit)
                        pred_val = np.asarray(model_inner.predict(X_val_proc), dtype=np.int32)
                        return float(np.mean(pred_val == y_val))
                    except Exception:
                        return 0.0

                timeout = float(args.optuna_timeout_sec) if float(args.optuna_timeout_sec) > 0 else None
                study.optimize(_objective, n_trials=int(args.optuna_trials), timeout=timeout)
                optuna_trials_used = len(study.trials)
                if optuna_trials_used > 0:
                    best_params.update(study.best_params)
                    best_params["branching_penalty"] = 0.0
                    best_val_accuracy = float(study.best_value)

            hpo_by_dataset_depth[(str(dataset_name), depth_i)] = {
                "best_params": dict(best_params),
                "val_accuracy": float(best_val_accuracy) if np.isfinite(best_val_accuracy) else np.nan,
                "optuna_trials": int(optuna_trials_used),
            }

    def _fit_shapecart_trial(
        dataset_name: str,
        payload: dict[str, object],
        depth: int,
        seed: int,
    ) -> dict[str, object]:
        X = payload["X"]
        y_bin = np.asarray(payload["y_bin"], dtype=np.int32)
        split_idx = split_idx_cache[(str(dataset_name), int(seed))]
        idx_train = split_idx["idx_train"]
        idx_fit = split_idx["idx_fit"]
        idx_test = split_idx["idx_test"]
        idx_train_used = idx_fit if bool(args.paper_split_protocol) else idx_train

        X_train = _slice_rows(X, idx_train_used)
        X_test = _slice_rows(X, idx_test)
        y_train = y_bin[idx_train_used]
        y_test = y_bin[idx_test]

        shared_hpo = hpo_by_dataset_depth[(str(dataset_name), int(depth))]
        best_params = dict(shared_hpo["best_params"])
        best_params["random_state"] = int(seed)
        best_params["branching_penalty"] = 0.0
        best_val_accuracy = float(shared_hpo["val_accuracy"]) if np.isfinite(shared_hpo["val_accuracy"]) else np.nan
        optuna_trials_used = int(shared_hpo["optuna_trials"])

        pre = make_preprocessor(X_train)
        X_train_proc = np.asarray(pre.fit_transform(X_train), dtype=np.float32)
        X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)

        model = ShapeCARTClassifier(**best_params)
        model.fit(X_train_proc, y_train)
        pred_train = model.predict(X_train_proc).astype(np.int32)
        pred_test = model.predict(X_test_proc).astype(np.int32)
        return {
            "dataset": dataset_name,
            "depth_budget": int(depth),
            "seed": int(seed),
            "shapecart_train_accuracy": float(np.mean(pred_train == y_train)),
            "shapecart_accuracy": float(np.mean(pred_test == y_test)),
            "shapecart_val_accuracy": float(best_val_accuracy) if np.isfinite(best_val_accuracy) else np.nan,
            "shapecart_optuna_trials": int(optuna_trials_used),
            "shapecart_optuna_best_params": json.dumps(best_params, sort_keys=True),
        }

    tasks: list[tuple[str, int, int]] = []
    for dataset_name in args.datasets:
        for depth in args.depth_budgets:
            for seed in args.seeds:
                tasks.append((dataset_name, int(depth), int(seed)))

    shapecart_workers = _resolve_worker_count(
        num_tasks=len(tasks),
        base_parallel_trials=int(args.parallel_trials),
        threads_per_worker=int(args.threads_per_trial),
        explicit_workers=int(args.xgb_parallel_trials),
    )
    print(
        (
            f"[compare] ShapeCART parallel workers={shapecart_workers}, "
            f"trials={len(tasks)}"
        ),
        flush=True,
    )

    if shapecart_workers == 1:
        for dataset_name, depth, seed in tasks:
            rows.append(_fit_shapecart_trial(dataset_name, dataset_payload[dataset_name], depth, seed))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=shapecart_workers) as executor:
            future_map = {}
            for dataset_name, depth, seed in tasks:
                future = executor.submit(_fit_shapecart_trial, dataset_name, dataset_payload[dataset_name], depth, seed)
                future_map[future] = (dataset_name, depth, seed)
            for future in concurrent.futures.as_completed(future_map):
                rows.append(future.result())

    seed_df = pd.DataFrame(rows)
    summary_df = (
        seed_df.groupby(["dataset", "depth_budget"], as_index=False)
        .agg(
            shapecart_mean_train_accuracy=("shapecart_train_accuracy", "mean"),
            shapecart_mean_accuracy=("shapecart_accuracy", "mean"),
            shapecart_mean_val_accuracy=("shapecart_val_accuracy", "mean"),
            shapecart_std_accuracy=("shapecart_accuracy", "std"),
            shapecart_n_success=("shapecart_accuracy", "count"),
            shapecart_mean_optuna_trials=("shapecart_optuna_trials", "mean"),
        )
        .sort_values(["dataset", "depth_budget"])
        .reset_index(drop=True)
    )
    summary_df["shapecart_std_accuracy"] = summary_df["shapecart_std_accuracy"].fillna(0.0)
    return seed_df, summary_df


def _plot_accuracy_vs_depth(
    df: pd.DataFrame,
    datasets: list[str],
    depth_budgets: list[int],
    out_path: Path,
    metric: str,
) -> None:
    plt.rcParams.update({"figure.facecolor": "#f7f7f3", "axes.facecolor": "#f7f7f3", "font.family": "DejaVu Sans"})

    metric = str(metric).lower().strip()
    if metric not in {"test", "train"}:
        raise ValueError(f"Unknown metric '{metric}', expected 'test' or 'train'.")

    series_map_test = [
        ("lightgbm_mssplit_accuracy", "MSPLIT (LGB bins)", "#ff1493", "^", 2.8),
        ("xgboost_mean_accuracy", "XGBoost", "#2ca02c", "D", 2.0),
        ("shapecart_mean_accuracy", "ShapeCART", "#ff8c00", "s", 2.0),
    ]
    series_map_train = [
        ("lightgbm_mssplit_train_accuracy", "MSPLIT (LGB bins)", "#ff1493", "^", 2.8),
        ("xgboost_mean_train_accuracy", "XGBoost", "#2ca02c", "D", 2.0),
        ("shapecart_mean_train_accuracy", "ShapeCART", "#ff8c00", "s", 2.0),
    ]
    series_map = series_map_test if metric == "test" else series_map_train

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.2), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    legend_handles: dict[str, object] = {}
    for ax, ds in zip(axes, datasets):
        s = df[df["dataset"] == ds].sort_values("depth_budget")
        depths = s["depth_budget"].to_numpy(dtype=int)
        for col, label, color, marker, lw in series_map:
            if col not in s.columns:
                continue
            y = s[col].to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                continue
            (line,) = ax.plot(depths, y * 100.0, marker=marker, linewidth=lw, color=color, label=label)
            if label not in legend_handles:
                legend_handles[label] = line

        ax.set_title(ds)
        ax.set_xlabel("Depth budget")
        ax.set_xticks(depth_budgets)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"{metric.capitalize()} accuracy (%)")
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[label] for label in labels]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=10,
        )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_detailed_report(
    summary_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    out_path: Path,
    label: str,
    datasets: list[str],
    depth_budgets: list[int],
) -> None:
    lines = [f"{label} detailed report", ""]
    lines.append("Status counts:")
    if "status" in seed_df.columns:
        for status, count in seed_df["status"].value_counts().to_dict().items():
            lines.append(f"- {status}: {count}")
    lines.append("")

    lines.append("Per-depth summary:")
    grid = pd.MultiIndex.from_product([datasets, depth_budgets], names=["dataset", "depth_budget"]).to_frame(index=False)
    merged = grid.merge(summary_df, on=["dataset", "depth_budget"], how="left")

    for ds in datasets:
        lines.append(f"Dataset: {ds}")
        s = merged[merged["dataset"] == ds].sort_values("depth_budget")
        for _, row in s.iterrows():
            d = int(row["depth_budget"])
            n = int(row["n_success"]) if not pd.isna(row.get("n_success")) else 0
            acc = row.get("mean_accuracy", np.nan)
            bal = row.get("mean_balanced_accuracy", np.nan)
            t = row.get("mean_fit_time_sec", np.nan)
            if n == 0 or pd.isna(acc):
                lines.append(f"- depth={d}: n_success=0, acc=NA, bal_acc=NA, fit_time=NA")
            else:
                lines.append(f"- depth={d}: n_success={n}, acc={acc:.6f}, bal_acc={bal:.6f}, fit_time={t:.2f}s")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_msplit_runtime_logs(
    lgb_seed_df: pd.DataFrame,
    datasets: list[str],
    depth_budgets: list[int],
    out_dir: Path,
) -> tuple[Path, Path]:
    seed_runtime_csv = out_dir / "lightgbm_mssplit_seed_runtime.csv"
    depth_runtime_csv = out_dir / "lightgbm_mssplit_runtime_vs_depth.csv"

    seed_cols = [
        "dataset",
        "depth_budget",
        "seed",
        "msplit_variant",
        "status",
        "fit_time_sec",
        "accuracy",
        "train_accuracy",
    ]
    available_seed_cols = [c for c in seed_cols if c in lgb_seed_df.columns]
    lgb_seed_df.loc[:, available_seed_cols].to_csv(seed_runtime_csv, index=False)

    ok = lgb_seed_df[lgb_seed_df.get("status", "ok") == "ok"].copy()
    runtime_stats = (
        ok.groupby(["dataset", "depth_budget"], as_index=False)
        .agg(
            msplit_n_success=("fit_time_sec", "count"),
            msplit_fit_time_mean_sec=("fit_time_sec", "mean"),
            msplit_fit_time_median_sec=("fit_time_sec", "median"),
            msplit_fit_time_std_sec=("fit_time_sec", "std"),
        )
        .sort_values(["dataset", "depth_budget"])
        .reset_index(drop=True)
    )
    runtime_stats["msplit_fit_time_std_sec"] = runtime_stats["msplit_fit_time_std_sec"].fillna(0.0)

    grid = pd.MultiIndex.from_product([datasets, depth_budgets], names=["dataset", "depth_budget"]).to_frame(index=False)
    runtime_grid = grid.merge(runtime_stats, on=["dataset", "depth_budget"], how="left")
    runtime_grid.to_csv(depth_runtime_csv, index=False)
    return seed_runtime_csv, depth_runtime_csv


def _select_best_rows(
    seed_df: pd.DataFrame,
    datasets: list[str],
    accuracy_col: str,
    min_depth: int,
    default_depth: int,
    default_seed: int,
    require_ok: bool,
) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for ds in datasets:
        subset = seed_df[seed_df["dataset"] == ds].copy()
        if require_ok and "status" in subset.columns:
            subset = subset[subset["status"] == "ok"]
        subset = subset[pd.notna(subset[accuracy_col])]

        if subset.empty:
            selected[ds] = {
                "depth_budget": int(default_depth),
                "seed": int(default_seed),
                "accuracy": float("nan"),
                "selection_note": "fallback_no_rows",
                "tree_artifact_path": "",
            }
            continue

        filtered = subset[subset["depth_budget"] >= int(min_depth)]
        if filtered.empty:
            filtered = subset
            note = "fallback_no_rows_at_min_depth"
        else:
            note = "best_at_or_above_min_depth"

        ranked = filtered.sort_values(
            [accuracy_col, "depth_budget", "seed"],
            ascending=[False, False, True],
        )
        best = ranked.iloc[0]
        selected[ds] = {
            "depth_budget": int(best["depth_budget"]),
            "seed": int(best["seed"]),
            "accuracy": float(best[accuracy_col]),
            "selection_note": note,
            "tree_artifact_path": str(best.get("tree_artifact_path", "") or ""),
        }
    return selected


def _select_fixed_rows(
    seed_df: pd.DataFrame,
    datasets: list[str],
    accuracy_col: str,
    fixed_depth: int,
    fixed_seed: int,
    require_ok: bool,
) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for ds in datasets:
        subset = seed_df[
            (seed_df["dataset"] == ds)
            & (seed_df["depth_budget"] == int(fixed_depth))
            & (seed_df["seed"] == int(fixed_seed))
        ].copy()
        if require_ok and "status" in subset.columns:
            subset = subset[subset["status"] == "ok"]
        subset = subset[pd.notna(subset[accuracy_col])]

        if subset.empty:
            selected[ds] = {
                "depth_budget": int(fixed_depth),
                "seed": int(fixed_seed),
                "accuracy": float("nan"),
                "selection_note": "fixed_fallback_missing_row",
                "tree_artifact_path": "",
            }
            continue

        best = subset.sort_values(accuracy_col, ascending=False).iloc[0]
        selected[ds] = {
            "depth_budget": int(best["depth_budget"]),
            "seed": int(best["seed"]),
            "accuracy": float(best[accuracy_col]),
            "selection_note": "fixed_from_seed_rows",
            "tree_artifact_path": str(best.get("tree_artifact_path", "") or ""),
        }
    return selected


def _build_viz_config_map(
    args: argparse.Namespace,
    lgb_seed_df: pd.DataFrame,
    xgb_seed_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    if args.viz_select == "fixed":
        return {
            "xgboost": _select_fixed_rows(
                xgb_seed_df,
                args.datasets,
                accuracy_col="xgboost_accuracy",
                fixed_depth=args.viz_depth,
                fixed_seed=args.viz_seed,
                require_ok=False,
            ),
            "lightgbm": _select_fixed_rows(
                lgb_seed_df,
                args.datasets,
                accuracy_col="accuracy",
                fixed_depth=args.viz_depth,
                fixed_seed=args.viz_seed,
                require_ok=True,
            ),
        }

    return {
        "xgboost": _select_best_rows(
            xgb_seed_df,
            args.datasets,
            accuracy_col="xgboost_accuracy",
            min_depth=args.viz_min_depth,
            default_depth=args.viz_depth,
            default_seed=args.viz_seed,
            require_ok=False,
        ),
        "lightgbm": _select_best_rows(
            lgb_seed_df,
            args.datasets,
            accuracy_col="accuracy",
            min_depth=args.viz_min_depth,
            default_depth=args.viz_depth,
            default_seed=args.viz_seed,
            require_ok=True,
        ),
    }


def _tree_stats_from_node(node: dict[str, object]) -> tuple[int, int, int]:
    node_type = str(node.get("node_type", ""))
    if node_type == "leaf":
        return 0, 1, 0

    children = node.get("children")
    if not isinstance(children, list) or not children:
        return 0, 1, 0

    child_depths: list[int] = []
    leaf_count = 0
    internal_count = 1
    for entry in children:
        if not isinstance(entry, dict):
            continue
        child = entry.get("child")
        if not isinstance(child, dict):
            continue
        d, l, i = _tree_stats_from_node(child)
        child_depths.append(d)
        leaf_count += l
        internal_count += i

    if not child_depths:
        return 0, 1, 0
    return 1 + max(child_depths), leaf_count, internal_count


def _read_artifact_tree_stats(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "realized_depth": None,
        "realized_leaves": None,
        "realized_internal_nodes": None,
    }
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    tree_artifact = payload.get("tree_artifact")
    if not isinstance(tree_artifact, dict):
        return out
    if isinstance(tree_artifact.get("tree"), dict):
        root = tree_artifact["tree"]
    else:
        root = tree_artifact
    if not isinstance(root, dict):
        return out

    depth, leaves, internal = _tree_stats_from_node(root)
    out["realized_depth"] = int(depth)
    out["realized_leaves"] = int(leaves)
    out["realized_internal_nodes"] = int(internal)
    return out


def _generate_trees(
    args: argparse.Namespace,
    out_dir: Path,
    viz_config_map: dict[str, dict[str, dict[str, object]]],
    all_artifact_dirs: dict[str, Path],
) -> tuple[Path, Path, Path]:
    xgb_dir = out_dir / "XGBoost_trees"
    lgb_dir = out_dir / "lightgbm_mssplit_trees"
    artifacts_dir = out_dir / "tree_artifacts"
    xgb_artifacts_dir = artifacts_dir / "xgboost"
    lgb_artifacts_dir = artifacts_dir / "lightgbm_mssplit"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    lgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_artifacts_dir.mkdir(parents=True, exist_ok=True)
    lgb_artifacts_dir.mkdir(parents=True, exist_ok=True)

    base = [
        "--test-size",
        str(args.test_size),
        "--time-limit",
        str(args.time_limit),
        "--lookahead-cap",
        str(args.lookahead_cap),
        "--max-bins",
        str(args.max_bins),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--min-child-size",
        str(args.min_child_size),
        "--min-split-size",
        str(args.min_split_size),
        "--max-branching",
        str(args.max_branching),
        "--reg",
        str(args.reg),
        "--msplit-variant",
        str(args.msplit_variant),
        "--xgb-n-estimators",
        str(args.xgb_n_estimators),
        "--xgb-learning-rate",
        str(args.xgb_learning_rate),
        "--xgb-num-threads",
        str(args.xgb_num_threads),
        "--lgb-device-type",
        str(args.lgb_device_type),
        "--lgb-ensemble-runs",
        str(args.lgb_ensemble_runs),
        "--lgb-ensemble-feature-fraction",
        str(args.lgb_ensemble_feature_fraction),
        "--lgb-ensemble-bagging-fraction",
        str(args.lgb_ensemble_bagging_fraction),
        "--lgb-ensemble-bagging-freq",
        str(args.lgb_ensemble_bagging_freq),
        "--lgb-threshold-dedup-eps",
        str(args.lgb_threshold_dedup_eps),
        "--lgb-gpu-platform-id",
        str(args.lgb_gpu_platform_id),
        "--lgb-gpu-device-id",
        str(args.lgb_gpu_device_id),
        ("--lgb-gpu-fallback" if args.lgb_gpu_fallback else "--no-lgb-gpu-fallback"),
    ]

    selected_configs: list[dict[str, object]] = []
    for ds in args.datasets:
        targets = [
            ("xgboost", xgb_dir / f"{ds}.png", xgb_artifacts_dir / f"{ds}.json"),
            ("lightgbm", lgb_dir / f"{ds}.png", lgb_artifacts_dir / f"{ds}.json"),
        ]
        for pipeline, out_path, artifact_out in targets:
            cfg = viz_config_map[pipeline][ds]
            depth_budget = int(cfg["depth_budget"])
            seed = int(cfg["seed"])
            artifact_in = str(cfg.get("tree_artifact_path", "") or "").strip()
            if artifact_in and Path(artifact_in).exists():
                cmd = [
                    sys.executable,
                    "visualize_multisplit_tree.py",
                    "--dataset",
                    ds,
                    "--pipeline",
                    pipeline,
                    "--artifact-in",
                    artifact_in,
                    "--out",
                    str(out_path),
                    "--artifact-out",
                    str(artifact_out),
                ]
                source_mode = "artifact"
            else:
                cmd = [
                    sys.executable,
                    "visualize_multisplit_tree.py",
                    "--dataset",
                    ds,
                    "--pipeline",
                    pipeline,
                    "--depth-budget",
                    str(depth_budget),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out_path),
                    "--artifact-out",
                    str(artifact_out),
                    *base,
                ]
                source_mode = "retrain_fallback"
            print(
                (
                    f"[compare] visualizing {pipeline} dataset={ds} "
                    f"depth={depth_budget} seed={seed} source={source_mode}"
                ),
                flush=True,
            )
            print(f"[compare] visualizing: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
            realized = _read_artifact_tree_stats(artifact_out)
            selected_configs.append(
                {
                    "pipeline": pipeline,
                    "dataset": ds,
                    "depth_budget": depth_budget,
                    "seed": seed,
                    "accuracy": cfg.get("accuracy"),
                    "selection_note": cfg.get("selection_note", ""),
                    "image_path": str(out_path),
                    "artifact_source_path": artifact_in,
                    "artifact_path": str(artifact_out),
                    "artifact_source_mode": source_mode,
                    "realized_depth": realized["realized_depth"],
                    "realized_leaves": realized["realized_leaves"],
                    "realized_internal_nodes": realized["realized_internal_nodes"],
                }
            )

    if selected_configs:
        selected_df = pd.DataFrame(selected_configs).sort_values(["pipeline", "dataset"]).reset_index(drop=True)
    else:
        selected_df = pd.DataFrame(
            columns=[
                "pipeline",
                "dataset",
                "depth_budget",
                "seed",
                "accuracy",
                "selection_note",
                "image_path",
                "artifact_source_path",
                "artifact_path",
                "artifact_source_mode",
                "realized_depth",
                "realized_leaves",
                "realized_internal_nodes",
            ]
        )
    (out_dir / "visualized_tree_selection.csv").write_text(
        selected_df.to_csv(index=False),
        encoding="utf-8",
    )
    selected_configs = selected_df.to_dict(orient="records")

    manifest = {
        "schema_version": 1,
        "datasets": args.datasets,
        "tree_visualization": {
            "selection_mode": args.viz_select,
            "viz_min_depth": int(args.viz_min_depth),
            "fixed_depth_budget": int(args.viz_depth),
            "fixed_seed": int(args.viz_seed),
            "test_size": float(args.test_size),
            "selected_configs": selected_configs,
        },
        "xgboost": {
            "n_estimators": int(args.xgb_n_estimators),
            "learning_rate": float(args.xgb_learning_rate),
            "num_threads": int(args.xgb_num_threads),
            "images_dir": str(xgb_dir),
            "artifacts_dir": str(xgb_artifacts_dir),
            "all_artifacts_dir": str(all_artifact_dirs["xgboost"]),
        },
        "lightgbm_mssplit": {
            "time_limit": float(args.time_limit),
            "lookahead_cap": int(args.lookahead_cap),
            "max_bins": int(args.max_bins),
            "min_samples_leaf": int(args.min_samples_leaf),
            "min_child_size": int(args.min_child_size),
            "max_branching": int(args.max_branching),
            "reg": float(args.reg),
            "msplit_variant": str(args.msplit_variant),
            "lgb_device_type": str(args.lgb_device_type),
            "lgb_gpu_platform_id": int(args.lgb_gpu_platform_id),
            "lgb_gpu_device_id": int(args.lgb_gpu_device_id),
            "lgb_gpu_fallback": bool(args.lgb_gpu_fallback),
            "lgb_max_gpu_jobs": int(args.lgb_max_gpu_jobs),
            "lgb_gpu_lock_dir": str(args.lgb_gpu_lock_dir),
            "images_dir": str(lgb_dir),
            "artifacts_dir": str(lgb_artifacts_dir),
            "all_artifacts_dir": str(all_artifact_dirs["lightgbm"]),
        },
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"RUN_DIR={str(out_dir)!r}",
        "PY=${PY:-.venv/bin/python}",
        "",
    ]
    for ds in args.datasets:
        for pipeline, out_subdir, artifact_subdir in [
            ("xgboost", "XGBoost_trees", "xgboost"),
            ("lightgbm", "lightgbm_mssplit_trees", "lightgbm_mssplit"),
        ]:
            cfg = viz_config_map[pipeline][ds]
            lines.extend(
                [
                    "\"$PY\" visualize_multisplit_tree.py \\",
                    f"  --dataset {ds} \\",
                    f"  --pipeline {pipeline} \\",
                    f"  --artifact-in \"$RUN_DIR/tree_artifacts/{'xgboost_all' if pipeline == 'xgboost' else 'lightgbm_mssplit_all'}/{ds}/depth_{int(cfg['depth_budget'])}/seed_{int(cfg['seed'])}.json\" \\",
                    f"  --out \"$RUN_DIR/{out_subdir}/{ds}.png\" \\",
                    f"  --artifact-out \"$RUN_DIR/tree_artifacts/{artifact_subdir}/{ds}.json\"",
                    "",
                ]
            )
    reproduce_script = artifacts_dir / "reproduce_trees.sh"
    reproduce_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reproduce_script.chmod(0o755)
    return xgb_dir, lgb_dir, artifacts_dir


def main() -> None:
    args = _parse_args()
    t_total_start = time.perf_counter()
    stage_sec: dict[str, float] = {}
    run_name = args.run_name or datetime.now().strftime("comparison_%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tree_artifacts_all = {
        "xgboost": out_dir / "tree_artifacts" / "xgboost_all",
        "lightgbm": out_dir / "tree_artifacts" / "lightgbm_mssplit_all",
    }
    for path in tree_artifacts_all.values():
        path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    lgb_summary, lgb_seed, lgb_run_dir = _run_split_pipeline(
        "run_multisplit_experiments_lightgbm.py",
        f"{run_name}_lightgbm",
        args,
        tmp_root=tmp_dir / "runs_lightgbm",
        stable_dir=tmp_dir / "stable_lightgbm",
        tree_artifacts_dir=tree_artifacts_all["lightgbm"],
        extra_args=[
            ("--approx-mode" if args.approx_mode else "--no-approx-mode"),
            "--lgb-device-type",
            str(args.lgb_device_type),
            "--lgb-ensemble-runs",
            str(args.lgb_ensemble_runs),
            "--lgb-ensemble-feature-fraction",
            str(args.lgb_ensemble_feature_fraction),
            "--lgb-ensemble-bagging-fraction",
            str(args.lgb_ensemble_bagging_fraction),
            "--lgb-ensemble-bagging-freq",
            str(args.lgb_ensemble_bagging_freq),
            "--lgb-threshold-dedup-eps",
            str(args.lgb_threshold_dedup_eps),
            "--lgb-gpu-platform-id",
            str(args.lgb_gpu_platform_id),
            "--lgb-gpu-device-id",
            str(args.lgb_gpu_device_id),
            ("--lgb-gpu-fallback" if args.lgb_gpu_fallback else "--no-lgb-gpu-fallback"),
            "--lgb-max-gpu-jobs",
            str(args.lgb_max_gpu_jobs),
            "--lgb-gpu-lock-dir",
            str(args.lgb_gpu_lock_dir),
        ],
    )
    lgb_optuna_dir = _persist_optuna_outputs(lgb_run_dir, out_dir, "lightgbm")
    stage_sec["lightgbm_pipeline_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    xgb_seed, xgb_summary = _run_xgboost_reference(args, tree_artifacts_dir=tree_artifacts_all["xgboost"])
    stage_sec["xgboost_reference_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    shapecart_seed, shapecart_summary = _run_shapecart_reference(args)
    stage_sec["shapecart_reference_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = pd.MultiIndex.from_product([args.datasets, args.depth_budgets], names=["dataset", "depth_budget"]).to_frame(index=False)
    combined = (
        grid.merge(xgb_summary, on=["dataset", "depth_budget"], how="left")
        .merge(
            shapecart_summary,
            on=["dataset", "depth_budget"],
            how="left",
        )
        .merge(
            lgb_summary[
                ["dataset", "depth_budget", "mean_accuracy", "mean_train_accuracy", "mean_fit_time_sec"]
            ].rename(
                columns={
                    "mean_accuracy": "lightgbm_mssplit_accuracy",
                    "mean_train_accuracy": "lightgbm_mssplit_train_accuracy",
                    "mean_fit_time_sec": "lightgbm_mssplit_mean_fit_time_sec",
                }
            ),
            on=["dataset", "depth_budget"],
            how="left",
        )
    )
    combined = combined.sort_values(["dataset", "depth_budget"]).reset_index(drop=True)

    exact_summary_csv = out_dir / "accuracy_vs_depth_all.csv"
    lgb_report_txt = out_dir / "lightgbm_mssplit_detailed_report.txt"
    shapecart_report_txt = out_dir / "shapecart_detailed_report.txt"
    msplit_seed_runtime_csv = out_dir / "lightgbm_mssplit_seed_runtime.csv"
    msplit_depth_runtime_csv = out_dir / "lightgbm_mssplit_runtime_vs_depth.csv"
    plot_test_png = out_dir / "accuracy_vs_depth_all.png"
    plot_train_png = out_dir / "accuracy_vs_depth_train_all.png"

    combined.to_csv(exact_summary_csv, index=False)
    _write_detailed_report(lgb_summary, lgb_seed, lgb_report_txt, "LightGBM + MSPLIT", args.datasets, args.depth_budgets)
    _write_detailed_report(
        shapecart_summary.rename(
            columns={
                "shapecart_n_success": "n_success",
                "shapecart_mean_accuracy": "mean_accuracy",
                "shapecart_mean_train_accuracy": "mean_train_accuracy",
                "shapecart_std_accuracy": "std_accuracy",
            }
        ),
        shapecart_seed.rename(columns={"shapecart_accuracy": "accuracy", "shapecart_train_accuracy": "train_accuracy"}),
        shapecart_report_txt,
        "ShapeCART",
        args.datasets,
        args.depth_budgets,
    )
    _plot_accuracy_vs_depth(combined, args.datasets, args.depth_budgets, plot_test_png, metric="test")
    _plot_accuracy_vs_depth(combined, args.datasets, args.depth_budgets, plot_train_png, metric="train")
    msplit_seed_runtime_csv, msplit_depth_runtime_csv = _write_msplit_runtime_logs(
        lgb_seed_df=lgb_seed,
        datasets=args.datasets,
        depth_budgets=args.depth_budgets,
        out_dir=out_dir,
    )
    stage_sec["aggregation_reporting_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    viz_config_map = _build_viz_config_map(
        args,
        lgb_seed_df=lgb_seed,
        xgb_seed_df=xgb_seed,
    )
    xgb_dir, lgb_dir, artifacts_dir = _generate_trees(args, out_dir, viz_config_map, tree_artifacts_all)
    stage_sec["tree_visualization_sec"] = time.perf_counter() - t0

    shutil.rmtree(tmp_dir, ignore_errors=True)
    stage_sec["total_sec"] = time.perf_counter() - t_total_start

    timing_profile_path = out_dir / "timing_profile.json"
    timing_payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_dir": str(out_dir),
        "command": " ".join(sys.argv),
        "config": vars(args),
        "stage_seconds": {k: float(v) for k, v in stage_sec.items()},
    }
    timing_profile_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("[compare] saved:")
    print(f"- {exact_summary_csv}")
    print(f"- {lgb_report_txt}")
    print(f"- {shapecart_report_txt}")
    print(f"- {plot_test_png}")
    print(f"- {plot_train_png}")
    print(f"- {msplit_seed_runtime_csv}")
    print(f"- {msplit_depth_runtime_csv}")
    print(f"- {xgb_dir}")
    print(f"- {lgb_dir}")
    if lgb_optuna_dir is not None:
        print(f"- {lgb_optuna_dir}")
    print(f"- {artifacts_dir}")
    print(f"- {timing_profile_path}")


if __name__ == "__main__":
    main()
