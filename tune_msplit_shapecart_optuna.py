#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import optuna
except Exception as exc:  # pragma: no cover - CLI guard
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_teacher_guided_atomcolor_cached import (
    _protocol_split_indices,
    _slice_rows,
    derive_min_child_size,
    derive_min_split_size,
    load_local_libgosdt,
    predict_tree,
)
from experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner

SHAPECART_ROOT = REPO_ROOT / "Empowering-DTs-via-Shape-Functions"
if str(SHAPECART_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPECART_ROOT))

from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore


N_TRIALS = 20
DEFAULT_SEED = 0
TEST_SIZE = 0.2
VAL_SIZE = 0.125
MAX_BINS = 1024
MIN_SAMPLES_LEAF = 8
LEAF_FRAC = 0.001
LOOKAHEAD_DEPTH_CAP = 3
BRANCHING_FACTOR = 3
TIME_LIMIT_SECONDS = 28800.0
LGB_NUM_THREADS = 3
SHAPECART_INNER_MAX_DEPTH = 6
SHAPECART_INNER_MAX_LEAF_NODES = 32
SHAPECART_MAX_ITER = 20
SHAPECART_PAIRWISE_CANDIDATES = 0.0

REG_RANGE = (1e-5, 1e-3)
MSPLIT_TOP_K_CHOICES = (1, 2, 4, 8, 16)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MSPLIT and ShapeCART for one dataset at one fixed depth."
    )
    parser.add_argument("--dataset", default="electricity", choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if optuna is None:
        raise RuntimeError("Optuna is not installed in this environment.") from _OPTUNA_IMPORT_ERROR
    if int(args.depth) < 1:
        raise ValueError("--depth must be at least 1")
    return args


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred, dtype=np.int32) == np.asarray(y_true, dtype=np.int32)))


def _run_study(objective, *, seed: int):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _safe_objective(trial) -> float:
        try:
            return float(objective(trial))
        except Exception:
            return 0.0

    study.optimize(_safe_objective, n_trials=N_TRIALS)
    return study


@lru_cache(maxsize=None)
def _prepare_dataset(dataset_name: str) -> dict[str, object]:
    X, y = DATASET_LOADERS[dataset_name]()
    y_bin = encode_binary_target(y, dataset_name)
    split_idx = _protocol_split_indices(
        y_bin=np.asarray(y_bin, dtype=np.int32),
        seed=DEFAULT_SEED,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
    )
    idx_fit = split_idx["idx_fit"]
    idx_val = split_idx["idx_val"]
    idx_test = split_idx["idx_test"]

    X_fit = _slice_rows(X, idx_fit)
    X_val = _slice_rows(X, idx_val)
    X_test = _slice_rows(X, idx_test)
    y_fit = np.asarray(y_bin[idx_fit], dtype=np.int32)
    y_val = np.asarray(y_bin[idx_val], dtype=np.int32)
    y_test = np.asarray(y_bin[idx_test], dtype=np.int32)

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.asarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.asarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)

    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=MAX_BINS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=DEFAULT_SEED,
        n_estimators=10000,
        num_leaves=255,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        min_data_in_bin=1,
        min_data_in_leaf=2,
        lambda_l2=0.0,
        early_stopping_rounds=100,
        num_threads=LGB_NUM_THREADS,
        device_type="cpu",
        collect_teacher_logit=True,
    )

    resolved_min_child_size = derive_min_child_size(
        leaf_frac=LEAF_FRAC,
        n_fit=int(idx_fit.shape[0]),
    )
    resolved_min_split_size = derive_min_split_size(
        leaf_frac=LEAF_FRAC,
        n_fit=int(idx_fit.shape[0]),
    )

    return {
        "X_fit_proc": X_fit_proc,
        "X_val_proc": X_val_proc,
        "X_test_proc": X_test_proc,
        "Z_fit": np.asarray(binner.transform(X_fit_proc), dtype=np.int32),
        "Z_val": np.asarray(binner.transform(X_val_proc), dtype=np.int32),
        "Z_test": np.asarray(binner.transform(X_test_proc), dtype=np.int32),
        "y_fit": y_fit,
        "y_val": y_val,
        "y_test": y_test,
        "teacher_logit": np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        "teacher_boundary_gain": np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        "teacher_boundary_cover": np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(
            getattr(binner, "boundary_value_jump_per_feature"),
            dtype=np.float64,
        ),
        "min_child_size": int(resolved_min_child_size),
        "min_split_size": int(resolved_min_split_size),
        "n_fit": int(idx_fit.shape[0]),
        "n_val": int(idx_val.shape[0]),
        "n_test": int(idx_test.shape[0]),
    }


def _fit_msplit_candidate(
    payload: dict[str, object],
    *,
    depth: int,
    regularization: float,
    exactify_top_k: int,
) -> dict[str, object]:
    libgosdt = load_local_libgosdt()
    z_fit = np.asarray(payload["Z_fit"], dtype=np.int32)
    z_val = np.asarray(payload["Z_val"], dtype=np.int32)
    z_test = np.asarray(payload["Z_test"], dtype=np.int32)
    y_fit = np.asarray(payload["y_fit"], dtype=np.int32)
    y_val = np.asarray(payload["y_val"], dtype=np.int32)
    y_test = np.asarray(payload["y_test"], dtype=np.int32)
    sample_weight = np.full(z_fit.shape[0], 1.0 / float(max(1, z_fit.shape[0])), dtype=np.float64)

    started = time.perf_counter()
    cpp_result = libgosdt.msplit_fit(
        z_fit,
        y_fit,
        sample_weight,
        np.asarray(payload["teacher_logit"], dtype=np.float64),
        np.asarray(payload["teacher_boundary_gain"], dtype=np.float64),
        np.asarray(payload["teacher_boundary_cover"], dtype=np.float64),
        np.asarray(payload["teacher_boundary_value_jump"], dtype=np.float64),
        int(depth),
        min(int(depth), LOOKAHEAD_DEPTH_CAP),
        float(regularization),
        int(payload["min_split_size"]),
        int(payload["min_child_size"]),
        TIME_LIMIT_SECONDS,
        BRANCHING_FACTOR,
        int(exactify_top_k),
    )
    fit_time_sec = time.perf_counter() - started
    tree = json.loads(str(cpp_result["tree"]))
    pred_val = predict_tree(tree, z_val)
    pred_test = predict_tree(tree, z_test)
    return {
        "val_accuracy": _accuracy(y_val, pred_val),
        "test_accuracy": _accuracy(y_test, pred_test),
        "fit_time_sec": float(fit_time_sec),
    }


def _fit_shapecart_candidate(
    payload: dict[str, object],
    *,
    depth: int,
    regularization: float,
) -> dict[str, object]:
    X_fit = np.asarray(payload["X_fit_proc"], dtype=np.float32)
    X_val = np.asarray(payload["X_val_proc"], dtype=np.float32)
    X_test = np.asarray(payload["X_test_proc"], dtype=np.float32)
    y_fit = np.asarray(payload["y_fit"], dtype=np.int32)
    y_val = np.asarray(payload["y_val"], dtype=np.int32)
    y_test = np.asarray(payload["y_test"], dtype=np.int32)

    model = ShapeCARTClassifier(
        max_depth=int(depth),
        min_samples_leaf=int(payload["min_child_size"]),
        min_samples_split=max(2, int(payload["min_split_size"])),
        inner_min_samples_leaf=int(payload["min_child_size"]),
        inner_min_samples_split=max(2, int(payload["min_split_size"])),
        inner_max_depth=SHAPECART_INNER_MAX_DEPTH,
        inner_max_leaf_nodes=SHAPECART_INNER_MAX_LEAF_NODES,
        max_iter=SHAPECART_MAX_ITER,
        k=BRANCHING_FACTOR,
        branching_penalty=float(regularization),
        random_state=DEFAULT_SEED,
        verbose=False,
        pairwise_candidates=SHAPECART_PAIRWISE_CANDIDATES,
        smart_init=True,
        random_pairs=False,
        use_dpdt=False,
        use_tao=False,
    )
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_time_sec = time.perf_counter() - started

    pred_val = np.asarray(model.predict(X_val), dtype=np.int32)
    pred_test = np.asarray(model.predict(X_test), dtype=np.int32)
    return {
        "val_accuracy": _accuracy(y_val, pred_val),
        "test_accuracy": _accuracy(y_test, pred_test),
        "fit_time_sec": float(fit_time_sec),
    }


def _tune_msplit_payload(payload: dict[str, object], depth: int) -> dict[str, object]:
    def _objective(trial) -> float:
        regularization = trial.suggest_float("regularization", REG_RANGE[0], REG_RANGE[1], log=True)
        exactify_top_k = trial.suggest_categorical("exactify_top_k", MSPLIT_TOP_K_CHOICES)
        result = _fit_msplit_candidate(
            payload,
            depth=int(depth),
            regularization=float(regularization),
            exactify_top_k=int(exactify_top_k),
        )
        return float(result["val_accuracy"])

    study = _run_study(_objective, seed=DEFAULT_SEED)
    best_params = dict(study.best_params)
    final_result = _fit_msplit_candidate(
        payload,
        depth=int(depth),
        regularization=float(best_params["regularization"]),
        exactify_top_k=int(best_params["exactify_top_k"]),
    )
    return {
        "depth": int(depth),
        "best_params": {
            "regularization": float(best_params["regularization"]),
            "exactify_top_k": int(best_params["exactify_top_k"]),
        },
        "val_accuracy": float(final_result["val_accuracy"]),
        "test_accuracy": float(final_result["test_accuracy"]),
        "fit_time_sec": float(final_result["fit_time_sec"]),
        "optuna_trials": int(len(study.trials)),
    }


def tune_msplit(dataset: str, depth: int) -> dict[str, object]:
    return _tune_msplit_payload(_prepare_dataset(dataset), depth)


def _tune_shapecart_payload(payload: dict[str, object], depth: int) -> dict[str, object]:
    def _objective(trial) -> float:
        regularization = trial.suggest_float("regularization", REG_RANGE[0], REG_RANGE[1], log=True)
        result = _fit_shapecart_candidate(
            payload,
            depth=int(depth),
            regularization=float(regularization),
        )
        return float(result["val_accuracy"])

    study = _run_study(_objective, seed=DEFAULT_SEED + 1)
    best_params = dict(study.best_params)
    final_result = _fit_shapecart_candidate(
        payload,
        depth=int(depth),
        regularization=float(best_params["regularization"]),
    )
    return {
        "depth": int(depth),
        "best_params": {
            "regularization": float(best_params["regularization"]),
        },
        "val_accuracy": float(final_result["val_accuracy"]),
        "test_accuracy": float(final_result["test_accuracy"]),
        "fit_time_sec": float(final_result["fit_time_sec"]),
        "optuna_trials": int(len(study.trials)),
    }


def tune_shapecart(dataset: str, depth: int) -> dict[str, object]:
    return _tune_shapecart_payload(_prepare_dataset(dataset), depth)


def main() -> int:
    args = _parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    payload = _prepare_dataset(args.dataset)
    result = {
        "dataset": args.dataset,
        "depth": int(args.depth),
        "split_sizes": {
            "fit": int(payload["n_fit"]),
            "val": int(payload["n_val"]),
            "test": int(payload["n_test"]),
        },
        "branching_factor": BRANCHING_FACTOR,
        "msplit": _tune_msplit_payload(payload, int(args.depth)),
        "shapecart": _tune_shapecart_payload(payload, int(args.depth)),
    }
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
