"""LightGBM-guided per-feature binning for multiway decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.utils.validation import check_X_y, check_array


@dataclass
class LightGBMBinner:
    """Feature-wise binner built from LightGBM split thresholds."""

    bin_edges_per_feature: List[np.ndarray]
    fill_values_per_feature: np.ndarray
    max_bins: int
    min_samples_leaf: int
    random_state: Optional[int]
    device_type: str = "cpu"

    @property
    def n_bins_per_feature(self) -> np.ndarray:
        return np.array([int(edges.size + 1) if edges.size else 1 for edges in self.bin_edges_per_feature], dtype=np.int32)

    def transform(self, X) -> np.ndarray:
        X_arr = check_array(X, ensure_2d=True, dtype=float, ensure_all_finite="allow-nan")
        if X_arr.shape[1] != len(self.bin_edges_per_feature):
            raise ValueError(
                f"X has {X_arr.shape[1]} features but binner was fit with {len(self.bin_edges_per_feature)} features"
            )

        X_arr = X_arr.copy()
        nan_mask = np.isnan(X_arr)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X_arr[rows, cols] = self.fill_values_per_feature[cols]

        Z = np.zeros(X_arr.shape, dtype=np.int32)
        for j, edges in enumerate(self.bin_edges_per_feature):
            if edges.size == 0:
                Z[:, j] = 0
            else:
                Z[:, j] = np.digitize(X_arr[:, j], edges, right=False).astype(np.int32)
        return Z


def _binary_edge(unique_values: np.ndarray) -> np.ndarray:
    low = float(np.min(unique_values))
    high = float(np.max(unique_values))
    if low == 0.0 and high == 1.0:
        return np.array([0.5], dtype=float)
    return np.array([(low + high) / 2.0], dtype=float)


def _quantile_edges(values: np.ndarray, max_edges: int) -> np.ndarray:
    if max_edges <= 0:
        return np.array([], dtype=float)

    unique_values = np.unique(values)
    if unique_values.size <= 1:
        return np.array([], dtype=float)
    if unique_values.size <= 2:
        return _binary_edge(unique_values)

    quantiles = np.linspace(0.0, 1.0, max_edges + 2, dtype=float)[1:-1]
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges.astype(float))

    lo = float(np.min(values))
    hi = float(np.max(values))
    edges = edges[(edges > lo) & (edges < hi)]
    if edges.size > max_edges:
        edges = edges[:max_edges]
    return np.sort(edges)


def _collect_threshold_scores(tree_node: Dict, feature_scores: List[Dict[float, float]]) -> None:
    if "split_feature" not in tree_node:
        return

    feature_idx = int(tree_node["split_feature"])
    threshold = float(tree_node["threshold"])
    split_gain = float(tree_node.get("split_gain", 1.0))

    if np.isfinite(threshold):
        score = max(split_gain, 1e-12)
        feature_scores[feature_idx][threshold] = feature_scores[feature_idx].get(threshold, 0.0) + score

    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    if isinstance(left, dict):
        _collect_threshold_scores(left, feature_scores)
    if isinstance(right, dict):
        _collect_threshold_scores(right, feature_scores)


def _merge_threshold_score_maps(
    dst: List[Dict[float, float]],
    src: List[Dict[float, float]],
) -> None:
    for j, src_map in enumerate(src):
        dst_map = dst[j]
        for threshold, score in src_map.items():
            dst_map[threshold] = dst_map.get(threshold, 0.0) + float(score)


def _encode_binary_target(y_arr: np.ndarray) -> np.ndarray:
    y = np.asarray(y_arr)
    unique = np.unique(y)
    if unique.size != 2:
        raise ValueError(
            f"fit_lightgbm_binner expects binary labels; received {unique.size} classes: {unique.tolist()}"
        )
    ordered = sorted(unique.tolist(), key=lambda v: str(v))
    return np.where(y == ordered[0], 0, 1).astype(np.int32)


def fit_lightgbm_binner(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    max_bins: int = 6,
    min_samples_leaf: int = 10,
    random_state: Optional[int] = None,
    n_estimators: int = 10000,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    bagging_freq: int = 0,
    max_depth: int = -1,
    min_data_in_bin: int = 1,
    min_data_in_leaf: int = 2,
    lambda_l2: float = 0.0,
    early_stopping_rounds: int = 100,
    num_threads: int = 1,
    device_type: str = "cpu",
    gpu_platform_id: int = 0,
    gpu_device_id: int = 0,
    gpu_fallback_to_cpu: bool = True,
    ensemble_runs: int = 1,
    ensemble_feature_fraction: float = 0.8,
    ensemble_bagging_fraction: float = 0.8,
    ensemble_bagging_freq: int = 1,
    threshold_dedup_eps: float = 1e-9,
) -> LightGBMBinner:
    """Fit LightGBM and extract per-feature split thresholds as discretization edges.

    When ``ensemble_runs > 1``, this fits multiple stochastic LightGBM models and
    unions split thresholds by aggregating split-gain scores per feature.
    """
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    if ensemble_runs < 1:
        raise ValueError("ensemble_runs must be at least 1")
    if not (0.0 < ensemble_feature_fraction <= 1.0):
        raise ValueError("ensemble_feature_fraction must be in (0, 1]")
    if not (0.0 < ensemble_bagging_fraction <= 1.0):
        raise ValueError("ensemble_bagging_fraction must be in (0, 1]")
    if ensemble_bagging_freq < 0:
        raise ValueError("ensemble_bagging_freq must be >= 0")
    if threshold_dedup_eps < 0:
        raise ValueError("threshold_dedup_eps must be >= 0")
    if min_data_in_leaf < 1:
        raise ValueError("min_data_in_leaf must be at least 1")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be >= 0")
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be >= 0")

    X_arr, y_arr = check_X_y(X_train, y_train, ensure_2d=True, dtype=float)
    y_bin = _encode_binary_target(y_arr)
    ordered_classes = sorted(np.unique(y_arr).tolist(), key=lambda v: str(v))
    n_features = X_arr.shape[1]
    X_val_work: Optional[np.ndarray] = None
    y_val_bin: Optional[np.ndarray] = None
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must both be provided for validation early stopping.")
        X_val_arr, y_val_arr = check_X_y(X_val, y_val, ensure_2d=True, dtype=float)
        if X_val_arr.shape[1] != n_features:
            raise ValueError(
                f"validation feature count mismatch: train has {n_features}, val has {X_val_arr.shape[1]}"
            )
        X_val_work = X_val_arr.copy()
        y_val_unique = np.unique(y_val_arr)
        if y_val_unique.size == 1:
            y_val_bin = np.where(y_val_arr == ordered_classes[0], 0, 1).astype(np.int32)
        else:
            y_val_bin = _encode_binary_target(y_val_arr)

    X_work = X_arr.copy()
    fill_values = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        col = X_work[:, j]
        if np.isnan(col).any():
            finite = col[~np.isnan(col)]
            fill = float(np.median(finite)) if finite.size > 0 else 0.0
            fill_values[j] = fill
            col[np.isnan(col)] = fill
            X_work[:, j] = col
        else:
            fill_values[j] = 0.0
    if X_val_work is not None:
        for j in range(n_features):
            col = X_val_work[:, j]
            if np.isnan(col).any():
                col[np.isnan(col)] = fill_values[j]
                X_val_work[:, j] = col

    requested_device = str(device_type).strip().lower()
    if requested_device not in {"cpu", "gpu", "cuda"}:
        raise ValueError(f"device_type must be one of ['cpu', 'gpu', 'cuda'], got {device_type!r}")

    def _fit_for_device(
        device: str,
        run_seed: Optional[int],
        run_feature_fraction: float,
        run_bagging_fraction: float,
        run_bagging_freq: int,
    ) -> LGBMClassifier:
        subsample_freq = run_bagging_freq if run_bagging_fraction < 0.999 else 0
        params = {
            "objective": "binary",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_data_in_leaf,
            "min_data_in_bin": min_data_in_bin,
            "reg_lambda": lambda_l2,
            "colsample_bytree": run_feature_fraction,
            "subsample": run_bagging_fraction,
            "subsample_freq": subsample_freq,
            "random_state": run_seed,
            "n_jobs": num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
            "device_type": device,
        }
        if device != "cpu":
            params["gpu_platform_id"] = int(gpu_platform_id)
            params["gpu_device_id"] = int(gpu_device_id)
        model = LGBMClassifier(**params)
        fit_kwargs = {}
        if X_val_work is not None and y_val_bin is not None and int(early_stopping_rounds) > 0:
            fit_kwargs = {
                "eval_set": [(X_val_work, y_val_bin)],
                "eval_metric": "binary_logloss",
                "callbacks": [lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)],
            }
        model.fit(X_work, y_bin, **fit_kwargs)
        return model

    feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
    actual_device = requested_device
    run_device = requested_device
    for run_idx in range(int(ensemble_runs)):
        if run_idx == 0:
            run_feature_fraction = float(feature_fraction)
            run_bagging_fraction = float(bagging_fraction)
            run_bagging_freq = int(bagging_freq)
        else:
            run_feature_fraction = min(float(feature_fraction), float(ensemble_feature_fraction))
            run_bagging_fraction = min(float(bagging_fraction), float(ensemble_bagging_fraction))
            run_bagging_freq = max(int(bagging_freq), int(ensemble_bagging_freq))

        run_seed = None if random_state is None else int(random_state + 1009 * run_idx)
        try:
            lgbm = _fit_for_device(
                run_device,
                run_seed=run_seed,
                run_feature_fraction=run_feature_fraction,
                run_bagging_fraction=run_bagging_fraction,
                run_bagging_freq=run_bagging_freq,
            )
        except Exception:
            if run_device == "cpu" or not gpu_fallback_to_cpu:
                raise
            run_device = "cpu"
            actual_device = "cpu"
            lgbm = _fit_for_device(
                run_device,
                run_seed=run_seed,
                run_feature_fraction=run_feature_fraction,
                run_bagging_fraction=run_bagging_fraction,
                run_bagging_freq=run_bagging_freq,
            )

        run_feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
        dump = lgbm.booster_.dump_model()
        for tree_info in dump.get("tree_info", []):
            tree_structure = tree_info.get("tree_structure")
            if isinstance(tree_structure, dict):
                _collect_threshold_scores(tree_structure, run_feature_scores)
        _merge_threshold_score_maps(feature_scores, run_feature_scores)

    max_edges = max_bins - 1
    bin_edges_per_feature: List[np.ndarray] = []

    for j in range(n_features):
        feature_values = X_work[:, j]
        unique_values = np.unique(feature_values)

        if unique_values.size <= 1:
            edges = np.array([], dtype=float)
        elif unique_values.size <= 2:
            edges = _binary_edge(unique_values)
        else:
            lo = float(np.min(feature_values))
            hi = float(np.max(feature_values))

            candidates = [
                (thr, score)
                for thr, score in feature_scores[j].items()
                if np.isfinite(thr) and lo < float(thr) < hi
            ]

            if candidates:
                candidates.sort(key=lambda item: (-item[1], item[0]))
                chosen = []
                for threshold, _ in candidates:
                    t = float(threshold)
                    if threshold_dedup_eps > 0 and any(abs(prev - t) <= threshold_dedup_eps for prev in chosen):
                        continue
                    if threshold_dedup_eps == 0 and chosen and abs(chosen[-1] - t) < 1e-12:
                        continue
                    chosen.append(t)
                    if len(chosen) >= max_edges:
                        break
                edges = np.array(sorted(chosen), dtype=float)
            else:
                edges = _quantile_edges(feature_values, max_edges)

            if edges.size == 0:
                edges = _quantile_edges(feature_values, max_edges)

            if edges.size > max_edges:
                edges = edges[:max_edges]

        bin_edges_per_feature.append(np.sort(edges))

    return LightGBMBinner(
        bin_edges_per_feature=bin_edges_per_feature,
        fill_values_per_feature=fill_values,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        device_type=actual_device,
    )
