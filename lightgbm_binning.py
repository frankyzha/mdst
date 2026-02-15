"""LightGBM-guided per-feature binning for multiway decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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
    max_bins: int = 6,
    min_samples_leaf: int = 10,
    random_state: Optional[int] = None,
    n_estimators: int = 120,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    bagging_freq: int = 0,
    max_depth: int = -1,
    min_data_in_bin: int = 1,
    num_threads: int = 1,
) -> LightGBMBinner:
    """Fit LightGBM and extract per-feature split thresholds as discretization edges."""
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")

    X_arr, y_arr = check_X_y(X_train, y_train, ensure_2d=True, dtype=float)
    y_bin = _encode_binary_target(y_arr)
    n_features = X_arr.shape[1]

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

    subsample_freq = bagging_freq if bagging_fraction < 0.999 else 0

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_samples_leaf,
        min_data_in_bin=min_data_in_bin,
        colsample_bytree=feature_fraction,
        subsample=bagging_fraction,
        subsample_freq=subsample_freq,
        random_state=random_state,
        n_jobs=num_threads,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    lgbm.fit(X_work, y_bin)

    booster = lgbm.booster_
    dump = booster.dump_model()

    feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
    for tree_info in dump.get("tree_info", []):
        tree_structure = tree_info.get("tree_structure")
        if isinstance(tree_structure, dict):
            _collect_threshold_scores(tree_structure, feature_scores)

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
                    if chosen and abs(chosen[-1] - t) < 1e-12:
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
    )
