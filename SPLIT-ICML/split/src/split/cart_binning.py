"""CART-based per-feature binning for multiway decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array


@dataclass
class CARTBinner:
    """Feature-wise CART binner that maps continuous columns to integer bins."""

    bin_edges_per_feature: List[np.ndarray]
    fill_values_per_feature: np.ndarray
    max_bins: int
    min_samples_leaf: int
    random_state: Optional[int]

    @property
    def n_bins_per_feature(self) -> np.ndarray:
        return np.array([int(edges.size + 1) if edges.size else 1 for edges in self.bin_edges_per_feature], dtype=np.int32)

    def transform(self, X) -> np.ndarray:
        X_arr = check_array(X, ensure_2d=True, dtype=float, force_all_finite="allow-nan")
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


def fit_cart_binner(
    X_train,
    y_train,
    max_bins: int = 5,
    min_samples_leaf: int = 10,
    random_state: Optional[int] = None,
) -> CARTBinner:
    """Fit feature-wise 1D CARTs and return a reusable binner."""
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")

    X_arr, y_arr = check_X_y(X_train, y_train, ensure_2d=True, dtype=float)
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

    bin_edges_per_feature: List[np.ndarray] = []
    for j in range(n_features):
        feature_values = X_work[:, j]
        unique_values = np.unique(feature_values)

        if unique_values.size <= 1:
            edges = np.array([], dtype=float)
        elif unique_values.size <= 2:
            edges = _binary_edge(unique_values)
        else:
            seed = None if random_state is None else int(random_state + j)
            tree = DecisionTreeClassifier(
                max_leaf_nodes=max_bins,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
            tree.fit(feature_values.reshape(-1, 1), y_arr)
            thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
            edges = np.unique(thresholds.astype(float))
            edges = np.sort(edges)
            if edges.size > max_bins - 1:
                edges = edges[: max_bins - 1]

        bin_edges_per_feature.append(edges)

    return CARTBinner(
        bin_edges_per_feature=bin_edges_per_feature,
        fill_values_per_feature=fill_values,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
