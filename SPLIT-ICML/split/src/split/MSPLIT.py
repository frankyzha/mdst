"""Multiway SPLIT-style tree solver with CART discretization and lookahead greedy completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import json
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted

from .cart_binning import fit_cart_binner

try:
    from ._libgosdt import msplit_fit as _cpp_msplit_fit
except Exception:
    _cpp_msplit_fit = None


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


@dataclass
class MultiLeaf:
    prediction: int
    loss: float
    n_samples: int
    class_counts: Tuple[int, int]


@dataclass
class MultiNode:
    feature: int
    children: Dict[int, Union["MultiNode", MultiLeaf]]
    fallback_bin: int
    fallback_prediction: int
    group_count: int
    n_samples: int


@dataclass
class BoundResult:
    lb: float
    ub: float
    tree: Union[MultiNode, MultiLeaf]


class MSPLIT(ClassifierMixin, BaseEstimator):
    """True k-ary tree solver with SPLIT-style lookahead boundary behavior.

    Above lookahead depth, the solver performs systematic DP recursion.
    At exactly the lookahead depth, it computes a greedy completion and sets
    ``lb = ub = greedy_objective`` for that subproblem.
    """

    def __init__(
        self,
        lookahead_depth_budget: int = 2,
        full_depth_budget: int = 5,
        reg: float = 0.01,
        branch_penalty: float = 0.0,
        max_bins: int = 5,
        min_samples_leaf: int = 10,
        min_child_size: int = 5,
        max_branching: int = 0,
        time_limit: int = 100,
        verbose: bool = False,
        random_state: int = 0,
        input_is_binned: bool = False,
        use_cpp_solver: bool = True,
    ):
        self.lookahead_depth_budget = lookahead_depth_budget
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        self.branch_penalty = branch_penalty
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_child_size = min_child_size
        self.max_branching = max_branching
        self.time_limit = time_limit
        self.verbose = verbose
        self.random_state = random_state
        self.input_is_binned = input_is_binned
        self.use_cpp_solver = use_cpp_solver

    def fit(self, X, y):
        y_encoded, class_labels = self._encode_target(y)
        self.class_labels_ = class_labels
        self.classes_ = class_labels

        if self.input_is_binned:
            Z = check_array(X, ensure_2d=True, dtype=np.int32)
            if (Z < 0).any():
                raise ValueError("Binned input must be non-negative integer values")
            self.preprocessor_ = None
            self.binner_ = None
            self.feature_names_ = [f"x{i}" for i in range(Z.shape[1])]
        else:
            X_processed, feature_names = self._fit_and_transform_preprocessor(X)
            self.feature_names_ = feature_names
            self.binner_ = fit_cart_binner(
                X_processed,
                y_encoded,
                max_bins=self.max_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            Z = self.binner_.transform(X_processed)

        self._Z_train = np.asarray(Z, dtype=np.int32)
        self._y_train = np.asarray(y_encoded, dtype=np.int32)
        self._n_train = int(self._Z_train.shape[0])
        self._n_features = int(self._Z_train.shape[1])

        if self._n_train == 0:
            raise ValueError("Cannot fit MSPLIT on an empty dataset")
        if self.full_depth_budget < 1:
            raise ValueError("full_depth_budget must be at least 1")
        if self.min_child_size < 1:
            raise ValueError("min_child_size must be at least 1")
        if self.max_branching < 0:
            raise ValueError("max_branching must be >= 0 (0 means unlimited)")
        if self.reg < 0:
            raise ValueError("reg must be non-negative")
        if self.branch_penalty < 0:
            raise ValueError("branch_penalty must be non-negative")

        self.effective_lookahead_depth_ = max(1, min(self.lookahead_depth_budget, self.full_depth_budget))
        if self.use_cpp_solver and _cpp_msplit_fit is not None:
            cpp_result = _cpp_msplit_fit(
                self._Z_train,
                self._y_train,
                int(self.full_depth_budget),
                int(self.effective_lookahead_depth_),
                float(self.reg),
                float(self.branch_penalty),
                int(self.min_child_size),
                float(self.time_limit),
                int(self.max_branching),
            )
            tree_obj = json.loads(str(cpp_result["tree"]))
            self.tree_ = self._dict_to_tree(tree_obj)
            self.lower_bound_ = float(cpp_result["lowerbound"])
            self.upper_bound_ = float(cpp_result["upperbound"])
            self.objective_ = float(cpp_result["objective"])
            self.exact_internal_nodes_ = int(cpp_result.get("exact_internal_nodes", 0))
            self.greedy_internal_nodes_ = int(cpp_result.get("greedy_internal_nodes", 0))
        else:
            if self.use_cpp_solver and _cpp_msplit_fit is None:
                warnings.warn(
                    "MSPLIT C++ solver unavailable; falling back to Python DP solver.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            warnings.warn(
                "Python fallback solver uses per-bin branching; adaptive bin-group arity optimization is available in C++ mode.",
                RuntimeWarning,
                stacklevel=2,
            )

            self._start_time = time.perf_counter()
            self._dp_cache: Dict[Tuple[bytes, int], BoundResult] = {}
            self._greedy_cache: Dict[Tuple[bytes, int], Tuple[float, Union[MultiNode, MultiLeaf]]] = {}
            self._exact_internal_nodes_count = 0
            self._greedy_internal_nodes_count = 0

            root_indices = np.arange(self._n_train, dtype=np.int32)
            result = self._solve_subproblem(root_indices, self.full_depth_budget, current_depth=0)
            self.tree_ = result.tree
            self.lower_bound_ = float(result.lb)
            self.upper_bound_ = float(result.ub)
            self.objective_ = float(result.ub)
            self.exact_internal_nodes_ = int(self._exact_internal_nodes_count)
            self.greedy_internal_nodes_ = int(self._greedy_internal_nodes_count)

        self.tree = self._format_tree(self.tree_)
        self.n_features_in_ = self._n_features
        return self

    def predict(self, X):
        check_is_fitted(self, ["tree_", "classes_"])
        Z = self._prepare_features_for_predict(X)
        preds = np.zeros(Z.shape[0], dtype=np.int32)
        for i in range(Z.shape[0]):
            preds[i] = self._predict_row(Z[i], self.tree_)
        return self.classes_[preds]

    def _prepare_features_for_predict(self, X) -> np.ndarray:
        if self.input_is_binned:
            Z = check_array(X, ensure_2d=True, dtype=np.int32)
            if (Z < 0).any():
                raise ValueError("Binned input must be non-negative integer values")
            return Z

        X_processed = self._transform_preprocessor(X)
        return self.binner_.transform(X_processed)

    def _predict_row(self, row: np.ndarray, node: Union[MultiNode, MultiLeaf]) -> int:
        cur = node
        while isinstance(cur, MultiNode):
            bin_id = int(row[cur.feature])
            child = cur.children.get(bin_id)
            if child is None:
                # Route to nearest known bin if this bin was unseen during training.
                if cur.children:
                    nearest = min(cur.children.keys(), key=lambda b: (abs(int(b) - bin_id), int(b)))
                    child = cur.children.get(nearest)
                if child is None:
                    return cur.fallback_prediction
            cur = child
        return cur.prediction

    def _solve_subproblem(self, indices: np.ndarray, depth_remaining: int, current_depth: int) -> BoundResult:
        self._check_timeout()
        canonical_indices = np.sort(indices, kind="mergesort")
        key = (canonical_indices.tobytes(), depth_remaining)
        cached = self._dp_cache.get(key)
        if cached is not None:
            return cached

        leaf_objective, leaf_tree = self._leaf_solution(canonical_indices)
        pure = self._is_pure(canonical_indices)

        if depth_remaining <= 1 or pure:
            result = BoundResult(lb=leaf_objective, ub=leaf_objective, tree=leaf_tree)
            self._dp_cache[key] = result
            return result

        if current_depth == self.effective_lookahead_depth_:
            greedy_obj, greedy_tree = self._greedy_complete(indices, depth_remaining)
            result = BoundResult(lb=greedy_obj, ub=greedy_obj, tree=greedy_tree)
            self._dp_cache[key] = result
            return result

        best_tree: Union[MultiNode, MultiLeaf] = leaf_tree
        best_lb = leaf_objective
        best_ub = leaf_objective

        for feature in range(self._n_features):
            partition = self._partition_indices(canonical_indices, feature)
            if partition is None:
                continue

            split_lb = 0.0
            split_ub = 0.0
            children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
            largest_bin = -1
            largest_size = -1

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_result = self._solve_subproblem(child_indices, depth_remaining - 1, current_depth + 1)
                split_lb += child_result.lb
                split_ub += child_result.ub
                children[bin_id] = child_result.tree
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            split_penalty = self.branch_penalty * max(0, len(children) - 2)
            split_lb += split_penalty
            split_ub += split_penalty
            best_lb = min(best_lb, split_lb)
            if split_ub < best_ub:
                best_ub = split_ub
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    fallback_bin=largest_bin,
                    fallback_prediction=leaf_tree.prediction,
                    group_count=len(children),
                    n_samples=int(canonical_indices.size),
                )

        best_lb = min(best_lb, best_ub)
        result = BoundResult(lb=best_lb, ub=best_ub, tree=best_tree)
        if isinstance(best_tree, MultiNode):
            self._exact_internal_nodes_count += 1
        self._dp_cache[key] = result
        return result

    def _greedy_complete(self, indices: np.ndarray, depth_remaining: int) -> Tuple[float, Union[MultiNode, MultiLeaf]]:
        self._check_timeout()
        canonical_indices = np.sort(indices, kind="mergesort")
        key = (canonical_indices.tobytes(), depth_remaining)
        cached = self._greedy_cache.get(key)
        if cached is not None:
            return cached

        leaf_objective, leaf_tree = self._leaf_solution(canonical_indices)
        pure = self._is_pure(canonical_indices)
        if depth_remaining <= 1 or pure:
            result = (leaf_objective, leaf_tree)
            self._greedy_cache[key] = result
            return result

        best_objective = leaf_objective
        best_tree: Union[MultiNode, MultiLeaf] = leaf_tree

        for feature in range(self._n_features):
            partition = self._partition_indices(canonical_indices, feature)
            if partition is None:
                continue

            split_objective = 0.0
            children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
            largest_bin = -1
            largest_size = -1

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_obj, child_tree = self._greedy_complete(child_indices, depth_remaining - 1)
                split_objective += child_obj
                children[bin_id] = child_tree
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            split_objective += self.branch_penalty * max(0, len(children) - 2)
            if split_objective < best_objective:
                best_objective = split_objective
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    fallback_bin=largest_bin,
                    fallback_prediction=leaf_tree.prediction,
                    group_count=len(children),
                    n_samples=int(canonical_indices.size),
                )

        result = (best_objective, best_tree)
        if isinstance(best_tree, MultiNode):
            self._greedy_internal_nodes_count += 1
        self._greedy_cache[key] = result
        return result

    def _partition_indices(self, indices: np.ndarray, feature: int) -> Optional[Dict[int, np.ndarray]]:
        feature_values = self._Z_train[indices, feature]
        unique_bins = np.unique(feature_values)
        if unique_bins.size <= 1:
            return None

        children: Dict[int, np.ndarray] = {}
        for bin_id in unique_bins:
            mask = feature_values == bin_id
            child_indices = indices[mask]
            if child_indices.size < self.min_child_size:
                return None
            children[int(bin_id)] = child_indices
        return children

    def _leaf_solution(self, indices: np.ndarray) -> Tuple[float, MultiLeaf]:
        y_subset = self._y_train[indices]
        positives = int(y_subset.sum())
        negatives = int(y_subset.size - positives)

        if positives >= negatives:
            prediction = 1
            mistakes = negatives
        else:
            prediction = 0
            mistakes = positives

        loss = mistakes / float(self._n_train) + self.reg
        return loss, MultiLeaf(
            prediction=prediction,
            loss=loss,
            n_samples=int(indices.size),
            class_counts=(negatives, positives),
        )

    def _is_pure(self, indices: np.ndarray) -> bool:
        y_subset = self._y_train[indices]
        return bool(np.all(y_subset == y_subset[0]))

    def _check_timeout(self):
        if self.time_limit and self.time_limit > 0:
            elapsed = time.perf_counter() - self._start_time
            if elapsed > self.time_limit:
                raise TimeoutError(f"MSPLIT exceeded time_limit={self.time_limit} seconds")

    def _dict_to_tree(self, tree_obj: dict) -> Union[MultiNode, MultiLeaf]:
        node_type = tree_obj.get("type")
        if node_type == "leaf":
            class_counts = tree_obj.get("class_counts", [0, 0])
            return MultiLeaf(
                prediction=int(tree_obj["prediction"]),
                loss=float(tree_obj["loss"]),
                n_samples=int(tree_obj.get("n_samples", 0)),
                class_counts=(int(class_counts[0]), int(class_counts[1])),
            )

        children_raw = tree_obj.get("children", {})
        children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
        for key in sorted(children_raw.keys(), key=lambda x: int(x)):
            children[int(key)] = self._dict_to_tree(children_raw[key])

        return MultiNode(
            feature=int(tree_obj["feature"]),
            children=children,
            fallback_bin=int(tree_obj["fallback_bin"]),
            fallback_prediction=int(tree_obj["fallback_prediction"]),
            group_count=int(tree_obj.get("group_count", len(children))),
            n_samples=int(tree_obj.get("n_samples", 0)),
        )

    def _encode_target(self, y) -> Tuple[np.ndarray, np.ndarray]:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        classes = np.unique(y_arr)
        if classes.size == 0:
            raise ValueError("Target y must not be empty")

        if classes.size != 2:
            raise ValueError(
                f"MSPLIT currently supports binary targets only; received {classes.size} classes: {classes.tolist()}"
            )

        ordered = sorted(classes.tolist(), key=lambda v: str(v))
        mapping = {ordered[0]: 0, ordered[1]: 1}
        y_bin = np.array([mapping[val] for val in y_arr], dtype=np.int32)
        labels = np.array(
            [_to_python_scalar(ordered[0]), _to_python_scalar(ordered[1])],
            dtype=object,
        )
        return y_bin, labels

    def _fit_and_transform_preprocessor(self, X) -> Tuple[np.ndarray, list[str]]:
        if hasattr(X, "select_dtypes") and hasattr(X, "columns"):
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
            categorical_cols = [col for col in X.columns if col not in numeric_cols]

            transformers = []
            if numeric_cols:
                transformers.append(
                    (
                        "num",
                        Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                        numeric_cols,
                    )
                )
            if categorical_cols:
                try:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                except TypeError:
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                transformers.append(
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", ohe),
                            ]
                        ),
                        categorical_cols,
                    )
                )

            self.preprocessor_ = ColumnTransformer(transformers=transformers, remainder="drop")
            X_processed = self.preprocessor_.fit_transform(X)
            feature_names = self.preprocessor_.get_feature_names_out().tolist()
            return np.asarray(X_processed, dtype=float), feature_names

        self.preprocessor_ = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        X_arr = check_array(X, ensure_2d=True, dtype=float, force_all_finite="allow-nan")
        X_processed = self.preprocessor_.fit_transform(X_arr)
        feature_names = [f"x{i}" for i in range(X_processed.shape[1])]
        return np.asarray(X_processed, dtype=float), feature_names

    def _transform_preprocessor(self, X) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Preprocessor is not available; model was fit with input_is_binned=True")

        if hasattr(X, "select_dtypes") and hasattr(X, "columns"):
            X_processed = self.preprocessor_.transform(X)
        else:
            X_arr = check_array(X, ensure_2d=True, dtype=float, force_all_finite="allow-nan")
            X_processed = self.preprocessor_.transform(X_arr)
        return np.asarray(X_processed, dtype=float)

    def _format_tree(self, node: Union[MultiNode, MultiLeaf], depth: int = 0) -> str:
        indent = "  " * depth
        if isinstance(node, MultiLeaf):
            pred_label = self.class_labels_[node.prediction]
            return (
                f"{indent}Leaf(pred={pred_label!r}, n={node.n_samples}, "
                f"class_counts={node.class_counts}, loss={node.loss:.6f})"
            )

        lines = [
            f"{indent}Node(feature={node.feature}, groups={node.group_count}, fallback_bin={node.fallback_bin}, "
            f"fallback_pred={self.class_labels_[node.fallback_prediction]!r}, n={node.n_samples})"
        ]
        for bin_id in sorted(node.children.keys()):
            lines.append(f"{indent}  bin {bin_id} ->")
            lines.append(self._format_tree(node.children[bin_id], depth + 2))
        return "\n".join(lines)
