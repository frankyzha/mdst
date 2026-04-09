from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _minority_count(y: np.ndarray) -> int:
    y_int = np.asarray(y, dtype=np.int32).reshape(-1)
    if y_int.size == 0:
        return 0
    pos = int(np.sum(y_int == 1))
    neg = int(y_int.size - pos)
    return min(pos, neg)


def _leaf_objective(y: np.ndarray, reg: float) -> float:
    return float(_minority_count(y) + reg)


def _teacher_screening(X: np.ndarray, teacher: np.ndarray) -> list[int]:
    teacher_arr = np.asarray(teacher, dtype=np.float64).reshape(-1)
    ranked: list[tuple[float, int]] = []
    for feature_idx in range(X.shape[1]):
        col = np.asarray(X[:, feature_idx], dtype=np.float64)
        if col.size == 0:
            ranked.append((float("inf"), feature_idx))
            continue
        quantiles = np.unique(np.quantile(col, [0.2, 0.4, 0.6, 0.8]))
        bins = np.digitize(col, quantiles, right=False)
        distortion = 0.0
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            if not np.any(mask):
                continue
            segment = teacher_arr[mask]
            distortion += (float(segment.size) / float(teacher_arr.size)) * float(np.var(segment))
        ranked.append((distortion, feature_idx))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [feature_idx for _, feature_idx in ranked]


def _ordered_groups_from_tree(
    tree: Any,
    X_fit: np.ndarray,
    X_any: np.ndarray,
    order_signal_fit: np.ndarray,
    feature_subset: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    fit_leaf = tree.apply(X_fit[:, list(feature_subset)])
    ordered_leaf_ids = sorted(
        np.unique(fit_leaf),
        key=lambda leaf_id: (
            float(np.mean(order_signal_fit[fit_leaf == leaf_id])),
            int(leaf_id),
        ),
    )
    leaf_map = {int(leaf_id): mapped for mapped, leaf_id in enumerate(ordered_leaf_ids)}
    fit_group = np.fromiter((leaf_map[int(v)] for v in fit_leaf), dtype=np.int32)
    any_group = np.fromiter(
        (leaf_map.get(int(v), 0) for v in tree.apply(X_any[:, list(feature_subset)])),
        dtype=np.int32,
    )
    return fit_group, any_group, leaf_map


def _ordered_three_group_map(
    ordered_leaf_ids: list[int],
    fit_leaf: np.ndarray,
    y_fit: np.ndarray,
) -> dict[int, int]:
    leaf_ids = [int(v) for v in ordered_leaf_ids]
    if not leaf_ids:
        return {}
    if len(leaf_ids) <= 3:
        return {leaf_id: idx for idx, leaf_id in enumerate(leaf_ids)}

    y_arr = np.asarray(y_fit, dtype=np.int32).reshape(-1)
    fit_leaf_arr = np.asarray(fit_leaf).reshape(-1)
    pos_prefix = np.zeros(len(leaf_ids) + 1, dtype=np.float64)
    neg_prefix = np.zeros(len(leaf_ids) + 1, dtype=np.float64)
    for idx, leaf_id in enumerate(leaf_ids, start=1):
        mask = fit_leaf_arr == int(leaf_id)
        pos = float(np.sum(y_arr[mask] == 1))
        neg = float(np.sum(mask) - pos)
        pos_prefix[idx] = pos_prefix[idx - 1] + pos
        neg_prefix[idx] = neg_prefix[idx - 1] + neg

    def seg_loss(lo: int, hi: int) -> float:
        pos = pos_prefix[hi] - pos_prefix[lo]
        neg = neg_prefix[hi] - neg_prefix[lo]
        return min(pos, neg)

    best = (float("inf"), 1, len(leaf_ids) - 1)
    for cut1 in range(1, len(leaf_ids) - 1):
        for cut2 in range(cut1 + 1, len(leaf_ids)):
            loss = seg_loss(0, cut1) + seg_loss(cut1, cut2) + seg_loss(cut2, len(leaf_ids))
            cand = (loss, cut1, cut2)
            if cand < best:
                best = cand
    _, cut1, cut2 = best
    leaf_map: dict[int, int] = {}
    for idx, leaf_id in enumerate(leaf_ids):
        if idx < cut1:
            group = 0
        elif idx < cut2:
            group = 1
        else:
            group = 2
        leaf_map[int(leaf_id)] = group
    return leaf_map


def _collapsed_groups_from_tree(
    tree: Any,
    X_fit: np.ndarray,
    X_any: np.ndarray,
    order_signal_fit: np.ndarray,
    y_fit: np.ndarray,
    feature_subset: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    fit_leaf = tree.apply(X_fit[:, list(feature_subset)])
    ordered_leaf_ids = sorted(
        np.unique(fit_leaf),
        key=lambda leaf_id: (
            float(np.mean(order_signal_fit[fit_leaf == leaf_id])),
            int(leaf_id),
        ),
    )
    leaf_map = _ordered_three_group_map(ordered_leaf_ids, fit_leaf, y_fit)
    fit_group = np.fromiter((leaf_map[int(v)] for v in fit_leaf), dtype=np.int32)
    any_group = np.fromiter(
        (leaf_map.get(int(v), 0) for v in tree.apply(X_any[:, list(feature_subset)])),
        dtype=np.int32,
    )
    return fit_group, any_group, leaf_map


def _teacher_distortion(teacher: np.ndarray, groups: np.ndarray) -> float:
    teacher_arr = np.asarray(teacher, dtype=np.float64).reshape(-1)
    groups_arr = np.asarray(groups, dtype=np.int32).reshape(-1)
    if teacher_arr.size == 0 or teacher_arr.size != groups_arr.size:
        return float("inf")
    total = float(teacher_arr.size)
    distortion = 0.0
    for group_id in np.unique(groups_arr):
        mask = groups_arr == group_id
        if not np.any(mask):
            continue
        segment = teacher_arr[mask]
        distortion += (float(segment.size) / total) * float(np.var(segment))
    return float(distortion)


def _label_partition_loss(y: np.ndarray, groups: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    groups_arr = np.asarray(groups, dtype=np.int32).reshape(-1)
    if y_arr.size == 0 or y_arr.size != groups_arr.size:
        return float("inf")
    total = float(y_arr.size)
    loss = 0.0
    for group_id in np.unique(groups_arr):
        mask = groups_arr == group_id
        if not np.any(mask):
            continue
        segment = y_arr[mask]
        pos = float(np.sum(segment == 1))
        neg = float(segment.size - pos)
        loss += min(pos, neg) / total
    return float(loss)


@dataclass
class LeafNode:
    prediction: int
    objective: float


@dataclass
class SubtreeNode:
    model: Any
    objective: float


@dataclass
class SplitNode:
    feature_subset: tuple[int, ...]
    template_tree: DecisionTreeRegressor
    leaf_map: dict[int, int]
    children: dict[int, Any]
    objective: float


@dataclass
class TemplateCandidate:
    teacher_distortion: float
    label_loss: float
    feature_subset: tuple[int, ...]
    template_tree: Any
    fit_group: np.ndarray
    leaf_map: dict[int, int]


class TeacherTemplateHybridTree:
    def __init__(
        self,
        *,
        reg: float,
        min_child_size: int,
        template_depth: int = 2,
        screen_feature_count: int = 4,
        shortlist_size: int = 3,
        template_min_leaf: int = 0,
        pair_shape_max_leaves: int = 8,
        pair_only: bool = False,
        random_state: int = 0,
    ) -> None:
        self.reg = float(reg)
        self.min_child_size = int(min_child_size)
        self.template_depth = int(template_depth)
        self.screen_feature_count = int(screen_feature_count)
        self.shortlist_size = int(shortlist_size)
        self.template_min_leaf = int(template_min_leaf)
        self.pair_shape_max_leaves = int(pair_shape_max_leaves)
        self.pair_only = bool(pair_only)
        self.random_state = int(random_state)
        self.root_: LeafNode | SubtreeNode | SplitNode | None = None

    def fit(
        self,
        X_fit: np.ndarray,
        Z_fit: np.ndarray,
        y_fit: np.ndarray,
        teacher_fit: np.ndarray,
        depth_budget: int,
        fit_subtree_fn,
    ) -> "TeacherTemplateHybridTree":
        self.root_ = self._build_node(
            X_fit=np.asarray(X_fit, dtype=np.float32),
            Z_fit=np.asarray(Z_fit, dtype=np.int32),
            y_fit=np.asarray(y_fit, dtype=np.int32),
            teacher_fit=np.asarray(teacher_fit, dtype=np.float64),
            depth_remaining=int(depth_budget),
            template_depth_remaining=int(self.template_depth),
            fit_subtree_fn=fit_subtree_fn,
        )
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("model is not fit")
        out = np.zeros(Z.shape[0], dtype=np.int32)
        self._predict_into(self.root_, np.asarray(X, dtype=np.float32), np.asarray(Z, dtype=np.int32), np.arange(Z.shape[0]), out)
        return out

    def _build_node(
        self,
        *,
        X_fit: np.ndarray,
        Z_fit: np.ndarray,
        y_fit: np.ndarray,
        teacher_fit: np.ndarray,
        depth_remaining: int,
        template_depth_remaining: int,
        fit_subtree_fn,
    ) -> LeafNode | SubtreeNode | SplitNode:
        if (
            depth_remaining <= 0
            or template_depth_remaining <= 0
            or y_fit.size < max(2, self.min_child_size * 2)
            or np.unique(y_fit).size <= 1
        ):
            return fit_subtree_fn(Z_fit, y_fit, depth_remaining, teacher_fit)

        ranked = _teacher_screening(X_fit, teacher_fit)[: max(1, self.screen_feature_count)]
        candidate_features: list[tuple[int, ...]] = []
        if self.pair_only:
            candidate_features.extend(combinations(ranked, 2))
        else:
            candidate_features.extend((feature_idx,) for feature_idx in ranked)
            candidate_features.extend(combinations(ranked, 2))
            for prefix_size in range(3, len(ranked) + 1):
                candidate_features.append(tuple(ranked[:prefix_size]))
        candidate_features = list(dict.fromkeys(tuple(int(v) for v in subset) for subset in candidate_features))

        candidates: list[TemplateCandidate] = []
        fit_min_leaf = max(2, self.template_min_leaf if self.template_min_leaf > 0 else self.min_child_size)
        for feature_subset in candidate_features:
            template_tree = DecisionTreeRegressor(
                max_depth=2,
                max_leaf_nodes=3,
                min_samples_leaf=fit_min_leaf,
                random_state=self.random_state,
            )
            template_tree.fit(X_fit[:, list(feature_subset)], teacher_fit)
            fit_group, _, leaf_map = _ordered_groups_from_tree(
                template_tree,
                X_fit,
                X_fit,
                teacher_fit,
                feature_subset,
            )
            if int(np.unique(fit_group).size) >= 2:
                candidates.append(
                    TemplateCandidate(
                        teacher_distortion=_teacher_distortion(teacher_fit, fit_group),
                        label_loss=_label_partition_loss(y_fit, fit_group),
                        feature_subset=tuple(int(v) for v in feature_subset),
                        template_tree=template_tree,
                        fit_group=fit_group,
                        leaf_map=leaf_map,
                    )
                )
            label_tree = DecisionTreeClassifier(
                max_depth=2,
                max_leaf_nodes=3,
                min_samples_leaf=fit_min_leaf,
                random_state=self.random_state,
            )
            label_tree.fit(X_fit[:, list(feature_subset)], y_fit)
            fit_group_lbl, _, leaf_map_lbl = _ordered_groups_from_tree(
                label_tree,
                X_fit,
                X_fit,
                y_fit.astype(np.float64, copy=False),
                feature_subset,
            )
            if int(np.unique(fit_group_lbl).size) < 2:
                pass
            else:
                candidates.append(
                    TemplateCandidate(
                        teacher_distortion=_teacher_distortion(teacher_fit, fit_group_lbl),
                        label_loss=_label_partition_loss(y_fit, fit_group_lbl),
                        feature_subset=tuple(int(v) for v in feature_subset),
                        template_tree=label_tree,
                        fit_group=fit_group_lbl,
                        leaf_map=leaf_map_lbl,
                    )
                )

            if len(feature_subset) == 2 and self.pair_shape_max_leaves > 3:
                teacher_shape_tree = DecisionTreeRegressor(
                    max_depth=3,
                    max_leaf_nodes=max(4, self.pair_shape_max_leaves),
                    min_samples_leaf=fit_min_leaf,
                    random_state=self.random_state,
                )
                teacher_shape_tree.fit(X_fit[:, list(feature_subset)], teacher_fit)
                fit_group_shape, _, leaf_map_shape = _collapsed_groups_from_tree(
                    teacher_shape_tree,
                    X_fit,
                    X_fit,
                    teacher_fit,
                    y_fit,
                    feature_subset,
                )
                if int(np.unique(fit_group_shape).size) >= 2:
                    candidates.append(
                        TemplateCandidate(
                            teacher_distortion=_teacher_distortion(teacher_fit, fit_group_shape),
                            label_loss=_label_partition_loss(y_fit, fit_group_shape),
                            feature_subset=tuple(int(v) for v in feature_subset),
                            template_tree=teacher_shape_tree,
                            fit_group=fit_group_shape,
                            leaf_map=leaf_map_shape,
                        )
                    )
                label_shape_tree = DecisionTreeClassifier(
                    max_depth=3,
                    max_leaf_nodes=max(4, self.pair_shape_max_leaves),
                    min_samples_leaf=fit_min_leaf,
                    random_state=self.random_state,
                )
                label_shape_tree.fit(X_fit[:, list(feature_subset)], y_fit)
                fit_group_lbl_shape, _, leaf_map_lbl_shape = _collapsed_groups_from_tree(
                    label_shape_tree,
                    X_fit,
                    X_fit,
                    y_fit.astype(np.float64, copy=False),
                    y_fit,
                    feature_subset,
                )
                if int(np.unique(fit_group_lbl_shape).size) >= 2:
                    candidates.append(
                        TemplateCandidate(
                            teacher_distortion=_teacher_distortion(teacher_fit, fit_group_lbl_shape),
                            label_loss=_label_partition_loss(y_fit, fit_group_lbl_shape),
                            feature_subset=tuple(int(v) for v in feature_subset),
                            template_tree=label_shape_tree,
                            fit_group=fit_group_lbl_shape,
                            leaf_map=leaf_map_lbl_shape,
                        )
                    )

        if not candidates:
            return fit_subtree_fn(Z_fit, y_fit, depth_remaining, teacher_fit)

        ordered_teacher = sorted(
            candidates,
            key=lambda item: (
                item.teacher_distortion,
                item.label_loss,
                len(item.feature_subset),
                item.feature_subset,
            ),
        )
        ordered_label = sorted(
            candidates,
            key=lambda item: (
                item.label_loss,
                item.teacher_distortion,
                len(item.feature_subset),
                item.feature_subset,
            ),
        )
        shortlist: list[TemplateCandidate] = []
        seen: set[tuple[tuple[int, ...], int, tuple[int, ...]]] = set()
        for ordered in (ordered_teacher, ordered_label):
            for item in ordered[: max(1, self.shortlist_size)]:
                key = (
                    item.feature_subset,
                    int(getattr(item.template_tree.tree_, "node_count", 0)),
                    tuple(int(v) for v in np.unique(item.fit_group)),
                )
                if key in seen:
                    continue
                seen.add(key)
                shortlist.append(item)

        best_node: SplitNode | LeafNode | SubtreeNode = fit_subtree_fn(Z_fit, y_fit, depth_remaining, teacher_fit)
        best_objective = float(getattr(best_node, "objective", _leaf_objective(y_fit, self.reg)))

        for item in shortlist:
            feature_subset = item.feature_subset
            template_tree = item.template_tree
            fit_group = item.fit_group
            leaf_map = item.leaf_map
            children: dict[int, Any] = {}
            total_objective = 0.0
            feasible = True
            for group_id in np.unique(fit_group):
                mask = fit_group == group_id
                child = self._build_node(
                    X_fit=X_fit[mask],
                    Z_fit=Z_fit[mask],
                    y_fit=y_fit[mask],
                    teacher_fit=teacher_fit[mask],
                    depth_remaining=depth_remaining - 1,
                    template_depth_remaining=template_depth_remaining - 1,
                    fit_subtree_fn=fit_subtree_fn,
                )
                child_objective = float(getattr(child, "objective", np.nan))
                if not np.isfinite(child_objective):
                    feasible = False
                    break
                children[int(group_id)] = child
                total_objective += child_objective
            if not feasible:
                continue
            if total_objective + 1e-12 < best_objective:
                best_objective = total_objective
                best_node = SplitNode(
                    feature_subset=feature_subset,
                    template_tree=template_tree,
                    leaf_map=leaf_map,
                    children=children,
                    objective=total_objective,
                )

        return best_node

    def _predict_into(
        self,
        node: LeafNode | SubtreeNode | SplitNode,
        X: np.ndarray,
        Z: np.ndarray,
        row_idx: np.ndarray,
        out: np.ndarray,
    ) -> None:
        if row_idx.size == 0:
            return
        if isinstance(node, LeafNode):
            out[row_idx] = node.prediction
            return
        if isinstance(node, SubtreeNode):
            out[row_idx] = np.asarray(node.model.predict(Z[row_idx]), dtype=np.int32)
            return
        groups = np.fromiter(
            (
                node.leaf_map.get(int(v), 0)
                for v in node.template_tree.apply(X[row_idx][:, list(node.feature_subset)])
            ),
            dtype=np.int32,
        )
        for group_id, child in node.children.items():
            child_rows = row_idx[groups == int(group_id)]
            self._predict_into(child, X, Z, child_rows, out)

    def count_nodes(self) -> tuple[int, int, int]:
        if self.root_ is None:
            return 0, 0, 0
        return self._count_nodes(self.root_)

    def _count_nodes(self, node: LeafNode | SubtreeNode | SplitNode) -> tuple[int, int, int]:
        if isinstance(node, LeafNode):
            return 1, 0, 0
        if isinstance(node, SubtreeNode):
            tree = getattr(node.model, "tree_", None)
            if tree is None:
                return 0, 0, 0
            return self._count_msplit_tree(tree)
        leaves = 0
        internal = 1
        max_arity = max(2, len(node.children))
        for child in node.children.values():
            child_leaves, child_internal, child_arity = self._count_nodes(child)
            leaves += child_leaves
            internal += child_internal
            max_arity = max(max_arity, child_arity)
        return leaves, internal, max_arity

    def _count_msplit_tree(self, node: Any) -> tuple[int, int, int]:
        if not hasattr(node, "children"):
            return 1, 0, 0
        leaves = 0
        internal = 1
        max_arity = int(max(getattr(node, "group_count", len(node.children)), 2))
        for child in node.children.values():
            child_leaves, child_internal, child_arity = self._count_msplit_tree(child)
            leaves += child_leaves
            internal += child_internal
            max_arity = max(max_arity, child_arity)
        return leaves, internal, max_arity
