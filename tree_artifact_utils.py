"""Shared helpers for serializing fitted trees into JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def to_label_text(value: object) -> str:
    return str(value)


def write_artifact_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def feature_display_name(name: str, max_len: int = 24) -> str:
    if name.startswith("num__"):
        out = name[5:]
    elif name.startswith("cat__"):
        out = name[5:]
    else:
        out = name
    out = out.replace("__", "_")
    if len(out) <= max_len:
        return out
    return out[: max_len - 2] + ".."


def _format_float(value: float) -> str:
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = f"{float(value):.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _bins_to_spans(bins: Iterable[int]) -> List[Tuple[int, int]]:
    sorted_bins = sorted(set(int(b) for b in bins))
    if not sorted_bins:
        return []

    spans: List[Tuple[int, int]] = []
    start = sorted_bins[0]
    end = sorted_bins[0]
    for b in sorted_bins[1:]:
        if b == end + 1:
            end = b
            continue
        spans.append((start, end))
        start, end = b, b
    spans.append((start, end))
    return spans


def _span_to_bounds(start_bin: int, end_bin: int, edges: np.ndarray) -> Tuple[float, float]:
    lo = -np.inf if start_bin == 0 else float(edges[start_bin - 1])
    hi = np.inf if end_bin >= len(edges) else float(edges[end_bin])
    return lo, hi


def _format_numeric_union(spans: List[Tuple[int, int]], edges: np.ndarray) -> str:
    parts: List[str] = []
    for start_bin, end_bin in spans:
        lo, hi = _span_to_bounds(start_bin, end_bin, edges)
        if np.isneginf(lo) and np.isfinite(hi):
            parts.append(f"<= {_format_float(hi)}")
        elif np.isfinite(lo) and np.isposinf(hi):
            parts.append(f"> {_format_float(lo)}")
        elif np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"[{_format_float(lo)}, {_format_float(hi)}]")
        else:
            parts.append("all")
    return " U ".join(parts)


def _is_binary_onehot_feature(feature_name: str, edges: np.ndarray) -> bool:
    if not feature_name.startswith("cat__"):
        return False
    if edges is None or len(edges) != 1:
        return False
    return abs(float(edges[0]) - 0.5) < 1e-6


def _parse_onehot_name(feature_name: str) -> Tuple[str, str]:
    raw = feature_name[5:] if feature_name.startswith("cat__") else feature_name
    base, sep, category = raw.rpartition("_")
    if not sep:
        return feature_display_name(raw), "1"
    return feature_display_name(base), category


def format_msplit_condition(feature_idx: int, bins: Iterable[int], binner, feature_names: List[str]) -> str:
    feature_name = feature_names[feature_idx] if 0 <= feature_idx < len(feature_names) else f"x{feature_idx}"
    bins_list = sorted(set(int(b) for b in bins))
    bins_for_display = bins_list
    if len(bins_list) >= 2:
        # Merge gaps from unseen bins so labels stay readable.
        bins_for_display = list(range(bins_list[0], bins_list[-1] + 1))
    edges = binner.bin_edges_per_feature[feature_idx]

    if edges is None or len(edges) == 0:
        cond = "all"
    elif _is_binary_onehot_feature(feature_name, edges):
        _, category = _parse_onehot_name(feature_name)
        bins_set = set(bins_list)
        if bins_set == {0}:
            cond = f"!= {category}"
        elif bins_set == {1}:
            cond = f"= {category}"
        else:
            cond = f"in {{{category}, others}}"
        return cond
    else:
        spans = _bins_to_spans(bins_for_display)
        cond = _format_numeric_union(spans, edges)

    return cond


def is_leaf_node(node: object) -> bool:
    return not hasattr(node, "children")


def _expand_spans(spans: Iterable[Tuple[int, int]]) -> List[int]:
    bins: List[int] = []
    for lo, hi in spans:
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        bins.extend(range(lo_i, hi_i + 1))
    return bins


def _class_dist_from_counts(counts: Iterable[int], class_labels: np.ndarray) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, cnt in enumerate(counts):
        label = class_labels[i] if i < len(class_labels) else i
        out.append(
            {
                "class_index": int(i),
                "class_label": to_label_text(label),
                "count": int(cnt),
            }
        )
    return out


def _subtree_signature(node: object) -> Tuple:
    if is_leaf_node(node):
        prediction = int(getattr(node, "prediction", 0))
        class_counts = tuple(int(v) for v in getattr(node, "class_counts", (0, 0)))
        return ("leaf", prediction, class_counts)
    child_sigs = tuple(
        (
            tuple(int(b) for b in bins),
            _subtree_signature(child),
        )
        for bins, child in group_children(node)
    )
    return ("node", int(getattr(node, "feature", -1)), child_sigs)


def group_children(node: object) -> List[Tuple[List[int], object]]:
    if is_leaf_node(node):
        return []

    child_spans = getattr(node, "child_spans", None)
    if isinstance(child_spans, dict) and child_spans:
        grouped_children: List[Tuple[List[int], object]] = []
        for group_id in sorted(node.children.keys()):
            spans = child_spans.get(group_id, ())
            bins = _expand_spans(spans)
            if not bins:
                continue
            grouped_children.append((bins, node.children[group_id]))
        return grouped_children

    # C++ MSPLIT serializes one child per observed bin; merge adjacent observed-bin
    # entries when they share the same subtree.
    items = [(int(bin_id), child, _subtree_signature(child)) for bin_id, child in node.children.items()]
    items.sort(key=lambda item: item[0])
    if not items:
        return []

    grouped: List[Tuple[List[int], object, Tuple]] = []
    cur_bins = [items[0][0]]
    cur_child = items[0][1]
    cur_sig = items[0][2]
    for bin_id, child, sig in items[1:]:
        if sig == cur_sig:
            cur_bins.append(bin_id)
            continue
        grouped.append((cur_bins, cur_child, cur_sig))
        cur_bins = [bin_id]
        cur_child = child
        cur_sig = sig
    grouped.append((cur_bins, cur_child, cur_sig))
    return [(bins, child) for bins, child, _ in grouped]


def serialize_msplit_node(
    node: object,
    binner,
    feature_names: List[str],
    class_labels: np.ndarray,
    z_train: Optional[np.ndarray],
    idxs: Optional[np.ndarray],
    path_conditions: List[str],
) -> Dict[str, object]:
    n_samples = int(idxs.size) if idxs is not None else int(getattr(node, "n_samples", 0))
    if is_leaf_node(node):
        pred = int(node.prediction)
        pred_label = class_labels[pred] if pred < len(class_labels) else pred
        counts = [int(v) for v in getattr(node, "class_counts", ())]
        return {
            "node_type": "leaf",
            "n_samples": n_samples,
            "predicted_class_index": pred,
            "predicted_class_label": to_label_text(pred_label),
            "true_class_dist": _class_dist_from_counts(counts, class_labels),
            "path_conditions": path_conditions,
        }

    feature_idx = int(node.feature)
    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"x{feature_idx}"
    feature_display = feature_display_name(feature_name)
    groups = group_children(node)
    feature_values = z_train[idxs, feature_idx] if (z_train is not None and idxs is not None and idxs.size > 0) else None

    children: List[Dict[str, object]] = []
    for bins, child in groups:
        bins_arr = np.asarray(bins, dtype=np.int32)
        child_idxs = None
        if feature_values is not None:
            child_idxs = idxs[np.isin(feature_values, bins_arr)]
        cond = format_msplit_condition(feature_idx, bins, binner, feature_names)
        children.append(
            {
                "branch": {
                    "condition": cond,
                    "bins": [int(b) for b in bins],
                },
                "child": serialize_msplit_node(
                    child,
                    binner,
                    feature_names,
                    class_labels,
                    z_train,
                    child_idxs,
                    path_conditions + [f"{feature_display} {cond}"],
                ),
            }
        )

    return {
        "node_type": "internal",
        "feature_index": feature_idx,
        "feature_name": to_label_text(feature_name),
        "feature_display_name": feature_display,
        "n_samples": n_samples,
        "n_way": int(len(groups)),
        "children": children,
    }


def build_msplit_artifact(
    *,
    dataset: str,
    pipeline: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    lookahead: int,
    time_limit: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
    max_branching: int,
    reg: float,
    branch_penalty: float,
    msplit_variant: Optional[str],
    tree_root: object,
    binner,
    z_train: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    root_idxs = np.arange(z_train.shape[0], dtype=np.int32)
    bin_edges: List[Optional[List[float]]] = []
    for edges in binner.bin_edges_per_feature:
        if edges is None:
            bin_edges.append(None)
        else:
            bin_edges.append([float(v) for v in np.asarray(edges).tolist()])

    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": str(pipeline),
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "lookahead": int(lookahead),
            "time_limit": float(time_limit),
            "max_bins": int(max_bins),
            "min_samples_leaf": int(min_samples_leaf),
            "min_child_size": int(min_child_size),
            "max_branching": int(max_branching),
            "reg": float(reg),
            "branch_penalty": float(branch_penalty),
            "msplit_variant": str(msplit_variant) if msplit_variant is not None else None,
        },
        "binner": {
            "n_features": int(len(feature_names)),
            "bin_edges_per_feature": bin_edges,
        },
        "tree_artifact": {
            "tree": serialize_msplit_node(
                tree_root,
                binner,
                feature_names,
                np.asarray(class_labels, dtype=object),
                z_train=z_train,
                idxs=root_idxs,
                path_conditions=[],
            )
        },
    }


def xgb_parse_feature_idx(token: str) -> int:
    text = str(token)
    if text.startswith("f"):
        return int(text[1:])
    return int(text)


def serialize_xgb_tree(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    class_labels: np.ndarray,
) -> Dict[str, object]:
    tree_df = model.get_booster().trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == 0].copy()
    rows = {str(row["ID"]): row for _, row in tree_df.iterrows()}
    root_id = "0-0"

    children: Dict[str, Tuple[str, str]] = {}
    for node_id, row in rows.items():
        if str(row["Feature"]) == "Leaf":
            continue
        children[node_id] = (str(row["Yes"]), str(row["No"]))

    node_sample_counts: Dict[str, int] = {}
    leaf_class_counts: Dict[str, np.ndarray] = {}

    def _route_and_count(node_id: str, idxs: np.ndarray) -> None:
        node_sample_counts[node_id] = int(idxs.size)
        if node_id not in children:
            counts = np.bincount(y_train[idxs], minlength=len(class_labels)).astype(np.int64)
            leaf_class_counts[node_id] = counts
            return

        row = rows[node_id]
        yes_id, no_id = children[node_id]
        missing_id = str(row["Missing"])
        feat_idx = xgb_parse_feature_idx(str(row["Feature"]))
        thr = float(row["Split"])

        col = x_train[idxs, feat_idx]
        nan_mask = np.isnan(col)
        left_mask = (col < thr) & (~nan_mask)
        right_mask = (col >= thr) & (~nan_mask)

        left_idxs = idxs[left_mask]
        right_idxs = idxs[right_mask]
        if nan_mask.any():
            miss_idxs = idxs[nan_mask]
            if missing_id == yes_id:
                left_idxs = np.concatenate([left_idxs, miss_idxs])
            elif missing_id == no_id:
                right_idxs = np.concatenate([right_idxs, miss_idxs])

        _route_and_count(yes_id, left_idxs)
        _route_and_count(no_id, right_idxs)

    _route_and_count(root_id, np.arange(x_train.shape[0], dtype=np.int32))

    def _build_node(node_id: str, path_conditions: List[str]) -> Dict[str, object]:
        row = rows[node_id]
        if node_id not in children:
            leaf_score = float(row["Gain"])
            pred_idx = 1 if leaf_score >= 0.0 else 0
            pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else pred_idx
            counts = leaf_class_counts.get(node_id, np.zeros(len(class_labels), dtype=np.int64))
            return {
                "node_type": "leaf",
                "id": node_id,
                "n_samples": int(node_sample_counts.get(node_id, 0)),
                "cover": float(row["Cover"]),
                "leaf_score": leaf_score,
                "predicted_class_index": int(pred_idx),
                "predicted_class_label": to_label_text(pred_label),
                "true_class_dist": _class_dist_from_counts(counts.tolist(), class_labels),
                "path_conditions": path_conditions,
            }

        yes_id, no_id = children[node_id]
        feat_idx = xgb_parse_feature_idx(str(row["Feature"]))
        feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else str(row["Feature"])
        feature_disp = feature_display_name(feature_name)
        thr = float(row["Split"])
        yes_cond = f"<= {_format_float(thr)}"
        no_cond = f"> {_format_float(thr)}"
        return {
            "node_type": "internal",
            "id": node_id,
            "feature_index": int(feat_idx),
            "feature_name": to_label_text(feature_name),
            "feature_display_name": feature_disp,
            "n_samples": int(node_sample_counts.get(node_id, int(round(float(row["Cover"]))))),
            "cover": float(row["Cover"]),
            "threshold": thr,
            "n_way": 2,
            "missing_goes_to": str(row["Missing"]),
            "children": [
                {
                    "branch": {"condition": yes_cond},
                    "child": _build_node(yes_id, path_conditions + [f"{feature_disp} {yes_cond}"]),
                },
                {
                    "branch": {"condition": no_cond},
                    "child": _build_node(no_id, path_conditions + [f"{feature_disp} {no_cond}"]),
                },
            ],
        }

    return {
        "tree_index": 0,
        "root_id": root_id,
        "tree": _build_node(root_id, []),
    }


def build_xgb_artifact(
    *,
    dataset: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    n_estimators: int,
    learning_rate: float,
    num_threads: int,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": "xgboost",
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_threads": int(num_threads),
        },
        "tree_artifact": serialize_xgb_tree(
            model,
            x_train,
            y_train,
            feature_names,
            class_labels=np.asarray(class_labels, dtype=object),
        ),
    }
