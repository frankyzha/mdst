#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_teacher_guided_atomcolor_cached import load_cache, load_local_libgosdt


def _route_row(tree: dict[str, object], row: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    node = tree
    path: list[dict[str, object]] = []
    while node.get("type") == "node":
        feature = int(node["feature"])
        bin_id = int(row[feature])
        chosen_group = -1
        child = None
        chosen_spans: list[list[int]] = []
        nearest_group_idx = -1
        nearest_spans: list[list[int]] = []
        best_dist = None
        best_lo = None
        for group_idx, group in enumerate(node.get("groups", [])):
            spans = [[int(lo), int(hi)] for lo, hi in group.get("spans", [])]
            matched = any(lo <= bin_id <= hi for lo, hi in spans)
            if matched:
                chosen_group = int(group_idx)
                chosen_spans = spans
                child = group["child"]
                break
            if not spans:
                continue
            group_lo = min(lo for lo, _ in spans)
            group_dist = min(0 if lo <= bin_id <= hi else min(abs(bin_id - lo), abs(bin_id - hi)) for lo, hi in spans)
            if best_dist is None or group_dist < best_dist or (
                group_dist == best_dist and (best_lo is None or group_lo < best_lo)
            ):
                best_dist = group_dist
                best_lo = group_lo
                nearest_group_idx = int(group_idx)
                nearest_spans = spans
        path.append(
            {
                "feature": feature,
                "bin_id": bin_id,
                "group_idx": chosen_group,
                "spans": chosen_spans,
                "fallback_prediction": int(node.get("fallback_prediction", 0)),
            }
        )
        if child is None:
            if nearest_group_idx >= 0:
                path[-1]["group_idx"] = nearest_group_idx
                path[-1]["spans"] = nearest_spans
                child = node.get("groups", [])[nearest_group_idx]["child"]
            else:
                return int(node.get("fallback_prediction", 0)), path
        node = child
    return int(node.get("prediction", 0)), path


def _leaf_key(path: list[dict[str, object]], prediction: int) -> str:
    if not path:
        return f"leaf:root:pred={prediction}"
    parts = []
    for step in path:
        parts.append(f"f{step['feature']}=g{step['group_idx']}")
    return "|".join(parts) + f"|pred={prediction}"


def _path_signature(path: list[dict[str, object]]) -> str:
    if not path:
        return "root"
    parts = []
    for step in path:
        span_text = ",".join(f"{lo}-{hi}" for lo, hi in step["spans"])
        parts.append(f"f{step['feature']}=g{step['group_idx']}[{span_text}]")
    return "|".join(parts)


def _tree_depth(tree: dict[str, object]) -> int:
    if tree.get("type") != "node":
        return 0
    return 1 + max(_tree_depth(group["child"]) for group in tree.get("groups", []))


def _collect_feature_usage(tree: dict[str, object], depth: int = 0, counts: Counter | None = None,
                           depth_counts: dict[int, Counter] | None = None,
                           repeated_path_counts: Counter | None = None,
                           ancestor_features: tuple[int, ...] = (),
                           repeated_nodes: list[dict[str, object]] | None = None,
                           path_desc: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...] = (),
                           ) -> tuple[Counter, dict[int, Counter], Counter, list[dict[str, object]]]:
    if counts is None:
        counts = Counter()
    if depth_counts is None:
        depth_counts = defaultdict(Counter)
    if repeated_path_counts is None:
        repeated_path_counts = Counter()
    if repeated_nodes is None:
        repeated_nodes = []
    if tree.get("type") != "node":
        return counts, depth_counts, repeated_path_counts, repeated_nodes
    feature = int(tree["feature"])
    counts[feature] += 1
    depth_counts[depth][feature] += 1
    if feature in ancestor_features:
        repeated_path_counts[feature] += 1
        repeated_nodes.append(
            {
                "feature": feature,
                "depth": int(depth),
                "n_samples": int(tree.get("n_samples", 0)),
                "group_count": int(len(tree.get("groups", []))),
                "fallback_bin": int(tree.get("fallback_bin", -1)),
                "fallback_prediction": int(tree.get("fallback_prediction", 0)),
                "path_signature": "|".join(
                    f"f{feat}=g{group_idx}[{','.join(f'{lo}-{hi}' for lo, hi in spans)}]"
                    for feat, group_idx, spans in path_desc
                ),
            }
        )
    next_ancestors = ancestor_features + (feature,)
    for group_id, group in enumerate(tree.get("groups", [])):
        spans = tuple((int(lo), int(hi)) for lo, hi in group.get("spans", []))
        _collect_feature_usage(
            group["child"],
            depth=depth + 1,
            counts=counts,
            depth_counts=depth_counts,
            repeated_path_counts=repeated_path_counts,
            ancestor_features=next_ancestors,
            repeated_nodes=repeated_nodes,
            path_desc=path_desc + ((feature, int(group_id), spans),),
        )
    return counts, depth_counts, repeated_path_counts, repeated_nodes


def _analyze_split(tree: dict[str, object], z: np.ndarray, y: np.ndarray, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    leaf_stats: dict[str, dict[str, object]] = {}
    node_stats: dict[str, dict[str, object]] = {}
    for idx in range(z.shape[0]):
        pred, path = _route_row(tree, z[idx])
        leaf_id = _leaf_key(path, pred)
        correct = int(pred == int(y[idx]))
        entry = leaf_stats.setdefault(
            leaf_id,
            {
                "leaf_id": leaf_id,
                "prediction": int(pred),
                "path_length": len(path),
                "path": _path_signature(path),
                "support": 0,
                "correct": 0,
                "errors": 0,
                f"{split_name}_y0": 0,
                f"{split_name}_y1": 0,
            },
        )
        entry["support"] += 1
        entry["correct"] += correct
        entry["errors"] += 1 - correct
        entry[f"{split_name}_y{int(y[idx])}"] += 1

        prefix: list[str] = []
        for depth, step in enumerate(path):
            prefix.append(f"f{step['feature']}=g{step['group_idx']}")
            node_id = "|".join(prefix)
            node_entry = node_stats.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "depth": depth,
                    "feature": int(step["feature"]),
                    "group_idx": int(step["group_idx"]),
                    "support": 0,
                },
            )
            node_entry["support"] += 1

    leaf_df = pd.DataFrame(leaf_stats.values())
    if not leaf_df.empty:
        leaf_df[f"{split_name}_support"] = leaf_df["support"].astype(int)
        leaf_df[f"{split_name}_accuracy"] = leaf_df["correct"] / leaf_df["support"]
        leaf_df[f"{split_name}_error_rate"] = leaf_df["errors"] / leaf_df["support"]
        leaf_df = leaf_df.drop(columns=["support", "correct", "errors"])
    node_df = pd.DataFrame(node_stats.values())
    if not node_df.empty:
        node_df[f"{split_name}_support"] = node_df["support"].astype(int)
        node_df = node_df.drop(columns=["support"])
    return leaf_df, node_df


def _merge_leaf_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_df.merge(test_df, on=["leaf_id", "prediction", "path_length", "path"], how="outer")
    for col in ["train_support", "test_support"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in ["train_accuracy", "test_accuracy", "train_error_rate", "test_error_rate"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(np.nan)
    merged["support_gap"] = merged["train_support"] - merged["test_support"]
    merged["accuracy_gap"] = merged["train_accuracy"] - merged["test_accuracy"]
    merged["test_error_contribution"] = merged["test_support"] * merged["test_error_rate"].fillna(0.0)
    return merged.sort_values(
        ["test_error_contribution", "train_support", "path_length"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _merge_node_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_df.merge(test_df, on=["node_id", "depth", "feature", "group_idx"], how="outer")
    for col in ["train_support", "test_support"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    merged["coverage_ratio_test_over_train"] = np.where(
        merged["train_support"] > 0,
        merged["test_support"] / merged["train_support"],
        np.nan,
    )
    return merged.sort_values(["depth", "train_support"], ascending=[True, False]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MSPLIT generalization gap on a cached benchmark split.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--min-split-size", type=int, required=True)
    parser.add_argument("--min-child-size", type=int, required=True)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cache = load_cache(args.cache_path)
    z_fit = np.asarray(cache["Z_fit"], dtype=np.int32)
    z_test = np.asarray(cache["Z_test"], dtype=np.int32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    y_test = np.asarray(cache["y_test"], dtype=np.int32)
    sample_weight = np.full(z_fit.shape[0], 1.0 / float(max(1, z_fit.shape[0])), dtype=np.float64)

    libgosdt = load_local_libgosdt()
    started = time.perf_counter()
    cpp_result = libgosdt.msplit_fit(
        z_fit,
        y_fit,
        sample_weight,
        np.asarray(cache["teacher_logit"], dtype=np.float64),
        np.asarray(cache["teacher_boundary_gain"], dtype=np.float64),
        np.asarray(cache["teacher_boundary_cover"], dtype=np.float64),
        np.asarray(cache["teacher_boundary_value_jump"], dtype=np.float64),
        int(args.depth),
        int(args.lookahead_depth),
        float(args.reg),
        int(args.min_split_size),
        int(args.min_child_size),
        28800.0,
        int(args.max_branching),
    )
    fit_seconds = time.perf_counter() - started
    tree = json.loads(str(cpp_result["tree"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")

    train_leaf_df, train_node_df = _analyze_split(tree, z_fit, y_fit, "train")
    test_leaf_df, test_node_df = _analyze_split(tree, z_test, y_test, "test")
    leaf_df = _merge_leaf_stats(train_leaf_df, test_leaf_df)
    node_df = _merge_node_stats(train_node_df, test_node_df)
    leaf_df.to_csv(args.out_dir / "leaf_diagnostics.csv", index=False)
    node_df.to_csv(args.out_dir / "node_coverage.csv", index=False)

    feature_counts, depth_feature_counts, repeated_feature_counts, repeated_nodes = _collect_feature_usage(tree)
    feature_rows = []
    for feature, count in sorted(feature_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        row = {
            "feature": int(feature),
            "count": int(count),
            "repeated_on_path_count": int(repeated_feature_counts.get(feature, 0)),
        }
        for depth, counter in sorted(depth_feature_counts.items()):
            row[f"depth_{depth}_count"] = int(counter.get(feature, 0))
        feature_rows.append(row)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(args.out_dir / "feature_usage.csv", index=False)
    repeated_nodes_df = pd.DataFrame(repeated_nodes)
    repeated_nodes_df.to_csv(args.out_dir / "repeated_feature_nodes.csv", index=False)
    unstable_leaves_df = leaf_df[
        (leaf_df["train_support"] > 0) & (leaf_df["test_support"] > 0)
    ].sort_values(["accuracy_gap", "test_error_contribution"], ascending=[False, False]).reset_index(drop=True)
    unstable_leaves_df.to_csv(args.out_dir / "worst_gap_leaves.csv", index=False)

    pred_train = np.asarray([_route_row(tree, row)[0] for row in z_fit], dtype=np.int32)
    pred_test = np.asarray([_route_row(tree, row)[0] for row in z_test], dtype=np.int32)
    train_accuracy = float(np.mean(pred_train == y_fit))
    test_accuracy = float(np.mean(pred_test == y_test))

    summary = {
        "fit_seconds": float(fit_seconds),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "generalization_gap": train_accuracy - test_accuracy,
        "tree_depth": int(_tree_depth(tree)),
        "n_structural_leaves_observed": int(leaf_df.shape[0]),
        "n_train_supported_leaves": int((leaf_df["train_support"] > 0).sum()),
        "n_test_supported_leaves": int((leaf_df["test_support"] > 0).sum()),
        "n_shared_leaves": int(((leaf_df["train_support"] > 0) & (leaf_df["test_support"] > 0)).sum()),
        "n_test_only_leaves": int(((leaf_df["train_support"] == 0) & (leaf_df["test_support"] > 0)).sum()),
        "n_train_only_leaves": int(((leaf_df["test_support"] == 0) & (leaf_df["train_support"] > 0)).sum()),
        "n_internal": int(sum(feature_counts.values())),
        "n_train_leaves_with_lt_50_support": int(((leaf_df["train_support"] > 0) & (leaf_df["train_support"] < 50)).sum()),
        "n_train_leaves_with_lt_20_support": int(((leaf_df["train_support"] > 0) & (leaf_df["train_support"] < 20)).sum()),
        "test_error_concentrated_top_5_leaves": float(leaf_df["test_error_contribution"].head(5).sum()),
        "median_train_leaf_support": float(leaf_df.loc[leaf_df["train_support"] > 0, "train_support"].median()),
        "median_test_leaf_support": float(leaf_df.loc[leaf_df["test_support"] > 0, "test_support"].median()),
        "features_reused_on_paths": {str(k): int(v) for k, v in repeated_feature_counts.items()},
        "n_repeated_feature_nodes": int(len(repeated_nodes)),
        "cpp_metrics": {
            "greedy_subproblem_calls": int(cpp_result.get("greedy_subproblem_calls", 0)),
            "greedy_cache_hits": int(cpp_result.get("greedy_cache_hits", 0)),
            "greedy_unique_states": int(cpp_result.get("greedy_unique_states", 0)),
            "atomized_final_candidates": int(cpp_result.get("atomized_final_candidates", 0)),
            "debr_refine_calls": int(cpp_result.get("debr_refine_calls", 0)),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nTop error-contributing leaves:")
    cols = [
        "leaf_id",
        "prediction",
        "path_length",
        "train_support",
        "test_support",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
        "test_error_contribution",
    ]
    print(leaf_df[cols].head(10).to_string(index=False))
    print("\nFeature usage:")
    if not feature_df.empty:
        print(feature_df.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
