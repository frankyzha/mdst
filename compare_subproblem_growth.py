#!/usr/bin/env python3
"""Compare MSPLIT subproblem growth across experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_KEYS = [
    "msplit_variant",
    "lookahead_depth_budget",
    "used_max_bins",
    "used_min_samples_leaf",
    "used_min_child_size",
    "used_max_branching",
    "used_reg",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare subproblem counts and growth rates across seed_results.csv files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed-csv",
        nargs="+",
        required=True,
        help="One or more seed_results.csv files from multisplit runs.",
    )
    parser.add_argument(
        "--label",
        nargs="+",
        default=None,
        help="Optional labels (same count as --seed-csv). Defaults to run directory name.",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "dp_subproblem_calls",
            "dp_unique_states",
            "greedy_subproblem_calls",
            "greedy_unique_states",
            "exact_internal_nodes",
            "greedy_internal_nodes",
        ],
        default="dp_subproblem_calls",
        help="Primary metric for growth-rate columns.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=DEFAULT_CONFIG_KEYS,
        help="Columns that define a config when comparing growth.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional path to write the detailed comparison CSV.",
    )
    parser.add_argument(
        "--only-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only successful trials (status=ok).",
    )
    return parser.parse_args()


def _auto_label(path: Path) -> str:
    parent = path.parent
    return parent.name if parent.name else path.stem


def _sanitize_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    return out


def main() -> None:
    args = _parse_args()
    csv_paths = [Path(p).expanduser().resolve() for p in args.seed_csv]

    if args.label is not None and len(args.label) != len(csv_paths):
        raise ValueError("--label count must match --seed-csv count")

    labels = list(args.label) if args.label is not None else [_auto_label(p) for p in csv_paths]

    frames: list[pd.DataFrame] = []
    needed = [
        "dataset",
        "depth_budget",
        "status",
        "lookahead_depth_budget",
        "msplit_variant",
        "used_max_bins",
        "used_min_samples_leaf",
        "used_min_child_size",
        "used_max_branching",
        "used_reg",
        "dp_subproblem_calls",
        "dp_cache_hits",
        "dp_unique_states",
        "greedy_subproblem_calls",
        "greedy_cache_hits",
        "greedy_unique_states",
        "exact_internal_nodes",
        "greedy_internal_nodes",
    ]

    for label, path in zip(labels, csv_paths):
        if not path.exists():
            raise FileNotFoundError(f"missing CSV: {path}")
        df = pd.read_csv(path)
        df = _sanitize_columns(df, needed)
        if args.only_ok:
            df = df[df["status"] == "ok"].copy()
        df["config_label"] = label
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    group_keys = ["config_label", "dataset", "depth_budget", *args.group_by]

    agg = (
        all_df.groupby(group_keys, dropna=False)
        .agg(
            n_trials=("depth_budget", "size"),
            mean_dp_subproblem_calls=("dp_subproblem_calls", "mean"),
            mean_dp_cache_hits=("dp_cache_hits", "mean"),
            mean_dp_unique_states=("dp_unique_states", "mean"),
            mean_greedy_subproblem_calls=("greedy_subproblem_calls", "mean"),
            mean_greedy_cache_hits=("greedy_cache_hits", "mean"),
            mean_greedy_unique_states=("greedy_unique_states", "mean"),
            mean_exact_internal_nodes=("exact_internal_nodes", "mean"),
            mean_greedy_internal_nodes=("greedy_internal_nodes", "mean"),
        )
        .reset_index()
        .sort_values(["config_label", "dataset", "depth_budget"])
    )

    metric_col = f"mean_{args.metric}"
    prev_col = f"{metric_col}_prev_depth"
    ratio_col = f"{metric_col}_growth_ratio"
    delta_col = f"{metric_col}_growth_delta"

    by_config_no_depth = [k for k in group_keys if k != "depth_budget"]
    agg[prev_col] = agg.groupby(by_config_no_depth, dropna=False)[metric_col].shift(1)
    agg[ratio_col] = agg[metric_col] / agg[prev_col]
    agg[delta_col] = agg[metric_col] - agg[prev_col]

    print("=== Subproblem Growth Comparison ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
