#!/usr/bin/env python3
"""Benchmark a minimal weighted primary-DP prototype against the current local atom-color solver."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SPLIT_SRC = REPO_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

from split import _libgosdt  # type: ignore


EPS = 1e-9


@dataclass(frozen=True)
class AtomCase:
    row_counts: tuple[int, ...]
    pos_w: tuple[float, ...]
    neg_w: tuple[float, ...]
    groups: int
    regularization: float
    min_child_size: int
    margin_mode: str
    weighted: bool


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def probability_draws(rng: np.random.Generator, atom_count: int, mode: str) -> np.ndarray:
    if mode == "strong":
        return rng.beta(0.25, 0.25, size=atom_count)
    if mode == "ambiguous":
        return np.clip(0.5 + rng.normal(0.0, 0.07, size=atom_count), 0.02, 0.98)
    if mode == "mixed":
        hard = rng.beta(0.35, 0.35, size=atom_count)
        soft = np.clip(0.5 + rng.normal(0.0, 0.14, size=atom_count), 0.02, 0.98)
        mask = rng.random(atom_count) < 0.55
        return np.where(mask, hard, soft)
    raise ValueError(f"Unsupported margin mode: {mode}")


def generate_case(
    rng: np.random.Generator,
    atom_count: int,
    groups: int,
    min_child_size: int,
    regularization: float,
    weighted: bool,
    margin_mode: str,
) -> AtomCase:
    while True:
        row_counts = rng.integers(1, 2 * max(2, min_child_size) + 1, size=atom_count, dtype=np.int32)
        if int(np.sum(row_counts)) >= groups * min_child_size:
            break

    total_weight = row_counts.astype(np.float64)
    if weighted:
        total_weight = total_weight * rng.uniform(0.35, 2.1, size=atom_count)

    pos_prob = probability_draws(rng, atom_count, margin_mode)
    pos_w = total_weight * pos_prob
    neg_w = total_weight - pos_w
    return AtomCase(
        row_counts=tuple(int(v) for v in row_counts.tolist()),
        pos_w=tuple(float(v) for v in pos_w.tolist()),
        neg_w=tuple(float(v) for v in neg_w.tolist()),
        groups=int(groups),
        regularization=float(regularization),
        min_child_size=int(min_child_size),
        margin_mode=margin_mode,
        weighted=bool(weighted),
    )


def initial_histogram_state(positive_groups: int, negative_groups: int, min_child_size: int) -> tuple[int, ...]:
    plus = [0] * (min_child_size + 1)
    minus = [0] * (min_child_size + 1)
    plus[0] = positive_groups
    minus[0] = negative_groups
    return tuple(plus + minus)


def goal_histogram_state(positive_groups: int, negative_groups: int, min_child_size: int) -> tuple[int, ...]:
    plus = [0] * (min_child_size + 1)
    minus = [0] * (min_child_size + 1)
    plus[min_child_size] = positive_groups
    minus[min_child_size] = negative_groups
    return tuple(plus + minus)


def run_weighted_primary_dp(case: AtomCase) -> dict[str, Any]:
    supports = case.row_counts
    signed_margin = tuple(pos - neg for pos, neg in zip(case.pos_w, case.neg_w))
    total_weight = float(sum(pos + neg for pos, neg in zip(case.pos_w, case.neg_w)))
    m = case.min_child_size
    groups = case.groups

    best_reward = -math.inf
    best_template = None
    template_results: list[dict[str, Any]] = []
    total_frontier_states = 0
    total_transitions = 0
    max_frontier = 0

    for positive_groups in range(groups + 1):
        negative_groups = groups - positive_groups
        dp: dict[tuple[int, ...], float] = {
            initial_histogram_state(positive_groups, negative_groups, m): 0.0
        }

        for support, margin in zip(supports, signed_margin):
            next_dp: dict[tuple[int, ...], float] = {}
            for state, reward in dp.items():
                plus = state[: m + 1]
                minus = state[m + 1 :]

                for level, count in enumerate(plus):
                    if count <= 0:
                        continue
                    next_level = min(m, level + support)
                    plus_next = list(plus)
                    plus_next[level] -= 1
                    plus_next[next_level] += 1
                    next_state = tuple(plus_next + list(minus))
                    candidate = reward + margin
                    if candidate > next_dp.get(next_state, -math.inf):
                        next_dp[next_state] = candidate
                    total_transitions += 1

                for level, count in enumerate(minus):
                    if count <= 0:
                        continue
                    next_level = min(m, level + support)
                    minus_next = list(minus)
                    minus_next[level] -= 1
                    minus_next[next_level] += 1
                    next_state = tuple(list(plus) + minus_next)
                    candidate = reward - margin
                    if candidate > next_dp.get(next_state, -math.inf):
                        next_dp[next_state] = candidate
                    total_transitions += 1

            dp = next_dp
            total_frontier_states += len(dp)
            max_frontier = max(max_frontier, len(dp))

        goal = goal_histogram_state(positive_groups, negative_groups, m)
        reward = dp.get(goal, -math.inf)
        template_results.append(
            {
                "positive_groups": positive_groups,
                "negative_groups": negative_groups,
                "reward": reward,
                "reachable_final_states": len(dp),
            }
        )
        if reward > best_reward:
            best_reward = reward
            best_template = positive_groups

    leaf_cost = math.inf
    if math.isfinite(best_reward):
        leaf_cost = groups * case.regularization + 0.5 * (total_weight - best_reward)

    return {
        "leaf_cost": leaf_cost,
        "best_reward": best_reward,
        "best_template_positive_groups": best_template,
        "total_frontier_states": total_frontier_states,
        "total_transitions": total_transitions,
        "max_frontier": max_frontier,
        "template_results": template_results,
    }


def run_current_cpp_solver(case: AtomCase) -> dict[str, Any]:
    raw = _libgosdt.msplit_debug_teacher_guided_atomcolor_exact_bnb_synthetic(
        list(case.row_counts),
        list(case.pos_w),
        list(case.neg_w),
        int(case.groups),
        float(case.regularization),
        int(case.min_child_size),
    )
    return json.loads(raw)


def timed_call(fn, repeats: int) -> tuple[Any, float]:
    best_result = None
    elapsed: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        best_result = fn()
        elapsed.append(time.perf_counter() - started)
    return best_result, min(elapsed) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


def benchmark_setting(
    rng: np.random.Generator,
    atom_count: int,
    groups: int,
    trials: int,
    repeats: int,
    min_child_size: int,
    regularization: float,
    weighted: bool,
    margin_mode: str,
) -> dict[str, Any]:
    current_ms: list[float] = []
    dp_ms: list[float] = []
    current_dfs_calls: list[float] = []
    dp_frontier_states: list[float] = []
    dp_transitions: list[float] = []
    mismatches: list[dict[str, Any]] = []

    for trial_idx in range(trials):
        case = generate_case(
            rng=rng,
            atom_count=atom_count,
            groups=groups,
            min_child_size=min_child_size,
            regularization=regularization,
            weighted=weighted,
            margin_mode=margin_mode,
        )

        current_result, current_time_ms = timed_call(lambda: run_current_cpp_solver(case), repeats=repeats)
        dp_result, dp_time_ms = timed_call(lambda: run_weighted_primary_dp(case), repeats=repeats)

        current_leaf = (
            float(current_result["best_leaf_cost"]) if bool(current_result["has_feasible"]) else math.inf
        )
        dp_leaf = float(dp_result["leaf_cost"])
        if not math.isclose(current_leaf, dp_leaf, rel_tol=1e-8, abs_tol=1e-8):
            mismatches.append(
                {
                    "trial": trial_idx,
                    "current_leaf_cost": current_leaf,
                    "dp_leaf_cost": dp_leaf,
                    "row_counts": case.row_counts,
                    "pos_w": case.pos_w,
                    "neg_w": case.neg_w,
                    "cpp_debug": current_result,
                    "dp_debug": dp_result,
                }
            )

        current_ms.append(current_time_ms)
        dp_ms.append(dp_time_ms)
        current_dfs_calls.append(float(current_result.get("dfs_calls", 0)))
        dp_frontier_states.append(float(dp_result["total_frontier_states"]))
        dp_transitions.append(float(dp_result["total_transitions"]))

    return {
        "atom_count": atom_count,
        "groups": groups,
        "trials": trials,
        "repeats": repeats,
        "margin_mode": margin_mode,
        "weighted": weighted,
        "current_time_ms": summarize(current_ms),
        "dp_time_ms": summarize(dp_ms),
        "current_dfs_calls": summarize(current_dfs_calls),
        "dp_frontier_states": summarize(dp_frontier_states),
        "dp_transitions": summarize(dp_transitions),
        "speedup_current_over_dp_mean": (
            float(statistics.fmean(current_ms) / statistics.fmean(dp_ms))
            if statistics.fmean(dp_ms) > 0.0
            else math.inf
        ),
        "speedup_current_over_dp_median": (
            float(statistics.median(current_ms) / statistics.median(dp_ms))
            if statistics.median(dp_ms) > 0.0
            else math.inf
        ),
        "mismatches": mismatches,
    }


def print_summary(result: dict[str, Any]) -> None:
    label = (
        f"A={result['atom_count']:>2} "
        f"k={result['groups']} "
        f"mode={result['margin_mode']:<10} "
        f"weighted={int(result['weighted'])}"
    )
    current_time = result["current_time_ms"]
    dp_time = result["dp_time_ms"]
    print(label, flush=True)
    print(
        "  time_ms"
        f" current(mean/med)={format_float(current_time['mean'])}/{format_float(current_time['median'])}"
        f" dp(mean/med)={format_float(dp_time['mean'])}/{format_float(dp_time['median'])}"
        f" ratio(current/dp mean)={format_float(result['speedup_current_over_dp_mean'])}"
    , flush=True)
    print(
        "  work"
        f" current_dfs_median={format_float(result['current_dfs_calls']['median'])}"
        f" dp_frontier_median={format_float(result['dp_frontier_states']['median'])}"
        f" dp_transitions_median={format_float(result['dp_transitions']['median'])}"
    , flush=True)
    mismatch_count = len(result["mismatches"])
    print(f"  objective_mismatches={mismatch_count}/{result['trials']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a minimal weighted primary-DP prototype against the current local atom-color solver."
    )
    parser.add_argument("--atom-counts", default="8,12,16,20,24")
    parser.add_argument("--groups", default="2,3")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-child-size", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin-mode", choices=["strong", "mixed", "ambiguous"], default="mixed")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    atom_counts = parse_int_list(args.atom_counts)
    groups_list = parse_int_list(args.groups)
    rng = np.random.default_rng(args.seed)

    all_results: list[dict[str, Any]] = []
    for groups in groups_list:
        for atom_count in atom_counts:
            if atom_count < groups:
                continue
            result = benchmark_setting(
                rng=rng,
                atom_count=atom_count,
                groups=groups,
                trials=args.trials,
                repeats=args.repeats,
                min_child_size=args.min_child_size,
                regularization=args.regularization,
                weighted=args.weighted,
                margin_mode=args.margin_mode,
            )
            print_summary(result)
            all_results.append(result)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote raw benchmark results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
