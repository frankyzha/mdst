#!/usr/bin/env python3
"""Exhaustively validate the local exact teacher_guided_atomcolor solver on small cases."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SPLIT_SRC = REPO_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

from split import _libgosdt  # type: ignore


EPS = 1e-8


@dataclass(frozen=True)
class Case:
    row_counts: tuple[int, ...]
    pos_w: tuple[float, ...]
    neg_w: tuple[float, ...]
    groups: int
    regularization: float
    min_child_size: int


def canonicalize_assignment(raw: tuple[int, ...], groups: int) -> tuple[int, ...] | None:
    remap = [-1] * groups
    next_group = 0
    out: list[int] = []
    for child in raw:
        if child < 0 or child >= groups:
            return None
        mapped = remap[child]
        if mapped < 0:
            mapped = next_group
            remap[child] = mapped
            next_group += 1
        out.append(mapped)
    if next_group != groups:
        return None
    return tuple(out)


def assignment_to_group_spans(assignment: tuple[int, ...], groups: int) -> list[list[list[int]]]:
    spans: list[list[list[int]]] = [[] for _ in range(groups)]
    for idx, child in enumerate(assignment):
        group = spans[child]
        if group and group[-1][1] + 1 >= idx:
            group[-1][1] = idx
        else:
            group.append([idx, idx])
    return spans


def score_assignment(case: Case, assignment: tuple[int, ...]) -> tuple[bool, float, float]:
    row_counts = [0] * case.groups
    pos = [0.0] * case.groups
    neg = [0.0] * case.groups
    for idx, child in enumerate(assignment):
        row_counts[child] += case.row_counts[idx]
        pos[child] += case.pos_w[idx]
        neg[child] += case.neg_w[idx]
    if any(count < case.min_child_size for count in row_counts):
        return False, math.inf, math.inf
    leaf_cost = case.groups * case.regularization
    imp_cost = 0.0
    for p, n in zip(pos, neg):
        leaf_cost += min(p, n)
        total = p + n
        if total > EPS:
            imp_cost += total - (p * p + n * n) / total
    return True, leaf_cost, imp_cost


def candidate_better(
    lhs: tuple[float, float, tuple[int, ...]] | None,
    rhs: tuple[float, float, tuple[int, ...]] | None,
) -> bool:
    if lhs is None:
        return False
    if rhs is None:
        return True
    if lhs[0] < rhs[0] - EPS:
        return True
    if rhs[0] < lhs[0] - EPS:
        return False
    if lhs[1] < rhs[1] - EPS:
        return True
    if rhs[1] < lhs[1] - EPS:
        return False
    return lhs[2] < rhs[2]


def brute_force_best(case: Case) -> dict[str, Any]:
    best: tuple[float, float, tuple[int, ...]] | None = None
    seen: set[tuple[int, ...]] = set()
    for raw in itertools.product(range(case.groups), repeat=len(case.row_counts)):
        assignment = canonicalize_assignment(raw, case.groups)
        if assignment is None or assignment in seen:
            continue
        seen.add(assignment)
        feasible, leaf_cost, imp_cost = score_assignment(case, assignment)
        if not feasible:
            continue
        candidate = (leaf_cost, imp_cost, assignment)
        if candidate_better(candidate, best):
            best = candidate
    if best is None:
        return {
            "has_feasible": False,
            "best_leaf_cost": math.inf,
            "best_imp_cost": math.inf,
            "best_assignment": [],
            "best_group_spans": [],
        }
    _, _, assignment = best
    return {
        "has_feasible": True,
        "best_leaf_cost": best[0],
        "best_imp_cost": best[1],
        "best_assignment": list(assignment),
        "best_group_spans": assignment_to_group_spans(assignment, case.groups),
    }


def random_case(rng: np.random.Generator, max_atoms: int, weighted: bool) -> Case:
    atom_count = int(rng.integers(2, max_atoms + 1))
    groups = int(rng.integers(2, min(3, atom_count) + 1))
    min_child_size = int(rng.integers(1, 5))
    row_counts = rng.integers(1, 7, size=atom_count, dtype=np.int32)
    total_weight = row_counts.astype(np.float64)
    if weighted:
        total_weight = total_weight * rng.uniform(0.2, 2.5, size=atom_count)
    pos_prob = np.clip(0.5 + rng.normal(0.0, 0.22, size=atom_count), 0.02, 0.98)
    pos_w = total_weight * pos_prob
    neg_w = total_weight - pos_w
    regularization = float(rng.uniform(0.0, 0.5))
    return Case(
        row_counts=tuple(int(v) for v in row_counts.tolist()),
        pos_w=tuple(float(v) for v in pos_w.tolist()),
        neg_w=tuple(float(v) for v in neg_w.tolist()),
        groups=groups,
        regularization=regularization,
        min_child_size=min_child_size,
    )


def run_cpp(case: Case) -> dict[str, Any]:
    raw = _libgosdt.msplit_debug_teacher_guided_atomcolor_exact_bnb_synthetic(
        list(case.row_counts),
        list(case.pos_w),
        list(case.neg_w),
        int(case.groups),
        float(case.regularization),
        int(case.min_child_size),
    )
    return json.loads(raw)


def approx_equal(lhs: float, rhs: float) -> bool:
    if math.isinf(lhs) or math.isinf(rhs):
        return math.isinf(lhs) and math.isinf(rhs)
    return abs(lhs - rhs) <= EPS


def validate_case(case: Case) -> tuple[bool, str]:
    brute = brute_force_best(case)
    cpp = run_cpp(case)
    if bool(cpp["has_feasible"]) != bool(brute["has_feasible"]):
        return False, "has_feasible mismatch"
    if not brute["has_feasible"]:
        return True, ""
    if not approx_equal(float(cpp["best_leaf_cost"]), float(brute["best_leaf_cost"])):
        return False, "best_leaf_cost mismatch"
    if not approx_equal(float(cpp["best_imp_cost"]), float(brute["best_imp_cost"])):
        return False, "best_imp_cost mismatch"
    if list(cpp["best_assignment"]) != list(brute["best_assignment"]):
        return False, "best_assignment mismatch"
    if cpp["best_group_spans"] != brute["best_group_spans"]:
        return False, "best_group_spans mismatch"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unweighted-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checked = 0
    weighted_modes = [False] if args.unweighted_only else [False, True]
    for weighted in weighted_modes:
        for _ in range(args.cases):
            case = random_case(rng, args.max_atoms, weighted)
            ok, reason = validate_case(case)
            checked += 1
            if not ok:
                print("Validation failed")
                print("reason:", reason)
                print("case:", case)
                print("cpp:", run_cpp(case))
                print("brute:", brute_force_best(case))
                return 1
    print(f"Validated {checked} cases with exhaustive brute force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
