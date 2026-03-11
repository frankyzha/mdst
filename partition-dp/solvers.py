"""Core partition-DP solvers for binned binary labels."""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

INF = float("inf")


@dataclass(frozen=True)
class CaseInput:
    pos_bins: List[int]
    neg_bins: List[int]
    m: int
    lam: float
    ub_best: float
    max_branch_n: int


@dataclass(frozen=True)
class PreparedCase:
    inp: CaseInput
    B: int
    p: List[int]
    n: List[int]
    s: List[int]
    epb_suffix: List[float]
    ub_local: float
    k_max: int
    M: int


@dataclass(frozen=True)
class SolverResult:
    status: str
    raw_best_obj: float
    final_obj: float
    k: Optional[int]
    intervals: List[Tuple[int, int]]


def build_prefix(pos_bins: Sequence[int], neg_bins: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    B = len(pos_bins)
    p = [0] * (B + 1)
    n = [0] * (B + 1)
    s = [0] * (B + 1)
    for j in range(1, B + 1):
        p[j] = p[j - 1] + int(pos_bins[j - 1])
        n[j] = n[j - 1] + int(neg_bins[j - 1])
        s[j] = p[j] + n[j]
    return p, n, s


def interval_error(p: Sequence[int], n: Sequence[int], u: int, v: int) -> float:
    return float(min(p[v] - p[u - 1], n[v] - n[u - 1]))


def prepare_case(inp: CaseInput) -> PreparedCase:
    if len(inp.pos_bins) != len(inp.neg_bins):
        raise ValueError("pos_bins and neg_bins must have same length")
    if not inp.pos_bins:
        raise ValueError("B must be >= 1")
    if any(int(v) < 0 for v in inp.pos_bins) or any(int(v) < 0 for v in inp.neg_bins):
        raise ValueError("bin counts must be non-negative")
    if int(inp.m) < 1:
        raise ValueError("m must be >= 1")
    if float(inp.lam) <= 0:
        raise ValueError("lambda must be > 0")
    if int(inp.max_branch_n) < 1:
        raise ValueError("max_branch_n must be >= 1")

    B = len(inp.pos_bins)
    p, n, s = build_prefix(inp.pos_bins, inp.neg_bins)

    epb_suffix = [0.0] * (B + 2)
    for j in range(B, 0, -1):
        epb_suffix[j] = epb_suffix[j + 1] + float(min(int(inp.pos_bins[j - 1]), int(inp.neg_bins[j - 1])))

    ub_baseline = interval_error(p, n, 1, B) + float(inp.lam)
    ub_local = min(float(inp.ub_best), ub_baseline)

    if math.isinf(ub_local):
        k_max = B
    else:
        # K_max = max(0, ceil((UB_local - EPB(1,B))/lambda) - 1)
        k_max = max(0, int(math.ceil((ub_local - epb_suffix[1]) / float(inp.lam)) - 1))

    M = max(0, min(int(k_max), B, int(inp.max_branch_n)))
    return PreparedCase(inp, B, p, n, s, epb_suffix, float(ub_local), int(k_max), int(M))


def apply_incumbent_rule(prep: PreparedCase, raw_best_obj: float) -> Tuple[str, float]:
    """Strict tie-break: accept iff raw_best_obj < ub_best."""
    if math.isfinite(raw_best_obj) and raw_best_obj < float(prep.inp.ub_best):
        return "accept", float(raw_best_obj)
    if math.isfinite(prep.inp.ub_best):
        return "discard", float(prep.inp.ub_best)
    return "discard", float(raw_best_obj)


def build_b_prime(prep: PreparedCase) -> List[int]:
    """Majority-shift boundaries among non-empty bins."""
    out = [0]
    prev_nonempty = 0
    prev_majority = -1
    for j in range(1, prep.B + 1):
        if prep.s[j] == prep.s[j - 1]:
            continue
        majority = 1 if int(prep.inp.pos_bins[j - 1]) > int(prep.inp.neg_bins[j - 1]) else 0
        if prev_nonempty > 0 and majority != prev_majority and out[-1] != prev_nonempty:
            out.append(prev_nonempty)
        prev_nonempty = j
        prev_majority = majority
    if out[-1] != prep.B:
        out.append(prep.B)
    return out


def _discard(prep: PreparedCase) -> SolverResult:
    final_obj = float(prep.inp.ub_best) if math.isfinite(prep.inp.ub_best) else INF
    return SolverResult("discard", INF, final_obj, None, [])


def _reconstruct(parent: Sequence[Sequence[int]], endpoints: Sequence[int], k: int, t: int) -> Optional[List[Tuple[int, int]]]:
    intervals: List[Tuple[int, int]] = []
    i = int(k)
    cur_t = int(t)
    while i > 0:
        if i == 1:
            intervals.append((1, int(endpoints[cur_t])))
            break
        prev_t = int(parent[i][cur_t])
        if prev_t < 0:
            return None
        intervals.append((int(endpoints[prev_t]) + 1, int(endpoints[cur_t])))
        cur_t = prev_t
        i -= 1
    intervals.reverse()
    return intervals


def _finish(prep: PreparedCase, best_k: Optional[int], best_obj: float, parent, endpoints, t_final) -> SolverResult:
    if best_k is None:
        return _discard(prep)
    intervals = _reconstruct(parent, endpoints, best_k, t_final)
    if intervals is None:
        return _discard(prep)
    status, final_obj = apply_incumbent_rule(prep, best_obj)
    return SolverResult(status, float(best_obj), float(final_obj), int(best_k), intervals)


def _solve_linearized(prep: PreparedCase, endpoints: Sequence[int]) -> SolverResult:
    """Exact O(M*T) DP for binary loss on ordered endpoints, T=len(endpoints)-1."""
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    m = int(prep.inp.m)
    lam = float(prep.inp.lam)
    T = len(endpoints) - 1

    d = [0] * (prep.B + 1)
    for j in range(1, prep.B + 1):
        d[j] = prep.p[j] - prep.n[j]

    S = [int(prep.s[v]) for v in endpoints]
    D = [int(d[v]) for v in endpoints]
    total_support = int(prep.s[prep.B])

    dp2_prev = [INF] * (T + 1)
    dp2_prev[0] = 0.0
    parent = [[-1] * (T + 1) for _ in range(prep.M + 1)]

    best_obj = INF
    best_k: Optional[int] = None

    for i in range(1, prep.M + 1):
        dp2_cur = [INF] * (T + 1)

        a_min = INF  # min(g[k] + D[k])
        b_min = INF  # min(g[k] - D[k])
        a_arg = -1
        b_arg = -1
        k_ptr = max(0, i - 1)

        for t in range(i, T + 1):
            if S[t] < i * m:
                continue

            while k_ptr <= t - 1 and S[k_ptr] <= S[t] - m:
                prev = dp2_prev[k_ptr]
                if math.isfinite(prev):
                    g = prev - float(S[k_ptr])
                    dk = float(D[k_ptr])
                    a_val = g + dk
                    b_val = g - dk
                    if a_val < a_min:
                        a_min, a_arg = a_val, k_ptr
                    if b_val < b_min:
                        b_min, b_arg = b_val, k_ptr
                k_ptr += 1

            if not (math.isfinite(a_min) or math.isfinite(b_min)):
                continue

            x = float(D[t])
            if a_min - x <= b_min + x:
                best_inner = a_min - x
                best_prev = a_arg
            else:
                best_inner = b_min + x
                best_prev = b_arg

            dp2 = float(S[t]) + best_inner
            if not math.isfinite(dp2):
                continue

            # If not at full length yet, enforce feasibility of at least one remaining leaf
            # and prune using EPB lower bound against UB_local.
            if t < T:
                v = int(endpoints[t])
                if total_support - int(prep.s[v]) < m:
                    continue
                lb = (dp2 / 2.0) + float(prep.epb_suffix[v + 1]) + (i + 1) * lam
                if lb >= float(prep.ub_local):
                    continue

            dp2_cur[t] = dp2
            parent[i][t] = int(best_prev)

        if i >= 2 and math.isfinite(dp2_cur[T]):
            obj = (dp2_cur[T] / 2.0) + lam * i
            if obj < best_obj:
                best_obj = obj
                best_k = i

        dp2_prev = dp2_cur

    return _finish(prep, best_k, best_obj, parent, endpoints, T)


def solve_candidate_majority_shift(prep: PreparedCase) -> SolverResult:
    """Majority-shift projected solver. Exact on projected boundary set B'."""
    return _solve_linearized(prep, build_b_prime(prep))


def solve_linear_full_dp(prep: PreparedCase) -> SolverResult:
    """Exact full-boundary linear-time-in-B solver (for fixed M)."""
    return _solve_linearized(prep, list(range(prep.B + 1)))


def solve_baseline_full_dp(prep: PreparedCase) -> SolverResult:
    """Exact full-boundary DP. Complexity O(M*B^2)."""
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    B, M = prep.B, prep.M
    m, lam = int(prep.inp.m), float(prep.inp.lam)

    dp = [[INF] * (B + 1) for _ in range(M + 1)]
    parent = [[-1] * (B + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for t in range(1, B + 1):
        if prep.s[t] >= m:
            dp[1][t] = interval_error(prep.p, prep.n, 1, t)

    for i in range(2, M + 1):
        for t in range(i, B + 1):
            if prep.s[t] < i * m:
                continue
            best = INF
            best_prev = -1
            for k in range(i - 1, t):
                prev = dp[i - 1][k]
                if not math.isfinite(prev):
                    continue
                u = k + 1
                if prep.s[t] - prep.s[u - 1] < m:
                    continue
                cand = prev + interval_error(prep.p, prep.n, u, t)
                if cand < best:
                    best, best_prev = cand, k
            if math.isfinite(best):
                dp[i][t] = best
                parent[i][t] = best_prev

    best_obj = INF
    best_k: Optional[int] = None
    for i in range(2, M + 1):
        if math.isfinite(dp[i][B]):
            obj = dp[i][B] + lam * i
            if obj < best_obj:
                best_obj = obj
                best_k = i

    return _finish(prep, best_k, best_obj, parent, list(range(B + 1)), B)


def solve_snip_full_dp(prep: PreparedCase) -> SolverResult:
    """Exact constrained DP with SNIP-style pruning. Worst-case O(M*B^2)."""
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    B, M = prep.B, prep.M
    m, lam = int(prep.inp.m), float(prep.inp.lam)

    dp = [[INF] * (B + 1) for _ in range(M + 1)]
    parent = [[-1] * (B + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for t in range(1, B + 1):
        if prep.s[t] >= m:
            dp[1][t] = interval_error(prep.p, prep.n, 1, t)

    next_feasible = [B + 1] * (B + 1)
    right = 1
    for t in range(B):
        right = max(right, t + 1)
        while right <= B and prep.s[right] - prep.s[t] < m:
            right += 1
        next_feasible[t] = right if right <= B else B + 1

    for i in range(2, M + 1):
        active: List[int] = []
        remove_at = [B + 1] * (B + 1)

        for t in range(i, B + 1):
            if prep.s[t] < i * m:
                continue

            k_new = t - 1
            if math.isfinite(dp[i - 1][k_new]):
                active.append(k_new)
            active = [k for k in active if remove_at[k] > t]

            best = INF
            best_prev = -1
            for k in active:
                if prep.s[t] - prep.s[k] < m:
                    continue
                cand = dp[i - 1][k] + interval_error(prep.p, prep.n, k + 1, t)
                if cand < best:
                    best, best_prev = cand, k
            if math.isfinite(best):
                dp[i][t] = best
                parent[i][t] = best_prev

            prev_t = dp[i - 1][t]
            if not math.isfinite(prev_t):
                continue
            start = next_feasible[t]
            if start > B:
                continue
            for k in active:
                if prep.s[t] - prep.s[k] < m:
                    continue
                lhs = dp[i - 1][k] + interval_error(prep.p, prep.n, k + 1, t)
                if lhs >= prev_t:
                    remove_at[k] = min(remove_at[k], start)

    best_obj = INF
    best_k: Optional[int] = None
    for i in range(2, M + 1):
        if math.isfinite(dp[i][B]):
            obj = dp[i][B] + lam * i
            if obj < best_obj:
                best_obj = obj
                best_k = i

    return _finish(prep, best_k, best_obj, parent, list(range(B + 1)), B)


def solve_gosdt_dpb(prep: PreparedCase) -> SolverResult:
    """GOSDT-style bounded best-first search on prefix states.

    State: (i, t) = i leaves cover bins 1..t.
    Transition: add one leaf [t+1..v], v>t, with support >= m.
    Cost: E(t+1, v) + lambda.

    This is exact for the same capped problem (k<=M, min-support, strict incumbent rule),
    with GOSDT-like lower/upper-bound pruning:
    - lower bound via EPB suffix + at least one future lambda
    - global incumbent from best complete state found so far
    """
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    B, M = prep.B, prep.M
    m, lam = int(prep.inp.m), float(prep.inp.lam)
    s = prep.s

    def h(i: int, t: int) -> float:
        if t == B:
            return 0.0
        # Need at least one additional leaf to cover a non-empty suffix.
        if s[B] - s[t] < m or i >= M:
            return INF
        return float(prep.epb_suffix[t + 1]) + lam

    best_goal = INF
    best_state: Optional[Tuple[int, int]] = None
    pq: List[Tuple[float, float, int, int]] = []
    g_best: Dict[Tuple[int, int], float] = {(0, 0): 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    h0 = h(0, 0)
    if not math.isfinite(h0):
        return _discard(prep)
    heapq.heappush(pq, (h0, 0.0, 0, 0))

    while pq:
        f_cur, g_cur, i, t = heapq.heappop(pq)
        key = (i, t)
        if g_cur != g_best.get(key, INF):
            continue

        if math.isfinite(best_goal) and f_cur >= best_goal:
            continue

        # Goal state requires at least 2 leaves.
        if t == B and i >= 2:
            if g_cur < best_goal:
                best_goal = g_cur
                best_state = key
            continue

        if i >= M:
            continue

        u = t + 1
        # Extend next leaf endpoint v; enforce support >= m.
        for v in range(u, B + 1):
            if s[v] - s[u - 1] < m:
                continue

            j = i + 1
            nxt = (j, v)
            step = interval_error(prep.p, prep.n, u, v) + lam
            g_nxt = g_cur + step
            if g_nxt >= g_best.get(nxt, INF):
                continue

            h_nxt = h(j, v)
            if not math.isfinite(h_nxt):
                continue
            f_nxt = g_nxt + h_nxt
            if math.isfinite(best_goal) and f_nxt >= best_goal:
                continue

            g_best[nxt] = g_nxt
            parent[nxt] = key
            heapq.heappush(pq, (f_nxt, g_nxt, j, v))

    if best_state is None or not math.isfinite(best_goal):
        return _discard(prep)

    intervals: List[Tuple[int, int]] = []
    cur = best_state
    while cur != (0, 0):
        prev = parent.get(cur)
        if prev is None:
            return _discard(prep)
        _, t_prev = prev
        _, t_cur = cur
        intervals.append((t_prev + 1, t_cur))
        cur = prev
    intervals.reverse()

    status, final_obj = apply_incumbent_rule(prep, best_goal)
    return SolverResult(status, float(best_goal), float(final_obj), int(best_state[0]), intervals)


def solve_dl85_cache_bnb(prep: PreparedCase) -> SolverResult:
    """DL8.5-style cache branch-and-bound over suffix states.

    State: (u, r) => best objective for bins u..B using exactly r leaves.
    Uses recursion + memoization + admissible EPB lower bound pruning.
    """
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    B, M = prep.B, prep.M
    p, n, s = prep.p, prep.n, prep.s
    m, lam = int(prep.inp.m), float(prep.inp.lam)
    choice: Dict[Tuple[int, int], int] = {}

    @lru_cache(maxsize=None)
    def f(u: int, r: int) -> float:
        rem = int(s[B] - s[u - 1])
        if rem < r * m:
            return INF

        if r == 1:
            if rem < m:
                return INF
            return interval_error(p, n, u, B) + lam

        best = INF
        best_split = -1

        for v in range(u, B):
            left_support = int(s[v] - s[u - 1])
            if left_support < m:
                continue

            right_support = int(s[B] - s[v])
            if right_support < (r - 1) * m:
                break

            left_cost = interval_error(p, n, u, v) + lam
            lb_right = float(prep.epb_suffix[v + 1]) + (r - 1) * lam
            if left_cost + lb_right >= best:
                continue

            cand = left_cost + f(v + 1, r - 1)
            if cand < best:
                best = cand
                best_split = v

        if best_split >= 0:
            choice[(u, r)] = best_split
        return best

    best_obj = INF
    best_k: Optional[int] = None
    for k in range(2, M + 1):
        obj = f(1, k)
        if obj < best_obj:
            best_obj = obj
            best_k = k

    if best_k is None or not math.isfinite(best_obj):
        return _discard(prep)

    intervals: List[Tuple[int, int]] = []
    u, r = 1, best_k
    while r > 1:
        v = choice.get((u, r), -1)
        if v < u:
            return _discard(prep)
        intervals.append((u, v))
        u = v + 1
        r -= 1
    intervals.append((u, B))

    status, final_obj = apply_incumbent_rule(prep, best_obj)
    return SolverResult(status, float(best_obj), float(final_obj), int(best_k), intervals)


def solve_binoct_flow_dp(prep: PreparedCase) -> SolverResult:
    """BinOCT-style exact flow-DP on a segment DAG.

    Formulation view:
      nodes: prefix endpoints 0..B
      arc (t -> v): segment [t+1, v], support>=m, cost E(t+1,v)+lambda
      objective: min-cost path 0->B with 2..M arcs
    """
    if prep.k_max < 2 or prep.M < 2:
        return _discard(prep)

    B, M = prep.B, prep.M
    s = prep.s
    m, lam = int(prep.inp.m), float(prep.inp.lam)

    dp = [[INF] * (B + 1) for _ in range(M + 1)]
    parent = [[-1] * (B + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    best_complete = INF

    for i in range(1, M + 1):
        for t in range(i - 1, B):
            prev = dp[i - 1][t]
            if not math.isfinite(prev):
                continue
            if s[t] < (i - 1) * m:
                continue

            for v in range(t + 1, B + 1):
                seg_support = int(s[v] - s[t])
                if seg_support < m:
                    continue
                if s[v] < i * m:
                    continue

                cand = prev + interval_error(prep.p, prep.n, t + 1, v) + lam
                if v < B:
                    lb = cand + float(prep.epb_suffix[v + 1]) + lam
                    if lb >= best_complete:
                        continue

                if cand < dp[i][v]:
                    dp[i][v] = cand
                    parent[i][v] = t
                    if v == B and i >= 2 and cand < best_complete:
                        best_complete = cand

    best_obj = INF
    best_k: Optional[int] = None
    for i in range(2, M + 1):
        if dp[i][B] < best_obj:
            best_obj = dp[i][B]
            best_k = i

    return _finish(prep, best_k, best_obj, parent, list(range(B + 1)), B)


SOLVERS = {
    "baseline": solve_baseline_full_dp,
    "dl85_cbb": solve_dl85_cache_bnb,
    "binoct_flow": solve_binoct_flow_dp,
    "snip": solve_snip_full_dp,
    "gosdt_dpb": solve_gosdt_dpb,
    "linear_fallback": solve_linear_full_dp,
    "candidate": solve_candidate_majority_shift,
}

SOLVER_DISPLAY = {
    "baseline": "Baseline",
    "dl85_cbb": "DL8.5-CBB",
    "binoct_flow": "BinOCT-Flow",
    "snip": "SNIP",
    "gosdt_dpb": "GOSDT-DPB",
    "linear_fallback": "Grid-DP",
    "candidate": "RUSH-DP",
}

SOLVER_MARKER = {
    "baseline": "o",
    "dl85_cbb": "P",
    "binoct_flow": "v",
    "snip": "D",
    "gosdt_dpb": "^",
    "linear_fallback": "x",
    "candidate": "s",
}
