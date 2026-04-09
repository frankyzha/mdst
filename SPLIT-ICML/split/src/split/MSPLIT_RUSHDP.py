"""Compatibility alias for the native MSPLIT solver."""

from __future__ import annotations

from .MSPLIT import MSPLIT


class MSPLIT_RUSHDP(MSPLIT):
    """MSPLIT compatibility alias preserved for older call sites."""

    def __init__(
        self,
        lookahead_depth_budget: int | None = None,
        full_depth_budget: int = 5,
        reg: float = 0.01,
        min_child_size: int = 5,
        max_branching: int = 0,
        time_limit: int = 100,
        verbose: bool = False,
        random_state: int = 0,
        use_cpp_solver: bool = True,
        **legacy_kwargs,
    ):
        del legacy_kwargs
        super().__init__(
            lookahead_depth_budget=lookahead_depth_budget,
            full_depth_budget=full_depth_budget,
            reg=reg,
            min_child_size=min_child_size,
            max_branching=max_branching,
            time_limit=time_limit,
            verbose=verbose,
            random_state=random_state,
            use_cpp_solver=use_cpp_solver,
        )
