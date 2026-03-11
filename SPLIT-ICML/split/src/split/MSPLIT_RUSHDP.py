"""MSPLIT variant that uses RUSH-DP interval partitioning in the native solver."""

from __future__ import annotations

from .MSPLIT import MSPLIT


class MSPLIT_RUSHDP(MSPLIT):
    """MSPLIT with RUSH-DP interval partitioning (C++ mode)."""

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
        approx_mode: bool = False,
        patch_budget_per_feature: int = 12,
        exactify_top_m: int = 2,
        tau_mode: str = "lambda_sqrt_r",
        approx_feature_scan_limit: int = 0,
        approx_ref_shortlist_enabled: bool = True,
        approx_ref_widen_max: int = 1,
        approx_challenger_sweep_enabled: bool = False,
        approx_challenger_sweep_max_features: int = 3,
        approx_challenger_sweep_max_patch_calls_per_node: int = 0,
    ):
        super().__init__(
            lookahead_depth_budget=lookahead_depth_budget,
            full_depth_budget=full_depth_budget,
            reg=reg,
            branch_penalty=branch_penalty,
            max_bins=max_bins,
            min_samples_leaf=min_samples_leaf,
            min_child_size=min_child_size,
            max_branching=max_branching,
            time_limit=time_limit,
            verbose=verbose,
            random_state=random_state,
            input_is_binned=input_is_binned,
            use_cpp_solver=use_cpp_solver,
            interval_partition_solver="rush_dp",
            approx_mode=approx_mode,
            patch_budget_per_feature=patch_budget_per_feature,
            exactify_top_m=exactify_top_m,
            tau_mode=tau_mode,
            approx_feature_scan_limit=approx_feature_scan_limit,
            approx_ref_shortlist_enabled=approx_ref_shortlist_enabled,
            approx_ref_widen_max=approx_ref_widen_max,
            approx_challenger_sweep_enabled=approx_challenger_sweep_enabled,
            approx_challenger_sweep_max_features=approx_challenger_sweep_max_features,
            approx_challenger_sweep_max_patch_calls_per_node=(
                approx_challenger_sweep_max_patch_calls_per_node
            ),
        )
