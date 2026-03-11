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
    child_spans: Dict[int, Tuple[Tuple[int, int], ...]]
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
        interval_partition_solver: str = "rush_dp",
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
        self.interval_partition_solver = interval_partition_solver
        self.approx_mode = approx_mode
        self.patch_budget_per_feature = patch_budget_per_feature
        self.exactify_top_m = exactify_top_m
        self.tau_mode = tau_mode
        self.approx_feature_scan_limit = approx_feature_scan_limit
        self.approx_ref_shortlist_enabled = approx_ref_shortlist_enabled
        self.approx_ref_widen_max = approx_ref_widen_max
        self.approx_challenger_sweep_enabled = approx_challenger_sweep_enabled
        self.approx_challenger_sweep_max_features = approx_challenger_sweep_max_features
        self.approx_challenger_sweep_max_patch_calls_per_node = (
            approx_challenger_sweep_max_patch_calls_per_node
        )

    def fit(self, X, y, sample_weight=None):
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
        if sample_weight is None:
            self._w_train = np.full(self._n_train, 1.0 / float(self._n_train), dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w.shape[0] != self._n_train:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.any(w < 0):
                raise ValueError("sample_weight must be non-negative")
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                raise ValueError("sample_weight must have a positive finite sum")
            self._w_train = (w / w_sum).astype(np.float64, copy=False)

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
        if int(self.patch_budget_per_feature) < 0:
            raise ValueError("patch_budget_per_feature must be >= 0")
        if int(self.exactify_top_m) < 0:
            raise ValueError("exactify_top_m must be >= 0")
        if int(self.approx_feature_scan_limit) < 0:
            raise ValueError("approx_feature_scan_limit must be >= 0")
        if int(self.approx_ref_widen_max) < 0:
            raise ValueError("approx_ref_widen_max must be >= 0")
        if int(self.approx_challenger_sweep_max_features) < 1:
            raise ValueError("approx_challenger_sweep_max_features must be >= 1")
        if int(self.approx_challenger_sweep_max_patch_calls_per_node) < 0:
            raise ValueError("approx_challenger_sweep_max_patch_calls_per_node must be >= 0")
        tau_mode_raw = str(getattr(self, "tau_mode", "lambda_sqrt_r")).strip().lower()
        tau_mode_map = {"lambda": 0, "lambda_sqrt_r": 1}
        if tau_mode_raw not in tau_mode_map:
            raise ValueError("tau_mode must be one of {'lambda', 'lambda_sqrt_r'}")
        tau_mode_code = int(tau_mode_map[tau_mode_raw])

        solver_name = str(getattr(self, "interval_partition_solver", "rush_dp")).strip().lower()
        if solver_name in {"optimal_dp", "optimal", "dp"}:
            partition_strategy = 0
            solver_name = "optimal_dp"
        elif solver_name in {"rush_dp", "rushdp", "rush"}:
            partition_strategy = 1
            solver_name = "rush_dp"
        else:
            raise ValueError(
                "interval_partition_solver must be one of "
                "{'optimal_dp', 'rush_dp'}"
            )
        self.interval_partition_solver_ = solver_name

        self.effective_lookahead_depth_ = max(1, min(self.lookahead_depth_budget, self.full_depth_budget))
        if self.use_cpp_solver and _cpp_msplit_fit is not None:
            cpp_result = _cpp_msplit_fit(
                self._Z_train,
                self._y_train,
                self._w_train,
                int(self.full_depth_budget),
                int(self.effective_lookahead_depth_),
                float(self.reg),
                float(self.branch_penalty),
                int(self.min_child_size),
                float(self.time_limit),
                int(self.max_branching),
                int(partition_strategy),
                bool(self.approx_mode),
                int(self.patch_budget_per_feature),
                int(self.exactify_top_m),
                int(tau_mode_code),
                int(self.approx_feature_scan_limit),
                bool(self.approx_ref_shortlist_enabled),
                int(self.approx_ref_widen_max),
                bool(self.approx_challenger_sweep_enabled),
                int(self.approx_challenger_sweep_max_features),
                int(self.approx_challenger_sweep_max_patch_calls_per_node),
            )
            tree_obj = json.loads(str(cpp_result["tree"]))
            self.tree_ = self._dict_to_tree(tree_obj)
            self.lower_bound_ = float(cpp_result["lowerbound"])
            self.upper_bound_ = float(cpp_result["upperbound"])
            self.objective_ = float(cpp_result["objective"])
            self.exact_internal_nodes_ = int(cpp_result.get("exact_internal_nodes", 0))
            self.greedy_internal_nodes_ = int(cpp_result.get("greedy_internal_nodes", 0))
            self.dp_subproblem_calls_ = int(cpp_result.get("dp_subproblem_calls", 0))
            self.dp_cache_hits_ = int(cpp_result.get("dp_cache_hits", 0))
            self.dp_unique_states_ = int(cpp_result.get("dp_unique_states", 0))
            self.dp_cache_profile_enabled_ = int(cpp_result.get("dp_cache_profile_enabled", 0))
            self.dp_cache_lookup_calls_ = int(cpp_result.get("dp_cache_lookup_calls", 0))
            self.dp_cache_miss_no_bucket_ = int(cpp_result.get("dp_cache_miss_no_bucket", 0))
            self.dp_cache_miss_bucket_present_ = int(cpp_result.get("dp_cache_miss_bucket_present", 0))
            self.dp_cache_miss_depth_mismatch_only_ = int(
                cpp_result.get("dp_cache_miss_depth_mismatch_only", 0)
            )
            self.dp_cache_miss_indices_mismatch_ = int(cpp_result.get("dp_cache_miss_indices_mismatch", 0))
            self.dp_cache_depth_match_candidates_ = int(cpp_result.get("dp_cache_depth_match_candidates", 0))
            self.dp_cache_bucket_entries_scanned_ = int(cpp_result.get("dp_cache_bucket_entries_scanned", 0))
            self.dp_cache_bucket_max_size_ = int(cpp_result.get("dp_cache_bucket_max_size", 0))
            self.greedy_subproblem_calls_ = int(cpp_result.get("greedy_subproblem_calls", 0))
            self.greedy_cache_hits_ = int(cpp_result.get("greedy_cache_hits", 0))
            self.greedy_unique_states_ = int(cpp_result.get("greedy_unique_states", 0))
            self.greedy_cache_entries_peak_ = int(cpp_result.get("greedy_cache_entries_peak", 0))
            self.greedy_cache_clears_ = int(cpp_result.get("greedy_cache_clears", 0))
            self.dp_interval_evals_ = int(cpp_result.get("dp_interval_evals", 0))
            self.greedy_interval_evals_ = int(cpp_result.get("greedy_interval_evals", 0))
            self.rush_incumbent_feature_aborts_ = int(cpp_result.get("rush_incumbent_feature_aborts", 0))
            self.rush_total_time_sec_ = float(cpp_result.get("rush_total_time_sec", 0.0))
            self.rush_refinement_child_time_sec_ = 0.0
            self.rush_refinement_child_time_fraction_ = 0.0
            self.rush_refinement_child_calls_ = int(cpp_result.get("rush_refinement_child_calls", 0))
            self.rush_refinement_recursive_calls_ = int(cpp_result.get("rush_refinement_recursive_calls", 0))
            self.rush_refinement_recursive_unique_states_ = int(
                cpp_result.get("rush_refinement_recursive_unique_states", 0)
            )
            self.rush_ub_rescue_picks_ = int(cpp_result.get("rush_ub_rescue_picks", 0))
            self.rush_global_fallback_picks_ = int(cpp_result.get("rush_global_fallback_picks", 0))
            self.rush_profile_enabled_ = int(cpp_result.get("rush_profile_enabled", 0))
            self.rush_profile_ub0_ordering_sec_ = float(cpp_result.get("rush_profile_ub0_ordering_sec", 0.0))
            self.rush_profile_exact_lazy_eval_sec_ = float(cpp_result.get("rush_profile_exact_lazy_eval_sec", 0.0))
            self.rush_profile_exact_lazy_eval_exclusive_sec_ = float(
                cpp_result.get("rush_profile_exact_lazy_eval_exclusive_sec", 0.0)
            )
            self.rush_profile_exact_lazy_eval_sec_depth0_ = float(
                cpp_result.get("rush_profile_exact_lazy_eval_sec_depth0", 0.0)
            )
            self.rush_profile_exact_lazy_eval_exclusive_sec_depth0_ = float(
                cpp_result.get("rush_profile_exact_lazy_eval_exclusive_sec_depth0", 0.0)
            )
            self.rush_profile_exact_lazy_table_init_sec_ = float(
                cpp_result.get("rush_profile_exact_lazy_table_init_sec", 0.0)
            )
            self.rush_profile_exact_lazy_dp_recompute_sec_ = float(
                cpp_result.get("rush_profile_exact_lazy_dp_recompute_sec", 0.0)
            )
            self.rush_profile_exact_lazy_child_solve_sec_ = float(
                cpp_result.get("rush_profile_exact_lazy_child_solve_sec", 0.0)
            )
            self.rush_profile_exact_lazy_child_solve_sec_depth0_ = float(
                cpp_result.get("rush_profile_exact_lazy_child_solve_sec_depth0", 0.0)
            )
            self.rush_profile_exact_lazy_closure_sec_ = float(
                cpp_result.get("rush_profile_exact_lazy_closure_sec", 0.0)
            )
            self.rush_profile_exact_lazy_dp_recompute_calls_ = int(
                cpp_result.get("rush_profile_exact_lazy_dp_recompute_calls", 0)
            )
            self.rush_profile_exact_lazy_closure_passes_ = int(
                cpp_result.get("rush_profile_exact_lazy_closure_passes", 0)
            )
            self.interval_refinements_attempted_ = int(
                cpp_result.get("interval_refinements_attempted", 0)
            )
            self.expensive_child_calls_ = int(cpp_result.get("expensive_child_calls", 0))
            self.expensive_child_sec_ = float(cpp_result.get("expensive_child_sec", 0.0))
            self.expensive_child_exactify_calls_ = int(
                cpp_result.get("expensive_child_exactify_calls", 0)
            )
            self.expensive_child_exactify_sec_ = float(
                cpp_result.get("expensive_child_exactify_sec", 0.0)
            )
            self.approx_mode_enabled_ = int(cpp_result.get("approx_mode_enabled", 0))
            self.approx_ref_shortlist_enabled_ = int(
                cpp_result.get("approx_ref_shortlist_enabled", 0)
            )
            self.approx_challenger_sweep_enabled_ = int(
                cpp_result.get("approx_challenger_sweep_enabled", 0)
            )
            self.approx_lhat_computed_ = int(cpp_result.get("approx_lhat_computed", 0))
            self.approx_greedy_patch_calls_ = int(cpp_result.get("approx_greedy_patch_calls", 0))
            self.approx_greedy_patches_applied_ = int(cpp_result.get("approx_greedy_patches_applied", 0))
            self.approx_greedy_ub_updates_total_ = int(
                cpp_result.get("approx_greedy_ub_updates_total", 0)
            )
            self.approx_greedy_patch_sec_ = float(cpp_result.get("approx_greedy_patch_sec", 0.0))
            self.approx_exactify_triggered_nodes_ = int(cpp_result.get("approx_exactify_triggered_nodes", 0))
            self.approx_exactify_features_exact_solved_ = int(
                cpp_result.get("approx_exactify_features_exact_solved", 0)
            )
            self.approx_exactify_stops_by_separation_ = int(
                cpp_result.get("approx_exactify_stops_by_separation", 0)
            )
            self.approx_exactify_stops_by_cap_ = int(
                cpp_result.get("approx_exactify_stops_by_cap", 0)
            )
            self.approx_exactify_stops_by_ambiguous_empty_ = int(
                cpp_result.get("approx_exactify_stops_by_ambiguous_empty", 0)
            )
            self.approx_exactify_stops_by_no_improve_ = int(
                cpp_result.get("approx_exactify_stops_by_no_improve", 0)
            )
            self.approx_exactify_stops_by_separation_depth0_ = int(
                cpp_result.get("approx_exactify_stops_by_separation_depth0", 0)
            )
            self.approx_exactify_stops_by_separation_depth1_ = int(
                cpp_result.get("approx_exactify_stops_by_separation_depth1", 0)
            )
            self.approx_exactify_stops_by_cap_depth0_ = int(
                cpp_result.get("approx_exactify_stops_by_cap_depth0", 0)
            )
            self.approx_exactify_stops_by_cap_depth1_ = int(
                cpp_result.get("approx_exactify_stops_by_cap_depth1", 0)
            )
            self.approx_exactify_features_exact_solved_depth0_ = int(
                cpp_result.get("approx_exactify_features_exact_solved_depth0", 0)
            )
            self.approx_exactify_features_exact_solved_depth1_ = int(
                cpp_result.get("approx_exactify_features_exact_solved_depth1", 0)
            )
            self.approx_exactify_set_size_depth0_min_ = int(
                cpp_result.get("approx_exactify_set_size_depth0_min", 0)
            )
            self.approx_exactify_set_size_depth0_mean_ = float(
                cpp_result.get("approx_exactify_set_size_depth0_mean", 0.0)
            )
            self.approx_exactify_set_size_depth0_max_ = int(
                cpp_result.get("approx_exactify_set_size_depth0_max", 0)
            )
            self.approx_exactify_set_size_depth1_min_ = int(
                cpp_result.get("approx_exactify_set_size_depth1_min", 0)
            )
            self.approx_exactify_set_size_depth1_mean_ = float(
                cpp_result.get("approx_exactify_set_size_depth1_mean", 0.0)
            )
            self.approx_exactify_set_size_depth1_max_ = int(
                cpp_result.get("approx_exactify_set_size_depth1_max", 0)
            )
            self.approx_exactify_avg_features_per_triggered_node_ = float(
                cpp_result.get("approx_exactify_avg_features_per_triggered_node", 0.0)
            )
            self.approx_exactify_ambiguous_set_size_min_ = float(
                cpp_result.get("approx_exactify_ambiguous_set_size_min", 0.0)
            )
            self.approx_exactify_ambiguous_set_size_mean_ = float(
                cpp_result.get("approx_exactify_ambiguous_set_size_mean", 0.0)
            )
            self.approx_exactify_ambiguous_set_size_max_ = int(
                cpp_result.get("approx_exactify_ambiguous_set_size_max", 0)
            )
            self.approx_exactify_ambiguous_set_shrank_steps_ = int(
                cpp_result.get("approx_exactify_ambiguous_set_shrank_steps", 0)
            )
            self.approx_exactify_cap_effective_depth0_ = float(
                cpp_result.get("approx_exactify_cap_effective_depth0", 0.0)
            )
            self.approx_exactify_cap_effective_depth1_ = float(
                cpp_result.get("approx_exactify_cap_effective_depth1", 0.0)
            )
            self.approx_challenger_sweep_invocations_ = int(
                cpp_result.get("approx_challenger_sweep_invocations", 0)
            )
            self.approx_challenger_sweep_features_processed_ = int(
                cpp_result.get("approx_challenger_sweep_features_processed", 0)
            )
            self.approx_challenger_sweep_sec_ = float(
                cpp_result.get("approx_challenger_sweep_sec", 0.0)
            )
            self.approx_challenger_sweep_skipped_large_ambiguous_ = int(
                cpp_result.get("approx_challenger_sweep_skipped_large_ambiguous", 0)
            )
            self.approx_challenger_sweep_patch_cap_hit_ = int(
                cpp_result.get("approx_challenger_sweep_patch_cap_hit", 0)
            )
            self.approx_uncertainty_triggered_nodes_ = int(
                cpp_result.get("approx_uncertainty_triggered_nodes", 0)
            )
            self.approx_exactify_trigger_rate_depth0_ = float(
                cpp_result.get("approx_exactify_trigger_rate_depth0", 0.0)
            )
            self.approx_exactify_trigger_rate_depth1_ = float(
                cpp_result.get("approx_exactify_trigger_rate_depth1", 0.0)
            )
            self.approx_uncertainty_trigger_rate_depth0_ = float(
                cpp_result.get("approx_uncertainty_trigger_rate_depth0", 0.0)
            )
            self.approx_uncertainty_trigger_rate_depth1_ = float(
                cpp_result.get("approx_uncertainty_trigger_rate_depth1", 0.0)
            )
            self.approx_eligible_nodes_depth0_ = int(cpp_result.get("approx_eligible_nodes_depth0", 0))
            self.approx_eligible_nodes_depth1_ = int(cpp_result.get("approx_eligible_nodes_depth1", 0))
            self.approx_exactify_triggered_nodes_depth0_ = int(
                cpp_result.get("approx_exactify_triggered_nodes_depth0", 0)
            )
            self.approx_exactify_triggered_nodes_depth1_ = int(
                cpp_result.get("approx_exactify_triggered_nodes_depth1", 0)
            )
            self.approx_uncertainty_triggered_nodes_depth0_ = int(
                cpp_result.get("approx_uncertainty_triggered_nodes_depth0", 0)
            )
            self.approx_uncertainty_triggered_nodes_depth1_ = int(
                cpp_result.get("approx_uncertainty_triggered_nodes_depth1", 0)
            )
            self.approx_pub_unrefined_cells_on_pub_total_ = int(
                cpp_result.get("approx_pub_unrefined_cells_on_pub_total", 0)
            )
            self.approx_pub_patchable_cells_total_ = int(
                cpp_result.get("approx_pub_patchable_cells_total", 0)
            )
            self.approx_pub_cells_skipped_by_childrows_ = int(
                cpp_result.get("approx_pub_cells_skipped_by_childrows", 0)
            )
            self.approx_nodes_with_patchable_pub_ = int(
                cpp_result.get("approx_nodes_with_patchable_pub", 0)
            )
            self.approx_nodes_with_patch_calls_ = int(
                cpp_result.get("approx_nodes_with_patch_calls", 0)
            )
            self.approx_patch_cell_cache_hits_ = int(
                cpp_result.get("approx_patch_cell_cache_hits", 0)
            )
            self.approx_patch_cell_cache_misses_ = int(
                cpp_result.get("approx_patch_cell_cache_misses", 0)
            )
            self.approx_patch_cache_hit_updates_ = int(
                cpp_result.get("approx_patch_cache_hit_updates", 0)
            )
            self.approx_patch_cache_miss_oracle_calls_ = int(
                cpp_result.get("approx_patch_cache_miss_oracle_calls", 0)
            )
            self.approx_patch_subset_materializations_ = int(
                cpp_result.get("approx_patch_subset_materializations", 0)
            )
            self.approx_patch_skipped_already_tight_ = int(
                cpp_result.get("approx_patch_skipped_already_tight", 0)
            )
            self.approx_patch_skipped_no_possible_improve_ = int(
                cpp_result.get("approx_patch_skipped_no_possible_improve", 0)
            )
            self.approx_patch_skipped_cached_ = int(
                cpp_result.get("approx_patch_skipped_cached", 0)
            )
            self.approx_patch_budget_effective_min_ = int(
                cpp_result.get("approx_patch_budget_effective_min", 0)
            )
            self.approx_patch_budget_effective_avg_ = float(
                cpp_result.get("approx_patch_budget_effective_avg", 0.0)
            )
            self.approx_patch_budget_effective_max_ = int(
                cpp_result.get("approx_patch_budget_effective_max", 0)
            )
            self.approx_ref_neff_mean_ = float(cpp_result.get("approx_ref_neff_mean", 0.0))
            self.approx_ref_neff_max_ = float(cpp_result.get("approx_ref_neff_max", 0.0))
            self.approx_ref_k0_min_ = int(cpp_result.get("approx_ref_k0_min", 0))
            self.approx_ref_k0_mean_ = float(cpp_result.get("approx_ref_k0_mean", 0.0))
            self.approx_ref_k0_max_ = int(cpp_result.get("approx_ref_k0_max", 0))
            self.approx_ref_k_final_min_ = int(cpp_result.get("approx_ref_k_final_min", 0))
            self.approx_ref_k_final_mean_ = float(cpp_result.get("approx_ref_k_final_mean", 0.0))
            self.approx_ref_k_final_max_ = int(cpp_result.get("approx_ref_k_final_max", 0))
            self.approx_ref_k_depth0_mean_ = float(cpp_result.get("approx_ref_k_depth0_mean", 0.0))
            self.approx_ref_k_depth1_mean_ = float(cpp_result.get("approx_ref_k_depth1_mean", 0.0))
            self.approx_ref_widen_count_ = int(cpp_result.get("approx_ref_widen_count", 0))
            self.approx_ref_widen_count_depth0_ = int(
                cpp_result.get("approx_ref_widen_count_depth0", 0)
            )
            self.approx_ref_widen_count_depth1_ = int(
                cpp_result.get("approx_ref_widen_count_depth1", 0)
            )
            self.approx_ref_chosen_feature_rank_depth0_ = float(
                cpp_result.get("approx_ref_chosen_feature_rank_depth0", 0.0)
            )
            self.approx_ref_chosen_feature_rank_depth1_ = float(
                cpp_result.get("approx_ref_chosen_feature_rank_depth1", 0.0)
            )
            self.approx_ref_chosen_in_initial_shortlist_rate_depth0_ = float(
                cpp_result.get("approx_ref_chosen_in_initial_shortlist_rate_depth0", 0.0)
            )
            self.approx_ref_chosen_in_initial_shortlist_rate_depth1_ = float(
                cpp_result.get("approx_ref_chosen_in_initial_shortlist_rate_depth1", 0.0)
            )
            self.fast100_exactify_nodes_allowed_ = int(
                cpp_result.get("fast100_exactify_nodes_allowed", 0)
            )
            self.fast100_exactify_nodes_skipped_small_support_ = int(
                cpp_result.get("fast100_exactify_nodes_skipped_small_support", 0)
            )
            self.fast100_exactify_nodes_skipped_dominant_gain_ = int(
                cpp_result.get("fast100_exactify_nodes_skipped_dominant_gain", 0)
            )
            self.depth1_skipped_by_low_global_ambiguity_ = int(
                cpp_result.get("depth1_skipped_by_low_global_ambiguity", 0)
            )
            self.depth1_skipped_by_large_gap_ = int(
                cpp_result.get("depth1_skipped_by_large_gap", 0)
            )
            self.depth1_exactify_challenger_nodes_ = int(
                cpp_result.get("depth1_exactify_challenger_nodes", 0)
            )
            self.depth1_exactified_nodes_ = int(
                cpp_result.get("depth1_exactified_nodes", 0)
            )
            self.depth1_exactified_features_mean_ = float(
                cpp_result.get("depth1_exactified_features_mean", 0.0)
            )
            self.depth1_exactified_features_max_ = int(
                cpp_result.get("depth1_exactified_features_max", 0)
            )
            self.depth1_teacher_replaced_runnerup_ = int(
                cpp_result.get("depth1_teacher_replaced_runnerup", 0)
            )
            self.depth1_teacher_rejected_by_uhat_gate_ = int(
                cpp_result.get("depth1_teacher_rejected_by_uhat_gate", 0)
            )
            self.depth1_exactify_set_size_mean_ = float(
                cpp_result.get("depth1_exactify_set_size_mean", 0.0)
            )
            self.depth1_exactify_set_size_max_ = int(
                cpp_result.get("depth1_exactify_set_size_max", 0)
            )
            self.fast100_skipped_by_ub_lb_separation_ = int(
                cpp_result.get("fast100_skipped_by_ub_lb_separation", 0)
            )
            self.fast100_widen_forbidden_depth_gt0_attempts_ = int(
                cpp_result.get("fast100_widen_forbidden_depth_gt0_attempts", 0)
            )
            self.fast100_frontier_size_mean_ = float(
                cpp_result.get("fast100_frontier_size_mean", 0.0)
            )
            self.fast100_frontier_size_max_ = int(
                cpp_result.get("fast100_frontier_size_max", 0)
            )
            self.fast100_stopped_midloop_separation_ = int(
                cpp_result.get("fast100_stopped_midloop_separation", 0)
            )
            self.fast100_M_depth0_mean_ = float(cpp_result.get("fast100_M_depth0_mean", 0.0))
            self.fast100_M_depth0_max_ = int(cpp_result.get("fast100_M_depth0_max", 0))
            self.fast100_M_depth1_mean_ = float(cpp_result.get("fast100_M_depth1_mean", 0.0))
            self.fast100_M_depth1_max_ = int(cpp_result.get("fast100_M_depth1_max", 0))
            self.fast100_cf_exactify_nodes_depth0_ = int(
                cpp_result.get("fast100_cf_exactify_nodes_depth0", 0)
            )
            self.fast100_cf_exactify_nodes_depth1_ = int(
                cpp_result.get("fast100_cf_exactify_nodes_depth1", 0)
            )
            self.fast100_cf_skipped_agreement_ = int(
                cpp_result.get("fast100_cf_skipped_agreement", 0)
            )
            self.fast100_cf_skipped_small_regret_ = int(
                cpp_result.get("fast100_cf_skipped_small_regret", 0)
            )
            self.fast100_cf_skipped_low_impact_ = int(
                cpp_result.get("fast100_cf_skipped_low_impact", 0)
            )
            self.fast100_cf_frontier_size_mean_ = float(
                cpp_result.get("fast100_cf_frontier_size_mean", 0.0)
            )
            self.fast100_cf_frontier_size_max_ = int(
                cpp_result.get("fast100_cf_frontier_size_max", 0)
            )
            self.fast100_cf_exactified_features_mean_ = float(
                cpp_result.get("fast100_cf_exactified_features_mean", 0.0)
            )
            self.fast100_cf_exactified_features_max_ = int(
                cpp_result.get("fast100_cf_exactified_features_max", 0)
            )
            self.rootsafe_exactified_features_ = int(
                cpp_result.get("rootsafe_exactified_features", 0)
            )
            self.rootsafe_root_winner_changed_vs_proxy_ = int(
                cpp_result.get("rootsafe_root_winner_changed_vs_proxy", 0)
            )
            self.rootsafe_root_candidates_K_ = int(
                cpp_result.get("rootsafe_root_candidates_K", 0)
            )
            self.fast100_used_lgb_prior_tiebreak_ = int(
                cpp_result.get("fast100_used_lgb_prior_tiebreak", 0)
            )
            self.gini_dp_calls_root_ = int(cpp_result.get("gini_dp_calls_root", 0))
            self.gini_dp_calls_depth1_ = int(cpp_result.get("gini_dp_calls_depth1", 0))
            self.gini_teacher_chosen_depth1_ = int(
                cpp_result.get("gini_teacher_chosen_depth1", 0)
            )
            self.gini_tiebreak_used_in_shortlist_ = int(
                cpp_result.get("gini_tiebreak_used_in_shortlist", 0)
            )
            self.gini_dp_sec_ = float(cpp_result.get("gini_dp_sec", 0.0))
            self.gini_root_k0_ = int(cpp_result.get("gini_root_k0", 0))
            self.gini_endpoints_added_root_ = int(
                cpp_result.get("gini_endpoints_added_root", 0)
            )
            self.gini_endpoints_added_depth1_ = int(
                cpp_result.get("gini_endpoints_added_depth1", 0)
            )
            self.gini_endpoints_features_touched_root_ = int(
                cpp_result.get("gini_endpoints_features_touched_root", 0)
            )
            self.gini_endpoints_features_touched_depth1_ = int(
                cpp_result.get("gini_endpoints_features_touched_depth1", 0)
            )
            self.gini_endpoints_added_per_feature_max_ = int(
                cpp_result.get("gini_endpoints_added_per_feature_max", 0)
            )
            self.gini_endpoint_sec_ = float(cpp_result.get("gini_endpoint_sec", 0.0))
            self.rush_feature_logs_root_ = []
            self.rush_feature_logs_depth1_ = []
            self.rush_refinement_depth_logs_ = []
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
            self.dp_subproblem_calls_ = int(len(self._dp_cache))
            self.dp_cache_hits_ = 0
            self.dp_unique_states_ = int(len(self._dp_cache))
            self.dp_cache_profile_enabled_ = 0
            self.dp_cache_lookup_calls_ = 0
            self.dp_cache_miss_no_bucket_ = 0
            self.dp_cache_miss_bucket_present_ = 0
            self.dp_cache_miss_depth_mismatch_only_ = 0
            self.dp_cache_miss_indices_mismatch_ = 0
            self.dp_cache_depth_match_candidates_ = 0
            self.dp_cache_bucket_entries_scanned_ = 0
            self.dp_cache_bucket_max_size_ = 0
            self.greedy_subproblem_calls_ = int(len(self._greedy_cache))
            self.greedy_cache_hits_ = 0
            self.greedy_unique_states_ = int(len(self._greedy_cache))
            self.greedy_cache_entries_peak_ = int(len(self._greedy_cache))
            self.greedy_cache_clears_ = 0
            self.dp_interval_evals_ = 0
            self.greedy_interval_evals_ = 0
            self.rush_incumbent_feature_aborts_ = 0
            self.rush_total_time_sec_ = 0.0
            self.rush_refinement_child_time_sec_ = 0.0
            self.rush_refinement_child_time_fraction_ = 0.0
            self.rush_refinement_child_calls_ = 0
            self.rush_refinement_recursive_calls_ = 0
            self.rush_refinement_recursive_unique_states_ = 0
            self.rush_ub_rescue_picks_ = 0
            self.rush_global_fallback_picks_ = 0
            self.rush_profile_enabled_ = 0
            self.rush_profile_ub0_ordering_sec_ = 0.0
            self.rush_profile_exact_lazy_eval_sec_ = 0.0
            self.rush_profile_exact_lazy_eval_exclusive_sec_ = 0.0
            self.rush_profile_exact_lazy_eval_sec_depth0_ = 0.0
            self.rush_profile_exact_lazy_eval_exclusive_sec_depth0_ = 0.0
            self.rush_profile_exact_lazy_table_init_sec_ = 0.0
            self.rush_profile_exact_lazy_dp_recompute_sec_ = 0.0
            self.rush_profile_exact_lazy_child_solve_sec_ = 0.0
            self.rush_profile_exact_lazy_child_solve_sec_depth0_ = 0.0
            self.rush_profile_exact_lazy_closure_sec_ = 0.0
            self.rush_profile_exact_lazy_dp_recompute_calls_ = 0
            self.rush_profile_exact_lazy_closure_passes_ = 0
            self.interval_refinements_attempted_ = 0
            self.expensive_child_calls_ = 0
            self.expensive_child_sec_ = 0.0
            self.expensive_child_exactify_calls_ = 0
            self.expensive_child_exactify_sec_ = 0.0
            self.approx_mode_enabled_ = int(bool(self.approx_mode))
            self.approx_ref_shortlist_enabled_ = int(bool(self.approx_ref_shortlist_enabled))
            self.approx_challenger_sweep_enabled_ = int(bool(self.approx_challenger_sweep_enabled))
            self.approx_lhat_computed_ = 0
            self.approx_greedy_patch_calls_ = 0
            self.approx_greedy_patches_applied_ = 0
            self.approx_greedy_ub_updates_total_ = 0
            self.approx_greedy_patch_sec_ = 0.0
            self.approx_exactify_triggered_nodes_ = 0
            self.approx_exactify_features_exact_solved_ = 0
            self.approx_exactify_stops_by_separation_ = 0
            self.approx_exactify_stops_by_cap_ = 0
            self.approx_exactify_stops_by_ambiguous_empty_ = 0
            self.approx_exactify_stops_by_no_improve_ = 0
            self.approx_exactify_stops_by_separation_depth0_ = 0
            self.approx_exactify_stops_by_separation_depth1_ = 0
            self.approx_exactify_stops_by_cap_depth0_ = 0
            self.approx_exactify_stops_by_cap_depth1_ = 0
            self.approx_exactify_features_exact_solved_depth0_ = 0
            self.approx_exactify_features_exact_solved_depth1_ = 0
            self.approx_exactify_set_size_depth0_min_ = 0
            self.approx_exactify_set_size_depth0_mean_ = 0.0
            self.approx_exactify_set_size_depth0_max_ = 0
            self.approx_exactify_set_size_depth1_min_ = 0
            self.approx_exactify_set_size_depth1_mean_ = 0.0
            self.approx_exactify_set_size_depth1_max_ = 0
            self.approx_exactify_avg_features_per_triggered_node_ = 0.0
            self.approx_exactify_ambiguous_set_size_min_ = 0.0
            self.approx_exactify_ambiguous_set_size_mean_ = 0.0
            self.approx_exactify_ambiguous_set_size_max_ = 0
            self.approx_exactify_ambiguous_set_shrank_steps_ = 0
            self.approx_exactify_cap_effective_depth0_ = 0.0
            self.approx_exactify_cap_effective_depth1_ = 0.0
            self.approx_challenger_sweep_invocations_ = 0
            self.approx_challenger_sweep_features_processed_ = 0
            self.approx_challenger_sweep_sec_ = 0.0
            self.approx_challenger_sweep_skipped_large_ambiguous_ = 0
            self.approx_challenger_sweep_patch_cap_hit_ = 0
            self.approx_uncertainty_triggered_nodes_ = 0
            self.approx_exactify_trigger_rate_depth0_ = 0.0
            self.approx_exactify_trigger_rate_depth1_ = 0.0
            self.approx_uncertainty_trigger_rate_depth0_ = 0.0
            self.approx_uncertainty_trigger_rate_depth1_ = 0.0
            self.approx_eligible_nodes_depth0_ = 0
            self.approx_eligible_nodes_depth1_ = 0
            self.approx_exactify_triggered_nodes_depth0_ = 0
            self.approx_exactify_triggered_nodes_depth1_ = 0
            self.approx_uncertainty_triggered_nodes_depth0_ = 0
            self.approx_uncertainty_triggered_nodes_depth1_ = 0
            self.approx_pub_unrefined_cells_on_pub_total_ = 0
            self.approx_pub_patchable_cells_total_ = 0
            self.approx_pub_cells_skipped_by_childrows_ = 0
            self.approx_nodes_with_patchable_pub_ = 0
            self.approx_nodes_with_patch_calls_ = 0
            self.approx_patch_cell_cache_hits_ = 0
            self.approx_patch_cell_cache_misses_ = 0
            self.approx_patch_cache_hit_updates_ = 0
            self.approx_patch_cache_miss_oracle_calls_ = 0
            self.approx_patch_subset_materializations_ = 0
            self.approx_patch_skipped_already_tight_ = 0
            self.approx_patch_skipped_no_possible_improve_ = 0
            self.approx_patch_skipped_cached_ = 0
            self.approx_patch_budget_effective_min_ = 0
            self.approx_patch_budget_effective_avg_ = 0.0
            self.approx_patch_budget_effective_max_ = 0
            self.approx_ref_neff_mean_ = 0.0
            self.approx_ref_neff_max_ = 0.0
            self.approx_ref_k0_min_ = 0
            self.approx_ref_k0_mean_ = 0.0
            self.approx_ref_k0_max_ = 0
            self.approx_ref_k_final_min_ = 0
            self.approx_ref_k_final_mean_ = 0.0
            self.approx_ref_k_final_max_ = 0
            self.approx_ref_k_depth0_mean_ = 0.0
            self.approx_ref_k_depth1_mean_ = 0.0
            self.approx_ref_widen_count_ = 0
            self.approx_ref_widen_count_depth0_ = 0
            self.approx_ref_widen_count_depth1_ = 0
            self.approx_ref_chosen_feature_rank_depth0_ = 0.0
            self.approx_ref_chosen_feature_rank_depth1_ = 0.0
            self.approx_ref_chosen_in_initial_shortlist_rate_depth0_ = 0.0
            self.approx_ref_chosen_in_initial_shortlist_rate_depth1_ = 0.0
            self.fast100_exactify_nodes_allowed_ = 0
            self.fast100_exactify_nodes_skipped_small_support_ = 0
            self.fast100_exactify_nodes_skipped_dominant_gain_ = 0
            self.depth1_skipped_by_low_global_ambiguity_ = 0
            self.depth1_skipped_by_large_gap_ = 0
            self.depth1_exactify_challenger_nodes_ = 0
            self.depth1_exactified_nodes_ = 0
            self.depth1_exactified_features_mean_ = 0.0
            self.depth1_exactified_features_max_ = 0
            self.depth1_teacher_replaced_runnerup_ = 0
            self.depth1_teacher_rejected_by_uhat_gate_ = 0
            self.depth1_exactify_set_size_mean_ = 0.0
            self.depth1_exactify_set_size_max_ = 0
            self.fast100_skipped_by_ub_lb_separation_ = 0
            self.fast100_widen_forbidden_depth_gt0_attempts_ = 0
            self.fast100_frontier_size_mean_ = 0.0
            self.fast100_frontier_size_max_ = 0
            self.fast100_stopped_midloop_separation_ = 0
            self.fast100_M_depth0_mean_ = 0.0
            self.fast100_M_depth0_max_ = 0
            self.fast100_M_depth1_mean_ = 0.0
            self.fast100_M_depth1_max_ = 0
            self.fast100_cf_exactify_nodes_depth0_ = 0
            self.fast100_cf_exactify_nodes_depth1_ = 0
            self.fast100_cf_skipped_agreement_ = 0
            self.fast100_cf_skipped_small_regret_ = 0
            self.fast100_cf_skipped_low_impact_ = 0
            self.fast100_cf_frontier_size_mean_ = 0.0
            self.fast100_cf_frontier_size_max_ = 0
            self.fast100_cf_exactified_features_mean_ = 0.0
            self.fast100_cf_exactified_features_max_ = 0
            self.rootsafe_exactified_features_ = 0
            self.rootsafe_root_winner_changed_vs_proxy_ = 0
            self.rootsafe_root_candidates_K_ = 0
            self.fast100_used_lgb_prior_tiebreak_ = 0
            self.gini_dp_calls_root_ = 0
            self.gini_dp_calls_depth1_ = 0
            self.gini_teacher_chosen_depth1_ = 0
            self.gini_tiebreak_used_in_shortlist_ = 0
            self.gini_dp_sec_ = 0.0
            self.gini_root_k0_ = 0
            self.gini_endpoints_added_root_ = 0
            self.gini_endpoints_added_depth1_ = 0
            self.gini_endpoints_features_touched_root_ = 0
            self.gini_endpoints_features_touched_depth1_ = 0
            self.gini_endpoints_added_per_feature_max_ = 0
            self.gini_endpoint_sec_ = 0.0
            self.rush_feature_logs_root_ = []
            self.rush_feature_logs_depth1_ = []
            self.rush_refinement_depth_logs_ = []

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
            child = None
            for group_id in sorted(cur.children.keys()):
                spans = cur.child_spans.get(group_id, ())
                matched = False
                for lo, hi in spans:
                    if int(lo) <= bin_id <= int(hi):
                        matched = True
                        break
                if matched:
                    child = cur.children.get(group_id)
                    break
            if child is None:
                # Route to nearest known span if this bin was unseen during training.
                nearest_group = None
                best_dist = None
                best_lo = None
                for group_id in sorted(cur.children.keys()):
                    spans = cur.child_spans.get(group_id, ())
                    if not spans:
                        continue
                    group_dist = None
                    group_lo = None
                    for lo, hi in spans:
                        lo_i = int(lo)
                        hi_i = int(hi)
                        if group_lo is None or lo_i < group_lo:
                            group_lo = lo_i
                        if lo_i <= bin_id <= hi_i:
                            dist = 0
                        elif bin_id < lo_i:
                            dist = lo_i - bin_id
                        else:
                            dist = bin_id - hi_i
                        if group_dist is None or dist < group_dist:
                            group_dist = dist
                    if group_dist is None:
                        continue
                    if (
                        best_dist is None
                        or group_dist < best_dist
                        or (group_dist == best_dist and group_lo is not None and (best_lo is None or group_lo < best_lo))
                    ):
                        best_dist = group_dist
                        best_lo = group_lo
                        nearest_group = group_id
                if nearest_group is not None:
                    child = cur.children.get(nearest_group)
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
            child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
            largest_bin = -1
            largest_size = -1
            group_id = 0

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_result = self._solve_subproblem(child_indices, depth_remaining - 1, current_depth + 1)
                split_lb += child_result.lb
                split_ub += child_result.ub
                children[group_id] = child_result.tree
                child_spans[group_id] = ((int(bin_id), int(bin_id)),)
                group_id += 1
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            split_penalty = self.branch_penalty * (len(children) - 2)
            split_lb += split_penalty
            split_ub += split_penalty
            best_lb = min(best_lb, split_lb)
            if split_ub < best_ub:
                best_ub = split_ub
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    child_spans=child_spans,
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
            child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
            largest_bin = -1
            largest_size = -1
            group_id = 0

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_obj, child_tree = self._greedy_complete(child_indices, depth_remaining - 1)
                split_objective += child_obj
                children[group_id] = child_tree
                child_spans[group_id] = ((int(bin_id), int(bin_id)),)
                group_id += 1
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            split_objective += self.branch_penalty * (len(children) - 2)
            if split_objective < best_objective:
                best_objective = split_objective
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    child_spans=child_spans,
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
        w_subset = self._w_train[indices]
        positives = int(y_subset.sum())
        negatives = int(y_subset.size - positives)
        pos_w = float(np.sum(w_subset[y_subset == 1]))
        neg_w = float(np.sum(w_subset[y_subset == 0]))

        if pos_w >= neg_w:
            prediction = 1
            mistakes_w = neg_w
        else:
            prediction = 0
            mistakes_w = pos_w

        loss = mistakes_w + self.reg
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

        children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
        child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        groups_raw = tree_obj.get("groups")
        if isinstance(groups_raw, list):
            for group_id, entry in enumerate(groups_raw):
                if not isinstance(entry, dict):
                    continue
                child_obj = entry.get("child")
                if not isinstance(child_obj, dict):
                    continue
                spans_raw = entry.get("spans", [])
                spans: list[Tuple[int, int]] = []
                if isinstance(spans_raw, list):
                    for span in spans_raw:
                        if not isinstance(span, (list, tuple)) or len(span) != 2:
                            continue
                        lo = int(span[0])
                        hi = int(span[1])
                        if hi < lo:
                            lo, hi = hi, lo
                        spans.append((lo, hi))
                if not spans:
                    continue
                children[group_id] = self._dict_to_tree(child_obj)
                child_spans[group_id] = tuple(spans)
        else:
            # Backward compatibility: legacy per-bin child map.
            children_raw = tree_obj.get("children", {})
            if isinstance(children_raw, dict):
                group_id = 0
                for key in sorted(children_raw.keys(), key=lambda x: int(x)):
                    bin_id = int(key)
                    child_obj = children_raw[key]
                    if not isinstance(child_obj, dict):
                        continue
                    children[group_id] = self._dict_to_tree(child_obj)
                    child_spans[group_id] = ((bin_id, bin_id),)
                    group_id += 1

        return MultiNode(
            feature=int(tree_obj["feature"]),
            children=children,
            child_spans=child_spans,
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
        for group_id in sorted(node.children.keys()):
            spans = node.child_spans.get(group_id, ())
            span_text_parts = []
            for lo, hi in spans:
                if int(lo) == int(hi):
                    span_text_parts.append(str(int(lo)))
                else:
                    span_text_parts.append(f"{int(lo)}-{int(hi)}")
            span_text = ",".join(span_text_parts) if span_text_parts else "?"
            lines.append(f"{indent}  group {group_id} bins {span_text} ->")
            lines.append(self._format_tree(node.children[group_id], depth + 2))
        return "\n".join(lines)
