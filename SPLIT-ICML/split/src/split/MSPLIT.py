"""Multiway SPLIT-style tree solver with LightGBM bins and lookahead greedy completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import os
import json
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

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
        lookahead_depth_budget: int | None = None,
        lookahead_depth: int | None = None,
        full_depth_budget: int = 5,
        reg: float = 0.01,
        family1_soft_weight: float = 0.25,
        min_child_size: int = 5,
        min_split_size: int | None = None,
        max_branching: int = 0,
        time_limit: int = 100,
        verbose: bool = False,
        random_state: int = 0,
        use_cpp_solver: bool = True,
        **legacy_kwargs,
    ):
        del legacy_kwargs
        self.lookahead_depth_budget = lookahead_depth_budget
        self.lookahead_depth = lookahead_depth
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        self.family1_soft_weight = family1_soft_weight
        self.min_child_size = min_child_size
        self.min_split_size = min_split_size
        self.max_branching = max_branching
        self.time_limit = time_limit
        self.verbose = verbose
        self.random_state = random_state
        self.use_cpp_solver = use_cpp_solver

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        teacher_logit=None,
        teacher_boundary_gain=None,
        teacher_boundary_cover=None,
        teacher_boundary_value_jump=None,
        **kwargs,
    ):
        y_encoded, class_labels = self._encode_target(y)
        self.class_labels_ = class_labels
        self.classes_ = class_labels
        Z = check_array(X, ensure_2d=True, dtype=None)
        if not np.issubdtype(np.asarray(Z).dtype, np.integer):
            raise ValueError("MSPLIT expects LightGBM-binned integer features")
        Z = np.asarray(Z, dtype=np.int32)
        if (Z < 0).any():
            raise ValueError("Binned input must be non-negative integer values")
        self.feature_names_ = [f"x{i}" for i in range(Z.shape[1])]

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

        if self.lookahead_depth is not None:
            resolved_lookahead_depth = int(self.lookahead_depth)
        elif self.lookahead_depth_budget is not None:
            resolved_lookahead_depth = int(self.lookahead_depth_budget)
        else:
            resolved_lookahead_depth = max(1, (self.full_depth_budget + 1) // 2)
        self.effective_lookahead_depth_ = max(1, min(resolved_lookahead_depth, self.full_depth_budget))
        if self.use_cpp_solver and _cpp_msplit_fit is not None:
            restore_cache_env = False
            previous_cache_env = os.environ.get("MSPLIT_GREEDY_CACHE_MAX_DEPTH")
            if previous_cache_env is None and self.full_depth_budget <= 2:
                # Tiny exact trees repeatedly revisit the same subproblems.
                # The native solver benefits from a shallow greedy cache here,
                # but we do not want to leak that process-wide setting into
                # larger fits, so set it only for the duration of this call.
                os.environ["MSPLIT_GREEDY_CACHE_MAX_DEPTH"] = str(self.full_depth_budget)
                restore_cache_env = True
            try:
                cpp_result = _cpp_msplit_fit(
                    self._Z_train,
                    self._y_train,
                    self._w_train,
                    teacher_logit,
                    teacher_boundary_gain,
                    teacher_boundary_cover,
                    teacher_boundary_value_jump,
                    int(self.full_depth_budget),
                    int(self.effective_lookahead_depth_),
                    float(self.reg),
                    int(getattr(self, "min_split_size", 0) or 0),
                    int(self.min_child_size),
                    float(self.time_limit),
                    int(self.max_branching),
                    float(getattr(self, "family1_soft_weight", 0.25)),
                )
            finally:
                if restore_cache_env:
                    os.environ.pop("MSPLIT_GREEDY_CACHE_MAX_DEPTH", None)
            tree_obj = json.loads(str(cpp_result["tree"]))
            self.tree_ = self._dict_to_tree(tree_obj)
            objective_value = float(cpp_result.get("objective", 0.0))
            self.lower_bound_ = float(cpp_result.get("lowerbound", objective_value))
            self.upper_bound_ = float(cpp_result.get("upperbound", objective_value))
            self.objective_ = objective_value
            self.family_compare_total_ = int(cpp_result.get("family_compare_total", 0))
            self.family_compare_equivalent_ = int(cpp_result.get("family_compare_equivalent", 0))
            self.family1_both_wins_ = int(cpp_result.get("family1_both_wins", 0))
            self.family2_hard_loss_wins_ = int(cpp_result.get("family2_hard_loss_wins", 0))
            self.family2_hard_impurity_wins_ = int(cpp_result.get("family2_hard_impurity_wins", 0))
            self.family2_both_wins_ = int(cpp_result.get("family2_both_wins", 0))
            self.family_metric_disagreement_ = int(cpp_result.get("family_metric_disagreement", 0))
            self.family_hard_loss_ties_ = int(cpp_result.get("family_hard_loss_ties", 0))
            self.family_hard_impurity_ties_ = int(cpp_result.get("family_hard_impurity_ties", 0))
            self.family_joint_impurity_ties_ = int(cpp_result.get("family_joint_impurity_ties", 0))
            self.family_neither_both_wins_ = int(cpp_result.get("family_neither_both_wins", 0))
            self.family1_selected_by_equivalence_ = int(cpp_result.get("family1_selected_by_equivalence", 0))
            self.family1_selected_by_dominance_ = int(cpp_result.get("family1_selected_by_dominance", 0))
            self.family2_selected_by_dominance_ = int(cpp_result.get("family2_selected_by_dominance", 0))
            self.family_sent_both_ = int(cpp_result.get("family_sent_both", 0))
            self.family1_hard_loss_sum_ = float(cpp_result.get("family1_hard_loss_sum", 0.0))
            self.family2_hard_loss_sum_ = float(cpp_result.get("family2_hard_loss_sum", 0.0))
            self.family_hard_loss_delta_sum_ = float(cpp_result.get("family_hard_loss_delta_sum", 0.0))
            self.family1_hard_impurity_sum_ = float(cpp_result.get("family1_hard_impurity_sum", 0.0))
            self.family2_hard_impurity_sum_ = float(cpp_result.get("family2_hard_impurity_sum", 0.0))
            self.family_hard_impurity_delta_sum_ = float(cpp_result.get("family_hard_impurity_delta_sum", 0.0))
            self.family1_soft_impurity_sum_ = float(cpp_result.get("family1_soft_impurity_sum", 0.0))
            self.family2_soft_impurity_sum_ = float(cpp_result.get("family2_soft_impurity_sum", 0.0))
            self.family_soft_impurity_delta_sum_ = float(cpp_result.get("family_soft_impurity_delta_sum", 0.0))
            self.family1_joint_impurity_sum_ = float(cpp_result.get("family1_joint_impurity_sum", 0.0))
            self.family2_joint_impurity_sum_ = float(cpp_result.get("family2_joint_impurity_sum", 0.0))
            self.family_joint_impurity_delta_sum_ = float(cpp_result.get("family_joint_impurity_delta_sum", 0.0))
            self.debr_refine_calls_ = int(cpp_result.get("debr_refine_calls", 0))
            self.debr_refine_improved_ = int(cpp_result.get("debr_refine_improved", 0))
            self.debr_total_moves_ = int(cpp_result.get("debr_total_moves", 0))
            self.debr_bridge_policy_calls_ = int(cpp_result.get("debr_bridge_policy_calls", 0))
            self.debr_refine_windowed_calls_ = int(
                cpp_result.get("debr_refine_windowed_calls", 0)
            )
            self.debr_refine_unwindowed_calls_ = int(
                cpp_result.get("debr_refine_unwindowed_calls", 0)
            )
            self.debr_refine_overlap_segments_ = int(
                cpp_result.get("debr_refine_overlap_segments", 0)
            )
            self.debr_refine_calls_with_overlap_ = int(
                cpp_result.get("debr_refine_calls_with_overlap", 0)
            )
            self.debr_refine_calls_without_overlap_ = int(
                cpp_result.get("debr_refine_calls_without_overlap", 0)
            )
            self.debr_candidate_total_ = int(cpp_result.get("debr_candidate_total", 0))
            self.debr_candidate_legal_ = int(cpp_result.get("debr_candidate_legal", 0))
            self.debr_candidate_source_size_rejects_ = int(
                cpp_result.get("debr_candidate_source_size_rejects", 0)
            )
            self.debr_candidate_target_size_rejects_ = int(
                cpp_result.get("debr_candidate_target_size_rejects", 0)
            )
            self.debr_candidate_descent_eligible_ = int(
                cpp_result.get("debr_candidate_descent_eligible", 0)
            )
            self.debr_candidate_descent_rejected_ = int(
                cpp_result.get("debr_candidate_descent_rejected", 0)
            )
            self.debr_candidate_bridge_eligible_ = int(
                cpp_result.get("debr_candidate_bridge_eligible", 0)
            )
            self.debr_candidate_bridge_window_blocked_ = int(
                cpp_result.get("debr_candidate_bridge_window_blocked", 0)
            )
            self.debr_candidate_bridge_used_blocked_ = int(
                cpp_result.get("debr_candidate_bridge_used_blocked", 0)
            )
            self.debr_candidate_bridge_guide_rejected_ = int(
                cpp_result.get("debr_candidate_bridge_guide_rejected", 0)
            )
            self.debr_candidate_cleanup_eligible_ = int(
                cpp_result.get("debr_candidate_cleanup_eligible", 0)
            )
            self.debr_candidate_cleanup_primary_rejected_ = int(
                cpp_result.get("debr_candidate_cleanup_primary_rejected", 0)
            )
            self.debr_candidate_cleanup_complexity_rejected_ = int(
                cpp_result.get("debr_candidate_cleanup_complexity_rejected", 0)
            )
            self.debr_candidate_score_rejected_ = int(
                cpp_result.get("debr_candidate_score_rejected", 0)
            )
            self.debr_descent_moves_ = int(cpp_result.get("debr_descent_moves", 0))
            self.debr_bridge_moves_ = int(cpp_result.get("debr_bridge_moves", 0))
            self.debr_simplify_moves_ = int(cpp_result.get("debr_simplify_moves", 0))
            self.debr_source_group_row_size_histogram_ = list(
                cpp_result.get("debr_source_group_row_size_histogram", [])
            )
            self.debr_source_component_atom_size_histogram_ = list(
                cpp_result.get("debr_source_component_atom_size_histogram", [])
            )
            self.debr_source_component_row_size_histogram_ = list(
                cpp_result.get("debr_source_component_row_size_histogram", [])
            )
            self.debr_total_hard_gain_ = float(cpp_result.get("debr_total_hard_gain", 0.0))
            self.debr_total_soft_gain_ = float(cpp_result.get("debr_total_soft_gain", 0.0))
            self.debr_total_delta_j_ = float(cpp_result.get("debr_total_delta_j", 0.0))
            self.debr_total_component_delta_ = int(cpp_result.get("debr_total_component_delta", 0))
            self.debr_final_geo_wins_ = int(cpp_result.get("debr_final_geo_wins", 0))
            self.debr_final_block_wins_ = int(cpp_result.get("debr_final_block_wins", 0))
            self.native_n_classes_ = int(cpp_result.get("native_n_classes", len(class_labels)))
            self.native_teacher_class_count_ = int(cpp_result.get("native_teacher_class_count", 0))
            self.native_binary_mode_ = int(cpp_result.get("native_binary_mode", int(len(class_labels) == 2)))
            self.atomized_features_prepared_ = int(cpp_result.get("atomized_features_prepared", 0))
            self.atomized_coarse_candidates_ = int(cpp_result.get("atomized_coarse_candidates", 0))
            self.atomized_final_candidates_ = int(cpp_result.get("atomized_final_candidates", 0))
            self.atomized_coarse_pruned_candidates_ = int(
                cpp_result.get("atomized_coarse_pruned_candidates", 0)
            )
            self.greedy_feature_survivor_histogram_ = list(
                cpp_result.get("greedy_feature_survivor_histogram", [])
            )
            self.nominee_unique_total_ = int(cpp_result.get("nominee_unique_total", 0))
            self.nominee_child_interval_lookups_ = int(
                cpp_result.get("nominee_child_interval_lookups", 0)
            )
            self.nominee_child_interval_unique_ = int(
                cpp_result.get("nominee_child_interval_unique", 0)
            )
            self.nominee_exactified_total_ = int(cpp_result.get("nominee_exactified_total", 0))
            self.nominee_incumbent_updates_ = int(cpp_result.get("nominee_incumbent_updates", 0))
            self.nominee_threatening_samples_ = int(cpp_result.get("nominee_threatening_samples", 0))
            self.nominee_threatening_sum_ = float(cpp_result.get("nominee_threatening_sum", 0.0))
            self.nominee_threatening_max_ = int(cpp_result.get("nominee_threatening_max", 0))
            self.atomized_feature_atom_count_histogram_ = list(
                cpp_result.get("atomized_feature_atom_count_histogram", [])
            )
            self.atomized_feature_block_atom_count_histogram_ = list(
                cpp_result.get("atomized_feature_block_atom_count_histogram", [])
            )
            self.atomized_feature_q_effective_histogram_ = list(
                cpp_result.get("atomized_feature_q_effective_histogram", [])
            )
            self.greedy_feature_preserved_histogram_ = list(
                cpp_result.get("greedy_feature_preserved_histogram", [])
            )
            self.greedy_candidate_count_histogram_ = list(
                cpp_result.get("greedy_candidate_count_histogram", [])
            )
            self.per_node_prepared_features_ = list(
                cpp_result.get(
                    "per_node_prepared_features",
                    self.greedy_feature_preserved_histogram_,
                )
            )
            self.per_node_candidate_count_ = list(
                cpp_result.get(
                    "per_node_candidate_count",
                    self.greedy_candidate_count_histogram_,
                )
            )
            self.per_node_total_weight_ = list(cpp_result.get("per_node_total_weight", []))
            self.per_node_mu_node_ = list(cpp_result.get("per_node_mu_node", []))
            self.per_node_candidate_upper_bounds_ = list(
                cpp_result.get("per_node_candidate_upper_bounds", [])
            )
            self.per_node_candidate_lower_bounds_ = list(
                cpp_result.get("per_node_candidate_lower_bounds", [])
            )
            self.nominee_elbow_prefix_total_ = int(cpp_result.get("nominee_elbow_prefix_total", 0))
            self.nominee_elbow_prefix_max_ = int(cpp_result.get("nominee_elbow_prefix_max", 0))
            self.nominee_elbow_prefix_histogram_ = list(
                cpp_result.get("nominee_elbow_prefix_histogram", [])
            )
            self.profiling_lp_solve_calls_ = int(cpp_result.get("profiling_lp_solve_calls", 0))
            self.profiling_lp_solve_sec_ = float(cpp_result.get("profiling_lp_solve_sec", 0.0))
            self.profiling_pricing_calls_ = int(cpp_result.get("profiling_pricing_calls", 0))
            self.profiling_pricing_sec_ = float(cpp_result.get("profiling_pricing_sec", 0.0))
            self.profiling_greedy_complete_calls_ = int(
                cpp_result.get("profiling_greedy_complete_calls", 0)
            )
            self.profiling_greedy_complete_sec_ = float(
                cpp_result.get("profiling_greedy_complete_sec", 0.0)
            )
            self.profiling_greedy_complete_calls_by_depth_ = list(
                cpp_result.get("profiling_greedy_complete_calls_by_depth", [])
            )
            self.profiling_feature_prepare_sec_ = float(
                cpp_result.get("profiling_feature_prepare_sec", 0.0)
            )
            self.profiling_candidate_nomination_sec_ = float(
                cpp_result.get("profiling_candidate_nomination_sec", 0.0)
            )
            self.profiling_candidate_shortlist_sec_ = float(
                cpp_result.get("profiling_candidate_shortlist_sec", 0.0)
            )
            self.profiling_candidate_generation_sec_ = float(
                cpp_result.get("profiling_candidate_generation_sec", 0.0)
            )
            self.profiling_recursive_child_eval_sec_ = float(
                cpp_result.get("profiling_recursive_child_eval_sec", 0.0)
            )
            self.heuristic_selector_nodes_ = int(cpp_result.get("heuristic_selector_nodes", 0))
            self.heuristic_selector_candidate_total_ = int(
                cpp_result.get("heuristic_selector_candidate_total", 0)
            )
            self.heuristic_selector_candidate_pruned_total_ = int(
                cpp_result.get("heuristic_selector_candidate_pruned_total", 0)
            )
            self.heuristic_selector_survivor_total_ = int(
                cpp_result.get("heuristic_selector_survivor_total", 0)
            )
            self.heuristic_selector_leaf_optimal_nodes_ = int(
                cpp_result.get("heuristic_selector_leaf_optimal_nodes", 0)
            )
            self.heuristic_selector_improving_split_nodes_ = int(
                cpp_result.get("heuristic_selector_improving_split_nodes", 0)
            )
            self.heuristic_selector_improving_split_retained_nodes_ = int(
                cpp_result.get("heuristic_selector_improving_split_retained_nodes", 0)
            )
            self.heuristic_selector_improving_split_margin_sum_ = float(
                cpp_result.get("heuristic_selector_improving_split_margin_sum", 0.0)
            )
            self.heuristic_selector_improving_split_margin_max_ = float(
                cpp_result.get("heuristic_selector_improving_split_margin_max", 0.0)
            )
            self.profiling_refine_calls_ = int(cpp_result.get("profiling_refine_calls", 0))
            self.profiling_refine_sec_ = float(cpp_result.get("profiling_refine_sec", 0.0))
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
            self.exact_dp_subproblem_calls_above_lookahead_ = int(
                cpp_result.get("exact_dp_subproblem_calls_above_lookahead", 0)
            )
            self.greedy_cache_hits_ = int(cpp_result.get("greedy_cache_hits", 0))
            self.greedy_unique_states_ = int(cpp_result.get("greedy_unique_states", 0))
            self.greedy_cache_entries_peak_ = int(cpp_result.get("greedy_cache_entries_peak", 0))
            self.greedy_cache_clears_ = int(cpp_result.get("greedy_cache_clears", 0))
            self.dp_interval_evals_ = int(cpp_result.get("dp_interval_evals", 0))
            self.greedy_interval_evals_ = int(cpp_result.get("greedy_interval_evals", 0))
            self.prism_rollout_calls_ = int(cpp_result.get("prism_rollout_calls", 0))
            self.prism_exact_calls_ = int(cpp_result.get("prism_exact_calls", 0))
            self.prism_rollout_candidate_evals_ = int(
                cpp_result.get("prism_rollout_candidate_evals", 0)
            )
            self.prism_exact_candidate_evals_ = int(
                cpp_result.get("prism_exact_candidate_evals", 0)
            )
            self.prism_shortlist_states_ = int(cpp_result.get("prism_shortlist_states", 0))
            self.prism_shortlist_candidates_ = int(
                cpp_result.get("prism_shortlist_candidates", 0)
            )
            self.prism_rollout_cache_hits_ = int(cpp_result.get("prism_rollout_cache_hits", 0))
            self.prism_rollout_cache_entries_peak_ = int(
                cpp_result.get("prism_rollout_cache_entries_peak", 0)
            )
            self.prism_rollout_cache_bytes_peak_ = int(
                cpp_result.get("prism_rollout_cache_bytes_peak", 0)
            )
            self.prism_exact_cache_hits_ = int(cpp_result.get("prism_exact_cache_hits", 0))
            self.prism_exact_cache_entries_peak_ = int(
                cpp_result.get("prism_exact_cache_entries_peak", 0)
            )
            self.prism_exact_cache_bytes_peak_ = int(
                cpp_result.get("prism_exact_cache_bytes_peak", 0)
            )
            self.prism_rollout_sec_ = float(cpp_result.get("prism_rollout_sec", 0.0))
            self.prism_exact_sec_ = float(cpp_result.get("prism_exact_sec", 0.0))
            self.prism_child_state_profiled_nodes_ = int(
                cpp_result.get("prism_child_state_profiled_nodes", 0)
            )
            self.prism_child_state_edges_ = int(cpp_result.get("prism_child_state_edges", 0))
            self.prism_child_state_unique_ = int(cpp_result.get("prism_child_state_unique", 0))
            self.prism_child_state_duplicate_edges_ = int(
                cpp_result.get("prism_child_state_duplicate_edges", 0)
            )
            self.prism_child_state_multiuse_states_ = int(
                cpp_result.get("prism_child_state_multiuse_states", 0)
            )
            self.prism_child_state_max_uses_peak_ = int(
                cpp_result.get("prism_child_state_max_uses_peak", 0)
            )
            self.prism_child_state_use_histogram_ = list(
                cpp_result.get("prism_child_state_use_histogram", [])
            )
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
            self.exact_dp_subproblem_calls_above_lookahead_ = 0
            self.greedy_subproblem_calls_ = int(len(self._greedy_cache))
            self.greedy_cache_hits_ = 0
            self.greedy_unique_states_ = int(len(self._greedy_cache))
            self.greedy_cache_entries_peak_ = int(len(self._greedy_cache))
            self.greedy_cache_clears_ = 0
            self.dp_interval_evals_ = 0
            self.greedy_interval_evals_ = 0
            self.prism_rollout_calls_ = 0
            self.prism_exact_calls_ = 0
            self.prism_rollout_candidate_evals_ = 0
            self.prism_exact_candidate_evals_ = 0
            self.prism_shortlist_states_ = 0
            self.prism_shortlist_candidates_ = 0
            self.prism_rollout_cache_hits_ = 0
            self.prism_rollout_cache_entries_peak_ = 0
            self.prism_rollout_cache_bytes_peak_ = 0
            self.prism_exact_cache_hits_ = 0
            self.prism_exact_cache_entries_peak_ = 0
            self.prism_exact_cache_bytes_peak_ = 0
            self.prism_rollout_sec_ = 0.0
            self.prism_exact_sec_ = 0.0
            self.prism_child_state_profiled_nodes_ = 0
            self.prism_child_state_edges_ = 0
            self.prism_child_state_unique_ = 0
            self.prism_child_state_duplicate_edges_ = 0
            self.prism_child_state_multiuse_states_ = 0
            self.prism_child_state_max_uses_peak_ = 0
            self.prism_child_state_use_histogram_ = []
            self.profiling_lp_solve_calls_ = 0
            self.profiling_lp_solve_sec_ = 0.0
            self.profiling_pricing_calls_ = 0
            self.profiling_pricing_sec_ = 0.0
            self.profiling_greedy_complete_calls_ = 0
            self.profiling_greedy_complete_sec_ = 0.0
            self.profiling_greedy_complete_calls_by_depth_ = []
            self.profiling_feature_prepare_sec_ = 0.0
            self.profiling_candidate_nomination_sec_ = 0.0
            self.profiling_candidate_shortlist_sec_ = 0.0
            self.profiling_candidate_generation_sec_ = 0.0
            self.profiling_recursive_child_eval_sec_ = 0.0
            self.heuristic_selector_nodes_ = 0
            self.heuristic_selector_candidate_total_ = 0
            self.heuristic_selector_candidate_pruned_total_ = 0
            self.heuristic_selector_survivor_total_ = 0
            self.heuristic_selector_leaf_optimal_nodes_ = 0
            self.heuristic_selector_improving_split_nodes_ = 0
            self.heuristic_selector_improving_split_retained_nodes_ = 0
            self.heuristic_selector_improving_split_margin_sum_ = 0.0
            self.heuristic_selector_improving_split_margin_max_ = 0.0
            self.profiling_refine_calls_ = 0
            self.profiling_refine_sec_ = 0.0
            self.nominee_unique_total_ = 0
            self.nominee_child_interval_lookups_ = 0
            self.nominee_child_interval_unique_ = 0
            self.nominee_exactified_total_ = 0
            self.nominee_incumbent_updates_ = 0
            self.nominee_threatening_samples_ = 0
            self.nominee_threatening_sum_ = 0.0
            self.nominee_threatening_max_ = 0
            self.atomized_feature_atom_count_histogram_ = []
            self.atomized_feature_block_atom_count_histogram_ = []
            self.atomized_feature_q_effective_histogram_ = []
            self.greedy_feature_preserved_histogram_ = []
            self.greedy_candidate_count_histogram_ = []
            self.greedy_feature_survivor_histogram_ = []
            self.per_node_prepared_features_ = []
            self.per_node_candidate_count_ = []
            self.per_node_total_weight_ = []
            self.per_node_mu_node_ = []
            self.per_node_candidate_upper_bounds_ = []
            self.per_node_candidate_lower_bounds_ = []
            self.nominee_elbow_prefix_total_ = 0
            self.nominee_elbow_prefix_max_ = 0
            self.nominee_elbow_prefix_histogram_ = []
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
        Z = check_array(X, ensure_2d=True, dtype=None)
        if not np.issubdtype(np.asarray(Z).dtype, np.integer):
            raise ValueError("MSPLIT expects LightGBM-binned integer features")
        Z = np.asarray(Z, dtype=np.int32)
        if (Z < 0).any():
            raise ValueError("Binned input must be non-negative integer values")
        preds = np.zeros(Z.shape[0], dtype=np.int32)
        for i in range(Z.shape[0]):
            preds[i] = self._predict_row(Z[i], self.tree_)
        return self.classes_[preds]

    def _predict_row(self, row: np.ndarray, node: Union[MultiNode, MultiLeaf]) -> int:
        cur = node
        while isinstance(cur, MultiNode):
            feature_index = self._resolve_feature_index(cur.feature, row.shape[0])
            bin_id = int(row[feature_index])
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

    def _resolve_feature_index(self, feature_index: int, row_width: int) -> int:
        feature_index = int(feature_index)
        row_width = int(row_width)
        if row_width <= 0:
            raise ValueError("Cannot predict on an empty feature row")
        if 0 <= feature_index < row_width:
            return feature_index
        if feature_index == row_width:
            warnings.warn(
                f"MSPLIT tree feature index {feature_index} matched the row width; "
                f"clamping to the last available feature {row_width - 1}.",
                RuntimeWarning,
                stacklevel=3,
            )
            return row_width - 1
        raise IndexError(
            f"MSPLIT tree feature index {feature_index} is out of bounds for input width {row_width}"
        )

    def _solve_subproblem(self, indices: np.ndarray, depth_remaining: int, current_depth: int) -> BoundResult:
        self._check_timeout()
        if current_depth < self.effective_lookahead_depth_:
            self.exact_dp_subproblem_calls_above_lookahead_ += 1
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
