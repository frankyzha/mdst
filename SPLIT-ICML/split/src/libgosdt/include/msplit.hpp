#ifndef MSPLIT_H
#define MSPLIT_H

#include <vector>

#include "nlohmann/json.hpp"

namespace msplit {

struct FitResult {
    nlohmann::json tree;
    double lowerbound = 0.0;
    double upperbound = 0.0;
    double objective = 0.0;
    int exact_internal_nodes = 0;
    int greedy_internal_nodes = 0;
    long long dp_subproblem_calls = 0;
    long long dp_cache_hits = 0;
    long long dp_unique_states = 0;
    int dp_cache_profile_enabled = 0;
    long long dp_cache_lookup_calls = 0;
    long long dp_cache_miss_no_bucket = 0;
    long long dp_cache_miss_bucket_present = 0;
    long long dp_cache_miss_depth_mismatch_only = 0;
    long long dp_cache_miss_indices_mismatch = 0;
    long long dp_cache_depth_match_candidates = 0;
    long long dp_cache_bucket_entries_scanned = 0;
    long long dp_cache_bucket_max_size = 0;
    long long greedy_subproblem_calls = 0;
    long long greedy_cache_hits = 0;
    long long greedy_unique_states = 0;
    long long greedy_cache_entries_peak = 0;
    long long greedy_cache_clears = 0;
    long long dp_interval_evals = 0;
    long long greedy_interval_evals = 0;
    long long rush_incumbent_feature_aborts = 0;
    double rush_total_time_sec = 0.0;
    long long rush_refinement_child_calls = 0;
    long long rush_refinement_recursive_calls = 0;
    long long rush_refinement_recursive_unique_states = 0;
    long long rush_ub_rescue_picks = 0;
    long long rush_global_fallback_picks = 0;
    int rush_profile_enabled = 0;
    double rush_profile_ub0_ordering_sec = 0.0;
    double rush_profile_exact_lazy_eval_sec = 0.0;
    double rush_profile_exact_lazy_eval_exclusive_sec = 0.0;
    double rush_profile_exact_lazy_eval_sec_depth0 = 0.0;
    double rush_profile_exact_lazy_eval_exclusive_sec_depth0 = 0.0;
    double rush_profile_exact_lazy_table_init_sec = 0.0;
    double rush_profile_exact_lazy_dp_recompute_sec = 0.0;
    double rush_profile_exact_lazy_child_solve_sec = 0.0;
    double rush_profile_exact_lazy_child_solve_sec_depth0 = 0.0;
    double rush_profile_exact_lazy_closure_sec = 0.0;
    long long rush_profile_exact_lazy_dp_recompute_calls = 0;
    long long rush_profile_exact_lazy_closure_passes = 0;
    long long interval_refinements_attempted = 0;
    long long expensive_child_calls = 0;
    double expensive_child_sec = 0.0;
    long long expensive_child_exactify_calls = 0;
    double expensive_child_exactify_sec = 0.0;
    int approx_mode_enabled = 0;
    int approx_ref_shortlist_enabled = 0;
    int approx_challenger_sweep_enabled = 0;
    int approx_lhat_computed = 0;
    long long approx_greedy_patch_calls = 0;
    long long approx_greedy_patches_applied = 0;
    long long approx_greedy_ub_updates_total = 0;
    double approx_greedy_patch_sec = 0.0;
    long long approx_exactify_triggered_nodes = 0;
    long long approx_exactify_features_exact_solved = 0;
    long long approx_exactify_stops_by_separation = 0;
    long long approx_exactify_stops_by_cap = 0;
    long long approx_exactify_stops_by_ambiguous_empty = 0;
    long long approx_exactify_stops_by_no_improve = 0;
    long long approx_exactify_stops_by_separation_depth0 = 0;
    long long approx_exactify_stops_by_separation_depth1 = 0;
    long long approx_exactify_stops_by_cap_depth0 = 0;
    long long approx_exactify_stops_by_cap_depth1 = 0;
    long long approx_exactify_features_exact_solved_depth0 = 0;
    long long approx_exactify_features_exact_solved_depth1 = 0;
    int approx_exactify_set_size_depth0_min = 0;
    double approx_exactify_set_size_depth0_mean = 0.0;
    int approx_exactify_set_size_depth0_max = 0;
    int approx_exactify_set_size_depth1_min = 0;
    double approx_exactify_set_size_depth1_mean = 0.0;
    int approx_exactify_set_size_depth1_max = 0;
    double approx_exactify_avg_features_per_triggered_node = 0.0;
    double approx_exactify_ambiguous_set_size_min = 0.0;
    double approx_exactify_ambiguous_set_size_mean = 0.0;
    long long approx_exactify_ambiguous_set_size_max = 0;
    long long approx_exactify_ambiguous_set_shrank_steps = 0;
    double approx_exactify_cap_effective_depth0 = 0.0;
    double approx_exactify_cap_effective_depth1 = 0.0;
    long long approx_challenger_sweep_invocations = 0;
    long long approx_challenger_sweep_features_processed = 0;
    double approx_challenger_sweep_sec = 0.0;
    long long approx_challenger_sweep_skipped_large_ambiguous = 0;
    long long approx_challenger_sweep_patch_cap_hit = 0;
    long long approx_uncertainty_triggered_nodes = 0;
    double approx_exactify_trigger_rate_depth0 = 0.0;
    double approx_exactify_trigger_rate_depth1 = 0.0;
    double approx_uncertainty_trigger_rate_depth0 = 0.0;
    double approx_uncertainty_trigger_rate_depth1 = 0.0;
    long long approx_eligible_nodes_depth0 = 0;
    long long approx_eligible_nodes_depth1 = 0;
    long long approx_exactify_triggered_nodes_depth0 = 0;
    long long approx_exactify_triggered_nodes_depth1 = 0;
    long long approx_uncertainty_triggered_nodes_depth0 = 0;
    long long approx_uncertainty_triggered_nodes_depth1 = 0;
    long long approx_pub_unrefined_cells_on_pub_total = 0;
    long long approx_pub_patchable_cells_total = 0;
    long long approx_pub_cells_skipped_by_childrows = 0;
    long long approx_nodes_with_patchable_pub = 0;
    long long approx_nodes_with_patch_calls = 0;
    long long approx_patch_cell_cache_hits = 0;
    long long approx_patch_cell_cache_misses = 0;
    long long approx_patch_cache_hit_updates = 0;
    long long approx_patch_cache_miss_oracle_calls = 0;
    long long approx_patch_subset_materializations = 0;
    long long approx_patch_skipped_already_tight = 0;
    long long approx_patch_skipped_no_possible_improve = 0;
    long long approx_patch_skipped_cached = 0;
    int approx_patch_budget_effective_min = 0;
    double approx_patch_budget_effective_avg = 0.0;
    int approx_patch_budget_effective_max = 0;
    double approx_ref_neff_mean = 0.0;
    double approx_ref_neff_max = 0.0;
    int approx_ref_k0_min = 0;
    double approx_ref_k0_mean = 0.0;
    int approx_ref_k0_max = 0;
    int approx_ref_k_final_min = 0;
    double approx_ref_k_final_mean = 0.0;
    int approx_ref_k_final_max = 0;
    double approx_ref_k_depth0_mean = 0.0;
    double approx_ref_k_depth1_mean = 0.0;
    long long approx_ref_widen_count = 0;
    long long approx_ref_widen_count_depth0 = 0;
    long long approx_ref_widen_count_depth1 = 0;
    double approx_ref_chosen_feature_rank_depth0 = 0.0;
    double approx_ref_chosen_feature_rank_depth1 = 0.0;
    double approx_ref_chosen_in_initial_shortlist_rate_depth0 = 0.0;
    double approx_ref_chosen_in_initial_shortlist_rate_depth1 = 0.0;
    long long fast100_exactify_nodes_allowed = 0;
    long long fast100_exactify_nodes_skipped_small_support = 0;
    long long fast100_exactify_nodes_skipped_dominant_gain = 0;
    long long depth1_skipped_by_low_global_ambiguity = 0;
    long long depth1_skipped_by_large_gap = 0;
    long long depth1_exactify_challenger_nodes = 0;
    long long depth1_exactified_nodes = 0;
    double depth1_exactified_features_mean = 0.0;
    int depth1_exactified_features_max = 0;
    long long depth1_teacher_replaced_runnerup = 0;
    long long depth1_teacher_rejected_by_uhat_gate = 0;
    double depth1_exactify_set_size_mean = 0.0;
    int depth1_exactify_set_size_max = 0;
    long long fast100_skipped_by_ub_lb_separation = 0;
    long long fast100_widen_forbidden_depth_gt0_attempts = 0;
    double fast100_frontier_size_mean = 0.0;
    int fast100_frontier_size_max = 0;
    long long fast100_stopped_midloop_separation = 0;
    double fast100_M_depth0_mean = 0.0;
    int fast100_M_depth0_max = 0;
    double fast100_M_depth1_mean = 0.0;
    int fast100_M_depth1_max = 0;
    long long fast100_cf_exactify_nodes_depth0 = 0;
    long long fast100_cf_exactify_nodes_depth1 = 0;
    long long fast100_cf_skipped_agreement = 0;
    long long fast100_cf_skipped_small_regret = 0;
    long long fast100_cf_skipped_low_impact = 0;
    double fast100_cf_frontier_size_mean = 0.0;
    int fast100_cf_frontier_size_max = 0;
    double fast100_cf_exactified_features_mean = 0.0;
    int fast100_cf_exactified_features_max = 0;
    int rootsafe_exactified_features = 0;
    int rootsafe_root_winner_changed_vs_proxy = 0;
    int rootsafe_root_candidates_K = 0;
    int fast100_used_lgb_prior_tiebreak = 0;
    long long gini_dp_calls_root = 0;
    long long gini_dp_calls_depth1 = 0;
    long long gini_teacher_chosen_depth1 = 0;
    long long gini_tiebreak_used_in_shortlist = 0;
    double gini_dp_sec = 0.0;
    int gini_root_k0 = 0;
    long long gini_endpoints_added_root = 0;
    long long gini_endpoints_added_depth1 = 0;
    long long gini_endpoints_features_touched_root = 0;
    long long gini_endpoints_features_touched_depth1 = 0;
    int gini_endpoints_added_per_feature_max = 0;
    double gini_endpoint_sec = 0.0;
};

FitResult fit(
    const std::vector<int> &x_flat,
    int n_rows,
    int n_features,
    const std::vector<int> &y,
    const std::vector<double> &sample_weight,
    int full_depth_budget,
    int lookahead_depth_budget,
    double regularization,
    double branch_penalty,
    int min_child_size,
    double time_limit_seconds,
    int max_branching,
    int partition_strategy = 0,
    bool approx_mode = false,
    int patch_budget_per_feature = 12,
    int exactify_top_m = 2,
    int tau_mode = 1,
    int approx_feature_scan_limit = 0,
    bool approx_ref_shortlist_enabled = true,
    int approx_ref_widen_max = 1,
    bool approx_challenger_sweep_enabled = false,
    int approx_challenger_sweep_max_features = 3,
    int approx_challenger_sweep_max_patch_calls_per_node = 0
);

}  // namespace msplit

#endif
