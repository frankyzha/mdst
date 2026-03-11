#include "msplit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace msplit {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kEpsCert = 1e-9;
constexpr double kEpsUpdate = 1e-12;

struct Node {
    bool is_leaf = true;

    int prediction = 0;
    double loss = 0.0;
    int n_samples = 0;
    int neg_count = 0;
    int pos_count = 0;

    int feature = -1;
    int fallback_bin = -1;
    int fallback_prediction = 0;
    int group_count = 0;
    // One child per logical group; each group stores compressed observed-bin spans.
    std::vector<std::vector<std::pair<int, int>>> group_bin_spans;
    std::vector<std::shared_ptr<Node>> group_nodes;
};

struct BoundResult {
    double lb = 0.0;
    double lb_mis = 0.0;
    double ub = 0.0;
    std::shared_ptr<Node> tree;
};

struct GreedyResult {
    double objective = 0.0;
    std::shared_ptr<Node> tree;
};

struct OrderedBins {
    std::vector<int> values;
    std::vector<std::vector<int>> members;
    std::vector<int> prefix_counts;
};

struct PartitionResult {
    bool feasible = false;
    double cost = kInfinity;
    int groups = 0;
    std::vector<std::pair<int, int>> intervals;
};

struct ProjectedPartitionResult {
    bool feasible = false;
    double cost = kInfinity;
    int groups = 0;
    // Each interval is [p, t) over projected endpoint indices.
    std::vector<std::pair<int, int>> intervals;
};

enum PartitionStrategy : int {
    kPartitionOptimalDp = 0,
    kPartitionRushDp = 1,
};

enum TauMode : int {
    kTauLambda = 0,
    kTauLambdaSqrtR = 1,
};

class Solver {
   public:
    Solver(
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
        int partition_strategy,
        bool approx_mode,
        int patch_budget_per_feature,
        int exactify_top_m,
        int tau_mode,
        int approx_feature_scan_limit,
        bool approx_ref_shortlist_enabled,
        int approx_ref_widen_max,
        bool approx_challenger_sweep_enabled,
        int approx_challenger_sweep_max_features,
        int approx_challenger_sweep_max_patch_calls_per_node
    )
        : x_flat_(x_flat),
          n_rows_(n_rows),
          n_features_(n_features),
          y_(y),
          sample_weight_raw_(sample_weight),
          full_depth_budget_(full_depth_budget),
          lookahead_depth_budget_(lookahead_depth_budget),
          regularization_(regularization),
          branch_penalty_(branch_penalty),
          min_child_size_(min_child_size),
          time_limit_seconds_(time_limit_seconds),
          max_branching_(max_branching),
          partition_strategy_(partition_strategy),
          approx_mode_(approx_mode),
          patch_budget_per_feature_(patch_budget_per_feature),
          exactify_top_m_(exactify_top_m),
          tau_mode_(tau_mode),
          approx_feature_scan_limit_(approx_feature_scan_limit),
          approx_ref_shortlist_enabled_(approx_ref_shortlist_enabled),
          approx_ref_widen_max_(approx_ref_widen_max),
          approx_challenger_sweep_enabled_(approx_challenger_sweep_enabled),
          approx_challenger_sweep_max_features_(approx_challenger_sweep_max_features),
          approx_challenger_sweep_max_patch_calls_per_node_(approx_challenger_sweep_max_patch_calls_per_node),
          start_time_(Clock::now()) {
        if (n_rows_ <= 0 || n_features_ <= 0) {
            throw std::invalid_argument("MSPLIT requires a non-empty matrix with at least one feature.");
        }
        if ((int)y_.size() != n_rows_) {
            throw std::invalid_argument("MSPLIT y length must match number of rows in X.");
        }
        if ((int)x_flat_.size() != n_rows_ * n_features_) {
            throw std::invalid_argument("MSPLIT x_flat size does not match n_rows * n_features.");
        }
        if (full_depth_budget_ < 1) {
            throw std::invalid_argument("MSPLIT full_depth_budget must be at least 1.");
        }
        if (lookahead_depth_budget_ < 1) {
            throw std::invalid_argument("MSPLIT lookahead_depth_budget must be at least 1.");
        }
        if (regularization_ < 0.0) {
            throw std::invalid_argument("MSPLIT regularization must be non-negative.");
        }
        if (branch_penalty_ < 0.0) {
            throw std::invalid_argument("MSPLIT branch_penalty must be non-negative.");
        }
        if (min_child_size_ < 1) {
            throw std::invalid_argument("MSPLIT min_child_size must be at least 1.");
        }
        if (max_branching_ < 0) {
            throw std::invalid_argument("MSPLIT max_branching must be >= 0 (0 means unlimited).");
        }
        if (partition_strategy_ != kPartitionOptimalDp && partition_strategy_ != kPartitionRushDp) {
            throw std::invalid_argument("MSPLIT partition_strategy must be 0 (optimal_dp) or 1 (rush_dp).");
        }
        if (patch_budget_per_feature_ < 0) {
            throw std::invalid_argument("MSPLIT patch_budget_per_feature must be >= 0.");
        }
        if (exactify_top_m_ < 0) {
            throw std::invalid_argument("MSPLIT exactify_top_m must be >= 0.");
        }
        if (tau_mode_ != kTauLambda && tau_mode_ != kTauLambdaSqrtR) {
            throw std::invalid_argument("MSPLIT tau_mode must be 0 (lambda) or 1 (lambda_sqrt_r).");
        }
        if (approx_feature_scan_limit_ < 0) {
            throw std::invalid_argument("MSPLIT approx_feature_scan_limit must be >= 0.");
        }
        if (approx_ref_widen_max_ < 0) {
            throw std::invalid_argument("MSPLIT approx_ref_widen_max must be >= 0.");
        }
        if (approx_challenger_sweep_max_features_ < 1) {
            throw std::invalid_argument("MSPLIT approx_challenger_sweep_max_features must be >= 1.");
        }
        if (approx_challenger_sweep_max_patch_calls_per_node_ < 0) {
            throw std::invalid_argument(
                "MSPLIT approx_challenger_sweep_max_patch_calls_per_node must be >= 0.");
        }
        if (!sample_weight_raw_.empty() && (int)sample_weight_raw_.size() != n_rows_) {
            throw std::invalid_argument("MSPLIT sample_weight must have length n_rows when provided.");
        }
        if (const char *raw = std::getenv("MSPLIT_FORCE_RUSH_LEGACY")) {
            const std::string value(raw);
            if (!value.empty() && value != "0" && value != "false" && value != "False") {
                force_rush_legacy_ = true;
            }
        }
        if (const char *raw = std::getenv("MSPLIT_RUSH_PROFILE")) {
            const std::string value(raw);
            if (!value.empty() && value != "0" && value != "false" && value != "False") {
                rush_profile_enabled_ = true;
            }
        }
        if (const char *raw = std::getenv("MSPLIT_DP_CACHE_PROFILE")) {
            const std::string value(raw);
            if (!value.empty() && value != "0" && value != "false" && value != "False") {
                dp_cache_profile_enabled_ = true;
            }
        }
        if (const char *raw = std::getenv("MSPLIT_GREEDY_CACHE_MAX_ENTRIES_FOR_APPROX")) {
            try {
                const long long parsed = std::stoll(std::string(raw));
                if (parsed > 0) {
                    greedy_cache_max_entries_for_approx_ = parsed;
                }
            } catch (const std::exception &) {
                // Ignore invalid overrides and keep the default guardrail.
            }
        }
        if (const char *raw = std::getenv("MSPLIT_FAST100_DEBUG_DEPTH1_NODES")) {
            try {
                fast100_debug_depth1_limit_ = std::max(0, std::stoi(std::string(raw)));
            } catch (const std::exception &) {
                fast100_debug_depth1_limit_ = 0;
            }
        }
        if (const char *raw = std::getenv("MSPLIT_ENABLE_GINI_SCOUT")) {
            const std::string value(raw);
            if (!value.empty() && value != "0" && value != "false" && value != "False") {
                gini_scout_enabled_ = true;
            }
        }
        effective_lookahead_ = std::min(full_depth_budget_, lookahead_depth_budget_);
        initialize_weights();
        {
            const double root_weight_sum =
                std::accumulate(sample_weight_.begin(), sample_weight_.end(), 0.0);
            if (root_weight_sum > 0.0 && n_rows_ > 0) {
                one_mistake_unit_ = root_weight_sum / (double)n_rows_;
            } else {
                one_mistake_unit_ = 1.0 / (double)std::max(1, n_rows_);
            }
        }
        if (fast100_debug_depth1_limit_ > 0) {
            const double root_weight_sum =
                std::accumulate(sample_weight_.begin(), sample_weight_.end(), 0.0);
            const int weights_normalized = (std::fabs(root_weight_sum - 1.0) <= 1e-6) ? 1 : 0;
            std::fprintf(
                stderr,
                "[FAST100_DBG_START] n_rows=%d min_child_size=%d lambda=%.12g one_mistake_unit=%.12g "
                "root_weight_sum=%.12g weights_normalized=%d approx_auto=%d approx_ref_widen_max=%d allow_widen=%d\n",
                n_rows_,
                min_child_size_,
                regularization_,
                one_mistake_unit_,
                root_weight_sum,
                weights_normalized,
                approx_gain_mass_auto_mode_enabled() ? 1 : 0,
                approx_ref_widen_max_,
                (approx_ref_widen_max_ > 0) ? 1 : 0);
        }
        preprocess_signatures();
        // Depth-indexed workspaces are referenced across recursive calls in the
        // exact-lazy loop. Pre-size once to avoid vector reallocations that
        // would invalidate references while child subproblems are being solved.
        projected_dp_workspaces_.resize((size_t)full_depth_budget_ + 1U);
        rush_interval_workspaces_.resize((size_t)full_depth_budget_ + 1U);
    }

    FitResult fit() {
        std::vector<int> root_indices(n_rows_);
        for (int i = 0; i < n_rows_; ++i) {
            root_indices[i] = i;
        }

        BoundResult solved = solve_subproblem(root_indices, full_depth_budget_);
        const std::chrono::duration<double> total_elapsed = Clock::now() - start_time_;
        const double total_sec = total_elapsed.count();
#ifndef NDEBUG
        if (approx_greedy_patches_applied_ > approx_greedy_patch_calls_) {
            throw std::runtime_error(
                "MSPLIT approx invariant violated: approx_greedy_patches_applied exceeds approx_greedy_patch_calls.");
        }
#endif

        FitResult out;
        out.tree = to_json(solved.tree);
        out.lowerbound = solved.lb;
        out.upperbound = solved.ub;
        out.objective = solved.ub;
        out.exact_internal_nodes = exact_internal_nodes_;
        out.greedy_internal_nodes = greedy_internal_nodes_;
        out.dp_subproblem_calls = dp_subproblem_calls_;
        out.dp_cache_hits = dp_cache_hits_;
        out.dp_unique_states = dp_cache_states_;
        out.dp_cache_profile_enabled = dp_cache_profile_enabled_ ? 1 : 0;
        out.dp_cache_lookup_calls = dp_cache_lookup_calls_;
        out.dp_cache_miss_no_bucket = dp_cache_miss_no_bucket_;
        out.dp_cache_miss_bucket_present = dp_cache_miss_bucket_present_;
        out.dp_cache_miss_depth_mismatch_only = dp_cache_miss_depth_mismatch_only_;
        out.dp_cache_miss_indices_mismatch = dp_cache_miss_indices_mismatch_;
        out.dp_cache_depth_match_candidates = dp_cache_depth_match_candidates_;
        out.dp_cache_bucket_entries_scanned = dp_cache_bucket_entries_scanned_;
        out.dp_cache_bucket_max_size = dp_cache_bucket_max_size_;
        out.greedy_subproblem_calls = greedy_subproblem_calls_;
        out.greedy_cache_hits = greedy_cache_hits_;
        out.greedy_unique_states = greedy_cache_states_;
        out.greedy_cache_entries_peak = greedy_cache_entries_peak_;
        out.greedy_cache_clears = greedy_cache_clears_;
        out.dp_interval_evals = dp_interval_evals_;
        out.greedy_interval_evals = greedy_interval_evals_;
        out.rush_incumbent_feature_aborts = rush_incumbent_feature_aborts_;
        out.rush_total_time_sec = total_sec;
        out.rush_refinement_child_calls = rush_refinement_child_calls_;
        out.rush_refinement_recursive_calls = rush_refinement_recursive_calls_;
        out.rush_refinement_recursive_unique_states = rush_refinement_recursive_unique_states_;
        out.rush_ub_rescue_picks = rush_ub_rescue_picks_;
        out.rush_global_fallback_picks = rush_global_fallback_picks_;
        out.rush_profile_enabled = rush_profile_enabled_ ? 1 : 0;
        out.rush_profile_ub0_ordering_sec = rush_profile_ub0_ordering_sec_;
        out.rush_profile_exact_lazy_eval_sec = rush_profile_exact_lazy_eval_sec_;
        out.rush_profile_exact_lazy_eval_exclusive_sec = rush_profile_exact_lazy_eval_exclusive_sec_;
        out.rush_profile_exact_lazy_eval_sec_depth0 = rush_profile_exact_lazy_eval_sec_depth0_;
        out.rush_profile_exact_lazy_eval_exclusive_sec_depth0 = rush_profile_exact_lazy_eval_exclusive_sec_depth0_;
        out.rush_profile_exact_lazy_table_init_sec = rush_profile_exact_lazy_table_init_sec_;
        out.rush_profile_exact_lazy_dp_recompute_sec = rush_profile_exact_lazy_dp_recompute_sec_;
        out.rush_profile_exact_lazy_child_solve_sec = rush_profile_exact_lazy_child_solve_sec_;
        out.rush_profile_exact_lazy_child_solve_sec_depth0 = rush_profile_exact_lazy_child_solve_sec_depth0_;
        out.rush_profile_exact_lazy_closure_sec = rush_profile_exact_lazy_closure_sec_;
        out.rush_profile_exact_lazy_dp_recompute_calls = rush_profile_exact_lazy_dp_recompute_calls_;
        out.rush_profile_exact_lazy_closure_passes = rush_profile_exact_lazy_closure_passes_;
        out.interval_refinements_attempted = interval_refinements_attempted_;
        out.expensive_child_calls = expensive_child_calls_;
        out.expensive_child_sec = expensive_child_sec_;
        out.expensive_child_exactify_calls = expensive_child_exactify_calls_;
        out.expensive_child_exactify_sec = expensive_child_exactify_sec_;
        out.approx_mode_enabled = approx_mode_ ? 1 : 0;
        out.approx_ref_shortlist_enabled = approx_ref_shortlist_enabled_ ? 1 : 0;
        out.approx_challenger_sweep_enabled = approx_challenger_sweep_enabled_ ? 1 : 0;
        out.approx_lhat_computed = approx_lhat_computed_ ? 1 : 0;
        out.approx_greedy_patch_calls = approx_greedy_patch_calls_;
        out.approx_greedy_patches_applied = approx_greedy_patches_applied_;
        out.approx_greedy_ub_updates_total = approx_greedy_ub_updates_total_;
        out.approx_greedy_patch_sec = approx_greedy_patch_sec_;
        out.approx_exactify_triggered_nodes = approx_exactify_triggered_nodes_;
        out.approx_exactify_features_exact_solved = approx_exactify_features_exact_solved_;
        out.approx_exactify_stops_by_separation = approx_exactify_stops_by_separation_;
        out.approx_exactify_stops_by_cap = approx_exactify_stops_by_cap_;
        out.approx_exactify_stops_by_ambiguous_empty = approx_exactify_stops_by_ambiguous_empty_;
        out.approx_exactify_stops_by_no_improve = approx_exactify_stops_by_no_improve_;
        out.approx_exactify_stops_by_separation_depth0 = approx_exactify_stops_by_separation_depth0_;
        out.approx_exactify_stops_by_separation_depth1 = approx_exactify_stops_by_separation_depth1_;
        out.approx_exactify_stops_by_cap_depth0 = approx_exactify_stops_by_cap_depth0_;
        out.approx_exactify_stops_by_cap_depth1 = approx_exactify_stops_by_cap_depth1_;
        out.approx_exactify_features_exact_solved_depth0 = approx_exactify_features_exact_solved_depth0_;
        out.approx_exactify_features_exact_solved_depth1 = approx_exactify_features_exact_solved_depth1_;
        out.approx_exactify_set_size_depth0_min =
            (approx_exactify_set_size_depth0_count_ > 0) ? approx_exactify_set_size_depth0_min_ : 0;
        out.approx_exactify_set_size_depth0_mean =
            (approx_exactify_set_size_depth0_count_ > 0)
                ? (double)approx_exactify_set_size_depth0_sum_ /
                      (double)approx_exactify_set_size_depth0_count_
                : 0.0;
        out.approx_exactify_set_size_depth0_max =
            (approx_exactify_set_size_depth0_count_ > 0) ? approx_exactify_set_size_depth0_max_ : 0;
        out.approx_exactify_set_size_depth1_min =
            (approx_exactify_set_size_depth1_count_ > 0) ? approx_exactify_set_size_depth1_min_ : 0;
        out.approx_exactify_set_size_depth1_mean =
            (approx_exactify_set_size_depth1_count_ > 0)
                ? (double)approx_exactify_set_size_depth1_sum_ /
                      (double)approx_exactify_set_size_depth1_count_
                : 0.0;
        out.approx_exactify_set_size_depth1_max =
            (approx_exactify_set_size_depth1_count_ > 0) ? approx_exactify_set_size_depth1_max_ : 0;
        out.approx_exactify_avg_features_per_triggered_node =
            (approx_exactify_triggered_nodes_ > 0)
                ? (double)approx_exactify_features_exact_solved_ / (double)approx_exactify_triggered_nodes_
                : 0.0;
        out.approx_exactify_ambiguous_set_size_min =
            (approx_exactify_ambiguous_set_size_seen_ > 0)
                ? (double)approx_exactify_ambiguous_set_size_min_
                : 0.0;
        out.approx_exactify_ambiguous_set_size_mean =
            (approx_exactify_ambiguous_set_size_mean_node_count_ > 0)
                ? approx_exactify_ambiguous_set_size_mean_node_sum_ /
                      (double)approx_exactify_ambiguous_set_size_mean_node_count_
                : 0.0;
        out.approx_exactify_ambiguous_set_size_max = approx_exactify_ambiguous_set_size_max_;
        out.approx_exactify_ambiguous_set_shrank_steps = approx_exactify_ambiguous_set_shrank_steps_;
        out.approx_exactify_cap_effective_depth0 =
            (approx_exactify_triggered_nodes_depth0_ > 0)
                ? (double)approx_exactify_cap_effective_sum_depth0_ /
                      (double)approx_exactify_triggered_nodes_depth0_
                : 0.0;
        out.approx_exactify_cap_effective_depth1 =
            (approx_exactify_triggered_nodes_depth1_ > 0)
                ? (double)approx_exactify_cap_effective_sum_depth1_ /
                      (double)approx_exactify_triggered_nodes_depth1_
                : 0.0;
        out.approx_challenger_sweep_invocations = approx_challenger_sweep_invocations_;
        out.approx_challenger_sweep_features_processed = approx_challenger_sweep_features_processed_;
        out.approx_challenger_sweep_sec = approx_challenger_sweep_sec_;
        out.approx_challenger_sweep_skipped_large_ambiguous =
            approx_challenger_sweep_skipped_large_ambiguous_;
        out.approx_challenger_sweep_patch_cap_hit = approx_challenger_sweep_patch_cap_hit_;
        out.approx_uncertainty_triggered_nodes = approx_uncertainty_triggered_nodes_;
        out.approx_eligible_nodes_depth0 = approx_eligible_nodes_depth0_;
        out.approx_eligible_nodes_depth1 = approx_eligible_nodes_depth1_;
        out.approx_exactify_triggered_nodes_depth0 = approx_exactify_triggered_nodes_depth0_;
        out.approx_exactify_triggered_nodes_depth1 = approx_exactify_triggered_nodes_depth1_;
        out.approx_uncertainty_triggered_nodes_depth0 = approx_uncertainty_triggered_nodes_depth0_;
        out.approx_uncertainty_triggered_nodes_depth1 = approx_uncertainty_triggered_nodes_depth1_;
        out.approx_exactify_trigger_rate_depth0 =
            (approx_eligible_nodes_depth0_ > 0)
                ? (double)approx_exactify_triggered_nodes_depth0_ / (double)approx_eligible_nodes_depth0_
                : 0.0;
        out.approx_exactify_trigger_rate_depth1 =
            (approx_eligible_nodes_depth1_ > 0)
                ? (double)approx_exactify_triggered_nodes_depth1_ / (double)approx_eligible_nodes_depth1_
                : 0.0;
        out.approx_uncertainty_trigger_rate_depth0 =
            (approx_eligible_nodes_depth0_ > 0)
                ? (double)approx_uncertainty_triggered_nodes_depth0_ / (double)approx_eligible_nodes_depth0_
                : 0.0;
        out.approx_uncertainty_trigger_rate_depth1 =
            (approx_eligible_nodes_depth1_ > 0)
                ? (double)approx_uncertainty_triggered_nodes_depth1_ / (double)approx_eligible_nodes_depth1_
                : 0.0;
        out.approx_pub_unrefined_cells_on_pub_total = approx_pub_unrefined_cells_on_pub_total_;
        out.approx_pub_patchable_cells_total = approx_pub_patchable_cells_total_;
        out.approx_pub_cells_skipped_by_childrows = approx_pub_cells_skipped_by_childrows_;
        out.approx_nodes_with_patchable_pub = approx_nodes_with_patchable_pub_;
        out.approx_nodes_with_patch_calls = approx_nodes_with_patch_calls_;
        out.approx_patch_cell_cache_hits = approx_patch_cell_cache_hits_;
        out.approx_patch_cell_cache_misses = approx_patch_cell_cache_misses_;
        out.approx_patch_cache_hit_updates = approx_patch_cache_hit_updates_;
        out.approx_patch_cache_miss_oracle_calls = approx_patch_cache_miss_oracle_calls_;
        out.approx_patch_subset_materializations = approx_patch_subset_materializations_;
        out.approx_patch_skipped_already_tight = approx_patch_skipped_already_tight_;
        out.approx_patch_skipped_no_possible_improve = approx_patch_skipped_no_possible_improve_;
        out.approx_patch_skipped_cached = approx_patch_skipped_cached_;
        out.approx_patch_budget_effective_min =
            (approx_patch_budget_effective_seen_ > 0) ? approx_patch_budget_effective_min_ : 0;
        out.approx_patch_budget_effective_avg =
            (approx_patch_budget_effective_seen_ > 0)
                ? (double)approx_patch_budget_effective_sum_ / (double)approx_patch_budget_effective_seen_
                : 0.0;
        out.approx_patch_budget_effective_max =
            (approx_patch_budget_effective_seen_ > 0) ? approx_patch_budget_effective_max_ : 0;
        out.approx_ref_neff_mean =
            (approx_ref_neff_count_ > 0) ? (approx_ref_neff_sum_ / (double)approx_ref_neff_count_) : 0.0;
        out.approx_ref_neff_max = (approx_ref_neff_count_ > 0) ? approx_ref_neff_max_ : 0.0;
        out.approx_ref_k0_min = (approx_ref_k0_count_ > 0) ? approx_ref_k0_min_ : 0;
        out.approx_ref_k0_mean =
            (approx_ref_k0_count_ > 0) ? ((double)approx_ref_k0_sum_ / (double)approx_ref_k0_count_) : 0.0;
        out.approx_ref_k0_max = (approx_ref_k0_count_ > 0) ? approx_ref_k0_max_ : 0;
        out.approx_ref_k_final_min = (approx_ref_k_final_count_ > 0) ? approx_ref_k_final_min_ : 0;
        out.approx_ref_k_final_mean =
            (approx_ref_k_final_count_ > 0)
                ? ((double)approx_ref_k_final_sum_ / (double)approx_ref_k_final_count_)
                : 0.0;
        out.approx_ref_k_final_max = (approx_ref_k_final_count_ > 0) ? approx_ref_k_final_max_ : 0;
        out.approx_ref_k_depth0_mean =
            (approx_ref_k_depth0_count_ > 0)
                ? ((double)approx_ref_k_depth0_sum_ / (double)approx_ref_k_depth0_count_)
                : 0.0;
        out.approx_ref_k_depth1_mean =
            (approx_ref_k_depth1_count_ > 0)
                ? ((double)approx_ref_k_depth1_sum_ / (double)approx_ref_k_depth1_count_)
                : 0.0;
        out.approx_ref_widen_count = approx_ref_widen_count_;
        out.approx_ref_widen_count_depth0 = approx_ref_widen_count_depth0_;
        out.approx_ref_widen_count_depth1 = approx_ref_widen_count_depth1_;
        out.approx_ref_chosen_feature_rank_depth0 =
            (approx_ref_chosen_rank_depth0_count_ > 0)
                ? (approx_ref_chosen_rank_depth0_sum_ / (double)approx_ref_chosen_rank_depth0_count_)
                : 0.0;
        out.approx_ref_chosen_feature_rank_depth1 =
            (approx_ref_chosen_rank_depth1_count_ > 0)
                ? (approx_ref_chosen_rank_depth1_sum_ / (double)approx_ref_chosen_rank_depth1_count_)
                : 0.0;
        out.approx_ref_chosen_in_initial_shortlist_rate_depth0 =
            (approx_ref_chosen_depth0_total_ > 0)
                ? ((double)approx_ref_chosen_depth0_in_initial_ / (double)approx_ref_chosen_depth0_total_)
                : 0.0;
        out.approx_ref_chosen_in_initial_shortlist_rate_depth1 =
            (approx_ref_chosen_depth1_total_ > 0)
                ? ((double)approx_ref_chosen_depth1_in_initial_ / (double)approx_ref_chosen_depth1_total_)
                : 0.0;
        out.fast100_exactify_nodes_allowed = fast100_exactify_nodes_allowed_;
        out.fast100_exactify_nodes_skipped_small_support = fast100_exactify_nodes_skipped_small_support_;
        out.fast100_exactify_nodes_skipped_dominant_gain = fast100_exactify_nodes_skipped_dominant_gain_;
        out.depth1_skipped_by_low_global_ambiguity = depth1_skipped_by_low_global_ambiguity_;
        out.depth1_skipped_by_large_gap = depth1_skipped_by_large_gap_;
        out.depth1_exactify_challenger_nodes = depth1_exactify_challenger_nodes_;
        out.depth1_exactified_nodes = depth1_exactified_nodes_;
        out.depth1_exactified_features_mean =
            (depth1_exactified_nodes_ > 0)
                ? ((double)depth1_exactified_features_sum_ / (double)depth1_exactified_nodes_)
                : 0.0;
        out.depth1_exactified_features_max = depth1_exactified_features_max_;
        out.depth1_teacher_replaced_runnerup = depth1_teacher_replaced_runnerup_;
        out.depth1_teacher_rejected_by_uhat_gate = depth1_teacher_rejected_by_uhat_gate_;
        out.depth1_exactify_set_size_mean = out.depth1_exactified_features_mean;
        out.depth1_exactify_set_size_max = depth1_exactified_features_max_;
        out.fast100_skipped_by_ub_lb_separation = fast100_skipped_by_ub_lb_separation_;
        out.fast100_widen_forbidden_depth_gt0_attempts = fast100_widen_forbidden_depth_gt0_attempts_;
        out.fast100_frontier_size_mean =
            (fast100_frontier_size_count_ > 0)
                ? ((double)fast100_frontier_size_sum_ / (double)fast100_frontier_size_count_)
                : 0.0;
        out.fast100_frontier_size_max = fast100_frontier_size_max_;
        out.fast100_stopped_midloop_separation = fast100_stopped_midloop_separation_;
        out.fast100_M_depth0_mean =
            (fast100_m_depth0_count_ > 0)
                ? ((double)fast100_m_depth0_sum_ / (double)fast100_m_depth0_count_)
                : 0.0;
        out.fast100_M_depth0_max = fast100_m_depth0_max_;
        out.fast100_M_depth1_mean =
            (fast100_m_depth1_count_ > 0)
                ? ((double)fast100_m_depth1_sum_ / (double)fast100_m_depth1_count_)
                : 0.0;
        out.fast100_M_depth1_max = fast100_m_depth1_max_;
        out.fast100_cf_exactify_nodes_depth0 = fast100_cf_exactify_nodes_depth0_;
        out.fast100_cf_exactify_nodes_depth1 = fast100_cf_exactify_nodes_depth1_;
        out.fast100_cf_skipped_agreement = fast100_cf_skipped_agreement_;
        out.fast100_cf_skipped_small_regret = fast100_cf_skipped_small_regret_;
        out.fast100_cf_skipped_low_impact = fast100_cf_skipped_low_impact_;
        out.fast100_cf_frontier_size_mean =
            (fast100_cf_frontier_size_count_ > 0)
                ? ((double)fast100_cf_frontier_size_sum_ / (double)fast100_cf_frontier_size_count_)
                : 0.0;
        out.fast100_cf_frontier_size_max = fast100_cf_frontier_size_max_;
        out.fast100_cf_exactified_features_mean =
            (fast100_cf_exactified_features_count_ > 0)
                ? ((double)fast100_cf_exactified_features_sum_ /
                   (double)fast100_cf_exactified_features_count_)
                : 0.0;
        out.fast100_cf_exactified_features_max = fast100_cf_exactified_features_max_;
        out.rootsafe_exactified_features = rootsafe_exactified_features_;
        out.rootsafe_root_winner_changed_vs_proxy = rootsafe_root_winner_changed_vs_proxy_;
        out.rootsafe_root_candidates_K = rootsafe_root_candidates_k_;
        out.fast100_used_lgb_prior_tiebreak = fast100_used_lgb_prior_tiebreak_ ? 1 : 0;
        out.gini_dp_calls_root = gini_dp_calls_root_;
        out.gini_dp_calls_depth1 = gini_dp_calls_depth1_;
        out.gini_teacher_chosen_depth1 = gini_teacher_chosen_depth1_;
        out.gini_tiebreak_used_in_shortlist = gini_tiebreak_used_in_shortlist_;
        out.gini_dp_sec = gini_dp_sec_;
        out.gini_root_k0 = gini_root_k0_;
        out.gini_endpoints_added_root = gini_endpoints_added_root_;
        out.gini_endpoints_added_depth1 = gini_endpoints_added_depth1_;
        out.gini_endpoints_features_touched_root = gini_endpoints_features_touched_root_;
        out.gini_endpoints_features_touched_depth1 = gini_endpoints_features_touched_depth1_;
        out.gini_endpoints_added_per_feature_max = gini_endpoints_added_per_feature_max_;
        out.gini_endpoint_sec = gini_endpoint_sec_;
#ifndef NDEBUG
        const long long gini_calls_total = gini_dp_calls_root_ + gini_dp_calls_depth1_;
        if (gini_calls_total > 0) {
            const double mean_k =
                (gini_dp_scored_count_ > 0)
                    ? ((double)gini_dp_k_sum_ / (double)gini_dp_scored_count_)
                    : 0.0;
            const double mean_b =
                (gini_dp_scored_count_ > 0)
                    ? ((double)gini_dp_b_sum_ / (double)gini_dp_scored_count_)
                    : 0.0;
            std::fprintf(
                stderr,
                "[GINI_DP_SUMMARY] calls_root=%lld calls_depth1=%lld sec=%.6f "
                "mean_k=%.4f max_k=%d mean_b=%.4f max_b=%d depth1_teacher_changed=%lld\n",
                (long long)gini_dp_calls_root_,
                (long long)gini_dp_calls_depth1_,
                gini_dp_sec_,
                mean_k,
                gini_dp_k_max_,
                mean_b,
                gini_dp_b_max_,
                (long long)gini_depth1_teacher_changed_count_);
        }
#endif
        return out;
    }

   private:
    const std::vector<int> &x_flat_;
    int n_rows_;
    int n_features_;
    const std::vector<int> &y_;
    const std::vector<double> &sample_weight_raw_;
    std::vector<double> sample_weight_;
    bool non_uniform_weights_ = false;

    int full_depth_budget_;
    int lookahead_depth_budget_;
    int effective_lookahead_;
    double regularization_;
    double branch_penalty_;
    double one_mistake_unit_ = 1.0;
    int min_child_size_;
    double time_limit_seconds_;
    int max_branching_;
    int partition_strategy_ = kPartitionOptimalDp;
    bool approx_mode_ = false;
    bool approx_lhat_computed_ = false;
    int patch_budget_per_feature_ = 12;
    int exactify_top_m_ = 2;
    int tau_mode_ = kTauLambdaSqrtR;
    int approx_feature_scan_limit_ = 0;
    bool approx_ref_shortlist_enabled_ = true;
    int approx_ref_widen_max_ = 1;
    bool approx_challenger_sweep_enabled_ = false;
    int approx_challenger_sweep_max_features_ = 3;
    int approx_challenger_sweep_max_patch_calls_per_node_ = 0;
    long long greedy_cache_max_entries_for_approx_ = 200000;
    bool force_rush_legacy_ = false;
    bool rush_profile_enabled_ = false;
    bool dp_cache_profile_enabled_ = false;
    int exact_internal_nodes_ = 0;
    int greedy_internal_nodes_ = 0;
    long long dp_subproblem_calls_ = 0;
    long long dp_cache_hits_ = 0;
    long long greedy_subproblem_calls_ = 0;
    long long greedy_cache_hits_ = 0;
    long long greedy_cache_entries_live_ = 0;
    long long greedy_cache_entries_peak_ = 0;
    long long greedy_cache_clears_ = 0;
    int cheap_oracle_context_depth_ = 0;
    long long dp_interval_evals_ = 0;
    long long greedy_interval_evals_ = 0;
    long long rush_incumbent_feature_aborts_ = 0;
    long long rush_refinement_child_calls_ = 0;
    long long rush_refinement_recursive_calls_ = 0;
    long long rush_refinement_recursive_unique_states_ = 0;
    long long rush_ub_rescue_picks_ = 0;
    long long rush_global_fallback_picks_ = 0;
    double rush_profile_ub0_ordering_sec_ = 0.0;
    double rush_profile_exact_lazy_eval_sec_ = 0.0;
    double rush_profile_exact_lazy_eval_exclusive_sec_ = 0.0;
    double rush_profile_exact_lazy_eval_sec_depth0_ = 0.0;
    double rush_profile_exact_lazy_eval_exclusive_sec_depth0_ = 0.0;
    double rush_profile_exact_lazy_table_init_sec_ = 0.0;
    double rush_profile_exact_lazy_dp_recompute_sec_ = 0.0;
    double rush_profile_exact_lazy_child_solve_sec_ = 0.0;
    double rush_profile_exact_lazy_child_solve_sec_depth0_ = 0.0;
    double rush_profile_exact_lazy_closure_sec_ = 0.0;
    long long rush_profile_exact_lazy_dp_recompute_calls_ = 0;
    long long rush_profile_exact_lazy_closure_passes_ = 0;
    long long interval_refinements_attempted_ = 0;
    long long expensive_child_calls_ = 0;
    double expensive_child_sec_ = 0.0;
    long long expensive_child_exactify_calls_ = 0;
    double expensive_child_exactify_sec_ = 0.0;
    int exactify_eval_context_depth_ = 0;
    long long approx_greedy_patch_calls_ = 0;
    long long approx_greedy_patches_applied_ = 0;
    long long approx_greedy_ub_updates_total_ = 0;
    double approx_greedy_patch_sec_ = 0.0;
    long long approx_exactify_triggered_nodes_ = 0;
    long long approx_exactify_features_exact_solved_ = 0;
    long long approx_exactify_stops_by_separation_ = 0;
    long long approx_exactify_stops_by_cap_ = 0;
    long long approx_exactify_stops_by_ambiguous_empty_ = 0;
    long long approx_exactify_stops_by_no_improve_ = 0;
    long long approx_exactify_stops_by_separation_depth0_ = 0;
    long long approx_exactify_stops_by_separation_depth1_ = 0;
    long long approx_exactify_stops_by_cap_depth0_ = 0;
    long long approx_exactify_stops_by_cap_depth1_ = 0;
    long long approx_exactify_features_exact_solved_depth0_ = 0;
    long long approx_exactify_features_exact_solved_depth1_ = 0;
    long long approx_exactify_set_size_depth0_count_ = 0;
    long long approx_exactify_set_size_depth0_sum_ = 0;
    int approx_exactify_set_size_depth0_min_ = std::numeric_limits<int>::max();
    int approx_exactify_set_size_depth0_max_ = 0;
    long long approx_exactify_set_size_depth1_count_ = 0;
    long long approx_exactify_set_size_depth1_sum_ = 0;
    int approx_exactify_set_size_depth1_min_ = std::numeric_limits<int>::max();
    int approx_exactify_set_size_depth1_max_ = 0;
    long long approx_exactify_ambiguous_set_size_seen_ = 0;
    long long approx_exactify_ambiguous_set_size_min_ = std::numeric_limits<long long>::max();
    double approx_exactify_ambiguous_set_size_mean_node_sum_ = 0.0;
    long long approx_exactify_ambiguous_set_size_mean_node_count_ = 0;
    long long approx_exactify_ambiguous_set_size_max_ = 0;
    long long approx_exactify_ambiguous_set_shrank_steps_ = 0;
    long long approx_exactify_cap_effective_sum_depth0_ = 0;
    long long approx_exactify_cap_effective_sum_depth1_ = 0;
    long long approx_challenger_sweep_invocations_ = 0;
    long long approx_challenger_sweep_features_processed_ = 0;
    double approx_challenger_sweep_sec_ = 0.0;
    long long approx_challenger_sweep_skipped_large_ambiguous_ = 0;
    long long approx_challenger_sweep_patch_cap_hit_ = 0;
    long long approx_uncertainty_triggered_nodes_ = 0;
    long long approx_eligible_nodes_depth0_ = 0;
    long long approx_eligible_nodes_depth1_ = 0;
    long long approx_exactify_triggered_nodes_depth0_ = 0;
    long long approx_exactify_triggered_nodes_depth1_ = 0;
    long long approx_uncertainty_triggered_nodes_depth0_ = 0;
    long long approx_uncertainty_triggered_nodes_depth1_ = 0;
    long long approx_pub_unrefined_cells_on_pub_total_ = 0;
    long long approx_pub_patchable_cells_total_ = 0;
    long long approx_pub_cells_skipped_by_childrows_ = 0;
    long long approx_nodes_with_patchable_pub_ = 0;
    long long approx_nodes_with_patch_calls_ = 0;
    long long approx_patch_cell_cache_hits_ = 0;
    long long approx_patch_cell_cache_misses_ = 0;
    long long approx_patch_cache_hit_updates_ = 0;
    long long approx_patch_cache_miss_oracle_calls_ = 0;
    long long approx_patch_subset_materializations_ = 0;
    long long approx_patch_skipped_already_tight_ = 0;
    long long approx_patch_skipped_no_possible_improve_ = 0;
    long long approx_patch_skipped_cached_ = 0;
    long long approx_patch_budget_effective_seen_ = 0;
    long long approx_patch_budget_effective_sum_ = 0;
    int approx_patch_budget_effective_min_ = std::numeric_limits<int>::max();
    int approx_patch_budget_effective_max_ = 0;
    long long approx_ref_neff_count_ = 0;
    double approx_ref_neff_sum_ = 0.0;
    double approx_ref_neff_max_ = 0.0;
    long long approx_ref_k0_count_ = 0;
    long long approx_ref_k0_sum_ = 0;
    int approx_ref_k0_min_ = std::numeric_limits<int>::max();
    int approx_ref_k0_max_ = 0;
    long long approx_ref_k_final_count_ = 0;
    long long approx_ref_k_final_sum_ = 0;
    int approx_ref_k_final_min_ = std::numeric_limits<int>::max();
    int approx_ref_k_final_max_ = 0;
    long long approx_ref_k_depth0_count_ = 0;
    long long approx_ref_k_depth0_sum_ = 0;
    long long approx_ref_k_depth1_count_ = 0;
    long long approx_ref_k_depth1_sum_ = 0;
    long long approx_ref_widen_count_ = 0;
    long long approx_ref_widen_count_depth0_ = 0;
    long long approx_ref_widen_count_depth1_ = 0;
    long long approx_ref_chosen_rank_depth0_count_ = 0;
    double approx_ref_chosen_rank_depth0_sum_ = 0.0;
    long long approx_ref_chosen_rank_depth1_count_ = 0;
    double approx_ref_chosen_rank_depth1_sum_ = 0.0;
    long long approx_ref_chosen_depth0_total_ = 0;
    long long approx_ref_chosen_depth0_in_initial_ = 0;
    long long approx_ref_chosen_depth1_total_ = 0;
    long long approx_ref_chosen_depth1_in_initial_ = 0;
    long long fast100_exactify_nodes_allowed_ = 0;
    long long fast100_exactify_nodes_skipped_small_support_ = 0;
    long long fast100_exactify_nodes_skipped_dominant_gain_ = 0;
    long long depth1_skipped_by_low_global_ambiguity_ = 0;
    long long depth1_skipped_by_large_gap_ = 0;
    long long depth1_exactify_challenger_nodes_ = 0;
    long long depth1_exactified_nodes_ = 0;
    long long depth1_exactified_features_sum_ = 0;
    int depth1_exactified_features_max_ = 0;
    long long depth1_teacher_replaced_runnerup_ = 0;
    long long depth1_teacher_rejected_by_uhat_gate_ = 0;
    long long fast100_skipped_by_ub_lb_separation_ = 0;
    long long fast100_widen_forbidden_depth_gt0_attempts_ = 0;
    long long fast100_frontier_size_count_ = 0;
    long long fast100_frontier_size_sum_ = 0;
    int fast100_frontier_size_max_ = 0;
    long long fast100_stopped_midloop_separation_ = 0;
    long long fast100_m_depth0_count_ = 0;
    long long fast100_m_depth0_sum_ = 0;
    int fast100_m_depth0_max_ = 0;
    long long fast100_m_depth1_count_ = 0;
    long long fast100_m_depth1_sum_ = 0;
    int fast100_m_depth1_max_ = 0;
    long long fast100_cf_exactify_nodes_depth0_ = 0;
    long long fast100_cf_exactify_nodes_depth1_ = 0;
    long long fast100_cf_skipped_agreement_ = 0;
    long long fast100_cf_skipped_small_regret_ = 0;
    long long fast100_cf_skipped_low_impact_ = 0;
    long long fast100_cf_frontier_size_count_ = 0;
    long long fast100_cf_frontier_size_sum_ = 0;
    int fast100_cf_frontier_size_max_ = 0;
    long long fast100_cf_exactified_features_count_ = 0;
    long long fast100_cf_exactified_features_sum_ = 0;
    int fast100_cf_exactified_features_max_ = 0;
    int rootsafe_exactified_features_ = 0;
    int rootsafe_root_winner_changed_vs_proxy_ = 0;
    int rootsafe_root_candidates_k_ = 0;
    bool fast100_used_lgb_prior_tiebreak_ = false;
    long long gini_dp_calls_root_ = 0;
    long long gini_dp_calls_depth1_ = 0;
    long long gini_teacher_chosen_depth1_ = 0;
    long long gini_tiebreak_used_in_shortlist_ = 0;
    double gini_dp_sec_ = 0.0;
    int gini_root_k0_ = 0;
    long long gini_endpoints_added_root_ = 0;
    long long gini_endpoints_added_depth1_ = 0;
    long long gini_endpoints_features_touched_root_ = 0;
    long long gini_endpoints_features_touched_depth1_ = 0;
    int gini_endpoints_added_per_feature_max_ = 0;
    double gini_endpoint_sec_ = 0.0;
    long long gini_dp_scored_count_ = 0;
    long long gini_dp_k_sum_ = 0;
    int gini_dp_k_max_ = 0;
    long long gini_dp_b_sum_ = 0;
    int gini_dp_b_max_ = 0;
    long long gini_depth1_teacher_changed_count_ = 0;
    bool gini_scout_enabled_ = false;
    int fast100_debug_depth1_limit_ = 0;
    int fast100_debug_depth1_nodes_logged_ = 0;
    bool approx_ref_root_ready_ = false;
    std::vector<int> approx_ref_root_order_;
    double approx_ref_neff_root_ = 0.0;

    Clock::time_point start_time_;

    enum class IntervalStatus : unsigned char {
        kInfeasible = 0,
        kCertifiedLeaf = 1,
        kUnrefined = 2,
        kRefined = 3,
    };

    struct DpCacheEntry {
        int depth_remaining = 0;
        std::vector<int> indices;
        BoundResult result;
    };
    struct GreedyCacheEntry {
        int depth_remaining = 0;
        std::vector<int> indices;
        GreedyResult result;
    };

    struct CheapUbResult {
        double objective = kInfinity;
        std::shared_ptr<Node> tree;
    };

    struct ApproxFeatureResult {
        bool feasible = false;
        int feature = -1;
        double lhat = kInfinity;
        double uhat = kInfinity;
        int projected_endpoint_count = 0;
        std::vector<std::vector<std::pair<int, int>>> group_spans;
    };

    struct ApproxFeaturePrep {
        int feature = -1;
        bool feasible = false;
        int n_bins_dense = 0;
        int q_effective = 0;
        int q_projected = 0;
        double leaf_cost = kInfinity;
        double best_2way_split_cost = kInfinity;
        double g = 0.0;
        bool gini_endpoints_augmented = false;
        int gini_endpoints_added = 0;
        std::vector<int> row_cnt_bin;
        std::vector<int> pos_cnt_bin;
        std::vector<int> neg_cnt_bin;
        std::vector<double> pos_w_bin;
        std::vector<double> neg_w_bin;
        std::vector<int> row_cnt_prefix;
        std::vector<int> pos_cnt_prefix;
        std::vector<int> neg_cnt_prefix;
        std::vector<double> pos_w_prefix;
        std::vector<double> neg_w_prefix;
        std::vector<int> endpoints;
    };

    struct GiniDPResult {
        bool feasible = false;
        double best_cost = kInfinity;
        int best_k = 0;
        std::vector<int> cuts;
    };

    struct ApproxPatchCellKey {
        int feature = -1;
        int p = -1;
        int t = -1;
        int child_depth_remaining = -1;

        bool operator==(const ApproxPatchCellKey &other) const {
            return feature == other.feature &&
                   p == other.p &&
                   t == other.t &&
                   child_depth_remaining == other.child_depth_remaining;
        }
    };

    struct ApproxPatchCellKeyHash {
        size_t operator()(const ApproxPatchCellKey &key) const noexcept {
            size_t h = 1469598103934665603ULL;
            h ^= static_cast<size_t>(static_cast<uint32_t>(key.feature));
            h *= 1099511628211ULL;
            h ^= static_cast<size_t>(static_cast<uint32_t>(key.p));
            h *= 1099511628211ULL;
            h ^= static_cast<size_t>(static_cast<uint32_t>(key.t));
            h *= 1099511628211ULL;
            h ^= static_cast<size_t>(static_cast<uint32_t>(key.child_depth_remaining));
            h *= 1099511628211ULL;
            return h;
        }
    };

    struct ApproxPatchCellCacheEntry {
        double ub_child = kInfinity;
        bool no_further_improve = false;
    };

    using ApproxPatchCellCache = std::unordered_map<
        ApproxPatchCellKey,
        ApproxPatchCellCacheEntry,
        ApproxPatchCellKeyHash>;

    long long dp_cache_states_ = 0;
    long long greedy_cache_states_ = 0;
    long long dp_cache_lookup_calls_ = 0;
    long long dp_cache_miss_no_bucket_ = 0;
    long long dp_cache_miss_bucket_present_ = 0;
    long long dp_cache_miss_depth_mismatch_only_ = 0;
    long long dp_cache_miss_indices_mismatch_ = 0;
    long long dp_cache_depth_match_candidates_ = 0;
    long long dp_cache_bucket_entries_scanned_ = 0;
    long long dp_cache_bucket_max_size_ = 0;
    std::unordered_map<uint64_t, std::vector<DpCacheEntry>> dp_cache_;
    std::unordered_map<uint64_t, std::vector<GreedyCacheEntry>> greedy_cache_;

    struct ProjectedDpWorkspace {
        std::vector<std::vector<double>> dp;
        std::vector<std::vector<int>> parent;
    };

    struct GiniDPWorkspace {
        std::vector<std::vector<double>> dp;
        std::vector<std::vector<int>> parent;
    };

    struct RushIntervalWorkspace {
        std::vector<std::vector<IntervalStatus>> status;
        std::vector<std::vector<bool>> valid;
        std::vector<std::vector<double>> ub_leaf_obj;
        std::vector<std::vector<double>> static_mis_lb;
        std::vector<std::vector<double>> lb_obj;
        std::vector<std::vector<double>> ub_obj;
        std::vector<std::vector<double>> interval_support_w;
        std::vector<std::vector<double>> ssb_mis_cur;
        std::vector<std::vector<int>> leaf_prediction;
        std::vector<std::vector<int>> leaf_support_count;
        std::vector<std::vector<int>> leaf_pos_count;
        std::vector<std::vector<int>> leaf_neg_count;
        std::vector<std::vector<std::shared_ptr<Node>>> projected_trees;
        std::vector<int> batch_selected_stamp;
        int batch_selection_token = 0;
        std::vector<int> index_rank;
        std::vector<int> subset_row_stamp;
        int subset_row_token = 0;
        std::vector<int> subset_scratch;
    };
    std::vector<ProjectedDpWorkspace> projected_dp_workspaces_;
    std::vector<RushIntervalWorkspace> rush_interval_workspaces_;
    GiniDPWorkspace gini_dp_workspace_;

    std::vector<int> sig_id_;
    std::vector<std::vector<int>> sig_bin_;
    std::vector<int> feature_bin_max_;
    std::vector<double> sig_pos_acc_;
    std::vector<double> sig_neg_acc_;
    std::vector<int> sig_cnt_acc_;
    std::vector<int> sig_pos_cnt_acc_;
    std::vector<int> sig_neg_cnt_acc_;
    std::vector<int> sig_stamp_;
    int sig_stamp_token_ = 0;
    std::vector<int> touched_signatures_;
    // Reused scratch for build_ordered_bins to avoid per-call hash-map/vector churn.
    mutable std::vector<int> ordered_bin_stamp_;
    mutable int ordered_bin_stamp_token_ = 0;
    mutable std::vector<std::vector<int>> ordered_bin_members_;
    mutable std::vector<int> ordered_bin_touched_;
    mutable std::vector<int> ordered_bin_last_idx_;
    mutable std::vector<unsigned char> ordered_bin_needs_sort_;
    // Dense reusable lookup: bin value -> position in ordered bins for current feature eval.
    mutable std::vector<int> bin_value_pos_stamp_;
    mutable std::vector<int> bin_value_pos_;
    mutable int bin_value_pos_token_ = 0;

    int x(int row, int feature) const { return x_flat_[row * n_features_ + feature]; }

    struct SubproblemStats {
        int total_count = 0;
        int pos_count = 0;
        int neg_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        int prediction = 0;
        bool pure = true;
        double leaf_mis = 0.0;
        double leaf_objective = 0.0;
        double epb_mis = 0.0;
        std::vector<int> touched_signature_ids;
        std::vector<double> touched_signature_epb;
    };

    struct RushFeatureRootCache {
        bool ready = false;
        OrderedBins bins;
        int n_bins = 0;
        int q_effective = 0;
        int t_max = 0;
        int q_projected = 0;
        std::vector<int> endpoints;
        std::vector<int> pos_cnt_prefix;
        std::vector<int> neg_cnt_prefix;
        std::vector<int> s_cnt_prefix;
        std::vector<double> pos_w_prefix;
        std::vector<double> neg_w_prefix;
        std::vector<double> s_w_prefix;
        std::vector<double> epb_prefix;
        double ub0 = kInfinity;
        double lb0 = kInfinity;
        std::shared_ptr<Node> ub0_tree;
    };

    void assert_interval_bound_sanity(
        const std::vector<std::vector<bool>> &valid,
        const std::vector<std::vector<IntervalStatus>> &status,
        const std::vector<std::vector<double>> &lb_obj,
        const std::vector<std::vector<double>> &ub_obj,
        const std::vector<std::vector<double>> &ub_leaf_obj,
        int t_max
    ) const {
        for (int p = 0; p < t_max; ++p) {
            for (int t = p + 1; t <= t_max; ++t) {
                if (!valid[(size_t)p][(size_t)t]) {
                    continue;
                }
                const double lb = lb_obj[(size_t)p][(size_t)t];
                const double ub = ub_obj[(size_t)p][(size_t)t];
                if (lb > ub + kEpsCert) {
                    throw std::runtime_error("RUSH exact-lazy invariant violated: LB_obj > UB_obj for interval.");
                }
                if (status[(size_t)p][(size_t)t] == IntervalStatus::kCertifiedLeaf) {
                    const double leaf = ub_leaf_obj[(size_t)p][(size_t)t];
                    if (std::fabs(lb - ub) > kEpsCert || std::fabs(lb - leaf) > kEpsCert) {
                        throw std::runtime_error(
                            "RUSH exact-lazy invariant violated: CERTIFIED_LEAF interval bounds not equal to leaf UB.");
                    }
                }
            }
        }
    }

    void assert_feature_bound_sanity(double feature_lb, double feature_ub) const {
        if (feature_lb > feature_ub + kEpsCert) {
            throw std::runtime_error("RUSH exact-lazy invariant violated: feature_LB > feature_UB.");
        }
    }

    void initialize_weights() {
        sample_weight_.assign((size_t)n_rows_, 0.0);
        if (sample_weight_raw_.empty()) {
            const double uniform = 1.0 / static_cast<double>(n_rows_);
            std::fill(sample_weight_.begin(), sample_weight_.end(), uniform);
            non_uniform_weights_ = false;
            return;
        }

        double sum = 0.0;
        for (double w : sample_weight_raw_) {
            if (!std::isfinite(w) || w < 0.0) {
                throw std::invalid_argument("MSPLIT sample_weight entries must be finite and non-negative.");
            }
            sum += w;
        }
        if (!std::isfinite(sum) || sum <= 0.0) {
            throw std::invalid_argument("MSPLIT sample_weight must have positive finite sum.");
        }

        const double inv_sum = 1.0 / sum;
        non_uniform_weights_ = false;
        for (int i = 0; i < n_rows_; ++i) {
            sample_weight_[(size_t)i] = sample_weight_raw_[(size_t)i] * inv_sum;
        }
        const double first = sample_weight_.empty() ? 0.0 : sample_weight_.front();
        for (double w : sample_weight_) {
            if (std::fabs(w - first) > kEpsUpdate) {
                non_uniform_weights_ = true;
                break;
            }
        }
    }

    void preprocess_signatures() {
        sig_id_.assign((size_t)n_rows_, -1);
        sig_bin_.clear();
        sig_bin_.reserve((size_t)n_rows_);
        feature_bin_max_.assign((size_t)n_features_, 0);

        std::unordered_map<std::string, int> signature_to_id;
        signature_to_id.reserve((size_t)n_rows_);
        for (int row = 0; row < n_rows_; ++row) {
            std::string key;
            key.reserve((size_t)n_features_ * sizeof(int));
            for (int feature = 0; feature < n_features_; ++feature) {
                const int value = x(row, feature);
                key.append(reinterpret_cast<const char *>(&value), sizeof(int));
            }
            auto it = signature_to_id.find(key);
            int id = -1;
            if (it == signature_to_id.end()) {
                id = static_cast<int>(sig_bin_.size());
                signature_to_id.emplace(std::move(key), id);
                sig_bin_.push_back(std::vector<int>((size_t)n_features_, 0));
                for (int feature = 0; feature < n_features_; ++feature) {
                    const int value = x(row, feature);
                    if (value < 0) {
                        throw std::invalid_argument("MSPLIT expects non-negative integer bins in x_flat.");
                    }
                    sig_bin_[(size_t)id][(size_t)feature] = value;
                    if (value > feature_bin_max_[(size_t)feature]) {
                        feature_bin_max_[(size_t)feature] = value;
                    }
                }
            } else {
                id = it->second;
            }
            sig_id_[(size_t)row] = id;
        }

        const size_t n_sig = sig_bin_.size();
        sig_pos_acc_.assign(n_sig, 0.0);
        sig_neg_acc_.assign(n_sig, 0.0);
        sig_cnt_acc_.assign(n_sig, 0);
        sig_pos_cnt_acc_.assign(n_sig, 0);
        sig_neg_cnt_acc_.assign(n_sig, 0);
        sig_stamp_.assign(n_sig, 0);
        touched_signatures_.clear();
        touched_signatures_.reserve(n_sig);
        sig_stamp_token_ = 0;
    }

    int next_signature_stamp() {
        ++sig_stamp_token_;
        if (sig_stamp_token_ == std::numeric_limits<int>::max()) {
            std::fill(sig_stamp_.begin(), sig_stamp_.end(), 0);
            sig_stamp_token_ = 1;
        }
        return sig_stamp_token_;
    }

    SubproblemStats compute_subproblem_stats(const std::vector<int> &indices) {
        SubproblemStats out;
        out.total_count = static_cast<int>(indices.size());
        out.pure = true;
        const int stamp = next_signature_stamp();
        touched_signatures_.clear();
        touched_signatures_.reserve(indices.size());

        int first_label = -1;
        for (int idx : indices) {
            const int label = y_[(size_t)idx];
            const double w = sample_weight_[(size_t)idx];
            if (label == 1) {
                out.pos_count += 1;
                out.pos_weight += w;
            } else {
                out.neg_count += 1;
                out.neg_weight += w;
            }
            if (first_label < 0) {
                first_label = label;
            } else if (label != first_label) {
                out.pure = false;
            }

            const int s = sig_id_[(size_t)idx];
            if (sig_stamp_[(size_t)s] != stamp) {
                sig_stamp_[(size_t)s] = stamp;
                sig_pos_acc_[(size_t)s] = 0.0;
                sig_neg_acc_[(size_t)s] = 0.0;
                sig_cnt_acc_[(size_t)s] = 0;
                sig_pos_cnt_acc_[(size_t)s] = 0;
                sig_neg_cnt_acc_[(size_t)s] = 0;
                touched_signatures_.push_back(s);
            }
            sig_cnt_acc_[(size_t)s] += 1;
            if (label == 1) {
                sig_pos_acc_[(size_t)s] += w;
                sig_pos_cnt_acc_[(size_t)s] += 1;
            } else {
                sig_neg_acc_[(size_t)s] += w;
                sig_neg_cnt_acc_[(size_t)s] += 1;
            }
        }

        out.prediction = (out.pos_weight >= out.neg_weight) ? 1 : 0;
        out.leaf_mis = (out.prediction == 1) ? out.neg_weight : out.pos_weight;
        out.leaf_objective = out.leaf_mis + regularization_;

        out.touched_signature_ids.reserve(touched_signatures_.size());
        out.touched_signature_epb.reserve(touched_signatures_.size());
        for (int s : touched_signatures_) {
            const double epb = std::min(sig_pos_acc_[(size_t)s], sig_neg_acc_[(size_t)s]);
            out.epb_mis += epb;
            out.touched_signature_ids.push_back(s);
            out.touched_signature_epb.push_back(epb);
        }
        return out;
    }

    std::shared_ptr<Node> make_leaf_node(
        int prediction,
        int total_count,
        int pos_count,
        int neg_count,
        double leaf_objective
    ) const {
        auto leaf = std::make_shared<Node>();
        leaf->is_leaf = true;
        leaf->prediction = prediction;
        leaf->n_samples = total_count;
        leaf->neg_count = neg_count;
        leaf->pos_count = pos_count;
        leaf->loss = leaf_objective;
        return leaf;
    }

    void check_timeout() const {
        if (time_limit_seconds_ <= 0.0) {
            return;
        }
        std::chrono::duration<double> elapsed = Clock::now() - start_time_;
        if (elapsed.count() > time_limit_seconds_) {
            throw std::runtime_error("MSPLIT exceeded time_limit during C++ solve.");
        }
    }

    int max_groups_for_bins(int n_bins) const {
        if (n_bins <= 0) {
            return 0;
        }
        if (max_branching_ <= 0) {
            return n_bins;
        }
        return std::min(n_bins, max_branching_);
    }

    static uint64_t hash_mix_u64(uint64_t seed, uint64_t value) {
        return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    }

    static uint64_t state_hash(const std::vector<int> &indices, int depth_remaining) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(depth_remaining)));
        h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(indices.size())));
        for (int value : indices) {
            h = hash_mix_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(value)));
        }
        return h;
    }

    int find_dp_cache_slot(
        uint64_t key_hash,
        int depth_remaining,
        const std::vector<int> &indices,
        std::vector<DpCacheEntry> **bucket_out = nullptr
    ) {
        if (dp_cache_profile_enabled_) {
            ++dp_cache_lookup_calls_;
        }
        auto it = dp_cache_.find(key_hash);
        if (it == dp_cache_.end()) {
            if (dp_cache_profile_enabled_) {
                ++dp_cache_miss_no_bucket_;
            }
            if (bucket_out != nullptr) {
                *bucket_out = nullptr;
            }
            return -1;
        }
        if (bucket_out != nullptr) {
            *bucket_out = &it->second;
        }
        auto &bucket = it->second;
        if (dp_cache_profile_enabled_) {
            dp_cache_bucket_entries_scanned_ += static_cast<long long>(bucket.size());
            dp_cache_bucket_max_size_ =
                std::max(dp_cache_bucket_max_size_, static_cast<long long>(bucket.size()));
        }
        bool has_depth_match = false;
        for (size_t i = 0; i < bucket.size(); ++i) {
            if (bucket[i].depth_remaining != depth_remaining) {
                continue;
            }
            has_depth_match = true;
            if (dp_cache_profile_enabled_) {
                ++dp_cache_depth_match_candidates_;
            }
            if (bucket[i].indices == indices) {
                return static_cast<int>(i);
            }
        }
        if (dp_cache_profile_enabled_) {
            ++dp_cache_miss_bucket_present_;
            if (has_depth_match) {
                ++dp_cache_miss_indices_mismatch_;
            } else {
                ++dp_cache_miss_depth_mismatch_only_;
            }
        }
        return -1;
    }

    ProjectedDpWorkspace &projected_dp_workspace_for_depth(int depth) {
        const int safe_depth = std::max(0, depth);
        if ((int)projected_dp_workspaces_.size() <= safe_depth) {
            projected_dp_workspaces_.resize((size_t)safe_depth + 1);
        }
        return projected_dp_workspaces_[(size_t)safe_depth];
    }

    RushIntervalWorkspace &rush_interval_workspace_for_depth(int depth) {
        const int safe_depth = std::max(0, depth);
        if ((int)rush_interval_workspaces_.size() <= safe_depth) {
            rush_interval_workspaces_.resize((size_t)safe_depth + 1);
        }
        return rush_interval_workspaces_[(size_t)safe_depth];
    }

    double split_penalty_for_groups(int groups) const {
        return branch_penalty_ * static_cast<double>(groups - 2);
    }

    bool build_approx_feature_prep_from_signatures(
        const SubproblemStats &node_stats,
        int feature,
        ApproxFeaturePrep &prep
    ) const {
        prep = ApproxFeaturePrep{};
        prep.feature = feature;
        if (feature < 0 || feature >= n_features_) {
            return false;
        }

        const int max_bin = feature_bin_max_[(size_t)feature];
        prep.n_bins_dense = max_bin + 1;
        if (prep.n_bins_dense <= 1) {
            return false;
        }

        prep.row_cnt_bin.assign((size_t)prep.n_bins_dense, 0);
        prep.pos_cnt_bin.assign((size_t)prep.n_bins_dense, 0);
        prep.neg_cnt_bin.assign((size_t)prep.n_bins_dense, 0);
        prep.pos_w_bin.assign((size_t)prep.n_bins_dense, 0.0);
        prep.neg_w_bin.assign((size_t)prep.n_bins_dense, 0.0);

        int observed_bins = 0;
        for (int s : node_stats.touched_signature_ids) {
            const int bin = sig_bin_[(size_t)s][(size_t)feature];
            if (bin < 0 || bin >= prep.n_bins_dense) {
                return false;
            }
            if (prep.row_cnt_bin[(size_t)bin] == 0) {
                ++observed_bins;
            }
            prep.row_cnt_bin[(size_t)bin] += sig_cnt_acc_[(size_t)s];
            prep.pos_cnt_bin[(size_t)bin] += sig_pos_cnt_acc_[(size_t)s];
            prep.neg_cnt_bin[(size_t)bin] += sig_neg_cnt_acc_[(size_t)s];
            prep.pos_w_bin[(size_t)bin] += sig_pos_acc_[(size_t)s];
            prep.neg_w_bin[(size_t)bin] += sig_neg_acc_[(size_t)s];
        }
        if (observed_bins <= 1) {
            return false;
        }

        prep.row_cnt_prefix.assign((size_t)prep.n_bins_dense + 1, 0);
        prep.pos_cnt_prefix.assign((size_t)prep.n_bins_dense + 1, 0);
        prep.neg_cnt_prefix.assign((size_t)prep.n_bins_dense + 1, 0);
        prep.pos_w_prefix.assign((size_t)prep.n_bins_dense + 1, 0.0);
        prep.neg_w_prefix.assign((size_t)prep.n_bins_dense + 1, 0.0);
        for (int b = 0; b < prep.n_bins_dense; ++b) {
            prep.row_cnt_prefix[(size_t)(b + 1)] = prep.row_cnt_prefix[(size_t)b] + prep.row_cnt_bin[(size_t)b];
            prep.pos_cnt_prefix[(size_t)(b + 1)] = prep.pos_cnt_prefix[(size_t)b] + prep.pos_cnt_bin[(size_t)b];
            prep.neg_cnt_prefix[(size_t)(b + 1)] = prep.neg_cnt_prefix[(size_t)b] + prep.neg_cnt_bin[(size_t)b];
            prep.pos_w_prefix[(size_t)(b + 1)] = prep.pos_w_prefix[(size_t)b] + prep.pos_w_bin[(size_t)b];
            prep.neg_w_prefix[(size_t)(b + 1)] = prep.neg_w_prefix[(size_t)b] + prep.neg_w_bin[(size_t)b];
        }

        const int total_rows = prep.row_cnt_prefix[(size_t)prep.n_bins_dense];
        const int q_base = max_groups_for_bins(prep.n_bins_dense);
        const int q_support = std::max(0, total_rows / std::max(1, min_child_size_));
        prep.q_effective = std::min(q_base, q_support);
        if (prep.q_effective < 2) {
            return false;
        }

        prep.endpoints = build_majority_shift_endpoints(prep.pos_w_bin, prep.neg_w_bin, prep.row_cnt_bin);
        if (prep.endpoints.size() < 2 || prep.endpoints.front() != 0 || prep.endpoints.back() != prep.n_bins_dense) {
            return false;
        }
        const int t_max = (int)prep.endpoints.size() - 1;
        prep.q_projected = std::min(prep.q_effective, t_max);
        if (prep.q_projected < 2) {
            return false;
        }

        const double pos_total = prep.pos_w_prefix[(size_t)prep.n_bins_dense];
        const double neg_total = prep.neg_w_prefix[(size_t)prep.n_bins_dense];
        prep.leaf_cost = regularization_ + std::min(pos_total, neg_total);

        double best_split = kInfinity;
        for (int cut = 0; cut + 1 < prep.n_bins_dense; ++cut) {
            const int left_rows = prep.row_cnt_prefix[(size_t)(cut + 1)];
            const int right_rows = total_rows - left_rows;
            if (left_rows < min_child_size_ || right_rows < min_child_size_) {
                continue;
            }
            const double left_pos_w = prep.pos_w_prefix[(size_t)(cut + 1)];
            const double left_neg_w = prep.neg_w_prefix[(size_t)(cut + 1)];
            const double right_pos_w = pos_total - left_pos_w;
            const double right_neg_w = neg_total - left_neg_w;
            const double split_cost =
                2.0 * regularization_ +
                std::min(left_pos_w, left_neg_w) +
                std::min(right_pos_w, right_neg_w);
            if (split_cost < best_split) {
                best_split = split_cost;
            }
        }
        prep.best_2way_split_cost = best_split;
        prep.g = std::isfinite(best_split) ? std::max(0.0, prep.leaf_cost - best_split) : 0.0;
        prep.feasible = true;
        return true;
    }

    static void segment_stats_from_prefix(
        const std::vector<double> &prefix_pos_w,
        const std::vector<double> &prefix_neg_w,
        const std::vector<int> &prefix_row_cnt,
        int a,
        int b,
        double &seg_p,
        double &seg_n,
        int &seg_cnt
    ) {
        seg_p = prefix_pos_w[(size_t)(b + 1)] - prefix_pos_w[(size_t)a];
        seg_n = prefix_neg_w[(size_t)(b + 1)] - prefix_neg_w[(size_t)a];
        seg_cnt = prefix_row_cnt[(size_t)(b + 1)] - prefix_row_cnt[(size_t)a];
    }

    double gini_seg_cost_norm(double seg_p, double seg_n, double w_total_node) const {
        const double w = seg_p + seg_n;
        if (w <= kEpsUpdate) {
            return 0.0;
        }
        const double denom_node = std::max(w_total_node, kEpsUpdate);
        const double denom_seg = std::max(w, kEpsUpdate);
        const double impurity_weighted =
            w - ((seg_p * seg_p + seg_n * seg_n) / denom_seg);
        return impurity_weighted / denom_node;
    }

    GiniDPResult gini_dp_best_partition(
        const ApproxFeaturePrep &prep,
        int k_max,
        int min_child_size_m,
        double w_total_node,
        GiniDPWorkspace &workspace
    ) const {
        GiniDPResult out;
        const int b_count = prep.n_bins_dense;
        if (b_count <= 1) {
            return out;
        }
        const int total_cnt = prep.row_cnt_prefix[(size_t)b_count];
        if (total_cnt < 2 * std::max(1, min_child_size_m)) {
            return out;
        }
        const int k_support = std::max(0, total_cnt / std::max(1, min_child_size_m));
        const int k_upper = std::min(std::max(0, k_max), k_support);
        if (k_upper < 2) {
            return out;
        }

        workspace.dp.resize((size_t)k_upper + 1);
        workspace.parent.resize((size_t)k_upper + 1);
        for (int k = 0; k <= k_upper; ++k) {
            workspace.dp[(size_t)k].assign((size_t)b_count + 1, kInfinity);
            workspace.parent[(size_t)k].assign((size_t)b_count + 1, -1);
        }
        workspace.dp[0][0] = 0.0;

        for (int k = 1; k <= k_upper; ++k) {
            for (int t = k; t <= b_count; ++t) {
                double best_cost = kInfinity;
                int best_p = -1;
                for (int p = k - 1; p <= t - 1; ++p) {
                    const double prev = workspace.dp[(size_t)(k - 1)][(size_t)p];
                    if (!std::isfinite(prev)) {
                        continue;
                    }
                    const int seg_cnt =
                        prep.row_cnt_prefix[(size_t)t] - prep.row_cnt_prefix[(size_t)p];
                    if (seg_cnt < min_child_size_m) {
                        continue;
                    }
                    double seg_p = 0.0;
                    double seg_n = 0.0;
                    int seg_cnt_check = 0;
                    segment_stats_from_prefix(
                        prep.pos_w_prefix,
                        prep.neg_w_prefix,
                        prep.row_cnt_prefix,
                        p,
                        t - 1,
                        seg_p,
                        seg_n,
                        seg_cnt_check);
                    if (seg_cnt_check < min_child_size_m) {
                        continue;
                    }
                    const double cand =
                        prev + gini_seg_cost_norm(seg_p, seg_n, w_total_node);
                    if (cand < best_cost - kEpsUpdate ||
                        (std::fabs(cand - best_cost) <= kEpsUpdate &&
                         (best_p < 0 || p < best_p))) {
                        best_cost = cand;
                        best_p = p;
                    }
                }
                if (best_p >= 0) {
                    workspace.dp[(size_t)k][(size_t)t] = best_cost;
                    workspace.parent[(size_t)k][(size_t)t] = best_p;
                }
            }
        }

        int best_k = -1;
        double best_obj = kInfinity;
        for (int k = 2; k <= k_upper; ++k) {
            const double cand = workspace.dp[(size_t)k][(size_t)b_count];
            if (!std::isfinite(cand)) {
                continue;
            }
            if (cand < best_obj - kEpsUpdate ||
                (std::fabs(cand - best_obj) <= kEpsUpdate &&
                 (best_k < 0 || k < best_k))) {
                best_obj = cand;
                best_k = k;
            }
        }
        if (best_k < 2) {
            return out;
        }

        out.feasible = true;
        out.best_cost = best_obj;
        out.best_k = best_k;
        out.cuts.clear();
        int t = b_count;
        int k = best_k;
        while (k > 1) {
            const int p = workspace.parent[(size_t)k][(size_t)t];
            if (p < 0) {
                out.feasible = false;
                out.best_cost = kInfinity;
                out.best_k = 0;
                out.cuts.clear();
                return out;
            }
            out.cuts.push_back(p - 1);
            t = p;
            --k;
        }
        std::reverse(out.cuts.begin(), out.cuts.end());
        return out;
    }

    static std::vector<std::pair<int, int>> compress_observed_bin_spans_from_hist(
        const std::vector<int> &row_cnt_bin,
        int left,
        int right
    ) {
        std::vector<std::pair<int, int>> spans;
        if (left < 0 || right < left || right >= (int)row_cnt_bin.size()) {
            return spans;
        }
        int span_lo = -1;
        int span_hi = -1;
        for (int b = left; b <= right; ++b) {
            if (row_cnt_bin[(size_t)b] <= 0) {
                continue;
            }
            if (span_lo < 0) {
                span_lo = b;
                span_hi = b;
                continue;
            }
            if (b == span_hi + 1) {
                span_hi = b;
            } else {
                spans.push_back({span_lo, span_hi});
                span_lo = b;
                span_hi = b;
            }
        }
        if (span_lo >= 0) {
            spans.push_back({span_lo, span_hi});
        }
        return spans;
    }

    static void append_sorted_members(std::vector<int> &dst, const std::vector<int> &members_sorted) {
        const size_t mid = dst.size();
        dst.insert(dst.end(), members_sorted.begin(), members_sorted.end());
        std::inplace_merge(dst.begin(), dst.begin() + static_cast<std::ptrdiff_t>(mid), dst.end());
    }

    std::pair<double, std::shared_ptr<Node>> leaf_solution(const SubproblemStats &stats) const {
        auto leaf = make_leaf_node(
            stats.prediction,
            stats.total_count,
            stats.pos_count,
            stats.neg_count,
            stats.leaf_objective);
        return {stats.leaf_objective, leaf};
    }

    bool build_ordered_bins(const std::vector<int> &indices, int feature, OrderedBins &out) const {
        if (feature < 0 || feature >= n_features_) {
            return false;
        }
        const int max_bin = feature_bin_max_[(size_t)feature];
        if (max_bin < 0) {
            return false;
        }
        const size_t dense_size = (size_t)max_bin + 1U;
        if (ordered_bin_stamp_.size() < dense_size) {
            ordered_bin_stamp_.resize(dense_size, 0);
            ordered_bin_members_.resize(dense_size);
            ordered_bin_last_idx_.resize(dense_size, 0);
            ordered_bin_needs_sort_.resize(dense_size, 0);
        }

        ++ordered_bin_stamp_token_;
        if (ordered_bin_stamp_token_ == std::numeric_limits<int>::max()) {
            std::fill(ordered_bin_stamp_.begin(), ordered_bin_stamp_.end(), 0);
            ordered_bin_stamp_token_ = 1;
        }
        const int stamp = ordered_bin_stamp_token_;

        ordered_bin_touched_.clear();
        ordered_bin_touched_.reserve(std::min((size_t)indices.size(), dense_size));

        for (int idx : indices) {
            const int bin = x(idx, feature);
            if (bin < 0 || bin > max_bin) {
                return false;
            }
            if (ordered_bin_stamp_[(size_t)bin] != stamp) {
                ordered_bin_stamp_[(size_t)bin] = stamp;
                ordered_bin_members_[(size_t)bin].clear();
                ordered_bin_needs_sort_[(size_t)bin] = 0;
                ordered_bin_last_idx_[(size_t)bin] = idx;
                ordered_bin_touched_.push_back(bin);
            } else if (idx < ordered_bin_last_idx_[(size_t)bin]) {
                ordered_bin_needs_sort_[(size_t)bin] = 1;
            }
            ordered_bin_last_idx_[(size_t)bin] = idx;
            ordered_bin_members_[(size_t)bin].push_back(idx);
        }

        if (ordered_bin_touched_.size() <= 1U) {
            return false;
        }

        std::sort(ordered_bin_touched_.begin(), ordered_bin_touched_.end());
        out.values.clear();
        out.values.reserve(ordered_bin_touched_.size());
        out.members.clear();
        out.members.reserve(ordered_bin_touched_.size());
        out.prefix_counts.assign(ordered_bin_touched_.size() + 1U, 0);

        for (size_t i = 0; i < ordered_bin_touched_.size(); ++i) {
            const int bin = ordered_bin_touched_[i];
            auto &members = ordered_bin_members_[(size_t)bin];
            if (ordered_bin_needs_sort_[(size_t)bin]) {
                std::sort(members.begin(), members.end());
            }
            out.values.push_back(bin);
            out.members.push_back(members);
            out.prefix_counts[i + 1U] = out.prefix_counts[i] + (int)members.size();
        }
        return true;
    }

    static std::vector<int> build_majority_shift_endpoints(
        const std::vector<double> &pos_mass,
        const std::vector<double> &neg_mass,
        const std::vector<int> &support_cnt
    ) {
        const int n_bins = static_cast<int>(pos_mass.size());
        std::vector<int> endpoints;
        endpoints.reserve((size_t)n_bins + 1);
        endpoints.push_back(0);

        int prev_nonempty = 0;
        int prev_majority = -1;
        for (int j = 1; j <= n_bins; ++j) {
            const int support = support_cnt[(size_t)(j - 1)];
            if (support <= 0) {
                continue;
            }
            const double pos = pos_mass[(size_t)(j - 1)];
            const double neg = neg_mass[(size_t)(j - 1)];
            const int majority = (pos > neg) ? 1 : 0;
            if (prev_nonempty > 0 && majority != prev_majority && endpoints.back() != prev_nonempty) {
                endpoints.push_back(prev_nonempty);
            }
            prev_nonempty = j;
            prev_majority = majority;
        }
        if (endpoints.back() != n_bins) {
            endpoints.push_back(n_bins);
        }
        return endpoints;
    }

    static int add_gini_cut_endpoints(
        std::vector<int> &endpoints,
        const std::vector<int> &cuts,
        int n_bins,
        int endpoint_budget
    ) {
        if (endpoint_budget <= 0 || cuts.empty() || endpoints.empty() || n_bins <= 1) {
            return 0;
        }
        std::vector<int> extras;
        extras.reserve(cuts.size());
        for (int cut : cuts) {
            const int endpoint = cut + 1;
            if (endpoint <= 0 || endpoint >= n_bins) {
                continue;
            }
            if (std::binary_search(endpoints.begin(), endpoints.end(), endpoint)) {
                continue;
            }
            if (std::find(extras.begin(), extras.end(), endpoint) != extras.end()) {
                continue;
            }
            extras.push_back(endpoint);
        }
        if (extras.empty()) {
            return 0;
        }
        std::sort(extras.begin(), extras.end());
        if ((int)extras.size() > endpoint_budget) {
            extras.resize((size_t)endpoint_budget);
        }
        for (int endpoint : extras) {
            endpoints.push_back(endpoint);
        }
        std::sort(endpoints.begin(), endpoints.end());
        endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
        return (int)extras.size();
    }

    bool solve_projected_leaf_partition_running_minima(
        const std::vector<int> &pos_bins,
        const std::vector<int> &neg_bins,
        int q_max,
        std::vector<std::pair<int, int>> &best_intervals
    ) const {
        best_intervals.clear();
        const int n_bins = (int)pos_bins.size();
        if (n_bins <= 1) {
            return false;
        }

        std::vector<double> pos_mass((size_t)n_bins, 0.0);
        std::vector<double> neg_mass((size_t)n_bins, 0.0);
        std::vector<int> support_cnt((size_t)n_bins, 0);
        for (int b = 0; b < n_bins; ++b) {
            pos_mass[(size_t)b] = static_cast<double>(pos_bins[(size_t)b]);
            neg_mass[(size_t)b] = static_cast<double>(neg_bins[(size_t)b]);
            support_cnt[(size_t)b] = pos_bins[(size_t)b] + neg_bins[(size_t)b];
        }
        const std::vector<int> endpoints = build_majority_shift_endpoints(pos_mass, neg_mass, support_cnt);
        if (endpoints.size() < 2 || endpoints.front() != 0 || endpoints.back() != n_bins) {
            return false;
        }

        const int t_max = (int)endpoints.size() - 1;
        q_max = std::min(q_max, t_max);
        if (q_max < 2) {
            return false;
        }

        std::vector<int> prefix_support((size_t)n_bins + 1, 0);
        std::vector<int> prefix_pos((size_t)n_bins + 1, 0);
        std::vector<int> prefix_diff((size_t)n_bins + 1, 0);
        for (int b = 0; b < n_bins; ++b) {
            const int total = pos_bins[(size_t)b] + neg_bins[(size_t)b];
            prefix_support[(size_t)(b + 1)] = prefix_support[(size_t)b] + total;
            prefix_pos[(size_t)(b + 1)] = prefix_pos[(size_t)b] + pos_bins[(size_t)b];
            prefix_diff[(size_t)(b + 1)] = prefix_diff[(size_t)b] + (pos_bins[(size_t)b] - neg_bins[(size_t)b]);
        }

        std::vector<int> support_by_t((size_t)t_max + 1, 0);
        std::vector<int> diff_by_t((size_t)t_max + 1, 0);
        for (int t = 0; t <= t_max; ++t) {
            const int endpoint = endpoints[(size_t)t];
            support_by_t[(size_t)t] = prefix_support[(size_t)endpoint];
            diff_by_t[(size_t)t] = prefix_diff[(size_t)endpoint];
        }

        std::vector<std::vector<int>> parent(
            (size_t)q_max + 1,
            std::vector<int>((size_t)t_max + 1, -1));
        std::vector<double> dp2_prev((size_t)t_max + 1, kInfinity);
        dp2_prev[0] = 0.0;

        int best_groups = -1;
        double best_objective = kInfinity;

        for (int groups = 1; groups <= q_max; ++groups) {
            check_timeout();
            std::vector<double> dp2_cur((size_t)t_max + 1, kInfinity);

            double a_min = kInfinity;  // min(g[k] + D[k])
            double b_min = kInfinity;  // min(g[k] - D[k])
            int a_arg = -1;
            int b_arg = -1;
            int k_ptr = std::max(0, groups - 1);

            for (int t = groups; t <= t_max; ++t) {
                const int support_t = support_by_t[(size_t)t];
                if (support_t < groups * min_child_size_) {
                    continue;
                }

                while (k_ptr <= t - 1 && support_by_t[(size_t)k_ptr] <= support_t - min_child_size_) {
                    const double prev = dp2_prev[(size_t)k_ptr];
                    if (std::isfinite(prev)) {
                        const double g = prev - static_cast<double>(support_by_t[(size_t)k_ptr]);
                        const double d_k = static_cast<double>(diff_by_t[(size_t)k_ptr]);
                        const double a_val = g + d_k;
                        const double b_val = g - d_k;
                        if (a_val < a_min) {
                            a_min = a_val;
                            a_arg = k_ptr;
                        }
                        if (b_val < b_min) {
                            b_min = b_val;
                            b_arg = k_ptr;
                        }
                    }
                    ++k_ptr;
                }

                if (!(std::isfinite(a_min) || std::isfinite(b_min))) {
                    continue;
                }

                const double d_t = static_cast<double>(diff_by_t[(size_t)t]);
                double best_inner = a_min - d_t;
                int best_prev = a_arg;
                if (b_min + d_t < best_inner) {
                    best_inner = b_min + d_t;
                    best_prev = b_arg;
                }
                if (best_prev < 0) {
                    continue;
                }

                const double dp2_val = static_cast<double>(support_t) + best_inner;
                if (!std::isfinite(dp2_val)) {
                    continue;
                }
                dp2_cur[(size_t)t] = dp2_val;
                parent[(size_t)groups][(size_t)t] = best_prev;
            }

            if (groups >= 2 && std::isfinite(dp2_cur[(size_t)t_max])) {
                const double misclassification =
                    dp2_cur[(size_t)t_max] / (2.0 * static_cast<double>(n_rows_));
                const double objective = misclassification +
                                         static_cast<double>(groups) * regularization_ +
                                         split_penalty_for_groups(groups);
                if (objective < best_objective) {
                    best_objective = objective;
                    best_groups = groups;
                }
            }

            dp2_prev = std::move(dp2_cur);
        }

        if (best_groups < 2) {
            return false;
        }

        int t = t_max;
        int groups = best_groups;
        while (groups > 0) {
            int prev_t = 0;
            if (groups > 1) {
                prev_t = parent[(size_t)groups][(size_t)t];
                if (prev_t < 0) {
                    best_intervals.clear();
                    return false;
                }
            }
            const int left = endpoints[(size_t)prev_t];
            const int right = endpoints[(size_t)t] - 1;
            if (left < 0 || right < left || right >= n_bins) {
                best_intervals.clear();
                return false;
            }
            if (prefix_support[(size_t)(right + 1)] - prefix_support[(size_t)left] < min_child_size_) {
                best_intervals.clear();
                return false;
            }
            best_intervals.push_back({left, right});
            t = prev_t;
            --groups;
        }
        std::reverse(best_intervals.begin(), best_intervals.end());
        return true;
    }

    bool evaluate_feature_dp_depth_two(
        const OrderedBins &bins,
        int feature,
        int fallback_prediction,
        int q_max,
        const std::vector<int> &pos_bins,
        const std::vector<int> &neg_bins,
        double &split_lb,
        double &split_ub,
        std::shared_ptr<Node> &split_tree
    ) const {
        const int n_bins = (int)bins.values.size();
        if (n_bins <= 1) {
            return false;
        }

        std::vector<std::pair<int, int>> best_intervals;
        if (!solve_projected_leaf_partition_running_minima(pos_bins, neg_bins, q_max, best_intervals)) {
            return false;
        }
        if (best_intervals.size() < 2) {
            return false;
        }

        std::vector<int> prefix_pos((size_t)n_bins + 1, 0);
        std::vector<double> prefix_pos_w((size_t)n_bins + 1, 0.0);
        std::vector<double> prefix_neg_w((size_t)n_bins + 1, 0.0);
        for (int b = 0; b < n_bins; ++b) {
            prefix_pos[(size_t)(b + 1)] = prefix_pos[(size_t)b] + pos_bins[(size_t)b];
            double bin_pos_w = 0.0;
            double bin_neg_w = 0.0;
            for (int idx : bins.members[(size_t)b]) {
                if (y_[(size_t)idx] == 1) {
                    bin_pos_w += sample_weight_[(size_t)idx];
                } else {
                    bin_neg_w += sample_weight_[(size_t)idx];
                }
            }
            prefix_pos_w[(size_t)(b + 1)] = prefix_pos_w[(size_t)b] + bin_pos_w;
            prefix_neg_w[(size_t)(b + 1)] = prefix_neg_w[(size_t)b] + bin_neg_w;
        }

        std::vector<std::vector<std::shared_ptr<Node>>> interval_trees(
            (size_t)n_bins,
            std::vector<std::shared_ptr<Node>>((size_t)n_bins, nullptr));

        double interval_cost = 0.0;
        for (const auto &interval : best_intervals) {
            const int left = interval.first;
            const int right = interval.second;

            const int total = bins.prefix_counts[(size_t)(right + 1)] - bins.prefix_counts[(size_t)left];
            const int positives = prefix_pos[(size_t)(right + 1)] - prefix_pos[(size_t)left];
            const int negatives = total - positives;
            const double positives_w = prefix_pos_w[(size_t)(right + 1)] - prefix_pos_w[(size_t)left];
            const double negatives_w = prefix_neg_w[(size_t)(right + 1)] - prefix_neg_w[(size_t)left];
            const int prediction = (positives_w >= negatives_w) ? 1 : 0;
            const double mistakes_w = (prediction == 1) ? negatives_w : positives_w;

            auto leaf = std::make_shared<Node>();
            leaf->is_leaf = true;
            leaf->prediction = prediction;
            leaf->n_samples = total;
            leaf->neg_count = negatives;
            leaf->pos_count = positives;
            leaf->loss = mistakes_w + regularization_;

            interval_trees[(size_t)left][(size_t)right] = leaf;
            interval_cost += leaf->loss;
        }

        split_tree = build_internal_node(
            feature,
            bins,
            best_intervals,
            interval_trees,
            fallback_prediction,
            (int)bins.prefix_counts[(size_t)n_bins]);
        if (!split_tree) {
            return false;
        }

        const int groups = (int)best_intervals.size();
        const double objective = interval_cost + split_penalty_for_groups(groups);
        split_lb = objective;
        split_ub = objective;
        return true;
    }

    PartitionResult optimize_partition_full(
        const std::vector<std::vector<double>> &cost,
        const std::vector<std::vector<bool>> &valid,
        int q_max,
        bool keep_path
    ) const {
        PartitionResult result;
        const int n = (int)cost.size();
        if (n <= 1) {
            return result;
        }

        q_max = std::min(q_max, n);
        if (q_max < 2) {
            return result;
        }

        std::vector<std::vector<double>> dp(q_max + 1, std::vector<double>(n + 1, kInfinity));
        std::vector<std::vector<int>> parent;
        if (keep_path) {
            parent.assign(q_max + 1, std::vector<int>(n + 1, -1));
        }

        dp[0][0] = 0.0;

        for (int q = 1; q <= q_max; ++q) {
            for (int t = 1; t <= n; ++t) {
                const int p_start = q - 1;
                for (int p = p_start; p <= t - 1; ++p) {
                    if (!std::isfinite(dp[q - 1][p])) {
                        continue;
                    }
                    if (!valid[p][t - 1]) {
                        continue;
                    }
                    double interval_cost = cost[p][t - 1];
                    if (!std::isfinite(interval_cost)) {
                        continue;
                    }
                    double candidate = dp[q - 1][p] + interval_cost;
                    if (candidate < dp[q][t]) {
                        dp[q][t] = candidate;
                        if (keep_path) {
                            parent[q][t] = p;
                        }
                    }
                }
            }
        }

        int best_q = -1;
        double best_cost = kInfinity;
        for (int q = 2; q <= q_max; ++q) {
            if (dp[q][n] < best_cost) {
                best_cost = dp[q][n];
                best_q = q;
            }
        }

        if (best_q < 0) {
            return result;
        }

        result.feasible = true;
        result.cost = best_cost;
        result.groups = best_q;

        if (!keep_path) {
            return result;
        }

        int t = n;
        int q = best_q;
        while (q > 0) {
            int p = parent[q][t];
            if (p < 0) {
                result.feasible = false;
                result.intervals.clear();
                return result;
            }
            result.intervals.push_back({p, t - 1});
            t = p;
            --q;
        }
        std::reverse(result.intervals.begin(), result.intervals.end());

        return result;
    }

    ProjectedPartitionResult optimize_partition_projected(
        const std::vector<std::vector<double>> &cost,
        const std::vector<std::vector<bool>> &valid,
        int q_max,
        bool keep_path,
        ProjectedDpWorkspace *workspace = nullptr
    ) const {
        ProjectedPartitionResult result;
        const int t_max = (int)cost.size() - 1;
        if (t_max <= 1) {
            return result;
        }
        if ((int)valid.size() != t_max + 1) {
            return result;
        }
        q_max = std::min(q_max, t_max);
        if (q_max < 2) {
            return result;
        }

        std::vector<std::vector<double>> local_dp;
        std::vector<std::vector<int>> local_parent;
        std::vector<std::vector<double>> &dp = workspace ? workspace->dp : local_dp;
        dp.resize((size_t)q_max + 1);
        for (int q = 0; q <= q_max; ++q) {
            dp[(size_t)q].assign((size_t)t_max + 1, kInfinity);
        }

        std::vector<std::vector<int>> *parent_ptr = nullptr;
        if (keep_path) {
            std::vector<std::vector<int>> &parent = workspace ? workspace->parent : local_parent;
            parent.resize((size_t)q_max + 1);
            for (int q = 0; q <= q_max; ++q) {
                parent[(size_t)q].assign((size_t)t_max + 1, -1);
            }
            parent_ptr = &parent;
        }

        dp[(size_t)0][(size_t)0] = 0.0;
        for (int q = 1; q <= q_max; ++q) {
            for (int t = q; t <= t_max; ++t) {
                for (int p = q - 1; p <= t - 1; ++p) {
                    if (!std::isfinite(dp[(size_t)(q - 1)][(size_t)p])) {
                        continue;
                    }
                    if (!valid[p][t]) {
                        continue;
                    }

                    const double interval_cost = cost[p][t];
                    if (!std::isfinite(interval_cost)) {
                        continue;
                    }
                    const double candidate = dp[(size_t)(q - 1)][(size_t)p] + interval_cost;
                    if (candidate < dp[(size_t)q][(size_t)t]) {
                        dp[(size_t)q][(size_t)t] = candidate;
                        if (parent_ptr != nullptr) {
                            (*parent_ptr)[(size_t)q][(size_t)t] = p;
                        }
                    }
                }
            }
        }

        int best_q = -1;
        double best_cost = kInfinity;
        for (int q = 2; q <= q_max; ++q) {
            if (dp[(size_t)q][(size_t)t_max] < best_cost) {
                best_cost = dp[(size_t)q][(size_t)t_max];
                best_q = q;
            }
        }
        if (best_q < 0) {
            return result;
        }

        result.feasible = true;
        result.cost = best_cost;
        result.groups = best_q;
        if (!keep_path) {
            return result;
        }
        if (parent_ptr == nullptr) {
            return result;
        }

        int t = t_max;
        int q = best_q;
        while (q > 0) {
            const int p = (*parent_ptr)[(size_t)q][(size_t)t];
            if (p < 0) {
                result.feasible = false;
                result.intervals.clear();
                return result;
            }
            result.intervals.push_back({p, t});
            t = p;
            --q;
        }
        std::reverse(result.intervals.begin(), result.intervals.end());
        return result;
    }

    std::shared_ptr<Node> build_internal_node(
        int feature,
        const OrderedBins &bins,
        const std::vector<std::pair<int, int>> &intervals,
        const std::vector<std::vector<std::shared_ptr<Node>>> &interval_trees,
        int fallback_prediction,
        int n_samples
    ) const {
        if (intervals.empty()) {
            return nullptr;
        }

        auto internal = std::make_shared<Node>();
        internal->is_leaf = false;
        internal->feature = feature;
        internal->fallback_prediction = fallback_prediction;
        internal->n_samples = n_samples;
        internal->group_count = (int)intervals.size();

        int largest_child_size = -1;
        int fallback_bin = -1;

        for (const auto &interval : intervals) {
            int left = interval.first;
            int right = interval.second;

            auto child = interval_trees[left][right];
            if (!child) {
                return nullptr;
            }

            int child_size = bins.prefix_counts[right + 1] - bins.prefix_counts[left];
            if (child_size < min_child_size_) {
                throw std::runtime_error(
                    "RUSH invariant violated: partition reconstruction produced child smaller than min_child_size.");
            }
            if (child_size > largest_child_size) {
                largest_child_size = child_size;
                fallback_bin = bins.values[left];
            }

            std::vector<std::pair<int, int>> spans;
            spans.reserve((size_t)(right - left + 1));
            int span_lo = bins.values[(size_t)left];
            int span_hi = span_lo;
            for (int pos = left + 1; pos <= right; ++pos) {
                const int bin_value = bins.values[(size_t)pos];
                if (bin_value == span_hi + 1) {
                    span_hi = bin_value;
                } else {
                    spans.push_back({span_lo, span_hi});
                    span_lo = bin_value;
                    span_hi = bin_value;
                }
            }
            spans.push_back({span_lo, span_hi});
            internal->group_bin_spans.push_back(std::move(spans));
            internal->group_nodes.push_back(child);
        }

        if (fallback_bin < 0 && !bins.values.empty()) {
            fallback_bin = bins.values.front();
        }
        internal->fallback_bin = fallback_bin;

        return internal;
    }

    std::shared_ptr<Node> build_internal_node_projected(
        int feature,
        const OrderedBins &bins,
        const std::vector<int> &endpoints,
        const std::vector<std::pair<int, int>> &projected_intervals,
        const std::vector<std::vector<std::shared_ptr<Node>>> &projected_trees,
        int fallback_prediction,
        int n_samples
    ) const {
        if (projected_intervals.empty()) {
            return nullptr;
        }

        auto internal = std::make_shared<Node>();
        internal->is_leaf = false;
        internal->feature = feature;
        internal->fallback_prediction = fallback_prediction;
        internal->n_samples = n_samples;
        internal->group_count = (int)projected_intervals.size();

        int largest_child_size = -1;
        int fallback_bin = -1;
        const int n_bins = (int)bins.values.size();
        const int t_max = (int)endpoints.size() - 1;

        for (const auto &interval : projected_intervals) {
            const int p = interval.first;
            const int t = interval.second;
            if (p < 0 || t <= p || p > t_max || t > t_max) {
                return nullptr;
            }
            const int left = endpoints[(size_t)p];
            const int right = endpoints[(size_t)t] - 1;
            if (left < 0 || right < left || right >= n_bins) {
                return nullptr;
            }

            auto child = projected_trees[(size_t)p][(size_t)t];
            if (!child) {
                return nullptr;
            }

            const int child_size = bins.prefix_counts[(size_t)(right + 1)] - bins.prefix_counts[(size_t)left];
            if (child_size < min_child_size_) {
                throw std::runtime_error(
                    "RUSH invariant violated: projected partition reconstruction produced child smaller than min_child_size.");
            }
            if (child_size > largest_child_size) {
                largest_child_size = child_size;
                fallback_bin = bins.values[(size_t)left];
            }

            std::vector<std::pair<int, int>> spans;
            spans.reserve((size_t)(right - left + 1));
            int span_lo = bins.values[(size_t)left];
            int span_hi = span_lo;
            for (int pos = left + 1; pos <= right; ++pos) {
                const int bin_value = bins.values[(size_t)pos];
                if (bin_value == span_hi + 1) {
                    span_hi = bin_value;
                } else {
                    spans.push_back({span_lo, span_hi});
                    span_lo = bin_value;
                    span_hi = bin_value;
                }
            }
            spans.push_back({span_lo, span_hi});
            internal->group_bin_spans.push_back(std::move(spans));
            internal->group_nodes.push_back(child);
        }

        if (fallback_bin < 0 && !bins.values.empty()) {
            fallback_bin = bins.values.front();
        }
        internal->fallback_bin = fallback_bin;
        return internal;
    }

    bool build_rush_feature_root_cache(
        const std::vector<int> &indices,
        const SubproblemStats &node_stats,
        int feature,
        int current_depth,
        int fallback_prediction,
        RushFeatureRootCache &out_cache
    ) {
        out_cache = RushFeatureRootCache{};

        OrderedBins bins;
        if (!build_ordered_bins(indices, feature, bins)) {
            return false;
        }

        const int n_bins = (int)bins.values.size();
        const int q_base = max_groups_for_bins(n_bins);
        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        const int q_effective = std::min(q_base, q_support);
        if (q_effective < 2) {
            return false;
        }

        std::vector<int> pos_cnt((size_t)n_bins, 0);
        std::vector<int> neg_cnt((size_t)n_bins, 0);
        std::vector<int> support_cnt((size_t)n_bins, 0);
        std::vector<double> pos_w((size_t)n_bins, 0.0);
        std::vector<double> neg_w((size_t)n_bins, 0.0);
        for (int bin_pos = 0; bin_pos < n_bins; ++bin_pos) {
            int positives = 0;
            int negatives = 0;
            double pos_mass = 0.0;
            double neg_mass = 0.0;
            for (int idx : bins.members[(size_t)bin_pos]) {
                const int y = y_[(size_t)idx];
                const double w = sample_weight_[(size_t)idx];
                if (y == 1) {
                    ++positives;
                    pos_mass += w;
                } else {
                    ++negatives;
                    neg_mass += w;
                }
            }
            pos_cnt[(size_t)bin_pos] = positives;
            neg_cnt[(size_t)bin_pos] = negatives;
            support_cnt[(size_t)bin_pos] = positives + negatives;
            pos_w[(size_t)bin_pos] = pos_mass;
            neg_w[(size_t)bin_pos] = neg_mass;
        }

        std::vector<int> endpoints = build_majority_shift_endpoints(pos_w, neg_w, support_cnt);
        if (endpoints.size() < 2 || endpoints.front() != 0 || endpoints.back() != n_bins) {
            return false;
        }
        const int t_max = (int)endpoints.size() - 1;
        const int q_projected = std::min(q_effective, t_max);
        if (q_projected < 2) {
            return false;
        }

        std::vector<int> pos_cnt_prefix((size_t)n_bins + 1, 0);
        std::vector<int> neg_cnt_prefix((size_t)n_bins + 1, 0);
        std::vector<int> s_cnt_prefix((size_t)n_bins + 1, 0);
        std::vector<double> pos_w_prefix((size_t)n_bins + 1, 0.0);
        std::vector<double> neg_w_prefix((size_t)n_bins + 1, 0.0);
        std::vector<double> s_w_prefix((size_t)n_bins + 1, 0.0);
        for (int b = 0; b < n_bins; ++b) {
            pos_cnt_prefix[(size_t)(b + 1)] = pos_cnt_prefix[(size_t)b] + pos_cnt[(size_t)b];
            neg_cnt_prefix[(size_t)(b + 1)] = neg_cnt_prefix[(size_t)b] + neg_cnt[(size_t)b];
            s_cnt_prefix[(size_t)(b + 1)] = s_cnt_prefix[(size_t)b] + support_cnt[(size_t)b];
            pos_w_prefix[(size_t)(b + 1)] = pos_w_prefix[(size_t)b] + pos_w[(size_t)b];
            neg_w_prefix[(size_t)(b + 1)] = neg_w_prefix[(size_t)b] + neg_w[(size_t)b];
            s_w_prefix[(size_t)(b + 1)] = s_w_prefix[(size_t)b] + pos_w[(size_t)b] + neg_w[(size_t)b];
        }

        std::vector<double> epb_bin((size_t)n_bins, 0.0);
        const int max_bin_value = feature_bin_max_[(size_t)feature];
        const size_t dense_size = (size_t)std::max(0, max_bin_value) + 1U;
        if (bin_value_pos_stamp_.size() < dense_size) {
            bin_value_pos_stamp_.resize(dense_size, 0);
            bin_value_pos_.resize(dense_size, -1);
        }
        ++bin_value_pos_token_;
        if (bin_value_pos_token_ == std::numeric_limits<int>::max()) {
            std::fill(bin_value_pos_stamp_.begin(), bin_value_pos_stamp_.end(), 0);
            bin_value_pos_token_ = 1;
        }
        const int pos_token = bin_value_pos_token_;
        for (int p = 0; p < n_bins; ++p) {
            const int v = bins.values[(size_t)p];
            if (v >= 0 && (size_t)v < dense_size) {
                bin_value_pos_stamp_[(size_t)v] = pos_token;
                bin_value_pos_[(size_t)v] = p;
            }
        }
        for (size_t k = 0; k < node_stats.touched_signature_ids.size(); ++k) {
            const int s = node_stats.touched_signature_ids[k];
            const double e = node_stats.touched_signature_epb[k];
            const int bin_val = sig_bin_[(size_t)s][(size_t)feature];
            if (bin_val >= 0 &&
                (size_t)bin_val < dense_size &&
                bin_value_pos_stamp_[(size_t)bin_val] == pos_token) {
                const int pos = bin_value_pos_[(size_t)bin_val];
                epb_bin[(size_t)pos] += e;
            }
        }
        std::vector<double> epb_prefix((size_t)n_bins + 1, 0.0);
        for (int b = 0; b < n_bins; ++b) {
            epb_prefix[(size_t)(b + 1)] = epb_prefix[(size_t)b] + epb_bin[(size_t)b];
        }

        const double uniform_row_weight = (!non_uniform_weights_ && !sample_weight_.empty())
            ? sample_weight_.front()
            : 0.0;
        std::vector<std::vector<bool>> valid(
            (size_t)t_max + 1,
            std::vector<bool>((size_t)t_max + 1, false));
        std::vector<std::vector<double>> ub_leaf_obj(
            (size_t)t_max + 1,
            std::vector<double>((size_t)t_max + 1, kInfinity));
        std::vector<std::vector<double>> lb_init_obj(
            (size_t)t_max + 1,
            std::vector<double>((size_t)t_max + 1, kInfinity));
        std::vector<std::vector<int>> leaf_prediction(
            (size_t)t_max + 1,
            std::vector<int>((size_t)t_max + 1, 0));
        std::vector<std::vector<int>> leaf_support_count(
            (size_t)t_max + 1,
            std::vector<int>((size_t)t_max + 1, 0));
        std::vector<std::vector<int>> leaf_pos_count(
            (size_t)t_max + 1,
            std::vector<int>((size_t)t_max + 1, 0));
        std::vector<std::vector<int>> leaf_neg_count(
            (size_t)t_max + 1,
            std::vector<int>((size_t)t_max + 1, 0));
        std::vector<std::vector<std::shared_ptr<Node>>> projected_trees(
            (size_t)t_max + 1,
            std::vector<std::shared_ptr<Node>>((size_t)t_max + 1, nullptr));

        for (int p = 0; p < t_max; ++p) {
            const int left = endpoints[(size_t)p];
            for (int t = p + 1; t <= t_max; ++t) {
                const int right = endpoints[(size_t)t] - 1;
                const int support = s_cnt_prefix[(size_t)(right + 1)] - s_cnt_prefix[(size_t)left];
                if (support < min_child_size_) {
                    continue;
                }
                valid[(size_t)p][(size_t)t] = true;

                const int p_cnt = pos_cnt_prefix[(size_t)(right + 1)] - pos_cnt_prefix[(size_t)left];
                const int n_cnt = neg_cnt_prefix[(size_t)(right + 1)] - neg_cnt_prefix[(size_t)left];
                const double p_w = pos_w_prefix[(size_t)(right + 1)] - pos_w_prefix[(size_t)left];
                const double n_w = neg_w_prefix[(size_t)(right + 1)] - neg_w_prefix[(size_t)left];
                const double leaf_obj = regularization_ + std::min(p_w, n_w);
                ub_leaf_obj[(size_t)p][(size_t)t] = leaf_obj;

                const int prediction = (p_w >= n_w) ? 1 : 0;
                leaf_prediction[(size_t)p][(size_t)t] = prediction;
                leaf_support_count[(size_t)p][(size_t)t] = support;
                leaf_pos_count[(size_t)p][(size_t)t] = p_cnt;
                leaf_neg_count[(size_t)p][(size_t)t] = n_cnt;

                const double epb_mis = epb_prefix[(size_t)(right + 1)] - epb_prefix[(size_t)left];
                double msb_mis = 0.0;
                if (!non_uniform_weights_) {
                    const int minor_cnt = std::min(p_cnt, n_cnt);
                    const int msb_cnt = std::min(minor_cnt, std::max(0, min_child_size_ - minor_cnt));
                    msb_mis = static_cast<double>(msb_cnt) * uniform_row_weight;
                }
                const double static_mis_lb = std::max(epb_mis, msb_mis);
                if (support < 2 * min_child_size_ || leaf_obj <= 2.0 * regularization_ + kEpsCert) {
                    lb_init_obj[(size_t)p][(size_t)t] = leaf_obj;
                } else {
                    const double lb_spb = std::min(leaf_obj, static_mis_lb + 2.0 * regularization_);
                    lb_init_obj[(size_t)p][(size_t)t] = std::max(regularization_, lb_spb);
                }
            }
        }

        ProjectedDpWorkspace &projected_ws = projected_dp_workspace_for_depth(current_depth);
        ProjectedPartitionResult lb_partition =
            optimize_partition_projected(lb_init_obj, valid, q_projected, false, &projected_ws);
        ProjectedPartitionResult ub_partition =
            optimize_partition_projected(ub_leaf_obj, valid, q_projected, true, &projected_ws);
        if (!ub_partition.feasible) {
            return false;
        }
        double ub0 = ub_partition.cost + split_penalty_for_groups(ub_partition.groups);
        double lb0 = ub0;
        if (lb_partition.feasible) {
            lb0 = lb_partition.cost + split_penalty_for_groups(lb_partition.groups);
            if (lb0 > ub0 + kEpsCert) {
                throw std::runtime_error(
                    "RUSH exact-lazy invariant violated: LB0 prepass produced LB0 > UB0.");
            }
            if (lb0 > ub0) {
                lb0 = ub0;
            }
        }

        for (const auto &interval : ub_partition.intervals) {
            const int p = interval.first;
            const int t = interval.second;
            projected_trees[(size_t)p][(size_t)t] = make_leaf_node(
                leaf_prediction[(size_t)p][(size_t)t],
                leaf_support_count[(size_t)p][(size_t)t],
                leaf_pos_count[(size_t)p][(size_t)t],
                leaf_neg_count[(size_t)p][(size_t)t],
                ub_leaf_obj[(size_t)p][(size_t)t]);
        }
        std::shared_ptr<Node> ub0_tree = build_internal_node_projected(
            feature,
            bins,
            endpoints,
            ub_partition.intervals,
            projected_trees,
            fallback_prediction,
            (int)indices.size());
        if (!ub0_tree) {
            return false;
        }

        out_cache.ready = true;
        out_cache.bins = std::move(bins);
        out_cache.n_bins = n_bins;
        out_cache.q_effective = q_effective;
        out_cache.t_max = t_max;
        out_cache.q_projected = q_projected;
        out_cache.endpoints = std::move(endpoints);
        out_cache.pos_cnt_prefix = std::move(pos_cnt_prefix);
        out_cache.neg_cnt_prefix = std::move(neg_cnt_prefix);
        out_cache.s_cnt_prefix = std::move(s_cnt_prefix);
        out_cache.pos_w_prefix = std::move(pos_w_prefix);
        out_cache.neg_w_prefix = std::move(neg_w_prefix);
        out_cache.s_w_prefix = std::move(s_w_prefix);
        out_cache.epb_prefix = std::move(epb_prefix);
        out_cache.ub0 = ub0;
        out_cache.lb0 = lb0;
        out_cache.ub0_tree = ub0_tree;
        return true;
    }

    bool evaluate_feature_dp_legacy(
        const std::vector<int> &indices,
        int feature,
        int depth_remaining,
        int fallback_prediction,
        double &split_lb,
        double &split_ub,
        std::shared_ptr<Node> &split_tree
    ) {
        OrderedBins bins;
        if (!build_ordered_bins(indices, feature, bins)) {
            return false;
        }

        const int n_bins = (int)bins.values.size();
        const int q_base = max_groups_for_bins(n_bins);
        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        const int q_effective = std::min(q_base, q_support);
        if (q_effective < 2) {
            return false;
        }

        std::vector<int> pos_bins;
        std::vector<int> neg_bins;
        auto ensure_bin_label_counts = [&]() {
            if (!pos_bins.empty() || !neg_bins.empty()) {
                return;
            }
            pos_bins.assign((size_t)n_bins, 0);
            neg_bins.assign((size_t)n_bins, 0);
            for (int bin_pos = 0; bin_pos < n_bins; ++bin_pos) {
                int positives = 0;
                for (int idx : bins.members[(size_t)bin_pos]) {
                    positives += y_[(size_t)idx];
                }
                const int total = (int)bins.members[(size_t)bin_pos].size();
                pos_bins[(size_t)bin_pos] = positives;
                neg_bins[(size_t)bin_pos] = total - positives;
            }
        };

        if (depth_remaining == 2 && !non_uniform_weights_) {
            ensure_bin_label_counts();
            return evaluate_feature_dp_depth_two(
                bins,
                feature,
                fallback_prediction,
                q_effective,
                pos_bins,
                neg_bins,
                split_lb,
                split_ub,
                split_tree);
        }

        std::vector<int> rush_endpoints;
        const std::vector<int> *rush_endpoints_ptr = nullptr;
        if (partition_strategy_ == kPartitionRushDp) {
            ensure_bin_label_counts();
            std::vector<double> pos_mass((size_t)n_bins, 0.0);
            std::vector<double> neg_mass((size_t)n_bins, 0.0);
            std::vector<int> support_cnt((size_t)n_bins, 0);
            for (int b = 0; b < n_bins; ++b) {
                double p_w = 0.0;
                double n_w = 0.0;
                for (int idx : bins.members[(size_t)b]) {
                    if (y_[(size_t)idx] == 1) {
                        p_w += sample_weight_[(size_t)idx];
                    } else {
                        n_w += sample_weight_[(size_t)idx];
                    }
                }
                pos_mass[(size_t)b] = p_w;
                neg_mass[(size_t)b] = n_w;
                support_cnt[(size_t)b] = pos_bins[(size_t)b] + neg_bins[(size_t)b];
            }
            rush_endpoints = build_majority_shift_endpoints(pos_mass, neg_mass, support_cnt);
            rush_endpoints_ptr = &rush_endpoints;
        }

        if (rush_endpoints_ptr != nullptr) {
            const std::vector<int> &endpoints = *rush_endpoints_ptr;
            const int t_max = (int)endpoints.size() - 1;
            const int q_projected = std::min(q_effective, t_max);
            if (q_projected < 2) {
                return false;
            }

            std::vector<std::vector<bool>> projected_valid(
                (size_t)t_max + 1,
                std::vector<bool>((size_t)t_max + 1, false));
            std::vector<std::vector<double>> projected_lb(
                (size_t)t_max + 1,
                std::vector<double>((size_t)t_max + 1, kInfinity));
            std::vector<std::vector<double>> projected_ub(
                (size_t)t_max + 1,
                std::vector<double>((size_t)t_max + 1, kInfinity));
            std::vector<std::vector<std::shared_ptr<Node>>> projected_trees(
                (size_t)t_max + 1,
                std::vector<std::shared_ptr<Node>>((size_t)t_max + 1, nullptr));

            for (int p = 0; p < t_max; ++p) {
                check_timeout();
                const int left = endpoints[(size_t)p];
                std::vector<int> subset_sorted;
                subset_sorted.reserve((size_t)(bins.prefix_counts[(size_t)n_bins] - bins.prefix_counts[(size_t)left]));
                int prev_right = left - 1;
                for (int t = p + 1; t <= t_max; ++t) {
                    const int right = endpoints[(size_t)t] - 1;
                    for (int b = prev_right + 1; b <= right; ++b) {
                        append_sorted_members(subset_sorted, bins.members[(size_t)b]);
                    }
                    prev_right = right;

                    const int child_count = bins.prefix_counts[(size_t)(right + 1)] - bins.prefix_counts[(size_t)left];
                    if (child_count < min_child_size_) {
                        continue;
                    }

                    ++dp_interval_evals_;
                    BoundResult child = solve_subproblem(subset_sorted, depth_remaining - 1);
                    projected_valid[(size_t)p][(size_t)t] = true;
                    projected_lb[(size_t)p][(size_t)t] = child.lb;
                    projected_ub[(size_t)p][(size_t)t] = child.ub;
                    projected_trees[(size_t)p][(size_t)t] = child.tree;
                }
            }

            ProjectedPartitionResult lb_partition =
                optimize_partition_projected(projected_lb, projected_valid, q_projected, false);
            ProjectedPartitionResult ub_partition =
                optimize_partition_projected(projected_ub, projected_valid, q_projected, true);
            if (!ub_partition.feasible) {
                return false;
            }

            split_ub = ub_partition.cost + split_penalty_for_groups(ub_partition.groups);
            if (lb_partition.feasible) {
                split_lb = lb_partition.cost + split_penalty_for_groups(lb_partition.groups);
            } else {
                split_lb = split_ub;
            }
            split_tree = build_internal_node_projected(
                feature,
                bins,
                endpoints,
                ub_partition.intervals,
                projected_trees,
                fallback_prediction,
                (int)indices.size());
            return split_tree != nullptr;
        }

        std::vector<std::vector<bool>> valid((size_t)n_bins, std::vector<bool>((size_t)n_bins, false));
        std::vector<std::vector<double>> interval_lb((size_t)n_bins, std::vector<double>((size_t)n_bins, kInfinity));
        std::vector<std::vector<double>> interval_ub((size_t)n_bins, std::vector<double>((size_t)n_bins, kInfinity));
        std::vector<std::vector<std::shared_ptr<Node>>> interval_trees(
            (size_t)n_bins,
            std::vector<std::shared_ptr<Node>>((size_t)n_bins, nullptr));
        for (int left = 0; left < n_bins; ++left) {
            std::vector<int> subset_sorted;
            subset_sorted.reserve((size_t)(bins.prefix_counts[(size_t)n_bins] - bins.prefix_counts[(size_t)left]));
            for (int right = left; right < n_bins; ++right) {
                append_sorted_members(subset_sorted, bins.members[(size_t)right]);
                const int child_count = bins.prefix_counts[(size_t)(right + 1)] - bins.prefix_counts[(size_t)left];
                if (child_count < min_child_size_) {
                    continue;
                }

                ++dp_interval_evals_;
                BoundResult child = solve_subproblem(subset_sorted, depth_remaining - 1);
                valid[(size_t)left][(size_t)right] = true;
                interval_lb[(size_t)left][(size_t)right] = child.lb;
                interval_ub[(size_t)left][(size_t)right] = child.ub;
                interval_trees[(size_t)left][(size_t)right] = child.tree;
            }
        }

        PartitionResult lb_partition = optimize_partition_full(interval_lb, valid, q_effective, false);
        PartitionResult ub_partition = optimize_partition_full(interval_ub, valid, q_effective, true);
        if (!ub_partition.feasible) {
            return false;
        }

        split_ub = ub_partition.cost + split_penalty_for_groups(ub_partition.groups);
        if (lb_partition.feasible) {
            split_lb = lb_partition.cost + split_penalty_for_groups(lb_partition.groups);
        } else {
            split_lb = split_ub;
        }
        split_tree = build_internal_node(
            feature,
            bins,
            ub_partition.intervals,
            interval_trees,
            fallback_prediction,
            (int)indices.size());
        return split_tree != nullptr;
    }

    bool evaluate_feature_dp_rush_exact_lazy(
        const std::vector<int> &indices,
        const SubproblemStats &node_stats,
        int feature,
        int depth_remaining,
        int current_depth,
        int fallback_prediction,
        double incumbent_ub,
        double &split_lb,
        double &split_ub,
        std::shared_ptr<Node> &split_tree,
        bool &aborted_by_incumbent,
        const RushFeatureRootCache *root_cache
    ) {
        const Clock::time_point exact_lazy_eval_start = rush_profile_enabled_ ? Clock::now() : Clock::time_point{};
        double exact_lazy_child_solve_local_sec = 0.0;
        bool exact_lazy_eval_time_added = false;
        auto add_exact_lazy_eval_time = [&]() {
            if (!rush_profile_enabled_ || exact_lazy_eval_time_added) {
                return;
            }
            const double elapsed_sec =
                std::chrono::duration<double>(Clock::now() - exact_lazy_eval_start).count();
            const double exclusive_sec = std::max(0.0, elapsed_sec - exact_lazy_child_solve_local_sec);
            rush_profile_exact_lazy_eval_sec_ += elapsed_sec;
            rush_profile_exact_lazy_eval_exclusive_sec_ += exclusive_sec;
            if (current_depth == 0) {
                rush_profile_exact_lazy_eval_sec_depth0_ += elapsed_sec;
                rush_profile_exact_lazy_eval_exclusive_sec_depth0_ += exclusive_sec;
            }
            exact_lazy_eval_time_added = true;
        };
        const double uniform_row_weight = (!non_uniform_weights_ && !sample_weight_.empty())
            ? sample_weight_.front()
            : 0.0;
        const OrderedBins *bins_ptr = nullptr;
        const std::vector<int> *endpoints_ptr = nullptr;
        const std::vector<int> *pos_cnt_prefix_ptr = nullptr;
        const std::vector<int> *neg_cnt_prefix_ptr = nullptr;
        const std::vector<int> *s_cnt_prefix_ptr = nullptr;
        const std::vector<double> *pos_w_prefix_ptr = nullptr;
        const std::vector<double> *neg_w_prefix_ptr = nullptr;
        const std::vector<double> *s_w_prefix_ptr = nullptr;
        const std::vector<double> *epb_prefix_ptr = nullptr;
        int n_bins = 0;
        int t_max = 0;
        int q_projected = 0;

        OrderedBins bins_storage;
        std::vector<int> endpoints_storage;
        std::vector<int> pos_cnt_prefix_storage;
        std::vector<int> neg_cnt_prefix_storage;
        std::vector<int> s_cnt_prefix_storage;
        std::vector<double> pos_w_prefix_storage;
        std::vector<double> neg_w_prefix_storage;
        std::vector<double> s_w_prefix_storage;
        std::vector<double> epb_prefix_storage;

        const bool use_prebuilt_root_cache =
            (root_cache != nullptr) &&
            root_cache->ready &&
            current_depth == 0 &&
            std::fabs(branch_penalty_) <= kEpsUpdate;
        if (use_prebuilt_root_cache) {
            bins_ptr = &root_cache->bins;
            endpoints_ptr = &root_cache->endpoints;
            pos_cnt_prefix_ptr = &root_cache->pos_cnt_prefix;
            neg_cnt_prefix_ptr = &root_cache->neg_cnt_prefix;
            s_cnt_prefix_ptr = &root_cache->s_cnt_prefix;
            pos_w_prefix_ptr = &root_cache->pos_w_prefix;
            neg_w_prefix_ptr = &root_cache->neg_w_prefix;
            s_w_prefix_ptr = &root_cache->s_w_prefix;
            epb_prefix_ptr = &root_cache->epb_prefix;
            n_bins = root_cache->n_bins;
            t_max = root_cache->t_max;
            q_projected = root_cache->q_projected;
        } else {
            OrderedBins bins;
            if (!build_ordered_bins(indices, feature, bins)) {
                add_exact_lazy_eval_time();
                return false;
            }
            n_bins = (int)bins.values.size();
            const int q_base = max_groups_for_bins(n_bins);
            const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
            const int q_effective = std::min(q_base, q_support);
            if (q_effective < 2) {
                add_exact_lazy_eval_time();
                return false;
            }

            std::vector<int> pos_cnt((size_t)n_bins, 0);
            std::vector<int> neg_cnt((size_t)n_bins, 0);
            std::vector<int> support_cnt((size_t)n_bins, 0);
            std::vector<double> pos_w((size_t)n_bins, 0.0);
            std::vector<double> neg_w((size_t)n_bins, 0.0);
            for (int bin_pos = 0; bin_pos < n_bins; ++bin_pos) {
                int positives = 0;
                int negatives = 0;
                double pos_mass = 0.0;
                double neg_mass = 0.0;
                for (int idx : bins.members[(size_t)bin_pos]) {
                    const int y = y_[(size_t)idx];
                    const double w = sample_weight_[(size_t)idx];
                    if (y == 1) {
                        ++positives;
                        pos_mass += w;
                    } else {
                        ++negatives;
                        neg_mass += w;
                    }
                }
                pos_cnt[(size_t)bin_pos] = positives;
                neg_cnt[(size_t)bin_pos] = negatives;
                support_cnt[(size_t)bin_pos] = positives + negatives;
                pos_w[(size_t)bin_pos] = pos_mass;
                neg_w[(size_t)bin_pos] = neg_mass;
            }

            endpoints_storage = build_majority_shift_endpoints(pos_w, neg_w, support_cnt);
            if (endpoints_storage.size() < 2 || endpoints_storage.front() != 0 || endpoints_storage.back() != n_bins) {
                add_exact_lazy_eval_time();
                return false;
            }
            t_max = (int)endpoints_storage.size() - 1;
            q_projected = std::min(q_effective, t_max);
            if (q_projected < 2) {
                add_exact_lazy_eval_time();
                return false;
            }

            const Clock::time_point table_init_start = rush_profile_enabled_ ? Clock::now() : Clock::time_point{};
            pos_cnt_prefix_storage.assign((size_t)n_bins + 1, 0);
            neg_cnt_prefix_storage.assign((size_t)n_bins + 1, 0);
            s_cnt_prefix_storage.assign((size_t)n_bins + 1, 0);
            pos_w_prefix_storage.assign((size_t)n_bins + 1, 0.0);
            neg_w_prefix_storage.assign((size_t)n_bins + 1, 0.0);
            s_w_prefix_storage.assign((size_t)n_bins + 1, 0.0);
            for (int b = 0; b < n_bins; ++b) {
                pos_cnt_prefix_storage[(size_t)(b + 1)] = pos_cnt_prefix_storage[(size_t)b] + pos_cnt[(size_t)b];
                neg_cnt_prefix_storage[(size_t)(b + 1)] = neg_cnt_prefix_storage[(size_t)b] + neg_cnt[(size_t)b];
                s_cnt_prefix_storage[(size_t)(b + 1)] = s_cnt_prefix_storage[(size_t)b] + support_cnt[(size_t)b];
                pos_w_prefix_storage[(size_t)(b + 1)] = pos_w_prefix_storage[(size_t)b] + pos_w[(size_t)b];
                neg_w_prefix_storage[(size_t)(b + 1)] = neg_w_prefix_storage[(size_t)b] + neg_w[(size_t)b];
                s_w_prefix_storage[(size_t)(b + 1)] = s_w_prefix_storage[(size_t)b] + pos_w[(size_t)b] + neg_w[(size_t)b];
            }

            std::vector<double> epb_bin((size_t)n_bins, 0.0);
            const int max_bin_value = feature_bin_max_[(size_t)feature];
            const size_t dense_size = (size_t)std::max(0, max_bin_value) + 1U;
            if (bin_value_pos_stamp_.size() < dense_size) {
                bin_value_pos_stamp_.resize(dense_size, 0);
                bin_value_pos_.resize(dense_size, -1);
            }
            ++bin_value_pos_token_;
            if (bin_value_pos_token_ == std::numeric_limits<int>::max()) {
                std::fill(bin_value_pos_stamp_.begin(), bin_value_pos_stamp_.end(), 0);
                bin_value_pos_token_ = 1;
            }
            const int pos_token = bin_value_pos_token_;
            for (int p = 0; p < n_bins; ++p) {
                const int v = bins.values[(size_t)p];
                if (v >= 0 && (size_t)v < dense_size) {
                    bin_value_pos_stamp_[(size_t)v] = pos_token;
                    bin_value_pos_[(size_t)v] = p;
                }
            }
            for (size_t k = 0; k < node_stats.touched_signature_ids.size(); ++k) {
                const int s = node_stats.touched_signature_ids[k];
                const double e = node_stats.touched_signature_epb[k];
                const int bin_val = sig_bin_[(size_t)s][(size_t)feature];
                if (bin_val >= 0 &&
                    (size_t)bin_val < dense_size &&
                    bin_value_pos_stamp_[(size_t)bin_val] == pos_token) {
                    const int pos = bin_value_pos_[(size_t)bin_val];
                    epb_bin[(size_t)pos] += e;
                }
            }
            epb_prefix_storage.assign((size_t)n_bins + 1, 0.0);
            for (int b = 0; b < n_bins; ++b) {
                epb_prefix_storage[(size_t)(b + 1)] = epb_prefix_storage[(size_t)b] + epb_bin[(size_t)b];
            }
            if (rush_profile_enabled_) {
                rush_profile_exact_lazy_table_init_sec_ +=
                    std::chrono::duration<double>(Clock::now() - table_init_start).count();
            }

            bins_storage = std::move(bins);
            bins_ptr = &bins_storage;
            endpoints_ptr = &endpoints_storage;
            pos_cnt_prefix_ptr = &pos_cnt_prefix_storage;
            neg_cnt_prefix_ptr = &neg_cnt_prefix_storage;
            s_cnt_prefix_ptr = &s_cnt_prefix_storage;
            pos_w_prefix_ptr = &pos_w_prefix_storage;
            neg_w_prefix_ptr = &neg_w_prefix_storage;
            s_w_prefix_ptr = &s_w_prefix_storage;
            epb_prefix_ptr = &epb_prefix_storage;
        }

        const OrderedBins &bins = *bins_ptr;
        const std::vector<int> &endpoints = *endpoints_ptr;
        const std::vector<int> &pos_cnt_prefix = *pos_cnt_prefix_ptr;
        const std::vector<int> &neg_cnt_prefix = *neg_cnt_prefix_ptr;
        const std::vector<int> &s_cnt_prefix = *s_cnt_prefix_ptr;
        const std::vector<double> &pos_w_prefix = *pos_w_prefix_ptr;
        const std::vector<double> &neg_w_prefix = *neg_w_prefix_ptr;
        const std::vector<double> &s_w_prefix = *s_w_prefix_ptr;
        const std::vector<double> &epb_prefix = *epb_prefix_ptr;

        RushIntervalWorkspace &interval_ws = rush_interval_workspace_for_depth(current_depth);
        auto reset_matrix = [&](auto &matrix, const auto &value) {
            const size_t dim = (size_t)t_max + 1U;
            if (matrix.size() != dim) {
                matrix.resize(dim);
            }
            for (size_t p = 0; p < dim; ++p) {
                if (matrix[p].size() != dim) {
                    matrix[p].resize(dim);
                }
                std::fill(matrix[p].begin(), matrix[p].end(), value);
            }
        };
        reset_matrix(interval_ws.status, IntervalStatus::kInfeasible);
        reset_matrix(interval_ws.valid, false);
        reset_matrix(interval_ws.ub_leaf_obj, kInfinity);
        reset_matrix(interval_ws.static_mis_lb, 0.0);
        reset_matrix(interval_ws.lb_obj, kInfinity);
        reset_matrix(interval_ws.ub_obj, kInfinity);
        reset_matrix(interval_ws.interval_support_w, 0.0);
        reset_matrix(interval_ws.ssb_mis_cur, -kInfinity);
        reset_matrix(interval_ws.leaf_prediction, 0);
        reset_matrix(interval_ws.leaf_support_count, 0);
        reset_matrix(interval_ws.leaf_pos_count, 0);
        reset_matrix(interval_ws.leaf_neg_count, 0);
        reset_matrix(interval_ws.projected_trees, std::shared_ptr<Node>(nullptr));

        auto &status = interval_ws.status;
        auto &valid = interval_ws.valid;
        auto &ub_leaf_obj = interval_ws.ub_leaf_obj;
        auto &static_mis_lb = interval_ws.static_mis_lb;
        auto &lb_obj = interval_ws.lb_obj;
        auto &ub_obj = interval_ws.ub_obj;
        auto &interval_support_w = interval_ws.interval_support_w;
        auto &ssb_mis_cur = interval_ws.ssb_mis_cur;
        auto &leaf_prediction = interval_ws.leaf_prediction;
        auto &leaf_support_count = interval_ws.leaf_support_count;
        auto &leaf_pos_count = interval_ws.leaf_pos_count;
        auto &leaf_neg_count = interval_ws.leaf_neg_count;
        auto &projected_trees = interval_ws.projected_trees;

        for (int p = 0; p < t_max; ++p) {
            const int left = endpoints[(size_t)p];
            for (int t = p + 1; t <= t_max; ++t) {
                const int right = endpoints[(size_t)t] - 1;
                const int support = s_cnt_prefix[(size_t)(right + 1)] - s_cnt_prefix[(size_t)left];
                if (support < min_child_size_) {
                    status[(size_t)p][(size_t)t] = IntervalStatus::kInfeasible;
                    continue;
                }
                valid[(size_t)p][(size_t)t] = true;

                const int p_cnt = pos_cnt_prefix[(size_t)(right + 1)] - pos_cnt_prefix[(size_t)left];
                const int n_cnt = neg_cnt_prefix[(size_t)(right + 1)] - neg_cnt_prefix[(size_t)left];
                const double p_w = pos_w_prefix[(size_t)(right + 1)] - pos_w_prefix[(size_t)left];
                const double n_w = neg_w_prefix[(size_t)(right + 1)] - neg_w_prefix[(size_t)left];
                interval_support_w[(size_t)p][(size_t)t] = p_w + n_w;
                const double minor_w = std::min(p_w, n_w);
                const double leaf_obj = regularization_ + minor_w;
                ub_leaf_obj[(size_t)p][(size_t)t] = leaf_obj;
                ub_obj[(size_t)p][(size_t)t] = leaf_obj;

                const int prediction = (p_w >= n_w) ? 1 : 0;
                leaf_prediction[(size_t)p][(size_t)t] = prediction;
                leaf_support_count[(size_t)p][(size_t)t] = support;
                leaf_pos_count[(size_t)p][(size_t)t] = p_cnt;
                leaf_neg_count[(size_t)p][(size_t)t] = n_cnt;

                const double epb_mis = epb_prefix[(size_t)(right + 1)] - epb_prefix[(size_t)left];
                double msb_mis = 0.0;
                if (!non_uniform_weights_) {
                    const int minor_cnt = std::min(p_cnt, n_cnt);
                    const int msb_cnt = std::min(minor_cnt, std::max(0, min_child_size_ - minor_cnt));
                    msb_mis = static_cast<double>(msb_cnt) * uniform_row_weight;
                }
                static_mis_lb[(size_t)p][(size_t)t] = std::max(epb_mis, msb_mis);

                if (support < 2 * min_child_size_ || leaf_obj <= 2.0 * regularization_ + kEpsCert) {
                    status[(size_t)p][(size_t)t] = IntervalStatus::kCertifiedLeaf;
                    lb_obj[(size_t)p][(size_t)t] = leaf_obj;
                    ub_obj[(size_t)p][(size_t)t] = leaf_obj;
                    continue;
                }

                status[(size_t)p][(size_t)t] = IntervalStatus::kUnrefined;
                const double ssb_obj = regularization_;
                const double ssb_mis = 0.0;
                const double m_base = std::max(static_mis_lb[(size_t)p][(size_t)t], ssb_mis);
                const double lb_spb = std::min(leaf_obj, m_base + 2.0 * regularization_);
                lb_obj[(size_t)p][(size_t)t] = std::max(ssb_obj, lb_spb);
            }
        }
        ProjectedDpWorkspace &projected_ws = projected_dp_workspace_for_depth(current_depth);
        auto recompute_partitions = [&](ProjectedPartitionResult &lb_part, ProjectedPartitionResult &ub_part, double &feat_lb, double &feat_ub) -> bool {
            const Clock::time_point recompute_start = rush_profile_enabled_ ? Clock::now() : Clock::time_point{};
            lb_part = optimize_partition_projected(lb_obj, valid, q_projected, false, &projected_ws);
            ub_part = optimize_partition_projected(ub_obj, valid, q_projected, true, &projected_ws);
            if (!ub_part.feasible) {
                if (rush_profile_enabled_) {
                    rush_profile_exact_lazy_dp_recompute_sec_ +=
                        std::chrono::duration<double>(Clock::now() - recompute_start).count();
                    rush_profile_exact_lazy_dp_recompute_calls_ += 1;
                }
                return false;
            }
            feat_ub = ub_part.cost;
            const double raw_lb = lb_part.feasible ? lb_part.cost : feat_ub;
            if (raw_lb > feat_ub + kEpsCert) {
                throw std::runtime_error(
                    "RUSH exact-lazy invariant violated: projected DP returned feature_LB > feature_UB.");
            }
            feat_lb = (raw_lb > feat_ub) ? feat_ub : raw_lb;
            if (rush_profile_enabled_) {
                rush_profile_exact_lazy_dp_recompute_sec_ +=
                    std::chrono::duration<double>(Clock::now() - recompute_start).count();
                rush_profile_exact_lazy_dp_recompute_calls_ += 1;
            }
            return true;
        };

        ProjectedPartitionResult lb_partition;
        ProjectedPartitionResult ub_partition;
        double feature_lb = kInfinity;
        double feature_ub = kInfinity;
        if (!recompute_partitions(lb_partition, ub_partition, feature_lb, feature_ub)) {
            add_exact_lazy_eval_time();
            return false;
        }
        assert_interval_bound_sanity(valid, status, lb_obj, ub_obj, ub_leaf_obj, t_max);
        assert_feature_bound_sanity(feature_lb, feature_ub);
        if (feature_lb >= incumbent_ub - kEpsCert) {
            aborted_by_incumbent = true;
            add_exact_lazy_eval_time();
            return false;
        }

        auto interval_delta = [&](int p_a, int t_a, int p_b, int t_b) -> double {
            const int left_a = endpoints[(size_t)p_a];
            const int right_a = endpoints[(size_t)t_a] - 1;
            const int left_b = endpoints[(size_t)p_b];
            const int right_b = endpoints[(size_t)t_b] - 1;
            const int u = std::max(left_a, left_b);
            const int v = std::min(right_a, right_b);
            const double support_a = interval_support_w[(size_t)p_a][(size_t)t_a];
            const double support_b = interval_support_w[(size_t)p_b][(size_t)t_b];
            const double intersect_w = (u <= v)
                ? (s_w_prefix[(size_t)(v + 1)] - s_w_prefix[(size_t)u])
                : 0.0;
            const double delta = support_a + support_b - 2.0 * intersect_w;
            return (delta < 0.0) ? 0.0 : delta;
        };

#ifndef NDEBUG
        auto debug_validate_interval_delta = [&]() {
            std::vector<std::pair<int, int>> feasible_intervals;
            feasible_intervals.reserve((size_t)t_max * (size_t)(t_max + 1) / 2U);
            for (int p = 0; p < t_max; ++p) {
                for (int t = p + 1; t <= t_max; ++t) {
                    if (valid[(size_t)p][(size_t)t]) {
                        feasible_intervals.push_back({p, t});
                    }
                }
            }
            if (feasible_intervals.size() < 2) {
                return;
            }

            const int check_count = std::min<int>(6, (int)feasible_intervals.size());
            std::unordered_set<int> rows_a;
            rows_a.reserve(indices.size());
            for (int c = 0; c < check_count; ++c) {
                const auto &a = feasible_intervals[(size_t)((c * 17) % (int)feasible_intervals.size())];
                const auto &b = feasible_intervals[(size_t)((c * 43 + 7) % (int)feasible_intervals.size())];
                const double delta_prefix = interval_delta(a.first, a.second, b.first, b.second);

                rows_a.clear();
                double support_a = 0.0;
                const int left_a = endpoints[(size_t)a.first];
                const int right_a = endpoints[(size_t)a.second] - 1;
                for (int bin = left_a; bin <= right_a; ++bin) {
                    for (int idx : bins.members[(size_t)bin]) {
                        rows_a.insert(idx);
                        support_a += sample_weight_[(size_t)idx];
                    }
                }

                double support_b = 0.0;
                double support_intersect = 0.0;
                const int left_b = endpoints[(size_t)b.first];
                const int right_b = endpoints[(size_t)b.second] - 1;
                for (int bin = left_b; bin <= right_b; ++bin) {
                    for (int idx : bins.members[(size_t)bin]) {
                        const double w = sample_weight_[(size_t)idx];
                        support_b += w;
                        if (rows_a.find(idx) != rows_a.end()) {
                            support_intersect += w;
                        }
                    }
                }

                double delta_materialized = support_a + support_b - 2.0 * support_intersect;
                if (delta_materialized < 0.0) {
                    delta_materialized = 0.0;
                }
                if (std::fabs(delta_prefix - delta_materialized) > 1e-9) {
                    throw std::runtime_error(
                        "RUSH exact-lazy invariant violated: interval delta prefix/materialized mismatch.");
                }
            }
        };
        debug_validate_interval_delta();
#endif

#ifndef NDEBUG
        auto num_unrefined_global = [&]() -> int {
            int count = 0;
            for (int p = 0; p < t_max; ++p) {
                for (int t = p + 1; t <= t_max; ++t) {
                    if (!valid[(size_t)p][(size_t)t]) {
                        continue;
                    }
                    if (status[(size_t)p][(size_t)t] == IntervalStatus::kUnrefined) {
                        ++count;
                    }
                }
            }
            return count;
        };
#endif

        const int batch_grid = t_max + 1;
        if ((int)interval_ws.batch_selected_stamp.size() < batch_grid * batch_grid) {
            interval_ws.batch_selected_stamp.assign((size_t)batch_grid * (size_t)batch_grid, 0);
            interval_ws.batch_selection_token = 0;
        }
        auto &batch_selected_stamp = interval_ws.batch_selected_stamp;
        int &batch_selection_token = interval_ws.batch_selection_token;
        auto next_batch_token = [&]() -> int {
            ++batch_selection_token;
            if (batch_selection_token == std::numeric_limits<int>::max()) {
                std::fill(batch_selected_stamp.begin(), batch_selected_stamp.end(), 0);
                batch_selection_token = 1;
            }
            return batch_selection_token;
        };

        auto add_unrefined_from_partition = [&](const std::vector<std::pair<int, int>> &intervals,
                                                int batch_target,
                                                int batch_token,
                                                std::vector<std::pair<int, int>> &batch) {
            if ((int)batch.size() >= batch_target) {
                return;
            }
            for (const auto &interval : intervals) {
                if ((int)batch.size() >= batch_target) {
                    break;
                }
                const int p = interval.first;
                const int t = interval.second;
                if (status[(size_t)p][(size_t)t] != IntervalStatus::kUnrefined) {
                    continue;
                }
                const size_t cell = (size_t)p * (size_t)(t_max + 1) + (size_t)t;
                if (batch_selected_stamp[(size_t)cell] == batch_token) {
                    continue;
                }
                batch.push_back({p, t});
                batch_selected_stamp[(size_t)cell] = batch_token;
            }
        };

        auto consider_unrefined_candidate = [&](int p,
                                                int t,
                                                int batch_token,
                                                int &best_p,
                                                int &best_t,
                                                bool &found,
                                                double &best_gap,
                                                double &best_support) {
            if (p < 0 || t > t_max || p >= t) {
                return;
            }
            if (status[(size_t)p][(size_t)t] != IntervalStatus::kUnrefined) {
                return;
            }
            if (!valid[(size_t)p][(size_t)t]) {
                return;
            }
            const size_t cell = (size_t)p * (size_t)(t_max + 1) + (size_t)t;
            if (batch_selected_stamp[(size_t)cell] == batch_token) {
                return;
            }
            const double gap = ub_obj[(size_t)p][(size_t)t] - lb_obj[(size_t)p][(size_t)t];
            const double support_w = interval_support_w[(size_t)p][(size_t)t];
            if (!found ||
                gap > best_gap + kEpsUpdate ||
                (std::fabs(gap - best_gap) <= kEpsUpdate &&
                 support_w > best_support + kEpsUpdate)) {
                found = true;
                best_gap = gap;
                best_support = support_w;
                best_p = p;
                best_t = t;
            }
        };

        auto add_unrefined_ub_rescue = [&](const std::vector<std::pair<int, int>> &ub_intervals,
                                           int batch_target,
                                           int batch_token,
                                           std::vector<std::pair<int, int>> &batch) {
            if ((int)batch.size() >= batch_target) {
                return;
            }
            constexpr int kRescueWindow = 8;
            const int picks_before = (int)batch.size();
            while ((int)batch.size() < batch_target) {
                int best_p = -1;
                int best_t = -1;
                bool found = false;
                double best_gap = -kInfinity;
                double best_support = -kInfinity;
                for (const auto &interval : ub_intervals) {
                    const int p = interval.first;
                    const int t = interval.second;

                    consider_unrefined_candidate(p - 1, t, batch_token, best_p, best_t, found, best_gap, best_support);
                    consider_unrefined_candidate(p + 1, t, batch_token, best_p, best_t, found, best_gap, best_support);
                    consider_unrefined_candidate(p, t - 1, batch_token, best_p, best_t, found, best_gap, best_support);
                    consider_unrefined_candidate(p, t + 1, batch_token, best_p, best_t, found, best_gap, best_support);

                    const int t_lo = std::max(p + 1, t - kRescueWindow);
                    const int t_hi = std::min(t_max, t + kRescueWindow);
                    for (int t2 = t_lo; t2 <= t_hi; ++t2) {
                        if (t2 == t) {
                            continue;
                        }
                        consider_unrefined_candidate(p, t2, batch_token, best_p, best_t, found, best_gap, best_support);
                    }

                    const int p_lo = std::max(0, p - kRescueWindow);
                    const int p_hi = std::min(t - 1, p + kRescueWindow);
                    for (int p2 = p_lo; p2 <= p_hi; ++p2) {
                        if (p2 == p) {
                            continue;
                        }
                        consider_unrefined_candidate(p2, t, batch_token, best_p, best_t, found, best_gap, best_support);
                    }
                }
                if (!found) {
                    break;
                }
                batch.push_back({best_p, best_t});
                const size_t cell = (size_t)best_p * (size_t)(t_max + 1) + (size_t)best_t;
                batch_selected_stamp[(size_t)cell] = batch_token;
            }
            rush_ub_rescue_picks_ += std::max(0, (int)batch.size() - picks_before);
        };

        auto add_unrefined_global = [&](int batch_target,
                                        int batch_token,
                                        std::vector<std::pair<int, int>> &batch) {
            while ((int)batch.size() < batch_target) {
                int best_p = -1;
                int best_t = -1;
                bool found = false;
                double best_gap = -kInfinity;
                double best_support = -kInfinity;
                for (int p = 0; p < t_max; ++p) {
                    for (int t = p + 1; t <= t_max; ++t) {
                        consider_unrefined_candidate(
                            p, t, batch_token, best_p, best_t, found, best_gap, best_support);
                    }
                }
                if (!found) {
                    break;
                }
                batch.push_back({best_p, best_t});
                const size_t cell = (size_t)best_p * (size_t)(t_max + 1) + (size_t)best_t;
                batch_selected_stamp[(size_t)cell] = batch_token;
            }
        };
        const bool allow_ub_transfer = (min_child_size_ <= 1);
        if ((int)interval_ws.index_rank.size() != n_rows_) {
            interval_ws.index_rank.assign((size_t)n_rows_, 0);
        }
        auto &index_rank = interval_ws.index_rank;
        for (int pos = 0; pos < (int)indices.size(); ++pos) {
            index_rank[(size_t)indices[(size_t)pos]] = pos;
        }
        if ((int)interval_ws.subset_row_stamp.size() != n_rows_) {
            interval_ws.subset_row_stamp.assign((size_t)n_rows_, 0);
            interval_ws.subset_row_token = 0;
        }
        auto &subset_scratch = interval_ws.subset_scratch;
        subset_scratch.clear();
        subset_scratch.reserve(indices.size());
        auto &subset_row_stamp = interval_ws.subset_row_stamp;
        int &subset_row_token = interval_ws.subset_row_token;
        auto next_subset_token = [&]() -> int {
            ++subset_row_token;
            if (subset_row_token == std::numeric_limits<int>::max()) {
                std::fill(subset_row_stamp.begin(), subset_row_stamp.end(), 0);
                subset_row_token = 1;
            }
            return subset_row_token;
        };
        auto materialize_interval_subset = [&](int p_refine, int t_refine, int support_expected, std::vector<int> &subset_out) {
            const int left = endpoints[(size_t)p_refine];
            const int right = endpoints[(size_t)t_refine] - 1;
            subset_out.clear();
            subset_out.reserve((size_t)support_expected);
            if (support_expected <= 0) {
                return;
            }
            if (support_expected == (int)indices.size()) {
                subset_out = indices;
                return;
            }
            if (left == right) {
                subset_out = bins.members[(size_t)left];
                return;
            }
            const bool use_gather_sort =
                (2LL * static_cast<long long>(support_expected) < static_cast<long long>(indices.size())) &&
                (right > left);
            if (use_gather_sort) {
                for (int b = left; b <= right; ++b) {
                    const auto &members = bins.members[(size_t)b];
                    subset_out.insert(subset_out.end(), members.begin(), members.end());
                }
                std::sort(
                    subset_out.begin(),
                    subset_out.end(),
                    [&](int lhs, int rhs) { return index_rank[(size_t)lhs] < index_rank[(size_t)rhs]; });
            } else {
                const int subset_token = next_subset_token();
                for (int b = left; b <= right; ++b) {
                    const auto &members = bins.members[(size_t)b];
                    for (int idx : members) {
                        subset_row_stamp[(size_t)idx] = subset_token;
                    }
                }
                for (int idx : indices) {
                    if (subset_row_stamp[(size_t)idx] == subset_token) {
                        subset_out.push_back(idx);
                    }
                }
            }
#ifndef NDEBUG
            if ((int)subset_out.size() != support_expected) {
                throw std::runtime_error(
                    "RUSH exact-lazy invariant violated: stamped subset size mismatch for projected interval.");
            }
#endif
        };

        bool stopped_certified = false;
        bool stopped_fully_refined = false;
        // Exact-lazy termination is only safe in three cases:
        // (1) certified feature bounds, (2) incumbent-dominated feature LB,
        // or (3) no feasible unrefined intervals remain globally.
        while (true) {
            check_timeout();
            if (feature_ub - feature_lb <= kEpsCert) {
                stopped_certified = true;
                break;
            }
            if (feature_lb >= incumbent_ub - kEpsCert) {
                aborted_by_incumbent = true;
                add_exact_lazy_eval_time();
                return false;
            }

            int unresolved_on_lb = 0;
            for (const auto &interval : lb_partition.intervals) {
                if (status[(size_t)interval.first][(size_t)interval.second] == IntervalStatus::kUnrefined) {
                    ++unresolved_on_lb;
                }
            }
            int unresolved_on_ub = 0;
            for (const auto &interval : ub_partition.intervals) {
                if (status[(size_t)interval.first][(size_t)interval.second] == IntervalStatus::kUnrefined) {
                    ++unresolved_on_ub;
                }
            }

            constexpr int kMaxBatchRefines = 3;
            const int batch_target =
                std::max(1, std::min(kMaxBatchRefines, std::max(unresolved_on_lb, unresolved_on_ub)));
            const int batch_token = next_batch_token();
            std::vector<std::pair<int, int>> batch;
            batch.reserve((size_t)batch_target);
            add_unrefined_from_partition(lb_partition.intervals, batch_target, batch_token, batch);
            add_unrefined_from_partition(ub_partition.intervals, batch_target, batch_token, batch);
            // UB-rescue targets local competitors around the current UB-optimal partition
            // before paying for a global max-gap scan across the full projected grid.
            if (batch.empty()) {
                add_unrefined_ub_rescue(ub_partition.intervals, batch_target, batch_token, batch);
            }
            if (batch.empty()) {
                const int fallback_before = (int)batch.size();
                add_unrefined_global(batch_target, batch_token, batch);
                rush_global_fallback_picks_ += std::max(0, (int)batch.size() - fallback_before);
            }

            if (batch.empty()) {
                stopped_fully_refined = true;
                break;
            }

            for (size_t anchor_idx = 0; anchor_idx < batch.size(); ++anchor_idx) {
                const int p_refine = batch[anchor_idx].first;
                const int t_refine = batch[anchor_idx].second;
                if (status[(size_t)p_refine][(size_t)t_refine] != IntervalStatus::kUnrefined) {
                    continue;
                }

                const int left_a = endpoints[(size_t)p_refine];
                const int right_a = endpoints[(size_t)t_refine] - 1;
                const int support_a = s_cnt_prefix[(size_t)(right_a + 1)] - s_cnt_prefix[(size_t)left_a];
                ++interval_refinements_attempted_;
                materialize_interval_subset(p_refine, t_refine, support_a, subset_scratch);
                const std::vector<int> *subset_ptr = &subset_scratch;

                ++dp_interval_evals_;
                const long long calls_before = dp_subproblem_calls_;
                const long long states_before = dp_cache_states_;
                const Clock::time_point child_solve_start = Clock::now();
                BoundResult child = solve_subproblem(*subset_ptr, depth_remaining - 1);
                const double child_solve_sec =
                    std::chrono::duration<double>(Clock::now() - child_solve_start).count();
                ++expensive_child_calls_;
                expensive_child_sec_ += child_solve_sec;
                if (exactify_eval_context_depth_ > 0) {
                    ++expensive_child_exactify_calls_;
                    expensive_child_exactify_sec_ += child_solve_sec;
                }
                if (rush_profile_enabled_) {
                    rush_profile_exact_lazy_child_solve_sec_ += child_solve_sec;
                    exact_lazy_child_solve_local_sec += child_solve_sec;
                    if (current_depth == 0) {
                        rush_profile_exact_lazy_child_solve_sec_depth0_ += child_solve_sec;
                    }
                }
                const long long calls_after = dp_subproblem_calls_;
                const long long states_after = dp_cache_states_;
                const long long spawned_calls = std::max(0LL, calls_after - calls_before);
                const long long spawned_states = std::max(0LL, states_after - states_before);
                rush_refinement_child_calls_ += 1;
                rush_refinement_recursive_calls_ += spawned_calls;
                rush_refinement_recursive_unique_states_ += spawned_states;
                status[(size_t)p_refine][(size_t)t_refine] = IntervalStatus::kRefined;
                lb_obj[(size_t)p_refine][(size_t)t_refine] =
                    std::max(lb_obj[(size_t)p_refine][(size_t)t_refine], child.lb);
                ub_obj[(size_t)p_refine][(size_t)t_refine] =
                    std::min(ub_obj[(size_t)p_refine][(size_t)t_refine], child.ub);
                if (lb_obj[(size_t)p_refine][(size_t)t_refine] >
                    ub_obj[(size_t)p_refine][(size_t)t_refine] + kEpsCert) {
                    throw std::runtime_error(
                        "RUSH exact-lazy invariant violated: refined interval LB_obj > UB_obj.");
                }
                if (lb_obj[(size_t)p_refine][(size_t)t_refine] > ub_obj[(size_t)p_refine][(size_t)t_refine]) {
                    lb_obj[(size_t)p_refine][(size_t)t_refine] = ub_obj[(size_t)p_refine][(size_t)t_refine];
                }
                projected_trees[(size_t)p_refine][(size_t)t_refine] = child.tree;

                const double lb_mis_anchor = child.lb_mis;
                const Clock::time_point closure_start = rush_profile_enabled_ ? Clock::now() : Clock::time_point{};
                for (int p = 0; p < t_max; ++p) {
                    for (int t = p + 1; t <= t_max; ++t) {
                        if (!valid[(size_t)p][(size_t)t]) {
                            continue;
                        }
                        const double delta = interval_delta(p_refine, t_refine, p, t);

                        // Refined intervals are immutable and owned by exact child results.
                        // Closure propagation must never modify them.
                        if (status[(size_t)p][(size_t)t] != IntervalStatus::kUnrefined) {
                            continue;
                        }

                        const double cand_ssb_mis = lb_mis_anchor - delta;
                        if (cand_ssb_mis > ssb_mis_cur[(size_t)p][(size_t)t] + kEpsUpdate) {
                            ssb_mis_cur[(size_t)p][(size_t)t] = cand_ssb_mis;
                        }

                        const double ssb_mis = std::max(0.0, ssb_mis_cur[(size_t)p][(size_t)t]);
                        const double m_base = std::max(static_mis_lb[(size_t)p][(size_t)t], ssb_mis);
                        const double lb_spb = std::min(
                            ub_leaf_obj[(size_t)p][(size_t)t],
                            m_base + 2.0 * regularization_);
                        const double tightened = std::max(regularization_ + m_base, lb_spb);
                        if (tightened > lb_obj[(size_t)p][(size_t)t] + kEpsUpdate) {
                            lb_obj[(size_t)p][(size_t)t] = std::min(ub_obj[(size_t)p][(size_t)t], tightened);
                        }

                        // Reusing an anchor subtree as a UB witness on another interval is only
                        // guaranteed feasible when the per-leaf min_child_size constraint is vacuous.
                        if (allow_ub_transfer) {
                            const double ub_candidate = child.ub + delta;
                            const double tightened_ub = std::max(lb_obj[(size_t)p][(size_t)t], ub_candidate);
                            if (tightened_ub + kEpsUpdate < ub_obj[(size_t)p][(size_t)t]) {
                                ub_obj[(size_t)p][(size_t)t] = tightened_ub;
                                projected_trees[(size_t)p][(size_t)t] = child.tree;
                            }
                        }
                    }
                }
                if (rush_profile_enabled_) {
                    rush_profile_exact_lazy_closure_sec_ +=
                        std::chrono::duration<double>(Clock::now() - closure_start).count();
                    rush_profile_exact_lazy_closure_passes_ += 1;
                }
            }

            if (!recompute_partitions(lb_partition, ub_partition, feature_lb, feature_ub)) {
                throw std::runtime_error(
                    "RUSH exact-lazy invariant violated: lost feasible UB partition during refinement.");
            }
            assert_interval_bound_sanity(valid, status, lb_obj, ub_obj, ub_leaf_obj, t_max);
            assert_feature_bound_sanity(feature_lb, feature_ub);
            if (feature_lb >= incumbent_ub - kEpsCert) {
                aborted_by_incumbent = true;
                add_exact_lazy_eval_time();
                return false;
            }
        }

#ifdef NDEBUG
        (void)stopped_certified;
        (void)stopped_fully_refined;
#endif
#ifndef NDEBUG
        if (!stopped_certified && !stopped_fully_refined) {
            throw std::runtime_error(
                "RUSH exact-lazy invariant violated: exited feature refinement loop without certified/full-refined stop.");
        }
        if (feature_ub - feature_lb > kEpsCert && num_unrefined_global() > 0) {
            throw std::runtime_error(
                "RUSH exact-lazy invariant violated: exited feature refinement loop with unresolved intervals and non-certified feature.");
        }
#endif

        split_lb = feature_lb;
        split_ub = feature_ub;
        for (const auto &interval : ub_partition.intervals) {
            const int p = interval.first;
            const int t = interval.second;
            if (!projected_trees[(size_t)p][(size_t)t]) {
                projected_trees[(size_t)p][(size_t)t] = make_leaf_node(
                    leaf_prediction[(size_t)p][(size_t)t],
                    leaf_support_count[(size_t)p][(size_t)t],
                    leaf_pos_count[(size_t)p][(size_t)t],
                    leaf_neg_count[(size_t)p][(size_t)t],
                    ub_leaf_obj[(size_t)p][(size_t)t]);
            }
        }
        split_tree = build_internal_node_projected(
            feature,
            bins,
            endpoints,
            ub_partition.intervals,
            projected_trees,
            fallback_prediction,
            (int)indices.size());
        if (!split_tree) {
            add_exact_lazy_eval_time();
            return false;
        }

        assert_interval_bound_sanity(valid, status, lb_obj, ub_obj, ub_leaf_obj, t_max);
        assert_feature_bound_sanity(split_lb, split_ub);
        add_exact_lazy_eval_time();
        return true;
    }

    bool evaluate_feature_dp(
        const std::vector<int> &indices,
        const SubproblemStats &node_stats,
        int feature,
        int depth_remaining,
        int current_depth,
        int fallback_prediction,
        double incumbent_ub,
        double &split_lb,
        double &split_ub,
        std::shared_ptr<Node> &split_tree,
        bool &aborted_by_incumbent,
        const RushFeatureRootCache *root_cache
    ) {
        aborted_by_incumbent = false;
        bool feasible = false;

        // Keep the exact legacy depth-2 shortcut active even in rush/exact-lazy mode.
        // Without this gate, depth-2 subproblems may trigger massive interval refinement
        // overhead despite having a closed-form exact evaluator.
        if (depth_remaining == 2 && !non_uniform_weights_) {
            feasible = evaluate_feature_dp_legacy(
                indices,
                feature,
                depth_remaining,
                fallback_prediction,
                split_lb,
                split_ub,
                split_tree);
        } else if (partition_strategy_ == kPartitionRushDp &&
            std::fabs(branch_penalty_) <= kEpsUpdate &&
            !force_rush_legacy_) {
            feasible = evaluate_feature_dp_rush_exact_lazy(
                indices,
                node_stats,
                feature,
                depth_remaining,
                current_depth,
                fallback_prediction,
                incumbent_ub,
                split_lb,
                split_ub,
                split_tree,
                aborted_by_incumbent,
                root_cache);
        } else {
            feasible = evaluate_feature_dp_legacy(
                indices,
                feature,
                depth_remaining,
                fallback_prediction,
                split_lb,
                split_ub,
                split_tree);
        }
        if (feasible) {
            assert_feature_bound_sanity(split_lb, split_ub);
        }
        return feasible;
    }

    double tau_for_depth(int depth_remaining) const {
        if (tau_mode_ == kTauLambdaSqrtR) {
            return regularization_ * std::sqrt((double)std::max(1, depth_remaining));
        }
        return regularization_;
    }

    bool approx_gain_mass_auto_mode_enabled() const {
        return approx_mode_ && patch_budget_per_feature_ == 0 && exactify_top_m_ == 0;
    }

    int auto_patch_budget_for_feature(int /*scanned_features*/, int projected_endpoint_count) const {
        const int t = std::max(2, projected_endpoint_count);
        const long long intervals = std::max(1, t - 1);
        const long long projected_partitions =
            intervals + (intervals * std::max(0LL, intervals - 1LL)) / 2LL;
        const int rounds_cap = std::max(1, (int)std::ceil(std::log2((double)projected_partitions + 1.0)));
        const int branching = std::max(1, (max_branching_ > 0) ? max_branching_ : 2);
        const long long budget = static_cast<long long>(branching) * static_cast<long long>(rounds_cap);
        return (int)std::max(1LL, budget);
    }

    int effective_patch_budget_for_feature(int scanned_features, int projected_endpoint_count) const {
        if (approx_gain_mass_auto_mode_enabled()) {
            return 0;
        }
        if (patch_budget_per_feature_ > 0) {
            return patch_budget_per_feature_;
        }
        return auto_patch_budget_for_feature(scanned_features, projected_endpoint_count);
    }

    int effective_exactify_cap(int scanned_features, int current_depth) const {
        scanned_features = std::max(1, scanned_features);
        if (exactify_top_m_ > 0) {
            return std::min(scanned_features, exactify_top_m_);
        }
        const int d = std::max(2, scanned_features);
        int auto_cap = std::max(2, (int)std::ceil(std::log2((double)d + 1.0)));
        if (current_depth <= 0) {
            auto_cap += 1;
        }
        return std::max(1, std::min(scanned_features, auto_cap));
    }

    bool in_cheap_oracle_context() const {
        return cheap_oracle_context_depth_ > 0;
    }

    void maybe_clear_greedy_cache_for_approx_insert() {
        if (!in_cheap_oracle_context()) {
            return;
        }
        const long long cap = std::max(1LL, greedy_cache_max_entries_for_approx_);
        if (greedy_cache_entries_live_ < cap) {
            return;
        }
        greedy_cache_.clear();
        greedy_cache_entries_live_ = 0;
        ++greedy_cache_clears_;
    }

    void register_greedy_cache_insert() {
        ++greedy_cache_states_;
        ++greedy_cache_entries_live_;
        greedy_cache_entries_peak_ = std::max(greedy_cache_entries_peak_, greedy_cache_entries_live_);
    }

    CheapUbResult cheap_complete_ub(const std::vector<int> &indices, int depth_remaining) {
        struct CheapOracleScope {
            explicit CheapOracleScope(int &depth) : depth_ref(depth) { ++depth_ref; }
            ~CheapOracleScope() { --depth_ref; }
            int &depth_ref;
        } scope(cheap_oracle_context_depth_);
        GreedyResult greedy = greedy_complete(indices, depth_remaining);
        return CheapUbResult{greedy.objective, greedy.tree};
    }

    static bool bin_matches_spans(int bin_value, const std::vector<std::pair<int, int>> &spans) {
        for (const auto &span : spans) {
            int lo = span.first;
            int hi = span.second;
            if (hi < lo) {
                std::swap(lo, hi);
            }
            if (bin_value >= lo && bin_value <= hi) {
                return true;
            }
        }
        return false;
    }

    static std::vector<std::pair<int, int>> compress_bin_value_spans(const OrderedBins &bins, int left, int right) {
        std::vector<std::pair<int, int>> spans;
        if (left < 0 || right < left || right >= (int)bins.values.size()) {
            return spans;
        }
        spans.reserve((size_t)(right - left + 1));
        int span_lo = bins.values[(size_t)left];
        int span_hi = span_lo;
        for (int pos = left + 1; pos <= right; ++pos) {
            const int bin_value = bins.values[(size_t)pos];
            if (bin_value == span_hi + 1) {
                span_hi = bin_value;
            } else {
                spans.push_back({span_lo, span_hi});
                span_lo = bin_value;
                span_hi = bin_value;
            }
        }
        spans.push_back({span_lo, span_hi});
        return spans;
    }

    bool build_split_from_group_spans_exact_children(
        const std::vector<int> &indices,
        int feature,
        int depth_remaining,
        int fallback_prediction,
        const std::vector<std::vector<std::pair<int, int>>> &group_spans,
        double &split_lb,
        double &split_ub,
        std::shared_ptr<Node> &split_tree
    ) {
        split_lb = 0.0;
        split_ub = 0.0;
        split_tree.reset();
        if (group_spans.size() < 2) {
            return false;
        }

        std::vector<std::vector<int>> grouped_indices(group_spans.size());
        for (int idx : indices) {
            const int bin_value = x(idx, feature);
            int matched_group = -1;
            for (size_t g = 0; g < group_spans.size(); ++g) {
                if (bin_matches_spans(bin_value, group_spans[g])) {
                    matched_group = (int)g;
                    break;
                }
            }
            if (matched_group < 0) {
                return false;
            }
            grouped_indices[(size_t)matched_group].push_back(idx);
        }

        auto internal = std::make_shared<Node>();
        internal->is_leaf = false;
        internal->feature = feature;
        internal->fallback_prediction = fallback_prediction;
        internal->n_samples = (int)indices.size();
        internal->group_count = (int)group_spans.size();
        internal->group_bin_spans = group_spans;
        internal->group_nodes.assign(group_spans.size(), nullptr);

        int largest_child_size = -1;
        int fallback_bin = -1;
        for (size_t g = 0; g < group_spans.size(); ++g) {
            auto &child_indices = grouped_indices[g];
            if ((int)child_indices.size() < min_child_size_) {
                return false;
            }
            BoundResult child = solve_subproblem(child_indices, depth_remaining - 1);
            split_lb += child.lb;
            split_ub += child.ub;
            internal->group_nodes[g] = child.tree;

            const int span_lo = group_spans[g].empty() ? -1 : group_spans[g].front().first;
            if ((int)child_indices.size() > largest_child_size ||
                ((int)child_indices.size() == largest_child_size && span_lo >= 0 &&
                 (fallback_bin < 0 || span_lo < fallback_bin))) {
                largest_child_size = (int)child_indices.size();
                fallback_bin = span_lo;
            }
        }
        if (fallback_bin < 0 && !group_spans.empty() && !group_spans.front().empty()) {
            fallback_bin = group_spans.front().front().first;
        }
        internal->fallback_bin = fallback_bin;
        split_lb += split_penalty_for_groups((int)group_spans.size());
        split_ub += split_penalty_for_groups((int)group_spans.size());
        split_tree = internal;
        return true;
    }

    bool evaluate_feature_approx_ub_only(
        const std::vector<int> &indices,
        const SubproblemStats &node_stats,
        const ApproxFeaturePrep &prep,
        int depth_remaining,
        int patch_budget,
        ApproxPatchCellCache *patch_cell_cache,
        ApproxFeatureResult &out
    ) {
        out = ApproxFeatureResult{};
        out.feature = prep.feature;
        if (!prep.feasible) {
            return false;
        }
        const int feature = prep.feature;
        const int n_bins = prep.n_bins_dense;
        const int q_projected = prep.q_projected;
        const std::vector<int> &endpoints = prep.endpoints;
        const std::vector<int> &row_cnt_prefix = prep.row_cnt_prefix;
        const std::vector<int> &pos_cnt_prefix = prep.pos_cnt_prefix;
        const std::vector<int> &neg_cnt_prefix = prep.neg_cnt_prefix;
        const std::vector<double> &pos_w_prefix = prep.pos_w_prefix;
        const std::vector<double> &neg_w_prefix = prep.neg_w_prefix;
        const std::vector<int> &row_cnt_bin = prep.row_cnt_bin;
        if (q_projected < 2 || n_bins <= 1) {
            return false;
        }
        const int t_max = (int)endpoints.size() - 1;
        out.projected_endpoint_count = t_max;
        std::vector<double> epb_bin((size_t)n_bins, 0.0);
        for (size_t k = 0; k < node_stats.touched_signature_ids.size(); ++k) {
            const int s = node_stats.touched_signature_ids[k];
            const double e = node_stats.touched_signature_epb[k];
            const int bin_val = sig_bin_[(size_t)s][(size_t)feature];
            if (bin_val >= 0 && bin_val < n_bins) {
                epb_bin[(size_t)bin_val] += e;
            }
        }
        std::vector<double> epb_prefix((size_t)n_bins + 1, 0.0);
        for (int b = 0; b < n_bins; ++b) {
            epb_prefix[(size_t)(b + 1)] = epb_prefix[(size_t)b] + epb_bin[(size_t)b];
        }

        const double uniform_row_weight = (!non_uniform_weights_ && !sample_weight_.empty())
            ? sample_weight_.front()
            : 0.0;
        std::vector<std::vector<IntervalStatus>> status(
            (size_t)t_max + 1,
            std::vector<IntervalStatus>((size_t)t_max + 1, IntervalStatus::kInfeasible));
        std::vector<std::vector<bool>> valid(
            (size_t)t_max + 1,
            std::vector<bool>((size_t)t_max + 1, false));
        std::vector<std::vector<double>> lb_obj(
            (size_t)t_max + 1,
            std::vector<double>((size_t)t_max + 1, kInfinity));
        std::vector<std::vector<double>> ub_obj(
            (size_t)t_max + 1,
            std::vector<double>((size_t)t_max + 1, kInfinity));

        for (int p = 0; p < t_max; ++p) {
            const int left = endpoints[(size_t)p];
            for (int t = p + 1; t <= t_max; ++t) {
                const int right = endpoints[(size_t)t] - 1;
                const int support = row_cnt_prefix[(size_t)(right + 1)] - row_cnt_prefix[(size_t)left];
                if (support < min_child_size_) {
                    continue;
                }
                valid[(size_t)p][(size_t)t] = true;

                const int p_cnt = pos_cnt_prefix[(size_t)(right + 1)] - pos_cnt_prefix[(size_t)left];
                const int n_cnt = neg_cnt_prefix[(size_t)(right + 1)] - neg_cnt_prefix[(size_t)left];
                const double p_w = pos_w_prefix[(size_t)(right + 1)] - pos_w_prefix[(size_t)left];
                const double n_w = neg_w_prefix[(size_t)(right + 1)] - neg_w_prefix[(size_t)left];
                const double leaf_obj = regularization_ + std::min(p_w, n_w);
                ub_obj[(size_t)p][(size_t)t] = leaf_obj;

                const double epb_mis = epb_prefix[(size_t)(right + 1)] - epb_prefix[(size_t)left];
                double msb_mis = 0.0;
                if (!non_uniform_weights_) {
                    const int minor_cnt = std::min(p_cnt, n_cnt);
                    const int msb_cnt = std::min(minor_cnt, std::max(0, min_child_size_ - minor_cnt));
                    msb_mis = static_cast<double>(msb_cnt) * uniform_row_weight;
                }
                const double static_mis_lb = std::max(epb_mis, msb_mis);
                if (support < 2 * min_child_size_ || leaf_obj <= 2.0 * regularization_ + kEpsCert) {
                    status[(size_t)p][(size_t)t] = IntervalStatus::kCertifiedLeaf;
                    lb_obj[(size_t)p][(size_t)t] = leaf_obj;
                    ub_obj[(size_t)p][(size_t)t] = leaf_obj;
                } else {
                    status[(size_t)p][(size_t)t] = IntervalStatus::kUnrefined;
                    const double lb_spb = std::min(leaf_obj, static_mis_lb + 2.0 * regularization_);
                    lb_obj[(size_t)p][(size_t)t] = std::max(regularization_, lb_spb);
                }
            }
        }

        ProjectedDpWorkspace &projected_ws = projected_dp_workspace_for_depth(
            full_depth_budget_ - depth_remaining);
        const ProjectedPartitionResult lb_partition =
            optimize_partition_projected(lb_obj, valid, q_projected, true, &projected_ws);
        if (!lb_partition.feasible) {
            return false;
        }
        out.lhat = lb_partition.cost + split_penalty_for_groups(lb_partition.groups);
        approx_lhat_computed_ = true;

        int budget_used = 0;
        ProjectedPartitionResult ub_partition;
        while (true) {
            ub_partition = optimize_partition_projected(ub_obj, valid, q_projected, true, &projected_ws);
            if (!ub_partition.feasible) {
                return false;
            }

            bool saw_unrefined = false;
            bool saw_patchable = false;
            bool tightened = false;
            for (const auto &interval : ub_partition.intervals) {
                const int p = interval.first;
                const int t = interval.second;
                if (status[(size_t)p][(size_t)t] != IntervalStatus::kUnrefined) {
                    continue;
                }
                saw_unrefined = true;
                ++approx_pub_unrefined_cells_on_pub_total_;
                const int left = endpoints[(size_t)p];
                const int right = endpoints[(size_t)t] - 1;
                const int child_rows =
                    row_cnt_prefix[(size_t)(right + 1)] - row_cnt_prefix[(size_t)left];
                if (child_rows < 2 * min_child_size_) {
                    ++approx_pub_cells_skipped_by_childrows_;
                    continue;
                }
                saw_patchable = true;
                ++approx_pub_patchable_cells_total_;
                const double lb_cell = lb_obj[(size_t)p][(size_t)t];
                const double ub_cell = ub_obj[(size_t)p][(size_t)t];
                if (ub_cell <= lb_cell + kEpsCert) {
                    ++approx_patch_skipped_already_tight_;
                    continue;
                }

                bool has_cached = false;
                double cached_ub_child = kInfinity;
                bool cached_no_further_improve = false;
                ApproxPatchCellKey cache_key;
                ApproxPatchCellCache::iterator cache_it;
                if (patch_cell_cache != nullptr && patch_budget > 0) {
                    cache_key.feature = feature;
                    cache_key.p = p;
                    cache_key.t = t;
                    cache_key.child_depth_remaining = depth_remaining - 1;
                    cache_it = patch_cell_cache->find(cache_key);
                    if (cache_it != patch_cell_cache->end()) {
                        ++approx_patch_cell_cache_hits_;
                        has_cached = true;
                        cached_ub_child = cache_it->second.ub_child;
                        cached_no_further_improve = cache_it->second.no_further_improve;
                    }
                }

                if (has_cached) {
                    if (cached_no_further_improve) {
                        ++approx_patch_skipped_cached_;
                        ++approx_patch_skipped_no_possible_improve_;
                        continue;
                    }
                    const double candidate_ub = std::max(lb_cell, cached_ub_child);
                    if (candidate_ub >= ub_obj[(size_t)p][(size_t)t] - kEpsUpdate) {
                        if (patch_cell_cache != nullptr && patch_budget > 0) {
                            cache_it->second.no_further_improve = true;
                        }
                        ++approx_patch_skipped_cached_;
                        ++approx_patch_skipped_no_possible_improve_;
                        continue;
                    }
                    ub_obj[(size_t)p][(size_t)t] = candidate_ub;
                    if (patch_cell_cache != nullptr && patch_budget > 0) {
                        cache_it->second.no_further_improve = true;
                    }
                    ++approx_patch_cache_hit_updates_;
                    ++approx_greedy_ub_updates_total_;
                    tightened = true;
                    continue;
                }

                if (budget_used >= patch_budget) {
                    continue;
                }

                std::vector<int> subset_sorted;
                ++approx_patch_subset_materializations_;
                subset_sorted.reserve((size_t)child_rows);
                for (int idx : indices) {
                    const int bin_value = x(idx, feature);
                    if (bin_value >= left && bin_value <= right) {
                        subset_sorted.push_back(idx);
                    }
                }
                const Clock::time_point patch_start = Clock::now();
                if (patch_cell_cache != nullptr && patch_budget > 0) {
                    ++approx_patch_cell_cache_misses_;
                }
                ++approx_greedy_patch_calls_;
                ++approx_patch_cache_miss_oracle_calls_;
                CheapUbResult cheap = cheap_complete_ub(subset_sorted, depth_remaining - 1);
                approx_greedy_patch_sec_ += std::chrono::duration<double>(Clock::now() - patch_start).count();
                ++budget_used;

                const double candidate_ub = std::max(lb_cell, cheap.objective);
                const bool can_improve = (candidate_ub < ub_obj[(size_t)p][(size_t)t] - kEpsUpdate);
                if (patch_cell_cache != nullptr && patch_budget > 0) {
                    (*patch_cell_cache)[cache_key] = ApproxPatchCellCacheEntry{
                        cheap.objective,
                        true};
                }
                if (!can_improve) {
                    ++approx_patch_skipped_no_possible_improve_;
                    continue;
                }
                ub_obj[(size_t)p][(size_t)t] = candidate_ub;
                ++approx_greedy_patches_applied_;
                ++approx_greedy_ub_updates_total_;
                tightened = true;
            }
            if (!saw_unrefined || !saw_patchable || !tightened) {
                break;
            }
        }
        ub_partition = optimize_partition_projected(ub_obj, valid, q_projected, true, &projected_ws);
        if (!ub_partition.feasible) {
            return false;
        }
        out.feasible = true;
        out.uhat = ub_partition.cost + split_penalty_for_groups(ub_partition.groups);
        out.group_spans.clear();
        out.group_spans.reserve(ub_partition.intervals.size());
        for (const auto &interval : ub_partition.intervals) {
            const int left = endpoints[(size_t)interval.first];
            const int right = endpoints[(size_t)interval.second] - 1;
            out.group_spans.push_back(compress_observed_bin_spans_from_hist(row_cnt_bin, left, right));
        }
        return out.feasible;
    }

    bool evaluate_feature_greedy(
        const std::vector<int> &indices,
        int feature,
        int depth_remaining,
        int fallback_prediction,
        double &split_objective,
        std::shared_ptr<Node> &split_tree
    ) {
        OrderedBins bins;
        if (!build_ordered_bins(indices, feature, bins)) {
            return false;
        }

        const int n_bins = (int)bins.values.size();
        const int q_base = max_groups_for_bins(n_bins);
        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        const int q_effective = std::min(q_base, q_support);
        if (q_effective < 2) {
            return false;
        }

        std::vector<int> rush_endpoints;
        const std::vector<int> *rush_endpoints_ptr = nullptr;
        if (partition_strategy_ == kPartitionRushDp) {
            std::vector<int> pos_bins((size_t)n_bins, 0);
            std::vector<int> neg_bins((size_t)n_bins, 0);
            std::vector<int> support_cnt((size_t)n_bins, 0);
            std::vector<double> pos_mass((size_t)n_bins, 0.0);
            std::vector<double> neg_mass((size_t)n_bins, 0.0);
            for (int bin_pos = 0; bin_pos < n_bins; ++bin_pos) {
                int positives = 0;
                double pos_weight = 0.0;
                double neg_weight = 0.0;
                for (int idx : bins.members[(size_t)bin_pos]) {
                    const int y = y_[(size_t)idx];
                    positives += y;
                    if (y == 1) {
                        pos_weight += sample_weight_[(size_t)idx];
                    } else {
                        neg_weight += sample_weight_[(size_t)idx];
                    }
                }
                const int total = (int)bins.members[(size_t)bin_pos].size();
                pos_bins[(size_t)bin_pos] = positives;
                neg_bins[(size_t)bin_pos] = total - positives;
                support_cnt[(size_t)bin_pos] = total;
                pos_mass[(size_t)bin_pos] = pos_weight;
                neg_mass[(size_t)bin_pos] = neg_weight;
            }
            rush_endpoints = build_majority_shift_endpoints(pos_mass, neg_mass, support_cnt);
            rush_endpoints_ptr = &rush_endpoints;
        }

        if (rush_endpoints_ptr != nullptr) {
            const std::vector<int> &endpoints = *rush_endpoints_ptr;
            const int t_max = (int)endpoints.size() - 1;
            const int q_projected = std::min(q_effective, t_max);
            if (q_projected < 2) {
                return false;
            }
            std::vector<std::vector<bool>> projected_valid(
                (size_t)t_max + 1,
                std::vector<bool>((size_t)t_max + 1, false));
            std::vector<std::vector<double>> projected_obj(
                (size_t)t_max + 1,
                std::vector<double>((size_t)t_max + 1, kInfinity));
            std::vector<std::vector<std::shared_ptr<Node>>> projected_trees(
                (size_t)t_max + 1,
                std::vector<std::shared_ptr<Node>>((size_t)t_max + 1, nullptr));
            for (int p = 0; p < t_max; ++p) {
                check_timeout();
                const int left = endpoints[(size_t)p];
                std::vector<int> subset_sorted;
                subset_sorted.reserve((size_t)(bins.prefix_counts[n_bins] - bins.prefix_counts[left]));
                int prev_right = left - 1;
                for (int t = p + 1; t <= t_max; ++t) {
                    const int right = endpoints[(size_t)t] - 1;
                    for (int b = prev_right + 1; b <= right; ++b) {
                        append_sorted_members(subset_sorted, bins.members[(size_t)b]);
                    }
                    prev_right = right;

                    const int child_count = bins.prefix_counts[right + 1] - bins.prefix_counts[left];
                    if (child_count < min_child_size_) {
                        continue;
                    }

                    ++greedy_interval_evals_;
                    GreedyResult child = greedy_complete(subset_sorted, depth_remaining - 1);
                    projected_valid[(size_t)p][(size_t)t] = true;
                    projected_obj[(size_t)p][(size_t)t] = child.objective;
                    projected_trees[(size_t)p][(size_t)t] = child.tree;
                }
            }

            ProjectedPartitionResult partition =
                optimize_partition_projected(projected_obj, projected_valid, q_projected, true);
            if (!partition.feasible) {
                return false;
            }

            split_objective = partition.cost + split_penalty_for_groups(partition.groups);
            split_tree = build_internal_node_projected(
                feature,
                bins,
                endpoints,
                partition.intervals,
                projected_trees,
                fallback_prediction,
                (int)indices.size());
            return split_tree != nullptr;
        } else {
            std::vector<std::vector<bool>> valid(n_bins, std::vector<bool>(n_bins, false));
            std::vector<std::vector<double>> interval_obj(n_bins, std::vector<double>(n_bins, kInfinity));
            std::vector<std::vector<std::shared_ptr<Node>>> interval_trees(
                n_bins,
                std::vector<std::shared_ptr<Node>>(n_bins, nullptr));
            for (int left = 0; left < n_bins; ++left) {
                std::vector<int> subset_sorted;
                subset_sorted.reserve((size_t)(bins.prefix_counts[n_bins] - bins.prefix_counts[left]));
                for (int right = left; right < n_bins; ++right) {
                    append_sorted_members(subset_sorted, bins.members[(size_t)right]);
                    int child_count = bins.prefix_counts[right + 1] - bins.prefix_counts[left];
                    if (child_count < min_child_size_) {
                        continue;
                    }

                    ++greedy_interval_evals_;
                    GreedyResult child = greedy_complete(subset_sorted, depth_remaining - 1);
                    valid[left][right] = true;
                    interval_obj[left][right] = child.objective;
                    interval_trees[left][right] = child.tree;
                }
            }
            PartitionResult partition = optimize_partition_full(interval_obj, valid, q_effective, true);
            if (!partition.feasible) {
                return false;
            }

            split_objective = partition.cost + split_penalty_for_groups(partition.groups);
            split_tree = build_internal_node(
                feature,
                bins,
                partition.intervals,
                interval_trees,
                fallback_prediction,
                (int)indices.size());
            return split_tree != nullptr;
        }
    }

    BoundResult solve_subproblem(const std::vector<int> &indices, int depth_remaining) {
        check_timeout();
        ++dp_subproblem_calls_;
        const int current_depth = full_depth_budget_ - depth_remaining;
        const uint64_t key_hash = state_hash(indices, depth_remaining);
        std::vector<DpCacheEntry> *dp_bucket_ptr = nullptr;
        const int dp_slot = find_dp_cache_slot(key_hash, depth_remaining, indices, &dp_bucket_ptr);
        if (dp_slot >= 0 && dp_bucket_ptr != nullptr) {
            const DpCacheEntry &entry = (*dp_bucket_ptr)[(size_t)dp_slot];
            ++dp_cache_hits_;
#ifndef NDEBUG
            const SubproblemStats cache_stats = compute_subproblem_stats(indices);
            if (std::fabs(entry.result.lb_mis - cache_stats.epb_mis) > kEpsCert) {
                throw std::runtime_error("RUSH exact-lazy invariant violated: cached lb_mis mismatch.");
            }
#endif
            return entry.result;
        }

        const SubproblemStats stats = compute_subproblem_stats(indices);
        auto [leaf_objective, leaf_tree] = leaf_solution(stats);
        if (depth_remaining <= 1 || stats.pure) {
            BoundResult solved{leaf_objective, stats.epb_mis, leaf_objective, leaf_tree};
            dp_cache_[key_hash].push_back(DpCacheEntry{depth_remaining, indices, solved});
            ++dp_cache_states_;
            return solved;
        }

        if (current_depth == effective_lookahead_) {
            GreedyResult greedy = greedy_complete(indices, depth_remaining);
            BoundResult solved{greedy.objective, stats.epb_mis, greedy.objective, greedy.tree};
            dp_cache_[key_hash].push_back(DpCacheEntry{depth_remaining, indices, solved});
            ++dp_cache_states_;
            return solved;
        }

        double best_lb = leaf_objective;
        double best_ub = leaf_objective;
        std::shared_ptr<Node> best_tree = leaf_tree;
        bool node_resolved = false;
        const bool approx_eligible =
            approx_mode_ &&
            (partition_strategy_ == kPartitionRushDp) &&
            (std::fabs(branch_penalty_) <= kEpsUpdate) &&
            !force_rush_legacy_ &&
            (depth_remaining >= 3);
        if (approx_eligible) {
            const long long node_patch_calls_before = approx_greedy_patch_calls_;
            const long long node_patchable_before = approx_pub_patchable_cells_total_;
            if (current_depth == 0) {
                ++approx_eligible_nodes_depth0_;
            } else if (current_depth == 1) {
                ++approx_eligible_nodes_depth1_;
            }
            bool fast100_debug_this_depth1_node = false;
            int fast100_debug_depth1_node_id = -1;
            if (current_depth == 1 &&
                fast100_debug_depth1_limit_ > 0 &&
                fast100_debug_depth1_nodes_logged_ < fast100_debug_depth1_limit_) {
                fast100_debug_this_depth1_node = true;
                fast100_debug_depth1_node_id = ++fast100_debug_depth1_nodes_logged_;
            }

            struct ApproxEntry {
                int feature = -1;
                int prep_index = -1;
                int budget_used = 0;
                int budget_cap = 0;
                int approx_budget_evaluated = -1;
                bool feasible = false;
                ApproxFeatureResult approx;
                bool patch_budget_recorded = false;
                bool exact_cached = false;
                bool exact_feasible = false;
                bool exact_aborted_by_incumbent = false;
                double exact_lb = 0.0;
                double exact_ub = kInfinity;
                std::shared_ptr<Node> exact_tree;
                bool gini_ready = false;
                double gini_best_cost = kInfinity;
                double gini_gain = -kInfinity;
                int gini_best_k = 0;
                int gini_endpoints_added = 0;
            };
            const int feature_scan_limit = (approx_feature_scan_limit_ > 0)
                ? std::min(n_features_, approx_feature_scan_limit_)
                : n_features_;
            if (feature_scan_limit > 0) {
                std::vector<ApproxFeaturePrep> approx_preps;
                std::vector<ApproxEntry> approx_entries;
                std::vector<int> feature_to_entry((size_t)feature_scan_limit, -1);
                ApproxPatchCellCache patch_cell_cache;

                auto record_patch_budget_effective = [&](int cap) {
                    approx_patch_budget_effective_seen_ += 1;
                    approx_patch_budget_effective_sum_ += cap;
                    approx_patch_budget_effective_min_ = std::min(approx_patch_budget_effective_min_, cap);
                    approx_patch_budget_effective_max_ = std::max(approx_patch_budget_effective_max_, cap);
                };
                auto ensure_entry = [&](int feature) -> ApproxEntry & {
                    int idx = feature_to_entry[(size_t)feature];
                    if (idx >= 0) {
                        return approx_entries[(size_t)idx];
                    }
                    ApproxEntry entry;
                    entry.feature = feature;
                    entry.approx.feature = feature;
                    ApproxFeaturePrep prep;
                    if (build_approx_feature_prep_from_signatures(stats, feature, prep)) {
                        entry.prep_index = (int)approx_preps.size();
                        entry.budget_cap = effective_patch_budget_for_feature(
                            feature_scan_limit,
                            std::max(0, (int)prep.endpoints.size() - 1));
                        approx_preps.push_back(std::move(prep));
                    }
                    idx = (int)approx_entries.size();
                    approx_entries.push_back(std::move(entry));
                    feature_to_entry[(size_t)feature] = idx;
                    return approx_entries[(size_t)idx];
                };
                auto refresh_feature_with_budget = [&](ApproxEntry &entry) {
                    if (entry.prep_index < 0) {
                        entry.feasible = false;
                        entry.approx_budget_evaluated = entry.budget_used;
                        return;
                    }
                    const ApproxFeaturePrep &prep = approx_preps[(size_t)entry.prep_index];
                    if (!entry.patch_budget_recorded) {
                        record_patch_budget_effective(entry.budget_cap);
                        entry.patch_budget_recorded = true;
                    }
                    if (entry.approx_budget_evaluated == entry.budget_used) {
                        return;
                    }
                    ApproxFeatureResult refreshed;
                    refreshed.feature = entry.feature;
                    entry.feasible = evaluate_feature_approx_ub_only(
                        indices,
                        stats,
                        prep,
                        depth_remaining,
                        entry.budget_used,
                        &patch_cell_cache,
                        refreshed);
                    if (entry.feasible) {
                        entry.approx = std::move(refreshed);
                    }
                    entry.approx_budget_evaluated = entry.budget_used;
                    if (entry.budget_used > entry.budget_cap) {
                        entry.budget_used = entry.budget_cap;
                    }
                };
                const double node_weight_sum = stats.pos_weight + stats.neg_weight;
                const double node_parent_gini =
                    gini_seg_cost_norm(stats.pos_weight, stats.neg_weight, node_weight_sum);
                auto ensure_gini_for_entry = [&](ApproxEntry &entry,
                                                 bool is_in_root_k0_block,
                                                 bool is_in_active_features,
                                                 bool node_is_depth1_ambiguous) -> bool {
                    if (!gini_scout_enabled_) {
                        return false;
                    }
                    bool endpoints_augmented = false;
                    const bool allow_root =
                        (current_depth == 0) && is_in_root_k0_block;
                    const bool allow_depth1 =
                        (current_depth == 1) &&
                        node_is_depth1_ambiguous &&
                        is_in_active_features;
                    if (!(allow_root || allow_depth1)) {
                        return false;
                    }
                    if (entry.gini_ready) {
                        return false;
                    }
                    if (entry.prep_index < 0) {
                        return false;
                    }

                    ApproxFeaturePrep &prep = approx_preps[(size_t)entry.prep_index];
                    const int gini_k_max = std::max(0, max_groups_for_bins(prep.n_bins_dense));
                    const Clock::time_point gini_start = Clock::now();
                    GiniDPResult gini_result = gini_dp_best_partition(
                        prep,
                        gini_k_max,
                        min_child_size_,
                        node_weight_sum,
                        gini_dp_workspace_);
                    gini_dp_sec_ +=
                        std::chrono::duration<double>(Clock::now() - gini_start).count();
                    if (current_depth == 0) {
                        ++gini_dp_calls_root_;
                    } else if (current_depth == 1) {
                        ++gini_dp_calls_depth1_;
                    }

                    entry.gini_ready = true;
                    if (!gini_result.feasible) {
                        entry.gini_best_cost = kInfinity;
                        entry.gini_gain = -kInfinity;
                        entry.gini_best_k = 0;
                        return false;
                    }

                    entry.gini_best_cost = gini_result.best_cost;
                    entry.gini_gain = node_parent_gini - gini_result.best_cost;
                    entry.gini_best_k = gini_result.best_k;
                    ++gini_dp_scored_count_;
                    gini_dp_k_sum_ += gini_result.best_k;
                    gini_dp_k_max_ = std::max(gini_dp_k_max_, gini_result.best_k);
                    gini_dp_b_sum_ += prep.n_bins_dense;
                    gini_dp_b_max_ = std::max(gini_dp_b_max_, prep.n_bins_dense);

                    // Endpoint augmentation is root-only to avoid depth1 per-call blowups.
                    if (allow_root) {
                        int branching_cap = max_groups_for_bins(prep.n_bins_dense);
                        if (max_branching_ > 0) {
                            branching_cap = std::min(branching_cap, max_branching_);
                        }
                        const int endpoint_budget = std::max(0, branching_cap - 1);
                        if (endpoint_budget > 0 && !gini_result.cuts.empty()) {
                            const Clock::time_point ep_start = Clock::now();
                            const int added = add_gini_cut_endpoints(
                                prep.endpoints,
                                gini_result.cuts,
                                prep.n_bins_dense,
                                endpoint_budget);
                            gini_endpoint_sec_ +=
                                std::chrono::duration<double>(Clock::now() - ep_start).count();
                            if (added > 0) {
                                prep.gini_endpoints_augmented = true;
                                prep.gini_endpoints_added += added;
                                prep.q_projected = std::min(
                                    prep.q_effective,
                                    std::max(0, (int)prep.endpoints.size() - 1));
                                entry.gini_endpoints_added = added;
                                entry.approx_budget_evaluated = -1;
                                gini_endpoints_added_per_feature_max_ =
                                    std::max(gini_endpoints_added_per_feature_max_, added);
                                gini_endpoints_added_root_ += added;
                                ++gini_endpoints_features_touched_root_;
                                endpoints_augmented = true;
                            }
                        }
                    }

                    return endpoints_augmented;
                };
                const bool gain_mass_auto_mode = approx_gain_mass_auto_mode_enabled();

                std::vector<int> root_order;
                root_order.reserve((size_t)feature_scan_limit);
                for (int feature = 0; feature < feature_scan_limit; ++feature) {
                    root_order.push_back(feature);
                }
                if (approx_ref_shortlist_enabled_) {
                    const bool root_order_compatible =
                        approx_ref_root_ready_ &&
                        ((int)approx_ref_root_order_.size() == feature_scan_limit);
                    if (!root_order_compatible && current_depth == 0) {
                        std::vector<std::pair<int, double>> root_scores;
                        root_scores.reserve((size_t)feature_scan_limit);
                        double sum_g = 0.0;
                        for (int feature = 0; feature < feature_scan_limit; ++feature) {
                            ApproxEntry &entry = ensure_entry(feature);
                            double g = 0.0;
                            if (entry.prep_index >= 0) {
                                g = std::max(0.0, approx_preps[(size_t)entry.prep_index].g);
                            }
                            sum_g += g;
                            root_scores.push_back({feature, g});
                        }
                        std::sort(
                            root_scores.begin(),
                            root_scores.end(),
                            [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) {
                                if (lhs.second > rhs.second + kEpsUpdate) {
                                    return true;
                                }
                                if (rhs.second > lhs.second + kEpsUpdate) {
                                    return false;
                                }
                                return lhs.first < rhs.first;
                            });
                        approx_ref_root_order_.clear();
                        approx_ref_root_order_.reserve(root_scores.size());
                        for (const auto &pair : root_scores) {
                            approx_ref_root_order_.push_back(pair.first);
                        }
                        if (sum_g <= kEpsUpdate) {
                            approx_ref_neff_root_ = (double)feature_scan_limit;
                        } else {
                            double entropy = 0.0;
                            for (const auto &pair : root_scores) {
                                if (pair.second <= 0.0) {
                                    continue;
                                }
                                const double p = pair.second / sum_g;
                                entropy -= p * std::log(p);
                            }
                            approx_ref_neff_root_ = std::exp(entropy);
                        }
                        approx_ref_root_ready_ = true;
                    }
                    if (approx_ref_root_ready_ &&
                        (int)approx_ref_root_order_.size() == feature_scan_limit) {
                        root_order = approx_ref_root_order_;
                    }
                }
                std::vector<int> prior_rank_by_feature((size_t)feature_scan_limit, 0);
                for (int pos = 0; pos < (int)root_order.size(); ++pos) {
                    const int feature = root_order[(size_t)pos];
                    if (feature >= 0 && feature < feature_scan_limit) {
                        prior_rank_by_feature[(size_t)feature] = pos + 1;
                    }
                }
                const bool prior_rank_available =
                    approx_ref_shortlist_enabled_ &&
                    approx_ref_root_ready_ &&
                    ((int)approx_ref_root_order_.size() == feature_scan_limit);
                if (prior_rank_available) {
                    fast100_used_lgb_prior_tiebreak_ = true;
                }

                const int r_split = std::max(1, depth_remaining - 1);
                auto clamp_prefix = [&](double neff_like) {
                    if (feature_scan_limit <= 0) {
                        return 0;
                    }
                    const double root_scale = std::max(1.0, neff_like);
                    int p = (int)std::ceil(root_scale * std::sqrt((double)r_split));
                    p = std::max(2, p);
                    p = std::min(feature_scan_limit, p);
                    if (feature_scan_limit < 2) {
                        p = feature_scan_limit;
                    }
                    return p;
                };

                int p_current = feature_scan_limit;
                if (approx_ref_shortlist_enabled_) {
                    const double neff_root = (approx_ref_root_ready_) ? approx_ref_neff_root_ : (double)feature_scan_limit;
                    p_current = clamp_prefix(neff_root);
                }
                auto prefix_features = [&](int prefix_len) {
                    prefix_len = std::max(0, std::min(prefix_len, (int)root_order.size()));
                    return std::vector<int>(root_order.begin(), root_order.begin() + prefix_len);
                };

                struct RefShortlistInfo {
                    std::vector<int> ref_rank;
                    double neff = 0.0;
                    int k0 = 0;
                    int k_target = 0;
                };
                auto build_ref_shortlist_info = [&](const std::vector<int> &universe) {
                    RefShortlistInfo info;
                    info.ref_rank = universe;
                    std::vector<std::pair<int, double>> scores;
                    scores.reserve(universe.size());
                    double sum_g = 0.0;
                    for (int feature : universe) {
                        ApproxEntry &entry = ensure_entry(feature);
                        double g = 0.0;
                        if (entry.prep_index >= 0) {
                            g = std::max(0.0, approx_preps[(size_t)entry.prep_index].g);
                        }
                        scores.push_back({feature, g});
                        sum_g += g;
                    }
                    std::sort(
                        scores.begin(),
                        scores.end(),
                        [&](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) {
                            if (lhs.second > rhs.second + kEpsUpdate) {
                                return true;
                            }
                            if (rhs.second > lhs.second + kEpsUpdate) {
                                return false;
                            }
                            const int lhs_rank =
                                (lhs.first >= 0 && lhs.first < feature_scan_limit)
                                    ? prior_rank_by_feature[(size_t)lhs.first]
                                    : std::numeric_limits<int>::max();
                            const int rhs_rank =
                                (rhs.first >= 0 && rhs.first < feature_scan_limit)
                                    ? prior_rank_by_feature[(size_t)rhs.first]
                                    : std::numeric_limits<int>::max();
                            if (lhs_rank != rhs_rank) {
                                return lhs_rank < rhs_rank;
                            }
                            return lhs.first < rhs.first;
                        });
                    info.ref_rank.clear();
                    info.ref_rank.reserve(scores.size());
                    for (const auto &pair : scores) {
                        info.ref_rank.push_back(pair.first);
                    }
                    const int u_size = (int)universe.size();
                    if (u_size == 0) {
                        info.neff = 0.0;
                        info.k0 = 0;
                        info.k_target = 0;
                        return info;
                    }
                    if (sum_g <= kEpsUpdate) {
                        info.neff = (double)u_size;
                        info.k0 = u_size;
                        info.k_target = u_size;
                    } else {
                        if (current_depth == 0) {
                            const double top_g = std::max(0.0, scores.front().second);
                            const double alpha = top_g / (sum_g + kEpsUpdate);
                            int neff_conc = u_size;
                            if (alpha > kEpsUpdate) {
                                neff_conc = (int)std::ceil(1.0 / alpha);
                            }
                            neff_conc = std::max(1, std::min(neff_conc, u_size));
                            info.neff = (double)neff_conc;
                            int k0 = std::max(std::min(2, u_size), std::min(neff_conc, u_size));
                            int k_target = std::max(k0, std::min(2 * neff_conc, u_size));
                            info.k0 = k0;
                            info.k_target = k_target;
                        } else {
                            double entropy = 0.0;
                            for (const auto &pair : scores) {
                                if (pair.second <= 0.0) {
                                    continue;
                                }
                                const double p = pair.second / sum_g;
                                entropy -= p * std::log(p);
                            }
                            info.neff = std::exp(entropy);
                            int k0 = (int)std::ceil(info.neff);
                            k0 = std::max(std::min(2, u_size), std::min(k0, u_size));
                            int k_target = (int)std::ceil(info.neff * std::sqrt((double)r_split));
                            k_target = std::max(k0, std::min(k_target, u_size));
                            info.k0 = k0;
                            info.k_target = k_target;
                        }
                    }
                    return info;
                };

                std::vector<int> universe = prefix_features(p_current);
                RefShortlistInfo shortlist_info = build_ref_shortlist_info(universe);
                const int initial_k0 = shortlist_info.k0;
                if (gini_scout_enabled_ && gain_mass_auto_mode && current_depth == 0) {
                    gini_root_k0_ = std::max(0, initial_k0);
                }
                if (gini_scout_enabled_ &&
                    gain_mass_auto_mode &&
                    current_depth == 0 && initial_k0 > 1 &&
                    initial_k0 <= (int)shortlist_info.ref_rank.size()) {
                    std::vector<int> root_k0_block(
                        shortlist_info.ref_rank.begin(),
                        shortlist_info.ref_rank.begin() + initial_k0);
                    for (int feature : root_k0_block) {
                        ApproxEntry &entry = ensure_entry(feature);
                        ensure_gini_for_entry(entry, true, false, false);
                    }
                    bool used_gini_tiebreak = false;
                    std::sort(
                        root_k0_block.begin(),
                        root_k0_block.end(),
                        [&](int lhs_feature, int rhs_feature) {
                            const ApproxEntry &lhs_entry =
                                approx_entries[(size_t)feature_to_entry[(size_t)lhs_feature]];
                            const ApproxEntry &rhs_entry =
                                approx_entries[(size_t)feature_to_entry[(size_t)rhs_feature]];
                            const double lhs_g =
                                (lhs_entry.prep_index >= 0)
                                    ? std::max(0.0, approx_preps[(size_t)lhs_entry.prep_index].g)
                                    : 0.0;
                            const double rhs_g =
                                (rhs_entry.prep_index >= 0)
                                    ? std::max(0.0, approx_preps[(size_t)rhs_entry.prep_index].g)
                                    : 0.0;
                            if (lhs_g > rhs_g + kEpsUpdate) {
                                return true;
                            }
                            if (rhs_g > lhs_g + kEpsUpdate) {
                                return false;
                            }
                            const bool lhs_gini_ok =
                                lhs_entry.gini_ready && std::isfinite(lhs_entry.gini_gain);
                            const bool rhs_gini_ok =
                                rhs_entry.gini_ready && std::isfinite(rhs_entry.gini_gain);
                            if (lhs_gini_ok && rhs_gini_ok) {
                                if (lhs_entry.gini_gain > rhs_entry.gini_gain + kEpsUpdate) {
                                    used_gini_tiebreak = true;
                                    return true;
                                }
                                if (rhs_entry.gini_gain > lhs_entry.gini_gain + kEpsUpdate) {
                                    used_gini_tiebreak = true;
                                    return false;
                                }
                            }
                            const int lhs_rank =
                                (lhs_feature >= 0 && lhs_feature < feature_scan_limit)
                                    ? prior_rank_by_feature[(size_t)lhs_feature]
                                    : std::numeric_limits<int>::max();
                            const int rhs_rank =
                                (rhs_feature >= 0 && rhs_feature < feature_scan_limit)
                                    ? prior_rank_by_feature[(size_t)rhs_feature]
                                    : std::numeric_limits<int>::max();
                            if (lhs_rank != rhs_rank) {
                                return lhs_rank < rhs_rank;
                            }
                            return lhs_feature < rhs_feature;
                        });
                    if (used_gini_tiebreak) {
                        ++gini_tiebreak_used_in_shortlist_;
                    }
                    std::copy(
                        root_k0_block.begin(),
                        root_k0_block.end(),
                        shortlist_info.ref_rank.begin());
                }
                std::vector<unsigned char> initial_active_mask((size_t)feature_scan_limit, 0);
                for (int i = 0; i < std::min(initial_k0, (int)shortlist_info.ref_rank.size()); ++i) {
                    const int feature = shortlist_info.ref_rank[(size_t)i];
                    if (feature >= 0 && feature < feature_scan_limit) {
                        initial_active_mask[(size_t)feature] = 1;
                    }
                }
                std::vector<int> root_rank_by_feature((size_t)feature_scan_limit, 0);
                for (size_t pos = 0; pos < root_order.size(); ++pos) {
                    const int feature = root_order[pos];
                    if (feature >= 0 && feature < feature_scan_limit) {
                        root_rank_by_feature[(size_t)feature] = (int)pos + 1;
                    }
                }
                approx_ref_neff_count_ += 1;
                approx_ref_neff_sum_ += shortlist_info.neff;
                approx_ref_neff_max_ = std::max(approx_ref_neff_max_, shortlist_info.neff);
                approx_ref_k0_count_ += 1;
                approx_ref_k0_sum_ += shortlist_info.k0;
                approx_ref_k0_min_ = std::min(approx_ref_k0_min_, shortlist_info.k0);
                approx_ref_k0_max_ = std::max(approx_ref_k0_max_, shortlist_info.k0);

                int k_current = shortlist_info.k0;
                int k_target = shortlist_info.k_target;
                int final_k_used = k_current;
                int widen_used = 0;
                int chosen_feature = -1;
                bool node_uncertainty_counted = false;
                bool node_exactify_counted = false;
                bool node_cap_stop_counted = false;
                std::vector<unsigned char> exact_order_mask((size_t)n_features_, 0);
                std::vector<unsigned char> exactified_feature_mask((size_t)n_features_, 0);

                struct ExactifyContextScope {
                    explicit ExactifyContextScope(int &depth_ref) : depth(depth_ref) {
                        ++depth;
                    }
                    ~ExactifyContextScope() { --depth; }
                    int &depth;
                };

                auto ensure_exact_feature_evaluated = [&](ApproxEntry &entry, double incumbent) {
                    if (entry.exact_cached) {
                        return;
                    }
                    double split_lb = 0.0;
                    double split_ub = 0.0;
                    std::shared_ptr<Node> split_tree;
                    bool aborted_by_incumbent = false;
                    ExactifyContextScope exactify_scope(exactify_eval_context_depth_);
                    entry.exact_feasible = evaluate_feature_dp(
                        indices,
                        stats,
                        entry.feature,
                        depth_remaining,
                        current_depth,
                        leaf_tree->prediction,
                        incumbent,
                        split_lb,
                        split_ub,
                        split_tree,
                        aborted_by_incumbent,
                        nullptr);
                    entry.exact_cached = true;
                    entry.exact_aborted_by_incumbent = aborted_by_incumbent;
                    entry.exact_lb = split_lb;
                    entry.exact_ub = split_ub;
                    entry.exact_tree = split_tree;
                    if (!entry.exact_feasible && aborted_by_incumbent) {
                        ++rush_incumbent_feature_aborts_;
                    }
                };

                auto rank_approx = [&](const std::vector<int> &active_features) {
                    std::vector<int> ranked;  // entry indices
                    ranked.reserve(active_features.size());
                    for (int feature : active_features) {
                        const int eidx = feature_to_entry[(size_t)feature];
                        if (eidx < 0) {
                            continue;
                        }
                        const ApproxEntry &entry = approx_entries[(size_t)eidx];
                        if (entry.feasible) {
                            ranked.push_back(eidx);
                        }
                    }
                    std::sort(
                        ranked.begin(),
                        ranked.end(),
                        [&](int lhs, int rhs) {
                            const ApproxEntry &l = approx_entries[(size_t)lhs];
                            const ApproxEntry &r = approx_entries[(size_t)rhs];
                            if (l.approx.uhat < r.approx.uhat - kEpsUpdate) {
                                return true;
                            }
                            if (r.approx.uhat < l.approx.uhat - kEpsUpdate) {
                                return false;
                            }
                            return l.feature < r.feature;
                        });
                    return ranked;
                };
                auto is_safe_accept = [&](const std::vector<int> &ranked) {
                    if (ranked.empty()) {
                        return false;
                    }
                    if (ranked.size() < 2U) {
                        return true;
                    }
                    const double best_uhat = approx_entries[(size_t)ranked[0]].approx.uhat;
                    double min_other_lhat = kInfinity;
                    for (size_t i = 1; i < ranked.size(); ++i) {
                        min_other_lhat = std::min(
                            min_other_lhat,
                            approx_entries[(size_t)ranked[i]].approx.lhat);
                    }
                    return best_uhat <= min_other_lhat + kEpsUpdate;
                };
                auto is_ambiguous = [&](const std::vector<int> &ranked) {
                    if (ranked.size() < 2U) {
                        return false;
                    }
                    const ApproxFeatureResult &best = approx_entries[(size_t)ranked[0]].approx;
                    const ApproxFeatureResult &second = approx_entries[(size_t)ranked[1]].approx;
                    const double delta = second.uhat - best.uhat;
                    const double uncertainty =
                        std::max(0.0, best.uhat - best.lhat) +
                        std::max(0.0, second.uhat - second.lhat);
                    return delta < uncertainty - kEpsUpdate;
                };
                bool done = false;
                while (!done) {
                    const int k_active = std::min(k_current, (int)shortlist_info.ref_rank.size());
                    std::vector<int> active_features;
                    active_features.reserve((size_t)k_active);
                    for (int i = 0; i < k_active; ++i) {
                        active_features.push_back(shortlist_info.ref_rank[(size_t)i]);
                    }
                    final_k_used = (int)active_features.size();
                    for (int feature : active_features) {
                        ApproxEntry &entry = ensure_entry(feature);
                        refresh_feature_with_budget(entry);
                    }

                    std::vector<int> ranked = rank_approx(active_features);
                    bool safe_accept = is_safe_accept(ranked);
                    bool ambiguous = is_ambiguous(ranked);
                    if (!gain_mass_auto_mode) {
                        while (!ranked.empty() && !safe_accept && ambiguous) {
                            bool progressed = false;
                            const int topk = std::min<int>(2, (int)ranked.size());
                            for (int pos = 0; pos < topk; ++pos) {
                                ApproxEntry &entry = approx_entries[(size_t)ranked[(size_t)pos]];
                                if (entry.budget_used >= entry.budget_cap) {
                                    continue;
                                }
                                ++entry.budget_used;
                                refresh_feature_with_budget(entry);
                                progressed = true;
                            }
                            if (!progressed) {
                                break;
                            }
                            ranked = rank_approx(active_features);
                            safe_accept = is_safe_accept(ranked);
                            ambiguous = is_ambiguous(ranked);
                        }
                    }

                    bool stage_exactify_entered = false;
                    bool stage_stopped_by_cap = false;
                    bool stage_resolved = false;
                    double stage_lb = kInfinity;
                    double stage_ub = kInfinity;
                    std::shared_ptr<Node> stage_tree;
                    int stage_feature = -1;
                    int stage_pre_exactify_best_feature = -1;
                    bool stage_root_gap_within_uncertainty = false;
                    bool stage_root_winner_changed_after_exactify = false;
                    int stage_winner_feature = -1;
                    double stage_winner_ub0 = kInfinity;
                    double stage_lb_best_challenger = kInfinity;
                    bool stage_skip_by_ub_lb_separation = false;
                    int stage_frontier_size = 0;
                    int stage_exactified_count = 0;
                    int stage_exactify_cap = 0;
                    const char *stage_stop_reason = "none";

                    bool gain_mass_handled = false;
                    bool stage_fast100_importance_passed = true;
                    if (!ranked.empty() && gain_mass_auto_mode) {
                        gain_mass_handled = true;
                        const int winner_u_eidx = ranked[0];
                        const int winner_u_feature = approx_entries[(size_t)winner_u_eidx].feature;
                        const double winner_u_ub0 = approx_entries[(size_t)winner_u_eidx].approx.uhat;
                        const double separation_margin = std::max(one_mistake_unit_, 2.0 * regularization_);
                        stage_winner_feature = winner_u_feature;
                        stage_winner_ub0 = winner_u_ub0;

                        auto accept_approx_winner = [&]() {
                            const ApproxFeatureResult &best_approx = approx_entries[(size_t)winner_u_eidx].approx;
                            double split_lb = 0.0;
                            double split_ub = 0.0;
                            std::shared_ptr<Node> split_tree;
                            if (build_split_from_group_spans_exact_children(
                                    indices,
                                    best_approx.feature,
                                    depth_remaining,
                                    leaf_tree->prediction,
                                    best_approx.group_spans,
                                    split_lb,
                                    split_ub,
                                    split_tree)) {
                                stage_lb = split_lb;
                                stage_ub = split_ub;
                                stage_tree = split_tree;
                                stage_feature = best_approx.feature;
                                stage_resolved = true;
                            }
                        };

                        double lb_challenger = kInfinity;
                        for (size_t ri = 1; ri < ranked.size(); ++ri) {
                            lb_challenger = std::min(lb_challenger, approx_entries[(size_t)ranked[ri]].approx.lhat);
                        }
                        const bool has_challenger = std::isfinite(lb_challenger);
                        const bool skip_by_ub_lb_separation =
                            has_challenger &&
                            lb_challenger >= winner_u_ub0 - separation_margin - kEpsUpdate;
                        stage_lb_best_challenger = lb_challenger;
                        stage_skip_by_ub_lb_separation = skip_by_ub_lb_separation;

                        if (!has_challenger || skip_by_ub_lb_separation) {
                            if (skip_by_ub_lb_separation) {
                                ++fast100_skipped_by_ub_lb_separation_;
                                stage_stop_reason = "skip_ub_lb_separation";
                            } else {
                                stage_stop_reason = "skip_no_challenger";
                            }
                            accept_approx_winner();
                        } else {
                            int exactify_cap =
                                std::max(1, effective_exactify_cap((int)ranked.size(), current_depth));
                            if (current_depth == 1) {
                                // FAST100 depth1 should stay tiny to prevent exactify blowups.
                                exactify_cap = std::min(exactify_cap, 2);
                            }
                            stage_exactify_cap = exactify_cap;
                            if (current_depth == 0) {
                                approx_exactify_cap_effective_sum_depth0_ += exactify_cap;
                            } else if (current_depth == 1) {
                                approx_exactify_cap_effective_sum_depth1_ += exactify_cap;
                            }

                            std::vector<int> frontier;
                            frontier.reserve(ranked.size());
                            bool run_exactify = true;
                            if (current_depth == 0) {
                                const int take = std::min<int>(exactify_cap, (int)ranked.size());
                                frontier.insert(frontier.end(), ranked.begin(), ranked.begin() + take);
                            } else if (current_depth == 1) {
                                if (ranked.size() < 2U) {
                                    ++depth1_skipped_by_low_global_ambiguity_;
                                    stage_stop_reason = "depth1_skip_no_runner_up";
                                    run_exactify = false;
                                    accept_approx_winner();
                                } else {
                                    const double ub1 = approx_entries[(size_t)ranked[0]].approx.uhat;
                                    const double ub2 = approx_entries[(size_t)ranked[1]].approx.uhat;
                                    const double gap = std::max(0.0, ub2 - ub1);
                                    double neff = shortlist_info.neff;
                                    if (!std::isfinite(neff) || neff < 1.0) {
                                        neff = (double)std::max<size_t>(1U, ranked.size());
                                    }
                                    const int mult_raw =
                                        1 + (int)std::ceil(std::log2(neff + 1.0));
                                    int mult = std::max(1, mult_raw);
                                    mult = std::min(mult, exactify_cap);
                                    const double amb = separation_margin * (double)mult;
                                    if (gap > amb + kEpsUpdate) {
                                        ++depth1_skipped_by_low_global_ambiguity_;
                                        ++depth1_skipped_by_large_gap_;
                                        stage_stop_reason = "depth1_skip_large_gap";
                                        run_exactify = false;
                                        accept_approx_winner();
                                    } else {
                                        auto push_frontier_eidx = [&](int eidx) {
                                            if (eidx < 0 || eidx >= (int)approx_entries.size()) {
                                                return;
                                            }
                                            const int feature = approx_entries[(size_t)eidx].feature;
                                            if (feature < 0 || feature >= n_features_) {
                                                return;
                                            }
                                            for (int seen_eidx : frontier) {
                                                if (approx_entries[(size_t)seen_eidx].feature == feature) {
                                                    return;
                                                }
                                            }
                                            frontier.push_back(eidx);
                                        };
                                        if (!gini_scout_enabled_) {
                                            push_frontier_eidx(ranked[0]);  // winner
                                            push_frontier_eidx(ranked[1]);  // runner-up
                                            if ((int)frontier.size() > exactify_cap) {
                                                frontier.resize((size_t)exactify_cap);
                                            }
                                        } else if (ranked.empty()) {
                                            stage_stop_reason = "depth1_skip_no_ranked_after_gini";
                                            run_exactify = false;
                                            accept_approx_winner();
                                        } else {
                                            for (int feature : active_features) {
                                                if (feature < 0 || feature >= feature_scan_limit) {
                                                    continue;
                                                }
                                                const int eidx = feature_to_entry[(size_t)feature];
                                                if (eidx < 0) {
                                                    continue;
                                                }
                                                (void)ensure_gini_for_entry(
                                                    approx_entries[(size_t)eidx],
                                                    false,
                                                    true,
                                                    true);
                                            }
                                            const int winner_eidx = ranked[0];
                                            const int runner_up_eidx =
                                                (ranked.size() >= 2U) ? ranked[1] : -1;

                                            push_frontier_eidx(winner_eidx);
                                            int challenger_eidx = runner_up_eidx;
                                            int teacher_eidx = -1;
                                            double best_teacher_gain = -kInfinity;
                                            if (runner_up_eidx >= 0) {
                                                for (int feature : active_features) {
                                                    if (feature < 0 || feature >= feature_scan_limit) {
                                                        continue;
                                                    }
                                                    if (feature == approx_entries[(size_t)winner_eidx].feature ||
                                                        feature == approx_entries[(size_t)runner_up_eidx].feature) {
                                                        continue;
                                                    }
                                                    const int eidx = feature_to_entry[(size_t)feature];
                                                    if (eidx < 0) {
                                                        continue;
                                                    }
                                                    const ApproxEntry &cand = approx_entries[(size_t)eidx];
                                                    if (!cand.feasible || !cand.gini_ready ||
                                                        !std::isfinite(cand.gini_gain)) {
                                                        continue;
                                                    }
                                                    if (cand.gini_gain > best_teacher_gain + kEpsUpdate ||
                                                        (std::fabs(cand.gini_gain - best_teacher_gain) <=
                                                             kEpsUpdate &&
                                                         (teacher_eidx < 0 ||
                                                          cand.feature <
                                                              approx_entries[(size_t)teacher_eidx].feature))) {
                                                        best_teacher_gain = cand.gini_gain;
                                                        teacher_eidx = eidx;
                                                    }
                                                }
                                            }

                                            if (runner_up_eidx >= 0 && teacher_eidx >= 0) {
                                                const ApproxEntry &runner = approx_entries[(size_t)runner_up_eidx];
                                                const ApproxEntry &teacher = approx_entries[(size_t)teacher_eidx];
                                                const double runner_gain =
                                                    (runner.gini_ready && std::isfinite(runner.gini_gain))
                                                        ? runner.gini_gain
                                                        : -kInfinity;
                                                const double teacher_gain =
                                                    (teacher.gini_ready && std::isfinite(teacher.gini_gain))
                                                        ? teacher.gini_gain
                                                        : -kInfinity;
                                                if (teacher_gain > runner_gain + kEpsUpdate) {
                                                    if (teacher.approx.uhat <=
                                                        runner.approx.uhat + separation_margin + kEpsUpdate) {
                                                        challenger_eidx = teacher_eidx;
                                                        ++depth1_teacher_replaced_runnerup_;
                                                        ++gini_teacher_chosen_depth1_;
                                                        if (teacher_eidx != runner_up_eidx) {
                                                            ++gini_depth1_teacher_changed_count_;
                                                        }
                                                    } else {
                                                        ++depth1_teacher_rejected_by_uhat_gate_;
                                                    }
                                                }
                                            }

                                            push_frontier_eidx(challenger_eidx);
                                            if ((int)frontier.size() > exactify_cap) {
                                                frontier.resize((size_t)exactify_cap);
                                            }
                                        }
                                    }
                                }
                            } else {
                                const double node_weight_sum = stats.pos_weight + stats.neg_weight;
                                const double ub_margin_node =
                                    one_mistake_unit_ / std::max(node_weight_sum, 1e-12);
                                const double ub_frontier_margin =
                                    std::max(2.0 * regularization_, ub_margin_node);
                                const double ub_frontier_threshold =
                                    winner_u_ub0 + ub_frontier_margin + kEpsUpdate;
                                for (int eidx : ranked) {
                                    const ApproxEntry &entry = approx_entries[(size_t)eidx];
                                    if (entry.approx.uhat <= ub_frontier_threshold) {
                                        frontier.push_back(eidx);
                                    }
                                }
                            }

                            if (run_exactify) {
                                ++fast100_exactify_nodes_allowed_;
                                stage_exactify_entered = true;
                                stage_pre_exactify_best_feature = winner_u_feature;
                                if (current_depth == 0) {
                                    ++fast100_cf_exactify_nodes_depth0_;
                                } else if (current_depth == 1) {
                                    ++fast100_cf_exactify_nodes_depth1_;
                                }
                                if (!node_exactify_counted) {
                                    ++approx_exactify_triggered_nodes_;
                                    if (current_depth == 0) {
                                        ++approx_exactify_triggered_nodes_depth0_;
                                    } else if (current_depth == 1) {
                                        ++approx_exactify_triggered_nodes_depth1_;
                                    }
                                    node_exactify_counted = true;
                                }
                                if (current_depth == 1) {
                                    ++depth1_exactify_challenger_nodes_;
                                    ++depth1_exactified_nodes_;
                                }

                                if (current_depth != 1) {
                                    std::sort(
                                        frontier.begin(),
                                        frontier.end(),
                                        [&](int lhs, int rhs) {
                                            const ApproxEntry &l = approx_entries[(size_t)lhs];
                                            const ApproxEntry &r = approx_entries[(size_t)rhs];
                                            if (l.approx.uhat < r.approx.uhat - kEpsUpdate) {
                                                return true;
                                            }
                                            if (r.approx.uhat < l.approx.uhat - kEpsUpdate) {
                                                return false;
                                            }
                                            return l.feature < r.feature;
                                        });
                                }
                                if (current_depth == 0 && !frontier.empty()) {
                                    int gini_best_eidx = -1;
                                    double gini_best_gain = -kInfinity;
                                    for (int feature = 0; feature < feature_scan_limit; ++feature) {
                                        if (!initial_active_mask[(size_t)feature]) {
                                            continue;
                                        }
                                        const int eidx = feature_to_entry[(size_t)feature];
                                        if (eidx < 0) {
                                            continue;
                                        }
                                        const ApproxEntry &cand = approx_entries[(size_t)eidx];
                                        if (!cand.feasible || !cand.gini_ready ||
                                            !std::isfinite(cand.gini_gain)) {
                                            continue;
                                        }
                                        if (cand.gini_gain > gini_best_gain + kEpsUpdate ||
                                            (std::fabs(cand.gini_gain - gini_best_gain) <= kEpsUpdate &&
                                             (gini_best_eidx < 0 ||
                                              cand.feature <
                                                  approx_entries[(size_t)gini_best_eidx].feature))) {
                                            gini_best_gain = cand.gini_gain;
                                            gini_best_eidx = eidx;
                                        }
                                    }
                                    if (gini_best_eidx >= 0) {
                                        bool already_present = false;
                                        for (int eidx : frontier) {
                                            if (approx_entries[(size_t)eidx].feature ==
                                                approx_entries[(size_t)gini_best_eidx].feature) {
                                                already_present = true;
                                                break;
                                            }
                                        }
                                        const int tail_eidx = frontier.back();
                                        const int tail_feature = approx_entries[(size_t)tail_eidx].feature;
                                        const bool tail_in_gini_root =
                                            (tail_feature >= 0 &&
                                             tail_feature < feature_scan_limit &&
                                             initial_active_mask[(size_t)tail_feature] != 0);
                                        const bool tail_has_gini =
                                            tail_in_gini_root &&
                                            approx_entries[(size_t)tail_eidx].gini_ready &&
                                            std::isfinite(approx_entries[(size_t)tail_eidx].gini_gain);
                                        const bool strict_gini_improve =
                                            tail_has_gini &&
                                            (approx_entries[(size_t)gini_best_eidx].gini_gain >
                                             approx_entries[(size_t)tail_eidx].gini_gain + kEpsUpdate);
                                        const bool uhat_competitive =
                                            (approx_entries[(size_t)gini_best_eidx].approx.uhat <=
                                             approx_entries[(size_t)tail_eidx].approx.uhat +
                                                 separation_margin + kEpsUpdate);
                                        if (!already_present &&
                                            strict_gini_improve &&
                                            uhat_competitive) {
                                            frontier.back() = gini_best_eidx;
                                            std::sort(
                                                frontier.begin(),
                                                frontier.end(),
                                                [&](int lhs, int rhs) {
                                                    const ApproxEntry &l = approx_entries[(size_t)lhs];
                                                    const ApproxEntry &r = approx_entries[(size_t)rhs];
                                                    if (l.approx.uhat < r.approx.uhat - kEpsUpdate) {
                                                        return true;
                                                    }
                                                    if (r.approx.uhat < l.approx.uhat - kEpsUpdate) {
                                                        return false;
                                                    }
                                                    return l.feature < r.feature;
                                                });
                                        }
                                    }
                                }
                                stage_frontier_size = (int)frontier.size();

                                ++fast100_frontier_size_count_;
                                fast100_frontier_size_sum_ += (long long)frontier.size();
                                fast100_frontier_size_max_ =
                                    std::max(fast100_frontier_size_max_, (int)frontier.size());
                                ++fast100_cf_frontier_size_count_;
                                fast100_cf_frontier_size_sum_ += (long long)frontier.size();
                                fast100_cf_frontier_size_max_ =
                                    std::max(fast100_cf_frontier_size_max_, (int)frontier.size());

                                std::fill(exactified_feature_mask.begin(), exactified_feature_mask.end(), 0);
                                int exactified_count = 0;
                                bool stopped_by_midloop_separation = false;
                                bool stopped_by_ambiguous_empty = false;
                                double best_certified_ub = kInfinity;
                                double best_certified_lb = kInfinity;
                                std::shared_ptr<Node> best_certified_tree;
                                int best_certified_feature = -1;

                                size_t frontier_cursor = 0;
                                while (frontier_cursor < frontier.size() && exactified_count < exactify_cap) {
                                    const int eidx = frontier[frontier_cursor++];
                                    ApproxEntry &entry = approx_entries[(size_t)eidx];
                                    const int feature = entry.feature;
                                    if (feature < 0 || feature >= n_features_) {
                                        continue;
                                    }
                                    if (exactified_feature_mask[(size_t)feature]) {
                                        continue;
                                    }
                                    exactified_feature_mask[(size_t)feature] = 1;
                                    ++exactified_count;
                                    ++approx_exactify_features_exact_solved_;
                                    if (current_depth == 0) {
                                        ++approx_exactify_features_exact_solved_depth0_;
                                    } else if (current_depth == 1) {
                                        ++approx_exactify_features_exact_solved_depth1_;
                                    }

                                    const double incumbent = std::min(best_ub, best_certified_ub);
                                    ensure_exact_feature_evaluated(entry, incumbent);
                                    if (entry.exact_feasible) {
                                        best_lb = std::min(best_lb, entry.exact_lb);
                                        if (entry.exact_ub < best_certified_ub - kEpsUpdate ||
                                            (std::fabs(entry.exact_ub - best_certified_ub) <= kEpsUpdate &&
                                             (best_certified_feature < 0 || feature < best_certified_feature))) {
                                            best_certified_ub = entry.exact_ub;
                                            best_certified_lb = entry.exact_lb;
                                            best_certified_tree = entry.exact_tree;
                                            best_certified_feature = feature;
                                        }
                                    }

                                    if (std::isfinite(best_certified_ub)) {
                                        double lb_remaining_min = kInfinity;
                                        for (size_t read = frontier_cursor; read < frontier.size(); ++read) {
                                            const int cand_eidx = frontier[read];
                                            const ApproxEntry &cand = approx_entries[(size_t)cand_eidx];
                                            const int cand_feature = cand.feature;
                                            if (cand_feature < 0 || cand_feature >= n_features_) {
                                                continue;
                                            }
                                            if (exactified_feature_mask[(size_t)cand_feature]) {
                                                continue;
                                            }
                                            lb_remaining_min = std::min(lb_remaining_min, cand.approx.lhat);
                                        }
                                        if (!std::isfinite(lb_remaining_min)) {
                                            stopped_by_ambiguous_empty = true;
                                            stage_stop_reason = "ambiguous_empty";
                                            break;
                                        }
                                        if (lb_remaining_min >= best_certified_ub - separation_margin - kEpsUpdate) {
                                            stopped_by_midloop_separation = true;
                                            ++fast100_stopped_midloop_separation_;
                                            stage_stop_reason = "midloop_separation";
                                            break;
                                        }
                                    }
                                }
                                stage_exactified_count = exactified_count;

                                if (!stopped_by_midloop_separation &&
                                    !stopped_by_ambiguous_empty &&
                                    frontier_cursor >= frontier.size()) {
                                    stopped_by_ambiguous_empty = true;
                                    stage_stop_reason = "ambiguous_empty";
                                }

                                if (!stopped_by_midloop_separation &&
                                    exactified_count >= exactify_cap) {
                                    bool has_unprocessed = false;
                                    for (size_t read = frontier_cursor; read < frontier.size(); ++read) {
                                        const int cand_feature = approx_entries[(size_t)frontier[read]].feature;
                                        if (cand_feature < 0 || cand_feature >= n_features_) {
                                            continue;
                                        }
                                        if (!exactified_feature_mask[(size_t)cand_feature]) {
                                            has_unprocessed = true;
                                            break;
                                        }
                                    }
                                    if (has_unprocessed) {
                                        stage_stopped_by_cap = true;
                                        stage_stop_reason = "cap";
                                    } else {
                                        stopped_by_ambiguous_empty = true;
                                        stage_stop_reason = "ambiguous_empty";
                                    }
                                }

                                if (stopped_by_ambiguous_empty) {
                                    ++approx_exactify_stops_by_ambiguous_empty_;
                                }
                                if (stopped_by_midloop_separation) {
                                    ++approx_exactify_stops_by_separation_;
                                    if (current_depth == 0) {
                                        ++approx_exactify_stops_by_separation_depth0_;
                                    } else if (current_depth == 1) {
                                        ++approx_exactify_stops_by_separation_depth1_;
                                    }
                                }
                                if (stage_stopped_by_cap) {
                                    if (!node_cap_stop_counted) {
                                        ++approx_exactify_stops_by_cap_;
                                        if (current_depth == 0) {
                                            ++approx_exactify_stops_by_cap_depth0_;
                                        } else if (current_depth == 1) {
                                            ++approx_exactify_stops_by_cap_depth1_;
                                        }
                                        node_cap_stop_counted = true;
                                    }
                                }
                                if (!stage_stopped_by_cap &&
                                    !stopped_by_midloop_separation &&
                                    !stopped_by_ambiguous_empty) {
                                    stage_stop_reason = "frontier_exhausted";
                                }

                                ++fast100_cf_exactified_features_count_;
                                fast100_cf_exactified_features_sum_ += exactified_count;
                                fast100_cf_exactified_features_max_ =
                                    std::max(fast100_cf_exactified_features_max_, exactified_count);
                                if (current_depth == 0) {
                                    ++fast100_m_depth0_count_;
                                    fast100_m_depth0_sum_ += exactified_count;
                                    fast100_m_depth0_max_ = std::max(fast100_m_depth0_max_, exactified_count);
                                    ++approx_exactify_set_size_depth0_count_;
                                    approx_exactify_set_size_depth0_sum_ += exactified_count;
                                    approx_exactify_set_size_depth0_min_ =
                                        std::min(approx_exactify_set_size_depth0_min_, exactified_count);
                                    approx_exactify_set_size_depth0_max_ =
                                        std::max(approx_exactify_set_size_depth0_max_, exactified_count);
                                } else if (current_depth == 1) {
                                    ++fast100_m_depth1_count_;
                                    fast100_m_depth1_sum_ += exactified_count;
                                    fast100_m_depth1_max_ = std::max(fast100_m_depth1_max_, exactified_count);
                                    depth1_exactified_features_sum_ += exactified_count;
                                    depth1_exactified_features_max_ =
                                        std::max(depth1_exactified_features_max_, exactified_count);
                                    ++approx_exactify_set_size_depth1_count_;
                                    approx_exactify_set_size_depth1_sum_ += exactified_count;
                                    approx_exactify_set_size_depth1_min_ =
                                        std::min(approx_exactify_set_size_depth1_min_, exactified_count);
                                    approx_exactify_set_size_depth1_max_ =
                                        std::max(approx_exactify_set_size_depth1_max_, exactified_count);
                                }

                                if (std::isfinite(best_certified_ub) && best_certified_tree) {
                                    stage_lb = best_certified_lb;
                                    stage_ub = best_certified_ub;
                                    stage_tree = best_certified_tree;
                                    stage_feature = best_certified_feature;
                                    stage_resolved = true;
                                } else {
                                    accept_approx_winner();
                                }
                            }
                        }
                    }
                    if (!gain_mass_handled && !ranked.empty() && (safe_accept || !ambiguous)) {
                        const ApproxFeatureResult &best_approx = approx_entries[(size_t)ranked[0]].approx;
                        double split_lb = 0.0;
                        double split_ub = 0.0;
                        std::shared_ptr<Node> split_tree;
                        if (build_split_from_group_spans_exact_children(
                                indices,
                                best_approx.feature,
                                depth_remaining,
                                leaf_tree->prediction,
                                best_approx.group_spans,
                                split_lb,
                                split_ub,
                                split_tree)) {
                            stage_lb = split_lb;
                            stage_ub = split_ub;
                            stage_tree = split_tree;
                            stage_feature = best_approx.feature;
                            stage_resolved = true;
                        }
                    } else if (!gain_mass_handled && !ranked.empty() && ambiguous) {
                        stage_exactify_entered = true;
                        stage_pre_exactify_best_feature = approx_entries[(size_t)ranked[0]].feature;
                        if (!node_uncertainty_counted) {
                            ++approx_uncertainty_triggered_nodes_;
                            if (current_depth == 0) {
                                ++approx_uncertainty_triggered_nodes_depth0_;
                            } else if (current_depth == 1) {
                                ++approx_uncertainty_triggered_nodes_depth1_;
                            }
                            node_uncertainty_counted = true;
                        }
                        if (!node_exactify_counted) {
                            ++approx_exactify_triggered_nodes_;
                            if (current_depth == 0) {
                                ++approx_exactify_triggered_nodes_depth0_;
                            } else if (current_depth == 1) {
                                ++approx_exactify_triggered_nodes_depth1_;
                            }
                            node_exactify_counted = true;
                        }

                        const int exactify_cap =
                            std::max(1, effective_exactify_cap((int)ranked.size(), current_depth));
                        if (current_depth == 0) {
                            approx_exactify_cap_effective_sum_depth0_ += exactify_cap;
                        } else if (current_depth == 1) {
                            approx_exactify_cap_effective_sum_depth1_ += exactify_cap;
                        }

                        std::fill(exactified_feature_mask.begin(), exactified_feature_mask.end(), 0);
                        int exactify_calls = 0;
                        bool stopped_by_separation = false;
                        bool stopped_by_ambiguous_empty = false;
                        bool stopped_by_no_improve = false;
                        double best_certified_ub = kInfinity;
                        double best_certified_lb = kInfinity;
                        std::shared_ptr<Node> best_certified_tree;
                        int best_certified_feature = -1;

                        double node_ambiguous_size_sum = 0.0;
                        int node_ambiguous_size_count = 0;
                        int node_ambiguous_size_max = 0;
                        int node_ambiguous_size_min = std::numeric_limits<int>::max();
                        int node_ambiguous_size_prev = -1;
                        int node_ambiguous_shrank_steps = 0;
                        auto record_ambiguous_size = [&](int size_now) {
                            node_ambiguous_size_sum += size_now;
                            node_ambiguous_size_count += 1;
                            node_ambiguous_size_max = std::max(node_ambiguous_size_max, size_now);
                            node_ambiguous_size_min = std::min(node_ambiguous_size_min, size_now);
                            if (node_ambiguous_size_prev >= 0 && size_now < node_ambiguous_size_prev) {
                                ++node_ambiguous_shrank_steps;
                            }
                            node_ambiguous_size_prev = size_now;
                        };

                        auto current_u_best = [&]() {
                            int radius_feature = best_certified_feature;
                            if (radius_feature < 0 && !ranked.empty()) {
                                radius_feature = approx_entries[(size_t)ranked[0]].feature;
                            }
                            if (radius_feature >= 0 && radius_feature < feature_scan_limit) {
                                const int eidx = feature_to_entry[(size_t)radius_feature];
                                if (eidx >= 0) {
                                    const ApproxEntry &entry = approx_entries[(size_t)eidx];
                                    return std::max(0.0, entry.approx.uhat - entry.approx.lhat);
                                }
                            }
                            return 0.0;
                        };
                        auto min_remaining_lhat = [&]() {
                            double min_lhat = kInfinity;
                            for (int eidx : ranked) {
                                const int feature = approx_entries[(size_t)eidx].feature;
                                if (feature < 0 || feature >= n_features_) {
                                    continue;
                                }
                                if (exactified_feature_mask[(size_t)feature]) {
                                    continue;
                                }
                                min_lhat = std::min(min_lhat, approx_entries[(size_t)eidx].approx.lhat);
                            }
                            return min_lhat;
                        };
                        auto ambiguous_candidates = [&]() {
                            std::vector<int> candidates;
                            candidates.reserve(ranked.size());
                            if (!std::isfinite(best_certified_ub)) {
                                for (int eidx : ranked) {
                                    const int feature = approx_entries[(size_t)eidx].feature;
                                    if (feature >= 0 && feature < n_features_ &&
                                        !exactified_feature_mask[(size_t)feature]) {
                                        candidates.push_back(eidx);
                                    }
                                }
                                return candidates;
                            }
                            const double u_best = current_u_best();
                            for (int eidx : ranked) {
                                const int feature = approx_entries[(size_t)eidx].feature;
                                if (feature < 0 || feature >= n_features_) {
                                    continue;
                                }
                                if (exactified_feature_mask[(size_t)feature]) {
                                    continue;
                                }
                                const ApproxFeatureResult &cand = approx_entries[(size_t)eidx].approx;
                                if (!(cand.lhat < best_certified_ub - kEpsUpdate)) {
                                    continue;
                                }
                                if (cand.lhat <= best_certified_ub + u_best + kEpsUpdate) {
                                    candidates.push_back(eidx);
                                }
                            }
                            return candidates;
                        };

                        while (exactify_calls < exactify_cap) {
                            std::vector<int> candidates = ambiguous_candidates();
                            if (std::isfinite(best_certified_ub)) {
                                record_ambiguous_size((int)candidates.size());
                            }
                            if (candidates.empty()) {
                                if (std::isfinite(best_certified_ub)) {
                                    stopped_by_ambiguous_empty = true;
                                } else {
                                    stopped_by_separation = true;
                                }
                                break;
                            }
                            const double cycle_start_best = best_certified_ub;
                            size_t candidate_cursor = 0;
                            while (candidate_cursor < candidates.size()) {
                                if (exactify_calls >= exactify_cap) {
                                    break;
                                }
                                const int eidx = candidates[candidate_cursor++];
                                ApproxEntry &entry = approx_entries[(size_t)eidx];
                                const int feature = entry.feature;
                                if (feature < 0 || feature >= n_features_) {
                                    continue;
                                }
                                if (exactified_feature_mask[(size_t)feature]) {
                                    continue;
                                }
                                exactified_feature_mask[(size_t)feature] = 1;
                                ++exactify_calls;
                                ++approx_exactify_features_exact_solved_;
                                if (current_depth == 0) {
                                    ++approx_exactify_features_exact_solved_depth0_;
                                } else if (current_depth == 1) {
                                    ++approx_exactify_features_exact_solved_depth1_;
                                }

                                const double incumbent = std::min(best_ub, best_certified_ub);
                                ensure_exact_feature_evaluated(entry, incumbent);

                                if (entry.exact_feasible) {
                                    best_lb = std::min(best_lb, entry.exact_lb);
                                    if (entry.exact_ub < best_certified_ub - kEpsUpdate ||
                                        (std::fabs(entry.exact_ub - best_certified_ub) <= kEpsUpdate &&
                                         (best_certified_feature < 0 || feature < best_certified_feature))) {
                                        best_certified_ub = entry.exact_ub;
                                        best_certified_lb = entry.exact_lb;
                                        best_certified_tree = entry.exact_tree;
                                        best_certified_feature = feature;
                                    }
                                }

                                if (std::isfinite(best_certified_ub)) {
                                    const double remaining_lhat = min_remaining_lhat();
                                    if (!std::isfinite(remaining_lhat) ||
                                        best_certified_ub <= remaining_lhat + kEpsUpdate) {
                                        stopped_by_separation = true;
                                        break;
                                    }
                                }
                            }
                            if (stopped_by_separation) {
                                break;
                            }
                            if (std::isfinite(cycle_start_best) &&
                                std::isfinite(best_certified_ub) &&
                                best_certified_ub >= cycle_start_best - kEpsUpdate) {
                                stopped_by_no_improve = true;
                                break;
                            }
                        }

                        if (node_ambiguous_size_count > 0) {
                            ++approx_exactify_ambiguous_set_size_seen_;
                            approx_exactify_ambiguous_set_size_mean_node_sum_ +=
                                node_ambiguous_size_sum / (double)node_ambiguous_size_count;
                            ++approx_exactify_ambiguous_set_size_mean_node_count_;
                            approx_exactify_ambiguous_set_size_max_ =
                                std::max(approx_exactify_ambiguous_set_size_max_, (long long)node_ambiguous_size_max);
                            if (node_ambiguous_size_min != std::numeric_limits<int>::max()) {
                                approx_exactify_ambiguous_set_size_min_ = std::min(
                                    approx_exactify_ambiguous_set_size_min_,
                                    (long long)node_ambiguous_size_min);
                            }
                            approx_exactify_ambiguous_set_shrank_steps_ += node_ambiguous_shrank_steps;
                        }
                        if (!stopped_by_separation &&
                            !stopped_by_ambiguous_empty &&
                            !stopped_by_no_improve &&
                            exactify_calls >= exactify_cap) {
                            bool has_unexactified = false;
                            for (int eidx : ranked) {
                                const int feature = approx_entries[(size_t)eidx].feature;
                                if (feature >= 0 && feature < n_features_ &&
                                    !exactified_feature_mask[(size_t)feature]) {
                                    has_unexactified = true;
                                    break;
                                }
                            }
                            if (has_unexactified) {
                                stage_stopped_by_cap = true;
                            } else {
                                stopped_by_separation = true;
                            }
                        }
                        if (stopped_by_ambiguous_empty) {
                            ++approx_exactify_stops_by_ambiguous_empty_;
                        }
                        if (stopped_by_no_improve) {
                            ++approx_exactify_stops_by_no_improve_;
                        }
                        if (stopped_by_separation) {
                            ++approx_exactify_stops_by_separation_;
                            if (current_depth == 0) {
                                ++approx_exactify_stops_by_separation_depth0_;
                            } else if (current_depth == 1) {
                                ++approx_exactify_stops_by_separation_depth1_;
                            }
                        }
                        if (stage_stopped_by_cap) {
                            if (!node_cap_stop_counted) {
                                ++approx_exactify_stops_by_cap_;
                                if (current_depth == 0) {
                                    ++approx_exactify_stops_by_cap_depth0_;
                                } else if (current_depth == 1) {
                                    ++approx_exactify_stops_by_cap_depth1_;
                                }
                                node_cap_stop_counted = true;
                            }
                        }

                        if (std::isfinite(best_certified_ub) && best_certified_tree) {
                            stage_lb = best_certified_lb;
                            stage_ub = best_certified_ub;
                            stage_tree = best_certified_tree;
                            stage_feature = best_certified_feature;
                            stage_resolved = true;
                        } else {
                            ranked = rank_approx(active_features);
                            if (!ranked.empty()) {
                                const ApproxFeatureResult &best_approx = approx_entries[(size_t)ranked[0]].approx;
                                double split_lb = 0.0;
                                double split_ub = 0.0;
                                std::shared_ptr<Node> split_tree;
                                if (build_split_from_group_spans_exact_children(
                                        indices,
                                        best_approx.feature,
                                        depth_remaining,
                                        leaf_tree->prediction,
                                        best_approx.group_spans,
                                        split_lb,
                                        split_ub,
                                        split_tree)) {
                                    stage_lb = split_lb;
                                    stage_ub = split_ub;
                                    stage_tree = split_tree;
                                    stage_feature = best_approx.feature;
                                    stage_resolved = true;
                                }
                            }
                        }
                        if (current_depth == 0 && stage_stopped_by_cap) {
                            std::vector<int> ranked_after = rank_approx(active_features);
                            if (ranked_after.size() >= 2U) {
                                const ApproxFeatureResult &best_after =
                                    approx_entries[(size_t)ranked_after[0]].approx;
                                const ApproxFeatureResult &second_after =
                                    approx_entries[(size_t)ranked_after[1]].approx;
                                const double delta = second_after.uhat - best_after.uhat;
                                const double uncertainty =
                                    std::max(0.0, best_after.uhat - best_after.lhat) +
                                    std::max(0.0, second_after.uhat - second_after.lhat);
                                stage_root_gap_within_uncertainty =
                                    delta <= uncertainty + kEpsUpdate;
                            }
                            stage_root_winner_changed_after_exactify =
                                (stage_pre_exactify_best_feature >= 0 &&
                                 stage_feature >= 0 &&
                                 stage_feature != stage_pre_exactify_best_feature);
                        }
                    }

                    if (stage_resolved && stage_tree) {
                        best_lb = std::min(best_lb, stage_lb);
                        if (stage_ub < best_ub) {
                            best_ub = stage_ub;
                            best_tree = stage_tree;
                            chosen_feature = stage_feature;
                        }
                        node_resolved = true;
                    }
                    bool widened = false;
                    const bool would_widen_by_cap =
                        stage_exactify_entered &&
                        stage_stopped_by_cap &&
                        widen_used < approx_ref_widen_max_;
                    bool allow_widen = would_widen_by_cap;
                    if (current_depth > 0) {
                        if (gain_mass_auto_mode && allow_widen) {
                            ++fast100_widen_forbidden_depth_gt0_attempts_;
                        }
                        // Safety rail: never widen/re-enter at depth >= 1.
                        allow_widen = false;
                    }
                    if (allow_widen && gain_mass_auto_mode) {
                        allow_widen = stage_fast100_importance_passed;
                    }
                    if (allow_widen) {
                        const bool stage_root_unstable =
                            (!stage_resolved) ||
                            stage_root_gap_within_uncertainty ||
                            stage_root_winner_changed_after_exactify;
                        allow_widen = stage_root_unstable;
                    }
                    if (allow_widen) {
                        if (k_current < k_target) {
                            k_current = std::min(
                                k_target,
                                std::max(k_current + 1, std::max(1, 2 * k_current)));
                            widened = true;
                        } else if (approx_ref_shortlist_enabled_ && p_current < feature_scan_limit) {
                            p_current = std::min(
                                feature_scan_limit,
                                std::max(p_current + 1, std::max(1, 2 * p_current)));
                            universe = prefix_features(p_current);
                            shortlist_info = build_ref_shortlist_info(universe);
                            k_target = shortlist_info.k_target;
                            k_current = std::min(shortlist_info.k_target, std::max(shortlist_info.k0, k_current + 1));
                            widened = true;
                        }
                    }
                    if (fast100_debug_this_depth1_node && gain_mass_auto_mode) {
                        const double s_w = stats.pos_weight + stats.neg_weight;
                        std::fprintf(
                            stderr,
                            "[FAST100_DBG_NODE] node=%d depth=%d round=%d S_cnt=%d S_w=%.12g "
                            "k0=%d k_final=%d k_target=%d winner_f=%d UB0=%.12g LB_best_chall=%.12g "
                            "skip_sep=%d frontier=%d exact_solved=%d stop=%s cap=%d allow_widen=%d widened=%d\n",
                            fast100_debug_depth1_node_id,
                            current_depth,
                            widen_used + 1,
                            stats.total_count,
                            s_w,
                            shortlist_info.k0,
                            final_k_used,
                            k_target,
                            stage_winner_feature,
                            stage_winner_ub0,
                            stage_lb_best_challenger,
                            stage_skip_by_ub_lb_separation ? 1 : 0,
                            stage_frontier_size,
                            stage_exactified_count,
                            stage_stop_reason,
                            stage_exactify_cap,
                            allow_widen ? 1 : 0,
                            widened ? 1 : 0);
                    }
                    if (widened) {
                        ++widen_used;
                        ++approx_ref_widen_count_;
                        if (current_depth == 0) {
                            ++approx_ref_widen_count_depth0_;
                        } else {
                            ++approx_ref_widen_count_depth1_;
                        }
                        continue;
                    }
                    done = true;
                }

                approx_ref_k_final_count_ += 1;
                approx_ref_k_final_sum_ += final_k_used;
                approx_ref_k_final_min_ = std::min(approx_ref_k_final_min_, final_k_used);
                approx_ref_k_final_max_ = std::max(approx_ref_k_final_max_, final_k_used);
                if (current_depth == 0) {
                    approx_ref_k_depth0_count_ += 1;
                    approx_ref_k_depth0_sum_ += final_k_used;
                } else if (current_depth == 1) {
                    approx_ref_k_depth1_count_ += 1;
                    approx_ref_k_depth1_sum_ += final_k_used;
                }
                if (chosen_feature < 0 && best_tree && !best_tree->is_leaf) {
                    chosen_feature = best_tree->feature;
                }
                if (chosen_feature >= 0) {
                    int rank = 0;
                    if (chosen_feature >= 0 && chosen_feature < feature_scan_limit) {
                        rank = root_rank_by_feature[(size_t)chosen_feature];
                    }
                    const bool in_initial_shortlist =
                        (chosen_feature >= 0 && chosen_feature < feature_scan_limit)
                            ? (initial_active_mask[(size_t)chosen_feature] != 0)
                            : false;
                    if (current_depth == 0) {
                        approx_ref_chosen_rank_depth0_count_ += 1;
                        approx_ref_chosen_rank_depth0_sum_ += (double)rank;
                        approx_ref_chosen_depth0_total_ += 1;
                        if (in_initial_shortlist) {
                            approx_ref_chosen_depth0_in_initial_ += 1;
                        }
                    } else if (current_depth == 1) {
                        approx_ref_chosen_rank_depth1_count_ += 1;
                        approx_ref_chosen_rank_depth1_sum_ += (double)rank;
                        approx_ref_chosen_depth1_total_ += 1;
                        if (in_initial_shortlist) {
                            approx_ref_chosen_depth1_in_initial_ += 1;
                        }
                    }
                }
            }
            const long long node_patchable_delta = approx_pub_patchable_cells_total_ - node_patchable_before;
            const long long node_patch_calls_delta = approx_greedy_patch_calls_ - node_patch_calls_before;
            if (node_patchable_delta > 0) {
                ++approx_nodes_with_patchable_pub_;
            }
            if (node_patch_calls_delta > 0) {
                ++approx_nodes_with_patch_calls_;
            }
        }

        if (!node_resolved) {
            const bool use_ub0_feature_ordering =
                (partition_strategy_ == kPartitionRushDp) &&
                (std::fabs(branch_penalty_) <= kEpsUpdate) &&
                (current_depth == 0) &&
                !force_rush_legacy_;
            std::vector<RushFeatureRootCache> root_feature_cache;
            std::vector<int> feature_order;
            feature_order.reserve((size_t)n_features_);
            for (int feature = 0; feature < n_features_; ++feature) {
                feature_order.push_back(feature);
            }
            if (use_ub0_feature_ordering) {
                const Clock::time_point ub0_start = rush_profile_enabled_ ? Clock::now() : Clock::time_point{};
                struct FeatureUb0Entry {
                    int feature = -1;
                    double ub0 = kInfinity;
                };
                std::vector<FeatureUb0Entry> ub0_entries;
                ub0_entries.reserve((size_t)n_features_);
                root_feature_cache.assign((size_t)n_features_, RushFeatureRootCache{});
                for (int feature = 0; feature < n_features_; ++feature) {
                    check_timeout();
                    RushFeatureRootCache &cache = root_feature_cache[(size_t)feature];
                    if (build_rush_feature_root_cache(
                            indices,
                            stats,
                            feature,
                            current_depth,
                            leaf_tree->prediction,
                            cache)) {
                        if (cache.ub0 < best_ub) {
                            best_ub = cache.ub0;
                            best_tree = cache.ub0_tree;
                        }
                        ub0_entries.push_back({feature, cache.ub0});
                    } else {
                        ub0_entries.push_back({feature, kInfinity});
                    }
                }
                std::sort(
                    ub0_entries.begin(),
                    ub0_entries.end(),
                    [](const FeatureUb0Entry &lhs, const FeatureUb0Entry &rhs) {
                        if (lhs.ub0 < rhs.ub0 - kEpsUpdate) {
                            return true;
                        }
                        if (rhs.ub0 < lhs.ub0 - kEpsUpdate) {
                            return false;
                        }
                        return lhs.feature < rhs.feature;
                    });
                feature_order.clear();
                feature_order.reserve(ub0_entries.size());
                for (const auto &entry : ub0_entries) {
                    feature_order.push_back(entry.feature);
                }
                if (rush_profile_enabled_) {
                    rush_profile_ub0_ordering_sec_ +=
                        std::chrono::duration<double>(Clock::now() - ub0_start).count();
                }
            }

            for (int feature : feature_order) {
                const RushFeatureRootCache *feature_root_cache = nullptr;
                if (use_ub0_feature_ordering) {
                    const RushFeatureRootCache &cache = root_feature_cache[(size_t)feature];
                    if (!cache.ready) {
                        continue;
                    }
                    if (cache.lb0 >= best_ub - kEpsCert) {
                        ++rush_incumbent_feature_aborts_;
                        continue;
                    }
                    feature_root_cache = &cache;
                }
                double split_lb = 0.0;
                double split_ub = 0.0;
                std::shared_ptr<Node> split_tree;
                bool aborted_by_incumbent = false;
                if (!evaluate_feature_dp(
                        indices,
                        stats,
                        feature,
                        depth_remaining,
                        current_depth,
                        leaf_tree->prediction,
                        best_ub,
                        split_lb,
                        split_ub,
                        split_tree,
                        aborted_by_incumbent,
                        feature_root_cache)) {
                    if (aborted_by_incumbent) {
                        ++rush_incumbent_feature_aborts_;
                    }
                    continue;
                }

                best_lb = std::min(best_lb, split_lb);
                if (split_ub < best_ub) {
                    best_ub = split_ub;
                    best_tree = split_tree;
                }
            }
        }

        best_lb = std::min(best_lb, best_ub);
        BoundResult solved{best_lb, stats.epb_mis, best_ub, best_tree};
        if (best_tree && !best_tree->is_leaf) {
            ++exact_internal_nodes_;
        }
        dp_cache_[key_hash].push_back(DpCacheEntry{depth_remaining, indices, solved});
        ++dp_cache_states_;
        return solved;
    }

    GreedyResult greedy_complete(const std::vector<int> &indices, int depth_remaining) {
        check_timeout();
        ++greedy_subproblem_calls_;
        const uint64_t key_hash = state_hash(indices, depth_remaining);
        auto greedy_bucket_it = greedy_cache_.find(key_hash);
        if (greedy_bucket_it != greedy_cache_.end()) {
            for (const auto &entry : greedy_bucket_it->second) {
                if (entry.depth_remaining == depth_remaining && entry.indices == indices) {
                    ++greedy_cache_hits_;
                    return entry.result;
                }
            }
        }

        const SubproblemStats stats = compute_subproblem_stats(indices);
        auto [leaf_objective, leaf_tree] = leaf_solution(stats);
        if (depth_remaining <= 1 || stats.pure) {
            GreedyResult solved{leaf_objective, leaf_tree};
            maybe_clear_greedy_cache_for_approx_insert();
            greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
            register_greedy_cache_insert();
            return solved;
        }

        double best_objective = leaf_objective;
        std::shared_ptr<Node> best_tree = leaf_tree;

        for (int feature = 0; feature < n_features_; ++feature) {
            double split_objective = 0.0;
            std::shared_ptr<Node> split_tree;
            if (!evaluate_feature_greedy(indices, feature, depth_remaining, leaf_tree->prediction, split_objective, split_tree)) {
                continue;
            }

            if (split_objective < best_objective) {
                best_objective = split_objective;
                best_tree = split_tree;
            }
        }

        GreedyResult solved{best_objective, best_tree};
        if (best_tree && !best_tree->is_leaf) {
            ++greedy_internal_nodes_;
        }
        maybe_clear_greedy_cache_for_approx_insert();
        greedy_cache_[key_hash].push_back(GreedyCacheEntry{depth_remaining, indices, solved});
        register_greedy_cache_insert();
        return solved;
    }

    static nlohmann::json to_json(const std::shared_ptr<Node> &node) {
        if (!node) {
            return nlohmann::json::object();
        }
        if (node->is_leaf) {
            return nlohmann::json{
                {"type", "leaf"},
                {"prediction", node->prediction},
                {"loss", node->loss},
                {"n_samples", node->n_samples},
                {"class_counts", nlohmann::json::array({node->neg_count, node->pos_count})},
            };
        }

        nlohmann::json groups = nlohmann::json::array();
        for (size_t i = 0; i < node->group_nodes.size(); ++i) {
            nlohmann::json spans = nlohmann::json::array();
            for (const auto &span : node->group_bin_spans[(size_t)i]) {
                spans.push_back(nlohmann::json::array({span.first, span.second}));
            }
            groups.push_back(nlohmann::json{
                {"spans", std::move(spans)},
                {"child", to_json(node->group_nodes[(size_t)i])},
            });
        }

        return nlohmann::json{
            {"type", "node"},
            {"feature", node->feature},
            {"fallback_bin", node->fallback_bin},
            {"fallback_prediction", node->fallback_prediction},
            {"group_count", node->group_count},
            {"n_samples", node->n_samples},
            {"groups", groups},
        };
    }
};

}  // namespace

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
    int partition_strategy,
    bool approx_mode,
    int patch_budget_per_feature,
    int exactify_top_m,
    int tau_mode,
    int approx_feature_scan_limit,
    bool approx_ref_shortlist_enabled,
    int approx_ref_widen_max,
    bool approx_challenger_sweep_enabled,
    int approx_challenger_sweep_max_features,
    int approx_challenger_sweep_max_patch_calls_per_node
) {
    Solver solver(
        x_flat,
        n_rows,
        n_features,
        y,
        sample_weight,
        full_depth_budget,
        lookahead_depth_budget,
        regularization,
        branch_penalty,
        min_child_size,
        time_limit_seconds,
        max_branching,
        partition_strategy,
        approx_mode,
        patch_budget_per_feature,
        exactify_top_m,
        tau_mode,
        approx_feature_scan_limit,
        approx_ref_shortlist_enabled,
        approx_ref_widen_max,
        approx_challenger_sweep_enabled,
        approx_challenger_sweep_max_features,
        approx_challenger_sweep_max_patch_calls_per_node);
    return solver.fit();
}

}  // namespace msplit
