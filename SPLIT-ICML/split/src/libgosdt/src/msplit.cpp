#include "msplit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace msplit {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

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
    std::vector<int> child_bins;
    std::vector<std::shared_ptr<Node>> child_nodes;
};

struct BoundResult {
    double lb = 0.0;
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

class Solver {
   public:
    Solver(
        const std::vector<int> &x_flat,
        int n_rows,
        int n_features,
        const std::vector<int> &y,
        int full_depth_budget,
        int lookahead_depth_budget,
        double regularization,
        double branch_penalty,
        int min_child_size,
        double time_limit_seconds,
        int max_branching
    )
        : x_flat_(x_flat),
          n_rows_(n_rows),
          n_features_(n_features),
          y_(y),
          full_depth_budget_(full_depth_budget),
          lookahead_depth_budget_(lookahead_depth_budget),
          regularization_(regularization),
          branch_penalty_(branch_penalty),
          min_child_size_(min_child_size),
          time_limit_seconds_(time_limit_seconds),
          max_branching_(max_branching),
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

        effective_lookahead_ = std::min(full_depth_budget_, lookahead_depth_budget_);
    }

    FitResult fit() {
        std::vector<int> root_indices(n_rows_);
        for (int i = 0; i < n_rows_; ++i) {
            root_indices[i] = i;
        }

        BoundResult solved = solve_subproblem(root_indices, full_depth_budget_);

        FitResult out;
        out.tree = to_json(solved.tree);
        out.lowerbound = solved.lb;
        out.upperbound = solved.ub;
        out.objective = solved.ub;
        out.exact_internal_nodes = exact_internal_nodes_;
        out.greedy_internal_nodes = greedy_internal_nodes_;
        return out;
    }

   private:
    const std::vector<int> &x_flat_;
    int n_rows_;
    int n_features_;
    const std::vector<int> &y_;

    int full_depth_budget_;
    int lookahead_depth_budget_;
    int effective_lookahead_;
    double regularization_;
    double branch_penalty_;
    int min_child_size_;
    double time_limit_seconds_;
    int max_branching_;
    int exact_internal_nodes_ = 0;
    int greedy_internal_nodes_ = 0;

    Clock::time_point start_time_;

    std::unordered_map<std::string, BoundResult> dp_cache_;
    std::unordered_map<std::string, GreedyResult> greedy_cache_;

    int x(int row, int feature) const { return x_flat_[row * n_features_ + feature]; }

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

    static std::string subset_key(const std::vector<int> &indices) {
        std::string key;
        int size = (int)indices.size();
        key.reserve((size + 1) * (int)sizeof(int));
        key.append(reinterpret_cast<const char *>(&size), sizeof(int));
        for (int value : indices) {
            key.append(reinterpret_cast<const char *>(&value), sizeof(int));
        }
        return key;
    }

    static std::string dp_key(const std::vector<int> &indices, int depth_remaining) {
        std::string key = subset_key(indices);
        key.append(reinterpret_cast<const char *>(&depth_remaining), sizeof(int));
        return key;
    }

    static std::string greedy_key(const std::vector<int> &indices, int depth_remaining) {
        std::string key = subset_key(indices);
        key.append(reinterpret_cast<const char *>(&depth_remaining), sizeof(int));
        return key;
    }

    double split_penalty_for_groups(int groups) const {
        if (groups <= 2) {
            return 0.0;
        }
        return branch_penalty_ * static_cast<double>(groups - 2);
    }

    bool is_pure(const std::vector<int> &indices) const {
        if (indices.empty()) {
            return true;
        }
        int first = y_[indices.front()];
        for (int idx : indices) {
            if (y_[idx] != first) {
                return false;
            }
        }
        return true;
    }

    std::pair<double, std::shared_ptr<Node>> leaf_solution(const std::vector<int> &indices) const {
        int positives = 0;
        int total = (int)indices.size();
        for (int idx : indices) {
            positives += y_[idx];
        }
        int negatives = total - positives;

        int prediction = (positives >= negatives) ? 1 : 0;
        int mistakes = (prediction == 1) ? negatives : positives;

        auto leaf = std::make_shared<Node>();
        leaf->is_leaf = true;
        leaf->prediction = prediction;
        leaf->n_samples = total;
        leaf->neg_count = negatives;
        leaf->pos_count = positives;
        leaf->loss = (double)mistakes / (double)n_rows_ + regularization_;

        return {leaf->loss, leaf};
    }

    bool build_ordered_bins(const std::vector<int> &indices, int feature, OrderedBins &out) const {
        std::unordered_map<int, std::vector<int>> by_bin;
        by_bin.reserve(indices.size());

        for (int idx : indices) {
            by_bin[x(idx, feature)].push_back(idx);
        }
        if (by_bin.size() <= 1) {
            return false;
        }

        out.values.clear();
        out.values.reserve(by_bin.size());
        for (const auto &entry : by_bin) {
            out.values.push_back(entry.first);
        }
        std::sort(out.values.begin(), out.values.end());

        out.members.clear();
        out.members.reserve(out.values.size());
        out.prefix_counts.assign(out.values.size() + 1, 0);
        for (size_t i = 0; i < out.values.size(); ++i) {
            auto it = by_bin.find(out.values[i]);
            if (it == by_bin.end()) {
                return false;
            }
            out.members.push_back(it->second);
            out.prefix_counts[i + 1] = out.prefix_counts[i] + (int)it->second.size();
        }

        return true;
    }

    PartitionResult optimize_partition(
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
            if (child_size > largest_child_size) {
                largest_child_size = child_size;
                fallback_bin = bins.values[left];
            }

            for (int pos = left; pos <= right; ++pos) {
                internal->child_bins.push_back(bins.values[pos]);
                internal->child_nodes.push_back(child);
            }
        }

        if (fallback_bin < 0 && !bins.values.empty()) {
            fallback_bin = bins.values.front();
        }
        internal->fallback_bin = fallback_bin;

        return internal;
    }

    bool evaluate_feature_dp(
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
        const int q_max = max_groups_for_bins(n_bins);
        if (q_max < 2) {
            return false;
        }

        std::vector<std::vector<bool>> valid(n_bins, std::vector<bool>(n_bins, false));
        std::vector<std::vector<double>> interval_lb(n_bins, std::vector<double>(n_bins, kInfinity));
        std::vector<std::vector<double>> interval_ub(n_bins, std::vector<double>(n_bins, kInfinity));
        std::vector<std::vector<std::shared_ptr<Node>>> interval_trees(
            n_bins,
            std::vector<std::shared_ptr<Node>>(n_bins, nullptr));

        for (int left = 0; left < n_bins; ++left) {
            std::vector<int> subset;
            subset.reserve((size_t)(bins.prefix_counts[n_bins] - bins.prefix_counts[left]));
            for (int right = left; right < n_bins; ++right) {
                subset.insert(subset.end(), bins.members[right].begin(), bins.members[right].end());
                int child_count = bins.prefix_counts[right + 1] - bins.prefix_counts[left];
                if (child_count < min_child_size_) {
                    continue;
                }

                std::vector<int> canonical_subset = subset;
                std::sort(canonical_subset.begin(), canonical_subset.end());
                BoundResult child = solve_subproblem(canonical_subset, depth_remaining - 1);
                valid[left][right] = true;
                interval_lb[left][right] = child.lb;
                interval_ub[left][right] = child.ub;
                interval_trees[left][right] = child.tree;
            }
        }

        PartitionResult lb_partition = optimize_partition(interval_lb, valid, q_max, false);
        PartitionResult ub_partition = optimize_partition(interval_ub, valid, q_max, true);
        if (!ub_partition.feasible) {
            return false;
        }

        const double ub_penalty = split_penalty_for_groups(ub_partition.groups);
        split_ub = ub_partition.cost + ub_penalty;
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
        const int q_max = max_groups_for_bins(n_bins);
        if (q_max < 2) {
            return false;
        }

        std::vector<std::vector<bool>> valid(n_bins, std::vector<bool>(n_bins, false));
        std::vector<std::vector<double>> interval_obj(n_bins, std::vector<double>(n_bins, kInfinity));
        std::vector<std::vector<std::shared_ptr<Node>>> interval_trees(
            n_bins,
            std::vector<std::shared_ptr<Node>>(n_bins, nullptr));

        for (int left = 0; left < n_bins; ++left) {
            std::vector<int> subset;
            subset.reserve((size_t)(bins.prefix_counts[n_bins] - bins.prefix_counts[left]));
            for (int right = left; right < n_bins; ++right) {
                subset.insert(subset.end(), bins.members[right].begin(), bins.members[right].end());
                int child_count = bins.prefix_counts[right + 1] - bins.prefix_counts[left];
                if (child_count < min_child_size_) {
                    continue;
                }

                std::vector<int> canonical_subset = subset;
                std::sort(canonical_subset.begin(), canonical_subset.end());
                GreedyResult child = greedy_complete(canonical_subset, depth_remaining - 1);
                valid[left][right] = true;
                interval_obj[left][right] = child.objective;
                interval_trees[left][right] = child.tree;
            }
        }

        PartitionResult partition = optimize_partition(interval_obj, valid, q_max, true);
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

    BoundResult solve_subproblem(const std::vector<int> &indices, int depth_remaining) {
        check_timeout();

        std::string key = dp_key(indices, depth_remaining);
        auto dp_it = dp_cache_.find(key);
        if (dp_it != dp_cache_.end()) {
            return dp_it->second;
        }

        auto [leaf_objective, leaf_tree] = leaf_solution(indices);
        if (depth_remaining <= 1 || is_pure(indices)) {
            BoundResult solved{leaf_objective, leaf_objective, leaf_tree};
            dp_cache_.insert({key, solved});
            return solved;
        }

        const int current_depth = full_depth_budget_ - depth_remaining;
        if (current_depth == effective_lookahead_) {
            GreedyResult greedy = greedy_complete(indices, depth_remaining);
            BoundResult solved{greedy.objective, greedy.objective, greedy.tree};
            dp_cache_.insert({key, solved});
            return solved;
        }

        double best_lb = leaf_objective;
        double best_ub = leaf_objective;
        std::shared_ptr<Node> best_tree = leaf_tree;

        for (int feature = 0; feature < n_features_; ++feature) {
            double split_lb = 0.0;
            double split_ub = 0.0;
            std::shared_ptr<Node> split_tree;
            if (!evaluate_feature_dp(
                    indices,
                    feature,
                    depth_remaining,
                    leaf_tree->prediction,
                    split_lb,
                    split_ub,
                    split_tree)) {
                continue;
            }

            best_lb = std::min(best_lb, split_lb);
            if (split_ub < best_ub) {
                best_ub = split_ub;
                best_tree = split_tree;
            }
        }

        best_lb = std::min(best_lb, best_ub);
        BoundResult solved{best_lb, best_ub, best_tree};
        if (best_tree && !best_tree->is_leaf) {
            ++exact_internal_nodes_;
        }
        dp_cache_.insert({key, solved});
        return solved;
    }

    GreedyResult greedy_complete(const std::vector<int> &indices, int depth_remaining) {
        check_timeout();

        std::string key = greedy_key(indices, depth_remaining);
        auto greedy_it = greedy_cache_.find(key);
        if (greedy_it != greedy_cache_.end()) {
            return greedy_it->second;
        }

        auto [leaf_objective, leaf_tree] = leaf_solution(indices);
        if (depth_remaining <= 1 || is_pure(indices)) {
            GreedyResult solved{leaf_objective, leaf_tree};
            greedy_cache_.insert({key, solved});
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
        greedy_cache_.insert({key, solved});
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

        nlohmann::json children = nlohmann::json::object();
        for (size_t i = 0; i < node->child_bins.size(); ++i) {
            children[std::to_string(node->child_bins[i])] = to_json(node->child_nodes[i]);
        }

        return nlohmann::json{
            {"type", "node"},
            {"feature", node->feature},
            {"fallback_bin", node->fallback_bin},
            {"fallback_prediction", node->fallback_prediction},
            {"group_count", node->group_count},
            {"n_samples", node->n_samples},
            {"children", children},
        };
    }
};

}  // namespace

FitResult fit(
    const std::vector<int> &x_flat,
    int n_rows,
    int n_features,
    const std::vector<int> &y,
    int full_depth_budget,
    int lookahead_depth_budget,
    double regularization,
    double branch_penalty,
    int min_child_size,
    double time_limit_seconds,
    int max_branching
) {
    Solver solver(
        x_flat,
        n_rows,
        n_features,
        y,
        full_depth_budget,
        lookahead_depth_budget,
        regularization,
        branch_penalty,
        min_child_size,
        time_limit_seconds,
        max_branching);
    return solver.fit();
}

}  // namespace msplit
