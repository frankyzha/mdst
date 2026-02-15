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
};

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
);

}  // namespace msplit

#endif
