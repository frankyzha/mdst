    AtomizedCandidate refine_atomized_candidate_debr(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedCandidate &seed,
        double mu_node,
        const std::vector<std::pair<int, int>> *active_windows = nullptr,
        AtomizedRefinementSummary *summary = nullptr,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (!seed.feasible) {
            return seed;
        }
        ++profiling_refine_calls_;
        ScopedTimer refine_timer(profiling_refine_sec_);

        AtomizedRefinementSummary local_summary;

        const int atom_count = (int)atoms.size();
        const int groups = seed.groups;
        if (atom_count <= 1 || groups <= 1) {
            return seed;
        }
        const bool hard_loss_mode = (mode == AtomizedObjectiveMode::kHardLoss);
        ++local_summary.bridge_policy_calls;
        auto bump_histogram = [](std::vector<long long> &hist, int bucket) {
            if (bucket < 0) {
                return;
            }
            const size_t idx = static_cast<size_t>(bucket);
            if (hist.size() <= idx) {
                hist.resize(idx + 1U, 0);
            }
            ++hist[idx];
        };

        struct RankedMove {
            AtomizedRefinementMove move;
            double primary_gain = 0.0;
            double guide_gain = 0.0;
            double complexity_gain = 0.0;
            double bridge_gain = 0.0;
            bool valid = false;
        };

        auto ranked_move_better_metadata = [](const RankedMove &lhs, const RankedMove &rhs) {
            if (lhs.move.length != rhs.move.length) {
                return lhs.move.length < rhs.move.length;
            }
            if (lhs.move.source_group != rhs.move.source_group) {
                return lhs.move.source_group < rhs.move.source_group;
            }
            if (lhs.move.target_group != rhs.move.target_group) {
                return lhs.move.target_group < rhs.move.target_group;
            }
            if (lhs.move.start != rhs.move.start) {
                return lhs.move.start < rhs.move.start;
            }
            return lhs.move.end < rhs.move.end;
        };

        auto ranked_move_better_descent = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.primary_gain > rhs.primary_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.primary_gain > lhs.primary_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        auto ranked_move_better_bridge = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.bridge_gain > rhs.bridge_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.bridge_gain > lhs.bridge_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        auto ranked_move_better_cleanup = [&](const RankedMove &lhs, const RankedMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.complexity_gain > rhs.complexity_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.complexity_gain > lhs.complexity_gain + kEpsUpdate) {
                return false;
            }
            if (lhs.guide_gain > rhs.guide_gain + kEpsUpdate) {
                return true;
            }
            if (rhs.guide_gain > lhs.guide_gain + kEpsUpdate) {
                return false;
            }
            return ranked_move_better_metadata(lhs, rhs);
        };

        const bool bridge_allowed = (active_windows == nullptr || active_windows->empty());
        bool bridge_used = false;
        const bool has_active_windows = active_windows != nullptr && !active_windows->empty();
        if (has_active_windows) {
            ++local_summary.refine_windowed_calls;
        } else {
            ++local_summary.refine_unwindowed_calls;
        }
        bool saw_overlap_segment = false;

        auto make_ranked_move = [&](AtomizedRefinementMove &&move,
                                    double primary_gain,
                                    double guide_gain) {
            RankedMove ranked;
            ranked.move = std::move(move);
            ranked.primary_gain = primary_gain;
            ranked.guide_gain = guide_gain;
            ranked.complexity_gain = -static_cast<double>(ranked.move.delta_components);
            ranked.bridge_gain = ranked.guide_gain + mu_node * ranked.complexity_gain;
            ranked.valid = true;
            return ranked;
        };

        auto consider_ranked_move = [&](AtomizedRefinementMove &&move,
                                        double primary_gain,
                                        double guide_gain,
                                        RankedMove &best_descent,
                                        RankedMove &best_bridge) {
            ++local_summary.candidate_legal;
            const bool is_descent = primary_gain > kEpsUpdate;
            RankedMove ranked = make_ranked_move(
                std::move(move),
                primary_gain,
                guide_gain);
            if (is_descent) {
                ++local_summary.candidate_descent_eligible;
                if (!best_descent.valid || ranked_move_better_descent(ranked, best_descent)) {
                    best_descent = std::move(ranked);
                }
                return;
            }

            ++local_summary.candidate_descent_rejected;
            const bool primary_near_zero = std::abs(primary_gain) <= kEpsUpdate;
            if (primary_near_zero) {
                if (active_windows != nullptr && !active_windows->empty()) {
                    ++local_summary.candidate_bridge_window_blocked;
                } else if (bridge_used) {
                    ++local_summary.candidate_bridge_used_blocked;
                } else if (ranked.guide_gain > kEpsUpdate) {
                    ++local_summary.candidate_bridge_eligible;
                    if (!best_bridge.valid || ranked_move_better_bridge(ranked, best_bridge)) {
                        best_bridge = std::move(ranked);
                    }
                    return;
                } else {
                    ++local_summary.candidate_bridge_guide_rejected;
                }
            }
            ++local_summary.candidate_score_rejected;
        };

        auto atomized_move_is_improving = [&](const AtomizedRefinementMove &move) {
            return move.valid &&
                (move.delta_j > kEpsUpdate ||
                 (std::abs(move.delta_j) <= kEpsUpdate && move.delta_soft > kEpsUpdate));
        };

        auto atomized_move_better = [&](const AtomizedRefinementMove &lhs,
                                        const AtomizedRefinementMove &rhs) {
            if (!rhs.valid) {
                return lhs.valid;
            }
            if (!lhs.valid) {
                return false;
            }
            if (lhs.delta_j > rhs.delta_j + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_j > lhs.delta_j + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_hard > rhs.delta_hard + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_hard > lhs.delta_hard + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_soft > rhs.delta_soft + kEpsUpdate) {
                return true;
            }
            if (rhs.delta_soft > lhs.delta_soft + kEpsUpdate) {
                return false;
            }
            if (lhs.delta_components != rhs.delta_components) {
                return lhs.delta_components < rhs.delta_components;
            }
            if (lhs.length != rhs.length) {
                return lhs.length < rhs.length;
            }
            if (lhs.source_group != rhs.source_group) {
                return lhs.source_group < rhs.source_group;
            }
            if (lhs.target_group != rhs.target_group) {
                return lhs.target_group < rhs.target_group;
            }
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.end < rhs.end;
        };

        if (!binary_mode_) {
            std::vector<int> assign = seed.assignment;
            std::vector<int> branch_rows((size_t)groups, 0);
            std::vector<double> branch_class_weight(
                static_cast<size_t>(groups) * static_cast<size_t>(n_classes_), 0.0);
            std::vector<double> branch_teacher_class_weight(
                static_cast<size_t>(groups) * static_cast<size_t>(n_classes_), 0.0);
            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                if (group_idx < 0 || group_idx >= groups) {
                    return seed;
                }
                branch_rows[(size_t)group_idx] += atoms[(size_t)atom_pos].row_count;
                const size_t base = static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[base + static_cast<size_t>(cls)] +=
                        atoms[(size_t)atom_pos].class_weight[(size_t)cls];
                    branch_teacher_class_weight[base + static_cast<size_t>(cls)] +=
                        atoms[(size_t)atom_pos].teacher_class_weight[(size_t)cls];
                }
            }

            std::vector<double> branch_loss((size_t)groups, 0.0);
            std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
            std::vector<double> branch_soft_impurity((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                const size_t group_base =
                    static_cast<size_t>(group_idx) * static_cast<size_t>(n_classes_);
                branch_loss[(size_t)group_idx] =
                    split_leaf_loss_flat(branch_class_weight, n_classes_, group_base);
                branch_hard_impurity[(size_t)group_idx] =
                    hard_label_impurity_flat(branch_class_weight, n_classes_, group_base);
                branch_soft_impurity[(size_t)group_idx] =
                    hard_label_impurity_flat(branch_teacher_class_weight, n_classes_, group_base);
            }

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_descent;
                RankedMove best_bridge;
                int component_start = 0;
                while (component_start < atom_count) {
                    const int source_group = assign[(size_t)component_start];
                    int component_end = component_start + 1;
                    while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                        ++component_end;
                    }
                    if (!seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(
                        local_summary.source_component_atom_size_histogram,
                        component_end - component_start);
                    int component_rows_total = 0;
                    for (int pos = component_start; pos < component_end; ++pos) {
                        component_rows_total += atoms[(size_t)pos].row_count;
                    }
                    bump_histogram(local_summary.source_component_row_size_histogram, component_rows_total);

                    for (int target_group = 0; target_group < groups; ++target_group) {
                        if (target_group == source_group) {
                            continue;
                        }
                        const int source_rows_total = branch_rows[(size_t)source_group];
                        const int target_rows_total = branch_rows[(size_t)target_group];
                        const double source_loss_before = branch_loss[(size_t)source_group];
                        const double target_loss_before = branch_loss[(size_t)target_group];
                        const double source_hard_before = branch_hard_impurity[(size_t)source_group];
                        const double target_hard_before = branch_hard_impurity[(size_t)target_group];
                        const double source_soft_before = branch_soft_impurity[(size_t)source_group];
                        const double target_soft_before = branch_soft_impurity[(size_t)target_group];
                        auto scan_component_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                std::vector<double> move_teacher_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                        move_teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
                                        continue;
                                    }

                                    const size_t source_base =
                                        static_cast<size_t>(source_group) * static_cast<size_t>(n_classes_);
                                    const size_t target_base =
                                        static_cast<size_t>(target_group) * static_cast<size_t>(n_classes_);
                                    const double source_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_teacher_class_weight,
                                        false);
                                    const double target_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_teacher_class_weight,
                                        true);
                                    const double primary_before = hard_loss_mode
                                        ? (source_loss_before + target_loss_before)
                                        : (source_hard_before + target_hard_before +
                                           family1_soft_weight_value_ *
                                               (source_soft_before + target_soft_before));
                                    const double primary_after = hard_loss_mode
                                        ? (source_loss_after + target_loss_after)
                                        : (source_hard_after + target_hard_after +
                                           family1_soft_weight_value_ *
                                               (source_soft_after + target_soft_after));
                                    const double guide_before = hard_loss_mode
                                        ? (source_hard_before + target_hard_before)
                                        : (source_loss_before + target_loss_before);
                                    const double guide_after = hard_loss_mode
                                        ? (source_hard_after + target_hard_after)
                                        : (source_loss_after + target_loss_after);
                                    const double delta_primary = primary_before - primary_after;
                                    const double delta_guide = guide_before - guide_after;
                                    const int source_delta_components =
                                        (start == component_start && end == component_end - 1)
                                            ? -1
                                            : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
                                    int target_delta_components = 0;
                                    if (!left_target && !right_target) {
                                        target_delta_components = 1;
                                    } else if (left_target && right_target) {
                                        target_delta_components = -1;
                                    }

                                    AtomizedRefinementMove move;
                                    move.valid = true;
                                    move.source_group = source_group;
                                    move.target_group = target_group;
                                    move.start = start;
                                    move.end = end;
                                    move.length = end - start + 1;
                                    move.delta_components = source_delta_components + target_delta_components;
                                    move.row_count = move_rows;
                                    move.class_weight = move_class_weight;
                                    move.teacher_class_weight = move_teacher_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.source_soft_impurity_after = source_soft_after;
                                move.target_soft_impurity_after = target_soft_after;
                                move.delta_hard = delta_primary;
                                move.delta_soft = delta_guide;
                                move.delta_j = delta_primary - mu_node * (double)move.delta_components;
                                consider_ranked_move(
                                    std::move(move),
                                    delta_primary,
                                    delta_guide,
                                    best_descent,
                                    best_bridge);
                                }
                            }
                        };

                        if (active_windows == nullptr || active_windows->empty()) {
                            scan_component_overlap(component_start, component_end - 1);
                            saw_overlap_segment = true;
                            ++local_summary.refine_overlap_segments;
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(component_start, window.first);
                                const int overlap_end = std::min(component_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    saw_overlap_segment = true;
                                    ++local_summary.refine_overlap_segments;
                                    scan_component_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                component_start = component_end;
            }

            if (saw_overlap_segment) {
                ++local_summary.refine_calls_with_overlap;
            } else {
                ++local_summary.refine_calls_without_overlap;
            }

            const RankedMove *chosen_ranked = nullptr;
            const AtomizedRefinementMove *best_move_ptr = nullptr;
                if (best_descent.valid) {
                    chosen_ranked = &best_descent;
                } else if (bridge_allowed && !bridge_used && best_bridge.valid) {
                    chosen_ranked = &best_bridge;
                    bridge_used = true;
                } else {
                    break;
                }
                best_move_ptr = &chosen_ranked->move;
                const AtomizedRefinementMove &best_move = *best_move_ptr;

                for (int atom_pos = best_move.start; atom_pos <= best_move.end; ++atom_pos) {
                    assign[(size_t)atom_pos] = best_move.target_group;
                }

                branch_rows[(size_t)best_move.source_group] -= best_move.row_count;
                branch_rows[(size_t)best_move.target_group] += best_move.row_count;
                const size_t source_base =
                    static_cast<size_t>(best_move.source_group) * static_cast<size_t>(n_classes_);
                const size_t target_base =
                    static_cast<size_t>(best_move.target_group) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_move.class_weight[(size_t)cls];
                    branch_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_move.class_weight[(size_t)cls];
                    branch_teacher_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_move.teacher_class_weight[(size_t)cls];
                    branch_teacher_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_move.teacher_class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
                branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
                branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;
                branch_soft_impurity[(size_t)best_move.source_group] = best_move.source_soft_impurity_after;
                branch_soft_impurity[(size_t)best_move.target_group] = best_move.target_soft_impurity_after;

                local_summary.moves += 1;
                local_summary.hard_gain += chosen_ranked->primary_gain;
                local_summary.soft_gain += chosen_ranked->guide_gain;
                if (chosen_ranked == &best_descent) {
                    ++local_summary.descent_moves;
                } else {
                    ++local_summary.bridge_moves;
                }
                local_summary.delta_j += best_move.delta_j;
                local_summary.component_delta += best_move.delta_components;
            }

            while (true) {
                check_timeout();
                std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

                RankedMove best_cleanup;
                int component_start = 0;
                while (component_start < atom_count) {
                    const int source_group = assign[(size_t)component_start];
                    int component_end = component_start + 1;
                    while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                        ++component_end;
                    }
                    if (!seen_source_groups[(size_t)source_group]) {
                        seen_source_groups[(size_t)source_group] = 1U;
                        bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                    }
                    bump_histogram(
                        local_summary.source_component_atom_size_histogram,
                        component_end - component_start);

                    for (int target_group = 0; target_group < groups; ++target_group) {
                        if (target_group == source_group) {
                            continue;
                        }
                        const int source_rows_total = branch_rows[(size_t)source_group];
                        const int target_rows_total = branch_rows[(size_t)target_group];
                        const double source_loss_before = branch_loss[(size_t)source_group];
                        const double target_loss_before = branch_loss[(size_t)target_group];
                        const double source_hard_before = branch_hard_impurity[(size_t)source_group];
                        const double target_hard_before = branch_hard_impurity[(size_t)target_group];
                        const double source_soft_before = branch_soft_impurity[(size_t)source_group];
                        const double target_soft_before = branch_soft_impurity[(size_t)target_group];
                        auto scan_component_overlap = [&](int lo, int hi) {
                            if (lo > hi) {
                                return;
                            }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                std::vector<double> move_class_weight((size_t)n_classes_, 0.0);
                                std::vector<double> move_teacher_class_weight((size_t)n_classes_, 0.0);
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    for (int cls = 0; cls < n_classes_; ++cls) {
                                        move_class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                                        move_teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                                    }

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
                                        continue;
                                    }

                                    const size_t source_base =
                                        static_cast<size_t>(source_group) * static_cast<size_t>(n_classes_);
                                    const size_t target_base =
                                        static_cast<size_t>(target_group) * static_cast<size_t>(n_classes_);
                                    const double source_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_loss_after = split_leaf_loss_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_class_weight,
                                        false);
                                    const double target_hard_after = hard_label_impurity_flat_delta(
                                        branch_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_class_weight,
                                        true);
                                    const double source_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        source_base,
                                        move_teacher_class_weight,
                                        false);
                                    const double target_soft_after = hard_label_impurity_flat_delta(
                                        branch_teacher_class_weight,
                                        n_classes_,
                                        target_base,
                                        move_teacher_class_weight,
                                        true);
                                    const double primary_gain = hard_loss_mode
                                        ? ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after))
                                        : ((source_hard_before + target_hard_before +
                                            family1_soft_weight_value_ *
                                                (source_soft_before + target_soft_before)) -
                                           (source_hard_after + target_hard_after +
                                            family1_soft_weight_value_ *
                                                (source_soft_after + target_soft_after)));
                                    const double guide_gain = hard_loss_mode
                                        ? ((source_hard_before + target_hard_before +
                                            source_soft_before + target_soft_before) -
                                           (source_hard_after + target_hard_after +
                                            source_soft_after + target_soft_after))
                                        : ((source_loss_before + target_loss_before) -
                                           (source_loss_after + target_loss_after));
                                    const int source_delta_components =
                                        (start == component_start && end == component_end - 1)
                                            ? -1
                                            : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                    const bool left_target =
                                        (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                    const bool right_target =
                                        (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
                                    int target_delta_components = 0;
                                    if (!left_target && !right_target) {
                                        target_delta_components = 1;
                                    } else if (left_target && right_target) {
                                        target_delta_components = -1;
                                    }

                                    AtomizedRefinementMove move;
                                    move.valid = true;
                                    move.source_group = source_group;
                                    move.target_group = target_group;
                                    move.start = start;
                                    move.end = end;
                                    move.length = end - start + 1;
                                    move.delta_components = source_delta_components + target_delta_components;
                                    move.row_count = move_rows;
                                    move.class_weight = move_class_weight;
                                    move.teacher_class_weight = move_teacher_class_weight;
                                    move.source_loss_after = source_loss_after;
                                    move.target_loss_after = target_loss_after;
                                    move.source_hard_impurity_after = source_hard_after;
                                    move.target_hard_impurity_after = target_hard_after;
                                    move.source_soft_impurity_after = source_soft_after;
                                    move.target_soft_impurity_after = target_soft_after;
                                    move.delta_hard = primary_gain;
                                    move.delta_soft = guide_gain;
                                    move.delta_j = primary_gain - mu_node * (double)move.delta_components;
                                    const double complexity_gain = -static_cast<double>(move.delta_components);
                                    ++local_summary.candidate_legal;
                                    if (std::abs(primary_gain) > kEpsUpdate) {
                                        ++local_summary.candidate_cleanup_primary_rejected;
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    if (complexity_gain <= kEpsUpdate) {
                                        ++local_summary.candidate_cleanup_complexity_rejected;
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    ++local_summary.candidate_cleanup_eligible;
                                    RankedMove ranked = make_ranked_move(
                                        std::move(move),
                                        primary_gain,
                                        guide_gain);
                                    if (!best_cleanup.valid || ranked_move_better_cleanup(ranked, best_cleanup)) {
                                        best_cleanup = std::move(ranked);
                                    }
                                }
                            }
                        };

                        if (active_windows == nullptr || active_windows->empty()) {
                            scan_component_overlap(component_start, component_end - 1);
                        } else {
                            for (const auto &window : *active_windows) {
                                const int overlap_start = std::max(component_start, window.first);
                                const int overlap_end = std::min(component_end - 1, window.second);
                                if (overlap_start <= overlap_end) {
                                    scan_component_overlap(overlap_start, overlap_end);
                                }
                            }
                        }
                    }

                    component_start = component_end;
                }

                if (!best_cleanup.valid) {
                    break;
                }

                const AtomizedRefinementMove &best_cleanup_move = best_cleanup.move;
                for (int atom_pos = best_cleanup_move.start; atom_pos <= best_cleanup_move.end; ++atom_pos) {
                    assign[(size_t)atom_pos] = best_cleanup_move.target_group;
                }

                branch_rows[(size_t)best_cleanup_move.source_group] -= best_cleanup_move.row_count;
                branch_rows[(size_t)best_cleanup_move.target_group] += best_cleanup_move.row_count;
                const size_t source_base =
                    static_cast<size_t>(best_cleanup_move.source_group) * static_cast<size_t>(n_classes_);
                const size_t target_base =
                    static_cast<size_t>(best_cleanup_move.target_group) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    branch_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_cleanup_move.class_weight[(size_t)cls];
                    branch_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_cleanup_move.class_weight[(size_t)cls];
                    branch_teacher_class_weight[source_base + static_cast<size_t>(cls)] -=
                        best_cleanup_move.teacher_class_weight[(size_t)cls];
                    branch_teacher_class_weight[target_base + static_cast<size_t>(cls)] +=
                        best_cleanup_move.teacher_class_weight[(size_t)cls];
                }
                branch_loss[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_loss_after;
                branch_loss[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_loss_after;
                branch_hard_impurity[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_hard_impurity_after;
                branch_hard_impurity[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_hard_impurity_after;
                branch_soft_impurity[(size_t)best_cleanup_move.source_group] = best_cleanup_move.source_soft_impurity_after;
                branch_soft_impurity[(size_t)best_cleanup_move.target_group] = best_cleanup_move.target_soft_impurity_after;

                local_summary.moves += 1;
                ++local_summary.simplify_moves;
                local_summary.hard_gain += best_cleanup.primary_gain;
                local_summary.soft_gain += best_cleanup.guide_gain;
                local_summary.delta_j += best_cleanup_move.delta_j;
                local_summary.component_delta += best_cleanup_move.delta_components;
            }

            if (local_summary.moves == 0) {
                if (summary != nullptr) {
                    *summary = local_summary;
                }
                return seed;
            }

            AtomizedCandidate refined = candidate_from_assignment(
                seed.feature,
                atoms,
                assign,
                groups,
                adjacency_bonus,
                adjacency_bonus_total);
            if (!refined.feasible) {
                return seed;
            }
            local_summary.improved =
                atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
            if (summary != nullptr) {
                *summary = local_summary;
            }
            return local_summary.improved ? refined : seed;
        }

        std::vector<int> assign = seed.assignment;
        std::vector<int> branch_rows((size_t)groups, 0);
        std::vector<double> branch_pos((size_t)groups, 0.0);
        std::vector<double> branch_neg((size_t)groups, 0.0);
        std::vector<double> branch_teacher_pos((size_t)groups, 0.0);
        std::vector<double> branch_teacher_neg((size_t)groups, 0.0);
        for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            if (group_idx < 0 || group_idx >= groups) {
                return seed;
            }
            const AtomizedAtom &atom = atoms[(size_t)atom_pos];
            branch_rows[(size_t)group_idx] += atom.row_count;
            branch_pos[(size_t)group_idx] += atom.pos_weight;
            branch_neg[(size_t)group_idx] += atom.neg_weight;
            branch_teacher_pos[(size_t)group_idx] += atom.teacher_pos_weight;
            branch_teacher_neg[(size_t)group_idx] += atom.teacher_neg_weight;
        }
        std::vector<double> branch_loss((size_t)groups, 0.0);
        std::vector<double> branch_hard_impurity((size_t)groups, 0.0);
        std::vector<double> branch_soft_impurity((size_t)groups, 0.0);
        for (int group_idx = 0; group_idx < groups; ++group_idx) {
            branch_loss[(size_t)group_idx] = split_leaf_loss(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
            branch_hard_impurity[(size_t)group_idx] = hard_label_impurity(
                branch_pos[(size_t)group_idx],
                branch_neg[(size_t)group_idx]);
            branch_soft_impurity[(size_t)group_idx] = hard_label_impurity(
                branch_teacher_pos[(size_t)group_idx],
                branch_teacher_neg[(size_t)group_idx]);
        }
        while (true) {
            check_timeout();
            std::vector<unsigned char> seen_source_groups((size_t)groups, 0);

            std::vector<double> branch_primary((size_t)groups, 0.0);
            std::vector<double> branch_secondary((size_t)groups, 0.0);
            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (hard_loss_mode) {
                    branch_primary[(size_t)group_idx] = split_leaf_loss(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                    branch_secondary[(size_t)group_idx] = hard_label_impurity(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                } else {
                    branch_primary[(size_t)group_idx] = hard_label_impurity(
                        branch_pos[(size_t)group_idx],
                        branch_neg[(size_t)group_idx]);
                    branch_secondary[(size_t)group_idx] = hard_label_impurity(
                        branch_teacher_pos[(size_t)group_idx],
                        branch_teacher_neg[(size_t)group_idx]);
                }
            }

            AtomizedRefinementMove best_move;
            int component_start = 0;
            while (component_start < atom_count) {
                const int source_group = assign[(size_t)component_start];
                int component_end = component_start + 1;
                while (component_end < atom_count && assign[(size_t)component_end] == source_group) {
                    ++component_end;
                }
                if (!seen_source_groups[(size_t)source_group]) {
                    seen_source_groups[(size_t)source_group] = 1U;
                    bump_histogram(local_summary.source_group_row_size_histogram, branch_rows[(size_t)source_group]);
                }
                bump_histogram(
                    local_summary.source_component_atom_size_histogram,
                    component_end - component_start);
                int component_rows_total = 0;
                for (int pos = component_start; pos < component_end; ++pos) {
                    component_rows_total += atoms[(size_t)pos].row_count;
                }
                bump_histogram(local_summary.source_component_row_size_histogram, component_rows_total);

                for (int target_group = 0; target_group < groups; ++target_group) {
                    if (target_group == source_group) {
                        continue;
                    }
                    const int source_rows_total = branch_rows[(size_t)source_group];
                    const int target_rows_total = branch_rows[(size_t)target_group];
                    const double source_pos_total = branch_pos[(size_t)source_group];
                    const double source_neg_total = branch_neg[(size_t)source_group];
                    const double target_pos_total = branch_pos[(size_t)target_group];
                    const double target_neg_total = branch_neg[(size_t)target_group];
                    const double source_teacher_pos_total = branch_teacher_pos[(size_t)source_group];
                    const double source_teacher_neg_total = branch_teacher_neg[(size_t)source_group];
                    const double target_teacher_pos_total = branch_teacher_pos[(size_t)target_group];
                    const double target_teacher_neg_total = branch_teacher_neg[(size_t)target_group];
                    const double source_primary_before = branch_primary[(size_t)source_group];
                    const double target_primary_before = branch_primary[(size_t)target_group];
                    const double source_secondary_before = branch_secondary[(size_t)source_group];
                    const double target_secondary_before = branch_secondary[(size_t)target_group];
                    auto scan_component_overlap = [&](int lo, int hi) {
                        if (lo > hi) {
                            return;
                        }
                            for (int start = lo; start <= hi; ++start) {
                                int move_rows = 0;
                                double move_pos = 0.0;
                                double move_neg = 0.0;
                                double move_teacher_pos = 0.0;
                                double move_teacher_neg = 0.0;
                                for (int end = start; end <= hi; ++end) {
                                    ++local_summary.candidate_total;
                                    const AtomizedAtom &atom = atoms[(size_t)end];
                                    move_rows += atom.row_count;
                                    move_pos += atom.pos_weight;
                                    move_neg += atom.neg_weight;
                                    move_teacher_pos += atom.teacher_pos_weight;
                                move_teacher_neg += atom.teacher_neg_weight;

                                    const int source_rows_after = source_rows_total - move_rows;
                                    if (source_rows_after < min_child_size_) {
                                        ++local_summary.candidate_source_size_rejects;
                                        break;
                                    }
                                    const int target_rows_after = target_rows_total + move_rows;
                                    if (target_rows_after < min_child_size_) {
                                        ++local_summary.candidate_target_size_rejects;
                                        continue;
                                    }

                                double source_primary_after = 0.0;
                                double target_primary_after = 0.0;
                                double source_secondary_after = 0.0;
                                double target_secondary_after = 0.0;
                                if (hard_loss_mode) {
                                    source_primary_after = split_leaf_loss(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_primary_after = split_leaf_loss(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                    source_secondary_after = hard_label_impurity(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_secondary_after = hard_label_impurity(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                } else {
                                    source_primary_after = hard_label_impurity(
                                        source_pos_total - move_pos,
                                        source_neg_total - move_neg);
                                    target_primary_after = hard_label_impurity(
                                        target_pos_total + move_pos,
                                        target_neg_total + move_neg);
                                    source_secondary_after = hard_label_impurity(
                                        source_teacher_pos_total - move_teacher_pos,
                                        source_teacher_neg_total - move_teacher_neg);
                                    target_secondary_after = hard_label_impurity(
                                        target_teacher_pos_total + move_teacher_pos,
                                        target_teacher_neg_total + move_teacher_neg);
                                }
                                const double delta_primary =
                                    (source_primary_before + target_primary_before) -
                                    (source_primary_after + target_primary_after);
                                const double delta_secondary =
                                    (source_secondary_before + target_secondary_before) -
                                    (source_secondary_after + target_secondary_after);
                                const int source_delta_components =
                                    (start == component_start && end == component_end - 1)
                                        ? -1
                                        : ((start == component_start || end == component_end - 1) ? 0 : 1);
                                const bool left_target = (start > 0 && assign[(size_t)(start - 1)] == target_group);
                                const bool right_target = (end + 1 < atom_count && assign[(size_t)(end + 1)] == target_group);
                                int target_delta_components = 0;
                                if (!left_target && !right_target) {
                                    target_delta_components = 1;
                                } else if (left_target && right_target) {
                                    target_delta_components = -1;
                                }

                                AtomizedRefinementMove move;
                                move.valid = true;
                                move.source_group = source_group;
                                move.target_group = target_group;
                                move.start = start;
                                move.end = end;
                                move.length = end - start + 1;
                                move.delta_components = source_delta_components + target_delta_components;
                                move.row_count = move_rows;
                                move.pos_weight = move_pos;
                                move.neg_weight = move_neg;
                                move.teacher_pos_weight = move_teacher_pos;
                                move.teacher_neg_weight = move_teacher_neg;
                                move.source_loss_after = hard_loss_mode ? source_primary_after : source_secondary_after;
                                move.target_loss_after = hard_loss_mode ? target_primary_after : target_secondary_after;
                                move.source_hard_impurity_after = hard_loss_mode ? source_secondary_after : source_primary_after;
                                    move.target_hard_impurity_after = hard_loss_mode ? target_secondary_after : target_primary_after;
                                    move.source_soft_impurity_after = 0.0;
                                    move.target_soft_impurity_after = 0.0;
                                    move.delta_hard = delta_primary;
                                    move.delta_soft = delta_secondary;
                                    move.delta_j = delta_primary - mu_node * (double)move.delta_components;
                                    ++local_summary.candidate_legal;
                                    if (!atomized_move_is_improving(move)) {
                                        ++local_summary.candidate_score_rejected;
                                        continue;
                                    }
                                    ++local_summary.candidate_descent_eligible;
                                    if (!best_move.valid || atomized_move_better(move, best_move)) {
                                        best_move = std::move(move);
                                    }
                                }
                            }
                        };

                    if (active_windows == nullptr || active_windows->empty()) {
                        scan_component_overlap(component_start, component_end - 1);
                    } else {
                        for (const auto &window : *active_windows) {
                            const int overlap_start = std::max(component_start, window.first);
                            const int overlap_end = std::min(component_end - 1, window.second);
                            if (overlap_start <= overlap_end) {
                                scan_component_overlap(overlap_start, overlap_end);
                            }
                        }
                    }
                }

                component_start = component_end;
            }

            if (!best_move.valid) {
                break;
            }

            for (int atom_pos = best_move.start; atom_pos <= best_move.end; ++atom_pos) {
                assign[(size_t)atom_pos] = best_move.target_group;
            }

            branch_rows[(size_t)best_move.source_group] -= best_move.row_count;
            branch_rows[(size_t)best_move.target_group] += best_move.row_count;
            branch_pos[(size_t)best_move.source_group] -= best_move.pos_weight;
            branch_pos[(size_t)best_move.target_group] += best_move.pos_weight;
            branch_neg[(size_t)best_move.source_group] -= best_move.neg_weight;
            branch_neg[(size_t)best_move.target_group] += best_move.neg_weight;
            branch_teacher_pos[(size_t)best_move.source_group] -= best_move.teacher_pos_weight;
            branch_teacher_pos[(size_t)best_move.target_group] += best_move.teacher_pos_weight;
            branch_teacher_neg[(size_t)best_move.source_group] -= best_move.teacher_neg_weight;
            branch_teacher_neg[(size_t)best_move.target_group] += best_move.teacher_neg_weight;
            branch_loss[(size_t)best_move.source_group] = best_move.source_loss_after;
            branch_loss[(size_t)best_move.target_group] = best_move.target_loss_after;
            branch_hard_impurity[(size_t)best_move.source_group] = best_move.source_hard_impurity_after;
            branch_hard_impurity[(size_t)best_move.target_group] = best_move.target_hard_impurity_after;
            branch_soft_impurity[(size_t)best_move.source_group] = best_move.source_soft_impurity_after;
            branch_soft_impurity[(size_t)best_move.target_group] = best_move.target_soft_impurity_after;

            local_summary.moves += 1;
            local_summary.hard_gain += best_move.delta_j + mu_node * (double)best_move.delta_components;
            local_summary.soft_gain += best_move.delta_soft;
            ++local_summary.descent_moves;
            local_summary.delta_j += best_move.delta_j;
            local_summary.component_delta += best_move.delta_components;
        }

        if (local_summary.moves == 0) {
            if (summary != nullptr) {
                *summary = local_summary;
            }
            return seed;
        }

        AtomizedCandidate refined = candidate_from_assignment(
            seed.feature,
            atoms,
            assign,
            groups,
            adjacency_bonus,
            adjacency_bonus_total);
        if (!refined.feasible) {
            return seed;
        }
        local_summary.improved = atomized_score_better_for_refinement(refined.score, seed.score, mu_node, mode);
        if (summary != nullptr) {
            *summary = local_summary;
        }
        return local_summary.improved ? refined : seed;
    }

    AtomizedCandidate solve_atomized_geometry_family(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedPrefixes &prefix,
        int feature,
        int groups,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        AtomizedCandidate out;
        const int m = (int)atoms.size();
        if (groups < 2 || groups > m) {
            return out;
        }

        const int stride = m + 1;
        auto at = [stride](int g, int t) -> size_t {
            return static_cast<size_t>(g) * static_cast<size_t>(stride) + static_cast<size_t>(t);
        };

        std::vector<AtomizedScore> dp(static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<int> parent(static_cast<size_t>(groups + 1) * static_cast<size_t>(stride), -1);
        dp[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= m; ++t) {
                AtomizedScore best;
                int best_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    const AtomizedScore &prev = dp[at(g - 1, p)];
                    if (!std::isfinite(prev.hard_impurity)) {
                        continue;
                    }
                    const int row_count = prefix.rows[(size_t)t] - prefix.rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }
                    AtomizedScore cand = prev;
                    if (binary_mode_) {
                        const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                        const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                        const double seg_teacher_pos = prefix.teacher_pos[(size_t)t] - prefix.teacher_pos[(size_t)p];
                        const double seg_teacher_neg = prefix.teacher_neg[(size_t)t] - prefix.teacher_neg[(size_t)p];
                        cand.hard_loss += split_leaf_loss(seg_pos, seg_neg);
                        cand.soft_loss += split_leaf_loss(seg_teacher_pos, seg_teacher_neg);
                        cand.hard_impurity += hard_label_impurity(seg_pos, seg_neg);
                        cand.soft_impurity += hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    } else {
                        cand.hard_loss += split_leaf_loss_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.soft_loss += split_leaf_loss_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.hard_impurity += hard_label_impurity_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        cand.soft_impurity += hard_label_impurity_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                    }
                    if (p > 0) {
                        cand.boundary_penalty -= contiguous_boundary_bonus(
                            feature,
                            atoms[(size_t)(p - 1)],
                            atoms[(size_t)p]);
                    }
                    cand.components += 1;
                    if (best_p < 0 || atomized_score_better(cand, best, mode)) {
                        best = cand;
                        best_p = p;
                    }
                }
                if (best_p >= 0) {
                    dp[at(g, t)] = best;
                    parent[at(g, t)] = best_p;
                }
            }
        }

        if (!std::isfinite(atomized_primary_objective(dp[at(groups, m)], mode))) {
            return out;
        }

        std::vector<int> assignment((size_t)m, -1);
        int t = m;
        int g = groups;
        int group_idx = groups - 1;
        while (g > 0) {
            const int p = parent[at(g, t)];
            if (p < 0) {
                return AtomizedCandidate{};
            }
            for (int pos = p; pos < t; ++pos) {
                assignment[(size_t)pos] = group_idx;
            }
            t = p;
            --g;
            --group_idx;
        }

        out.feasible = true;
        out.score = dp[at(groups, m)];
        out.groups = groups;
        out.assignment = std::move(assignment);
        return out;
    }

    AtomizedCandidatePair solve_atomized_geometry_family_pair(
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedPrefixes &prefix,
        int feature,
        int groups
    ) const {
        AtomizedCandidatePair out;
        const int m = (int)atoms.size();
        if (groups < 2 || groups > m) {
            return out;
        }

        const int stride = m + 1;
        auto at = [stride](int g, int t) -> size_t {
            return static_cast<size_t>(g) * static_cast<size_t>(stride) + static_cast<size_t>(t);
        };

        std::vector<AtomizedScore> dp_impurity(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<AtomizedScore> dp_hardloss(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride));
        std::vector<int> parent_impurity(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride),
            -1);
        std::vector<int> parent_hardloss(
            static_cast<size_t>(groups + 1) * static_cast<size_t>(stride),
            -1);
        dp_impurity[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};
        dp_hardloss[at(0, 0)] = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

        for (int g = 1; g <= groups; ++g) {
            for (int t = g; t <= m; ++t) {
                AtomizedScore best_impurity;
                AtomizedScore best_hardloss;
                int best_impurity_p = -1;
                int best_hardloss_p = -1;
                for (int p = g - 1; p <= t - 1; ++p) {
                    const AtomizedScore &prev_impurity = dp_impurity[at(g - 1, p)];
                    const AtomizedScore &prev_hardloss = dp_hardloss[at(g - 1, p)];
                    const bool impurity_prev_feasible =
                        std::isfinite(atomized_primary_objective(
                            prev_impurity,
                            AtomizedObjectiveMode::kImpurity));
                    const bool hardloss_prev_feasible =
                        std::isfinite(atomized_primary_objective(
                            prev_hardloss,
                            AtomizedObjectiveMode::kHardLoss));
                    if (!impurity_prev_feasible && !hardloss_prev_feasible) {
                        continue;
                    }

                    const int row_count = prefix.rows[(size_t)t] - prefix.rows[(size_t)p];
                    if (row_count < min_child_size_) {
                        continue;
                    }

                    double seg_hard_loss = 0.0;
                    double seg_soft_loss = 0.0;
                    double seg_hard_impurity = 0.0;
                    double seg_soft_impurity = 0.0;
                    if (binary_mode_) {
                        const double seg_pos = prefix.pos[(size_t)t] - prefix.pos[(size_t)p];
                        const double seg_neg = prefix.neg[(size_t)t] - prefix.neg[(size_t)p];
                        const double seg_teacher_pos =
                            prefix.teacher_pos[(size_t)t] - prefix.teacher_pos[(size_t)p];
                        const double seg_teacher_neg =
                            prefix.teacher_neg[(size_t)t] - prefix.teacher_neg[(size_t)p];
                        seg_hard_loss = split_leaf_loss(seg_pos, seg_neg);
                        seg_soft_loss = split_leaf_loss(seg_teacher_pos, seg_teacher_neg);
                        seg_hard_impurity = hard_label_impurity(seg_pos, seg_neg);
                        seg_soft_impurity = hard_label_impurity(seg_teacher_pos, seg_teacher_neg);
                    } else {
                        seg_hard_loss = split_leaf_loss_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_soft_loss = split_leaf_loss_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_hard_impurity = hard_label_impurity_prefix_segment(
                            prefix.class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                        seg_soft_impurity = hard_label_impurity_prefix_segment(
                            prefix.teacher_class_weight_prefix,
                            n_classes_,
                            p,
                            t);
                    }
                    const double seg_boundary_penalty =
                        (p > 0)
                            ? -contiguous_boundary_bonus(
                                  feature,
                                  atoms[(size_t)(p - 1)],
                                  atoms[(size_t)p])
                            : 0.0;

                    auto extend = [&](const AtomizedScore &prev) {
                        AtomizedScore cand = prev;
                        cand.hard_loss += seg_hard_loss;
                        cand.soft_loss += seg_soft_loss;
                        cand.hard_impurity += seg_hard_impurity;
                        cand.soft_impurity += seg_soft_impurity;
                        cand.boundary_penalty += seg_boundary_penalty;
                        cand.components += 1;
                        return cand;
                    };

                    if (impurity_prev_feasible) {
                        AtomizedScore cand = extend(prev_impurity);
                        if (best_impurity_p < 0 ||
                            atomized_score_better(
                                cand,
                                best_impurity,
                                AtomizedObjectiveMode::kImpurity)) {
                            best_impurity = cand;
                            best_impurity_p = p;
                        }
                    }
                    if (hardloss_prev_feasible) {
                        AtomizedScore cand = extend(prev_hardloss);
                        if (best_hardloss_p < 0 ||
                            atomized_score_better(
                                cand,
                                best_hardloss,
                                AtomizedObjectiveMode::kHardLoss)) {
                            best_hardloss = cand;
                            best_hardloss_p = p;
                        }
                    }
                }
                if (best_impurity_p >= 0) {
                    dp_impurity[at(g, t)] = best_impurity;
                    parent_impurity[at(g, t)] = best_impurity_p;
                }
                if (best_hardloss_p >= 0) {
                    dp_hardloss[at(g, t)] = best_hardloss;
                    parent_hardloss[at(g, t)] = best_hardloss_p;
                }
            }
        }

        auto build_candidate = [&](const std::vector<AtomizedScore> &dp,
                                   const std::vector<int> &parent,
                                   AtomizedObjectiveMode mode) {
            AtomizedCandidate candidate;
            if (!std::isfinite(atomized_primary_objective(dp[at(groups, m)], mode))) {
                return candidate;
            }
            std::vector<int> assignment((size_t)m, -1);
            int t = m;
            int g = groups;
            int group_idx = groups - 1;
            while (g > 0) {
                const int p = parent[at(g, t)];
                if (p < 0) {
                    return AtomizedCandidate{};
                }
                for (int pos = p; pos < t; ++pos) {
                    assignment[(size_t)pos] = group_idx;
                }
                t = p;
                --g;
                --group_idx;
            }
            candidate.feasible = true;
            candidate.score = dp[at(groups, m)];
            candidate.groups = groups;
            candidate.assignment = std::move(assignment);
            candidate.feature = feature;
            return candidate;
        };

        out.impurity = build_candidate(
            dp_impurity,
            parent_impurity,
            AtomizedObjectiveMode::kImpurity);
        out.misclassification = build_candidate(
            dp_hardloss,
            parent_hardloss,
            AtomizedObjectiveMode::kHardLoss);
        return out;
    }

    void record_refinement_summary(const AtomizedRefinementSummary &summary) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        auto accumulate_histogram = [](const std::vector<long long> &src, std::vector<long long> &dst) {
            if (src.size() > dst.size()) {
                dst.resize(src.size(), 0);
            }
            for (size_t i = 0; i < src.size(); ++i) {
                dst[i] += src[i];
            }
        };
        ++telemetry.debr_refine_calls;
        telemetry.debr_total_moves += summary.moves;
        telemetry.debr_bridge_policy_calls += summary.bridge_policy_calls;
        telemetry.debr_refine_windowed_calls += summary.refine_windowed_calls;
        telemetry.debr_refine_unwindowed_calls += summary.refine_unwindowed_calls;
        telemetry.debr_refine_overlap_segments += summary.refine_overlap_segments;
        telemetry.debr_refine_calls_with_overlap += summary.refine_calls_with_overlap;
        telemetry.debr_refine_calls_without_overlap += summary.refine_calls_without_overlap;
        telemetry.debr_candidate_total += summary.candidate_total;
        telemetry.debr_candidate_legal += summary.candidate_legal;
        telemetry.debr_candidate_source_size_rejects += summary.candidate_source_size_rejects;
        telemetry.debr_candidate_target_size_rejects += summary.candidate_target_size_rejects;
        telemetry.debr_candidate_descent_eligible += summary.candidate_descent_eligible;
        telemetry.debr_candidate_descent_rejected += summary.candidate_descent_rejected;
        telemetry.debr_candidate_bridge_eligible += summary.candidate_bridge_eligible;
        telemetry.debr_candidate_bridge_window_blocked += summary.candidate_bridge_window_blocked;
        telemetry.debr_candidate_bridge_used_blocked += summary.candidate_bridge_used_blocked;
        telemetry.debr_candidate_bridge_guide_rejected += summary.candidate_bridge_guide_rejected;
        telemetry.debr_candidate_cleanup_eligible += summary.candidate_cleanup_eligible;
        telemetry.debr_candidate_cleanup_primary_rejected += summary.candidate_cleanup_primary_rejected;
        telemetry.debr_candidate_cleanup_complexity_rejected += summary.candidate_cleanup_complexity_rejected;
        telemetry.debr_candidate_score_rejected += summary.candidate_score_rejected;
        telemetry.debr_descent_moves += summary.descent_moves;
        telemetry.debr_bridge_moves += summary.bridge_moves;
        telemetry.debr_simplify_moves += summary.simplify_moves;
        accumulate_histogram(summary.source_group_row_size_histogram, telemetry.debr_source_group_row_size_histogram);
        accumulate_histogram(
            summary.source_component_atom_size_histogram,
            telemetry.debr_source_component_atom_size_histogram);
        accumulate_histogram(
            summary.source_component_row_size_histogram,
            telemetry.debr_source_component_row_size_histogram);
        telemetry.debr_total_hard_gain += summary.hard_gain;
        telemetry.debr_total_soft_gain += summary.soft_gain;
        telemetry.debr_total_delta_j += summary.delta_j;
        telemetry.debr_total_component_delta += summary.component_delta;
        if (summary.improved) {
            ++telemetry.debr_refine_improved;
        }
    }

    AtomizedCoarseCandidate prepare_folded_family_coarse(
        int feature,
        const PreparedFeatureAtomized &prepared,
        int groups,
        double mu_node,
        AtomizedObjectiveMode mode
    ) const {
        AtomizedCoarseCandidate coarse;
        if (groups < 2 || groups > prepared.q_effective) {
            return coarse;
        }
        if (!prepared.has_block_compression) {
            AtomizedCandidate coarse_seed = solve_atomized_geometry_family(
                prepared.atoms,
                prepared.atom_prefix,
                feature,
                groups,
                mode);
            coarse_seed.feature = feature;
            if (coarse_seed.feasible) {
                coarse.geometry_seed_candidate = coarse_seed;
                coarse.initial_block_assignment = coarse_seed.assignment;
                coarse.refined_block_assignment = coarse_seed.assignment;
                coarse.candidate = coarse_seed;
            }
            return coarse;
        }
        if (groups > (int)prepared.block_atoms.size()) {
            return coarse;
        }

        AtomizedCandidate block_seed = solve_atomized_geometry_family(
            prepared.block_atoms,
            prepared.block_prefix,
            -1,
            groups,
            mode);
        if (!block_seed.feasible) {
            return coarse;
        }

        coarse.geometry_seed_candidate = lift_block_candidate_to_atoms(
            feature,
            prepared.blocks,
            prepared.atoms,
            block_seed,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total);
        if (!coarse.geometry_seed_candidate.feasible) {
            return AtomizedCoarseCandidate{};
        }

        coarse.initial_block_assignment = block_seed.assignment;

        AtomizedRefinementSummary block_summary;
        AtomizedCandidate refined_block = refine_atomized_candidate_debr(
            prepared.block_atoms,
            block_seed,
            mu_node,
            nullptr,
            &block_summary,
            nullptr,
            0.0,
            mode);
        record_refinement_summary(block_summary);

        coarse.refined_block_assignment = refined_block.assignment;

        coarse.block_candidate = lift_block_candidate_to_atoms(
            feature,
            prepared.blocks,
            prepared.atoms,
            refined_block,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total);
        coarse.candidate = coarse.geometry_seed_candidate;
        if (coarse.block_candidate.feasible &&
            atomized_candidate_better_for_objective(
                coarse.block_candidate,
                coarse.candidate,
                feature,
                feature,
                mode)) {
            coarse.candidate = coarse.block_candidate;
        }
        return coarse;
    }

    bool prepare_feature_atomized_local(
        const std::vector<int> &indices,
        int feature,
        double mu_node,
        PreparedFeatureAtomized &prepared
    ) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        prepared = PreparedFeatureAtomized{};
        if (!build_ordered_bins(indices, feature, prepared.bins)) {
            return false;
        }
        if (!build_atomized_atoms(prepared.bins, feature, prepared.atoms)) {
            return false;
        }
        ++telemetry.atomized_features_prepared;
        auto bump_count_histogram = [](std::vector<long long> &hist, size_t bucket) {
            const size_t idx = bucket;
            if (hist.size() <= idx) {
                hist.resize(idx + 1U, 0);
            }
            ++hist[idx];
        };
        bump_count_histogram(telemetry.atomized_feature_atom_count_histogram, prepared.atoms.size());

        const int q_support = std::max(0, (int)indices.size() / std::max(1, min_child_size_));
        prepared.q_effective = std::min(max_groups_for_bins((int)prepared.atoms.size()), q_support);
        bump_count_histogram(
            telemetry.atomized_feature_q_effective_histogram,
            static_cast<size_t>(std::max(0, prepared.q_effective)));
        if (prepared.q_effective < 2) {
            return false;
        }
        prepared.atom_prefix = build_atomized_prefixes(prepared.atoms);
        prepared.atom_adjacency_bonus.clear();
        prepared.atom_adjacency_bonus_total = 0.0;
        if (teacher_available_ && prepared.atoms.size() > 1U) {
            prepared.atom_adjacency_bonus.resize(prepared.atoms.size() - 1U, 0.0);
            for (size_t atom_pos = 0; atom_pos + 1U < prepared.atoms.size(); ++atom_pos) {
                const double bonus = contiguous_boundary_bonus(
                    feature,
                    prepared.atoms[atom_pos],
                    prepared.atoms[atom_pos + 1U]);
                prepared.atom_adjacency_bonus[atom_pos] = bonus;
                prepared.atom_adjacency_bonus_total += bonus;
            }
        }

        prepared.has_block_compression = has_atomized_block_compression(prepared.atoms);
        if (prepared.has_block_compression) {
            build_atomized_blocks_and_atoms(prepared.atoms, prepared.blocks, prepared.block_atoms);
            prepared.has_block_compression = prepared.block_atoms.size() < prepared.atoms.size();
        }
        bump_count_histogram(
            telemetry.atomized_feature_block_atom_count_histogram,
            prepared.has_block_compression && !prepared.block_atoms.empty()
                ? prepared.block_atoms.size()
                : prepared.atoms.size());
        if (prepared.has_block_compression) {
            prepared.block_prefix = build_atomized_prefixes(prepared.block_atoms);
        } else {
            prepared.blocks.clear();
            prepared.block_atoms.clear();
            prepared.block_prefix = AtomizedPrefixes{};
        }
        prepared.coarse_by_groups.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});
        prepared.coarse_by_groups_hardloss.assign((size_t)prepared.q_effective + 1, AtomizedCoarseCandidate{});

        bool any_feasible = false;
        for (int groups = 2; groups <= prepared.q_effective; ++groups) {
            AtomizedCoarseCandidate coarse = prepare_folded_family_coarse(
                feature,
                prepared,
                groups,
                mu_node,
                AtomizedObjectiveMode::kImpurity);
            AtomizedCoarseCandidate hardloss_coarse = prepare_folded_family_coarse(
                feature,
                prepared,
                groups,
                mu_node,
                AtomizedObjectiveMode::kHardLoss);
            if (!coarse.candidate.feasible && !hardloss_coarse.candidate.feasible) {
                continue;
            }
            prepared.coarse_by_groups[(size_t)groups] = std::move(coarse);
            prepared.coarse_by_groups_hardloss[(size_t)groups] = std::move(hardloss_coarse);
            ++telemetry.atomized_coarse_candidates;
            any_feasible = true;
        }

        prepared.valid = any_feasible;
        return any_feasible;
    }

    AtomizedCandidate nominate_folded_family_candidate(
        int feature,
        const PreparedFeatureAtomized &prepared,
        int groups,
        double mu_node,
        const AtomizedCoarseCandidate &coarse,
        AtomizedObjectiveMode mode,
        const AtomizedCandidate *raw_seed_override = nullptr
    ) const {
        AtomizedCandidate best;
        if (coarse.geometry_seed_candidate.feasible) {
            best = coarse.geometry_seed_candidate;
        }
        if (coarse.block_candidate.feasible &&
            atomized_candidate_better_for_objective(
                coarse.block_candidate,
                best,
                feature,
                feature,
                mode)) {
            best = coarse.block_candidate;
        }

        AtomizedCandidate raw_seed;
        if (raw_seed_override != nullptr && raw_seed_override->feasible) {
            raw_seed = *raw_seed_override;
            raw_seed.feature = feature;
        } else if (!prepared.has_block_compression && coarse.geometry_seed_candidate.feasible) {
            raw_seed = coarse.geometry_seed_candidate;
        } else {
            raw_seed = solve_atomized_geometry_family(
                prepared.atoms,
                prepared.atom_prefix,
                feature,
                groups,
                mode);
            raw_seed.feature = feature;
        }
        if (raw_seed.feasible) {
            best = raw_seed;
        }
        if (coarse.block_candidate.feasible &&
            atomized_candidate_better_for_objective(
                coarse.block_candidate,
                best,
                feature,
                feature,
                mode)) {
            best = coarse.block_candidate;
        }
        if (!raw_seed.feasible) {
            return best;
        }
        if (!prepared.has_block_compression || prepared.block_atoms.empty()) {
            AtomizedRefinementSummary raw_summary;
            const AtomizedCandidate refined = refine_atomized_candidate_debr(
                prepared.atoms,
                raw_seed,
                mu_node,
                nullptr,
                &raw_summary,
                &prepared.atom_adjacency_bonus,
                prepared.atom_adjacency_bonus_total,
                mode);
            record_refinement_summary(raw_summary);

            AtomizedCandidate geometry_candidate = raw_seed;
            if (refined.feasible &&
                atomized_score_better_for_refinement(refined.score, raw_seed.score, mu_node, mode)) {
                geometry_candidate = refined;
            }
            if (atomized_candidate_better_for_objective(
                    geometry_candidate,
                    best,
                    feature,
                    feature,
                    mode)) {
                best = geometry_candidate;
            }
            return best;
        }

        std::vector<int> projected_block_assignment;
        std::vector<unsigned char> mixed_block;
        if (!project_atom_assignment_to_blocks(
                prepared.blocks,
                raw_seed.assignment,
                projected_block_assignment,
                mixed_block)) {
            return best;
        }

        std::vector<int> refined_block_assignment = projected_block_assignment;
        if (groups <= (int)prepared.block_atoms.size()) {
            const bool can_reuse_initial =
                !coarse.initial_block_assignment.empty() &&
                coarse.initial_block_assignment.size() == projected_block_assignment.size() &&
                coarse.refined_block_assignment.size() == projected_block_assignment.size() &&
                projected_block_assignment == coarse.initial_block_assignment;
            const bool can_reuse_refined =
                !coarse.refined_block_assignment.empty() &&
                coarse.refined_block_assignment.size() == projected_block_assignment.size() &&
                projected_block_assignment == coarse.refined_block_assignment;
            if (can_reuse_initial || can_reuse_refined) {
                refined_block_assignment = coarse.refined_block_assignment;
            } else {
                AtomizedCandidate projected_block_seed = candidate_from_assignment(
                    feature,
                    prepared.block_atoms,
                    projected_block_assignment,
                    groups);
                if (projected_block_seed.feasible) {
                    AtomizedRefinementSummary block_summary;
                    const AtomizedCandidate refined_block = refine_atomized_candidate_debr(
                        prepared.block_atoms,
                        projected_block_seed,
                        mu_node,
                        nullptr,
                        &block_summary,
                        nullptr,
                        0.0,
                        mode);
                    record_refinement_summary(block_summary);
                    refined_block_assignment = refined_block.assignment;
                }
            }
        }

        std::vector<std::pair<int, int>> block_windows =
            build_active_block_windows(projected_block_assignment, refined_block_assignment, &mixed_block);
        if (block_windows.empty()) {
            return best;
        }

        const std::vector<std::pair<int, int>> atom_windows =
            block_windows_to_atom_windows(prepared.blocks, block_windows);
        AtomizedRefinementSummary raw_summary;
        const AtomizedCandidate refined = refine_atomized_candidate_debr(
            prepared.atoms,
            raw_seed,
            mu_node,
            &atom_windows,
            &raw_summary,
            &prepared.atom_adjacency_bonus,
            prepared.atom_adjacency_bonus_total,
            mode);
        record_refinement_summary(raw_summary);

        AtomizedCandidate geometry_candidate = raw_seed;
        if (refined.feasible &&
            atomized_score_better_for_refinement(refined.score, raw_seed.score, mu_node, mode)) {
            geometry_candidate = refined;
        }
        if (atomized_candidate_better_for_objective(
                geometry_candidate,
                best,
                feature,
                feature,
                mode)) {
            best = geometry_candidate;
        }
        return best;
    }

    std::vector<AtomizedCandidate> select_atomized_candidates_for_arity(
        int feature,
        const PreparedFeatureAtomized &prepared,
        int groups,
        double mu_node
    ) const {
        std::vector<AtomizedCandidate> selected;
        if (groups < 2 ||
            groups >= (int)prepared.coarse_by_groups.size() ||
            groups >= (int)prepared.coarse_by_groups_hardloss.size()) {
            return selected;
        }

        const AtomizedCoarseCandidate &impurity_coarse = prepared.coarse_by_groups[(size_t)groups];
        const AtomizedCoarseCandidate &hardloss_coarse = prepared.coarse_by_groups_hardloss[(size_t)groups];
        const AtomizedCandidatePair raw_seed_pair = solve_atomized_geometry_family_pair(
            prepared.atoms,
            prepared.atom_prefix,
            feature,
            groups);
        AtomizedCandidate impurity = nominate_folded_family_candidate(
            feature,
            prepared,
            groups,
            mu_node,
            impurity_coarse,
            AtomizedObjectiveMode::kImpurity,
            &raw_seed_pair.impurity);
        AtomizedCandidate misclassification = nominate_folded_family_candidate(
            feature,
            prepared,
            groups,
            mu_node,
            hardloss_coarse,
            AtomizedObjectiveMode::kHardLoss,
            &raw_seed_pair.misclassification);

        if (impurity.feasible && misclassification.feasible) {
            selected = select_family_nominees(std::move(impurity), std::move(misclassification));
            return selected;
        }
        if (impurity.feasible) {
            ++const_cast<Solver *>(this)->debr_final_geo_wins_;
            selected.push_back(std::move(impurity));
        } else if (misclassification.feasible) {
            ++const_cast<Solver *>(this)->debr_final_block_wins_;
            selected.push_back(std::move(misclassification));
        }
        return selected;
    }

    static std::vector<std::pair<int, int>> atom_positions_to_spans(
        const OrderedBins &bins,
        const std::vector<int> &group_positions
    ) {
        std::vector<std::pair<int, int>> spans;
        if (group_positions.empty()) {
            return spans;
        }
        int span_lo = bins.values[(size_t)group_positions.front()];
        int span_hi = span_lo;
        int prev_pos = group_positions.front();
        for (size_t i = 1; i < group_positions.size(); ++i) {
            const int atom_pos = group_positions[i];
            if (atom_pos == prev_pos + 1) {
                span_hi = bins.values[(size_t)atom_pos];
            } else {
                spans.push_back({span_lo, span_hi});
                span_lo = bins.values[(size_t)atom_pos];
                span_hi = span_lo;
            }
            prev_pos = atom_pos;
        }
        spans.push_back({span_lo, span_hi});
        return spans;
    }

    std::shared_ptr<Node> build_internal_node_from_group_spans(
        int feature,
        const std::vector<std::vector<std::pair<int, int>>> &group_spans,
        const std::vector<std::shared_ptr<Node>> &group_nodes,
        int fallback_prediction,
        int n_samples
    ) const {
        if (group_spans.empty() || group_spans.size() != group_nodes.size()) {
            return nullptr;
        }

        auto internal = std::make_shared<Node>();
        internal->is_leaf = false;
        internal->feature = feature;
        internal->fallback_prediction = fallback_prediction;
        internal->n_samples = n_samples;
        internal->group_count = (int)group_nodes.size();

        int largest_child_size = -1;
        for (size_t i = 0; i < group_nodes.size(); ++i) {
            if (!group_nodes[i] || group_spans[i].empty()) {
                return nullptr;
            }
            internal->group_bin_spans.push_back(group_spans[i]);
            internal->group_nodes.push_back(group_nodes[i]);
            if (group_nodes[i]->n_samples > largest_child_size) {
                largest_child_size = group_nodes[i]->n_samples;
                internal->fallback_bin = group_spans[i].front().first;
            }
        }
        if (internal->fallback_bin < 0) {
            internal->fallback_bin = group_spans.front().front().first;
        }
        return internal;
    }

    static bool subtree_is_constant_prediction(const std::shared_ptr<Node> &node, int &prediction_out) {
        if (!node) {
            return false;
        }
        if (node->is_leaf) {
            prediction_out = node->prediction;
            return true;
        }
        int shared_prediction = -1;
        bool first = true;
        for (const auto &child : node->group_nodes) {
            int child_prediction = -1;
            if (!subtree_is_constant_prediction(child, child_prediction)) {
                return false;
            }
            if (first) {
                shared_prediction = child_prediction;
                first = false;
            } else if (child_prediction != shared_prediction) {
                return false;
            }
        }
        if (first) {
            return false;
        }
        prediction_out = shared_prediction;
        return true;
    }

    std::string encode_signature_code(int row) const {
        std::string code;
        code.resize((size_t)n_features_ * sizeof(int));
        std::memcpy(code.data(), &x_flat_[(size_t)row * (size_t)n_features_], (size_t)n_features_ * sizeof(int));
        return code;
    }

    CanonicalSignatureState build_canonical_signature_state(const std::vector<int> &indices) const {
        CanonicalSignatureState state;
        state.blocks.reserve(indices.size());
        std::unordered_map<std::string, size_t> block_index;
        block_index.reserve(indices.size());

        for (int row : indices) {
            const std::string code = encode_signature_code(row);
            auto [it, inserted] = block_index.emplace(code, state.blocks.size());
            if (inserted) {
                CanonicalSignatureBlock block;
                block.code_key = code;
                if (!binary_mode_) {
                    block.class_weight.assign((size_t)n_classes_, 0.0);
                    block.teacher_class_weight.assign((size_t)n_classes_, 0.0);
                }
                state.blocks.push_back(std::move(block));
            }
            CanonicalSignatureBlock &block = state.blocks[it->second];
            const double w = sample_weight_[(size_t)row];
            const int label = y_[(size_t)row];
            ++block.row_count;
            if (binary_mode_) {
                if (label == 1) {
                    block.pos_weight += w;
                } else {
                    block.neg_weight += w;
                }
            } else {
                block.class_weight[(size_t)label] += w;
            }
        }

        std::sort(
            state.blocks.begin(),
            state.blocks.end(),
            [](const CanonicalSignatureBlock &lhs, const CanonicalSignatureBlock &rhs) {
                return lhs.code_key < rhs.code_key;
            });

        auto block_minority = [&](const CanonicalSignatureBlock &block) {
            if (binary_mode_) {
                return std::min(block.pos_weight, block.neg_weight);
            }
            double total = 0.0;
            double best = 0.0;
            for (double value : block.class_weight) {
                total += value;
                best = std::max(best, value);
            }
            return (total > best) ? (total - best) : 0.0;
        };

        state.signature_bound = regularization_;
        double total_pos = 0.0;
        double total_neg = 0.0;
        std::vector<double> total_class_weight((size_t)n_classes_, 0.0);
        for (const auto &block : state.blocks) {
            state.signature_bound += block_minority(block);
            if (binary_mode_) {
                total_pos += block.pos_weight;
                total_neg += block.neg_weight;
            } else {
                for (int cls = 0; cls < n_classes_; ++cls) {
                    total_class_weight[(size_t)cls] += block.class_weight[(size_t)cls];
                }
            }
        }

        if (binary_mode_) {
            state.leaf_objective = regularization_ + std::min(total_pos, total_neg);
        } else {
            double total = 0.0;
            double best = 0.0;
            for (double value : total_class_weight) {
                total += value;
                best = std::max(best, value);
            }
            state.leaf_objective = regularization_ + ((total > best) ? (total - best) : 0.0);
        }

        return state;
    }

    std::string canonical_state_key(
        const CanonicalSignatureState &state,
        int depth_remaining
    ) const {
        std::string key;
        size_t total = sizeof(int64_t) * 2U;
        for (const auto &block : state.blocks) {
            total += sizeof(int64_t) + block.code_key.size();
            total += sizeof(int64_t);
            total += sizeof(double) * 2U;
            if (!binary_mode_) {
                total += sizeof(int64_t) + block.class_weight.size() * sizeof(double);
            }
        }
        key.reserve(total);
        auto append_i64 = [&](int64_t value) {
            key.append(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        auto append_u64 = [&](uint64_t value) {
            key.append(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        auto append_double = [&](double value) {
            static_assert(sizeof(double) == sizeof(uint64_t));
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            append_u64(bits);
        };
        append_i64((int64_t)depth_remaining);
        append_i64((int64_t)state.blocks.size());
        for (const auto &block : state.blocks) {
            append_u64((uint64_t)block.code_key.size());
            key.append(block.code_key.data(), block.code_key.size());
            append_i64((int64_t)block.row_count);
            append_double(block.pos_weight);
            append_double(block.neg_weight);
            if (!binary_mode_) {
                append_u64((uint64_t)block.class_weight.size());
                for (double value : block.class_weight) {
                    append_double(value);
                }
            }
        }
        return key;
    }

    std::string serialized_indices_key(const std::vector<int> &indices) const {
        std::string key;
        key.reserve(sizeof(uint64_t) + indices.size() * 2U);
        auto append_varint = [&](uint64_t value) {
            while (value >= 0x80U) {
                key.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
                value >>= 7U;
            }
            key.push_back(static_cast<char>(value));
        };
        append_varint((uint64_t)indices.size());
        uint64_t previous = 0U;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            const uint64_t row = static_cast<uint64_t>(indices[idx]);
            append_varint(idx == 0U ? row : (row - previous));
            previous = row;
        }
        return key;
    }

    std::string serialized_indices_depth_key(
        const std::vector<int> &indices,
        int depth_remaining
    ) const {
        std::string key = serialized_indices_key(indices);
        const int64_t depth_value = depth_remaining;
        key.append(reinterpret_cast<const char *>(&depth_value), sizeof(depth_value));
        return key;
    }

    std::string serialized_block_key_set(std::vector<std::string> keys) const {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::string out;
        size_t total = sizeof(int64_t);
        for (const auto &key : keys) {
            total += sizeof(int64_t) + key.size();
        }
        out.reserve(total);
        auto append_i64 = [&](int64_t value) {
            out.append(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        append_i64((int64_t)keys.size());
        for (const auto &key : keys) {
            append_i64((int64_t)key.size());
            out.append(key.data(), key.size());
        }
        return out;
    }

    double signature_bound_for_indices(const std::vector<int> &indices, int depth_remaining) const {
        ++profiling_signature_bound_calls_;
        ScopedTimer signature_timer(profiling_signature_bound_sec_);
        if (depth_remaining <= 0 || indices.empty()) {
            return compute_subproblem_stats(indices).leaf_objective;
        }
        return get_canonical_signature_summary(indices).signature_bound;
    }

    struct PathMasterSolution {
        double objective = kInfinity;
        std::unordered_map<std::string, double> duals_by_block;
    };

    PathMasterSolution solve_path_master_lp(
        const CanonicalSignatureState &root_state,
        const std::vector<PathColumn> &columns
    ) const {
        ++profiling_lp_solve_calls_;
        ScopedTimer lp_timer(profiling_lp_solve_sec_);
        PathMasterSolution out;
        if (root_state.blocks.empty() || columns.empty()) {
            return out;
        }

        const int row_count = (int)root_state.blocks.size();
        const int col_count = (int)columns.size();

        std::unordered_map<std::string, int> row_index;
        row_index.reserve(root_state.blocks.size() * 2U);
        for (int row = 0; row < row_count; ++row) {
            row_index.emplace(root_state.blocks[(size_t)row].code_key, row + 1);
        }

        glp_prob *lp = glp_create_prob();
        if (lp == nullptr) {
            return out;
        }

        glp_set_obj_dir(lp, GLP_MIN);
        glp_add_rows(lp, row_count);
        for (int row = 0; row < row_count; ++row) {
            glp_set_row_bnds(lp, row + 1, GLP_FX, 1.0, 1.0);
        }
        glp_add_cols(lp, col_count);
        for (int col = 0; col < col_count; ++col) {
            glp_set_col_bnds(lp, col + 1, GLP_LO, 0.0, 0.0);
            glp_set_obj_coef(lp, col + 1, columns[(size_t)col].cost);
        }

        std::vector<int> ia(1, 0);
        std::vector<int> ja(1, 0);
        std::vector<double> ar(1, 0.0);
        ia.reserve(1U + columns.size() * 8U);
        ja.reserve(1U + columns.size() * 8U);
        ar.reserve(1U + columns.size() * 8U);
        for (int col = 0; col < col_count; ++col) {
            std::unordered_set<std::string> unique_coverage;
            unique_coverage.reserve(columns[(size_t)col].covered_block_keys.size() * 2U);
            for (const auto &block_key : columns[(size_t)col].covered_block_keys) {
                unique_coverage.insert(block_key);
            }
            for (const auto &block_key : unique_coverage) {
                auto row_it = row_index.find(block_key);
                if (row_it == row_index.end()) {
                    continue;
                }
                ia.push_back(row_it->second);
                ja.push_back(col + 1);
                ar.push_back(1.0);
            }
        }

        const int matrix_size = (int)ia.size() - 1;
        if (matrix_size <= 0) {
            glp_delete_prob(lp);
            return out;
        }
        glp_load_matrix(lp, matrix_size, ia.data(), ja.data(), ar.data());

        glp_smcp parm;
        glp_init_smcp(&parm);
        parm.msg_lev = GLP_MSG_OFF;
        parm.presolve = GLP_ON;
        parm.meth = GLP_PRIMAL;
        const int simplex_status = glp_simplex(lp, &parm);
        if (simplex_status != 0) {
            glp_delete_prob(lp);
            return out;
        }
        const int status = glp_get_status(lp);
        if (status != GLP_OPT && status != GLP_FEAS) {
            glp_delete_prob(lp);
            return out;
        }

        out.objective = glp_get_obj_val(lp);
        out.duals_by_block.reserve(root_state.blocks.size() * 2U);
        for (int row = 0; row < row_count; ++row) {
            out.duals_by_block.emplace(
                root_state.blocks[(size_t)row].code_key,
                glp_get_row_dual(lp, row + 1));
        }
        glp_delete_prob(lp);
        return out;
    }

    PathPricingResult price_terminal_region(
        const std::vector<int> &indices,
        int depth_remaining,
        const std::vector<std::string> &root_block_key_by_row,
        const std::unordered_map<std::string, double> &duals_by_block,
        std::unordered_map<std::string, PathPricingResult> &memo
    ) const {
        ++profiling_pricing_calls_;
        ScopedTimer pricing_timer(profiling_pricing_sec_);
        const auto summary = get_canonical_signature_summary(indices);
        const std::string memo_key = serialized_indices_depth_key(indices, depth_remaining);
        auto memo_it = memo.find(memo_key);
        if (memo_it != memo.end()) {
            return memo_it->second;
        }

        PathPricingResult best;
        best.key = memo_key;
        best.covered_block_keys.reserve(indices.size());
        double dual_mass = 0.0;
        std::unordered_set<std::string> unique_coverage;
        unique_coverage.reserve(indices.size() * 2U);
        for (int row : indices) {
            const std::string &block_key = root_block_key_by_row[(size_t)row];
            if (unique_coverage.insert(block_key).second) {
                best.covered_block_keys.push_back(block_key);
                auto dual_it = duals_by_block.find(block_key);
                if (dual_it != duals_by_block.end()) {
                    dual_mass += dual_it->second;
                }
            }
        }
        best.column_key = serialized_block_key_set(best.covered_block_keys);
        best.cost = summary.leaf_objective;
        best.reduced_cost = best.cost - dual_mass;

        if (depth_remaining <= 0 || summary.block_count <= 1U || (int)indices.size() < min_split_size_) {
            memo.emplace(memo_key, best);
            return best;
        }

        const double mu_node = effective_sample_unit(compute_subproblem_stats(indices));
        std::vector<CandidateEval> candidate_evals;
        collect_ranked_candidates(indices, mu_node, candidate_evals, false);
        for (const auto &eval : candidate_evals) {
            for (const auto &child_indices : eval.child_indices) {
                if (child_indices.empty()) {
                    continue;
                }
                PathPricingResult child = price_terminal_region(
                    child_indices,
                    depth_remaining - 1,
                    root_block_key_by_row,
                    duals_by_block,
                    memo);
                if (child.reduced_cost + kEpsUpdate < best.reduced_cost ||
                    (std::abs(child.reduced_cost - best.reduced_cost) <= kEpsUpdate &&
                     child.cost + kEpsUpdate < best.cost)) {
                    best = std::move(child);
                }
            }
        }

        memo.emplace(memo_key, best);
        return best;
    }

    double path_bound_for_indices(const std::vector<int> &indices, int depth_remaining) const {
        ++profiling_path_bound_calls_;
        ScopedTimer path_timer(profiling_path_bound_sec_);
        if (depth_remaining <= 0 || indices.empty()) {
            ++profiling_path_bound_skip_trivial_;
            return compute_subproblem_stats(indices).leaf_objective;
        }
        const auto summary = get_canonical_signature_summary(indices);
        if (summary.block_count > static_cast<size_t>(path_lp_max_blocks_)) {
            ++profiling_path_bound_skip_too_many_blocks_;
            return summary.signature_bound;
        }
        if (summary.block_count <= 1U || (int)indices.size() < min_split_size_) {
            ++profiling_path_bound_skip_small_state_;
            return summary.leaf_objective;
        }

        const CanonicalSignatureState state = build_canonical_signature_state(indices);

        std::vector<std::string> root_block_key_by_row((size_t)n_rows_);
        for (int row : indices) {
            root_block_key_by_row[(size_t)row] = encode_signature_code(row);
        }

        std::vector<PathColumn> columns;
        columns.reserve(8U);
        PathColumn root_column;
        std::vector<std::string> root_block_keys;
        root_block_keys.reserve(state.blocks.size());
        for (const auto &block : state.blocks) {
            root_block_keys.push_back(block.code_key);
        }
        root_column.key = serialized_block_key_set(root_block_keys);
        root_column.cost = summary.leaf_objective;
        root_column.covered_block_keys.reserve(state.blocks.size());
        for (const auto &block : state.blocks) {
            root_column.covered_block_keys.push_back(block.code_key);
        }
        columns.push_back(std::move(root_column));

        std::unordered_set<std::string> seen_columns;
        seen_columns.reserve(8U);
        seen_columns.insert(columns.back().key);

        const int max_iterations = std::max(16, (int)state.blocks.size() + 8);
        for (int iter = 0; iter < max_iterations; ++iter) {
            PathMasterSolution master = solve_path_master_lp(state, columns);
            if (!std::isfinite(master.objective)) {
                break;
            }

            std::unordered_map<std::string, PathPricingResult> memo;
            PathPricingResult priced = price_terminal_region(
                indices,
                depth_remaining,
                root_block_key_by_row,
                master.duals_by_block,
                memo);
            if (priced.reduced_cost >= -kEpsUpdate) {
                return master.objective;
            }
            if (!seen_columns.insert(priced.column_key).second) {
                break;
            }

            PathColumn column;
            column.key = priced.column_key;
            column.covered_block_keys = std::move(priced.covered_block_keys);
            column.cost = priced.cost;
            columns.push_back(std::move(column));
        }

        const PathMasterSolution master = solve_path_master_lp(state, columns);
        const double objective = std::isfinite(master.objective) ? master.objective : summary.leaf_objective;
        return objective;
    }

    double state_lower_bound_for_indices(
        const std::vector<int> &indices,
        int depth_remaining,
        bool allow_path
    ) const {
        if (depth_remaining <= 0 || indices.empty()) {
            ++profiling_path_bound_skip_trivial_;
            return compute_subproblem_stats(indices).leaf_objective;
        }
        const auto summary = get_canonical_signature_summary(indices);
        double sig_bound = summary.signature_bound;
        if (!allow_path) {
            ++profiling_path_bound_skip_disabled_;
            return sig_bound;
        }
        if (summary.block_count <= 1U || (int)indices.size() < min_split_size_) {
            ++profiling_path_bound_skip_small_state_;
            return sig_bound;
        }
        if (summary.block_count > static_cast<size_t>(path_lp_max_blocks_)) {
            ++profiling_path_bound_skip_too_many_blocks_;
            return sig_bound;
        }
        const double path_bound = path_bound_for_indices(indices, depth_remaining);
        return std::max(sig_bound, path_bound);
    }

    struct CandidateEval {
        int feature = -1;
        int groups = 0;
        AtomizedCandidate candidate;
        std::vector<std::vector<int>> group_atom_positions;
        std::vector<std::vector<int>> child_indices;
        std::vector<std::vector<std::pair<int, int>>> group_spans;
        std::vector<double> child_lower_bounds;
        std::vector<size_t> lower_bound_child_order;
        size_t total_child_rows = 0U;
        size_t max_child_rows = 0U;
        size_t lower_bound_child_cursor = 0U;
        double search_priority = kInfinity;
        double leaf_completion_objective = kInfinity;
        double lower_bound = kInfinity;
        bool lower_bound_complete = false;
    };

    size_t collect_ranked_candidates(
        const std::vector<int> &indices,
        double mu_node,
        std::vector<CandidateEval> &candidate_evals,
        bool count_telemetry = true
    ) const {
        std::vector<PreparedFeatureAtomized> prepared_by_feature((size_t)n_features_);
        candidate_evals.clear();
        size_t preserved_feature_count = 0U;

        for (int feature = 0; feature < n_features_; ++feature) {
            PreparedFeatureAtomized prepared;
            if (!prepare_feature_atomized_local(indices, feature, mu_node, prepared)) {
                continue;
            }
            ++preserved_feature_count;
            prepared_by_feature[(size_t)feature] = std::move(prepared);
        }

        for (int feature = 0; feature < n_features_; ++feature) {
            const PreparedFeatureAtomized &prepared = prepared_by_feature[(size_t)feature];
            if (!prepared.valid) {
                continue;
            }
            for (size_t groups = 2; groups < prepared.coarse_by_groups.size(); ++groups) {
                std::vector<AtomizedCandidate> candidates =
                    select_atomized_candidates_for_arity(
                        feature,
                        prepared,
                        (int)groups,
                        mu_node);
                if (candidates.empty()) {
                    continue;
                }
                if (count_telemetry) {
                    auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
                    telemetry.greedy_interval_evals += (long long)candidates.size();
                    telemetry.atomized_final_candidates += (long long)candidates.size();
                }
                for (auto &candidate : candidates) {
                    if (!candidate.feasible) {
                        continue;
                    }
                    CandidateEval eval;
                    eval.feature = feature;
                    eval.groups = (int)groups;
                    eval.candidate = std::move(candidate);
                    std::vector<int> group_counts;
                    if (!fill_groups_from_assignment(
                            eval.candidate.assignment,
                            eval.candidate.groups,
                            eval.group_atom_positions,
                            group_counts)) {
                        continue;
                    }
                    std::vector<int>().swap(eval.candidate.assignment);
                    const OrderedBins &bins = prepared.bins;
                    eval.child_indices.resize((size_t)eval.group_atom_positions.size());
                    eval.group_spans.resize((size_t)eval.group_atom_positions.size());
                    bool ok = true;
                    size_t total_child_rows = 0U;
                    size_t max_child_rows = 0U;
                    for (size_t group_idx = 0; group_idx < eval.group_atom_positions.size(); ++group_idx) {
                        gather_group_members_sorted(
                            bins,
                            eval.group_atom_positions[group_idx],
                            eval.child_indices[group_idx]);
                        if ((int)eval.child_indices[group_idx].size() < min_child_size_) {
                            ok = false;
                            break;
                        }
                        const size_t child_rows = eval.child_indices[group_idx].size();
                        total_child_rows += child_rows;
                        max_child_rows = std::max(max_child_rows, child_rows);
                        eval.group_spans[group_idx] =
                            atom_positions_to_spans(bins, eval.group_atom_positions[group_idx]);
                    }
                    if (!ok) {
                        continue;
                    }
                    std::vector<std::vector<int>>().swap(eval.group_atom_positions);
                    eval.total_child_rows = total_child_rows;
                    eval.max_child_rows = max_child_rows;
                    const double impurity_proxy = atomized_score_proxy(eval.candidate.score, mu_node);
                    const double hardloss_proxy = atomized_score_proxy(
                        eval.candidate.score,
                        mu_node,
                        AtomizedObjectiveMode::kHardLoss);
                    eval.search_priority = std::min(impurity_proxy, hardloss_proxy);
                    candidate_evals.push_back(std::move(eval));
                }
            }
        }
        return preserved_feature_count;
    }

    void initialize_candidate_lower_bounds(
        CandidateEval &eval,
        int depth_remaining
    ) const {
        eval.child_lower_bounds.resize(eval.child_indices.size());
        eval.lower_bound_child_order.resize(eval.child_indices.size());
        eval.lower_bound_child_cursor = 0U;
        eval.lower_bound = 0.0;
        for (size_t child_idx = 0; child_idx < eval.child_indices.size(); ++child_idx) {
            const double child_lower_bound =
                signature_bound_for_indices(eval.child_indices[child_idx], depth_remaining - 1);
            eval.child_lower_bounds[child_idx] = child_lower_bound;
            eval.lower_bound += child_lower_bound;
            eval.lower_bound_child_order[child_idx] = child_idx;
        }
        std::sort(
            eval.lower_bound_child_order.begin(),
            eval.lower_bound_child_order.end(),
            [&eval](size_t lhs_idx, size_t rhs_idx) {
                const size_t lhs_size = eval.child_indices[lhs_idx].size();
                const size_t rhs_size = eval.child_indices[rhs_idx].size();
                if (lhs_size != rhs_size) {
                    return lhs_size > rhs_size;
                }
                return lhs_idx < rhs_idx;
            });
        eval.lower_bound_child_cursor = eval.lower_bound_child_order.size();
        eval.lower_bound_complete = true;
    }

    bool refine_candidate_lower_bound(
        CandidateEval &eval,
        int depth_remaining
    ) const {
        if (eval.lower_bound_complete) {
            return false;
        }
        if (eval.lower_bound_child_cursor >= eval.lower_bound_child_order.size()) {
            eval.lower_bound_complete = true;
            return false;
        }
        const size_t child_idx = eval.lower_bound_child_order[eval.lower_bound_child_cursor++];
        const double child_lower_bound =
            signature_bound_for_indices(eval.child_indices[child_idx], depth_remaining - 1);
        eval.lower_bound += child_lower_bound - eval.child_lower_bounds[child_idx];
        eval.child_lower_bounds[child_idx] = child_lower_bound;
        if (eval.lower_bound_child_cursor >= eval.lower_bound_child_order.size()) {
            eval.lower_bound_complete = true;
        }
        return true;
    }

    double candidate_leaf_completion_objective(const CandidateEval &eval) const {
        double objective = 0.0;
        for (const auto &child_indices : eval.child_indices) {
            const SubproblemStats child_stats = compute_subproblem_stats(child_indices);
            objective += leaf_solution(child_stats).first;
        }
        return objective;
    }

    double tighten_candidate_lower_bound_with_path(
        const CandidateEval &eval,
        int depth_remaining
    ) const {
        ++profiling_path_bound_tighten_attempts_;
        if (depth_remaining <= 1 || eval.child_indices.empty()) {
            return eval.lower_bound;
        }
        size_t most_uncertain_child = 0U;
        size_t largest_child_size = 0U;
        double tightened = 0.0;
        for (size_t child_idx = 0; child_idx < eval.child_indices.size(); ++child_idx) {
            const auto &child_indices = eval.child_indices[child_idx];
            tightened += signature_bound_for_indices(child_indices, depth_remaining - 1);
            if (child_indices.size() > largest_child_size) {
                largest_child_size = child_indices.size();
                most_uncertain_child = child_idx;
            }
        }
        const double child_sig = signature_bound_for_indices(
            eval.child_indices[most_uncertain_child],
            depth_remaining - 1);
        if (eval.child_indices[most_uncertain_child].size() > static_cast<size_t>(path_lp_max_child_)) {
            ++profiling_path_bound_skip_large_child_;
            return tightened;
        }
        const double child_tight = state_lower_bound_for_indices(
            eval.child_indices[most_uncertain_child],
            depth_remaining - 1,
            true);
        if (child_tight > child_sig + kEpsUpdate) {
            ++profiling_path_bound_tighten_effective_;
            tightened += child_tight - child_sig;
        }
        return tightened;
    }

    GreedyResult greedy_complete_impl(
        std::vector<int> indices,
        int depth_remaining,
        bool exact_mode) {
        ++profiling_greedy_complete_calls_;
        record_greedy_complete_call(depth_remaining);
        const int current_depth = full_depth_budget_ - depth_remaining;
        if (exact_mode && current_depth < effective_lookahead_depth_) {
            ++exact_dp_subproblem_calls_above_lookahead_;
        } else {
            ++greedy_subproblem_calls_;
        }
        ScopedTimer greedy_timer(profiling_greedy_complete_sec_);
        check_timeout();

        const bool use_greedy_cache = greedy_cache_max_depth_ >= 0;
        const auto signature_summary = get_canonical_signature_summary(indices);
        const size_t block_count = signature_summary.block_count;
        std::string state_key;
        if (use_greedy_cache) {
            state_key = serialized_indices_depth_key(indices, depth_remaining);
            auto cache_it = greedy_cache_.find(state_key);
            if (cache_it != greedy_cache_.end()) {
                ++greedy_cache_hits_;
                trace_greedy_snapshot(
                    "cache_hit",
                    depth_remaining,
                    indices.size(),
                    block_count,
                    0U,
                    0U,
                    0U,
                    0U,
                    0U,
                    0U,
                    cache_it->second.result.objective,
                    cache_it->second.result.objective,
                    0U);
                return cache_it->second.result;
            }
        }
        trace_greedy_snapshot(
            "enter",
            depth_remaining,
            indices.size(),
            block_count,
            0U,
            0U,
            0U,
            0U,
            0U,
            0U,
            kInfinity,
            kInfinity,
            0U);

        const SubproblemStats stats = compute_subproblem_stats(indices);
        auto [leaf_objective, leaf_tree] = leaf_solution(stats);
        if (depth_remaining <= 0) {
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_leaf_depth",
                depth_remaining,
                indices.size(),
                block_count,
                0U,
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }
        if (stats.pure) {
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_pure",
                depth_remaining,
                indices.size(),
                block_count,
                0U,
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }
        if ((int)indices.size() < min_split_size_) {
            GreedyResult solved{leaf_objective, leaf_tree};
            trace_greedy_snapshot(
                "return_min_split",
                depth_remaining,
                indices.size(),
                block_count,
                0U,
                0U,
                0U,
                0U,
                0U,
                0U,
                solved.objective,
                solved.objective,
                0U);
            cache_store(state_key, solved, depth_remaining);
            return solved;
        }

        const double mu_node = effective_sample_unit(stats);
        const bool use_exact_frontier = exact_mode && current_depth < effective_lookahead_depth_;
        std::vector<CandidateEval> candidate_evals;
        size_t preserved_feature_count = 0U;
        auto release_candidate_payloads = [](CandidateEval &candidate) {
            std::vector<std::vector<int>>().swap(candidate.child_indices);
            std::vector<std::vector<std::pair<int, int>>>().swap(candidate.group_spans);
            std::vector<double>().swap(candidate.child_lower_bounds);
            std::vector<size_t>().swap(candidate.lower_bound_child_order);
            std::vector<int>().swap(candidate.candidate.assignment);
        };
        auto candidate_compare = [&](const CandidateEval &lhs, const CandidateEval &rhs) {
            if (lhs.leaf_completion_objective < rhs.leaf_completion_objective - kEpsUpdate) {
                return true;
            }
            if (rhs.leaf_completion_objective < lhs.leaf_completion_objective - kEpsUpdate) {
                return false;
            }
            if (lhs.lower_bound < rhs.lower_bound - kEpsUpdate) {
                return true;
            }
            if (rhs.lower_bound < lhs.lower_bound - kEpsUpdate) {
                return false;
            }
            if (lhs.search_priority < rhs.search_priority - kEpsUpdate) {
                return true;
            }
            if (rhs.search_priority < lhs.search_priority - kEpsUpdate) {
                return false;
            }
            if (atomized_candidate_better_global(
                    lhs.candidate,
                    rhs.candidate,
                    lhs.feature,
                    rhs.feature)) {
                return true;
            }
            if (atomized_candidate_better_global(
                    rhs.candidate,
                    lhs.candidate,
                    rhs.feature,
                    lhs.feature)) {
                return false;
            }
            if (lhs.max_child_rows != rhs.max_child_rows) {
                return lhs.max_child_rows < rhs.max_child_rows;
            }
            if (lhs.total_child_rows != rhs.total_child_rows) {
                return lhs.total_child_rows < rhs.total_child_rows;
            }
            if (lhs.groups != rhs.groups) {
                return lhs.groups < rhs.groups;
            }
            return false;
        };
        {
            ScopedTimer candidate_timer(profiling_candidate_generation_sec_);
            preserved_feature_count =
                collect_ranked_candidates(indices, mu_node, candidate_evals);
            if (use_exact_frontier) {
                for (auto &eval : candidate_evals) {
                    eval.leaf_completion_objective = candidate_leaf_completion_objective(eval);
                    initialize_candidate_lower_bounds(eval, depth_remaining);
                }
                std::sort(candidate_evals.begin(), candidate_evals.end(), candidate_compare);
            } else {
                auto greedy_compare = [&](const CandidateEval &lhs, const CandidateEval &rhs) {
                    if (lhs.search_priority < rhs.search_priority - kEpsUpdate) {
                        return true;
                    }
                    if (rhs.search_priority < lhs.search_priority - kEpsUpdate) {
                        return false;
                    }
                    if (atomized_candidate_better_global(
                            lhs.candidate,
                            rhs.candidate,
                            lhs.feature,
                            rhs.feature)) {
                        return true;
                    }
                    if (atomized_candidate_better_global(
                            rhs.candidate,
                            lhs.candidate,
                            rhs.feature,
                            lhs.feature)) {
                        return false;
                    }
                    if (lhs.max_child_rows != rhs.max_child_rows) {
                        return lhs.max_child_rows < rhs.max_child_rows;
                    }
                    if (lhs.total_child_rows != rhs.total_child_rows) {
                        return lhs.total_child_rows < rhs.total_child_rows;
                    }
                    if (lhs.groups != rhs.groups) {
                        return lhs.groups < rhs.groups;
                    }
                    return false;
                };
                if (!candidate_evals.empty()) {
                    auto best_it = std::min_element(candidate_evals.begin(), candidate_evals.end(), greedy_compare);
                    if (best_it != candidate_evals.begin()) {
                        std::iter_swap(candidate_evals.begin(), best_it);
                    }
                    for (size_t i = 1; i < candidate_evals.size(); ++i) {
                        release_candidate_payloads(candidate_evals[i]);
                    }
                    candidate_evals.resize(1);
                }
            }
        }
        auto bump_count_histogram = [](std::vector<long long> &hist, size_t bucket) {
            if (hist.size() <= bucket) {
                hist.resize(bucket + 1U, 0);
            }
            ++hist[bucket];
        };
        size_t promising_count = 0U;
        const double initial_incumbent_objective = leaf_objective;
        double best_lower_bound = candidate_evals.empty()
            ? kInfinity
            : (use_exact_frontier ? candidate_evals.front().leaf_completion_objective
                                  : candidate_evals.front().search_priority);
        promising_count = candidate_evals.size();
        bump_count_histogram(greedy_state_block_count_histogram_, block_count);
        bump_count_histogram(greedy_feature_preserved_histogram_, preserved_feature_count);
        bump_count_histogram(greedy_candidate_count_histogram_, candidate_evals.size());
        trace_greedy_snapshot(
            "candidate_ready",
            depth_remaining,
            indices.size(),
            block_count,
            preserved_feature_count,
            candidate_evals.size(),
            promising_count,
            0U,
            0U,
            0U,
            initial_incumbent_objective,
            best_lower_bound,
            0U);

        GreedyResult solved{leaf_objective, leaf_tree};
        double incumbent_objective = leaf_objective;
        std::shared_ptr<Node> incumbent_tree = leaf_tree;
        std::vector<std::shared_ptr<Node>> child_nodes;
        size_t processed_candidate_count = 0U;
        size_t recurse_attempt_count = 0U;
        size_t incumbent_update_count = 0U;
        size_t path_tighten_count = 0U;
        {
            ScopedTimer recursion_timer(profiling_recursive_child_eval_sec_);
            while (!candidate_evals.empty()) {
                if (use_exact_frontier) {
                    std::sort(candidate_evals.begin(), candidate_evals.end(), candidate_compare);
                }
                CandidateEval &eval = candidate_evals.front();
                ++processed_candidate_count;
                if (use_exact_frontier && eval.lower_bound >= incumbent_objective - kEpsUpdate) {
                    ++lp_skip_reason_not_promising_;
                    if (trace_enabled_ && depth_remaining <= trace_max_depth_ &&
                        (processed_candidate_count % 128U == 0U)) {
                        trace_greedy_snapshot(
                            "progress_pruned",
                            depth_remaining,
                            indices.size(),
                            block_count,
                            preserved_feature_count,
                            candidate_evals.size(),
                            promising_count,
                            processed_candidate_count,
                            recurse_attempt_count,
                            path_tighten_count,
                            incumbent_objective,
                            best_lower_bound,
                            incumbent_update_count);
                    }
                    release_candidate_payloads(eval);
                    candidate_evals.erase(candidate_evals.begin());
                    continue;
                }
                if (use_exact_frontier && !eval.lower_bound_complete) {
                    refine_candidate_lower_bound(eval, depth_remaining);
                    if (trace_enabled_ && depth_remaining <= trace_max_depth_ &&
                        (processed_candidate_count % 128U == 0U)) {
                        trace_greedy_snapshot(
                            "progress_refine",
                            depth_remaining,
                            indices.size(),
                            block_count,
                            preserved_feature_count,
                            candidate_evals.size(),
                            promising_count,
                            processed_candidate_count,
                            recurse_attempt_count,
                            path_tighten_count,
                            incumbent_objective,
                            candidate_evals.front().lower_bound,
                            incumbent_update_count);
                    }
                    continue;
                }
                double candidate_lower_bound = eval.lower_bound;
                if (use_exact_frontier) {
                    if (depth_remaining <= 1) {
                        ++lp_skip_reason_depth_gate_;
                    } else if (path_tighten_count < 2U) {
                        candidate_lower_bound = tighten_candidate_lower_bound_with_path(eval, depth_remaining);
                        ++path_tighten_count;
                        if (candidate_lower_bound >= incumbent_objective - kEpsUpdate) {
                            if (trace_enabled_ && depth_remaining <= trace_max_depth_ &&
                                (processed_candidate_count % 128U == 0U)) {
                                trace_greedy_snapshot(
                                    "progress_tightened_pruned",
                                    depth_remaining,
                                    indices.size(),
                                    block_count,
                                    preserved_feature_count,
                                    candidate_evals.size(),
                                    promising_count,
                                    processed_candidate_count,
                                    recurse_attempt_count,
                                    path_tighten_count,
                                    incumbent_objective,
                                    best_lower_bound,
                                    incumbent_update_count);
                            }
                            release_candidate_payloads(eval);
                            candidate_evals.erase(candidate_evals.begin());
                            continue;
                        }
                    } else {
                        ++lp_skip_reason_tighten_cap_;
                    }
                }
                double objective = 0.0;
                child_nodes.assign(eval.child_indices.size(), std::shared_ptr<Node>{});
                bool build_ok = true;
                bool has_prediction_flip = false;
                std::vector<size_t> child_order(eval.child_indices.size());
                std::iota(child_order.begin(), child_order.end(), 0U);
                auto child_order_compare = [&eval](size_t lhs_idx, size_t rhs_idx) {
                    const size_t lhs_size = eval.child_indices[lhs_idx].size();
                    const size_t rhs_size = eval.child_indices[rhs_idx].size();
                    if (lhs_size != rhs_size) {
                        return lhs_size > rhs_size;
                    }
                    return lhs_idx < rhs_idx;
                };
                std::sort(child_order.begin(), child_order.end(), child_order_compare);
                for (size_t order_pos = 0; order_pos < child_order.size(); ++order_pos) {
                    const size_t group_idx = child_order[order_pos];
                    std::vector<int> subset_sorted = eval.child_indices[group_idx];
                    if (subset_sorted.empty()) {
                        build_ok = false;
                        break;
                    }
                    if (depth_remaining == 1) {
                        const SubproblemStats child_stats = compute_subproblem_stats(subset_sorted);
                        auto [child_objective, child_tree] = leaf_solution(child_stats);
                        objective += child_objective;
                        if (child_stats.prediction != stats.prediction) {
                            has_prediction_flip = true;
                        }
                        child_nodes[group_idx] = child_tree;
                    } else {
                        ++recurse_attempt_count;
                        GreedyResult child;
                        if (exact_mode && current_depth + 1 < effective_lookahead_depth_) {
                            child = solve_subproblem(std::move(subset_sorted), depth_remaining - 1);
                        } else {
                            child = greedy_complete_impl(std::move(subset_sorted), depth_remaining - 1, false);
                        }
                        objective += child.objective;
                        if (objective >= incumbent_objective - kEpsUpdate) {
                            build_ok = false;
                            break;
                        }
                        child_nodes[group_idx] = child.tree;
                    }
                }
                if (!build_ok) {
                    release_candidate_payloads(eval);
                    candidate_evals.erase(candidate_evals.begin());
                    continue;
                }
                std::shared_ptr<Node> tree = build_internal_node_from_group_spans(
                    eval.feature,
                    eval.group_spans,
                    child_nodes,
                    leaf_tree->prediction,
                    (int)indices.size());
                if (depth_remaining == 1) {
                    if (tree && has_prediction_flip && objective + kEpsUpdate < incumbent_objective) {
                        incumbent_objective = objective;
                        incumbent_tree = tree;
                        ++incumbent_update_count;
                    }
                    if (trace_enabled_ && depth_remaining <= trace_max_depth_ &&
                        (processed_candidate_count % 128U == 0U)) {
                        trace_greedy_snapshot(
                            "progress_depth1",
                            depth_remaining,
                            indices.size(),
                            block_count,
                            preserved_feature_count,
                            candidate_evals.size(),
                            promising_count,
                            processed_candidate_count,
                            recurse_attempt_count,
                            path_tighten_count,
                            incumbent_objective,
                            best_lower_bound,
                            incumbent_update_count);
                    }
                    release_candidate_payloads(eval);
                    candidate_evals.erase(candidate_evals.begin());
                    continue;
                }
                int constant_prediction = -1;
                if (tree && !subtree_is_constant_prediction(tree, constant_prediction) &&
                    objective + kEpsUpdate < incumbent_objective) {
                    incumbent_objective = objective;
                    incumbent_tree = tree;
                    ++incumbent_update_count;
                }
                if (trace_enabled_ && depth_remaining <= trace_max_depth_ &&
                    (processed_candidate_count % 128U == 0U || incumbent_update_count > 0U)) {
                    trace_greedy_snapshot(
                        "progress",
                        depth_remaining,
                        indices.size(),
                        block_count,
                        preserved_feature_count,
                        candidate_evals.size(),
                        promising_count,
                        processed_candidate_count,
                        recurse_attempt_count,
                        path_tighten_count,
                        incumbent_objective,
                        best_lower_bound,
                        incumbent_update_count);
                }
                release_candidate_payloads(eval);
                candidate_evals.erase(candidate_evals.begin());
            }
        }

        solved = GreedyResult{incumbent_objective, incumbent_tree};
        trace_greedy_snapshot(
            "exit",
            depth_remaining,
            indices.size(),
            block_count,
            preserved_feature_count,
            candidate_evals.size(),
            promising_count,
            processed_candidate_count,
            recurse_attempt_count,
            path_tighten_count,
            incumbent_objective,
            best_lower_bound,
            incumbent_update_count);

        if (solved.tree && !solved.tree->is_leaf) {
            ++greedy_internal_nodes_;
        }
        cache_store(state_key, solved, depth_remaining);
        return solved;
    }

    GreedyResult greedy_complete(
        std::vector<int> indices,
        int depth_remaining) {
        return greedy_complete_impl(std::move(indices), depth_remaining, false);
    }

        GreedyResult solve_subproblem(
            std::vector<int> indices,
            int depth_remaining) {
        const int current_depth = full_depth_budget_ - depth_remaining;
        if (current_depth < effective_lookahead_depth_) {
            return greedy_complete_impl(std::move(indices), depth_remaining, true);
        }
        return greedy_complete_impl(std::move(indices), depth_remaining, false);
    }
