    struct AtomizedAtom {
        int atom_pos = -1;
        int bin_value = -1;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        double teacher_prob = 0.5;
        int empirical_prediction = 0;
        int teacher_prediction = 0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedBlock {
        std::vector<int> atom_positions;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        int empirical_prediction = 0;
        int teacher_prediction = 0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedScore {
        double hard_loss = kInfinity;
        double soft_loss = kInfinity;
        double hard_impurity = kInfinity;
        double soft_impurity = kInfinity;
        double boundary_penalty = kInfinity;
        int components = std::numeric_limits<int>::max();
    };

    struct AtomizedCandidate {
        bool feasible = false;
        AtomizedScore score;
        int feature = -1;
        int groups = 0;
        bool hard_loss_mode = false;
        std::vector<int> assignment;
        std::vector<double> branch_hard_losses;
    };

    struct AtomizedCoarseCandidate {
        AtomizedCandidate geometry_seed_candidate;
        AtomizedCandidate block_candidate;
        AtomizedCandidate hardloss_candidate;
        AtomizedCandidate candidate;
        std::vector<int> initial_block_assignment;
        std::vector<int> refined_block_assignment;
    };

    struct AtomizedCandidatePair {
        AtomizedCandidate impurity;
        AtomizedCandidate misclassification;
    };

    struct AtomizedPrefixes {
        std::vector<int> rows;
        std::vector<double> pos;
        std::vector<double> neg;
        std::vector<double> teacher_pos;
        std::vector<double> teacher_neg;
        std::vector<double> class_weight_prefix;
        std::vector<double> teacher_class_weight_prefix;
    };

    struct PreparedFeatureAtomized {
        bool valid = false;
        bool has_block_compression = false;
        OrderedBins bins;
        std::vector<AtomizedAtom> atoms;
        double atom_hard_floor = 0.0;
        double atom_imp_floor = 0.0;
        AtomizedPrefixes atom_prefix;
        std::vector<double> atom_adjacency_bonus;
        double atom_adjacency_bonus_total = 0.0;
        std::vector<AtomizedBlock> blocks;
        std::vector<AtomizedAtom> block_atoms;
        AtomizedPrefixes block_prefix;
        std::vector<AtomizedCoarseCandidate> coarse_by_groups;
        std::vector<AtomizedCoarseCandidate> coarse_by_groups_hardloss;
        int q_effective = 0;
    };

    struct AtomizedRefinementMove {
        bool valid = false;
        int source_group = -1;
        int target_group = -1;
        int start = -1;
        int end = -1;
        int length = 0;
        int delta_components = 0;
        int row_count = 0;
        double pos_weight = 0.0;
        double neg_weight = 0.0;
        double teacher_pos_weight = 0.0;
        double teacher_neg_weight = 0.0;
        double delta_j = 0.0;
        double delta_hard = 0.0;
        double delta_soft = 0.0;
        double source_loss_after = 0.0;
        double target_loss_after = 0.0;
        double source_hard_impurity_after = 0.0;
        double target_hard_impurity_after = 0.0;
        double source_soft_impurity_after = 0.0;
        double target_soft_impurity_after = 0.0;
        std::vector<double> class_weight;
        std::vector<double> teacher_class_weight;
    };

    struct AtomizedRefinementSummary {
        int moves = 0;
        int bridge_policy_calls = 0;
        int refine_windowed_calls = 0;
        int refine_unwindowed_calls = 0;
        int refine_overlap_segments = 0;
        int refine_calls_with_overlap = 0;
        int refine_calls_without_overlap = 0;
        int candidate_total = 0;
        int candidate_legal = 0;
        int candidate_source_size_rejects = 0;
        int candidate_target_size_rejects = 0;
        int candidate_descent_eligible = 0;
        int candidate_descent_rejected = 0;
        int candidate_bridge_eligible = 0;
        int candidate_bridge_window_blocked = 0;
        int candidate_bridge_used_blocked = 0;
        int candidate_bridge_guide_rejected = 0;
        int candidate_cleanup_eligible = 0;
        int candidate_cleanup_primary_rejected = 0;
        int candidate_cleanup_complexity_rejected = 0;
        int candidate_score_rejected = 0;
        int descent_moves = 0;
        int bridge_moves = 0;
        int simplify_moves = 0;
        std::vector<long long> source_group_row_size_histogram;
        std::vector<long long> source_component_atom_size_histogram;
        std::vector<long long> source_component_row_size_histogram;
        double hard_gain = 0.0;
        double soft_gain = 0.0;
        double delta_j = 0.0;
        int component_delta = 0;
        bool improved = false;
    };

    static constexpr double kBoundaryPenaltyWeight = 1.0;

    enum class AtomizedObjectiveMode {
        kImpurity,
        kHardLoss
    };

    double atomized_joint_impurity(const AtomizedScore &score) const {
        return score.hard_impurity + score.soft_impurity;
    }

    double atomized_primary_objective(
        const AtomizedScore &score,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (mode == AtomizedObjectiveMode::kHardLoss) {
            return score.hard_loss;
        }
        return score.hard_impurity + score.soft_impurity;
    }

    double atomized_candidate_primary_objective(const AtomizedCandidate &candidate) const {
        return candidate.hard_loss_mode
            ? candidate.score.hard_loss
            : atomized_primary_objective(candidate.score, AtomizedObjectiveMode::kImpurity);
    }

    double atomized_candidate_excess_components(const AtomizedCandidate &candidate) const {
        return std::max(0.0, (double)(candidate.score.components - candidate.groups));
    }

    double atomized_candidate_bad_boundary(const AtomizedCandidate &candidate) const {
        if (!teacher_available_) {
            return 0.0;
        }
        return candidate.score.boundary_penalty;
    }

    double atomized_score_bad_boundary(const AtomizedScore &score) const {
        if (!teacher_available_) {
            return 0.0;
        }
        return score.boundary_penalty;
    }

    bool atomized_score_better(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        const double lhs_primary = atomized_primary_objective(lhs, mode);
        const double rhs_primary = atomized_primary_objective(rhs, mode);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        const double lhs_boundary = atomized_score_bad_boundary(lhs);
        const double rhs_boundary = atomized_score_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        return false;
    }

    double atomized_score_proxy(
        const AtomizedScore &score,
        double mu_node,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        return atomized_primary_objective(score, mode) +
            mu_node * (kBoundaryPenaltyWeight * score.boundary_penalty + (double)score.components);
    }

    bool atomized_score_better_for_refinement(
        const AtomizedScore &lhs,
        const AtomizedScore &rhs,
        double mu_node,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        const double lhs_primary = atomized_primary_objective(lhs, mode);
        const double rhs_primary = atomized_primary_objective(rhs, mode);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        if (lhs.components != rhs.components) {
            return lhs.components < rhs.components;
        }
        const double lhs_boundary = atomized_score_bad_boundary(lhs);
        const double rhs_boundary = atomized_score_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        return false;
    }

    bool atomized_candidate_better_for_objective(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature,
        AtomizedObjectiveMode mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (!rhs.feasible) {
            return lhs.feasible;
        }
        if (!lhs.feasible) {
            return false;
        }
        const double lhs_primary = atomized_candidate_primary_objective(lhs);
        const double rhs_primary = atomized_candidate_primary_objective(rhs);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        const double lhs_excess = std::max(0.0, (double)(lhs.score.components - lhs.groups));
        const double rhs_excess = std::max(0.0, (double)(rhs.score.components - rhs.groups));
        if (lhs_excess < rhs_excess - kEpsUpdate) {
            return true;
        }
        if (rhs_excess < lhs_excess - kEpsUpdate) {
            return false;
        }
        const double lhs_boundary = atomized_candidate_bad_boundary(lhs);
        const double rhs_boundary = atomized_candidate_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        if (lhs.groups != rhs.groups) {
            return lhs.groups < rhs.groups;
        }
        if (lhs_feature != rhs_feature) {
            return lhs_feature < rhs_feature;
        }
        if (lhs.assignment != rhs.assignment) {
            return lhs.assignment < rhs.assignment;
        }
        return false;
    }

    bool atomized_candidate_dominates(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs
    ) const {
        if (!lhs.feasible || !rhs.feasible) {
            return false;
        }
        const bool loss_not_worse = lhs.score.hard_loss <= rhs.score.hard_loss + kEpsUpdate;
        const bool impurity_not_worse =
            lhs.score.hard_impurity <= rhs.score.hard_impurity + kEpsUpdate;
        const bool loss_better = lhs.score.hard_loss < rhs.score.hard_loss - kEpsUpdate;
        const bool impurity_better =
            lhs.score.hard_impurity < rhs.score.hard_impurity - kEpsUpdate;
        return loss_not_worse && impurity_not_worse && (loss_better || impurity_better);
    }

    bool atomized_candidate_better_global(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs,
        int lhs_feature,
        int rhs_feature
    ) const {
        if (!rhs.feasible) {
            return lhs.feasible;
        }
        if (!lhs.feasible) {
            return false;
        }
        const double lhs_primary = atomized_candidate_primary_objective(lhs);
        const double rhs_primary = atomized_candidate_primary_objective(rhs);
        if (lhs_primary < rhs_primary - kEpsUpdate) {
            return true;
        }
        if (rhs_primary < lhs_primary - kEpsUpdate) {
            return false;
        }
        const double lhs_excess = atomized_candidate_excess_components(lhs);
        const double rhs_excess = atomized_candidate_excess_components(rhs);
        if (lhs_excess < rhs_excess - kEpsUpdate) {
            return true;
        }
        if (rhs_excess < lhs_excess - kEpsUpdate) {
            return false;
        }
        const double lhs_boundary = atomized_candidate_bad_boundary(lhs);
        const double rhs_boundary = atomized_candidate_bad_boundary(rhs);
        if (lhs_boundary < rhs_boundary - kEpsUpdate) {
            return true;
        }
        if (rhs_boundary < lhs_boundary - kEpsUpdate) {
            return false;
        }
        if (lhs.groups != rhs.groups) {
            return lhs.groups < rhs.groups;
        }
        if (lhs_feature != rhs_feature) {
            return lhs_feature < rhs_feature;
        }
        if (lhs.assignment != rhs.assignment) {
            return lhs.assignment < rhs.assignment;
        }
        return false;
    }

    static bool atomized_assignment_equivalent(
        const AtomizedCandidate &lhs,
        const AtomizedCandidate &rhs
    ) {
        if (!lhs.feasible || !rhs.feasible) {
            return false;
        }
        if (lhs.assignment.size() != rhs.assignment.size()) {
            return false;
        }
        std::vector<int> lhs_to_rhs;
        std::vector<int> rhs_to_lhs;
        int lhs_groups = 0;
        int rhs_groups = 0;
        for (size_t i = 0; i < lhs.assignment.size(); ++i) {
            lhs_groups = std::max(lhs_groups, lhs.assignment[i] + 1);
            rhs_groups = std::max(rhs_groups, rhs.assignment[i] + 1);
        }
        lhs_to_rhs.assign((size_t)lhs_groups, -1);
        rhs_to_lhs.assign((size_t)rhs_groups, -1);
        for (size_t i = 0; i < lhs.assignment.size(); ++i) {
            const int lhs_group = lhs.assignment[i];
            const int rhs_group = rhs.assignment[i];
            if (lhs_group < 0 || rhs_group < 0) {
                return false;
            }
            int &mapped_rhs = lhs_to_rhs[(size_t)lhs_group];
            int &mapped_lhs = rhs_to_lhs[(size_t)rhs_group];
            if (mapped_rhs < 0 && mapped_lhs < 0) {
                mapped_rhs = rhs_group;
                mapped_lhs = lhs_group;
                continue;
            }
            if (mapped_rhs != rhs_group || mapped_lhs != lhs_group) {
                return false;
            }
        }
        return true;
    }

    void record_family_compare_stats(
        const AtomizedCandidate &impurity,
        const AtomizedCandidate &misclassification
    ) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        ++telemetry.family_compare_total;
        telemetry.family1_hard_loss_sum += impurity.score.hard_loss;
        telemetry.family2_hard_loss_sum += misclassification.score.hard_loss;
        telemetry.family_hard_loss_delta_sum +=
            (misclassification.score.hard_loss - impurity.score.hard_loss);
        telemetry.family1_hard_impurity_sum += impurity.score.hard_impurity;
        telemetry.family2_hard_impurity_sum += misclassification.score.hard_impurity;
        telemetry.family_hard_impurity_delta_sum +=
            (misclassification.score.hard_impurity - impurity.score.hard_impurity);
        telemetry.family1_soft_impurity_sum += impurity.score.soft_impurity;
        telemetry.family2_soft_impurity_sum += misclassification.score.soft_impurity;
        telemetry.family_soft_impurity_delta_sum +=
            (misclassification.score.soft_impurity - impurity.score.soft_impurity);
        telemetry.family1_joint_impurity_sum += atomized_joint_impurity(impurity.score);
        telemetry.family2_joint_impurity_sum += atomized_joint_impurity(misclassification.score);
        telemetry.family_joint_impurity_delta_sum +=
            (atomized_joint_impurity(misclassification.score) - atomized_joint_impurity(impurity.score));

        const bool family2_loss_better =
            misclassification.score.hard_loss < impurity.score.hard_loss - kEpsUpdate;
        const bool family1_loss_better =
            impurity.score.hard_loss < misclassification.score.hard_loss - kEpsUpdate;
        const bool family2_hard_impurity_better =
            misclassification.score.hard_impurity < impurity.score.hard_impurity - kEpsUpdate;
        const bool family1_hard_impurity_better =
            impurity.score.hard_impurity < misclassification.score.hard_impurity - kEpsUpdate;
        const double impurity_metric_1 = atomized_joint_impurity(impurity.score);
        const double impurity_metric_2 = atomized_joint_impurity(misclassification.score);
        const bool family2_joint_better = impurity_metric_2 < impurity_metric_1 - kEpsUpdate;
        const bool family1_joint_better = impurity_metric_1 < impurity_metric_2 - kEpsUpdate;
        if (!family2_loss_better && !family1_loss_better) {
            ++telemetry.family_hard_loss_ties;
        }
        if (!family2_hard_impurity_better && !family1_hard_impurity_better) {
            ++telemetry.family_hard_impurity_ties;
        }
        if (!family2_joint_better && !family1_joint_better) {
            ++telemetry.family_joint_impurity_ties;
        }
        if (family2_loss_better) {
            ++telemetry.family2_hard_loss_wins;
        }
        if (family2_hard_impurity_better) {
            ++telemetry.family2_hard_impurity_wins;
        }
        if (family2_joint_better) {
            ++telemetry.family2_joint_impurity_wins;
        }
        if (family2_loss_better && family2_joint_better) {
            ++telemetry.family2_both_wins;
        }
        if (family1_loss_better && family1_joint_better) {
            ++telemetry.family1_both_wins;
        }
        if (!family2_loss_better && !family1_loss_better &&
            !family2_joint_better && !family1_joint_better) {
            ++telemetry.family_neither_both_wins;
        }
    }

    void record_family1_hard_loss_inversion_trace(
        const AtomizedCandidate &final_impurity,
        const AtomizedCandidate &final_misclassification,
        const AtomizedCandidate &raw_impurity,
        const AtomizedCandidate &raw_misclassification,
        int feature,
        int groups
    ) const {
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        if (telemetry.family1_hard_loss_inversion_traces.size() >= 24) {
            return;
        }

        const double final_impurity_joint = atomized_joint_impurity(final_impurity.score);
        const double final_misclassification_joint = atomized_joint_impurity(final_misclassification.score);
        if (!(final_impurity.score.hard_loss < final_misclassification.score.hard_loss - kEpsUpdate &&
              final_impurity_joint < final_misclassification_joint - kEpsUpdate)) {
            return;
        }

        auto winner_label = [](double lhs, double rhs) {
            if (lhs < rhs - kEpsUpdate) {
                return 1;
            }
            if (rhs < lhs - kEpsUpdate) {
                return 2;
            }
            return 0;
        };

        nlohmann::json trace;
        trace["feature"] = feature;
        trace["groups"] = groups;
        trace["final"] = {
            {"family1", {
                {"hard_loss", final_impurity.score.hard_loss},
                {"hard_impurity", final_impurity.score.hard_impurity},
                {"soft_impurity", final_impurity.score.soft_impurity},
                {"joint_impurity", final_impurity_joint},
                {"boundary_penalty", final_impurity.score.boundary_penalty},
                {"components", final_impurity.score.components},
            }},
            {"family2", {
                {"hard_loss", final_misclassification.score.hard_loss},
                {"hard_impurity", final_misclassification.score.hard_impurity},
                {"soft_impurity", final_misclassification.score.soft_impurity},
                {"joint_impurity", final_misclassification_joint},
                {"boundary_penalty", final_misclassification.score.boundary_penalty},
                {"components", final_misclassification.score.components},
            }},
        };
        trace["raw"] = {
            {"family1", {
                {"hard_loss", raw_impurity.score.hard_loss},
                {"hard_impurity", raw_impurity.score.hard_impurity},
                {"soft_impurity", raw_impurity.score.soft_impurity},
                {"joint_impurity", atomized_joint_impurity(raw_impurity.score)},
                {"boundary_penalty", raw_impurity.score.boundary_penalty},
                {"components", raw_impurity.score.components},
            }},
            {"family2", {
                {"hard_loss", raw_misclassification.score.hard_loss},
                {"hard_impurity", raw_misclassification.score.hard_impurity},
                {"soft_impurity", raw_misclassification.score.soft_impurity},
                {"joint_impurity", atomized_joint_impurity(raw_misclassification.score)},
                {"boundary_penalty", raw_misclassification.score.boundary_penalty},
                {"components", raw_misclassification.score.components},
            }},
        };
        trace["winner"] = {
            {"final_hard_loss", winner_label(final_impurity.score.hard_loss, final_misclassification.score.hard_loss)},
            {"final_joint_impurity", winner_label(final_impurity_joint, final_misclassification_joint)},
            {"raw_hard_loss", winner_label(raw_impurity.score.hard_loss, raw_misclassification.score.hard_loss)},
            {"raw_joint_impurity", winner_label(
                atomized_joint_impurity(raw_impurity.score),
                atomized_joint_impurity(raw_misclassification.score))},
        };
        trace["deltas"] = {
            {"family1_hard_loss", final_impurity.score.hard_loss - raw_impurity.score.hard_loss},
            {"family2_hard_loss", final_misclassification.score.hard_loss - raw_misclassification.score.hard_loss},
            {"family1_joint_impurity", final_impurity_joint - atomized_joint_impurity(raw_impurity.score)},
            {"family2_joint_impurity", final_misclassification_joint - atomized_joint_impurity(raw_misclassification.score)},
        };

        telemetry.family1_hard_loss_inversion_traces.push_back(std::move(trace));
    }

    std::vector<AtomizedCandidate> select_family_nominees(
        AtomizedCandidate impurity,
        AtomizedCandidate misclassification
    ) const {
        std::vector<AtomizedCandidate> selected;
        selected.reserve(2);
        auto &telemetry = const_cast<Solver *>(this)->atomized_telemetry();
        if (!impurity.feasible && !misclassification.feasible) {
            return selected;
        }
        if (!impurity.feasible) {
            ++telemetry.atomized_coarse_pruned_candidates;
            ++telemetry.debr_final_block_wins;
            selected.push_back(std::move(misclassification));
            return selected;
        }
        if (!misclassification.feasible) {
            ++telemetry.atomized_coarse_pruned_candidates;
            ++telemetry.debr_final_geo_wins;
            selected.push_back(std::move(impurity));
            return selected;
        }

        record_family_compare_stats(impurity, misclassification);
        if (atomized_assignment_equivalent(impurity, misclassification)) {
            ++telemetry.family_compare_equivalent;
            ++telemetry.family1_selected_by_equivalence;
            ++telemetry.debr_final_geo_wins;
            ++telemetry.atomized_coarse_pruned_candidates;
            selected.push_back(std::move(impurity));
            return selected;
        }

        if (atomized_candidate_dominates(impurity, misclassification)) {
            ++telemetry.family1_selected_by_dominance;
            ++telemetry.debr_final_geo_wins;
            ++telemetry.atomized_coarse_pruned_candidates;
            selected.push_back(std::move(impurity));
            return selected;
        }
        if (atomized_candidate_dominates(misclassification, impurity)) {
            ++telemetry.family2_selected_by_dominance;
            ++telemetry.debr_final_block_wins;
            ++telemetry.atomized_coarse_pruned_candidates;
            selected.push_back(std::move(misclassification));
            return selected;
        }

        ++telemetry.family_metric_disagreement;
        ++telemetry.family_sent_both;
        selected.push_back(std::move(impurity));
        selected.push_back(std::move(misclassification));
        return selected;
    }

    double effective_sample_unit(const SubproblemStats &stats) const {
        if (sample_weight_uniform_) {
            return 1.0 / static_cast<double>(std::max(1, stats.total_count));
        }
        if (stats.sum_weight <= kEpsUpdate || stats.sum_weight_sq <= kEpsUpdate) {
            return 1.0 / static_cast<double>(std::max(1, stats.total_count));
        }
        const double n_eff = (stats.sum_weight * stats.sum_weight) / stats.sum_weight_sq;
        return 1.0 / std::max(1.0, n_eff);
    }

    double noncontiguous_boundary_penalty(
        int feature,
        const AtomizedAtom &left,
        const AtomizedAtom &right
    ) const {
        if (!teacher_available_) {
            return 0.0;
        }
        const int gap_width = right.bin_value - left.bin_value;
        if (gap_width <= 0) {
            return 0.0;
        }
        const double strength =
            boundary_strength_between_bins(feature, left.bin_value, right.bin_value);
        return strength / static_cast<double>(gap_width);
    }

    double contiguous_boundary_bonus(
        int feature,
        const AtomizedAtom &left,
        const AtomizedAtom &right
    ) const {
        if (feature < 0) {
            return 0.0;
        }
        return noncontiguous_boundary_penalty(feature, left, right);
    }

    bool build_atomized_atoms(
        const OrderedBins &bins,
        int feature,
        std::vector<AtomizedAtom> &atoms,
        double *hard_floor_out = nullptr,
        double *imp_floor_out = nullptr
    ) const {
        atoms.clear();
        if (bins.values.size() <= 1U) {
            return false;
        }

        atoms.reserve(bins.values.size());
        double hard_floor = 0.0;
        double imp_floor = 0.0;
        for (size_t atom_pos = 0; atom_pos < bins.values.size(); ++atom_pos) {
            AtomizedAtom atom;
            atom.atom_pos = (int)atom_pos;
            atom.bin_value = bins.values[atom_pos];
            atom.row_count = (int)bins.members[atom_pos].size();
            if (!binary_mode_) {
                atom.class_weight.assign((size_t)n_classes_, 0.0);
                atom.teacher_class_weight.assign((size_t)n_classes_, 0.0);
            }

            for (int idx : bins.members[atom_pos]) {
                const double w = sample_weight_[(size_t)idx];
                const int label = y_[(size_t)idx];
                if (binary_mode_ && label == 1) {
                    atom.pos_weight += w;
                } else if (binary_mode_) {
                    atom.neg_weight += w;
                } else {
                    atom.class_weight[(size_t)label] += w;
                }
                if (teacher_available_) {
                    if (binary_mode_) {
                        const double teacher_prob = teacher_prob_[(size_t)idx];
                        atom.teacher_pos_weight += w * teacher_prob;
                        atom.teacher_neg_weight += w * (1.0 - teacher_prob);
                    } else {
                        const size_t teacher_base = static_cast<size_t>(idx) * static_cast<size_t>(n_classes_);
                        for (int cls = 0; cls < n_classes_; ++cls) {
                            atom.teacher_class_weight[(size_t)cls] +=
                                w * teacher_prob_flat_[teacher_base + static_cast<size_t>(cls)];
                        }
                    }
                }
            }

            if (binary_mode_) {
                if (!teacher_available_) {
                    atom.teacher_pos_weight = atom.pos_weight;
                    atom.teacher_neg_weight = atom.neg_weight;
                }
                const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
                atom.teacher_prob = (teacher_total > kEpsUpdate) ? (atom.teacher_pos_weight / teacher_total) : 0.5;
                atom.empirical_prediction = (atom.pos_weight >= atom.neg_weight) ? 1 : 0;
                atom.teacher_prediction = (atom.teacher_prob >= 0.5) ? 1 : 0;
            } else {
                if (!teacher_available_) {
                    atom.teacher_class_weight = atom.class_weight;
                }
                atom.empirical_prediction = argmax_index(atom.class_weight);
                atom.teacher_prediction = argmax_index(atom.teacher_class_weight);
            }

            if (binary_mode_) {
                const double total = atom.pos_weight + atom.neg_weight;
                if (total > kEpsUpdate) {
                    hard_floor += total - std::max(atom.pos_weight, atom.neg_weight);
                    const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
                    if (teacher_total > kEpsUpdate) {
                        imp_floor += teacher_total - std::max(atom.teacher_pos_weight, atom.teacher_neg_weight);
                    } else {
                        imp_floor += total - std::max(atom.pos_weight, atom.neg_weight);
                    }
                }
            } else {
                double total = 0.0;
                double best = 0.0;
                for (double value : atom.class_weight) {
                    total += value;
                    best = std::max(best, value);
                }
                if (total > kEpsUpdate) {
                    hard_floor += total - best;
                }
                double teacher_total = 0.0;
                double teacher_best = 0.0;
                for (double value : atom.teacher_class_weight) {
                    teacher_total += value;
                    teacher_best = std::max(teacher_best, value);
                }
                if (teacher_total > kEpsUpdate) {
                    imp_floor += teacher_total - teacher_best;
                } else if (total > kEpsUpdate) {
                    imp_floor += total - best;
                }
            }
            atoms.push_back(std::move(atom));
        }
        if (hard_floor_out != nullptr) {
            *hard_floor_out = hard_floor;
        }
        if (imp_floor_out != nullptr) {
            *imp_floor_out = imp_floor;
        }
        return atoms.size() > 1U;
    }

    static void append_block_atom(
        const AtomizedBlock &block,
        int block_idx,
        std::vector<AtomizedAtom> &block_atoms
    ) {
        AtomizedAtom atom;
        atom.atom_pos = block_idx;
        atom.bin_value = block_idx;
        atom.row_count = block.row_count;
        atom.pos_weight = block.pos_weight;
        atom.neg_weight = block.neg_weight;
        atom.teacher_pos_weight = block.teacher_pos_weight;
        atom.teacher_neg_weight = block.teacher_neg_weight;
        atom.empirical_prediction = block.empirical_prediction;
        atom.teacher_prediction = block.teacher_prediction;
        atom.class_weight = block.class_weight;
        atom.teacher_class_weight = block.teacher_class_weight;
        if (!atom.class_weight.empty()) {
            atom.teacher_prob = 0.5;
        } else {
            const double teacher_total = atom.teacher_pos_weight + atom.teacher_neg_weight;
            atom.teacher_prob = (teacher_total > kEpsUpdate) ? (atom.teacher_pos_weight / teacher_total) : 0.5;
        }
        block_atoms.push_back(std::move(atom));
    }

    static bool has_atomized_block_compression(const std::vector<AtomizedAtom> &atoms) {
        for (size_t i = 1; i < atoms.size(); ++i) {
            if (atoms[i - 1].teacher_prediction == atoms[i].teacher_prediction &&
                atoms[i - 1].empirical_prediction == atoms[i].empirical_prediction) {
                return true;
            }
        }
        return false;
    }

    static void build_atomized_blocks_and_atoms(
        const std::vector<AtomizedAtom> &atoms,
        std::vector<AtomizedBlock> &blocks,
        std::vector<AtomizedAtom> &block_atoms
    ) {
        blocks.clear();
        block_atoms.clear();
        if (atoms.empty()) {
            return;
        }

        blocks.reserve(atoms.size());
        block_atoms.reserve(atoms.size());
        AtomizedBlock current;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (i > 0) {
                const bool same_type =
                    atoms[i - 1].teacher_prediction == atoms[i].teacher_prediction &&
                    atoms[i - 1].empirical_prediction == atoms[i].empirical_prediction;
                if (!same_type) {
                    append_block_atom(current, (int)blocks.size(), block_atoms);
                    blocks.push_back(std::move(current));
                    current = AtomizedBlock{};
                }
            }
            current.atom_positions.push_back(atoms[i].atom_pos);
            current.row_count += atoms[i].row_count;
            current.pos_weight += atoms[i].pos_weight;
            current.neg_weight += atoms[i].neg_weight;
            current.teacher_pos_weight += atoms[i].teacher_pos_weight;
            current.teacher_neg_weight += atoms[i].teacher_neg_weight;
            current.empirical_prediction = atoms[i].empirical_prediction;
            current.teacher_prediction = atoms[i].teacher_prediction;
            if (!atoms[i].class_weight.empty()) {
                if (current.class_weight.empty()) {
                    current.class_weight.assign(atoms[i].class_weight.size(), 0.0);
                    current.teacher_class_weight.assign(atoms[i].teacher_class_weight.size(), 0.0);
                }
                for (size_t cls = 0; cls < atoms[i].class_weight.size(); ++cls) {
                    current.class_weight[cls] += atoms[i].class_weight[cls];
                    current.teacher_class_weight[cls] += atoms[i].teacher_class_weight[cls];
                }
            }
        }

        append_block_atom(current, (int)blocks.size(), block_atoms);
        blocks.push_back(std::move(current));
    }

    AtomizedPrefixes build_atomized_prefixes(const std::vector<AtomizedAtom> &atoms) const {
        AtomizedPrefixes prefix;
        const size_t count = atoms.size();
        prefix.rows.assign(count + 1, 0);
        prefix.pos.assign(count + 1, 0.0);
        prefix.neg.assign(count + 1, 0.0);
        prefix.teacher_pos.assign(count + 1, 0.0);
        prefix.teacher_neg.assign(count + 1, 0.0);
        if (!binary_mode_) {
            prefix.class_weight_prefix.assign((count + 1U) * static_cast<size_t>(n_classes_), 0.0);
            prefix.teacher_class_weight_prefix.assign((count + 1U) * static_cast<size_t>(n_classes_), 0.0);
        }
        for (size_t i = 0; i < count; ++i) {
            prefix.rows[i + 1] = prefix.rows[i] + atoms[i].row_count;
            prefix.pos[i + 1] = prefix.pos[i] + atoms[i].pos_weight;
            prefix.neg[i + 1] = prefix.neg[i] + atoms[i].neg_weight;
            prefix.teacher_pos[i + 1] = prefix.teacher_pos[i] + atoms[i].teacher_pos_weight;
            prefix.teacher_neg[i + 1] = prefix.teacher_neg[i] + atoms[i].teacher_neg_weight;
            if (!binary_mode_) {
                const size_t prev_base = i * static_cast<size_t>(n_classes_);
                const size_t next_base = (i + 1U) * static_cast<size_t>(n_classes_);
                for (int cls = 0; cls < n_classes_; ++cls) {
                    prefix.class_weight_prefix[next_base + static_cast<size_t>(cls)] =
                        prefix.class_weight_prefix[prev_base + static_cast<size_t>(cls)] +
                        atoms[i].class_weight[(size_t)cls];
                    prefix.teacher_class_weight_prefix[next_base + static_cast<size_t>(cls)] =
                        prefix.teacher_class_weight_prefix[prev_base + static_cast<size_t>(cls)] +
                        atoms[i].teacher_class_weight[(size_t)cls];
                }
            }
        }
        return prefix;
    }

    static std::vector<int> lift_block_assignment_to_atoms(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<int> &block_assignment,
        int atom_count
    ) {
        std::vector<int> atom_assignment((size_t)atom_count, -1);
        for (size_t block_idx = 0; block_idx < blocks.size() && block_idx < block_assignment.size(); ++block_idx) {
            const int group_idx = block_assignment[block_idx];
            for (int atom_pos : blocks[block_idx].atom_positions) {
                atom_assignment[(size_t)atom_pos] = group_idx;
            }
        }
        return atom_assignment;
    }

    static bool project_atom_assignment_to_blocks(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<int> &atom_assignment,
        std::vector<int> &block_assignment,
        std::vector<unsigned char> &mixed_block
    ) {
        block_assignment.assign(blocks.size(), -1);
        mixed_block.assign(blocks.size(), 0);
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            const auto &positions = blocks[block_idx].atom_positions;
            if (positions.empty()) {
                return false;
            }
            const int first_group = atom_assignment[(size_t)positions.front()];
            if (first_group < 0) {
                return false;
            }
            block_assignment[block_idx] = first_group;
            for (int atom_pos : positions) {
                if (atom_assignment[(size_t)atom_pos] != first_group) {
                    mixed_block[block_idx] = 1;
                    break;
                }
            }
        }
        return true;
    }

    static std::vector<std::pair<int, int>> build_active_block_windows(
        const std::vector<int> &before,
        const std::vector<int> &after,
        const std::vector<unsigned char> *extra_active = nullptr
    ) {
        std::vector<std::pair<int, int>> windows;
        const int count = (int)after.size();
        if (count <= 0 || before.size() != after.size()) {
            return windows;
        }

        std::vector<unsigned char> active((size_t)count, 0);
        bool any_active = false;
        for (int idx = 0; idx < count; ++idx) {
            const bool changed = before[(size_t)idx] != after[(size_t)idx];
            const bool extra = extra_active != nullptr && idx < (int)extra_active->size() && (*extra_active)[(size_t)idx] != 0;
            const unsigned char is_active = (changed || extra) ? 1U : 0U;
            active[(size_t)idx] = is_active;
            any_active = any_active || (is_active != 0U);
        }
        if (!any_active) {
            return windows;
        }

        int idx = 0;
        while (idx < count) {
            if (!active[(size_t)idx]) {
                ++idx;
                continue;
            }
            int start = idx;
            int end = idx;
            while (end + 1 < count && active[(size_t)(end + 1)]) {
                ++end;
            }
            start = std::max(0, start - 1);
            end = std::min(count - 1, end + 1);
            while (start > 0 && after[(size_t)(start - 1)] == after[(size_t)start]) {
                --start;
            }
            while (end + 1 < count && after[(size_t)(end + 1)] == after[(size_t)end]) {
                ++end;
            }
            if (!windows.empty() && start <= windows.back().second + 1) {
                windows.back().second = std::max(windows.back().second, end);
            } else {
                windows.push_back({start, end});
            }
            idx = end + 1;
        }
        return windows;
    }

    static std::vector<std::pair<int, int>> block_windows_to_atom_windows(
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<std::pair<int, int>> &block_windows
    ) {
        std::vector<std::pair<int, int>> atom_windows;
        atom_windows.reserve(block_windows.size());
        for (const auto &window : block_windows) {
            const auto &left_positions = blocks[(size_t)window.first].atom_positions;
            const auto &right_positions = blocks[(size_t)window.second].atom_positions;
            if (left_positions.empty() || right_positions.empty()) {
                continue;
            }
            atom_windows.push_back({left_positions.front(), right_positions.back()});
        }
        return atom_windows;
    }

    AtomizedScore score_group_assignment(
        int feature,
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<std::vector<int>> &group_atom_positions,
        const std::vector<int> *assignment = nullptr,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        std::vector<double> *branch_hard_losses = nullptr
    ) const {
        AtomizedScore score{0.0, 0.0, 0.0, 0.0, 0.0, 0};
        double kept_adjacency_bonus = 0.0;
        const bool has_adjacency_bonus =
            adjacency_bonus != nullptr && !adjacency_bonus->empty();
        if (branch_hard_losses != nullptr) {
            branch_hard_losses->clear();
            branch_hard_losses->reserve(group_atom_positions.size());
        }
        for (const auto &group : group_atom_positions) {
            if (group.empty()) {
                return AtomizedScore{};
            }
            int row_count = 0;
            double pos_weight = 0.0;
            double neg_weight = 0.0;
            double teacher_pos_weight = 0.0;
            double teacher_neg_weight = 0.0;
            std::vector<double> class_weight;
            std::vector<double> teacher_class_weight;
            if (!binary_mode_) {
                class_weight.assign((size_t)n_classes_, 0.0);
                teacher_class_weight.assign((size_t)n_classes_, 0.0);
            }
            int components = 1;
            int prev_pos = group.front();
            for (size_t idx = 0; idx < group.size(); ++idx) {
                const int atom_pos = group[idx];
                if (idx > 0 && atom_pos != prev_pos + 1) {
                    ++components;
                    score.boundary_penalty += noncontiguous_boundary_penalty(
                        feature,
                        atoms[(size_t)prev_pos],
                        atoms[(size_t)atom_pos]);
                } else if (idx > 0 && has_adjacency_bonus && (size_t)prev_pos < adjacency_bonus->size()) {
                    kept_adjacency_bonus += (*adjacency_bonus)[(size_t)prev_pos];
                }
                const AtomizedAtom &atom = atoms[(size_t)atom_pos];
                row_count += atom.row_count;
                pos_weight += atom.pos_weight;
                neg_weight += atom.neg_weight;
                teacher_pos_weight += atom.teacher_pos_weight;
                teacher_neg_weight += atom.teacher_neg_weight;
                if (!binary_mode_) {
                    for (int cls = 0; cls < n_classes_; ++cls) {
                        class_weight[(size_t)cls] += atom.class_weight[(size_t)cls];
                        teacher_class_weight[(size_t)cls] += atom.teacher_class_weight[(size_t)cls];
                    }
                }
                prev_pos = atom_pos;
            }
            if (row_count < min_child_size_) {
                return AtomizedScore{};
            }
            if (binary_mode_) {
                const double branch_hard_loss = split_leaf_loss(pos_weight, neg_weight);
                score.hard_loss += branch_hard_loss;
                score.soft_loss += split_leaf_loss(teacher_pos_weight, teacher_neg_weight);
                score.hard_impurity += hard_label_impurity(pos_weight, neg_weight);
                score.soft_impurity += hard_label_impurity(teacher_pos_weight, teacher_neg_weight);
                if (branch_hard_losses != nullptr) {
                    branch_hard_losses->push_back(branch_hard_loss);
                }
            } else {
                const double branch_hard_loss = split_leaf_loss(class_weight);
                score.hard_loss += branch_hard_loss;
                score.soft_loss += split_leaf_loss(teacher_class_weight);
                score.hard_impurity += hard_label_impurity(class_weight);
                score.soft_impurity += hard_label_impurity(teacher_class_weight);
                if (branch_hard_losses != nullptr) {
                    branch_hard_losses->push_back(branch_hard_loss);
                }
            }
            score.components += components;
        }
        if (adjacency_bonus != nullptr) {
            score.boundary_penalty += kept_adjacency_bonus - adjacency_bonus_total;
        } else if (feature >= 0 && assignment != nullptr) {
            for (int atom_pos = 1; atom_pos < (int)atoms.size(); ++atom_pos) {
                if ((*assignment)[(size_t)(atom_pos - 1)] != (*assignment)[(size_t)atom_pos]) {
                    score.boundary_penalty -= contiguous_boundary_bonus(
                        feature,
                        atoms[(size_t)(atom_pos - 1)],
                        atoms[(size_t)atom_pos]);
                }
            }
        }
        return score;
    }

    static bool fill_groups_from_assignment(
        const std::vector<int> &assign,
        int groups,
        std::vector<std::vector<int>> &out,
        std::vector<int> &counts
    ) {
        out.resize((size_t)groups);
        counts.assign((size_t)groups, 0);
        for (auto &group : out) {
            group.clear();
        }
        for (int atom_pos = 0; atom_pos < (int)assign.size(); ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            if (group_idx >= 0 && group_idx < groups) {
                ++counts[(size_t)group_idx];
            } else {
                return false;
            }
        }
        for (int group_idx = 0; group_idx < groups; ++group_idx) {
            if (counts[(size_t)group_idx] <= 0) {
                return false;
            }
            out[(size_t)group_idx].reserve((size_t)counts[(size_t)group_idx]);
        }
        for (int atom_pos = 0; atom_pos < (int)assign.size(); ++atom_pos) {
            const int group_idx = assign[(size_t)atom_pos];
            out[(size_t)group_idx].push_back(atom_pos);
        }
        return true;
    }

    AtomizedCandidate candidate_from_assignment(
        int feature,
        const std::vector<AtomizedAtom> &atoms,
        const std::vector<int> &assign,
        int groups,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        bool compute_branch_hard_losses = true,
        AtomizedObjectiveMode objective_mode = AtomizedObjectiveMode::kImpurity
    ) const {
        AtomizedCandidate out;
        if (groups < 2) {
            return out;
        }
        if (binary_mode_) {
            const int atom_count = (int)assign.size();
            std::vector<int> counts((size_t)groups, 0);
            const bool has_adjacency_bonus =
                adjacency_bonus != nullptr && !adjacency_bonus->empty();
            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                if (group_idx < 0 || group_idx >= groups) {
                    return AtomizedCandidate{};
                }
                ++counts[(size_t)group_idx];
            }

            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (counts[(size_t)group_idx] <= 0) {
                    return AtomizedCandidate{};
                }
            }

            std::vector<int> group_rows((size_t)groups, 0);
            std::vector<int> group_last_pos((size_t)groups, -1);
            std::vector<int> group_components((size_t)groups, 0);
            std::vector<double> group_pos((size_t)groups, 0.0);
            std::vector<double> group_neg((size_t)groups, 0.0);
            std::vector<double> group_teacher_pos((size_t)groups, 0.0);
            std::vector<double> group_teacher_neg((size_t)groups, 0.0);
            std::vector<double> branch_hard_losses;
            if (compute_branch_hard_losses) {
                branch_hard_losses.reserve((size_t)groups);
            }
            double kept_adjacency_bonus = 0.0;
            out.score = AtomizedScore{0.0, 0.0, 0.0, 0.0, 0.0, 0};

            for (int atom_pos = 0; atom_pos < atom_count; ++atom_pos) {
                const int group_idx = assign[(size_t)atom_pos];
                const int last_pos = group_last_pos[(size_t)group_idx];
                if (last_pos >= 0) {
                    if (atom_pos != last_pos + 1) {
                        out.score.boundary_penalty += noncontiguous_boundary_penalty(
                            feature,
                            atoms[(size_t)last_pos],
                            atoms[(size_t)atom_pos]);
                        ++group_components[(size_t)group_idx];
                    } else if (has_adjacency_bonus && (size_t)last_pos < adjacency_bonus->size()) {
                        kept_adjacency_bonus += (*adjacency_bonus)[(size_t)last_pos];
                    }
                } else {
                    group_components[(size_t)group_idx] = 1;
                }

                group_last_pos[(size_t)group_idx] = atom_pos;
                const AtomizedAtom &atom = atoms[(size_t)atom_pos];
                group_rows[(size_t)group_idx] += atom.row_count;
                group_pos[(size_t)group_idx] += atom.pos_weight;
                group_neg[(size_t)group_idx] += atom.neg_weight;
                group_teacher_pos[(size_t)group_idx] += atom.teacher_pos_weight;
                group_teacher_neg[(size_t)group_idx] += atom.teacher_neg_weight;
            }

            for (int group_idx = 0; group_idx < groups; ++group_idx) {
                if (group_rows[(size_t)group_idx] < min_child_size_) {
                    return AtomizedCandidate{};
                }
                const double branch_hard_loss = split_leaf_loss(
                    group_pos[(size_t)group_idx],
                    group_neg[(size_t)group_idx]);
                out.score.hard_loss += branch_hard_loss;
                out.score.soft_loss += split_leaf_loss(
                    group_teacher_pos[(size_t)group_idx],
                    group_teacher_neg[(size_t)group_idx]);
                out.score.hard_impurity += hard_label_impurity(
                    group_pos[(size_t)group_idx],
                    group_neg[(size_t)group_idx]);
                out.score.soft_impurity += hard_label_impurity(
                    group_teacher_pos[(size_t)group_idx],
                    group_teacher_neg[(size_t)group_idx]);
                out.score.components += group_components[(size_t)group_idx];
                if (compute_branch_hard_losses) {
                    branch_hard_losses.push_back(branch_hard_loss);
                }
            }

            if (adjacency_bonus != nullptr) {
                out.score.boundary_penalty += kept_adjacency_bonus - adjacency_bonus_total;
            } else if (feature >= 0) {
                for (int atom_pos = 1; atom_pos < atom_count; ++atom_pos) {
                    if (assign[(size_t)(atom_pos - 1)] != assign[(size_t)atom_pos]) {
                        out.score.boundary_penalty -= contiguous_boundary_bonus(
                            feature,
                            atoms[(size_t)(atom_pos - 1)],
                            atoms[(size_t)atom_pos]);
                    }
                }
            }

            out.feasible = true;
            out.feature = feature;
            out.groups = groups;
            out.hard_loss_mode = (objective_mode == AtomizedObjectiveMode::kHardLoss);
            out.assignment = assign;
            if (compute_branch_hard_losses) {
                out.branch_hard_losses = std::move(branch_hard_losses);
            }
            return out;
        }
        std::vector<std::vector<int>> group_atom_positions;
        std::vector<int> group_counts;
        if (!fill_groups_from_assignment(assign, groups, group_atom_positions, group_counts)) {
            return AtomizedCandidate{};
        }
        out.score = score_group_assignment(
            feature,
            atoms,
            group_atom_positions,
            &assign,
            adjacency_bonus,
            adjacency_bonus_total,
            compute_branch_hard_losses ? &out.branch_hard_losses : nullptr);
        if (!std::isfinite(out.score.hard_impurity)) {
            return AtomizedCandidate{};
        }
        out.feasible = true;
        out.feature = feature;
        out.groups = groups;
        out.hard_loss_mode = (objective_mode == AtomizedObjectiveMode::kHardLoss);
        out.assignment = assign;
        return out;
    }

    AtomizedCandidate lift_block_candidate_to_atoms(
        int feature,
        const std::vector<AtomizedBlock> &blocks,
        const std::vector<AtomizedAtom> &atoms,
        const AtomizedCandidate &block_candidate,
        const std::vector<double> *adjacency_bonus = nullptr,
        double adjacency_bonus_total = 0.0,
        bool compute_branch_hard_losses = true,
        AtomizedObjectiveMode objective_mode = AtomizedObjectiveMode::kImpurity
    ) const {
        if (!block_candidate.feasible) {
            return AtomizedCandidate{};
        }
        const std::vector<int> atom_assignment =
            lift_block_assignment_to_atoms(blocks, block_candidate.assignment, (int)atoms.size());
        return candidate_from_assignment(
            feature,
            atoms,
            atom_assignment,
            block_candidate.groups,
            adjacency_bonus,
            adjacency_bonus_total,
            compute_branch_hard_losses,
            objective_mode);
    }
