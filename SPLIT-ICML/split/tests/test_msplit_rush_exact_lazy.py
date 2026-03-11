import numpy as np
from pathlib import Path

from split import MSPLIT_RUSHDP


def test_rush_exact_lazy_weighted_smoke():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    w = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    model = MSPLIT_RUSHDP(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=0.02,
        branch_penalty=0.0,
        min_child_size=1,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
    )
    model.fit(X, y, sample_weight=w)

    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert isinstance(model.rush_incumbent_feature_aborts_, int)
    assert model.rush_incumbent_feature_aborts_ >= 0
    assert model.rush_total_time_sec_ >= 0.0
    assert model.rush_refinement_child_time_sec_ >= 0.0
    assert 0.0 <= model.rush_refinement_child_time_fraction_ <= 1.0 + 1e-9
    assert model.rush_refinement_child_calls_ >= 0
    assert model.rush_refinement_recursive_calls_ >= model.rush_refinement_child_calls_
    assert model.rush_refinement_recursive_unique_states_ >= 0
    assert model.interval_refinements_attempted_ >= 0
    assert model.expensive_child_calls_ >= 0
    assert model.expensive_child_sec_ >= 0.0
    assert model.rush_feature_logs_root_ == []
    assert model.rush_feature_logs_depth1_ == []
    assert model.rush_refinement_depth_logs_ == []


def test_rush_branch_penalty_nonzero_uses_legacy_path():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)

    model = MSPLIT_RUSHDP(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=0.02,
        branch_penalty=0.1,
        min_child_size=1,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
    )
    model.fit(X, y)

    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert model.rush_incumbent_feature_aborts_ == 0
    assert model.rush_feature_logs_root_ == []
    assert model.rush_feature_logs_depth1_ == []
    assert model.rush_refinement_child_calls_ == 0


def test_rush_cache_hit_path_preserves_bounds():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(24, 4), dtype=np.int32)
    y = rng.integers(0, 2, size=24, dtype=np.int32)
    w = rng.random(24) + 0.1

    model = MSPLIT_RUSHDP(
        full_depth_budget=5,
        lookahead_depth_budget=3,
        reg=0.02,
        branch_penalty=0.0,
        min_child_size=2,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
    )
    model.fit(X, y, sample_weight=w)

    assert model.dp_cache_hits_ > 0
    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert model.greedy_cache_entries_peak_ >= 0
    assert model.greedy_cache_clears_ >= 0


def test_rush_exact_lazy_depth3_parity_with_forced_legacy(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.integers(0, 8, size=(96, 8), dtype=np.int32)
    y = (
        (X[:, 0] % 2)
        ^ (X[:, 1] % 2)
        ^ (X[:, 2] >= 4).astype(np.int32)
        ^ (X[:, 3] >= 4).astype(np.int32)
    ).astype(np.int32)
    y = np.where((X[:, 4] == 0) & (X[:, 5] >= 4), 1 - y, y).astype(np.int32)
    w = (rng.random(96) + 0.2).astype(np.float64)

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=1e-4,
        branch_penalty=0.0,
        min_child_size=4,
        max_branching=3,
        input_is_binned=True,
        use_cpp_solver=True,
        approx_mode=False,
    )

    monkeypatch.delenv("MSPLIT_FORCE_RUSH_LEGACY", raising=False)
    exact_model = MSPLIT_RUSHDP(**kwargs)
    exact_model.fit(X, y, sample_weight=w)

    monkeypatch.setenv("MSPLIT_FORCE_RUSH_LEGACY", "1")
    legacy_model = MSPLIT_RUSHDP(**kwargs)
    legacy_model.fit(X, y, sample_weight=w)

    monkeypatch.delenv("MSPLIT_FORCE_RUSH_LEGACY", raising=False)
    assert abs(exact_model.objective_ - legacy_model.objective_) <= 1e-9

    exact_root_feature = getattr(exact_model.tree_, "feature", None)
    legacy_root_feature = getattr(legacy_model.tree_, "feature", None)
    if exact_root_feature != legacy_root_feature:
        assert abs(exact_model.objective_ - legacy_model.objective_) <= 1e-12


def test_no_objective_to_mis_backdoor_pattern():
    source = Path(__file__).resolve().parents[1] / "src" / "libgosdt" / "src" / "msplit.cpp"
    text = source.read_text(encoding="utf-8")

    forbidden = [
        "lb_mis = child.lb - regularization_",
        "lb_mis = child.lb - lambda",
        "lb_mis = child.lb_obj -",
    ]
    for token in forbidden:
        assert token not in text


def test_approx_prep_builder_does_not_call_ordered_bins():
    source = Path(__file__).resolve().parents[1] / "src" / "libgosdt" / "src" / "msplit.cpp"
    text = source.read_text(encoding="utf-8")
    start = text.find("bool build_approx_feature_prep_from_signatures(")
    assert start >= 0
    end = text.find("static std::vector<std::pair<int, int>> compress_observed_bin_spans_from_hist", start)
    assert end > start
    body = text[start:end]
    assert "build_ordered_bins(" not in body
    assert "std::sort(" not in body


def test_approx_mode_false_exact_path_parity():
    rng = np.random.default_rng(123)
    X = rng.integers(0, 8, size=(80, 6), dtype=np.int32)
    y = (
        (X[:, 0] >= 4).astype(np.int32)
        ^ (X[:, 1] % 2)
        ^ (X[:, 2] >= 3).astype(np.int32)
    ).astype(np.int32)
    w = (rng.random(80) + 0.1).astype(np.float64)

    common = dict(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=1e-4,
        branch_penalty=0.0,
        min_child_size=3,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
    )

    baseline = MSPLIT_RUSHDP(**common)
    baseline.fit(X, y, sample_weight=w)

    explicit_off = MSPLIT_RUSHDP(
        **common,
        approx_mode=False,
        patch_budget_per_feature=12,
        exactify_top_m=2,
        tau_mode="lambda_sqrt_r",
        approx_feature_scan_limit=0,
    )
    explicit_off.fit(X, y, sample_weight=w)

    assert baseline.objective_ == explicit_off.objective_
    assert baseline.lower_bound_ == explicit_off.lower_bound_
    assert baseline.upper_bound_ == explicit_off.upper_bound_
    assert baseline.tree == explicit_off.tree


def test_approx_mode_deterministic_same_seed_config():
    rng = np.random.default_rng(99)
    X = rng.integers(0, 10, size=(120, 8), dtype=np.int32)
    y = (
        (X[:, 0] % 3 == 0).astype(np.int32)
        ^ (X[:, 1] >= 6).astype(np.int32)
        ^ (X[:, 2] % 2)
    ).astype(np.int32)
    w = (rng.random(120) + 0.05).astype(np.float64)

    kwargs = dict(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=1e-4,
        branch_penalty=0.0,
        min_child_size=3,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
        approx_mode=True,
        patch_budget_per_feature=6,
        exactify_top_m=2,
        tau_mode="lambda_sqrt_r",
        approx_feature_scan_limit=0,
    )

    model_a = MSPLIT_RUSHDP(**kwargs)
    model_a.fit(X, y, sample_weight=w)
    model_b = MSPLIT_RUSHDP(**kwargs)
    model_b.fit(X, y, sample_weight=w)

    assert model_a.tree == model_b.tree
    assert model_a.expensive_child_calls_ == model_b.expensive_child_calls_
    assert model_a.approx_greedy_patch_calls_ == model_b.approx_greedy_patch_calls_
    assert model_a.approx_greedy_patches_applied_ == model_b.approx_greedy_patches_applied_
    assert model_a.approx_pub_patchable_cells_total_ == model_b.approx_pub_patchable_cells_total_
    assert model_a.approx_uncertainty_triggered_nodes_ == model_b.approx_uncertainty_triggered_nodes_
    assert model_a.rootsafe_exactified_features_ == model_b.rootsafe_exactified_features_
    assert (
        model_a.rootsafe_root_winner_changed_vs_proxy_
        == model_b.rootsafe_root_winner_changed_vs_proxy_
    )
    assert model_a.rootsafe_root_candidates_K_ == model_b.rootsafe_root_candidates_K_
    assert (
        model_a.fast100_used_lgb_prior_tiebreak_
        == model_b.fast100_used_lgb_prior_tiebreak_
    )
    assert model_a.gini_dp_calls_root_ == model_b.gini_dp_calls_root_
    assert model_a.gini_dp_calls_depth1_ == model_b.gini_dp_calls_depth1_
    assert (
        model_a.gini_teacher_chosen_depth1_
        == model_b.gini_teacher_chosen_depth1_
    )
    assert (
        model_a.gini_tiebreak_used_in_shortlist_
        == model_b.gini_tiebreak_used_in_shortlist_
    )
    assert model_a.gini_dp_sec_ >= 0.0
    assert model_b.gini_dp_sec_ >= 0.0
    assert model_a.gini_root_k0_ == model_b.gini_root_k0_
    assert (
        model_a.depth1_teacher_replaced_runnerup_
        == model_b.depth1_teacher_replaced_runnerup_
    )
    assert (
        model_a.depth1_teacher_rejected_by_uhat_gate_
        == model_b.depth1_teacher_rejected_by_uhat_gate_
    )
    assert (
        model_a.depth1_exactify_set_size_mean_
        == model_b.depth1_exactify_set_size_mean_
    )
    assert (
        model_a.depth1_exactify_set_size_max_
        == model_b.depth1_exactify_set_size_max_
    )
    assert model_a.gini_endpoints_added_root_ == model_b.gini_endpoints_added_root_
    assert model_a.gini_endpoints_added_depth1_ == model_b.gini_endpoints_added_depth1_
    assert (
        model_a.gini_endpoints_features_touched_root_
        == model_b.gini_endpoints_features_touched_root_
    )
    assert (
        model_a.gini_endpoints_features_touched_depth1_
        == model_b.gini_endpoints_features_touched_depth1_
    )
    assert (
        model_a.gini_endpoints_added_per_feature_max_
        == model_b.gini_endpoints_added_per_feature_max_
    )
    assert model_a.gini_endpoint_sec_ >= 0.0
    assert model_b.gini_endpoint_sec_ >= 0.0
    assert abs(model_a.objective_ - model_b.objective_) <= 1e-9


def test_approx_mode_telemetry_fields_exposed():
    rng = np.random.default_rng(314)
    X = rng.integers(0, 6, size=(96, 7), dtype=np.int32)
    y = ((X[:, 0] >= 3).astype(np.int32) ^ (X[:, 4] % 2)).astype(np.int32)

    model = MSPLIT_RUSHDP(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=1e-4,
        branch_penalty=0.0,
        min_child_size=2,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
        approx_mode=True,
    )
    model.fit(X, y)

    assert model.approx_mode_enabled_ == 1
    assert model.approx_challenger_sweep_enabled_ == 0
    assert model.approx_ref_shortlist_enabled_ == 1
    assert model.approx_lhat_computed_ in (0, 1)
    assert model.approx_greedy_patch_calls_ >= 0
    assert model.approx_greedy_patches_applied_ >= 0
    assert model.approx_greedy_patches_applied_ <= model.approx_greedy_patch_calls_
    assert model.approx_greedy_ub_updates_total_ >= model.approx_greedy_patches_applied_
    assert model.approx_greedy_patch_sec_ >= 0.0
    assert model.expensive_child_exactify_calls_ >= 0
    assert model.expensive_child_exactify_sec_ >= 0.0
    assert model.expensive_child_exactify_calls_ <= model.expensive_child_calls_
    assert model.expensive_child_exactify_sec_ <= model.expensive_child_sec_ + 1e-9
    assert model.approx_exactify_triggered_nodes_ >= 0
    assert model.approx_exactify_features_exact_solved_ >= 0
    assert model.approx_exactify_stops_by_separation_ >= 0
    assert model.approx_exactify_stops_by_cap_ >= 0
    assert model.approx_exactify_stops_by_ambiguous_empty_ >= 0
    assert model.approx_exactify_stops_by_no_improve_ >= 0
    assert model.approx_exactify_stops_by_separation_depth0_ >= 0
    assert model.approx_exactify_stops_by_separation_depth1_ >= 0
    assert model.approx_exactify_stops_by_cap_depth0_ >= 0
    assert model.approx_exactify_stops_by_cap_depth1_ >= 0
    assert model.approx_exactify_features_exact_solved_depth0_ >= 0
    assert model.approx_exactify_features_exact_solved_depth1_ >= 0
    assert model.approx_exactify_set_size_depth0_min_ >= 0
    assert model.approx_exactify_set_size_depth0_mean_ >= 0.0
    assert model.approx_exactify_set_size_depth0_max_ >= 0
    assert model.approx_exactify_set_size_depth1_min_ >= 0
    assert model.approx_exactify_set_size_depth1_mean_ >= 0.0
    assert model.approx_exactify_set_size_depth1_max_ >= 0
    assert model.approx_exactify_avg_features_per_triggered_node_ >= 0.0
    assert model.approx_exactify_ambiguous_set_size_min_ >= 0.0
    assert model.approx_exactify_ambiguous_set_size_mean_ >= 0.0
    assert model.approx_exactify_ambiguous_set_size_max_ >= 0
    assert model.approx_exactify_ambiguous_set_shrank_steps_ >= 0
    assert model.approx_exactify_cap_effective_depth0_ >= 0.0
    assert model.approx_exactify_cap_effective_depth1_ >= 0.0
    assert model.approx_challenger_sweep_invocations_ >= 0
    assert model.approx_challenger_sweep_features_processed_ >= 0
    assert model.approx_challenger_sweep_sec_ >= 0.0
    assert model.approx_challenger_sweep_skipped_large_ambiguous_ >= 0
    assert model.approx_challenger_sweep_patch_cap_hit_ >= 0
    assert model.approx_uncertainty_triggered_nodes_ >= 0
    assert model.approx_eligible_nodes_depth0_ >= 0
    assert model.approx_eligible_nodes_depth1_ >= 0
    assert model.approx_exactify_triggered_nodes_depth0_ >= 0
    assert model.approx_exactify_triggered_nodes_depth1_ >= 0
    assert model.approx_uncertainty_triggered_nodes_depth0_ >= 0
    assert model.approx_uncertainty_triggered_nodes_depth1_ >= 0
    assert 0.0 <= model.approx_exactify_trigger_rate_depth0_ <= 1.0 + 1e-12
    assert 0.0 <= model.approx_exactify_trigger_rate_depth1_ <= 1.0 + 1e-12
    assert 0.0 <= model.approx_uncertainty_trigger_rate_depth0_ <= 1.0 + 1e-12
    assert 0.0 <= model.approx_uncertainty_trigger_rate_depth1_ <= 1.0 + 1e-12
    assert model.approx_pub_unrefined_cells_on_pub_total_ >= 0
    assert model.approx_pub_patchable_cells_total_ >= 0
    assert model.approx_pub_cells_skipped_by_childrows_ >= 0
    assert model.approx_nodes_with_patchable_pub_ >= 0
    assert model.approx_nodes_with_patch_calls_ >= 0
    assert model.approx_patch_cell_cache_hits_ >= 0
    assert model.approx_patch_cell_cache_misses_ >= 0
    assert model.approx_patch_cache_hit_updates_ >= 0
    assert model.approx_patch_cache_miss_oracle_calls_ >= 0
    assert model.approx_patch_subset_materializations_ >= 0
    assert model.approx_patch_cache_miss_oracle_calls_ == model.approx_patch_cell_cache_misses_
    assert model.approx_patch_subset_materializations_ == model.approx_patch_cache_miss_oracle_calls_
    assert model.approx_patch_skipped_already_tight_ >= 0
    assert model.approx_patch_skipped_no_possible_improve_ >= 0
    assert model.approx_patch_skipped_cached_ >= 0
    assert model.approx_patch_budget_effective_min_ >= 0
    assert model.approx_patch_budget_effective_avg_ >= 0.0
    assert model.approx_patch_budget_effective_max_ >= 0
    assert model.approx_ref_neff_mean_ >= 0.0
    assert model.approx_ref_neff_max_ >= 0.0
    assert model.approx_ref_k0_min_ >= 0
    assert model.approx_ref_k0_mean_ >= 0.0
    assert model.approx_ref_k0_max_ >= 0
    assert model.approx_ref_k_final_min_ >= 0
    assert model.approx_ref_k_final_mean_ >= 0.0
    assert model.approx_ref_k_final_max_ >= 0
    assert model.approx_ref_k_depth0_mean_ >= 0.0
    assert model.approx_ref_k_depth1_mean_ >= 0.0
    assert model.approx_ref_widen_count_ >= 0
    assert model.approx_ref_widen_count_depth0_ >= 0
    assert model.approx_ref_widen_count_depth1_ >= 0
    assert model.approx_ref_chosen_feature_rank_depth0_ >= 0.0
    assert model.approx_ref_chosen_feature_rank_depth1_ >= 0.0
    assert 0.0 <= model.approx_ref_chosen_in_initial_shortlist_rate_depth0_ <= 1.0 + 1e-12
    assert 0.0 <= model.approx_ref_chosen_in_initial_shortlist_rate_depth1_ <= 1.0 + 1e-12
    assert model.fast100_exactify_nodes_allowed_ >= 0
    assert model.fast100_exactify_nodes_skipped_small_support_ >= 0
    assert model.fast100_exactify_nodes_skipped_dominant_gain_ >= 0
    assert model.depth1_skipped_by_low_global_ambiguity_ >= 0
    assert model.depth1_skipped_by_large_gap_ >= 0
    assert model.depth1_exactify_challenger_nodes_ >= 0
    assert model.depth1_exactified_nodes_ >= 0
    assert model.depth1_exactified_features_mean_ >= 0.0
    assert model.depth1_exactified_features_max_ >= 0
    assert model.depth1_teacher_replaced_runnerup_ >= 0
    assert model.depth1_teacher_rejected_by_uhat_gate_ >= 0
    assert model.depth1_exactify_set_size_mean_ >= 0.0
    assert model.depth1_exactify_set_size_max_ >= 0
    assert model.fast100_skipped_by_ub_lb_separation_ >= 0
    assert model.fast100_widen_forbidden_depth_gt0_attempts_ >= 0
    assert model.fast100_frontier_size_mean_ >= 0.0
    assert model.fast100_frontier_size_max_ >= 0
    assert model.fast100_stopped_midloop_separation_ >= 0
    assert model.fast100_M_depth0_mean_ >= 0.0
    assert model.fast100_M_depth0_max_ >= 0
    assert model.fast100_M_depth1_mean_ >= 0.0
    assert model.fast100_M_depth1_max_ >= 0
    assert model.fast100_cf_exactify_nodes_depth0_ >= 0
    assert model.fast100_cf_exactify_nodes_depth1_ >= 0
    assert model.fast100_cf_skipped_agreement_ >= 0
    assert model.fast100_cf_skipped_small_regret_ >= 0
    assert model.fast100_cf_skipped_low_impact_ >= 0
    assert model.fast100_cf_frontier_size_mean_ >= 0.0
    assert model.fast100_cf_frontier_size_max_ >= 0
    assert model.fast100_cf_exactified_features_mean_ >= 0.0
    assert model.fast100_cf_exactified_features_max_ >= 0
    assert model.rootsafe_exactified_features_ >= 0
    assert model.rootsafe_root_winner_changed_vs_proxy_ in (0, 1)
    assert model.rootsafe_root_candidates_K_ >= 0
    assert model.fast100_used_lgb_prior_tiebreak_ in (0, 1)
    assert model.gini_dp_calls_root_ >= 0
    assert model.gini_dp_calls_depth1_ >= 0
    assert model.gini_teacher_chosen_depth1_ >= 0
    assert model.gini_tiebreak_used_in_shortlist_ >= 0
    assert model.gini_dp_sec_ >= 0.0
    assert model.gini_root_k0_ >= 0
    assert model.gini_endpoints_added_root_ >= 0
    assert model.gini_endpoints_added_depth1_ >= 0
    assert model.gini_endpoints_features_touched_root_ >= 0
    assert model.gini_endpoints_features_touched_depth1_ >= 0
    assert model.gini_endpoints_added_per_feature_max_ >= 0
    assert model.gini_endpoint_sec_ >= 0.0
    # Gini scout must stay off outside FAST100 AUTO.
    assert model.gini_dp_calls_root_ == 0
    assert model.gini_dp_calls_depth1_ == 0
    assert model.gini_teacher_chosen_depth1_ == 0
    assert model.gini_tiebreak_used_in_shortlist_ == 0
    assert model.gini_root_k0_ == 0
    assert model.gini_endpoints_added_root_ == 0
    assert model.gini_endpoints_added_depth1_ == 0
    assert model.gini_endpoints_features_touched_root_ == 0
    assert model.gini_endpoints_features_touched_depth1_ == 0
    assert model.gini_endpoints_added_per_feature_max_ == 0


def test_approx_mode_accepts_auto_caps_zero():
    rng = np.random.default_rng(2718)
    X = rng.integers(0, 8, size=(80, 6), dtype=np.int32)
    y = ((X[:, 0] >= 4).astype(np.int32) ^ (X[:, 2] % 2)).astype(np.int32)

    model = MSPLIT_RUSHDP(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=1e-4,
        branch_penalty=0.0,
        min_child_size=2,
        max_branching=4,
        input_is_binned=True,
        use_cpp_solver=True,
        approx_mode=True,
        patch_budget_per_feature=0,
        exactify_top_m=0,
        approx_ref_shortlist_enabled=True,
        approx_ref_widen_max=1,
    )
    model.fit(X, y)

    assert model.approx_mode_enabled_ == 1
    assert model.approx_lhat_computed_ in (0, 1)
    assert model.approx_patch_budget_effective_min_ >= 0
    assert model.approx_patch_budget_effective_max_ >= model.approx_patch_budget_effective_min_
    assert model.approx_greedy_patch_calls_ == 0
    assert model.approx_ref_widen_count_depth1_ == 0
    assert model.approx_exactify_stops_by_cap_depth1_ == 0
    assert model.depth1_skipped_by_low_global_ambiguity_ >= 0
    assert model.depth1_skipped_by_large_gap_ >= 0
    assert model.depth1_exactify_challenger_nodes_ >= 0
    assert model.depth1_exactified_nodes_ >= 0
    assert model.depth1_exactified_features_mean_ >= 0.0
    assert model.depth1_exactified_features_max_ >= 0
    assert model.depth1_exactified_features_max_ <= 3
    assert model.depth1_teacher_replaced_runnerup_ >= 0
    assert model.depth1_teacher_rejected_by_uhat_gate_ >= 0
    assert model.depth1_exactify_set_size_mean_ >= 0.0
    assert model.depth1_exactify_set_size_max_ <= 3
    assert model.rootsafe_root_candidates_K_ >= 0
    assert model.rootsafe_exactified_features_ >= 0
    if model.rootsafe_root_candidates_K_ > 0:
        assert model.rootsafe_exactified_features_ == model.rootsafe_root_candidates_K_
    assert model.rootsafe_root_winner_changed_vs_proxy_ in (0, 1)
    assert model.fast100_used_lgb_prior_tiebreak_ in (0, 1)
    assert model.fast100_cf_skipped_agreement_ >= 0
    assert model.fast100_cf_exactified_features_max_ >= 0
    assert model.gini_dp_calls_root_ >= 0
    assert model.gini_dp_calls_depth1_ >= 0
    assert model.gini_teacher_chosen_depth1_ >= 0
    assert model.gini_tiebreak_used_in_shortlist_ >= 0
    assert model.gini_dp_sec_ >= 0.0
    assert model.gini_root_k0_ >= 0
    assert model.gini_endpoints_added_root_ >= 0
    assert model.gini_endpoints_added_depth1_ >= 0
    assert model.gini_endpoints_features_touched_root_ >= 0
    assert model.gini_endpoints_features_touched_depth1_ >= 0
    assert model.gini_endpoints_added_per_feature_max_ <= model.max_branching - 1
    assert model.gini_endpoint_sec_ >= 0.0
    assert model.gini_endpoints_added_depth1_ == 0
    assert model.gini_endpoints_features_touched_depth1_ == 0
    assert model.gini_dp_calls_root_ <= model.gini_root_k0_
    assert model.gini_dp_calls_depth1_ <= (
        model.depth1_exactified_nodes_ * max(1, model.approx_ref_k0_max_)
    )
