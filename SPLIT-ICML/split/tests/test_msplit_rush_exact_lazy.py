import json
import numpy as np
import pytest
from pathlib import Path

from split import MSPLIT_RUSHDP
from split import _libgosdt


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
        min_child_size=1,
        max_branching=4,
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


def test_rush_nonzero_regularization_still_produces_valid_bounds():
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
        reg=0.1,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y)

    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert model.objective_ <= model.upper_bound_ + 1e-9
    assert model.objective_ >= model.lower_bound_ - 1e-9


def test_legacy_distilled_geometry_mode_is_rejected():
    with pytest.raises(ValueError, match="approx_distilled_geometry_mode"):
        MSPLIT_RUSHDP(approx_distilled_geometry_mode="legacy_mean_logit").fit(
            np.array([[0], [1]], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )
    with pytest.raises(ValueError, match="approx_distilled_geometry_mode"):
        MSPLIT_RUSHDP(approx_distilled_geometry_mode="teacher_atomcolor").fit(
            np.array([[0], [1]], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )


def test_teacher_guided_atomcolor_debug_finds_noncontiguous_winner():
    z = np.array(
        [
            [0], [0], [1], [1], [1], [2], [2], [2], [3], [3], [3], [3], [4], [4],
            [5], [5], [5], [5], [6], [6], [6], [6], [7], [7], [7], [7], [8], [8],
        ],
        dtype=np.int32,
    )
    y = np.array(
        [
            1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1,
        ],
        dtype=np.int32,
    )
    teacher_logit = np.array(
        [
            2.002392583645255, 0.18851919251246557, -0.6331940901922267, -0.37756350523280824,
            -1.0911461176191954, -1.277680166386608, 0.6304114907682319, 0.5811658124128057,
            1.294558819441117, -0.7546057912599311, 1.689107452443673, -0.2873877078086663,
            1.5744082788445868, -0.4327858471825968, -0.735483292342275, 0.24978537155866684,
            1.0314530848694723, 0.16100957671534466, -0.5855288241233366, -1.341219714076669,
            -1.401520214917428, 0.5026828498748657, 0.989713033285805, -0.1642945926252907,
            -1.0743648582284346, 0.8730421526217066, -1.2803939447145731, -0.7130680950592722,
        ],
        dtype=np.float64,
    )
    teacher_boundary_gain = np.array(
        [[
            0.7320061956565608, 0.6143732469489966, 0.028365365113521057, 0.7192197728267403,
            0.015991729523571974, 0.7579510023564281, 0.512758723262078, 0.9291042207970062,
        ]],
        dtype=np.float64,
    )
    teacher_boundary_cover = np.array(
        [[
            0.16608249672407474, 0.9413172796123832, 0.1666900087671014, 0.44430997880412515,
            0.5302987319478333, 1.0660620807840702, 0.662231842228457, 0.3588645931709322,
        ]],
        dtype=np.float64,
    )
    teacher_boundary_value_jump = np.array(
        [[
            -1.4267738509897323, -0.13504510003701392, -0.7695146401767057, -1.4227417685154136,
            0.25845279091298756, -0.5685494541476426, -1.0298044380114637, -1.0430010800715654,
        ]],
        dtype=np.float64,
    )

    raw = _libgosdt.msplit_debug_teacher_guided_atomcolor_root_feature(
        z=z,
        y=y,
        sample_weight=None,
        feature=0,
        depth_remaining=3,
        full_depth_budget=3,
        lookahead_depth_budget=2,
        regularization=0.0,
        min_child_size=2,
        min_atom_size=16,
        time_limit_seconds=30.0,
        max_branching=3,
        partition_strategy=1,
        approx_feature_scan_limit=1,
        approx_ref_shortlist_enabled=False,
        approx_ref_widen_max=0,
        approx_challenger_sweep_enabled=False,
        approx_challenger_sweep_max_features=0,
        approx_challenger_sweep_max_patch_calls_per_node=0,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.0,
        approx_distilled_max_depth=2,
        approx_distilled_geometry_mode=7,
        approx_score_order_enabled=False,
        teacher_logit=teacher_logit,
        teacher_boundary_gain=teacher_boundary_gain,
        teacher_boundary_cover=teacher_boundary_cover,
        teacher_boundary_value_jump=teacher_boundary_value_jump,
        teacher_boundary_left_delta=None,
        teacher_boundary_right_delta=None,
        teacher_boundary_left_conf=None,
        teacher_boundary_right_conf=None,
    )

    payload = json.loads(raw)
    winners = []
    contiguous = payload.get("contiguous", {})
    if contiguous.get("exact_ready"):
        winners.append(
            (
                float(contiguous["exact_ub"]),
                int(contiguous.get("noncontiguous_group_count", 0)),
                contiguous["group_spans"],
            )
        )
    for candidate in payload.get("candidates", []):
        if candidate.get("exact_ready"):
            winners.append(
                (
                    float(candidate["exact_ub"]),
                    int(candidate.get("noncontiguous_group_count", 0)),
                    candidate["group_spans"],
                )
            )

    assert winners
    winners.sort(key=lambda item: item[0])
    best_exact_ub = winners[0][0]
    best_tied = [item for item in winners if abs(item[0] - best_exact_ub) <= 1e-12]
    assert best_exact_ub < 1.0
    assert any(item[1] > 0 for item in best_tied)
    noncontiguous_winner = next(item for item in best_tied if item[1] > 0)
    assert sum(1 for group in noncontiguous_winner[2] if len(group) > 1) == noncontiguous_winner[1]


def test_distilled_leaf_objective_matches_soft_logloss():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    teacher_logit = np.array([-1.2, 0.7, 1.1, -0.4], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float64)
    alpha = 0.35
    reg = 0.02

    model = MSPLIT_RUSHDP(
        full_depth_budget=1,
        lookahead_depth_budget=1,
        reg=reg,
        min_child_size=1,
        max_branching=4,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=alpha,
    )
    model.fit(X, y, sample_weight=sample_weight, teacher_logit=teacher_logit)

    w = sample_weight / sample_weight.sum()
    q = 1.0 / (1.0 + np.exp(-np.clip(teacher_logit, -35.0, 35.0)))
    t = (1.0 - alpha) * y.astype(np.float64) + alpha * q
    total_w = float(w.sum())
    target_w = float(np.dot(w, t))
    p = np.clip(target_w / total_w, 1e-15, 1.0 - 1e-15)
    expected = reg - (target_w * np.log(p) + (total_w - target_w) * np.log(1.0 - p))

    assert abs(model.objective_ - expected) <= 1e-9
    assert model.tree_ is not None
    assert model.tree_.__class__.__name__ == "MultiLeaf"


def test_teacher_logits_do_not_change_nonapprox_objective():
    rng = np.random.default_rng(17)
    X = rng.integers(0, 6, size=(48, 5), dtype=np.int32)
    y = ((X[:, 0] >= 3).astype(np.int32) ^ (X[:, 1] % 2) ^ (X[:, 2] >= 2)).astype(np.int32)
    teacher_logit_a = (
        0.9 * (X[:, 0] >= 3).astype(np.float64) -
        0.6 * (X[:, 1] % 2).astype(np.float64) +
        0.4 * (X[:, 2] >= 2).astype(np.float64) -
        0.3
    )
    teacher_logit_b = -1.7 * teacher_logit_a + 0.25

    kwargs = dict(
        full_depth_budget=3,
        lookahead_depth_budget=2,
        reg=1e-4,
        min_child_size=3,
        max_branching=3,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.4,
    )

    baseline_model = MSPLIT_RUSHDP(**kwargs)
    baseline_model.fit(X, y)

    teacher_model_a = MSPLIT_RUSHDP(**kwargs)
    teacher_model_a.fit(X, y, teacher_logit=teacher_logit_a)

    teacher_model_b = MSPLIT_RUSHDP(**kwargs)
    teacher_model_b.fit(X, y, teacher_logit=teacher_logit_b)

    assert abs(baseline_model.objective_ - teacher_model_a.objective_) <= 1e-9
    assert abs(baseline_model.objective_ - teacher_model_b.objective_) <= 1e-9
    np.testing.assert_array_equal(baseline_model.predict(X), teacher_model_a.predict(X))
    np.testing.assert_array_equal(baseline_model.predict(X), teacher_model_b.predict(X))


def test_teacher_guided_atomcolor_geometry_keeps_hard_label_leaf_objective():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    teacher_logit_a = np.array([-1.2, 0.7, 1.1, -0.4], dtype=np.float64)
    teacher_logit_b = np.array([2.4, -0.3, 0.2, -1.7], dtype=np.float64)
    reg = 0.02

    kwargs = dict(
        full_depth_budget=1,
        lookahead_depth_budget=1,
        reg=reg,
        min_child_size=1,
        max_branching=3,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.8,
        approx_distilled_geometry_mode="teacher_guided_atomcolor",
    )

    baseline_model = MSPLIT_RUSHDP(**kwargs)
    baseline_model.fit(X, y)

    teacher_model_a = MSPLIT_RUSHDP(**kwargs)
    teacher_model_a.fit(X, y, teacher_logit=teacher_logit_a)

    teacher_model_b = MSPLIT_RUSHDP(**kwargs)
    teacher_model_b.fit(X, y, teacher_logit=teacher_logit_b)

    expected = reg + 0.5
    assert abs(baseline_model.objective_ - expected) <= 1e-9
    assert abs(baseline_model.objective_ - teacher_model_a.objective_) <= 1e-9
    assert abs(baseline_model.objective_ - teacher_model_b.objective_) <= 1e-9
    np.testing.assert_array_equal(baseline_model.predict(X), teacher_model_a.predict(X))
    np.testing.assert_array_equal(baseline_model.predict(X), teacher_model_b.predict(X))


def test_teacher_guided_atomcolor_fit_commits_noncontiguous_root_split():
    bins = np.repeat(np.arange(12, dtype=np.int32), 12)
    X = bins.reshape(-1, 1)
    positive_bins = {1, 2, 5, 8, 9}
    y = np.fromiter((int(int(bin_id) in positive_bins) for bin_id in bins), dtype=np.int32)
    teacher_signal = 2.0 * y.astype(np.float64) - 1.0
    teacher_logit = 2.75 * teacher_signal

    model = MSPLIT_RUSHDP(
        full_depth_budget=2,
        lookahead_depth_budget=1,
        reg=0.0,
        min_child_size=2,
        max_branching=2,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.8,
        approx_distilled_max_depth=1,
        approx_distilled_geometry_mode="teacher_guided_atomcolor",
    )
    model.fit(X, y, teacher_logit=teacher_logit)

    assert model.tree_.feature == 0
    assert model.approx_distilled_shape_candidates_considered_ > 0
    assert model.approx_distilled_shape_nodes_ > 0
    assert abs(model.objective_) <= 1e-12

    child_spans = set(model.tree_.child_spans.values())
    expected_positive = ((1, 2), (5, 5), (8, 9))
    expected_negative = ((0, 0), (3, 4), (6, 7), (10, 11))
    assert expected_positive in child_spans or expected_negative in child_spans
    assert any(len(spans) > 1 for spans in child_spans)
    np.testing.assert_array_equal(model.predict(X), y)


def test_teacher_guided_atomcolor_reports_prep_cache_metrics():
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
    teacher_logit = 1.5 * (2.0 * y.astype(np.float64) - 1.0)

    model = MSPLIT_RUSHDP(
        full_depth_budget=2,
        lookahead_depth_budget=1,
        reg=1e-4,
        min_child_size=1,
        max_branching=3,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.5,
        approx_distilled_max_depth=1,
        approx_distilled_geometry_mode="teacher_guided_atomcolor",
    )
    model.fit(X, y, teacher_logit=teacher_logit)

    assert model.objective_merge_cache_clears_ >= 0
    assert model.objective_merge_cache_skips_ >= 0
    assert model.approx_feature_prep_cache_clears_ >= 0
    assert model.approx_feature_prep_cache_skips_ >= 0


def test_distilled_entropy_lower_bound_prunes_recursive_calls():
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
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    teacher_logit = np.zeros(X.shape[0], dtype=np.float64)
    reg = 0.02

    model = MSPLIT_RUSHDP(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=reg,
        min_child_size=1,
        max_branching=3,
        use_cpp_solver=True,
        approx_distilled_mode=True,
        approx_distilled_alpha=1.0,
        # Keep the entire reachable subtree on the distilled objective so a
        # flat q=0.5 teacher truly makes every split unattractive.
        approx_distilled_max_depth=4,
    )
    model.fit(X, y, teacher_logit=teacher_logit)

    assert abs(model.objective_ - (reg + np.log(2.0))) <= 1e-9
    assert model.expensive_child_calls_ == 0
    assert model.rush_refinement_child_calls_ == 0
    assert model.approx_exactify_triggered_nodes_ == 0
    assert model.tree_.__class__.__name__ == "MultiLeaf"


def test_rush_cache_hit_path_preserves_bounds():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(24, 4), dtype=np.int32)
    y = rng.integers(0, 2, size=24, dtype=np.int32)
    w = rng.random(24) + 0.1

    model = MSPLIT_RUSHDP(
        full_depth_budget=5,
        lookahead_depth_budget=3,
        reg=0.02,
        min_child_size=2,
        max_branching=4,
        use_cpp_solver=True,
    )
    model.fit(X, y, sample_weight=w)

    assert model.dp_cache_hits_ > 0
    assert model.lower_bound_ <= model.upper_bound_ + 1e-9
    assert model.greedy_cache_entries_peak_ >= 0
    assert model.greedy_cache_clears_ >= 0


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
        min_child_size=3,
        max_branching=4,
        use_cpp_solver=True,
    )

    baseline = MSPLIT_RUSHDP(**common)
    baseline.fit(X, y, sample_weight=w)

    explicit_off = MSPLIT_RUSHDP(
        **common,
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
        min_child_size=3,
        max_branching=4,
        use_cpp_solver=True,
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
        min_child_size=2,
        max_branching=4,
        use_cpp_solver=True,
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
    assert model.approx_distilled_mode_active_ in (0, 1)
    assert model.approx_distilled_alpha_ >= 0.0
    assert model.approx_distilled_max_depth_ >= 0
    assert model.approx_distilled_target_type_ >= 0
    assert model.approx_distilled_nodes_considered_ >= 0
    assert model.approx_distilled_nodes_depth0_ >= 0
    assert model.approx_distilled_nodes_depth1_ >= 0
    assert model.approx_distilled_score_order_candidates_ >= 0
    assert model.approx_distilled_shape_nodes_ >= 0
    assert model.approx_distilled_shape_nodes_depth0_ >= 0
    assert model.approx_distilled_shape_nodes_depth1_ >= 0
    assert model.approx_distilled_shape_nodes_depth2_ >= 0
    assert model.approx_distilled_shape_candidates_considered_ >= 0
    assert model.approx_distilled_shape_candidates_solved_ >= 0
    assert model.approx_distilled_shape_beats_contiguous_ >= 0
    assert model.approx_distilled_shape_sec_ >= 0.0


def test_approx_mode_accepts_auto_caps_zero():
    rng = np.random.default_rng(2718)
    X = rng.integers(0, 8, size=(80, 6), dtype=np.int32)
    y = ((X[:, 0] >= 4).astype(np.int32) ^ (X[:, 2] % 2)).astype(np.int32)

    model = MSPLIT_RUSHDP(
        full_depth_budget=4,
        lookahead_depth_budget=2,
        reg=1e-4,
        min_child_size=2,
        max_branching=4,
        use_cpp_solver=True,
        approx_ref_shortlist_enabled=True,
        approx_ref_widen_max=1,
    )
    model.fit(X, y)

    assert model.approx_mode_enabled_ == 1
    assert model.approx_lhat_computed_ in (0, 1)
    assert model.approx_patch_budget_effective_min_ >= 0
    assert model.approx_patch_budget_effective_max_ >= model.approx_patch_budget_effective_min_
    assert model.approx_greedy_patch_calls_ == 0
    assert model.approx_exactify_features_exact_solved_ == 0
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
