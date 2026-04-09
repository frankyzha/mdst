from __future__ import annotations

from pathlib import Path


def _current_solver_source_text():
    root = Path(__file__).resolve().parents[1]
    core_source = root / "src" / "libgosdt" / "src" / "msplit_core.cpp"
    atomized_source = root / "src" / "libgosdt" / "src" / "msplit_atomized.cpp"
    return {
        "core": core_source.read_text(encoding="utf-8"),
        "atomized": atomized_source.read_text(encoding="utf-8"),
    }


def test_current_sources_have_no_force_legacy_or_atom_descent_symbols():
    texts = _current_solver_source_text()
    text = texts["core"] + "\n" + texts["atomized"]

    assert "MSPLIT_FORCE_RUSH_LEGACY" not in text
    assert "ensure_teacher_prior_hierarchy_for_prep_legacy" not in text
    assert "ensure_distilled_legacy_shape_for_prep" not in text
    assert "run_teacher_atom_descent(" not in text
    assert "run_teacher_atom_descent_family(" not in text

    py_source = Path(__file__).resolve().parents[1] / "src" / "split" / "MSPLIT.py"
    py_text = py_source.read_text(encoding="utf-8")
    assert "legacy_mean_logit" not in py_text


def test_current_atomized_source_has_no_objective_to_mis_backdoor_pattern():
    text = _current_solver_source_text()["atomized"]

    forbidden = [
        "lb_mis = child.lb - regularization_",
        "lb_mis = child.lb - lambda",
        "lb_mis = child.lb_obj -",
    ]
    for token in forbidden:
        assert token not in text


def test_current_sources_have_no_legacy_selector_or_path_lp_helpers():
    texts = _current_solver_source_text()
    text = texts["core"] + "\n" + texts["atomized"]

    forbidden = [
        "signature_bound_for_indices",
        "path_bound_for_indices",
        "state_lower_bound_for_indices",
        "tighten_candidate_lower_bound_with_path",
        "initialize_candidate_lower_bounds",
        "candidate_leaf_completion_objective",
        "get_canonical_signature_summary",
        "build_canonical_signature_state",
        "encode_signature_code",
        "CanonicalSignatureState",
        "CanonicalSignatureSummary",
        "profiling_signature_bound_calls",
        "profiling_signature_bound_sec",
        "signature_state_cache_entries",
        "profiling_path_bound_calls",
        "profiling_path_bound_sec",
        "profiling_path_bound_skip_trivial",
        "profiling_path_bound_skip_disabled",
        "profiling_path_bound_skip_small_state",
        "profiling_path_bound_skip_too_many_blocks",
        "profiling_path_bound_skip_large_child",
        "profiling_path_bound_tighten_attempts",
        "profiling_path_bound_tighten_effective",
        "lp_skip_reason_not_promising",
        "lp_skip_reason_depth_gate",
        "lp_skip_reason_tighten_cap",
        "greedy_state_block_count_histogram",
        "per_node_block_count",
    ]
    for token in forbidden:
        assert token not in text


def test_current_atomized_lookahead_boundary_switches_to_greedy_completion():
    text = _current_solver_source_text()["atomized"]

    assert "current_depth < effective_lookahead_depth_" in text
    assert "return greedy_complete_impl(std::move(indices), depth_remaining, true);" in text
    assert "return greedy_complete_impl(std::move(indices), depth_remaining, false);" in text
    assert "use_exact_frontier" in text
    assert "candidate_envelope_score(" in text
    assert "prism_envelope_value(" in text
    assert "prism_order_compare" in text


def test_current_core_default_lookahead_is_dynamic_half_depth():
    text = _current_solver_source_text()["core"]
    assert "std::max(1, (full_depth_budget_ + 1) / 2)" in text
    assert "effective_lookahead_depth_ =" in text
