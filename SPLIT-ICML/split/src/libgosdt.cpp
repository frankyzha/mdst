#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <matrix.hpp>
#include <configuration.hpp>
#include <gosdt.hpp>
#include <dataset.hpp>
#include <msplit.hpp>
#include <string>
#include <vector>
#include <cstring>

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_libgosdt, m) {

    using BoolMatrix = Matrix<bool>;
    using FloatMatrix = Matrix<float>;

    // Input binary matrix class
    py::class_<BoolMatrix>(m, "BoolMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, bool>())
        .def("__getitem__",
                [](const BoolMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](BoolMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](BoolMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(bool),
                        py::format_descriptor<bool>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(bool) * m.n_columns(), sizeof(bool) }
                );
        });

    // float matrix class
    py::class_<FloatMatrix>(m, "FloatMatrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, float>())
        .def("__getitem__",
                [](const FloatMatrix& bm, std::pair<size_t, size_t> tup) {
                    return bm(tup.first, tup.second);
                })
        .def("__setitem__",
                [](FloatMatrix& bm, std::pair<size_t, size_t> tup, bool value) {
                    bm(tup.first, tup.second) = value;
                })
        .def_buffer([](FloatMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(float),
                        py::format_descriptor<float>::format(),
                        2,
                        { m.n_rows(), m.n_columns() },
                        { sizeof(float) * m.n_columns(), sizeof(float) }
                );
        });

    // Configuration class
    py::class_<Configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("regularization",                &Configuration::regularization)
        .def_readwrite("upperbound",                    &Configuration::upperbound_guess)
        .def_readwrite("time_limit",                    &Configuration::time_limit)
        .def_readwrite("worker_limit",                  &Configuration::worker_limit)
        .def_readwrite("model_limit",                   &Configuration::model_limit)
        .def_readwrite("verbose",                       &Configuration::verbose)
        .def_readwrite("diagnostics",                   &Configuration::diagnostics)
        .def_readwrite("depth_budget",                  &Configuration::depth_budget)
        .def_readwrite("reference_LB",                  &Configuration::reference_LB)
        .def_readwrite("look_ahead",                    &Configuration::look_ahead)
        .def_readwrite("similar_support",               &Configuration::similar_support)
        .def_readwrite("cancellation",                  &Configuration::cancellation)
        .def_readwrite("feature_transform",             &Configuration::feature_transform)
        .def_readwrite("rule_list",                     &Configuration::rule_list)
        .def_readwrite("non_binary",                    &Configuration::non_binary)
        .def_readwrite("trace",                         &Configuration::trace)
        .def_readwrite("tree",                          &Configuration::tree)
        .def_readwrite("profile",                       &Configuration::profile)
        .def_readwrite("cart_lookahead_depth",          &Configuration::cart_lookahead_depth)
        .def("__repr__", [](const Configuration& config) { return config.to_json().dump(); })
        // Provides Pickling support for the Configuration class:
        .def(py::pickle(
            // __getstate__
            [](const Configuration& config) {
                // Return a tuple that fully encodes the state of the object
                return py::make_tuple(
                    config.regularization,
                    config.upperbound_guess,
                    config.time_limit,
                    config.worker_limit,
                    config.model_limit,
                    config.verbose,
                    config.diagnostics,
                    config.depth_budget,
                    config.reference_LB,
                    config.look_ahead,
                    config.similar_support,
                    config.cancellation,
                    config.feature_transform,
                    config.rule_list,
                    config.non_binary,
                    config.trace,
                    config.tree,
                    config.profile, 
                    config.cart_lookahead_depth
                );
            },
            // __setstate__
            [](const py::tuple& t) {
                if (t.size() != 18) {
                    throw std::runtime_error("Invalid state!");
                }
                Configuration config;
                config.regularization = t[0].cast<float>();
                config.upperbound_guess = t[1].cast<float>();
                config.time_limit = t[2].cast<unsigned int>();
                config.worker_limit = t[3].cast<unsigned int>();
                config.model_limit = t[4].cast<unsigned int>();
                config.verbose = t[5].cast<bool>();
                config.diagnostics = t[6].cast<bool>();
                config.depth_budget = t[7].cast<unsigned char>();
                config.reference_LB = t[8].cast<bool>();
                config.look_ahead = t[9].cast<bool>();
                config.similar_support = t[10].cast<bool>();
                config.cancellation = t[11].cast<bool>();
                config.feature_transform = t[12].cast<bool>();
                config.rule_list = t[13].cast<bool>();
                config.non_binary = t[14].cast<bool>();
                config.trace = t[15].cast<std::string>();
                config.tree = t[16].cast<std::string>();
                config.profile = t[17].cast<std::string>();
                config.cart_lookahead_depth = t[18].cast<unsigned char>();
                return config;
            }
        ))
        .def("save", &Configuration::save);

    // gosdt::Result Class
    py::class_<gosdt::Result>(m, "GOSDTResult")
        .def(py::init<gosdt::Result>())
        .def_readonly("model",          &gosdt::Result::model)
        .def_readonly("graph_size",     &gosdt::Result::graph_size)
        .def_readonly("n_iterations",   &gosdt::Result::n_iterations)
        .def_readonly("lowerbound",     &gosdt::Result::lower_bound)
        .def_readonly("upperbound",     &gosdt::Result::upper_bound)
        .def_readonly("model_loss",     &gosdt::Result::model_loss)
        .def_readonly("time",           &gosdt::Result::time_elapsed)
        .def_readonly("status",         &gosdt::Result::status)
        .def(py::pickle(
            [](const gosdt::Result& result) {
                return py::make_tuple(
                    result.model,
                    result.graph_size,
                    result.n_iterations,
                    result.lower_bound,
                    result.upper_bound,
                    result.model_loss,
                    result.time_elapsed,
                    result.status
                );
            },
            [](const py::tuple& t) {
                if (t.size() != 8) {
                    throw std::runtime_error("Invalid state!");
                }
                gosdt::Result result;
                result.model = t[0].cast<std::string>();
                result.graph_size = t[1].cast<size_t>();
                result.n_iterations = t[2].cast<size_t>();
                result.lower_bound = t[3].cast<double>();
                result.upper_bound = t[4].cast<double>();
                result.model_loss = t[5].cast<double>();
                result.time_elapsed = t[6].cast<double>();
                result.status = t[7].cast<gosdt::Status>();
                return result;
            }
        ));

    // gosdt::fit function
    m.def("gosdt_fit", &gosdt::fit);
    m.def(
        "msplit_fit",
        [](py::array_t<int, py::array::c_style | py::array::forcecast> z,
           py::array_t<int, py::array::c_style | py::array::forcecast> y,
           py::object sample_weight,
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
           int approx_challenger_sweep_max_patch_calls_per_node) {
            if (z.ndim() != 2) {
                throw std::runtime_error("msplit_fit expects z to be a 2D int array.");
            }
            if (y.ndim() != 1) {
                throw std::runtime_error("msplit_fit expects y to be a 1D int array.");
            }
            if (z.shape(0) != y.shape(0)) {
                throw std::runtime_error("msplit_fit expects z.shape[0] == y.shape[0].");
            }

            const int n_rows = static_cast<int>(z.shape(0));
            const int n_features = static_cast<int>(z.shape(1));

            std::vector<int> z_flat(static_cast<size_t>(n_rows) * static_cast<size_t>(n_features));
            std::memcpy(z_flat.data(), z.data(), z_flat.size() * sizeof(int));

            std::vector<int> y_vec(static_cast<size_t>(n_rows));
            std::memcpy(y_vec.data(), y.data(), y_vec.size() * sizeof(int));

            std::vector<double> sample_weight_vec;
            if (!sample_weight.is_none()) {
                py::array_t<double, py::array::c_style | py::array::forcecast> sw =
                    sample_weight.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (sw.ndim() != 1 || sw.shape(0) != z.shape(0)) {
                    throw std::runtime_error(
                        "msplit_fit expects sample_weight to be None or a 1D float array with shape[0] == z.shape[0].");
                }
                sample_weight_vec.resize(static_cast<size_t>(n_rows));
                std::memcpy(sample_weight_vec.data(), sw.data(), sample_weight_vec.size() * sizeof(double));
            }

            msplit::FitResult solved = msplit::fit(
                z_flat,
                n_rows,
                n_features,
                y_vec,
                sample_weight_vec,
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

            py::dict out;
            out["tree"] = py::str(solved.tree.dump());
            out["lowerbound"] = solved.lowerbound;
            out["upperbound"] = solved.upperbound;
            out["objective"] = solved.objective;
            out["exact_internal_nodes"] = solved.exact_internal_nodes;
            out["greedy_internal_nodes"] = solved.greedy_internal_nodes;
            out["dp_subproblem_calls"] = solved.dp_subproblem_calls;
            out["dp_cache_hits"] = solved.dp_cache_hits;
            out["dp_unique_states"] = solved.dp_unique_states;
            out["dp_cache_profile_enabled"] = solved.dp_cache_profile_enabled;
            out["dp_cache_lookup_calls"] = solved.dp_cache_lookup_calls;
            out["dp_cache_miss_no_bucket"] = solved.dp_cache_miss_no_bucket;
            out["dp_cache_miss_bucket_present"] = solved.dp_cache_miss_bucket_present;
            out["dp_cache_miss_depth_mismatch_only"] = solved.dp_cache_miss_depth_mismatch_only;
            out["dp_cache_miss_indices_mismatch"] = solved.dp_cache_miss_indices_mismatch;
            out["dp_cache_depth_match_candidates"] = solved.dp_cache_depth_match_candidates;
            out["dp_cache_bucket_entries_scanned"] = solved.dp_cache_bucket_entries_scanned;
            out["dp_cache_bucket_max_size"] = solved.dp_cache_bucket_max_size;
            out["greedy_subproblem_calls"] = solved.greedy_subproblem_calls;
            out["greedy_cache_hits"] = solved.greedy_cache_hits;
            out["greedy_unique_states"] = solved.greedy_unique_states;
            out["greedy_cache_entries_peak"] = solved.greedy_cache_entries_peak;
            out["greedy_cache_clears"] = solved.greedy_cache_clears;
            out["dp_interval_evals"] = solved.dp_interval_evals;
            out["greedy_interval_evals"] = solved.greedy_interval_evals;
            out["rush_incumbent_feature_aborts"] = solved.rush_incumbent_feature_aborts;
            out["rush_total_time_sec"] = solved.rush_total_time_sec;
            out["rush_refinement_child_calls"] = solved.rush_refinement_child_calls;
            out["rush_refinement_recursive_calls"] = solved.rush_refinement_recursive_calls;
            out["rush_refinement_recursive_unique_states"] = solved.rush_refinement_recursive_unique_states;
            out["rush_ub_rescue_picks"] = solved.rush_ub_rescue_picks;
            out["rush_global_fallback_picks"] = solved.rush_global_fallback_picks;
            out["rush_profile_enabled"] = solved.rush_profile_enabled;
            out["rush_profile_ub0_ordering_sec"] = solved.rush_profile_ub0_ordering_sec;
            out["rush_profile_exact_lazy_eval_sec"] = solved.rush_profile_exact_lazy_eval_sec;
            out["rush_profile_exact_lazy_eval_exclusive_sec"] = solved.rush_profile_exact_lazy_eval_exclusive_sec;
            out["rush_profile_exact_lazy_eval_sec_depth0"] = solved.rush_profile_exact_lazy_eval_sec_depth0;
            out["rush_profile_exact_lazy_eval_exclusive_sec_depth0"] =
                solved.rush_profile_exact_lazy_eval_exclusive_sec_depth0;
            out["rush_profile_exact_lazy_table_init_sec"] = solved.rush_profile_exact_lazy_table_init_sec;
            out["rush_profile_exact_lazy_dp_recompute_sec"] = solved.rush_profile_exact_lazy_dp_recompute_sec;
            out["rush_profile_exact_lazy_child_solve_sec"] = solved.rush_profile_exact_lazy_child_solve_sec;
            out["rush_profile_exact_lazy_child_solve_sec_depth0"] =
                solved.rush_profile_exact_lazy_child_solve_sec_depth0;
            out["rush_profile_exact_lazy_closure_sec"] = solved.rush_profile_exact_lazy_closure_sec;
            out["rush_profile_exact_lazy_dp_recompute_calls"] = solved.rush_profile_exact_lazy_dp_recompute_calls;
            out["rush_profile_exact_lazy_closure_passes"] = solved.rush_profile_exact_lazy_closure_passes;
            out["interval_refinements_attempted"] = solved.interval_refinements_attempted;
            out["expensive_child_calls"] = solved.expensive_child_calls;
            out["expensive_child_sec"] = solved.expensive_child_sec;
            out["expensive_child_exactify_calls"] = solved.expensive_child_exactify_calls;
            out["expensive_child_exactify_sec"] = solved.expensive_child_exactify_sec;
            out["approx_mode_enabled"] = solved.approx_mode_enabled;
            out["approx_ref_shortlist_enabled"] = solved.approx_ref_shortlist_enabled;
            out["approx_challenger_sweep_enabled"] = solved.approx_challenger_sweep_enabled;
            out["approx_lhat_computed"] = solved.approx_lhat_computed;
            out["approx_greedy_patch_calls"] = solved.approx_greedy_patch_calls;
            out["approx_greedy_patches_applied"] = solved.approx_greedy_patches_applied;
            out["approx_greedy_ub_updates_total"] = solved.approx_greedy_ub_updates_total;
            out["approx_greedy_patch_sec"] = solved.approx_greedy_patch_sec;
            out["approx_exactify_triggered_nodes"] = solved.approx_exactify_triggered_nodes;
            out["approx_exactify_features_exact_solved"] = solved.approx_exactify_features_exact_solved;
            out["approx_exactify_stops_by_separation"] = solved.approx_exactify_stops_by_separation;
            out["approx_exactify_stops_by_cap"] = solved.approx_exactify_stops_by_cap;
            out["approx_exactify_stops_by_ambiguous_empty"] = solved.approx_exactify_stops_by_ambiguous_empty;
            out["approx_exactify_stops_by_no_improve"] = solved.approx_exactify_stops_by_no_improve;
            out["approx_exactify_stops_by_separation_depth0"] = solved.approx_exactify_stops_by_separation_depth0;
            out["approx_exactify_stops_by_separation_depth1"] = solved.approx_exactify_stops_by_separation_depth1;
            out["approx_exactify_stops_by_cap_depth0"] = solved.approx_exactify_stops_by_cap_depth0;
            out["approx_exactify_stops_by_cap_depth1"] = solved.approx_exactify_stops_by_cap_depth1;
            out["approx_exactify_features_exact_solved_depth0"] =
                solved.approx_exactify_features_exact_solved_depth0;
            out["approx_exactify_features_exact_solved_depth1"] =
                solved.approx_exactify_features_exact_solved_depth1;
            out["approx_exactify_set_size_depth0_min"] = solved.approx_exactify_set_size_depth0_min;
            out["approx_exactify_set_size_depth0_mean"] = solved.approx_exactify_set_size_depth0_mean;
            out["approx_exactify_set_size_depth0_max"] = solved.approx_exactify_set_size_depth0_max;
            out["approx_exactify_set_size_depth1_min"] = solved.approx_exactify_set_size_depth1_min;
            out["approx_exactify_set_size_depth1_mean"] = solved.approx_exactify_set_size_depth1_mean;
            out["approx_exactify_set_size_depth1_max"] = solved.approx_exactify_set_size_depth1_max;
            out["approx_exactify_avg_features_per_triggered_node"] =
                solved.approx_exactify_avg_features_per_triggered_node;
            out["approx_exactify_ambiguous_set_size_min"] = solved.approx_exactify_ambiguous_set_size_min;
            out["approx_exactify_ambiguous_set_size_mean"] = solved.approx_exactify_ambiguous_set_size_mean;
            out["approx_exactify_ambiguous_set_size_max"] = solved.approx_exactify_ambiguous_set_size_max;
            out["approx_exactify_ambiguous_set_shrank_steps"] = solved.approx_exactify_ambiguous_set_shrank_steps;
            out["approx_exactify_cap_effective_depth0"] = solved.approx_exactify_cap_effective_depth0;
            out["approx_exactify_cap_effective_depth1"] = solved.approx_exactify_cap_effective_depth1;
            out["approx_challenger_sweep_invocations"] = solved.approx_challenger_sweep_invocations;
            out["approx_challenger_sweep_features_processed"] = solved.approx_challenger_sweep_features_processed;
            out["approx_challenger_sweep_sec"] = solved.approx_challenger_sweep_sec;
            out["approx_challenger_sweep_skipped_large_ambiguous"] =
                solved.approx_challenger_sweep_skipped_large_ambiguous;
            out["approx_challenger_sweep_patch_cap_hit"] =
                solved.approx_challenger_sweep_patch_cap_hit;
            out["approx_uncertainty_triggered_nodes"] = solved.approx_uncertainty_triggered_nodes;
            out["approx_exactify_trigger_rate_depth0"] = solved.approx_exactify_trigger_rate_depth0;
            out["approx_exactify_trigger_rate_depth1"] = solved.approx_exactify_trigger_rate_depth1;
            out["approx_uncertainty_trigger_rate_depth0"] = solved.approx_uncertainty_trigger_rate_depth0;
            out["approx_uncertainty_trigger_rate_depth1"] = solved.approx_uncertainty_trigger_rate_depth1;
            out["approx_eligible_nodes_depth0"] = solved.approx_eligible_nodes_depth0;
            out["approx_eligible_nodes_depth1"] = solved.approx_eligible_nodes_depth1;
            out["approx_exactify_triggered_nodes_depth0"] = solved.approx_exactify_triggered_nodes_depth0;
            out["approx_exactify_triggered_nodes_depth1"] = solved.approx_exactify_triggered_nodes_depth1;
            out["approx_uncertainty_triggered_nodes_depth0"] = solved.approx_uncertainty_triggered_nodes_depth0;
            out["approx_uncertainty_triggered_nodes_depth1"] = solved.approx_uncertainty_triggered_nodes_depth1;
            out["approx_pub_unrefined_cells_on_pub_total"] = solved.approx_pub_unrefined_cells_on_pub_total;
            out["approx_pub_patchable_cells_total"] = solved.approx_pub_patchable_cells_total;
            out["approx_pub_cells_skipped_by_childrows"] = solved.approx_pub_cells_skipped_by_childrows;
            out["approx_nodes_with_patchable_pub"] = solved.approx_nodes_with_patchable_pub;
            out["approx_nodes_with_patch_calls"] = solved.approx_nodes_with_patch_calls;
            out["approx_patch_cell_cache_hits"] = solved.approx_patch_cell_cache_hits;
            out["approx_patch_cell_cache_misses"] = solved.approx_patch_cell_cache_misses;
            out["approx_patch_cache_hit_updates"] = solved.approx_patch_cache_hit_updates;
            out["approx_patch_cache_miss_oracle_calls"] = solved.approx_patch_cache_miss_oracle_calls;
            out["approx_patch_subset_materializations"] = solved.approx_patch_subset_materializations;
            out["approx_patch_skipped_already_tight"] = solved.approx_patch_skipped_already_tight;
            out["approx_patch_skipped_no_possible_improve"] = solved.approx_patch_skipped_no_possible_improve;
            out["approx_patch_skipped_cached"] = solved.approx_patch_skipped_cached;
            out["approx_patch_budget_effective_min"] = solved.approx_patch_budget_effective_min;
            out["approx_patch_budget_effective_avg"] = solved.approx_patch_budget_effective_avg;
            out["approx_patch_budget_effective_max"] = solved.approx_patch_budget_effective_max;
            out["approx_ref_neff_mean"] = solved.approx_ref_neff_mean;
            out["approx_ref_neff_max"] = solved.approx_ref_neff_max;
            out["approx_ref_k0_min"] = solved.approx_ref_k0_min;
            out["approx_ref_k0_mean"] = solved.approx_ref_k0_mean;
            out["approx_ref_k0_max"] = solved.approx_ref_k0_max;
            out["approx_ref_k_final_min"] = solved.approx_ref_k_final_min;
            out["approx_ref_k_final_mean"] = solved.approx_ref_k_final_mean;
            out["approx_ref_k_final_max"] = solved.approx_ref_k_final_max;
            out["approx_ref_k_depth0_mean"] = solved.approx_ref_k_depth0_mean;
            out["approx_ref_k_depth1_mean"] = solved.approx_ref_k_depth1_mean;
            out["approx_ref_widen_count"] = solved.approx_ref_widen_count;
            out["approx_ref_widen_count_depth0"] = solved.approx_ref_widen_count_depth0;
            out["approx_ref_widen_count_depth1"] = solved.approx_ref_widen_count_depth1;
            out["approx_ref_chosen_feature_rank_depth0"] = solved.approx_ref_chosen_feature_rank_depth0;
            out["approx_ref_chosen_feature_rank_depth1"] = solved.approx_ref_chosen_feature_rank_depth1;
            out["approx_ref_chosen_in_initial_shortlist_rate_depth0"] =
                solved.approx_ref_chosen_in_initial_shortlist_rate_depth0;
            out["approx_ref_chosen_in_initial_shortlist_rate_depth1"] =
                solved.approx_ref_chosen_in_initial_shortlist_rate_depth1;
            out["fast100_exactify_nodes_allowed"] = solved.fast100_exactify_nodes_allowed;
            out["fast100_exactify_nodes_skipped_small_support"] =
                solved.fast100_exactify_nodes_skipped_small_support;
            out["fast100_exactify_nodes_skipped_dominant_gain"] =
                solved.fast100_exactify_nodes_skipped_dominant_gain;
            out["depth1_skipped_by_low_global_ambiguity"] =
                solved.depth1_skipped_by_low_global_ambiguity;
            out["depth1_skipped_by_large_gap"] =
                solved.depth1_skipped_by_large_gap;
            out["depth1_exactify_challenger_nodes"] =
                solved.depth1_exactify_challenger_nodes;
            out["depth1_exactified_nodes"] =
                solved.depth1_exactified_nodes;
            out["depth1_exactified_features_mean"] =
                solved.depth1_exactified_features_mean;
            out["depth1_exactified_features_max"] =
                solved.depth1_exactified_features_max;
            out["depth1_teacher_replaced_runnerup"] =
                solved.depth1_teacher_replaced_runnerup;
            out["depth1_teacher_rejected_by_uhat_gate"] =
                solved.depth1_teacher_rejected_by_uhat_gate;
            out["depth1_exactify_set_size_mean"] =
                solved.depth1_exactify_set_size_mean;
            out["depth1_exactify_set_size_max"] =
                solved.depth1_exactify_set_size_max;
            out["fast100_skipped_by_ub_lb_separation"] =
                solved.fast100_skipped_by_ub_lb_separation;
            out["fast100_widen_forbidden_depth_gt0_attempts"] =
                solved.fast100_widen_forbidden_depth_gt0_attempts;
            out["fast100_frontier_size_mean"] = solved.fast100_frontier_size_mean;
            out["fast100_frontier_size_max"] = solved.fast100_frontier_size_max;
            out["fast100_stopped_midloop_separation"] =
                solved.fast100_stopped_midloop_separation;
            out["fast100_M_depth0_mean"] = solved.fast100_M_depth0_mean;
            out["fast100_M_depth0_max"] = solved.fast100_M_depth0_max;
            out["fast100_M_depth1_mean"] = solved.fast100_M_depth1_mean;
            out["fast100_M_depth1_max"] = solved.fast100_M_depth1_max;
            out["fast100_cf_exactify_nodes_depth0"] = solved.fast100_cf_exactify_nodes_depth0;
            out["fast100_cf_exactify_nodes_depth1"] = solved.fast100_cf_exactify_nodes_depth1;
            out["fast100_cf_skipped_agreement"] = solved.fast100_cf_skipped_agreement;
            out["fast100_cf_skipped_small_regret"] = solved.fast100_cf_skipped_small_regret;
            out["fast100_cf_skipped_low_impact"] = solved.fast100_cf_skipped_low_impact;
            out["fast100_cf_frontier_size_mean"] = solved.fast100_cf_frontier_size_mean;
            out["fast100_cf_frontier_size_max"] = solved.fast100_cf_frontier_size_max;
            out["fast100_cf_exactified_features_mean"] = solved.fast100_cf_exactified_features_mean;
            out["fast100_cf_exactified_features_max"] = solved.fast100_cf_exactified_features_max;
            out["rootsafe_exactified_features"] = solved.rootsafe_exactified_features;
            out["rootsafe_root_winner_changed_vs_proxy"] =
                solved.rootsafe_root_winner_changed_vs_proxy;
            out["rootsafe_root_candidates_K"] = solved.rootsafe_root_candidates_K;
            out["fast100_used_lgb_prior_tiebreak"] =
                solved.fast100_used_lgb_prior_tiebreak;
            out["gini_dp_calls_root"] = solved.gini_dp_calls_root;
            out["gini_dp_calls_depth1"] = solved.gini_dp_calls_depth1;
            out["gini_teacher_chosen_depth1"] = solved.gini_teacher_chosen_depth1;
            out["gini_tiebreak_used_in_shortlist"] = solved.gini_tiebreak_used_in_shortlist;
            out["gini_dp_sec"] = solved.gini_dp_sec;
            out["gini_root_k0"] = solved.gini_root_k0;
            out["gini_endpoints_added_root"] = solved.gini_endpoints_added_root;
            out["gini_endpoints_added_depth1"] = solved.gini_endpoints_added_depth1;
            out["gini_endpoints_features_touched_root"] =
                solved.gini_endpoints_features_touched_root;
            out["gini_endpoints_features_touched_depth1"] =
                solved.gini_endpoints_features_touched_depth1;
            out["gini_endpoints_added_per_feature_max"] =
                solved.gini_endpoints_added_per_feature_max;
            out["gini_endpoint_sec"] = solved.gini_endpoint_sec;
            return out;
        },
        py::arg("z"),
        py::arg("y"),
        py::arg("sample_weight") = py::none(),
        py::arg("full_depth_budget"),
        py::arg("lookahead_depth_budget"),
        py::arg("regularization"),
        py::arg("branch_penalty") = 0.0,
        py::arg("min_child_size"),
        py::arg("time_limit_seconds") = 0.0,
        py::arg("max_branching") = 0,
        py::arg("partition_strategy") = 0,
        py::arg("approx_mode") = false,
        py::arg("patch_budget_per_feature") = 12,
        py::arg("exactify_top_m") = 2,
        py::arg("tau_mode") = 1,
        py::arg("approx_feature_scan_limit") = 0,
        py::arg("approx_ref_shortlist_enabled") = true,
        py::arg("approx_ref_widen_max") = 1,
        py::arg("approx_challenger_sweep_enabled") = false,
        py::arg("approx_challenger_sweep_max_features") = 3,
        py::arg("approx_challenger_sweep_max_patch_calls_per_node") = 0);

    // Define Status enum
    py::enum_<gosdt::Status>(m, "Status")
        .value("CONVERGED",         gosdt::Status::CONVERGED)
        .value("TIMEOUT",           gosdt::Status::TIMEOUT)
        .value("NON_CONVERGENCE",   gosdt::Status::NON_CONVERGENCE)
        .value("FALSE_CONVERGENCE", gosdt::Status::FALSE_CONVERGENCE)
        .value("UNINITIALIZED",     gosdt::Status::UNINITIALIZED)
        .export_values();

    // Encoding class for translating between original features and binarized features.
    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&>())
        .def(py::init<const Configuration&, const Matrix<bool>&, const Matrix<float>&, const std::vector<std::set<size_t>>&, const Matrix<bool>&>())
        .def_readonly("n_rows",     &Dataset::m_number_rows)
        .def_readonly("n_features", &Dataset::m_number_features)
        .def_readonly("n_targets",  &Dataset::m_number_targets)
        .def("save",                &Dataset::save);
        // .def_static("load",         &Dataset::load);

}
