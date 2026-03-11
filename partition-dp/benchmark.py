"""Lean benchmark runner for partition-DP solvers."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.font_manager import FontProperties

from binning import build_feature_bins, encode_binary_labels, load_dataset_feature_matrix
from solvers import CaseInput, PreparedCase, SOLVER_DISPLAY, SOLVER_MARKER, SOLVERS, SolverResult, interval_error, prepare_case


def fmt(v: float, digits: int = 6) -> str:
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}g}"


def display_name(solver_id: str) -> str:
    return SOLVER_DISPLAY.get(solver_id, solver_id)


def enabled_solvers(args: argparse.Namespace) -> List[str]:
    out = ["baseline"]
    if args.dl85:
        out.append("dl85_cbb")
    if args.binoct:
        out.append("binoct_flow")
    if args.snip:
        out.append("snip")
    if args.gosdt:
        out.append("gosdt_dpb")
    if args.linear_fallback:
        out.append("linear_fallback")
    out.append("candidate")
    return out


def legend_order(solver_names: Sequence[str]) -> List[str]:
    preferred = [
        "baseline",
        "dl85_cbb",
        "binoct_flow",
        "snip",
        "gosdt_dpb",
        "linear_fallback",
        "candidate",
    ]
    out = [s for s in preferred if s in solver_names]
    out.extend([s for s in solver_names if s not in out])
    return out


def is_valid_solution(res: SolverResult) -> bool:
    return res.status == "accept" and res.k is not None and res.k >= 2 and bool(res.intervals) and math.isfinite(res.raw_best_obj)


def accuracy(prep: PreparedCase, intervals: Sequence[Tuple[int, int]]) -> float:
    total = float(prep.s[prep.B])
    if total <= 0:
        return float("nan")
    err = sum(interval_error(prep.p, prep.n, int(u), int(v)) for u, v in intervals)
    return float(1.0 - float(err) / total)


def run_one_trial(
    feature_bins: Sequence[Tuple[List[int], List[int]]],
    solver_names: Sequence[str],
    m: int,
    lam: float,
    max_branch_n: int,
    ub_best: float,
    dynamic_ub: bool,
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    total_ns = {s: 0 for s in solver_names}
    incumbent = {s: float(ub_best) for s in solver_names}
    best: Dict[str, Tuple[Optional[PreparedCase], Optional[SolverResult]]] = {s: (None, None) for s in solver_names}

    for pos_bins, neg_bins in feature_bins:
        for s in solver_names:
            prep = prepare_case(
                CaseInput(
                    pos_bins=list(pos_bins),
                    neg_bins=list(neg_bins),
                    m=int(m),
                    lam=float(lam),
                    ub_best=float(incumbent[s]),
                    max_branch_n=int(max_branch_n),
                )
            )

            t0 = time.perf_counter_ns()
            res = SOLVERS[s](prep)
            total_ns[s] += int(time.perf_counter_ns() - t0)

            _, cur_res = best[s]
            if res.status == "accept" and (cur_res is None or res.raw_best_obj < cur_res.raw_best_obj):
                best[s] = (prep, res)
            if dynamic_ub and res.status == "accept" and res.raw_best_obj < incumbent[s]:
                incumbent[s] = float(res.raw_best_obj)

    out_time_ms: Dict[str, float] = {}
    out_acc: Dict[str, float] = {}
    for s in solver_names:
        prep, res = best[s]
        if prep is None or res is None or not is_valid_solution(res):
            return None
        out_time_ms[s] = float(total_ns[s]) / 1e6
        out_acc[s] = accuracy(prep, res.intervals)
    return out_time_ms, out_acc


def slope_loglog(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = (x_arr > 0) & (y_arr > 0) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(valid)) < 4:
        return float("nan")

    xv = x_arr[valid]
    yv = y_arr[valid]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]

    # Collapse repeated x-values using median y to stabilize the log-log slope.
    ux: List[float] = []
    uy: List[float] = []
    i = 0
    while i < len(xv):
        j = i + 1
        while j < len(xv) and xv[j] == xv[i]:
            j += 1
        ux.append(float(xv[i]))
        uy.append(float(np.median(yv[i:j])))
        i = j

    if len(ux) < 4:
        return float("nan")

    lx = np.log2(np.asarray(ux, dtype=float))
    ly = np.log10(np.asarray(uy, dtype=float))

    # Robust Theil-Sen slope (median of pairwise slopes).
    pair_slopes: List[float] = []
    n = len(lx)
    for a in range(n - 1):
        dx = lx[a + 1 :] - lx[a]
        dy = ly[a + 1 :] - ly[a]
        good = dx != 0
        if np.any(good):
            pair_slopes.extend((dy[good] / dx[good]).tolist())
    if not pair_slopes:
        return float("nan")
    return float(np.median(np.asarray(pair_slopes, dtype=float)))


def default_beff_path(plot_out: Path) -> Path:
    return plot_out.with_name(plot_out.stem + "_beff" + plot_out.suffix)


def default_acc_path(plot_out: Path) -> Path:
    return plot_out.with_name(plot_out.stem + "_accuracy" + plot_out.suffix)


def default_metrics_path(plot_out: Path) -> Path:
    return plot_out.with_name(plot_out.stem + "_metrics.json")


def setup_log2_x_axis(ax, ticks: Sequence[float]) -> None:
    ax.set_xscale("log", base=2)
    if ticks:
        ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def setup_log10_y_axis_plain(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def snapped_log2_bounds(values: Sequence[float], reference_ticks: Sequence[int]) -> Tuple[float, float]:
    vals = [float(v) for v in values if np.isfinite(v) and v > 0]
    refs = [float(v) for v in reference_ticks if v > 0]
    if not vals or not refs:
        return 32.0, 1024.0
    vmin = min(vals)
    vmax = max(vals)
    lo_candidates = [r for r in refs if r <= vmin]
    hi_candidates = [r for r in refs if r >= vmax]
    lo = lo_candidates[-1] if lo_candidates else refs[0]
    hi = hi_candidates[0] if hi_candidates else refs[-1]
    if hi < lo:
        hi = lo
    return lo, hi


def style_grid(ax) -> None:
    tick_size = 13
    tick_fp = FontProperties(size=tick_size, family="DejaVu Sans")
    ax.tick_params(axis="x", which="major", labelsize=tick_size)
    ax.tick_params(axis="y", which="major", labelsize=tick_size)
    ax.tick_params(axis="x", which="minor", labelsize=tick_size)
    ax.tick_params(axis="y", which="minor", labelsize=tick_size)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(tick_fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(tick_fp)
    # Remove tiny y-axis tick marks for cleaner panels.
    ax.tick_params(axis="y", which="both", length=0)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.32, linewidth=0.7, linestyle="-")
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.45, linestyle=":")


def style_for_solver(solver: str) -> Dict[str, object]:
    colors = {
        "baseline": "#1f77b4",
        "dl85_cbb": "#ff9800",
        "binoct_flow": "#00bcd4",
        "snip": "#f44336",
        "gosdt_dpb": "#7e57c2",
        "linear_fallback": "#00c853",  # Grid-DP
        "candidate": "#e91e63",  # RUSH-DP
    }
    color = colors.get(solver, "#333333")
    # Keep RUSH-DP and Grid-DP visually dominant.
    if solver == "candidate":
        return {
            "linewidth": 2.2,
            "linestyle": "-",
            "color": color,
            "markerfacecolor": color,
            "markeredgewidth": 1.0,
            "zorder": 4,
        }
    if solver == "linear_fallback":
        return {
            "linewidth": 2.0,
            "linestyle": "-",
            "color": color,
            "markerfacecolor": color,
            "markeredgewidth": 1.0,
            "zorder": 4,
        }
    return {
        "linewidth": 1.45,
        "linestyle": "-",
        "color": color,
        "markerfacecolor": "none",
        "markeredgewidth": 1.2,
        "zorder": 3,
    }


def plot_all(
    datasets: Sequence[str],
    b_values: Sequence[int],
    solver_names: Sequence[str],
    runtime_ms: Dict[str, Dict[str, List[float]]],
    acc_mean: Dict[str, Dict[str, List[float]]],
    beff_mean: Dict[str, List[float]],
    out_runtime: Path,
    out_beff: Path,
    out_acc: Path,
    dpi: int,
) -> None:
    n = len(datasets)
    axis_label_size = 13
    panel_w = 5.3
    panel_h = 5.8
    fig_runtime, axes_runtime = plt.subplots(1, n, figsize=(panel_w * n, panel_h), squeeze=False)
    fig_beff, axes_beff = plt.subplots(1, n, figsize=(panel_w * n, panel_h), squeeze=False)
    fig_acc, axes_acc = plt.subplots(1, n, figsize=(panel_w * n, panel_h), squeeze=False)

    runtime_handles: Dict[str, object] = {}
    beff_handles: Dict[str, object] = {}
    acc_handles: Dict[str, object] = {}
    ordered_legend = legend_order(solver_names)
    reference_ticks = [32, 64, 128, 256, 512, 1024]

    for i, ds in enumerate(datasets):
        ax_rt = axes_runtime[0, i]
        ax_bf = axes_beff[0, i]
        ax_ac = axes_acc[0, i]

        x_req = list(b_values)
        x_beff = list(beff_mean[ds])

        for s in solver_names:
            y_rt = runtime_ms[ds][s]
            style = style_for_solver(s)
            line_rt, = ax_rt.plot(
                x_beff,
                y_rt,
                marker=SOLVER_MARKER[s],
                label=SOLVER_DISPLAY[s],
                **style,
            )
            line_bf, = ax_bf.plot(
                x_req,
                y_rt,
                marker=SOLVER_MARKER[s],
                label=SOLVER_DISPLAY[s],
                **style,
            )
            line_ac, = ax_ac.plot(
                x_req,
                acc_mean[ds][s],
                marker=SOLVER_MARKER[s],
                linewidth=2.0 if s == "candidate" else 1.6,
                markerfacecolor=None if s == "candidate" else "none",
                markeredgewidth=1.2 if s != "candidate" else 1.0,
                label=SOLVER_DISPLAY[s],
            )

            if i == 0:
                runtime_handles[s] = line_rt
                beff_handles[s] = line_bf
                acc_handles[s] = line_ac

        rt_lo, rt_hi = snapped_log2_bounds(x_beff, reference_ticks)
        rt_ticks = [t for t in reference_ticks if rt_lo <= t <= rt_hi]
        ax_rt.set_title(ds)
        setup_log2_x_axis(ax_rt, rt_ticks)
        ax_rt.set_xlim(rt_lo, rt_hi)
        setup_log10_y_axis_plain(ax_rt)
        if i == 0:
            ax_rt.set_ylabel(r"Median Runtime (ms) [$\log_{10}$ scale]", fontsize=axis_label_size)
        style_grid(ax_rt)

        bf_lo, bf_hi = snapped_log2_bounds(x_req, reference_ticks)
        bf_ticks = [t for t in reference_ticks if bf_lo <= t <= bf_hi]
        ax_bf.set_title(ds)
        setup_log2_x_axis(ax_bf, bf_ticks)
        ax_bf.set_xlim(bf_lo, bf_hi)
        setup_log10_y_axis_plain(ax_bf)
        if i == 0:
            ax_bf.set_ylabel(r"Median Runtime (ms) [$\log_{10}$ scale]", fontsize=axis_label_size)
        style_grid(ax_bf)

        deltas = [
            abs(acc_mean[ds][s][j] - acc_mean[ds]["baseline"][j])
            for s in solver_names
            for j in range(len(b_values))
            if np.isfinite(acc_mean[ds][s][j]) and np.isfinite(acc_mean[ds]["baseline"][j])
        ]
        delta_med = float(np.median(np.asarray(deltas, dtype=float))) if deltas else 0.0

        ax_ac.set_title(rf"{ds} ($|\Delta\mathrm{{acc}}|\sim{fmt(delta_med, 4)}$)")
        setup_log2_x_axis(ax_ac, bf_ticks)
        ax_ac.set_xlim(bf_lo, bf_hi)
        if i == 0:
            ax_ac.set_ylabel("Mean Accuracy", fontsize=axis_label_size)
        ax_ac.set_ylim(0.0, 1.0)
        style_grid(ax_ac)

    fig_runtime.suptitle("Runtime Comparison of Optimal Interval Partitioning", fontsize=14, fontweight="normal", y=0.985)
    fig_runtime.supxlabel(r"Effective Number of Bins ($B_\text{eff}$) [$\log_2$ scale]", fontsize=axis_label_size)
    fig_runtime.legend(
        [runtime_handles[s] for s in ordered_legend if s in runtime_handles],
        [SOLVER_DISPLAY[s] for s in ordered_legend if s in runtime_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig_runtime.subplots_adjust(top=0.79, bottom=0.16, wspace=0.22)
    out_runtime.parent.mkdir(parents=True, exist_ok=True)
    fig_runtime.savefig(str(out_runtime), dpi=int(dpi))
    plt.close(fig_runtime)

    fig_beff.suptitle("Runtime Comparison of Optimal Interval Partitioning", fontsize=14, fontweight="normal", y=0.985)
    fig_beff.supxlabel(r"Number of Bins (B) [$\log_2$ scale]", fontsize=axis_label_size)
    fig_beff.legend(
        [beff_handles[s] for s in ordered_legend if s in beff_handles],
        [SOLVER_DISPLAY[s] for s in ordered_legend if s in beff_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig_beff.subplots_adjust(top=0.79, bottom=0.16, wspace=0.22)
    out_beff.parent.mkdir(parents=True, exist_ok=True)
    fig_beff.savefig(str(out_beff), dpi=int(dpi))
    plt.close(fig_beff)

    fig_acc.suptitle("Accuracy on OpenML Datasets (mean over shared-valid trials)", fontsize=14, fontweight="normal", y=0.985)
    fig_acc.supxlabel(r"Number of Bins ($B$) [log$_2$ scale]", fontsize=axis_label_size)
    fig_acc.legend(
        [acc_handles[s] for s in ordered_legend if s in acc_handles],
        [SOLVER_DISPLAY[s] for s in ordered_legend if s in acc_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig_acc.subplots_adjust(top=0.79, bottom=0.16, wspace=0.22)
    out_acc.parent.mkdir(parents=True, exist_ok=True)
    fig_acc.savefig(str(out_acc), dpi=int(dpi))
    plt.close(fig_acc)


def run(args: argparse.Namespace) -> int:
    if args.dataset_trials < 1:
        raise ValueError("--dataset-trials must be >= 1")
    if args.min_support < 1:
        raise ValueError("--min-support must be >= 1")
    if args.lambda_val <= 0:
        raise ValueError("--lambda-val must be > 0")
    if args.max_branch_n < 1:
        raise ValueError("--max-branch-n must be >= 1")
    if any(int(v) < 2 for v in args.b_sweep):
        raise ValueError("--b-sweep values must be >= 2")

    solver_names = enabled_solvers(args)
    datasets = list(dict.fromkeys(args.datasets))
    b_values = [int(v) for v in args.b_sweep]

    runtime_ms: Dict[str, Dict[str, List[float]]] = {ds: {s: [] for s in solver_names} for ds in datasets}
    acc_mean: Dict[str, Dict[str, List[float]]] = {ds: {s: [] for s in solver_names} for ds in datasets}
    beff_mean: Dict[str, List[float]] = {ds: [] for ds in datasets}
    trial_metrics: Dict[str, List[Dict[str, object]]] = {ds: [] for ds in datasets}

    print(
        "Dataset runtime mode: "
        f"datasets={datasets}, B_sweep={b_values}, trials={args.dataset_trials}, "
        f"m={args.min_support}, lambda={args.lambda_val}, max_branch_n={args.max_branch_n}, "
        f"ub_best={fmt(args.ub_best)}, binning={args.dataset_binning}, "
        f"dynamic_ub={bool(args.dynamic_ub)}, solvers={solver_names}"
    )

    for dataset_name in datasets:
        X_proc, y_raw = load_dataset_feature_matrix(dataset_name)
        y_bin, y_meta = encode_binary_labels(y_raw)
        n_rows, n_features = X_proc.shape

        print(
            f"\n[{dataset_name}] shape={n_rows}x{n_features} "
            f"label_mode={y_meta['mode']} classes={y_meta['n_classes']} "
            f"pos_label={y_meta['positive_label']} pos={y_meta['positive_count']} neg={y_meta['negative_count']}"
        )

        lgbm_cache: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}

        for B in b_values:
            feature_bins, n_bins_list = build_feature_bins(
                X_proc=X_proc,
                y_bin=y_bin,
                B=int(B),
                dataset_binning=str(args.dataset_binning),
                lgbm_min_data_in_bin=int(args.lgbm_min_data_in_bin),
                lgbm_cache=lgbm_cache,
            )

            beff = float(np.mean(np.asarray(n_bins_list, dtype=float))) if n_bins_list else float(B)
            beff_mean[dataset_name].append(beff)

            time_samples: Dict[str, List[float]] = {s: [] for s in solver_names}
            acc_samples: Dict[str, List[float]] = {s: [] for s in solver_names}
            shared_valid = 0

            for _ in range(int(args.dataset_trials)):
                trial = run_one_trial(
                    feature_bins=feature_bins,
                    solver_names=solver_names,
                    m=int(args.min_support),
                    lam=float(args.lambda_val),
                    max_branch_n=int(args.max_branch_n),
                    ub_best=float(args.ub_best),
                    dynamic_ub=bool(args.dynamic_ub),
                )
                if trial is None:
                    continue

                trial_time_ms, trial_acc = trial
                shared_valid += 1
                for s in solver_names:
                    time_samples[s].append(trial_time_ms[s])
                    acc_samples[s].append(trial_acc[s])

            for s in solver_names:
                runtime_ms[dataset_name][s].append(float(np.median(time_samples[s])) if time_samples[s] else float("nan"))
                acc_mean[dataset_name][s].append(float(np.mean(acc_samples[s])) if acc_samples[s] else float("nan"))
            trial_metrics[dataset_name].append(
                {
                    "requested_B": int(B),
                    "realized_beff_mean": float(beff),
                    "realized_beff_min": int(min(n_bins_list)),
                    "realized_beff_max": int(max(n_bins_list)),
                    "shared_valid_trials": int(shared_valid),
                    "runtime_samples_ms": {s: [float(v) for v in time_samples[s]] for s in solver_names},
                    "runtime_median_ms": {s: float(runtime_ms[dataset_name][s][-1]) for s in solver_names},
                    "accuracy_samples": {s: [float(v) for v in acc_samples[s]] for s in solver_names},
                    "accuracy_mean": {s: float(acc_mean[dataset_name][s][-1]) for s in solver_names},
                }
            )

            baseline_rt = runtime_ms[dataset_name]["baseline"][-1]
            speedups = {
                s: (
                    baseline_rt / runtime_ms[dataset_name][s][-1]
                    if s != "baseline"
                    and np.isfinite(baseline_rt)
                    and np.isfinite(runtime_ms[dataset_name][s][-1])
                    and runtime_ms[dataset_name][s][-1] > 0
                    else float("nan")
                )
                for s in solver_names
            }
            acc_gaps = {
                s: (
                    abs(acc_mean[dataset_name][s][-1] - acc_mean[dataset_name]["baseline"][-1])
                    if s != "baseline"
                    and np.isfinite(acc_mean[dataset_name][s][-1])
                    and np.isfinite(acc_mean[dataset_name]["baseline"][-1])
                    else float("nan")
                )
                for s in solver_names
            }

            rt_text = ", ".join(f"{display_name(s)}={fmt(runtime_ms[dataset_name][s][-1], 4)}ms" for s in solver_names)
            sp_text = ", ".join(f"{display_name(s)}={fmt(speedups[s], 4)}x" for s in solver_names if s != "baseline")
            gap_text = ", ".join(f"{display_name(s)}={fmt(acc_gaps[s], 4)}" for s in solver_names if s != "baseline")
            acc_text = ", ".join(f"{display_name(s)}={fmt(acc_mean[dataset_name][s][-1], 6)}" for s in solver_names)

            print(
                f"  B={B:>4} | B_eff_mean={fmt(beff, 4)} [{min(n_bins_list)}, {max(n_bins_list)}]\n"
                f"    runtime_ms_total_features: {rt_text}\n"
                f"    speedup_vs_baseline: {sp_text}\n"
                f"    acc_gap_vs_baseline: {gap_text}\n"
                f"    accuracy_mean: {acc_text}\n"
                f"    coverage: shared_valid={shared_valid}/{args.dataset_trials}, n_features={n_features}"
            )

    out_runtime = Path(args.plot_out)
    out_beff = Path(args.runtime_beff_plot_out) if args.runtime_beff_plot_out else default_beff_path(out_runtime)
    out_acc = Path(args.accuracy_plot_out) if args.accuracy_plot_out else default_acc_path(out_runtime)
    out_metrics = Path(args.metrics_log_out) if args.metrics_log_out else default_metrics_path(out_runtime)

    slope_summary: Dict[str, Dict[str, Dict[str, float]]] = {
        ds: {
            s: {
                "k_vs_requested_B": float(slope_loglog(b_values, runtime_ms[ds][s])),
                "k_vs_realized_beff": float(slope_loglog(beff_mean[ds], runtime_ms[ds][s])),
            }
            for s in solver_names
        }
        for ds in datasets
    }
    solver_labels = {s: display_name(s) for s in solver_names}
    runtime_ms_by_label = {
        ds: {solver_labels[s]: vals for s, vals in runtime_ms[ds].items()}
        for ds in datasets
    }
    acc_mean_by_label = {
        ds: {solver_labels[s]: vals for s, vals in acc_mean[ds].items()}
        for ds in datasets
    }
    slopes_by_label = {
        ds: {
            solver_labels[s]: vals
            for s, vals in slope_summary[ds].items()
        }
        for ds in datasets
    }
    trial_metrics_by_label: Dict[str, List[Dict[str, object]]] = {}
    for ds in datasets:
        trial_metrics_by_label[ds] = []
        for row in trial_metrics[ds]:
            trial_metrics_by_label[ds].append(
                {
                    **row,
                    "runtime_samples_ms": {
                        solver_labels[s]: v for s, v in row["runtime_samples_ms"].items()
                    },
                    "runtime_median_ms": {
                        solver_labels[s]: v for s, v in row["runtime_median_ms"].items()
                    },
                    "accuracy_samples": {
                        solver_labels[s]: v for s, v in row["accuracy_samples"].items()
                    },
                    "accuracy_mean": {
                        solver_labels[s]: v for s, v in row["accuracy_mean"].items()
                    },
                }
            )

    metrics_payload: Dict[str, object] = {
        "config": {
            "datasets": datasets,
            "b_sweep": b_values,
            "dataset_trials": int(args.dataset_trials),
            "dataset_binning": str(args.dataset_binning),
            "lgbm_min_data_in_bin": int(args.lgbm_min_data_in_bin),
            "min_support": int(args.min_support),
            "lambda_val": float(args.lambda_val),
            "max_branch_n": int(args.max_branch_n),
            "ub_best": float(args.ub_best),
            "dynamic_ub": bool(args.dynamic_ub),
            "solvers": solver_names,
            "solver_display_names": solver_labels,
        },
        "runtime_median_ms": runtime_ms_by_label,
        "accuracy_mean": acc_mean_by_label,
        "realized_beff_mean": beff_mean,
        "slopes": slopes_by_label,
        "per_b_trial_data": trial_metrics_by_label,
    }
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=False)

    plot_all(
        datasets=datasets,
        b_values=b_values,
        solver_names=solver_names,
        runtime_ms=runtime_ms,
        acc_mean=acc_mean,
        beff_mean=beff_mean,
        out_runtime=out_runtime,
        out_beff=out_beff,
        out_acc=out_acc,
        dpi=int(args.plot_dpi),
    )

    print(f"\nSaved runtime plot: {out_runtime}")
    print(f"Saved B_eff runtime plot: {out_beff}")
    print(f"Saved accuracy plot: {out_acc}")
    print(f"Saved metrics log: {out_metrics}")
    return 0


def parse_args() -> argparse.Namespace:
    default_plot_out = Path(__file__).resolve().parent / "results" / "runtime_full_comparison.pdf"
    p = argparse.ArgumentParser(description="Partition DP benchmark")

    p.add_argument("--datasets", nargs="+", default=["electricity", "eye-movements", "eye-state"])
    p.add_argument("--b-sweep", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--dataset-trials", type=int, default=50)

    p.add_argument("--dataset-binning", choices=["equal-width", "lightgbm"], default="equal-width")
    p.add_argument("--lgbm-min-data-in-bin", type=int, default=1)

    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--lambda-val", type=float, default=0.01)
    p.add_argument("--max-branch-n", "--max_branch_n", dest="max_branch_n", type=int, default=6)
    p.add_argument("--ub-best", type=float, default=float("inf"))

    p.add_argument("--linear-fallback", dest="linear_fallback", action="store_true", default=True)
    p.add_argument("--no-linear-fallback", dest="linear_fallback", action="store_false")
    p.add_argument("--dl85", dest="dl85", action="store_true", default=True)
    p.add_argument("--no-dl85", dest="dl85", action="store_false")
    p.add_argument("--binoct", dest="binoct", action="store_true", default=True)
    p.add_argument("--no-binoct", dest="binoct", action="store_false")
    p.add_argument("--snip", dest="snip", action="store_true", default=True)
    p.add_argument("--no-snip", dest="snip", action="store_false")
    p.add_argument("--gosdt", dest="gosdt", action="store_true", default=True)
    p.add_argument("--no-gosdt", dest="gosdt", action="store_false")
    p.add_argument("--dynamic-ub", dest="dynamic_ub", action="store_true", default=True)
    p.add_argument("--no-dynamic-ub", dest="dynamic_ub", action="store_false")

    p.add_argument("--plot-out", type=str, default=str(default_plot_out))
    p.add_argument("--runtime-beff-plot-out", type=str, default="")
    p.add_argument("--accuracy-plot-out", type=str, default="")
    p.add_argument("--metrics-log-out", type=str, default="")
    p.add_argument("--plot-dpi", type=int, default=300)

    return p.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
