"""Run LightGBM-binned multi-split SPLIT experiments with VCM-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Force non-interactive backend so this runs cleanly in headless/batch sessions.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dataset import load_electricity, load_eye_movements, load_eye_state
from lightgbm_binning import fit_lightgbm_binner

# Fallback import path if the package is not installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_SRC = PROJECT_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

from split import MSPLIT


PAPER_SOTA = {
    "eye-movements": {2: 0.601, 3: 0.636, 4: 0.662, 5: 0.675, 6: 0.666},
    "electricity": {2: 0.849, 3: 0.826, 4: 0.866, 5: 0.882, 6: 0.888},
    "eye-state": {2: 0.759, 3: 0.763, 4: 0.793, 5: 0.811, 6: 0.817},
}

DATASET_LOADERS = {
    "electricity": load_electricity,
    "eye-movements": load_eye_movements,
    "eye-state": load_eye_state,
}

SEED_COLUMNS = [
    "dataset",
    "depth_budget",
    "lookahead_depth_budget",
    "seed",
    "status",
    "error",
    "class_prevalence",
    "accuracy",
    "balanced_accuracy",
    "trivial_accuracy",
    "fit_time_sec",
    "n_leaves",
    "n_internal",
    "max_arity",
    "exact_internal_nodes",
    "greedy_internal_nodes",
]


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str, log_fp) -> None:
    stamped = f"[{_iso_now()}] {message}"
    print(stamped, flush=True)
    if log_fp is not None:
        log_fp.write(stamped + "\n")
        log_fp.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGBM-binned multi-split SPLIT experiments on OpenML datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS.keys()),
        default=["electricity", "eye-movements", "eye-state"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--depth-budgets",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="Tree depth budgets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds used for train/test splits.",
    )
    parser.add_argument("--max-bins", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-child-size", type=int, default=2)
    parser.add_argument(
        "--max-branching",
        type=int,
        default=0,
        help="0 means no cap in the C++ solver.",
    )
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--branch-penalty", type=float, default=0.0)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lookahead-cap", type=int, default=2)

    parser.add_argument("--lgb-n-estimators", type=int, default=120)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgb-feature-fraction", type=float, default=1.0)
    parser.add_argument("--lgb-bagging-fraction", type=float, default=1.0)
    parser.add_argument("--lgb-bagging-freq", type=int, default=0)
    parser.add_argument("--lgb-max-depth", type=int, default=-1)
    parser.add_argument("--lgb-min-data-in-bin", type=int, default=1)
    parser.add_argument("--lgb-num-threads", type=int, default=1)

    parser.add_argument(
        "--results-root",
        type=str,
        default="results/runs_lightgbm",
        help="Parent directory that will contain per-run folders.",
    )
    parser.add_argument(
        "--stable-results-dir",
        type=str,
        default="results/lightgbm",
        help="Directory for compatibility files with fixed names.",
    )
    parser.add_argument(
        "--openml-cache-dir",
        type=str,
        default="results/openml_cache",
        help="Persistent cache directory for OpenML downloads.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name. If omitted, uses timestamp.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run folder and skip completed trials.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on newly executed trials (0 means no cap).",
    )
    parser.add_argument(
        "--package-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a tar.gz bundle of the run artifacts.",
    )
    parser.add_argument(
        "--include-paper-sota",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay paper SOTA curve in a second plot.",
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        default=None,
        help="Optional destination directory to copy final artifacts for download.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on non-timeout errors.",
    )

    args = parser.parse_args()
    args.depth_budgets = sorted(set(args.depth_budgets))
    args.seeds = sorted(set(args.seeds))
    args.datasets = list(dict.fromkeys(args.datasets))
    return args


def _make_preprocessor(X_train):
    if not hasattr(X_train, "select_dtypes"):
        return Pipeline([("imputer", SimpleImputer(strategy="median"))])

    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _encode_binary_target(y, dataset_name: str, log_fp) -> np.ndarray:
    y_series = pd.Series(y).reset_index(drop=True)
    class_counts = y_series.value_counts(dropna=False)

    if class_counts.shape[0] != 2:
        raise ValueError(
            f"{dataset_name}: expected binary target with exactly 2 classes, "
            f"got {class_counts.shape[0]} classes. counts={class_counts.to_dict()}"
        )

    ordered = list(sorted(class_counts.index.tolist(), key=lambda v: str(v)))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    return y_series.map(mapping).astype(np.int32).to_numpy()


def _run_single_trial(
    X,
    y_bin: np.ndarray,
    depth_budget: int,
    seed: int,
    args: argparse.Namespace,
):
    lookahead = min(args.lookahead_cap, depth_budget - 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_bin,
        test_size=args.test_size,
        random_state=seed,
        stratify=y_bin,
    )

    preprocessor = _make_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_train_proc = np.asarray(X_train_proc, dtype=float)
    X_test_proc = np.asarray(X_test_proc, dtype=float)

    binner = fit_lightgbm_binner(
        X_train_proc,
        y_train,
        max_bins=args.max_bins,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
        n_estimators=args.lgb_n_estimators,
        num_leaves=args.lgb_num_leaves,
        learning_rate=args.lgb_learning_rate,
        feature_fraction=args.lgb_feature_fraction,
        bagging_fraction=args.lgb_bagging_fraction,
        bagging_freq=args.lgb_bagging_freq,
        max_depth=args.lgb_max_depth,
        min_data_in_bin=args.lgb_min_data_in_bin,
        num_threads=args.lgb_num_threads,
    )
    Z_train = binner.transform(X_train_proc)
    Z_test = binner.transform(X_test_proc)

    model = MSPLIT(
        lookahead_depth_budget=lookahead,
        full_depth_budget=depth_budget,
        reg=args.reg,
        branch_penalty=args.branch_penalty,
        max_bins=args.max_bins,
        min_samples_leaf=args.min_samples_leaf,
        min_child_size=args.min_child_size,
        max_branching=args.max_branching,
        time_limit=args.time_limit,
        verbose=False,
        random_state=seed,
        input_is_binned=True,
        use_cpp_solver=True,
    )

    start = time.time()
    model.fit(Z_train, y_train)
    fit_time = time.time() - start

    y_pred = model.predict(Z_test).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_test))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    trivial_acc = float(max(np.mean(y_test == 0), np.mean(y_test == 1)))

    n_leaves = 0
    n_internal = 0
    max_arity = 0
    stack = [model.tree_]
    while stack:
        node = stack.pop()
        if hasattr(node, "children"):
            n_internal += 1
            max_arity = max(max_arity, len(node.children))
            for child in node.children.values():
                stack.append(child)
        else:
            n_leaves += 1

    return {
        "lookahead_depth_budget": int(lookahead),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "trivial_accuracy": trivial_acc,
        "fit_time_sec": fit_time,
        "n_leaves": int(n_leaves),
        "n_internal": int(n_internal),
        "max_arity": int(max_arity),
        "exact_internal_nodes": int(getattr(model, "exact_internal_nodes_", 0)),
        "greedy_internal_nodes": int(getattr(model, "greedy_internal_nodes_", 0)),
    }


def _load_existing_seed_rows(seed_csv_path: Path):
    if not seed_csv_path.exists():
        return [], set()

    seed_df = pd.read_csv(seed_csv_path)
    rows = seed_df.to_dict(orient="records")
    done = set()
    for row in rows:
        if pd.isna(row.get("dataset")) or pd.isna(row.get("depth_budget")) or pd.isna(row.get("seed")):
            continue
        done.add((str(row["dataset"]), int(row["depth_budget"]), int(row["seed"])))
    return rows, done


def _append_seed_row(seed_csv_path: Path, row: dict[str, Any]) -> None:
    row_df = pd.DataFrame([row], columns=SEED_COLUMNS)
    header = not seed_csv_path.exists()
    row_df.to_csv(seed_csv_path, mode="a", header=header, index=False)


def _build_summary(seed_df: pd.DataFrame, datasets: list[str], depth_budgets: list[int]) -> pd.DataFrame:
    summary_rows = []

    ok_df = seed_df[seed_df["status"] == "ok"].copy()
    for dataset_name in datasets:
        for depth_budget in depth_budgets:
            subset = ok_df[(ok_df["dataset"] == dataset_name) & (ok_df["depth_budget"] == depth_budget)]
            if subset.empty:
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "depth_budget": depth_budget,
                        "lookahead_depth_budget": min(depth_budget - 1, int(seed_df["lookahead_depth_budget"].max()))
                        if not seed_df.empty
                        else min(depth_budget - 1, 2),
                        "n_success": 0,
                        "mean_accuracy": np.nan,
                        "mean_balanced_accuracy": np.nan,
                        "mean_trivial_accuracy": np.nan,
                        "std_accuracy": np.nan,
                        "mean_fit_time_sec": np.nan,
                        "mean_n_leaves": np.nan,
                        "mean_n_internal": np.nan,
                        "mean_max_arity": np.nan,
                        "mean_exact_internal_nodes": np.nan,
                        "mean_greedy_internal_nodes": np.nan,
                    }
                )
                continue

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "depth_budget": depth_budget,
                    "lookahead_depth_budget": int(subset["lookahead_depth_budget"].iloc[0]),
                    "n_success": int(subset.shape[0]),
                    "mean_accuracy": float(np.nanmean(subset["accuracy"])),
                    "mean_balanced_accuracy": float(np.nanmean(subset["balanced_accuracy"])),
                    "mean_trivial_accuracy": float(np.nanmean(subset["trivial_accuracy"])),
                    "std_accuracy": float(np.nanstd(subset["accuracy"], ddof=1)) if subset.shape[0] > 1 else 0.0,
                    "mean_fit_time_sec": float(np.nanmean(subset["fit_time_sec"])),
                    "mean_n_leaves": float(np.nanmean(subset["n_leaves"])),
                    "mean_n_internal": float(np.nanmean(subset["n_internal"])),
                    "mean_max_arity": float(np.nanmean(subset["max_arity"])),
                    "mean_exact_internal_nodes": float(np.nanmean(subset["exact_internal_nodes"])),
                    "mean_greedy_internal_nodes": float(np.nanmean(subset["greedy_internal_nodes"])),
                }
            )

    return pd.DataFrame(summary_rows).sort_values(["dataset", "depth_budget"]).reset_index(drop=True)


def _plot_accuracy(
    summary_df: pd.DataFrame,
    datasets: list[str],
    depth_budgets: list[int],
    out_path: Path,
    include_paper_sota: bool,
    dpi: int = 220,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, datasets):
        subset = summary_df[summary_df["dataset"] == dataset_name]
        depths = subset["depth_budget"].to_numpy(dtype=int)
        ours = subset["mean_accuracy"].to_numpy(dtype=float)

        ax.plot(
            depths,
            ours * 100.0,
            marker="o",
            linewidth=2,
            color="#0a9396",
            label="MSPLIT + LightGBM bins",
        )

        if include_paper_sota and dataset_name in PAPER_SOTA:
            paper = np.array([PAPER_SOTA[dataset_name].get(int(d), np.nan) for d in depths], dtype=float)
            ax.plot(
                depths,
                paper * 100.0,
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color="#ae2012",
                label="Paper SOTA",
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("Depth budget")
        ax.set_xticks(depth_budgets)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Test accuracy (%)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_depth_log(
    summary_df: pd.DataFrame,
    out_path: Path,
    datasets: list[str],
    args: argparse.Namespace,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("MSPLIT (LightGBM bins + C++ DP/lookahead) depth vs. accuracy\n")
        f.write(
            "CONFIG: "
            f"SEEDS={args.seeds}, TEST_SIZE={args.test_size}, MAX_BINS={args.max_bins}, "
            f"MIN_SAMPLES_LEAF={args.min_samples_leaf}, MIN_CHILD_SIZE={args.min_child_size}, "
            f"MAX_BRANCHING={args.max_branching}, REG={args.reg}, "
            f"BRANCH_PENALTY={args.branch_penalty}, LOOKAHEAD_CAP={args.lookahead_cap}, "
            f"TIME_LIMIT={args.time_limit}, "
            f"LGB_N_EST={args.lgb_n_estimators}, LGB_NUM_LEAVES={args.lgb_num_leaves}, "
            f"LGB_LR={args.lgb_learning_rate}, LGB_FEATURE_FRAC={args.lgb_feature_fraction}, "
            f"LGB_BAGGING_FRAC={args.lgb_bagging_fraction}, LGB_BAGGING_FREQ={args.lgb_bagging_freq}, "
            f"LGB_MAX_DEPTH={args.lgb_max_depth}, LGB_MIN_DATA_IN_BIN={args.lgb_min_data_in_bin}, "
            f"LGB_THREADS={args.lgb_num_threads}\n\n"
        )
        for dataset_name in datasets:
            f.write(f"Dataset: {dataset_name}\n")
            subset = summary_df[summary_df["dataset"] == dataset_name]
            for _, row in subset.iterrows():
                depth = int(row["depth_budget"])
                mean_acc = float(row["mean_accuracy"])
                mean_bal_acc = float(row["mean_balanced_accuracy"])
                mean_trivial_acc = float(row["mean_trivial_accuracy"])
                std_acc = float(row["std_accuracy"])
                n_success = int(row["n_success"])

                if dataset_name in PAPER_SOTA and depth in PAPER_SOTA[dataset_name]:
                    paper_acc = float(PAPER_SOTA[dataset_name][depth])
                    delta = mean_acc - paper_acc
                    paper_text = (
                        f", paper={paper_acc:.4f} ({paper_acc*100:.2f}%), "
                        f"delta={delta:+.4f} ({delta*100:+.2f} pp)"
                    )
                else:
                    paper_text = ""

                f.write(
                    f"Depth {depth}: n_success={n_success}, "
                    f"ours={mean_acc:.4f} ({mean_acc*100:.2f}%), "
                    f"bal_acc={mean_bal_acc:.4f} ({mean_bal_acc*100:.2f}%), "
                    f"trivial_acc={mean_trivial_acc:.4f} ({mean_trivial_acc*100:.2f}%), "
                    f"std={std_acc:.4f}, "
                    f"mean_leaves={float(row['mean_n_leaves']):.2f}, "
                    f"mean_max_arity={float(row['mean_max_arity']):.2f}, "
                    f"exact_nodes={float(row['mean_exact_internal_nodes']):.2f}, "
                    f"greedy_nodes={float(row['mean_greedy_internal_nodes']):.2f}"
                    f"{paper_text}\n"
                )
            f.write("\n")


def _configure_openml_cache(cache_dir: Path, log_fp) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import openml

        if hasattr(openml, "config") and hasattr(openml.config, "set_root_cache_directory"):
            openml.config.set_root_cache_directory(str(cache_dir))
            _log(f"OpenML cache directory set to {cache_dir}", log_fp)
        else:
            _log("OpenML cache directory API not available; using default openml cache.", log_fp)
    except Exception as exc:  # pragma: no cover - defensive logging only
        _log(f"[warning] Unable to configure OpenML cache directory: {exc}", log_fp)


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_manifest(
    manifest_path: Path,
    run_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    artifact_paths: dict[str, Path],
    start_time: float,
    end_time: float,
) -> None:
    artifacts = []
    for label, path in artifact_paths.items():
        if not path.exists():
            continue
        artifacts.append(
            {
                "label": label,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
        )

    manifest = {
        "created_at": _iso_now(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "command": " ".join(sys.argv),
        "elapsed_sec": end_time - start_time,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "artifacts": artifacts,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _create_artifact_bundle(archive_path: Path, files: list[Path]) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=path.name)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = _parse_args()

    run_name = args.run_name or datetime.now().strftime("run_lightgbm_%Y%m%d_%H%M%S")
    runs_root = (PROJECT_ROOT / args.results_root).resolve()
    run_dir = runs_root / run_name

    if run_dir.exists() and not args.resume:
        raise RuntimeError(
            f"Run directory already exists: {run_dir}. Use --resume or pick --run-name."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    stable_results_dir = (PROJECT_ROOT / args.stable_results_dir).resolve()
    stable_results_dir.mkdir(parents=True, exist_ok=True)

    stdout_log_path = run_dir / "run_stdout.log"
    with stdout_log_path.open("a", encoding="utf-8") as log_fp:
        start_time = time.time()
        _log("Starting LightGBM multisplit experiment run.", log_fp)
        _log(f"run_name={run_name}", log_fp)
        _log(f"run_dir={run_dir}", log_fp)

        _configure_openml_cache((PROJECT_ROOT / args.openml_cache_dir).resolve(), log_fp)

        config_path = run_dir / "config.json"
        config_payload = {
            "created_at": _iso_now(),
            "command": " ".join(sys.argv),
            "config": vars(args),
        }
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

        seed_csv_path = run_dir / "seed_results.csv"
        summary_csv_path = run_dir / "summary_results.csv"
        depth_log_path = run_dir / "multisplit_lightgbm_depth_vs_accuracy.log"
        plot_path = run_dir / "multisplit_lightgbm_dp_accuracy.png"
        plot_vs_paper_path = run_dir / "multisplit_lightgbm_dp_vs_paper_accuracy.png"

        seed_rows, completed_keys = _load_existing_seed_rows(seed_csv_path) if args.resume else ([], set())
        if seed_rows:
            _log(
                f"Resuming from existing seed results: {seed_csv_path} (rows={len(seed_rows)})",
                log_fp,
            )

        executed_trials = 0

        for dataset_name in args.datasets:
            loader = DATASET_LOADERS[dataset_name]
            _log(f"Loading dataset={dataset_name}", log_fp)
            X, y = loader()
            y_bin = _encode_binary_target(y, dataset_name, log_fp)
            class_prevalence = float(max(np.mean(y_bin == 0), np.mean(y_bin == 1)))
            _log(f"dataset={dataset_name}, class_prevalence(max class)={class_prevalence:.4f}", log_fp)

            for depth_budget in args.depth_budgets:
                lookahead = min(args.lookahead_cap, depth_budget - 1)
                _log(
                    f"dataset={dataset_name}, depth={depth_budget}, lookahead={lookahead}: running seeds {args.seeds}",
                    log_fp,
                )
                for seed in args.seeds:
                    key = (dataset_name, int(depth_budget), int(seed))
                    if key in completed_keys:
                        _log(
                            f"[resume] skipping completed trial dataset={dataset_name}, depth={depth_budget}, seed={seed}",
                            log_fp,
                        )
                        continue

                    row = {
                        "dataset": dataset_name,
                        "depth_budget": int(depth_budget),
                        "lookahead_depth_budget": int(lookahead),
                        "seed": int(seed),
                        "status": "ok",
                        "error": "",
                        "class_prevalence": class_prevalence,
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "trivial_accuracy": np.nan,
                        "fit_time_sec": np.nan,
                        "n_leaves": np.nan,
                        "n_internal": np.nan,
                        "max_arity": np.nan,
                        "exact_internal_nodes": np.nan,
                        "greedy_internal_nodes": np.nan,
                    }

                    try:
                        metrics = _run_single_trial(X, y_bin, depth_budget=depth_budget, seed=seed, args=args)
                        row.update(metrics)
                    except (TimeoutError, RuntimeError) as exc:
                        err_text = str(exc)
                        if isinstance(exc, RuntimeError) and "time_limit" not in err_text.lower():
                            row["status"] = "error"
                            row["error"] = err_text
                            if args.fail_fast:
                                raise
                        else:
                            row["status"] = "timeout"
                            row["error"] = err_text
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = repr(exc)
                        if args.fail_fast:
                            raise

                    seed_rows.append(row)
                    completed_keys.add(key)
                    _append_seed_row(seed_csv_path, row)

                    executed_trials += 1
                    _log(
                        (
                            f"trial dataset={dataset_name}, depth={depth_budget}, seed={seed}, "
                            f"status={row['status']}, acc={row['accuracy']}"
                        ),
                        log_fp,
                    )

                    if args.max_trials > 0 and executed_trials >= args.max_trials:
                        _log(
                            f"Reached --max-trials={args.max_trials}. Stopping early.",
                            log_fp,
                        )
                        break

                if args.max_trials > 0 and executed_trials >= args.max_trials:
                    break

            if args.max_trials > 0 and executed_trials >= args.max_trials:
                break

        seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS)
        seed_df.to_csv(seed_csv_path, index=False)

        summary_df = _build_summary(seed_df, datasets=args.datasets, depth_budgets=args.depth_budgets)
        summary_df.to_csv(summary_csv_path, index=False)

        _plot_accuracy(
            summary_df,
            datasets=args.datasets,
            depth_budgets=args.depth_budgets,
            out_path=plot_path,
            include_paper_sota=False,
        )

        if args.include_paper_sota:
            _plot_accuracy(
                summary_df,
                datasets=args.datasets,
                depth_budgets=args.depth_budgets,
                out_path=plot_vs_paper_path,
                include_paper_sota=True,
            )

        _write_depth_log(summary_df, depth_log_path, args.datasets, args)

        # Write fixed-name outputs under selected stable results directory.
        stable_csv = stable_results_dir / "multisplit_lightgbm_dp_results.csv"
        stable_plot = stable_results_dir / "multisplit_lightgbm_dp_accuracy.png"
        stable_log = stable_results_dir / "multisplit_lightgbm_depth_vs_accuracy.log"

        _safe_copy(summary_csv_path, stable_csv)
        _safe_copy(plot_path, stable_plot)
        _safe_copy(depth_log_path, stable_log)

        manifest_path = run_dir / "manifest.json"
        artifact_paths = {
            "seed_results_csv": seed_csv_path,
            "summary_results_csv": summary_csv_path,
            "depth_log": depth_log_path,
            "plot_accuracy": plot_path,
            "plot_accuracy_vs_paper": plot_vs_paper_path,
            "config_json": config_path,
            "run_stdout_log": stdout_log_path,
            "stable_summary_csv": stable_csv,
            "stable_plot_png": stable_plot,
            "stable_log_txt": stable_log,
        }

        end_time = time.time()
        _write_manifest(
            manifest_path=manifest_path,
            run_name=run_name,
            run_dir=run_dir,
            args=args,
            artifact_paths=artifact_paths,
            start_time=start_time,
            end_time=end_time,
        )

        archive_path = run_dir / f"{run_name}_artifacts.tar.gz"
        if args.package_artifacts:
            bundle_files = [
                config_path,
                manifest_path,
                seed_csv_path,
                summary_csv_path,
                depth_log_path,
                plot_path,
                plot_vs_paper_path,
                stdout_log_path,
            ]
            _create_artifact_bundle(archive_path, bundle_files)
            _log(f"Packaged artifact bundle: {archive_path}", log_fp)

        if args.copy_to:
            copy_dir = Path(args.copy_to).expanduser().resolve()
            copy_dir.mkdir(parents=True, exist_ok=True)
            for path in [summary_csv_path, depth_log_path, plot_path, manifest_path]:
                if path.exists():
                    _safe_copy(path, copy_dir / path.name)
            if plot_vs_paper_path.exists():
                _safe_copy(plot_vs_paper_path, copy_dir / plot_vs_paper_path.name)
            if archive_path.exists():
                _safe_copy(archive_path, copy_dir / archive_path.name)
            _log(f"Copied selected artifacts to {copy_dir}", log_fp)

        _log("=== Summary Table ===", log_fp)
        summary_text = summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        print(summary_text, flush=True)
        log_fp.write(summary_text + "\n")

        _log("=== Depth vs Accuracy (Exact) ===", log_fp)
        for dataset_name in args.datasets:
            _log(f"Dataset: {dataset_name}", log_fp)
            subset = summary_df[summary_df["dataset"] == dataset_name]
            for _, row in subset.iterrows():
                depth = int(row["depth_budget"])
                mean_acc = float(row["mean_accuracy"])
                mean_bal_acc = float(row["mean_balanced_accuracy"])
                mean_trivial_acc = float(row["mean_trivial_accuracy"])
                msg = (
                    f"Depth {depth}: ours={mean_acc:.4f} ({mean_acc*100:.2f}%), "
                    f"bal_acc={mean_bal_acc:.4f} ({mean_bal_acc*100:.2f}%), "
                    f"trivial_acc={mean_trivial_acc:.4f} ({mean_trivial_acc*100:.2f}%), "
                    f"mean_leaves={float(row['mean_n_leaves']):.2f}"
                )
                if dataset_name in PAPER_SOTA and depth in PAPER_SOTA[dataset_name]:
                    paper_acc = float(PAPER_SOTA[dataset_name][depth])
                    delta = mean_acc - paper_acc
                    msg += (
                        f", paper={paper_acc:.4f} ({paper_acc*100:.2f}%), "
                        f"delta={delta:+.4f} ({delta*100:+.2f} pp)"
                    )
                _log(msg, log_fp)

        _log(f"Saved run summary CSV: {summary_csv_path}", log_fp)
        _log(f"Saved run seed CSV: {seed_csv_path}", log_fp)
        _log(f"Saved run plot: {plot_path}", log_fp)
        _log(f"Saved run log: {depth_log_path}", log_fp)
        _log(f"Saved run manifest: {manifest_path}", log_fp)
        if plot_vs_paper_path.exists():
            _log(f"Saved run paper comparison plot: {plot_vs_paper_path}", log_fp)
        if archive_path.exists():
            _log(f"Saved run bundle: {archive_path}", log_fp)

        _log(f"Compatibility CSV: {stable_csv}", log_fp)
        _log(f"Compatibility plot: {stable_plot}", log_fp)
        _log(f"Compatibility log: {stable_log}", log_fp)


if __name__ == "__main__":
    main()
