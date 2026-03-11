"""Dataset loading and feature-to-bin conversion helpers."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dataset_loaders():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import load_electricity, load_eye_movements, load_eye_state

    return {
        "electricity": load_electricity,
        "eye-movements": load_eye_movements,
        "eye-state": load_eye_state,
    }


def _require_lightgbm() -> None:
    if lgb is None:
        raise ModuleNotFoundError("LightGBM binning mode requires 'lightgbm'.")


def _tokenize_object(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=object).reshape(-1)
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append("__nan__")
        else:
            out.append(str(v))
    return np.asarray(out, dtype=object)


def _factorize_to_float(values: np.ndarray) -> np.ndarray:
    _, inv = np.unique(_tokenize_object(values), return_inverse=True)
    return inv.astype(float)


def _sanitize_feature(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.isnan(arr).any():
        finite = arr[np.isfinite(arr)]
        fill = float(np.median(finite)) if finite.size else 0.0
        arr = arr.copy()
        arr[~np.isfinite(arr)] = fill
    return arr


def load_dataset_feature_matrix(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    loaders = _dataset_loaders()
    if dataset_name not in loaders:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {sorted(loaders)}")

    X, y = loaders[dataset_name]()

    if hasattr(X, "select_dtypes") and hasattr(X, "columns"):
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in X.columns if c not in num_cols]

        blocks: List[np.ndarray] = []
        if num_cols:
            num = np.asarray(X[num_cols], dtype=float)
            blocks.append(num if num.ndim == 2 else num.reshape(-1, 1))
        for c in cat_cols:
            blocks.append(_factorize_to_float(np.asarray(X[c], dtype=object)).reshape(-1, 1))

        X_proc = np.hstack(blocks) if blocks else np.zeros((len(X), 1), dtype=float)
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if np.issubdtype(arr.dtype, np.number):
            X_proc = arr.astype(float)
        else:
            cols = [_factorize_to_float(arr[:, j]).reshape(-1, 1) for j in range(arr.shape[1])]
            X_proc = np.hstack(cols) if cols else np.zeros((arr.shape[0], 1), dtype=float)

    X_proc = np.asarray(X_proc, dtype=float)
    if np.isnan(X_proc).any():
        med = np.nanmedian(X_proc, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        r, c = np.where(np.isnan(X_proc))
        X_proc[r, c] = med[c]

    return np.ascontiguousarray(X_proc, dtype=float), np.asarray(y, dtype=object).reshape(-1)


def encode_binary_labels(y_raw: np.ndarray):
    labels, inv, counts = np.unique(_tokenize_object(y_raw), return_inverse=True, return_counts=True)
    if labels.size < 2:
        raise ValueError("Need at least 2 classes for binary partitioning.")

    if labels.size == 2:
        pos_idx = 1
        mode = "binary"
    else:
        pos_idx = int(np.argmax(counts))
        mode = "one-vs-rest-majority"

    y_bin = (inv == pos_idx).astype(np.int8)
    return y_bin, {
        "mode": mode,
        "n_classes": int(labels.size),
        "positive_label": str(labels[pos_idx]),
        "positive_count": int(np.sum(y_bin == 1)),
        "negative_count": int(np.sum(y_bin == 0)),
    }


def equal_width_bin_ids(values: np.ndarray, B: int) -> Tuple[np.ndarray, int]:
    arr = _sanitize_feature(values)
    B = int(B)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int32), B

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo == hi:
        return np.zeros(arr.size, dtype=np.int32), B

    edges = np.linspace(lo, hi, B + 1, dtype=float)
    ids = np.searchsorted(edges, arr, side="right") - 1
    return np.clip(ids, 0, B - 1).astype(np.int32, copy=False), B


def _parse_lgb_dump_bin_ids(path: Path, n_rows: int) -> np.ndarray:
    feature_hdr = re.compile(r"^feature\s+0:\s*$")
    value_re = re.compile(r"^(-?\d+),?\s*$")

    in_feature = False
    out: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not in_feature:
                in_feature = bool(feature_hdr.match(line))
                continue
            if not line:
                continue
            m = value_re.match(line)
            if not m:
                break
            out.append(int(m.group(1)))
            if len(out) == n_rows:
                break

    if len(out) != n_rows:
        raise RuntimeError(f"Failed to parse LightGBM bin IDs: expected {n_rows}, got {len(out)}")
    return np.asarray(out, dtype=np.int32)


def lightgbm_bin_ids(values: np.ndarray, B: int, min_data_in_bin: int) -> Tuple[np.ndarray, int]:
    _require_lightgbm()
    assert lgb is not None

    arr = _sanitize_feature(values)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int32), 1

    ds = lgb.Dataset(
        arr.reshape(-1, 1),
        label=np.zeros(arr.size, dtype=np.float32),
        params={
            "max_bin": int(B),
            "min_data_in_bin": int(min_data_in_bin),
            "feature_pre_filter": False,
            "verbosity": -1,
            "bin_construct_sample_cnt": int(arr.size),
        },
        free_raw_data=False,
    )
    ds.construct()
    n_bins = max(1, int(ds.feature_num_bin(0)))

    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        ds._dump_text(str(tmp_path))
        bin_ids = _parse_lgb_dump_bin_ids(tmp_path, arr.size)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if bin_ids.size == 0:
        return bin_ids, n_bins
    return np.clip(bin_ids, 0, n_bins - 1).astype(np.int32, copy=False), n_bins


def pos_neg_from_bin_ids(bin_ids: np.ndarray, y_bin: np.ndarray, n_bins: int) -> Tuple[List[int], List[int]]:
    y = np.asarray(y_bin, dtype=np.int8).reshape(-1)
    if bin_ids.size != y.size:
        raise ValueError(f"Feature/label length mismatch: {bin_ids.size} vs {y.size}")

    pos = np.bincount(bin_ids[y == 1], minlength=int(n_bins)) if bin_ids.size else np.zeros(int(n_bins), dtype=int)
    neg = np.bincount(bin_ids[y == 0], minlength=int(n_bins)) if bin_ids.size else np.zeros(int(n_bins), dtype=int)
    return [int(v) for v in pos.tolist()], [int(v) for v in neg.tolist()]


def build_feature_bins(
    X_proc: np.ndarray,
    y_bin: np.ndarray,
    B: int,
    dataset_binning: str,
    lgbm_min_data_in_bin: int,
    lgbm_cache: Dict[Tuple[int, int], Tuple[np.ndarray, int]],
) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
    bins: List[Tuple[List[int], List[int]]] = []
    n_bins_list: List[int] = []

    for feat_idx in range(int(X_proc.shape[1])):
        if dataset_binning == "lightgbm":
            key = (int(B), int(feat_idx))
            hit = lgbm_cache.get(key)
            if hit is None:
                hit = lightgbm_bin_ids(X_proc[:, feat_idx], B=int(B), min_data_in_bin=int(lgbm_min_data_in_bin))
                lgbm_cache[key] = hit
            bin_ids, n_bins = hit
        else:
            bin_ids, n_bins = equal_width_bin_ids(X_proc[:, feat_idx], B=int(B))

        bins.append(pos_neg_from_bin_ids(bin_ids, y_bin, int(n_bins)))
        n_bins_list.append(int(n_bins))

    return bins, n_bins_list
