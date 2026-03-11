"""Shared utilities/constants for multisplit experiment scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dataset import load_electricity, load_eye_movements, load_eye_state


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


def make_preprocessor(X_train):
    if not hasattr(X_train, "select_dtypes"):
        return Pipeline([("imputer", SimpleImputer(strategy="median"))])

    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    transformers = []

    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))

    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def encode_binary_target(y, dataset_name: str) -> np.ndarray:
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
