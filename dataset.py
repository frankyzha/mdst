"""OpenML dataset loaders used by multisplit experiments."""

from __future__ import annotations

import openml


def _load_openml_dataset(dataset_id: int):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y


def load_electricity():
    """Electricity dataset (OpenML 151)."""
    return _load_openml_dataset(151)


def load_eye_movements():
    """Eye-Movements dataset (OpenML 45073)."""
    return _load_openml_dataset(45073)


def load_eye_state():
    """Eye-State dataset (OpenML 1471)."""
    return _load_openml_dataset(1471)
