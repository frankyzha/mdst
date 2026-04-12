from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from split import NumericBinarizer


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parents[2]


def _dataset_path(filename: str) -> Path:
    candidates = [TESTS_DIR / "datasets" / filename]
    if filename == "compas.csv":
        candidates.extend(
            [
                PROJECT_ROOT / "SPLIT-ICML" / "resplit" / "example" / "compas.csv",
                PROJECT_ROOT / "benchmark" / "datasets" / "compas" / "raw" / "compas_data_matrix.csv",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    pytest.skip(f"Missing optional test fixture: {filename}")


def test_spiral():
    spiral = pd.read_csv(_dataset_path("spiral.csv"))
    X = spiral.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 100
    assert Xt.shape[1] == 180
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_compas():
    compas = pd.read_csv(_dataset_path("compas.csv"))
    X = compas.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 6907
    assert Xt.shape[1] == 134
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_broward():
    broward = pd.read_csv(_dataset_path("broward_general_2y.csv"))
    X = broward.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 1954
    assert Xt.shape[1] == 588
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_iris():
    from sklearn.datasets import load_iris
    iris = load_iris().data
    enc = NumericBinarizer()
    Xt = enc.fit_transform(iris)
    assert Xt.shape[0] == 150
    assert Xt.shape[1] == 119
    assert np.array_equal(iris, enc.inverse_transform(Xt))
