"""
Classification Demo using ShapeCARTClassifier

This script demonstrates ShapeCARTClassifier on the Electricity dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import *
from src.data_utils import DataFactory_clf
from sklearn.metrics import accuracy_score
from src.ShapeCARTClassifier import ShapeCARTClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


def run_classification_demo():
    """Demo on Electricity dataset"""
    print("=" * 80)
    print("CLASSIFICATION: Electricity Dataset")
    print("=" * 80)
    print()

    dataset = 'electricity'
    data_factory = DataFactory_clf(dataset=dataset, cache=False)
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(0)
    feature_dict = data_factory.feature_dict
    n_classes = len(np.unique(y_train))

    print(f"Dataset: {dataset}")
    print(f"Number of classes: {n_classes}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {list(feature_dict.keys())}")
    print()

    # Store results for summary table
    results = []

    # 1. Baseline: sklearn DecisionTreeClassifier
    cart = DecisionTreeClassifier(max_depth=5, random_state=0)
    cart.fit(X_train, y_train)
    train_pred = cart.predict(X_train)
    test_pred = cart.predict(X_test)
    results.append(("DecisionTreeClassifier",
                    accuracy_score(y_train, train_pred),
                    accuracy_score(y_test, test_pred)))

    # 2. ShapeCARTClassifier (k=2)
    clf = ShapeCARTClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("ShapeCARTClassifier (k=2)",
                    accuracy_score(y_train, train_pred),
                    accuracy_score(y_test, test_pred)))

    # 3. ShapeCARTClassifier (k=3)
    clf = ShapeCARTClassifier(max_depth=5, random_state=0, k=3)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("ShapeCARTClassifier (k=3)",
                    accuracy_score(y_train, train_pred),
                    accuracy_score(y_test, test_pred)))

    # 4. Shape2CART (pairwise)
    clf = ShapeCARTClassifier(max_depth=5, random_state=0, pairwise_candidates=1.0)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("Shape2CART (pairwise)",
                    accuracy_score(y_train, train_pred),
                    accuracy_score(y_test, test_pred)))

    # 5. Shape2CART_3 (pairwise + k=3)
    clf = ShapeCARTClassifier(max_depth=5, random_state=0, pairwise_candidates=1.0, k=3)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("Shape2CART (pairwise, k=3)",
                    accuracy_score(y_train, train_pred),
                    accuracy_score(y_test, test_pred)))

    # Print summary table
    print("SUMMARY: Accuracy Results")
    print("-" * 60)
    print(f"{'Model':<30} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 60)
    for name, train_acc, test_acc in results:
        print(f"{name:<30} {train_acc:<15.4f} {test_acc:<15.4f}")
    print()


if __name__ == "__main__":
    run_classification_demo()
