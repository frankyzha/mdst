"""
Regression Demo using ShapeCARTRegressor

This script demonstrates ShapeCARTRegressor on two datasets:
1. California Housing (single-target regression)
2. ENB Energy Efficiency (multi-target regression)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.ShapeCARTRegressor import ShapeCARTRegressor
from sklearn.tree import DecisionTreeRegressor
from src.data_utils import DataFactory_rgr
import warnings
warnings.filterwarnings('ignore')


def run_single_target_demo():
    """Demo on California Housing dataset (single-target)"""
    print("=" * 80)
    print("SINGLE-TARGET REGRESSION: California Housing Dataset")
    print("=" * 80)
    print()

    dataset = 'california'
    data_factory = DataFactory_rgr(dataset=dataset, cache=False)
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(0)
    feature_dict = data_factory.feature_dict

    print(f"Dataset: {dataset}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {list(feature_dict.keys())}")
    print()

    # Store results for summary table
    results = []

    # 1. Baseline: sklearn DecisionTreeRegressor
    cart = DecisionTreeRegressor(max_depth=5, random_state=0)
    cart.fit(X_train, y_train)
    train_pred = cart.predict(X_train)
    test_pred = cart.predict(X_test)
    results.append(("DecisionTreeRegressor",
                    mean_squared_error(y_train, train_pred),
                    mean_squared_error(y_test, test_pred)))

    # 2. ShapeCARTRegressor (k=2)
    clf = ShapeCARTRegressor(max_depth=5, random_state=0)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("ShapeCARTRegressor (k=2)",
                    mean_squared_error(y_train, train_pred),
                    mean_squared_error(y_test, test_pred)))

    # 3. ShapeCARTRegressor (k=3)
    clf = ShapeCARTRegressor(max_depth=5, random_state=0, k=3)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("ShapeCARTRegressor (k=3)",
                    mean_squared_error(y_train, train_pred),
                    mean_squared_error(y_test, test_pred)))

    # 4. Shape2CART (pairwise)
    clf = ShapeCARTRegressor(max_depth=5, random_state=0, pairwise_candidates=1.0)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("Shape2CART (pairwise)",
                    mean_squared_error(y_train, train_pred),
                    mean_squared_error(y_test, test_pred)))

    # 5. Shape2CART_3 (pairwise + k=3)
    clf = ShapeCARTRegressor(max_depth=5, random_state=0, pairwise_candidates=1.0, k=3)
    clf.fit(X_train, y_train, feature_dict=feature_dict)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results.append(("Shape2CART (pairwise, k=3)",
                    mean_squared_error(y_train, train_pred),
                    mean_squared_error(y_test, test_pred)))

    # Print summary table
    print("SUMMARY: MSE Results")
    print("-" * 60)
    print(f"{'Model':<30} {'Train MSE':<15} {'Test MSE':<15}")
    print("-" * 60)
    for name, train_mse, test_mse in results:
        print(f"{name:<30} {train_mse:<15.4f} {test_mse:<15.4f}")
    print()


def run_multi_target_demo():
    """Demo on ENB Energy Efficiency dataset (multi-target)"""
    print("=" * 80)
    print("MULTI-TARGET REGRESSION: ENB Energy Efficiency Dataset")
    print("=" * 80)
    print()

    # Fetch the ENB (Energy Efficiency) dataset
    print("Loading ENB dataset from OpenML...")
    data = fetch_openml(name='ENB', as_frame=True, parser='auto')

    X = data.data.values.astype(np.float64)
    y = data.target.values.astype(np.float64)

    feature_names = list(data.data.columns)
    target_names = ['Heating Load (Y1)', 'Cooling Load (Y2)']

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")
    print()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()

    # Store results: (name, train_preds, test_preds)
    models = []

    # 1. Baseline: sklearn DecisionTreeRegressor
    cart = DecisionTreeRegressor(max_depth=5, random_state=42)
    cart.fit(X_train, y_train)
    models.append(("DecisionTreeRegressor", cart.predict(X_train), cart.predict(X_test)))

    # 2. ShapeCARTRegressor (k=2)
    scart = ShapeCARTRegressor(max_depth=5, random_state=42, k=2, verbose=False)
    scart.fit(X_train, y_train)
    models.append(("ShapeCARTRegressor (k=2)", scart.predict(X_train), scart.predict(X_test)))

    # 3. ShapeCARTRegressor (k=3)
    scart_k3 = ShapeCARTRegressor(max_depth=5, random_state=42, k=3, verbose=False)
    scart_k3.fit(X_train, y_train)
    models.append(("ShapeCARTRegressor (k=3)", scart_k3.predict(X_train), scart_k3.predict(X_test)))

    # 4. ShapeCARTRegressor with pairwise
    scart_pair = ShapeCARTRegressor(max_depth=5, random_state=42, k=2, pairwise_candidates=3, verbose=False)
    scart_pair.fit(X_train, y_train)
    models.append(("Shape2CART (pairwise)", scart_pair.predict(X_train), scart_pair.predict(X_test)))

    # Print summary table - Train MSE
    print("SUMMARY: Train MSE")
    print("-" * 75)
    print(f"{'Model':<30} {'Heating (Y1)':<15} {'Cooling (Y2)':<15} {'Overall':<15}")
    print("-" * 75)
    for name, train_pred, test_pred in models:
        mse_y1 = mean_squared_error(y_train[:, 0], train_pred[:, 0])
        mse_y2 = mean_squared_error(y_train[:, 1], train_pred[:, 1])
        mse_overall = mean_squared_error(y_train, train_pred)
        print(f"{name:<30} {mse_y1:<15.4f} {mse_y2:<15.4f} {mse_overall:<15.4f}")
    print()

    # Print summary table - Test MSE
    print("SUMMARY: Test MSE")
    print("-" * 75)
    print(f"{'Model':<30} {'Heating (Y1)':<15} {'Cooling (Y2)':<15} {'Overall':<15}")
    print("-" * 75)
    for name, train_pred, test_pred in models:
        mse_y1 = mean_squared_error(y_test[:, 0], test_pred[:, 0])
        mse_y2 = mean_squared_error(y_test[:, 1], test_pred[:, 1])
        mse_overall = mean_squared_error(y_test, test_pred)
        print(f"{name:<30} {mse_y1:<15.4f} {mse_y2:<15.4f} {mse_overall:<15.4f}")
    print()


if __name__ == "__main__":
    run_single_target_demo()
    print("\n")
    run_multi_target_demo()
