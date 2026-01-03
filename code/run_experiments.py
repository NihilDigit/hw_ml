"""Experiment runner for intrusion detection (main entrypoint).

This file is intentionally kept as an executable script. Reusable building
blocks (splitting, scaling, timing) live in `pipeline_utils.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    DATA_PROCESSED,
    FEATURES,
    FIGURES_DIR,
    LABEL_COL,
    RANDOM_SEED,
)
from metrics import binary_rates_from_multiclass, overall_accuracy
from plots import plot_2d_scatter
from preprocess import load_and_clean
from torch_reducers import TorchPCA, TorchLDA
from model_selection import grid_search_cuml, grid_search_sklearn
from pipeline_utils import (
    build_2d_viz_frame,
    fit_minmax_scaler,
    save_minmax_scaler_params,
    select_torch_device,
    split_train_test,
    timed,
    transform_with_scaler,
)
from reduction_metrics import class_separation_ratio

try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
except Exception:
    CuMLRandomForestClassifier = None
    CuMLLogisticRegression = None


def reducer_factory(name, n_components, n_classes, device="cuda"):
    """Create a dimensionality reducer for PCA/LDA (deployable reducers only)."""
    if name == "PCA":
        return TorchPCA(n_components=n_components, device=device)
    if name == "LDA":
        # LDA components cannot exceed n_classes - 1
        n_comp = min(n_components, max(n_classes - 1, 1))
        return TorchLDA(n_components=n_comp, device=device)
    raise ValueError(f"Unknown reducer: {name}")


def main():
    device = select_torch_device()
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    print(f"Loading data from {csv_path}...")
    df = load_and_clean(str(csv_path))
    print(f"Loaded {len(df)} samples with {len(df[LABEL_COL].unique())} classes")

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    split = split_train_test(X, y, test_size=0.2, seed=RANDOM_SEED)
    X_train, X_test, y_train, y_test = split.X_train, split.X_test, split.y_train, split.y_test
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    scaler = fit_minmax_scaler(X_train)
    X_train = transform_with_scaler(scaler, X_train)
    X_test = transform_with_scaler(scaler, X_test)
    save_minmax_scaler_params(scaler, DATA_PROCESSED / "minmax_scaler_params.json")

    # Experiment design: 18 combinations (6 reducers × 3 classifiers)
    # Dimensionality reduction methods: PCA, LDA (t-SNE moved to standalone viz)
    experiments = [
        ("PCA", 10),      # PCA-10D
        ("PCA", 15),      # PCA-15D
        ("PCA", 20),      # PCA-20D
        ("LDA", 6),       # LDA-6D (<= n_classes-1)
        ("LDA", 10),      # LDA-10D (<= n_classes-1)
        ("LDA", 14),      # LDA-14D (max for 15 classes)
    ]

    use_cuml = CuMLRandomForestClassifier is not None and CuMLLogisticRegression is not None
    if not use_cuml:
        print("cuML is not available; falling back to sklearn models (CPU).")

    # Classifier configurations (GPU via cuML)
    # Note: cuML SVC has a known bug with multi-class classification, so we use sklearn SVC
    classifiers = {
        "SVM": {
            "type": "sklearn",
            "model_cls": SVC,
            "param_grid": [
                {"C": 1, "kernel": "rbf", "gamma": "scale"},
                {"C": 10, "kernel": "rbf", "gamma": "scale"},
            ],
        },
        "RandomForest": {
            "type": "cuml" if use_cuml else "sklearn",
            "model_cls": CuMLRandomForestClassifier if use_cuml else RandomForestClassifier,
            "param_grid": [
                (
                    {"n_estimators": 200, "max_depth": 10, "n_streams": 1, "random_state": RANDOM_SEED}
                    if use_cuml
                    else {"n_estimators": 200, "max_depth": None, "random_state": RANDOM_SEED, "n_jobs": -1}
                ),
                (
                    {"n_estimators": 200, "max_depth": 20, "n_streams": 1, "random_state": RANDOM_SEED}
                    if use_cuml
                    else {"n_estimators": 200, "max_depth": 20, "random_state": RANDOM_SEED, "n_jobs": -1}
                ),
            ],
        },
        "LogisticRegression": {
            "type": "cuml" if use_cuml else "sklearn",
            "model_cls": CuMLLogisticRegression if use_cuml else LogisticRegression,
            "param_grid": [
                (
                    {"C": 0.1, "max_iter": 1000}
                    if use_cuml
                    else {"C": 0.1, "max_iter": 1000, "solver": "lbfgs", "multi_class": "auto"}
                ),
                (
                    {"C": 1.0, "max_iter": 1000}
                    if use_cuml
                    else {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "multi_class": "auto"}
                ),
                (
                    {"C": 10.0, "max_iter": 1000}
                    if use_cuml
                    else {"C": 10.0, "max_iter": 1000, "solver": "lbfgs", "multi_class": "auto"}
                ),
            ],
        },
    }

    metrics_rows = []
    reduction_rows = []

    n_classes = len(np.unique(y))

    # Track processed reducer combinations to avoid redundant computations
    processed_reductions = {}

    for exp_idx, (reducer_name, n_comp) in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx}/{len(experiments)}: {reducer_name}-{n_comp}D")
        print(f"{'='*60}")

        # Use cached reduction if already computed
        reduction_key = (reducer_name, n_comp)
        if reduction_key in processed_reductions:
            print(f"Using cached {reducer_name}-{n_comp}D reduction...")
            X_train_red, X_test_red = processed_reductions[reduction_key]
        else:
            print(f"\n--- Dimensions: {n_comp} ---")

            reducer = reducer_factory(reducer_name, n_comp, n_classes, device)

            # Apply dimensionality reduction
            reduced_train = timed(reducer.fit_transform, X_train, y_train)
            X_train_red = reduced_train.value
            X_test_red = reducer.transform(X_test)

            print(f"Reduced to shape: {X_train_red.shape}")
            print(f"Reduction time: {reduced_train.seconds:.3f}s")

            # Compute reduction metrics
            info_retention = ""
            if reducer_name in {"PCA", "LDA"} and hasattr(
                reducer, "explained_variance_ratio_"
            ):
                evr = reducer.explained_variance_ratio_
                if torch.is_tensor(evr):
                    evr = evr.cpu().numpy()
                info_retention = float(np.sum(evr))
                print(f"Explained variance ratio: {info_retention:.4f}")

            sep_ratio = class_separation_ratio(X_train_red, y_train)
            print(f"Class separation ratio: {sep_ratio:.4f}")

            reduction_rows.append(
                {
                    "Reducer": reducer_name,
                    "n_components": n_comp,
                    "Information_retention": info_retention,
                    "Class_separation": sep_ratio,
                }
            )

            # 2D visualization
            if X_train_red.shape[1] >= 2:
                print("Generating 2D visualization...")
                sample_size = min(5000, len(X_train_red))
                sample_idx = np.random.RandomState(RANDOM_SEED).choice(
                    len(X_train_red), size=sample_size, replace=False
                )
                viz_df = build_2d_viz_frame(X_train_red[sample_idx, :2], y_train[sample_idx])
                plot_path = FIGURES_DIR / f"{reducer_name}_{n_comp}_2d.png"
                plot_2d_scatter(viz_df, LABEL_COL, plot_path)
                print(f"Saved plot to {plot_path}")

            # Cache the reduction for reuse
            processed_reductions[reduction_key] = (X_train_red, X_test_red)

        # Train classifiers on reduced data
        for clf_name, clf_config in classifiers.items():
            print(f"\nTraining {clf_name}...")

            if clf_config["type"] == "cuml":
                trained = timed(
                    grid_search_cuml,
                    clf_config["model_cls"],
                    clf_config["param_grid"],
                    X_train_red,
                    y_train,
                    cv=3,
                )
                model, best_params, _ = trained.value
            elif clf_config["type"] == "sklearn":
                trained = timed(
                    grid_search_sklearn,
                    clf_config["model_cls"],
                    clf_config["param_grid"],
                    X_train_red,
                    y_train,
                    cv=3,
                )
                model, best_params, _ = trained.value
            elif clf_config["type"] == "cuml_or_sklearn":
                try:
                    trained = timed(
                        grid_search_cuml,
                        clf_config["model_cls"],
                        clf_config["param_grid"],
                        X_train_red,
                        y_train,
                        cv=3,
                    )
                    model, best_params, _ = trained.value
                except Exception as err:
                    print(f"cuML {clf_name} failed ({err}); falling back to sklearn...")
                    trained = timed(
                        grid_search_sklearn,
                        clf_config["fallback_cls"],
                        clf_config["fallback_param_grid"],
                        X_train_red,
                        y_train,
                        cv=3,
                    )
                    model, best_params, _ = trained.value

            train_time = trained.seconds

            # Prediction
            predicted = timed(model.predict, X_test_red)
            y_pred = predicted.value
            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            pred_time = predicted.seconds

            # Metrics
            acc = overall_accuracy(y_test, y_pred)
            fpr, fnr = binary_rates_from_multiclass(y_test, y_pred)

            print(
                f"  Accuracy: {acc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, "
                f"Train time: {train_time:.2f}s, Pred time: {pred_time:.4f}s"
            )
            print(f"  Best params: {best_params}")

            metrics_rows.append(
                {
                    "Reducer": reducer_name,
                    "n_components": n_comp,
                    "Classifier": clf_name,
                    "Accuracy": acc,
                    "FPR": fpr,
                    "FNR": fnr,
                    "Train_time_s": train_time,
                    "Predict_time_s": pred_time,
                    "Best_params": str(best_params),
                }
            )

    # Save results
    metrics_df = pd.DataFrame(metrics_rows)
    reduction_df = pd.DataFrame(reduction_rows)
    metrics_df.to_csv(DATA_PROCESSED / "metrics.csv", index=False)
    reduction_df.to_csv(DATA_PROCESSED / "reduction_metrics.csv", index=False)

    print(f"\n{'='*60}")
    print("Results saved!")
    print(f"Metrics: {DATA_PROCESSED / 'metrics.csv'}")
    print(f"Reduction metrics: {DATA_PROCESSED / 'reduction_metrics.csv'}")

    # Attack-specific metrics for best combination
    print(f"\n{'='*60}")
    print("Computing attack-specific metrics for best model...")
    best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    print(f"Best combination: {best_row['Reducer']} + {best_row['Classifier']}")
    print(f"  Accuracy: {best_row['Accuracy']:.4f}")

    # Refit best model
    reducer = reducer_factory(best_row["Reducer"], int(best_row["n_components"]), n_classes, device)
    X_train_red = reducer.fit_transform(X_train, y_train)
    X_test_red = reducer.transform(X_test)

    clf_config = classifiers[best_row["Classifier"]]

    if clf_config["type"] == "cuml":
        model, _, _ = grid_search_cuml(
            clf_config["model_cls"], clf_config["param_grid"], X_train_red, y_train, cv=3
        )
    elif clf_config["type"] == "sklearn":
        model, _, _ = grid_search_sklearn(
            clf_config["model_cls"], clf_config["param_grid"], X_train_red, y_train, cv=3
        )
    elif clf_config["type"] == "cuml_or_sklearn":
        try:
            model, _, _ = grid_search_cuml(
                clf_config["model_cls"], clf_config["param_grid"], X_train_red, y_train, cv=3
            )
        except Exception:
            model, _, _ = grid_search_sklearn(
                clf_config["fallback_cls"], clf_config["fallback_param_grid"], X_train_red, y_train, cv=3
            )

    y_pred = model.predict(X_test_red)
    if hasattr(y_pred, "get"):
        y_pred = y_pred.get()

    # Per-attack metrics for all attack labels (exclude Benign)
    labels = np.unique(y_test)
    attack_labels = [l for l in labels if l != "Benign"]

    if attack_labels:
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=attack_labels, average=None, zero_division=0
        )
        attack_df = pd.DataFrame(
            {
                "Attack_type": attack_labels,
                "Precision": pr.astype(float),
                "Recall": rc.astype(float),
                "F1": f1.astype(float),
            }
        ).sort_values("Attack_type")
        attack_df.to_csv(DATA_PROCESSED / "attack_metrics.csv", index=False)
        print(f"\nAttack-specific metrics saved to {DATA_PROCESSED / 'attack_metrics.csv'}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
