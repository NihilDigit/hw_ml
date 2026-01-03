"""Sklearn-based experiment pipeline for intrusion detection.

Strictly follows requirements:
- 3 reducers: PCA, LDA, t-SNE
- 3 dimensions: 10, 15, 20 (t-SNE only supports 2-3D, use 2D)
- 3 classifiers: SVM, RandomForest, LogisticRegression
- Metrics: Accuracy, FPR (误报率), FNR (漏报率), Detection time
- Grid search for hyperparameter optimization
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
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


def grid_search(model_cls, param_grid_list, X, y, cv=3, seed=RANDOM_SEED):
    """Simple manual grid search using macro F1 score.

    Much faster than GridSearchCV for small datasets due to less overhead.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_score = -1.0
    best_params = None

    for params in param_grid_list:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_cls(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit on full training data
    best_model = model_cls(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score


def class_separation_ratio(X, y):
    """Compute class separation ratio (between-class / within-class scatter).

    This metric evaluates the quality of dimensionality reduction by measuring
    how well the reduced features separate different classes.
    """
    labels = np.unique(y)
    centroids = {}
    for label in labels:
        centroids[label] = X[y == label].mean(axis=0)

    # Within-class scatter (类内散度)
    within = 0.0
    for label in labels:
        diffs = X[y == label] - centroids[label]
        within += np.sum(np.linalg.norm(diffs, axis=1) ** 2)
    within = within / max(len(X), 1)

    # Between-class scatter (类间散度)
    all_centroid = X.mean(axis=0)
    between = 0.0
    for label in labels:
        n = (y == label).sum()
        diff = centroids[label] - all_centroid
        between += n * (np.linalg.norm(diff) ** 2)
    between = between / max(len(X), 1)

    return between / within if within > 0 else 0.0


def reducer_factory(name, n_components, n_classes):
    """Create dimensionality reduction model.

    Args:
        name: One of "PCA", "LDA", "t-SNE"
        n_components: Target dimensions
        n_classes: Number of classes (for LDA constraint)
    """
    if name == "PCA":
        return PCA(n_components=n_components, svd_solver="auto", whiten=False)
    if name == "LDA":
        # LDA components cannot exceed n_classes - 1
        n_comp = min(n_components, max(n_classes - 1, 1))
        return LinearDiscriminantAnalysis(n_components=n_comp, solver="svd")
    if name == "t-SNE":
        # t-SNE only supports 2-3 dimensions in sklearn
        n_comp = min(n_components, 3)
        return TSNE(
            n_components=n_comp,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            n_iter=1000,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown reducer: {name}")


def main():
    print("=" * 60)
    print("Network Intrusion Detection System - Experiment Pipeline")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load data - use the stratified subset
    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    print(f"\nLoading data from {csv_path}...")
    df = load_and_clean(str(csv_path))
    print(f"Loaded {len(df)} samples with {len(df[LABEL_COL].unique())} classes")
    print(f"Class distribution:\n{df[LABEL_COL].value_counts()}")

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    # Train/test split (80/20 as required)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Min-Max normalization (as required)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Experiment design per requirements:
    # - 3 reducers: PCA, LDA, t-SNE
    # - PCA/LDA: 10, 15, 20 dimensions
    # - t-SNE: only 2D (t-SNE is for visualization, higher dims don't make sense)
    experiments = [
        ("PCA", 10),
        ("PCA", 15),
        ("PCA", 20),
        ("LDA", 10),
        ("LDA", 15),
        ("LDA", 20),
        ("t-SNE", 2),  # t-SNE only for 2D visualization
    ]

    # 3 classifiers with manual grid search (as required: 网格搜索优化超参数)
    # Using fixed parameter lists - same as torch version for fair comparison
    classifiers = {
        "SVM": {
            "model_cls": SVC,
            "param_grid": [
                {"C": 1, "kernel": "rbf", "gamma": "scale"},
                {"C": 10, "kernel": "rbf", "gamma": "scale"},
                {"C": 1, "kernel": "linear"},
                {"C": 10, "kernel": "linear"},
            ],
        },
        "RandomForest": {
            "model_cls": RandomForestClassifier,
            "param_grid": [
                {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_SEED, "n_jobs": -1},
                {"n_estimators": 200, "max_depth": 10, "random_state": RANDOM_SEED, "n_jobs": -1},
                {"n_estimators": 100, "max_depth": 20, "random_state": RANDOM_SEED, "n_jobs": -1},
                {"n_estimators": 200, "max_depth": 20, "random_state": RANDOM_SEED, "n_jobs": -1},
            ],
        },
        "LogisticRegression": {
            "model_cls": LogisticRegression,
            "param_grid": [
                {"C": 0.1, "max_iter": 1000, "random_state": RANDOM_SEED},
                {"C": 1.0, "max_iter": 1000, "random_state": RANDOM_SEED},
                {"C": 10.0, "max_iter": 1000, "random_state": RANDOM_SEED},
            ],
        },
    }

    metrics_rows = []
    reduction_rows = []
    n_classes = len(np.unique(y))

    # Cache for reduced data to avoid recomputation
    reduction_cache = {}

    total_experiments = len(experiments)
    exp_count = 0

    for reducer_name, n_comp in experiments:
        exp_count += 1
        print(f"\n{'=' * 60}")
        print(f"Experiment {exp_count}/{total_experiments}: {reducer_name}-{n_comp}D")
        print("=" * 60)

        cache_key = (reducer_name, n_comp)

        if cache_key in reduction_cache:
            print(f"Using cached {reducer_name}-{n_comp}D reduction...")
            X_train_red, X_test_red, actual_dims = reduction_cache[cache_key]
        else:
            reducer = reducer_factory(reducer_name, n_comp, n_classes)

            start_reduce = time.time()
            if reducer_name == "t-SNE":
                # t-SNE has no transform, must fit on full data
                X_full = np.vstack([X_train, X_test])
                print(f"Running t-SNE on {len(X_full)} samples...")
                X_red = reducer.fit_transform(X_full)
                X_train_red = X_red[: len(X_train)]
                X_test_red = X_red[len(X_train):]
            else:
                X_train_red = reducer.fit_transform(X_train, y_train)
                X_test_red = reducer.transform(X_test)
            reduce_time = time.time() - start_reduce

            actual_dims = X_train_red.shape[1]
            print(f"Reduced to {actual_dims}D in {reduce_time:.2f}s")

            # Compute reduction quality metrics
            if reducer_name == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
                info_retention = float(np.sum(reducer.explained_variance_ratio_))
                print(f"PCA explained variance ratio: {info_retention:.4f}")
            elif reducer_name == "LDA" and hasattr(reducer, "explained_variance_ratio_"):
                info_retention = float(np.sum(reducer.explained_variance_ratio_))
                print(f"LDA explained variance ratio: {info_retention:.4f}")
            else:
                info_retention = None  # t-SNE doesn't have this metric

            sep_ratio = class_separation_ratio(X_train_red, y_train)
            print(f"Class separation ratio: {sep_ratio:.4f}")

            reduction_rows.append({
                "Reducer": reducer_name,
                "n_components": n_comp,
                "Actual_dims": actual_dims,
                "Information_retention": info_retention if info_retention else "",
                "Class_separation": sep_ratio,
            })

            # Generate 2D visualization
            if actual_dims >= 2:
                print("Generating 2D visualization...")
                sample_size = min(5000, len(X_train_red))
                sample_idx = np.random.RandomState(RANDOM_SEED).choice(
                    len(X_train_red), size=sample_size, replace=False
                )
                viz_df = pd.DataFrame(X_train_red[sample_idx, :2], columns=["c1", "c2"])
                viz_df[LABEL_COL] = y_train[sample_idx]
                plot_path = FIGURES_DIR / f"{reducer_name}_{n_comp}_2d.png"
                plot_2d_scatter(viz_df, LABEL_COL, plot_path)
                print(f"Saved plot to {plot_path}")

            # Cache the reduction
            reduction_cache[cache_key] = (X_train_red, X_test_red, actual_dims)

        # Train and evaluate all classifiers
        for clf_name, clf_config in classifiers.items():
            print(f"\nTraining {clf_name}...")

            # Manual grid search (as required: 网格搜索优化超参数)
            start_train = time.time()
            model, best_params, _ = grid_search(
                clf_config["model_cls"],
                clf_config["param_grid"],
                X_train_red,
                y_train,
                cv=3,
            )
            train_time = time.time() - start_train

            # Prediction and timing
            start_pred = time.time()
            y_pred = model.predict(X_test_red)
            pred_time = time.time() - start_pred

            # Compute metrics as required:
            # - Accuracy (准确率)
            # - FPR (误报率): False Positive Rate
            # - FNR (漏报率): False Negative Rate
            # - Detection time (检测耗时)
            acc = overall_accuracy(y_test, y_pred)
            fpr, fnr = binary_rates_from_multiclass(y_test, y_pred)

            print(f"  Accuracy: {acc:.4f}")
            print(f"  FPR (误报率): {fpr:.4f}")
            print(f"  FNR (漏报率): {fnr:.4f}")
            print(f"  Train time: {train_time:.2f}s, Predict time: {pred_time:.4f}s")
            print(f"  Best params: {best_params}")

            metrics_rows.append({
                "Reducer": reducer_name,
                "n_components": n_comp,
                "Actual_dims": actual_dims,
                "Classifier": clf_name,
                "Accuracy": acc,
                "FPR": fpr,
                "FNR": fnr,
                "Train_time_s": train_time,
                "Predict_time_s": pred_time,
                "Best_params": str(best_params),
            })

    # Save results
    metrics_df = pd.DataFrame(metrics_rows)
    reduction_df = pd.DataFrame(reduction_rows)
    metrics_df.to_csv(DATA_PROCESSED / "metrics.csv", index=False)
    reduction_df.to_csv(DATA_PROCESSED / "reduction_metrics.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print("=" * 60)
    print(f"\nMetrics saved to: {DATA_PROCESSED / 'metrics.csv'}")
    print(f"Reduction metrics saved to: {DATA_PROCESSED / 'reduction_metrics.csv'}")

    # Find best combination
    best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    print(f"\nBest combination: {best_row['Reducer']}-{best_row['Actual_dims']}D + {best_row['Classifier']}")
    print(f"  Accuracy: {best_row['Accuracy']:.4f}")
    print(f"  FPR: {best_row['FPR']:.4f}")
    print(f"  FNR: {best_row['FNR']:.4f}")

    # Per-attack metrics (as required: 针对DDoS等高发攻击类型单独统计检测精度)
    print(f"\n{'=' * 60}")
    print("Computing per-attack metrics for best model...")
    print("=" * 60)

    # Refit best model
    reducer = reducer_factory(
        best_row["Reducer"], int(best_row["n_components"]), n_classes
    )
    if best_row["Reducer"] == "t-SNE":
        X_full = np.vstack([X_train, X_test])
        X_red = reducer.fit_transform(X_full)
        X_train_red = X_red[: len(X_train)]
        X_test_red = X_red[len(X_train):]
    else:
        X_train_red = reducer.fit_transform(X_train, y_train)
        X_test_red = reducer.transform(X_test)

    clf_name = best_row["Classifier"]
    clf_config = classifiers[clf_name]
    model, _, _ = grid_search(
        clf_config["model_cls"],
        clf_config["param_grid"],
        X_train_red,
        y_train,
        cv=3,
    )
    y_pred = model.predict(X_test_red)

    # Compute metrics for all attack types (exclude Benign)
    labels = np.unique(y_test)
    attack_labels = [l for l in labels if l != "Benign"]

    if attack_labels:
        pr, rc, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=attack_labels, average=None, zero_division=0
        )
        attack_df = pd.DataFrame({
            "Attack_type": attack_labels,
            "Precision": pr.astype(float),
            "Recall": rc.astype(float),
            "F1": f1.astype(float),
            "Support": support.astype(int),
        }).sort_values("Attack_type")

        attack_df.to_csv(DATA_PROCESSED / "attack_metrics.csv", index=False)
        print(f"\nPer-attack metrics:")
        print(attack_df.to_string(index=False))
        print(f"\nSaved to: {DATA_PROCESSED / 'attack_metrics.csv'}")

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
