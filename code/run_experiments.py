from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    DATA_PROCESSED,
    DATA_RAW,
    FEATURES,
    FIGURES_DIR,
    LABEL_COL,
    RANDOM_SEED,
)
from metrics import binary_rates_from_multiclass, overall_accuracy
from plots import plot_2d_scatter
from preprocess import load_and_clean


def class_separation_ratio(X, y):
    labels = np.unique(y)
    centroids = {}
    for label in labels:
        centroids[label] = X[y == label].mean(axis=0)
    # within-class scatter
    within = 0.0
    for label in labels:
        diffs = X[y == label] - centroids[label]
        within += np.sum(np.linalg.norm(diffs, axis=1) ** 2)
    within = within / max(len(X), 1)
    # between-class scatter
    all_centroid = X.mean(axis=0)
    between = 0.0
    for label in labels:
        n = (y == label).sum()
        diff = centroids[label] - all_centroid
        between += n * (np.linalg.norm(diff) ** 2)
    between = between / max(len(X), 1)
    return between / within if within > 0 else 0.0


def reducer_factory(name, n_components, n_classes):
    if name == "PCA":
        return PCA(n_components=n_components, svd_solver="auto", whiten=False)
    if name == "LDA":
        # LDA components cannot exceed n_classes - 1
        n_comp = min(n_components, max(n_classes - 1, 1))
        return LinearDiscriminantAnalysis(n_components=n_comp, solver="svd")
    if name == "t-SNE":
        return TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            n_iter=1000,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown reducer: {name}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_RAW / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    df = load_and_clean(str(csv_path))

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reducers = ["PCA", "LDA", "t-SNE"]
    dims = [10, 15, 20]

    classifiers = {
        "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        ),
    }

    metrics_rows = []
    reduction_rows = []

    n_classes = len(np.unique(y))

    for reducer_name in reducers:
        for n_comp in dims:
            reducer = reducer_factory(reducer_name, n_comp, n_classes)

            if reducer_name == "t-SNE":
                X_full = np.vstack([X_train, X_test])
                y_full = np.concatenate([y_train, y_test])
                X_red = reducer.fit_transform(X_full)
                X_train_red = X_red[: len(X_train)]
                X_test_red = X_red[len(X_train) :]
            else:
                X_train_red = reducer.fit_transform(X_train, y_train)
                X_test_red = reducer.transform(X_test)

            # Reduction metrics
            info_retention = ""
            if reducer_name in {"PCA", "LDA"} and hasattr(reducer, "explained_variance_ratio_"):
                info_retention = float(np.sum(reducer.explained_variance_ratio_))
            elif reducer_name == "t-SNE":
                info_retention = float(trustworthiness(X_full, X_red, n_neighbors=5))

            sep_ratio = class_separation_ratio(X_train_red, y_train)
            reduction_rows.append(
                {
                    "Reducer": reducer_name,
                    "n_components": n_comp,
                    "Information_retention": info_retention,
                    "Class_separation": sep_ratio,
                }
            )

            # 2D visualization for each reducer at n_components >= 2
            if X_train_red.shape[1] >= 2:
                sample_idx = np.random.RandomState(RANDOM_SEED).choice(
                    len(X_train_red), size=min(5000, len(X_train_red)), replace=False
                )
                viz_df = pd.DataFrame(X_train_red[sample_idx, :2], columns=["c1", "c2"])
                viz_df[LABEL_COL] = y_train[sample_idx]
                plot_path = FIGURES_DIR / f"{reducer_name}_{n_comp}_2d.png"
                plot_2d_scatter(viz_df, LABEL_COL, plot_path)

            for clf_name, (clf, grid) in classifiers.items():
                start_train = time.time()
                search = GridSearchCV(
                    clf, grid, cv=3, scoring="f1_macro", n_jobs=-1, refit=True
                )
                search.fit(X_train_red, y_train)
                train_time = time.time() - start_train

                start_pred = time.time()
                y_pred = search.predict(X_test_red)
                pred_time = time.time() - start_pred

                acc = overall_accuracy(y_test, y_pred)
                fpr, fnr = binary_rates_from_multiclass(y_test, y_pred)

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
                        "Best_params": search.best_params_,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    reduction_df = pd.DataFrame(reduction_rows)
    metrics_df.to_csv(DATA_PROCESSED / "metrics.csv", index=False)
    reduction_df.to_csv(DATA_PROCESSED / "reduction_metrics.csv", index=False)

    # Attack-specific metrics for best combination (by Accuracy)
    best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_mask = (
        (metrics_df["Reducer"] == best_row["Reducer"])
        & (metrics_df["n_components"] == best_row["n_components"])
        & (metrics_df["Classifier"] == best_row["Classifier"])
    )
    if best_mask.any():
        # Refit best model for per-attack metrics
        reducer = reducer_factory(best_row["Reducer"], int(best_row["n_components"]), n_classes)
        if best_row["Reducer"] == "t-SNE":
            X_full = np.vstack([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            X_red = reducer.fit_transform(X_full)
            X_train_red = X_red[: len(X_train)]
            X_test_red = X_red[len(X_train) :]
        else:
            X_train_red = reducer.fit_transform(X_train, y_train)
            X_test_red = reducer.transform(X_test)

        clf_name = best_row["Classifier"]
        clf, grid = classifiers[clf_name]
        search = GridSearchCV(clf, grid, cv=3, scoring="f1_macro", n_jobs=-1, refit=True)
        search.fit(X_train_red, y_train)
        y_pred = search.predict(X_test_red)

        attack_label = "DDoS"
        labels = np.unique(y_test)
        if attack_label in labels:
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=[attack_label], average=None
            )
            attack_df = pd.DataFrame(
                {
                    "Attack_type": [attack_label],
                    "Precision": [float(pr[0])],
                    "Recall": [float(rc[0])],
                    "F1": [float(f1[0])],
                }
            )
            attack_df.to_csv(DATA_PROCESSED / "attack_metrics.csv", index=False)


if __name__ == "__main__":
    main()
