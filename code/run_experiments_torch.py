"""PyTorch GPU-accelerated experiment pipeline for intrusion detection."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE as SklearnTSNE

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
from torch_reducers import TorchPCA, TorchLDA, TorchTSNE
from torch_classifiers import TorchLogisticRegression, grid_search_torch_lr

try:
    from cuml.manifold import TSNE as CuMLTSNE
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
    from cuml.svm import SVC as CuMLSVC
except Exception:
    CuMLTSNE = None
    CuMLRandomForestClassifier = None
    CuMLLogisticRegression = None
    CuMLSVC = None


def class_separation_ratio(X, y):
    """Compute class separation ratio (between-class / within-class scatter)."""
    labels = np.unique(y)
    centroids = {}
    for label in labels:
        centroids[label] = X[y == label].mean(axis=0)

    # Within-class scatter
    within = 0.0
    for label in labels:
        diffs = X[y == label] - centroids[label]
        within += np.sum(np.linalg.norm(diffs, axis=1) ** 2)
    within = within / max(len(X), 1)

    # Between-class scatter
    all_centroid = X.mean(axis=0)
    between = 0.0
    for label in labels:
        n = (y == label).sum()
        diff = centroids[label] - all_centroid
        between += n * (np.linalg.norm(diff) ** 2)
    between = between / max(len(X), 1)

    return between / within if within > 0 else 0.0


def reducer_factory(name, n_components, n_classes, device="cuda"):
    """Create dimensionality reduction model."""
    if name == "PCA":
        return TorchPCA(n_components=n_components, device=device)
    if name == "LDA":
        # LDA components cannot exceed n_classes - 1
        n_comp = min(n_components, max(n_classes - 1, 1))
        return TorchLDA(n_components=n_comp, device=device)
    if name == "t-SNE":
        # Use sklearn t-SNE (CPU) - stable and sufficient for 3k samples
        return SklearnTSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate="auto",
            max_iter=1000,
            init="pca",
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown reducer: {name}")


def grid_search_cuml(model_cls, param_grid_list, X, y, cv=3, seed=RANDOM_SEED):
    """Simple grid search for cuML models using macro F1."""
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
            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_model = model_cls(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score


def grid_search_sklearn(model_cls, param_grid_list, X, y, cv=3, seed=RANDOM_SEED):
    """Simple grid search for sklearn models using macro F1."""
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

    best_model = model_cls(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score


def main():
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    # Use the stratified 10k subset for multi-class experiments
    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    print(f"Loading data from {csv_path}...")
    df = load_and_clean(str(csv_path))
    print(f"Loaded {len(df)} samples with {len(df[LABEL_COL].unique())} classes")

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Sample training set to 10k for faster experimentation and t-SNE feasibility
    # Normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

    if CuMLRandomForestClassifier is None or CuMLLogisticRegression is None:
        raise RuntimeError("cuML is required for GPU classifiers but could not be imported.")

    # Classifier configurations (GPU via cuML)
    # Note: cuML SVC has a known bug with multi-class classification, so we use sklearn SVC
    classifiers = {
        "SVM": {
            "type": "sklearn",
            "model_cls": SVC,
            "param_grid": [
                {"C": 1, "kernel": "rbf", "gamma": "scale"},
                {"C": 10, "kernel": "rbf", "gamma": "scale"},
                {"C": 1, "kernel": "linear"},
                {"C": 10, "kernel": "linear"},
            ],
        },
        "RandomForest": {
            "type": "cuml",
            "model_cls": CuMLRandomForestClassifier,
            "param_grid": [
                {"n_estimators": 200, "max_depth": 10, "n_streams": 1, "random_state": RANDOM_SEED},
                {"n_estimators": 200, "max_depth": 20, "n_streams": 1, "random_state": RANDOM_SEED},
            ],
        },
        "LogisticRegression": {
            "type": "cuml",
            "model_cls": CuMLLogisticRegression,
            "param_grid": [
                {"C": 0.1, "max_iter": 1000},
                {"C": 1.0, "max_iter": 1000},
                {"C": 10.0, "max_iter": 1000},
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
            if reducer_name == "t-SNE":
                # t-SNE requires full dataset
                X_full = np.vstack([X_train, X_test])
                y_full = np.concatenate([y_train, y_test])
                X_red = reducer.fit_transform(X_full)
                # cuML t-SNE returns cupy array, convert to numpy
                if hasattr(X_red, "get"):
                    X_red = X_red.get()
                X_train_red = X_red[: len(X_train)]
                X_test_red = X_red[len(X_train) :]
                # Clean up GPU memory after t-SNE
                del reducer
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            else:
                X_train_red = reducer.fit_transform(X_train, y_train)
                X_test_red = reducer.transform(X_test)

            print(f"Reduced to shape: {X_train_red.shape}")

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
            elif reducer_name == "t-SNE":
                # Note: trustworthiness is expensive to compute, using placeholder
                info_retention = 0.0  # Can compute later if needed

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
                viz_df = pd.DataFrame(X_train_red[sample_idx, :2], columns=["c1", "c2"])
                viz_df[LABEL_COL] = y_train[sample_idx]
                plot_path = FIGURES_DIR / f"{reducer_name}_{n_comp}_2d.png"
                plot_2d_scatter(viz_df, LABEL_COL, plot_path)
                print(f"Saved plot to {plot_path}")

            # Cache the reduction for reuse
            processed_reductions[reduction_key] = (X_train_red, X_test_red)

        # Train classifiers on reduced data
        for clf_name, clf_config in classifiers.items():
            print(f"\nTraining {clf_name}...")

            start_train = time.time()

            if clf_config["type"] == "cuml":
                model, best_params, _ = grid_search_cuml(
                    clf_config["model_cls"],
                    clf_config["param_grid"],
                    X_train_red,
                    y_train,
                    cv=3,
                )
            elif clf_config["type"] == "sklearn":
                model, best_params, _ = grid_search_sklearn(
                    clf_config["model_cls"],
                    clf_config["param_grid"],
                    X_train_red,
                    y_train,
                    cv=3,
                )
            elif clf_config["type"] == "cuml_or_sklearn":
                try:
                    model, best_params, _ = grid_search_cuml(
                        clf_config["model_cls"],
                        clf_config["param_grid"],
                        X_train_red,
                        y_train,
                        cv=3,
                    )
                except Exception as err:
                    print(f"cuML {clf_name} failed ({err}); falling back to sklearn...")
                    model, best_params, _ = grid_search_sklearn(
                        clf_config["fallback_cls"],
                        clf_config["fallback_param_grid"],
                        X_train_red,
                        y_train,
                        cv=3,
                    )

            train_time = time.time() - start_train

            # Prediction
            start_pred = time.time()
            y_pred = model.predict(X_test_red)
            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            pred_time = time.time() - start_pred

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
    reducer = reducer_factory(
        best_row["Reducer"], int(best_row["n_components"]), n_classes, device
    )

    if best_row["Reducer"] == "t-SNE":
        X_full = np.vstack([X_train, X_test])
        X_red = reducer.fit_transform(X_full)
        X_train_red = X_red[: len(X_train)]
        X_test_red = X_red[len(X_train) :]
    else:
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
