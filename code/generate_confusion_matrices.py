"""Generate confusion matrix figures for selected model(s).

This script is intentionally separate from the main experiment runner so that
figures can be regenerated without re-running the full grid-search pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from config import DATA_PROCESSED, FIGURES_DIR, FEATURES, LABEL_COL, RANDOM_SEED
from plots import setup_ieee_style
from torch_reducers import TorchLDA, TorchPCA


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate cells
    vmax = float(cm.max()) if cm.size else 0.0
    threshold = vmax * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if val > threshold else "black"),
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_PROCESSED / "ids2018_subset_10k.csv"
    df = pd.read_csv(csv_path)

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    labels = ["Benign"] + [l for l in np.unique(y_test) if l != "Benign"]

    # Baseline: PCA-10D + RandomForest (strong and stable across reducers)
    pca = TorchPCA(n_components=10, device=device)
    X_train_red = pca.fit_transform(X_train)
    X_test_red = pca.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_red, y_train)
    y_pred = rf.predict(X_test_red)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    out_path = FIGURES_DIR / "ConfusionMatrix_PCA10_RandomForest.png"
    plot_confusion_matrix(cm, labels, out_path)

    # Additional comparisons (cheap to compute) for analysis:
    #   - PCA-10D + SVM (RBF)
    #   - LDA-10D + RandomForest (capped at n_classes - 1)
    svm = SVC(C=10, kernel="rbf", gamma="scale", random_state=RANDOM_SEED)
    svm.fit(X_train_red, y_train)
    y_pred_svm = svm.predict(X_test_red)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels)
    out_path_svm = FIGURES_DIR / "ConfusionMatrix_PCA10_SVM.png"
    plot_confusion_matrix(cm_svm, labels, out_path_svm)

    lda = TorchLDA(n_components=10, device=device)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    rf_lda = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_lda.fit(X_train_lda, y_train)
    y_pred_rf_lda = rf_lda.predict(X_test_lda)
    cm_rf_lda = confusion_matrix(y_test, y_pred_rf_lda, labels=labels)
    out_path_rf_lda = FIGURES_DIR / "ConfusionMatrix_LDA10_RandomForest.png"
    plot_confusion_matrix(cm_rf_lda, labels, out_path_rf_lda)

    print(
        "Saved confusion matrices to:\n"
        f"  - {out_path}\n"
        f"  - {out_path_svm}\n"
        f"  - {out_path_rf_lda}"
    )


if __name__ == "__main__":
    main()
