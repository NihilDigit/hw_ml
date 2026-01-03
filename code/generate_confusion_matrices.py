"""Generate confusion matrix figures for selected model(s).

This script is intentionally separate from the main experiment runner so that
figures can be regenerated without re-running the full grid-search pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from config import DATA_PROCESSED, FIGURES_DIR, FEATURES, LABEL_COL, RANDOM_SEED
from plots import setup_ieee_style
from preprocess import load_and_clean
from pipeline_utils import fit_minmax_scaler, split_train_test, transform_with_scaler
from torch_reducers import TorchLDA, TorchPCA


def _abbrev_label(label: str) -> str:
    mapping = {
        "Benign": "Benign",
        "Bot": "Bot",
        "Brute Force -Web": "BF-Web",
        "Brute Force -XSS": "BF-XSS",
        "DDOS attack-HOIC": "DDoS-HOIC",
        "DDOS attack-LOIC-UDP": "DDoS-UDP",
        "DDoS attacks-LOIC-HTTP": "DDoS-HTTP",
        "DoS attacks-GoldenEye": "DoS-GE",
        "DoS attacks-Hulk": "DoS-Hulk",
        "DoS attacks-SlowHTTPTest": "DoS-SlowHTTP",
        "DoS attacks-Slowloris": "DoS-Slowloris",
        "FTP-BruteForce": "FTP-BF",
        "Infilteration": "Infil",
        "SQL Injection": "SQLi",
        "SSH-Bruteforce": "SSH-BF",
    }
    short = mapping.get(label, label)
    return short if len(short) <= 14 else short[:13] + "…"


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    display_labels = [_abbrev_label(l) for l in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(display_labels)), labels=display_labels, rotation=60, ha="right")
    ax.set_yticks(range(len(display_labels)), labels=display_labels)
    ax.tick_params(axis="both", labelsize=7)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate cells
    vmax = float(cm.max()) if cm.size else 0.0
    threshold = vmax * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0:
                continue
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
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    df = load_and_clean(str(csv_path))

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    split = split_train_test(X, y, test_size=0.2, seed=RANDOM_SEED)
    X_train, X_test, y_train, y_test = split.X_train, split.X_test, split.y_train, split.y_test

    scaler = fit_minmax_scaler(X_train)
    X_train = transform_with_scaler(scaler, X_train)
    X_test = transform_with_scaler(scaler, X_test)

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
