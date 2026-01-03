"""Plot result figures from processed CSV outputs.

Reads:
  - data/processed/metrics.csv
  - data/processed/reduction_metrics.csv
  - data/processed/attack_metrics.csv (optional)

Writes figures into figures/ using SciencePlots IEEE style.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, FIGURES_DIR, LABEL_COL
from plots import setup_ieee_style


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_combination(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    df = metrics.copy()
    clf_short = {"RandomForest": "RF", "LogisticRegression": "LR", "SVM": "SVM"}
    df["Combo"] = (
        df["Reducer"].astype(str)
        + "-"
        + df["n_components"].astype(int).astype(str)
        + " + "
        + df["Classifier"].map(clf_short).fillna(df["Classifier"].astype(str))
    )
    df = df.sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(df["Combo"], df["Accuracy"], color="#4C78A8")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Reducer + Classifier")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    _savefig(out_path)


def plot_tradeoff_fpr_fnr(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    df = metrics.copy()
    df["Label"] = (
        df["Reducer"].astype(str)
        + "-"
        + df["n_components"].astype(int).astype(str)
        + "D/"
        + df["Classifier"].astype(str)
    )

    classifier_palette = {
        "SVM": "#F58518",
        "RandomForest": "#54A24B",
        "LogisticRegression": "#B279A2",
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for clf, sub in df.groupby("Classifier"):
        ax.scatter(
            sub["FPR"],
            sub["FNR"],
            s=30,
            alpha=0.9,
            label=clf,
            color=classifier_palette.get(clf, None),
        )

    # Annotate the top-accuracy point for readability
    best = df.sort_values("Accuracy", ascending=False).iloc[0]
    ax.annotate(
        best["Label"],
        xy=(best["FPR"], best["FNR"]),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("False Negative Rate (FNR)")
    ax.set_xlim(0.0, max(0.30, float(df["FPR"].max()) * 1.1))
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, ncol=1)
    _savefig(out_path)


def plot_predict_time(metrics: pd.DataFrame, out_path: Path, test_size: int = 65637) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    df = metrics.copy()
    df["ms_per_sample"] = df["Predict_time_s"] / float(test_size) * 1000.0
    clf_short = {"RandomForest": "RF", "LogisticRegression": "LR", "SVM": "SVM"}
    df["Combo"] = (
        df["Reducer"].astype(str)
        + "-"
        + df["n_components"].astype(int).astype(str)
        + " + "
        + df["Classifier"].map(clf_short).fillna(df["Classifier"].astype(str))
    )
    df = df.sort_values("ms_per_sample", ascending=True)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(df["Combo"], df["ms_per_sample"], color="#72B7B2")
    ax.set_ylabel("Prediction latency (ms/sample)")
    ax.set_xlabel("Prediction latency (ms/sample)")
    ax.tick_params(axis="y", labelsize=8)
    _savefig(out_path)


def plot_pca_information_retention(reduction: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    pca = reduction[reduction["Reducer"] == "PCA"].copy()
    pca = pca.sort_values("n_components")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(
        pca["n_components"].astype(int).to_numpy(),
        pca["Information_retention"].astype(float).to_numpy(),
        marker="o",
        color="#4C78A8",
    )
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Information retention")
    ax.set_ylim(0.90, 1.005)
    ax.grid(True, alpha=0.25)
    _savefig(out_path)


def plot_class_distribution(clean_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    counts = clean_df[LABEL_COL].value_counts()

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(counts.index.astype(str), counts.values.astype(int), color="#4C78A8")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    _savefig(out_path)


def plot_pipeline_diagram(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_ieee_style()

    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    ax.axis("off")

    boxes = [
        "Raw CSV",
        "Cleaning\n+ Feature selection",
        "Min-Max scaling",
        "Dimensionality\nreduction",
        "Classifier\n(GridSearchCV)",
        "Metrics\n+ Figures",
    ]

    xs = np.linspace(0.05, 0.95, len(boxes))
    y = 0.55
    for i, (x, text) in enumerate(zip(xs, boxes)):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.06, y),
                xytext=(x + 0.06, y),
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

    _savefig(out_path)


def main() -> None:
    metrics_path = DATA_PROCESSED / "metrics.csv"
    reduction_path = DATA_PROCESSED / "reduction_metrics.csv"

    metrics = pd.read_csv(metrics_path)
    reduction = pd.read_csv(reduction_path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_combination(metrics, FIGURES_DIR / "Accuracy_by_Combination.png")
    plot_tradeoff_fpr_fnr(metrics, FIGURES_DIR / "Tradeoff_FPR_vs_FNR.png")
    # The current experiment uses a 3,000-sample subset with an 80/20 split.
    plot_predict_time(metrics, FIGURES_DIR / "PredictLatency_ms_per_sample.png", test_size=600)
    plot_pca_information_retention(reduction, FIGURES_DIR / "PCA_InformationRetention.png")

    # Class distribution after applying the project's fixed cleaning rules.
    from preprocess import load_and_clean

    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    clean_df = load_and_clean(str(csv_path))
    plot_class_distribution(clean_df, FIGURES_DIR / "ClassDistribution_AfterCleaning.png")

    plot_pipeline_diagram(FIGURES_DIR / "Pipeline_Overview.png")

    print("Saved figures:")
    for name in [
        "Accuracy_by_Combination.png",
        "Tradeoff_FPR_vs_FNR.png",
        "PredictLatency_ms_per_sample.png",
        "PCA_InformationRetention.png",
        "ClassDistribution_AfterCleaning.png",
        "Pipeline_Overview.png",
    ]:
        print(f"  - {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
