from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def setup_ieee_style():
    # SciencePlots registers styles at import time.
    try:
        import scienceplots  # noqa: F401
    except Exception as exc:
        raise ImportError("SciencePlots is required to use the 'science' and 'ieee' styles.") from exc
    plt.style.use(["science", "ieee"])


def plot_2d_scatter(df_2d: pd.DataFrame, label_col: str, out_path):
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=150)
    labels = df_2d[label_col].unique()
    for label in labels:
        sub = df_2d[df_2d[label_col] == label]
        ax.scatter(sub.iloc[:, 0], sub.iloc[:, 1], s=6, alpha=0.7, label=str(label))
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
