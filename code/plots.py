from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_ieee_style():
    # Ensure Matplotlib can write its cache/config in sandboxed environments.
    os.environ.setdefault(
        "MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / ".mplconfig").resolve())
    )

    # SciencePlots registers styles at import time.
    try:
        import scienceplots  # noqa: F401
    except Exception as exc:
        raise ImportError("SciencePlots is required to use the 'science' and 'ieee' styles.") from exc
    plt.style.use(["science", "ieee"])
    # Avoid requiring a LaTeX installation (and TeX cache writes) for figure rendering.
    plt.rcParams.update({"text.usetex": False})
    # Prefer fonts that exist in minimal environments while keeping an IEEE-like serif look.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times"]})


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
