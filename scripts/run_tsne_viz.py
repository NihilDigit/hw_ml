"""Run t-SNE (2D) for visualization only."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))

from config import DATA_PROCESSED, FEATURES, LABEL_COL, FIGURES_DIR, RANDOM_SEED
from plots import plot_2d_scatter
from preprocess import load_and_clean
from pipeline_utils import build_2d_viz_frame, fit_minmax_scaler, transform_with_scaler


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    df = load_and_clean(str(csv_path))

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    scaler = fit_minmax_scaler(X)
    X = transform_with_scaler(scaler, X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=RANDOM_SEED,
    )

    X_emb = tsne.fit_transform(X)

    viz_df = build_2d_viz_frame(X_emb, y)

    out_path = FIGURES_DIR / "t-SNE_2_2d.png"
    plot_2d_scatter(viz_df, LABEL_COL, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
