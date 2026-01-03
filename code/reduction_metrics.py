from __future__ import annotations

import numpy as np


def class_separation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Between-class / within-class scatter ratio on a feature representation."""
    labels = np.unique(y)
    centroids = {label: X[y == label].mean(axis=0) for label in labels}

    within = 0.0
    for label in labels:
        diffs = X[y == label] - centroids[label]
        within += float(np.sum(np.linalg.norm(diffs, axis=1) ** 2))
    within = within / max(len(X), 1)

    all_centroid = X.mean(axis=0)
    between = 0.0
    for label in labels:
        n = int((y == label).sum())
        diff = centroids[label] - all_centroid
        between += float(n * (np.linalg.norm(diff) ** 2))
    between = between / max(len(X), 1)

    return between / within if within > 0 else 0.0

