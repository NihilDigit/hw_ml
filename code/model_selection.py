from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_SEED


def grid_search_sklearn(
    model_cls: Callable[..., Any],
    param_grid: Iterable[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any], float]:
    """Lightweight grid search (macro-F1) for sklearn-compatible estimators."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: dict[str, Any] | None = None

    for params in param_grid:
        scores: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            model = model_cls(**params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            scores.append(float(f1_score(y[val_idx], y_pred, average="macro", zero_division=0)))

        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise ValueError("Empty param grid.")

    best_model = model_cls(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score


def grid_search_cuml(
    model_cls: Callable[..., Any],
    param_grid: Iterable[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any], float]:
    """Lightweight grid search (macro-F1) for cuML estimators.

    cuML often returns device arrays; this helper normalizes predictions to numpy.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: dict[str, Any] | None = None

    for params in param_grid:
        scores: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            model = model_cls(**params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            scores.append(float(f1_score(y[val_idx], y_pred, average="macro", zero_division=0)))

        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise ValueError("Empty param grid.")

    best_model = model_cls(**best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_score

