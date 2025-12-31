"""PyTorch GPU-accelerated classifiers and sklearn wrappers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any


class TorchLogisticRegression:
    """GPU-accelerated Logistic Regression using PyTorch."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.device = device
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit logistic regression model."""
        torch.manual_seed(self.random_state)

        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y_encoded).long().to(self.device)

        n_features = X.shape[1]

        # Create model
        self.model = nn.Linear(n_features, n_classes).to(self.device)

        # Optimizer with L2 regularization
        weight_decay = 1.0 / self.C
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=10,
        )

        # Training
        def closure():
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = F.cross_entropy(logits, y_tensor)
            # Add L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + weight_decay * l2_reg
            loss.backward()
            return loss

        for _ in range(self.max_iter // 20):
            optimizer.step(closure)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

        # Decode back to original labels
        predictions_np = predictions.cpu().numpy()
        return self.label_encoder.inverse_transform(predictions_np)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = F.softmax(logits, dim=1)

        return probas.cpu().numpy()


def grid_search_torch_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 3,
    device: str = "cuda",
) -> tuple[TorchLogisticRegression, Dict[str, Any]]:
    """Simple grid search for TorchLogisticRegression."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    best_score = -np.inf
    best_params = None
    best_model = None

    # Generate all parameter combinations
    param_combinations = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    def generate_combinations(idx, current):
        if idx == len(keys):
            param_combinations.append(current.copy())
            return
        for val in values[idx]:
            current[keys[idx]] = val
            generate_combinations(idx + 1, current)

    generate_combinations(0, {})

    # Encode labels for stratification
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    for params in param_combinations:
        scores = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(X_train, y_encoded):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = TorchLogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                device=device,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            score = f1_score(y_val, y_pred, average='macro')
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit on full training data
    best_model = TorchLogisticRegression(
        C=best_params.get('C', 1.0),
        max_iter=best_params.get('max_iter', 1000),
        device=device,
    )
    best_model.fit(X_train, y_train)

    return best_model, best_params
