"""PyTorch GPU-accelerated dimensionality reduction implementations."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class TorchPCA:
    """GPU-accelerated PCA using PyTorch."""

    def __init__(self, n_components: int, device: str = "cuda"):
        self.n_components = n_components
        self.device = device
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, y=None):
        """Fit PCA on data."""
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Center the data
        self.mean_ = X_tensor.mean(dim=0)
        X_centered = X_tensor - self.mean_

        # Compute SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Store components
        self.components_ = Vt[:self.n_components]

        # Compute explained variance
        explained_variance = (S ** 2) / (X_tensor.shape[0] - 1)
        self.explained_variance_ = explained_variance[:self.n_components]
        total_variance = explained_variance.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to reduced dimensions."""
        X_tensor = torch.from_numpy(X).float().to(self.device)
        X_centered = X_tensor - self.mean_
        X_transformed = X_centered @ self.components_.T
        return X_transformed.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X, y)
        return self.transform(X)


class TorchLDA:
    """GPU-accelerated LDA using PyTorch."""

    def __init__(self, n_components: int, device: str = "cuda"):
        self.n_components = n_components
        self.device = device
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LDA on data."""
        from sklearn.preprocessing import LabelEncoder
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Encode labels if they are strings/objects
        if y.dtype == object or y.dtype.kind in ['U', 'S']:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y

        y_tensor = torch.from_numpy(y_encoded).long().to(self.device)

        classes = torch.unique(y_tensor)
        n_classes = len(classes)
        n_features = X_tensor.shape[1]

        # Limit components
        max_components = min(n_classes - 1, n_features)
        n_components = min(self.n_components, max_components)

        # Overall mean
        mean_overall = X_tensor.mean(dim=0)

        # Within-class scatter matrix
        S_W = torch.zeros(n_features, n_features, device=self.device)
        # Between-class scatter matrix
        S_B = torch.zeros(n_features, n_features, device=self.device)

        for c in classes:
            X_c = X_tensor[y_tensor == c]
            mean_c = X_c.mean(dim=0)

            # Within-class scatter
            X_c_centered = X_c - mean_c
            S_W += X_c_centered.T @ X_c_centered

            # Between-class scatter
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).unsqueeze(1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Solve generalized eigenvalue problem: S_B @ w = lambda * S_W @ w
        # Add regularization to S_W for numerical stability
        S_W_reg = S_W + 1e-6 * torch.eye(n_features, device=self.device)

        # Compute S_W^-1 @ S_B
        try:
            S_W_inv = torch.linalg.inv(S_W_reg)
            M = S_W_inv @ S_B

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(M)

            # Sort by eigenvalues (descending)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top components
            self.components_ = eigenvectors[:, :n_components].T

            # Compute explained variance ratio
            eigenvalues_pos = torch.clamp(eigenvalues[:n_components], min=0)
            total_variance = eigenvalues_pos.sum()
            self.explained_variance_ratio_ = eigenvalues_pos / (total_variance + 1e-10)

        except Exception:
            # Fallback to scikit-learn if numerical issues
            print("Warning: Numerical issues in TorchLDA, falling back to scikit-learn")
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA
            lda = SKLDA(n_components=n_components)
            lda.fit(X, y)
            self.components_ = torch.from_numpy(lda.scalings_.T).float().to(self.device)
            self.explained_variance_ratio_ = torch.from_numpy(lda.explained_variance_ratio_).float().to(self.device)

        self.mean_ = mean_overall
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to reduced dimensions."""
        X_tensor = torch.from_numpy(X).float().to(self.device)
        X_transformed = (X_tensor - self.mean_) @ self.components_.T
        return X_transformed.cpu().numpy()

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X, y)
        return self.transform(X)


class TorchTSNE:
    """GPU-accelerated t-SNE using PyTorch."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.device = device
        self.random_state = random_state
        self.embedding_ = None

    def _compute_pairwise_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances."""
        sum_X = (X ** 2).sum(dim=1)
        D = sum_X.unsqueeze(0) + sum_X.unsqueeze(1) - 2 * (X @ X.T)
        return torch.clamp(D, min=0.0)

    def _compute_joint_probabilities(self, D: torch.Tensor) -> torch.Tensor:
        """Compute joint probabilities from distances."""
        n = D.shape[0]

        # Binary search for sigma (perplexity-based)
        P = torch.zeros_like(D)
        beta = torch.ones(n, device=self.device)
        target_entropy = np.log(self.perplexity)

        for i in range(n):
            beta_min = torch.tensor(-np.inf, device=self.device)
            beta_max = torch.tensor(np.inf, device=self.device)
            Di = D[i].clone()
            Di[i] = 0

            for _ in range(50):  # Binary search iterations
                P_i = torch.exp(-Di * beta[i])
                P_i[i] = 0
                sum_P_i = P_i.sum()

                if sum_P_i == 0:
                    P_i = torch.ones_like(P_i) / n
                    sum_P_i = 1.0

                H = torch.log(sum_P_i) + beta[i] * (Di * P_i).sum() / sum_P_i
                H_diff = H - target_entropy

                if abs(H_diff.item()) < 1e-5:
                    break

                if H_diff > 0:
                    beta_min = beta[i]
                    beta[i] = (beta[i] + beta_max) / 2 if not torch.isinf(beta_max) else beta[i] * 2
                else:
                    beta_max = beta[i]
                    beta[i] = (beta[i] + beta_min) / 2 if not torch.isinf(beta_min) else beta[i] / 2

            P[i] = P_i / sum_P_i

        # Symmetrize
        P = (P + P.T) / (2 * n)
        P = torch.clamp(P, min=1e-12)
        return P

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform data using t-SNE."""
        torch.manual_seed(self.random_state)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        n = X_tensor.shape[0]

        # Compute pairwise distances in high-dimensional space
        D_high = self._compute_pairwise_distances(X_tensor)

        # Compute joint probabilities
        P = self._compute_joint_probabilities(D_high)

        # Initialize embedding with PCA
        pca = TorchPCA(n_components=self.n_components, device=self.device)
        Y = torch.from_numpy(pca.fit_transform(X)).float().to(self.device)
        Y = Y * 1e-4  # Small initialization

        # Optimize
        Y.requires_grad_(True)
        optimizer = torch.optim.Adam([Y], lr=self.learning_rate)

        for iteration in range(self.n_iter):
            optimizer.zero_grad()

            # Compute pairwise distances in low-dimensional space
            D_low = self._compute_pairwise_distances(Y)

            # Compute Q (Student t-distribution)
            Q = 1 / (1 + D_low)
            Q.fill_diagonal_(0)
            Q = Q / Q.sum()
            Q = torch.clamp(Q, min=1e-12)

            # Compute KL divergence
            loss = (P * torch.log(P / Q)).sum()

            loss.backward()
            optimizer.step()

            if (iteration + 1) % 100 == 0:
                print(f"t-SNE iteration {iteration + 1}/{self.n_iter}, loss: {loss.item():.4f}")

        self.embedding_ = Y.detach()
        return self.embedding_.cpu().numpy()
