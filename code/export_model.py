from __future__ import annotations

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_PROCESSED,
    FEATURES,
    LABEL_COL,
    RANDOM_SEED,
)
from preprocess import load_and_clean
from torch_reducers import TorchLDA
from pipeline_utils import (
    fit_minmax_scaler,
    select_torch_device,
    transform_with_scaler,
)

# Try to use cuml if available
try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    USE_CUML = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_CUML = False

def main():
    device = select_torch_device()
    print(f"Using device: {device}")
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = DATA_PROCESSED / "ids2018_subset_3k.csv"
    print(f"Loading data from {csv_path}...")
    df = load_and_clean(str(csv_path))
    
    X = df[FEATURES].values
    y = df[LABEL_COL].values
    
    # 1. Train Scaler
    print("Fitting MinMaxScaler...")
    scaler = fit_minmax_scaler(X)
    X_scaled = transform_with_scaler(scaler, X)
    joblib.dump(scaler, model_dir / "minmax_scaler.joblib")
    
    # 2. Train LDA (10D)
    print("Fitting TorchLDA (10D)...")
    n_classes = len(np.unique(y))
    reducer = TorchLDA(n_components=10, device=device)
    X_red = reducer.fit_transform(X_scaled, y)
    
    # Save LDA components and metadata
    lda_data = {
        "components": reducer.components_,
        "mean": reducer.mean_,
        "n_components": reducer.n_components,
        "explained_variance_ratio": reducer.explained_variance_ratio_
    }
    torch.save(lda_data, model_dir / "lda_10d.pt")
    
    # 3. Train RandomForest
    print(f"Fitting RandomForest (USE_CUML={USE_CUML})...")
    if USE_CUML:
        rf = CuMLRandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            n_streams=1, 
            random_state=RANDOM_SEED
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    
    rf.fit(X_red, y)
    joblib.dump(rf, model_dir / "random_forest.joblib")
    
    # 4. Save Label Encoding info
    labels = sorted(np.unique(y).tolist())
    with open(model_dir / "labels.json", "w") as f:
        json.dump(labels, f)
        
    print(f"Successfully exported models to {model_dir}")

if __name__ == "__main__":
    main()
