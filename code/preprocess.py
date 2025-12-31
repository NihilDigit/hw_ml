from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_COL, FEATURES


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    # Drop duplicated header rows that leaked into data
    df = df[df[LABEL_COL] != LABEL_COL]

    # Keep only required features + label first
    keep_cols = [c for c in FEATURES if c in df.columns] + [LABEL_COL]
    df = df[keep_cols].copy()

    # Convert all feature columns to numeric, coercing errors to NaN
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace inf/-inf with NaN, then drop rows with any NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop invalid zero-duration zero-rate rows
    if "Flow Byts/s" in df.columns and "Flow Pkts/s" in df.columns and "Flow Duration" in df.columns:
        zero_mask = (df["Flow Byts/s"] == 0) & (df["Flow Pkts/s"] == 0) & (df["Flow Duration"] == 0)
        df = df[~zero_mask]

    return df
