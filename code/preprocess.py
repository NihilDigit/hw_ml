from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_COL, FEATURES


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    # Drop duplicated header rows that leaked into data
    df = df[df[LABEL_COL] != LABEL_COL]

    # Replace inf/-inf with NaN, then drop rows with any NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop invalid zero-duration zero-rate rows
    zero_mask = (df["Flow Byts/s"] == 0) & (df["Flow Pkts/s"] == 0) & (df["Flow Duration"] == 0)
    df = df[~zero_mask]

    # Keep only required features + label
    keep_cols = [c for c in FEATURES if c in df.columns] + [LABEL_COL]
    df = df[keep_cols].copy()
    return df
