from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import LABEL_COL, RANDOM_SEED

T = TypeVar("T")


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class TimedResult:
    value: Any
    seconds: float


def timed(call: Callable[..., T], *args: Any, **kwargs: Any) -> TimedResult:
    start = time.perf_counter()
    value = call(*args, **kwargs)
    end = time.perf_counter()
    return TimedResult(value=value, seconds=end - start)


def select_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def fit_minmax_scaler(X_train: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def save_minmax_scaler_params(scaler: MinMaxScaler, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_range": list(getattr(scaler, "feature_range", (0, 1))),
        "data_min_": getattr(scaler, "data_min_", []).tolist(),
        "data_max_": getattr(scaler, "data_max_", []).tolist(),
        "data_range_": getattr(scaler, "data_range_", []).tolist(),
        "min_": getattr(scaler, "min_", []).tolist(),
        "scale_": getattr(scaler, "scale_", []).tolist(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latency_ms_per_sample(total_seconds: float, n_samples: int) -> float:
    if n_samples <= 0:
        return 0.0
    return (total_seconds / n_samples) * 1000.0


def build_2d_viz_frame(X_2d: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    if X_2d.shape[1] < 2:
        raise ValueError("X_2d must have at least 2 columns.")
    df = pd.DataFrame(X_2d[:, :2], columns=["c1", "c2"])
    df[LABEL_COL] = y
    return df

