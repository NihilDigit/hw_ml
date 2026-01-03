"""Build a stratified 10k subset from multiple IDS2018 processed CSVs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES, LABEL_COL, RANDOM_SEED


RAW_DIR = Path("data/raw/processed_csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep Benign + 14 attack types
LABEL_WHITELIST = {
    "Benign",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infilteration",
    "SQL Injection",
    "SSH-Bruteforce",
}


def iter_clean_chunks(path: Path, chunksize: int = 200_000):
    usecols = FEATURES + [LABEL_COL]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        # Drop duplicated header rows embedded in data
        chunk = chunk[chunk[LABEL_COL] != LABEL_COL]

        # Coerce numeric features and clean inf/NaN
        chunk[FEATURES] = chunk[FEATURES].apply(pd.to_numeric, errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=FEATURES + [LABEL_COL])

        # Drop zero-duration + zero-rate anomalies
        zero_mask = (
            (chunk["Flow Byts/s"] == 0)
            & (chunk["Flow Pkts/s"] == 0)
            & (chunk["Flow Duration"] == 0)
        )
        if zero_mask.any():
            chunk = chunk[~zero_mask]

        if len(chunk) > 0:
            # Keep only whitelist labels
            chunk = chunk[chunk[LABEL_COL].isin(LABEL_WHITELIST)]
            if len(chunk) > 0:
                yield chunk


def compute_label_counts(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path in files:
        for chunk in iter_clean_chunks(path):
            labels = chunk[LABEL_COL].astype(str)
            for label, cnt in labels.value_counts().items():
                counts[label] += int(cnt)
    return dict(counts)


def allocate_targets(
    counts: dict[str, int], total_samples: int, min_per_class: int
) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No samples found after cleaning.")

    labels = list(counts.keys())
    base = min_per_class * len(labels)
    if base > total_samples:
        raise ValueError(
            f"min_per_class={min_per_class} too large for total_samples={total_samples}"
        )

    remaining = total_samples - base
    # Proportional allocation with rounding
    raw = {k: (v / total) * remaining for k, v in counts.items()}
    targets = {k: min_per_class + int(np.floor(v)) for k, v in raw.items()}

    # Adjust to match total_samples
    current = sum(targets.values())
    if current < total_samples:
        # Distribute remaining based on largest fractional part
        frac = sorted(
            ((k, raw[k] - np.floor(raw[k])) for k in counts.keys()),
            key=lambda x: x[1],
            reverse=True,
        )
        i = 0
        while current < total_samples:
            k, _ = frac[i % len(frac)]
            if targets[k] < counts[k]:
                targets[k] += 1
                current += 1
            i += 1
    elif current > total_samples:
        # Remove extras from largest targets
        over = current - total_samples
        by_size = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        i = 0
        while over > 0:
            k, _ = by_size[i % len(by_size)]
            if targets[k] > 1:
                targets[k] -= 1
                over -= 1
            i += 1

    return targets


def reservoir_sample(
    files: list[Path], targets: dict[str, int], counts: dict[str, int], seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reservoirs = {k: [] for k in targets}
    seen = defaultdict(int)

    for path in files:
        for chunk in iter_clean_chunks(path):
            labels = chunk[LABEL_COL].astype(str).to_numpy()
            feats = chunk[FEATURES].to_numpy()
            for i, label in enumerate(labels):
                if label not in targets or targets[label] == 0:
                    continue
                seen[label] += 1
                k = targets[label]
                res = reservoirs[label]
                if len(res) < k and len(res) < counts[label]:
                    res.append(feats[i])
                else:
                    j = rng.integers(0, seen[label])
                    if j < k and j < counts[label]:
                        res[j] = feats[i]

    # Oversample with replacement if target > available samples
    for label, samples in reservoirs.items():
        target = targets[label]
        if len(samples) < target and len(samples) > 0:
            need = target - len(samples)
            extra_idx = rng.integers(0, len(samples), size=need)
            samples.extend([samples[i] for i in extra_idx])

    rows = []
    for label, samples in reservoirs.items():
        for sample in samples:
            rows.append((*sample.tolist(), label))

    df = pd.DataFrame(rows, columns=FEATURES + [LABEL_COL])
    return df


def main(total_samples: int = 10_000, min_per_class: int = 100) -> None:
    files = sorted(p for p in RAW_DIR.glob("*.csv") if not p.name.endswith(".part"))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")

    print(f"Using {len(files)} files from {RAW_DIR}")
    counts = compute_label_counts(files)
    targets = allocate_targets(counts, total_samples, min_per_class)

    print("Label counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("Target samples:")
    for k, v in sorted(targets.items()):
        print(f"  {k}: {v}")

    subset = reservoir_sample(files, targets, counts, RANDOM_SEED)
    subset = subset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    suffix = f"{total_samples//1000}k"
    out_csv = OUT_DIR / f"ids2018_subset_{suffix}.csv"
    subset.to_csv(out_csv, index=False)

    stats = subset[LABEL_COL].value_counts().rename_axis("Label").reset_index(name="Count")
    stats.to_csv(OUT_DIR / f"ids2018_subset_{suffix}_stats.csv", index=False)

    print(f"Saved subset to {out_csv}")
    print(f"Saved stats to {OUT_DIR / f'ids2018_subset_{suffix}_stats.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build stratified IDS2018 subset.")
    parser.add_argument("--total", type=int, default=10000, help="Total samples")
    parser.add_argument("--min-per-class", type=int, default=100, help="Min samples per class")
    args = parser.parse_args()
    main(total_samples=args.total, min_per_class=args.min_per_class)
