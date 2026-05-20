from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


LABEL_TO_INDEX = {"background": 0, "target": 1}
INDEX_TO_LABEL = {0: "background", 1: "target"}


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["eeg_file"].str.extract(r"sub(\d+)_sess(\d+)_epoch(\d+)")
    parsed.columns = ["sub", "sess", "epoch"]
    parsed = parsed.astype(int)
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


def load_raw_epochs(data_dir: str | Path, eeg_files: Iterable[str]) -> np.ndarray:
    data_dir = Path(data_dir)
    return np.stack([np.load(data_dir / eeg_file).astype("float32")[0] for eeg_file in eeg_files])


def build_metadata_features(df: pd.DataFrame) -> np.ndarray:
    rows: List[pd.DataFrame] = []
    for (subject, session), group0 in df.groupby(["sub", "sess"], sort=False):
        group = group0.sort_values("epoch")
        epochs = group["epoch"].to_numpy()
        n = len(epochs)
        prev_gap = np.r_[99, np.diff(epochs)]
        next_gap = np.r_[np.diff(epochs), 99]
        prev2_gap = np.r_[99, 99, epochs[2:] - epochs[:-2]] if n > 2 else np.full(n, 99)
        next2_gap = np.r_[epochs[2:] - epochs[:-2], 99, 99] if n > 2 else np.full(n, 99)
        rank = np.arange(n) / max(1, n - 1)

        local_density = []
        for radius in [3, 5, 8, 12, 20]:
            local_density.append(np.array([(np.abs(epochs - value) <= radius).sum() - 1 for value in epochs]))

        harmonics = []
        for period in [1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30]:
            harmonics.extend(
                [
                    np.sin(2 * np.pi * epochs / period),
                    np.cos(2 * np.pi * epochs / period),
                ]
            )

        features = np.c_[
            np.full(n, subject),
            np.full(n, session),
            epochs,
            epochs / 300.0,
            rank,
            np.full(n, n),
            epochs % 2,
            epochs % 3,
            epochs % 4,
            epochs % 5,
            epochs % 6,
            epochs % 7,
            epochs % 8,
            epochs % 9,
            epochs % 10,
            epochs % 12,
            epochs % 15,
            epochs % 20,
            prev_gap,
            next_gap,
            prev2_gap,
            next2_gap,
            np.asarray(local_density).T,
            np.asarray(harmonics).T,
        ]
        rows.append(pd.DataFrame(features, index=group.index))
    return pd.concat(rows).sort_index().to_numpy(dtype=np.float32)


def build_signal_features(epochs: np.ndarray) -> np.ndarray:
    features: List[np.ndarray] = [
        epochs.mean(axis=(1, 2))[:, None],
        epochs.std(axis=(1, 2))[:, None],
        np.abs(epochs).mean(axis=(1, 2))[:, None],
        epochs.max(axis=(1, 2))[:, None],
        epochs.min(axis=(1, 2))[:, None],
    ]

    windows = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 250),
        (250, 282),
        (60, 160),
        (120, 240),
    ]
    for start, end in windows:
        window = epochs[:, :, start:end]
        features.extend(
            [
                window.mean(axis=(1, 2))[:, None],
                window.std(axis=(1, 2))[:, None],
                np.abs(window).mean(axis=(1, 2))[:, None],
                window.mean(axis=2),
                window.std(axis=2),
            ]
        )

    return np.concatenate([item.reshape(len(epochs), -1) for item in features], axis=1).astype(np.float32)


def build_tabular_features(df: pd.DataFrame, data_dir: str | Path) -> np.ndarray:
    df = add_metadata(df) if "sub" not in df.columns else df
    epochs = load_raw_epochs(data_dir, df["eeg_file"])
    return np.concatenate([build_metadata_features(df), build_signal_features(epochs)], axis=1)


def fixed_background_predictions(
    df: pd.DataFrame,
    target_prob: np.ndarray,
    background_count: int = 90,
) -> np.ndarray:
    df = add_metadata(df) if "sub" not in df.columns else df.copy()
    df["target_prob"] = target_prob
    predictions = np.ones(len(df), dtype=np.int64)
    for _, group in df.groupby(["sub", "sess"], sort=False):
        sorted_index = group.sort_values("target_prob").index.to_numpy()
        local_positions = df.index.get_indexer(sorted_index[: min(background_count, len(sorted_index))])
        predictions[local_positions] = 0
    return predictions
