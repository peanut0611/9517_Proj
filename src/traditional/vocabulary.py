from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .sift import load_cached_descriptors


def sample_training_descriptors(
    train_frame: pd.DataFrame,
    cache_dir: Path,
    max_descriptors: int,
    seed: int,
) -> np.ndarray:
    """Reservoir-sample descriptors from the training split only."""
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    collected = 0

    # Shuffle image order for unbiased sampling across classes.
    order = rng.permutation(len(train_frame))
    for index in tqdm(order, desc="Sample descriptors", leave=False):
        if collected >= max_descriptors:
            break
        image_id = str(train_frame.iloc[int(index)]["image_id"])
        descriptors = load_cached_descriptors(cache_dir, image_id)
        if descriptors.shape[0] == 0:
            continue
        remaining = max_descriptors - collected
        if descriptors.shape[0] <= remaining:
            take = descriptors
        else:
            choice = rng.choice(descriptors.shape[0], size=remaining, replace=False)
            take = descriptors[choice]
        sampled.append(take)
        collected += int(take.shape[0])

    if not sampled:
        raise RuntimeError("No SIFT descriptors found in the training split")
    return np.concatenate(sampled, axis=0)


def build_vocabulary(
    descriptors: np.ndarray,
    vocabulary_size: int,
    seed: int,
    batch_size: int = 4096,
    max_iter: int = 100,
) -> MiniBatchKMeans:
    if descriptors.shape[0] < vocabulary_size:
        raise ValueError(
            f"Need at least {vocabulary_size} descriptors to build vocabulary, "
            f"got {descriptors.shape[0]}"
        )
    model = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        random_state=seed,
        batch_size=min(batch_size, descriptors.shape[0]),
        max_iter=max_iter,
        n_init="auto",
        reassignment_ratio=0.01,
        verbose=0,
    )
    model.fit(descriptors)
    return model


def save_vocabulary(model: MiniBatchKMeans, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_vocabulary(path: Path) -> MiniBatchKMeans:
    return joblib.load(path)
