from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .sift import load_cached_descriptors


def encode_histogram(
    descriptors: np.ndarray,
    vocabulary: MiniBatchKMeans,
) -> np.ndarray:
    size = int(vocabulary.n_clusters)
    histogram = np.zeros(size, dtype=np.float32)
    if descriptors.shape[0] == 0:
        return histogram
    words = vocabulary.predict(descriptors)
    np.add.at(histogram, words, 1.0)
    return histogram


def encode_frame(
    frame: pd.DataFrame,
    cache_dir: Path,
    vocabulary: MiniBatchKMeans,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(frame), int(vocabulary.n_clusters)), dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)

    for index, row in enumerate(
        tqdm(frame.itertuples(index=False), total=len(frame), desc="Encode BoVW", leave=False)
    ):
        descriptors = load_cached_descriptors(cache_dir, str(row.image_id))
        histogram = encode_histogram(descriptors, vocabulary)
        if l2_normalize:
            norm = float(np.linalg.norm(histogram))
            if norm > 0:
                histogram /= norm
        features[index] = histogram
    return features, labels


def fit_tfidf(train_features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return IDF weights from training histograms (raw or L2-normalized counts)."""
    document_frequency = np.maximum((train_features > 0).sum(axis=0), 1)
    n_docs = train_features.shape[0]
    idf = np.log((n_docs + 1) / (document_frequency + 1)) + 1.0
    return idf.astype(np.float32)


def apply_tfidf(
    features: np.ndarray,
    idf: np.ndarray,
    l2_normalize: bool = True,
) -> np.ndarray:
    weighted = features * idf[None, :]
    if l2_normalize:
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        weighted = weighted / norms
    return weighted.astype(np.float32)
