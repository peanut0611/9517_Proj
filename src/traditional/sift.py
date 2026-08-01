from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_sift(n_features: int = 0, contrast_threshold: float = 0.04) -> cv2.SIFT:
    # n_features=0 keeps OpenCV default (unlimited)
    return cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
    )


def extract_sift_descriptors(
    image_path: Path,
    sift: cv2.SIFT,
    max_side: int | None = 512,
) -> np.ndarray:
    """Return float32 descriptors with shape (N, 128), or empty (0, 128)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if max_side is not None:
        height, width = image.shape[:2]
        long_side = max(height, width)
        if long_side > max_side:
            scale = max_side / float(long_side)
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

    _keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    return descriptors.astype(np.float32, copy=False)


def cache_path_for_image(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / f"{image_id}.npy"


def extract_and_cache_split(
    frame: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    sift: cv2.SIFT,
    max_side: int | None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Extract SIFT for each row; cache by image_id. Returns simple stats."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_descriptors = 0
    empty_images = 0

    for row in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc=f"SIFT [{cache_dir.name}]",
        leave=False,
    ):
        out_path = cache_path_for_image(cache_dir, str(row.image_id))
        if out_path.exists() and not overwrite:
            descriptors = np.load(out_path)
        else:
            full_path = data_root / Path(str(row.image_path))
            descriptors = extract_sift_descriptors(full_path, sift, max_side=max_side)
            np.save(out_path, descriptors)

        n = int(descriptors.shape[0])
        total_descriptors += n
        if n == 0:
            empty_images += 1

    return {
        "images": int(len(frame)),
        "total_descriptors": int(total_descriptors),
        "empty_images": int(empty_images),
        "mean_descriptors_per_image": float(total_descriptors / max(len(frame), 1)),
    }


def load_cached_descriptors(cache_dir: Path, image_id: str) -> np.ndarray:
    path = cache_path_for_image(cache_dir, image_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing SIFT cache: {path}")
    return np.load(path)
