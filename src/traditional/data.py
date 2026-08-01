from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "image_path",
    "image_id",
    "class_index",
    "split",
}


def load_split_csv(
    csv_path: Path,
    expected_split: str | None = None,
    class_indices: set[int] | None = None,
    max_samples: int | None = None,
    sample_seed: int = 9517,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, dtype={"image_id": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    if expected_split is None:
        expected_split = csv_path.stem
    actual_splits = set(frame["split"].astype(str))
    if actual_splits != {expected_split}:
        raise ValueError(
            f"{csv_path} contains split values {sorted(actual_splits)}, "
            f"expected only {expected_split!r}"
        )

    frame = frame.copy()
    frame["class_index"] = frame["class_index"].astype(int)

    if class_indices is not None:
        frame = frame[frame["class_index"].isin(class_indices)].reset_index(drop=True)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        frame = (
            frame.sample(n=min(max_samples, len(frame)), random_state=sample_seed)
            .reset_index(drop=True)
        )

    return frame


def select_class_subset(
    metadata_dir: Path,
    num_classes: int | None,
    seed: int,
) -> set[int] | None:
    """Return a stable subset of class indices, or None for all 500 classes."""
    if num_classes is None:
        return None

    mapping_path = metadata_dir / "class_to_idx.json"
    import json

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    all_indices = sorted(int(v) for v in mapping.values())
    if num_classes > len(all_indices):
        raise ValueError(
            f"Requested {num_classes} classes but only {len(all_indices)} exist"
        )
    rng = np_random(seed)
    chosen = rng.choice(all_indices, size=num_classes, replace=False)
    return set(int(x) for x in chosen)


def np_random(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def remap_labels(frame: pd.DataFrame, class_indices: set[int] | None) -> pd.DataFrame:
    """Optionally remap sparse class_index values to contiguous 0..C-1 for pilot runs."""
    out = frame.copy()
    if class_indices is None:
        out["label"] = out["class_index"].astype(int)
        return out
    ordered = sorted(class_indices)
    remap = {old: new for new, old in enumerate(ordered)}
    out["label"] = out["class_index"].map(remap).astype(int)
    return out
