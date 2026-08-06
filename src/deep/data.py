from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common import seed_worker


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        transform: transforms.Compose,
        num_classes: int,
        max_samples: int | None = None,
        sample_seed: int = 9517,
    ) -> None:
        frame = pd.read_csv(csv_path, dtype={"image_id": str})
        required = {"image_path", "image_id", "class_index", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        expected_split = csv_path.stem
        actual_splits = set(frame["split"].astype(str))
        if actual_splits != {expected_split}:
            raise ValueError(
                f"{csv_path} contains split values {sorted(actual_splits)}, "
                f"expected only {expected_split!r}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            frame = frame.sample(
                n=min(max_samples, len(frame)), random_state=sample_seed
            ).reset_index(drop=True)

        labels = frame["class_index"].astype(int)
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Labels in {csv_path} fall outside 0..{num_classes - 1}")

        self.frame = frame
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative_path = str(row["image_path"])
        full_path = self.data_root / Path(relative_path)
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "target": int(row["class_index"]),
            "image_path": relative_path,
            "image_id": str(row["image_id"]),
        }


def build_transform(
    mode: str,
    image_size: int,
    resize_size: int,
    use_color_jitter: bool,
) -> transforms.Compose:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if mode == "standard":
        operations: list[Any] = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        if use_color_jitter:
            operations.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            )
        operations.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(operations)
    if mode in {"noaug", "eval"}:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    raise ValueError(f"Unknown transform mode: {mode}")


def validate_class_mapping(path: Path, num_classes: int) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    mapping = {str(key): int(value) for key, value in mapping.items()}
    if len(mapping) != num_classes:
        raise ValueError(f"Expected {num_classes} classes, found {len(mapping)}")
    if sorted(mapping.values()) != list(range(num_classes)):
        raise ValueError("class_to_idx values are not contiguous from 0")
    return mapping


def make_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
