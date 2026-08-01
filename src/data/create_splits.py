#!/usr/bin/env python3
"""Create the reproducible iNaturalist-2021 subset used by the group.

The script only reads COCO-style annotation JSON files and writes metadata.
Images do not need to be extracted before this script is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TAXONOMY_FIELDS = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specific_epithet",
)

CSV_FIELDS = (
    "image_path",
    "image_id",
    "original_category_id",
    "class_index",
    "kingdom",
    "phylum",
    "class_name",
    "order",
    "family",
    "genus",
    "species",
    "common_name",
    "split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed 40/10/10 iNaturalist-2021 split."
    )
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9517)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    print(f"Loading {path} ...")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_dataset(
    dataset: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images_by_id = {int(image["id"]): image for image in dataset["images"]}
    images_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        try:
            image = images_by_id[image_id]
        except KeyError as exc:
            raise ValueError(f"Annotation refers to missing image id {image_id}") from exc
        images_by_category[category_id].append(image)

    for images in images_by_category.values():
        images.sort(key=lambda item: (item["file_name"], int(item["id"])))

    return images_by_id, images_by_category


def taxonomy_record(category: dict[str, Any], class_index: int) -> dict[str, Any]:
    return {
        "original_category_id": int(category["id"]),
        "class_index": class_index,
        "kingdom": category.get("kingdom", ""),
        "phylum": category.get("phylum", ""),
        "class_name": category.get("class", ""),
        "order": category.get("order", ""),
        "family": category.get("family", ""),
        "genus": category.get("genus", ""),
        "species": category.get("name", ""),
        "specific_epithet": category.get("specific_epithet", ""),
        "common_name": category.get("common_name", ""),
    }


def image_record(
    image: dict[str, Any],
    category: dict[str, Any],
    class_index: int,
    split: str,
) -> dict[str, Any]:
    return {
        "image_path": image["file_name"],
        "image_id": int(image["id"]),
        "original_category_id": int(category["id"]),
        "class_index": class_index,
        "kingdom": category.get("kingdom", ""),
        "phylum": category.get("phylum", ""),
        "class_name": category.get("class", ""),
        "order": category.get("order", ""),
        "family": category.get("family", ""),
        "genus": category.get("genus", ""),
        "species": category.get("name", ""),
        "common_name": category.get("common_name", ""),
        "split": split,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if min(
        args.num_classes,
        args.train_per_class,
        args.val_per_class,
        args.test_per_class,
    ) <= 0:
        raise ValueError("All requested class and image counts must be positive.")

    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)

    train_categories = {
        int(category["id"]): category for category in train_data["categories"]
    }
    test_categories = {
        int(category["id"]): category for category in test_data["categories"]
    }
    common_category_ids = set(train_categories) & set(test_categories)

    _, train_images_by_category = index_dataset(train_data)
    _, test_images_by_category = index_dataset(test_data)

    required_train_pool = args.train_per_class + args.val_per_class
    eligible_category_ids = sorted(
        category_id
        for category_id in common_category_ids
        if len(train_images_by_category.get(category_id, [])) >= required_train_pool
        and len(test_images_by_category.get(category_id, [])) >= args.test_per_class
    )
    if len(eligible_category_ids) < args.num_classes:
        raise ValueError(
            f"Only {len(eligible_category_ids)} eligible categories; "
            f"cannot sample {args.num_classes}."
        )

    selection_rng = random.Random(args.seed)
    selected_category_ids = sorted(
        selection_rng.sample(eligible_category_ids, args.num_classes)
    )
    class_to_idx = {
        str(category_id): class_index
        for class_index, category_id in enumerate(selected_category_ids)
    }

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    selected_classes: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []

    for category_id in selected_category_ids:
        class_index = class_to_idx[str(category_id)]
        category = train_categories[category_id]

        # A category-specific RNG keeps each category split stable and reproducible.
        category_rng = random.Random(args.seed * 1_000_003 + category_id)
        train_pool = list(train_images_by_category[category_id])
        category_rng.shuffle(train_pool)
        train_images = train_pool[: args.train_per_class]
        val_images = train_pool[
            args.train_per_class : args.train_per_class + args.val_per_class
        ]

        test_pool = list(test_images_by_category[category_id])
        category_rng.shuffle(test_pool)
        test_images = test_pool[: args.test_per_class]

        split_rows["train"].extend(
            image_record(image, category, class_index, "train")
            for image in train_images
        )
        split_rows["val"].extend(
            image_record(image, category, class_index, "val") for image in val_images
        )
        split_rows["test"].extend(
            image_record(image, category, class_index, "test") for image in test_images
        )

        taxon = taxonomy_record(category, class_index)
        taxonomy_rows.append(taxon)
        selected_classes.append(taxon)

        # Detect an unexpected mismatch between train and official validation taxonomy.
        test_category = test_categories[category_id]
        for field in TAXONOMY_FIELDS:
            if category.get(field, "") != test_category.get(field, ""):
                raise ValueError(
                    f"Taxonomy mismatch for category {category_id}, field {field}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (int(row["class_index"]), row["image_path"]))
        write_csv(args.output_dir / f"{split}.csv", rows, CSV_FIELDS)

    write_csv(
        args.output_dir / "taxonomy.csv",
        taxonomy_rows,
        (
            "original_category_id",
            "class_index",
            "kingdom",
            "phylum",
            "class_name",
            "order",
            "family",
            "genus",
            "species",
            "specific_epithet",
            "common_name",
        ),
    )
    write_json(args.output_dir / "selected_classes.json", selected_classes)
    write_json(args.output_dir / "class_to_idx.json", class_to_idx)

    summary = {
        "dataset_version": "0.1-unverified",
        "status": "AWAITING_INDEPENDENT_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "num_classes": args.num_classes,
        "eligible_classes": len(eligible_category_ids),
        "images_per_class": {
            "train": args.train_per_class,
            "val": args.val_per_class,
            "test": args.test_per_class,
        },
        "image_counts": {split: len(rows) for split, rows in split_rows.items()},
        "sources": {
            "train_and_validation": args.train_json.name,
            "held_out_test": args.test_json.name,
        },
        "notes": [
            "Train and val are disjoint samples from the official train_mini split.",
            "Test uses only the official iNaturalist-2021 validation split.",
            "Image paths are relative to the common dataset root.",
        ],
    }
    write_json(args.output_dir / "dataset_summary.json", summary)

    readme = f"""# Dataset metadata v0.1

Status: **AWAITING INDEPENDENT AUDIT**

- Random seed: `{args.seed}`
- Classes: `{args.num_classes}`
- Train images: `{len(split_rows['train'])}` ({args.train_per_class} per class)
- Validation images: `{len(split_rows['val'])}` ({args.val_per_class} per class)
- Test images: `{len(split_rows['test'])}` ({args.test_per_class} per class)
- Train/validation source: official iNaturalist-2021 `train_mini`
- Test source: official iNaturalist-2021 validation split

All image paths in the CSV files are relative to a common dataset root. Member B must
run the independent audit before this metadata is renamed or marked as Dataset v1.0.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Eligible categories: {len(eligible_category_ids)}")
    print(f"Selected categories: {len(selected_category_ids)}")
    for split, rows in split_rows.items():
        print(f"{split}: {len(rows)} images")
    print(f"Metadata written to {args.output_dir}")


if __name__ == "__main__":
    main()
