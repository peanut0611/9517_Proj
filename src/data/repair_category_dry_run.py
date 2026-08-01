#!/usr/bin/env python3
"""Dry-run replacement of category 9918 with category 705.

This script only reads files and simulates the replacement in memory.
It does not modify CSV files, images, or metadata.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
METADATA = PROJECT / "data_metadata"
TRAIN_JSON = Path.home() / "Datasets/iNaturalist2021/train_mini.json"
TEST_JSON = Path.home() / "Datasets/iNaturalist2021/val.json"

OLD_CATEGORY_ID = 9918
NEW_CATEGORY_ID = 705
TARGET_CLASS_INDEX = 495
SEED = 9517

EXPECTED_PER_SPLIT = {
    "train": 40,
    "val": 10,
    "test": 10,
}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_images_by_category(dataset):
    images_by_id = {
        int(image["id"]): image
        for image in dataset["images"]
    }
    result = defaultdict(list)

    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        result[category_id].append(images_by_id[image_id])

    for images in result.values():
        images.sort(key=lambda image: (image["file_name"], int(image["id"])))

    return result


def main():
    train_data = load_json(TRAIN_JSON)
    test_data = load_json(TEST_JSON)

    train_categories = {
        int(category["id"]): category
        for category in train_data["categories"]
    }
    test_categories = {
        int(category["id"]): category
        for category in test_data["categories"]
    }

    split_rows = {
        split: load_csv(METADATA / f"{split}.csv")
        for split in EXPECTED_PER_SPLIT
    }

    old_rows = {
        split: [
            row for row in rows
            if int(row["original_category_id"]) == OLD_CATEGORY_ID
        ]
        for split, rows in split_rows.items()
    }

    new_existing_rows = [
        row
        for rows in split_rows.values()
        for row in rows
        if int(row["original_category_id"]) == NEW_CATEGORY_ID
    ]

    assert not new_existing_rows, (
        f"Category {NEW_CATEGORY_ID} already exists in current splits"
    )

    for split, expected in EXPECTED_PER_SPLIT.items():
        assert len(old_rows[split]) == expected, (
            f"Category {OLD_CATEGORY_ID}: expected {expected} {split} rows, "
            f"found {len(old_rows[split])}"
        )
        assert {
            int(row["class_index"]) for row in old_rows[split]
        } == {TARGET_CLASS_INDEX}, (
            f"Category {OLD_CATEGORY_ID} does not consistently use "
            f"class_index {TARGET_CLASS_INDEX}"
        )

    train_images = index_images_by_category(train_data)
    test_images = index_images_by_category(test_data)

    assert len(train_images[NEW_CATEGORY_ID]) >= 50
    assert len(test_images[NEW_CATEGORY_ID]) >= 10

    # Reproduce the category-specific selection logic in create_splits.py.
    rng = random.Random(SEED * 1_000_003 + NEW_CATEGORY_ID)

    replacement_train_pool = list(train_images[NEW_CATEGORY_ID])
    rng.shuffle(replacement_train_pool)

    replacement_images = {
        "train": replacement_train_pool[:40],
        "val": replacement_train_pool[40:50],
    }

    replacement_test_pool = list(test_images[NEW_CATEGORY_ID])
    rng.shuffle(replacement_test_pool)
    replacement_images["test"] = replacement_test_pool[:10]

    # Simulate the replacement in memory.
    simulated_rows = {}

    for split, rows in split_rows.items():
        kept_rows = [
            row for row in rows
            if int(row["original_category_id"]) != OLD_CATEGORY_ID
        ]

        simulated_replacements = [
            {
                "image_path": image["file_name"],
                "image_id": str(image["id"]),
                "original_category_id": str(NEW_CATEGORY_ID),
                "class_index": str(TARGET_CLASS_INDEX),
                "split": split,
            }
            for image in replacement_images[split]
        ]

        simulated_rows[split] = kept_rows + simulated_replacements

        assert len(simulated_rows[split]) == len(rows), (
            f"{split} total row count would change"
        )

    # Verify every class index would retain the required split count.
    for split, rows in simulated_rows.items():
        counts = Counter(int(row["class_index"]) for row in rows)
        expected = EXPECTED_PER_SPLIT[split]

        assert len(counts) == 500, (
            f"{split} would contain {len(counts)} classes instead of 500"
        )
        assert set(counts) == set(range(500)), (
            f"{split} class indices would no longer be exactly 0–499"
        )
        assert all(count == expected for count in counts.values()), (
            f"{split} would not contain exactly {expected} rows per class"
        )

    # Confirm all other class-index/category mappings remain unchanged.
    original_mapping = {}
    simulated_mapping = {}

    for rows in split_rows.values():
        for row in rows:
            original_mapping[int(row["class_index"])] = int(
                row["original_category_id"]
            )

    for rows in simulated_rows.values():
        for row in rows:
            simulated_mapping[int(row["class_index"])] = int(
                row["original_category_id"]
            )

    for class_index in range(500):
        if class_index == TARGET_CLASS_INDEX:
            continue
        assert simulated_mapping[class_index] == original_mapping[class_index], (
            f"class_index {class_index} would change unexpectedly"
        )

    assert simulated_mapping[TARGET_CLASS_INDEX] == NEW_CATEGORY_ID

    category = train_categories[NEW_CATEGORY_ID]

    print("DRY-RUN PASS")
    print(f"Would remove category: {OLD_CATEGORY_ID}")
    print(
        f"Would add category: {NEW_CATEGORY_ID} "
        f"({category.get('name', '')})"
    )
    print(f"Preserved class_index: {TARGET_CLASS_INDEX}")
    print("Other 499 class mappings: unchanged")
    print("Simulated counts: train=40, val=10, test=10")
    print("No files were modified.")


if __name__ == "__main__":
    main()
