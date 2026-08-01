#!/usr/bin/env python3
"""Prepare repaired metadata in a separate staging directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OLD_CATEGORY_ID = 9918
NEW_CATEGORY_ID = 705
TARGET_CLASS_INDEX = 495
SEED = 9517

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

TAXONOMY_FIELDS = (
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
)

EXPECTED = {"train": 40, "val": 10, "test": 10}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def index_images(dataset):
    images_by_id = {
        int(image["id"]): image for image in dataset["images"]
    }
    images_by_category = defaultdict(list)

    for annotation in dataset["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        images_by_category[category_id].append(images_by_id[image_id])

    for images in images_by_category.values():
        images.sort(key=lambda image: (image["file_name"], int(image["id"])))

    return images_by_category


def taxonomy_record(category):
    return {
        "original_category_id": NEW_CATEGORY_ID,
        "class_index": TARGET_CLASS_INDEX,
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


def image_record(image, category, split):
    return {
        "image_path": image["file_name"],
        "image_id": int(image["id"]),
        "original_category_id": NEW_CATEGORY_ID,
        "class_index": TARGET_CLASS_INDEX,
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


def validate_splits(rows_by_split):
    all_paths = {}
    all_ids = {}

    for split, rows in rows_by_split.items():
        expected_total = 500 * EXPECTED[split]
        assert len(rows) == expected_total

        counts = Counter(int(row["class_index"]) for row in rows)
        assert set(counts) == set(range(500))
        assert all(count == EXPECTED[split] for count in counts.values())

        assert not any(
            int(row["original_category_id"]) == OLD_CATEGORY_ID
            for row in rows
        )

        replacement_rows = [
            row for row in rows
            if int(row["original_category_id"]) == NEW_CATEGORY_ID
        ]
        assert len(replacement_rows) == EXPECTED[split]
        assert {
            int(row["class_index"]) for row in replacement_rows
        } == {TARGET_CLASS_INDEX}

        paths = [row["image_path"] for row in rows]
        image_ids = [str(row["image_id"]) for row in rows]
        assert len(paths) == len(set(paths))
        assert len(image_ids) == len(set(image_ids))

        all_paths[split] = set(paths)
        all_ids[split] = set(image_ids)

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        assert not (all_paths[left] & all_paths[right])
        assert not (all_ids[left] & all_ids[right])


def main():
    args = parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing staging directory: "
            f"{args.output_dir}"
        )

    building_dir = args.output_dir.with_name(args.output_dir.name + ".building")
    if building_dir.exists():
        raise FileExistsError(
            f"Previous incomplete build exists: {building_dir}"
        )

    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)

    train_categories = {
        int(category["id"]): category
        for category in train_data["categories"]
    }
    test_categories = {
        int(category["id"]): category
        for category in test_data["categories"]
    }

    assert NEW_CATEGORY_ID in train_categories
    assert NEW_CATEGORY_ID in test_categories

    category = train_categories[NEW_CATEGORY_ID]
    train_images = index_images(train_data)
    test_images = index_images(test_data)

    assert len(train_images[NEW_CATEGORY_ID]) >= 50
    assert len(test_images[NEW_CATEGORY_ID]) >= 10

    original_splits = {
        split: load_csv(args.metadata_dir / f"{split}.csv")
        for split in EXPECTED
    }

    for split, expected_count in EXPECTED.items():
        old_rows = [
            row for row in original_splits[split]
            if int(row["original_category_id"]) == OLD_CATEGORY_ID
        ]
        assert len(old_rows) == expected_count
        assert {
            int(row["class_index"]) for row in old_rows
        } == {TARGET_CLASS_INDEX}

        assert not any(
            int(row["original_category_id"]) == NEW_CATEGORY_ID
            for row in original_splits[split]
        )

    rng = random.Random(SEED * 1_000_003 + NEW_CATEGORY_ID)

    shuffled_train = list(train_images[NEW_CATEGORY_ID])
    rng.shuffle(shuffled_train)

    selected_images = {
        "train": shuffled_train[:40],
        "val": shuffled_train[40:50],
    }

    shuffled_test = list(test_images[NEW_CATEGORY_ID])
    rng.shuffle(shuffled_test)
    selected_images["test"] = shuffled_test[:10]

    repaired_splits = {}

    for split, original_rows in original_splits.items():
        kept_rows = [
            row for row in original_rows
            if int(row["original_category_id"]) != OLD_CATEGORY_ID
        ]

        new_rows = [
            image_record(image, category, split)
            for image in selected_images[split]
        ]

        repaired_rows = kept_rows + new_rows
        repaired_rows.sort(
            key=lambda row: (
                int(row["class_index"]),
                row["image_path"],
            )
        )
        repaired_splits[split] = repaired_rows

    validate_splits(repaired_splits)

    original_mapping = load_json(args.metadata_dir / "class_to_idx.json")
    assert str(OLD_CATEGORY_ID) in original_mapping
    assert str(NEW_CATEGORY_ID) not in original_mapping
    assert int(original_mapping[str(OLD_CATEGORY_ID)]) == TARGET_CLASS_INDEX

    repaired_mapping = {}
    for category_id, class_index in original_mapping.items():
        if category_id == str(OLD_CATEGORY_ID):
            repaired_mapping[str(NEW_CATEGORY_ID)] = TARGET_CLASS_INDEX
        else:
            repaired_mapping[category_id] = class_index

    assert len(repaired_mapping) == 500
    assert sorted(int(value) for value in repaired_mapping.values()) == list(
        range(500)
    )

    new_taxonomy = taxonomy_record(category)

    taxonomy_rows = load_csv(args.metadata_dir / "taxonomy.csv")
    taxonomy_matches = [
        index for index, row in enumerate(taxonomy_rows)
        if int(row["original_category_id"]) == OLD_CATEGORY_ID
    ]
    assert len(taxonomy_matches) == 1
    taxonomy_rows[taxonomy_matches[0]] = new_taxonomy

    selected_classes = load_json(
        args.metadata_dir / "selected_classes.json"
    )
    selected_matches = [
        index for index, row in enumerate(selected_classes)
        if int(row["original_category_id"]) == OLD_CATEGORY_ID
    ]
    assert len(selected_matches) == 1
    selected_classes[selected_matches[0]] = new_taxonomy

    assert len(taxonomy_rows) == 500
    assert len(selected_classes) == 500

    summary = load_json(args.metadata_dir / "dataset_summary.json")
    summary["dataset_version"] = "0.2-unverified"
    summary["status"] = "REPAIR_STAGED_AWAITING_AUDIT"
    summary["repaired_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["repair"] = {
        "reason": "Cross-split duplicate image content detected by SHA256",
        "removed_category_id": OLD_CATEGORY_ID,
        "replacement_category_id": NEW_CATEGORY_ID,
        "replacement_species": category.get("name", ""),
        "preserved_class_index": TARGET_CLASS_INDEX,
        "seed": SEED,
        "other_class_mappings_changed": False,
    }

    readme = f"""# Dataset metadata v0.2-unverified

Status: **REPAIR STAGED — AWAITING AUDIT**

- Random seed: `{SEED}`
- Classes: `500`
- Train images: `20000` (40 per class)
- Validation images: `5000` (10 per class)
- Test images: `5000` (10 per class)
- Removed category: `{OLD_CATEGORY_ID}`
- Replacement category: `{NEW_CATEGORY_ID}` ({category.get("name", "")})
- Preserved class index: `{TARGET_CLASS_INDEX}`
- Other 499 class mappings: unchanged
- Train/validation source: official iNaturalist-2021 `train_mini`
- Test source: official iNaturalist-2021 validation split

This staging version must pass metadata, readability, and SHA256 audits
before it replaces the active dataset metadata or is frozen as Dataset v1.0.
"""

    audit_placeholder = {
        "status": "NOT_RUN_AFTER_REPAIR",
        "dataset_version": "0.2-unverified",
        "message": "Run full metadata, readability, and SHA256 audits.",
    }

    building_dir.mkdir(parents=False)

    for split, rows in repaired_splits.items():
        write_csv(building_dir / f"{split}.csv", rows, CSV_FIELDS)

    write_csv(
        building_dir / "taxonomy.csv",
        taxonomy_rows,
        TAXONOMY_FIELDS,
    )
    write_json(
        building_dir / "selected_classes.json",
        selected_classes,
    )
    write_json(
        building_dir / "class_to_idx.json",
        repaired_mapping,
    )
    write_json(
        building_dir / "dataset_summary.json",
        summary,
    )
    write_json(
        building_dir / "audit_report.json",
        audit_placeholder,
    )

    with (building_dir / "README.md").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(readme)

    building_dir.rename(args.output_dir)

    print("STAGING METADATA CREATED")
    print(f"Output: {args.output_dir}")
    print(f"Removed category: {OLD_CATEGORY_ID}")
    print(
        f"Added category: {NEW_CATEGORY_ID} "
        f"({category.get('name', '')})"
    )
    print(f"Preserved class_index: {TARGET_CLASS_INDEX}")
    print("Other 499 class mappings: unchanged")
    print("Counts: train=20000, val=5000, test=5000")
    print("Active metadata and image files were not modified.")


if __name__ == "__main__":
    main()
