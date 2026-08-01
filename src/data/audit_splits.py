#!/usr/bin/env python3
"""Audit generated iNaturalist split metadata before Dataset v1.0 is frozen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated dataset metadata.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional common image root. When omitted, file checks are skipped.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_dir
    summary = load_json(metadata_dir / "dataset_summary.json")
    class_to_idx = load_json(metadata_dir / "class_to_idx.json")
    rows_by_split = {
        split: load_csv(metadata_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }

    failures: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    expected_classes = int(summary["num_classes"])
    expected_per_class = {
        split: int(summary["images_per_class"][split])
        for split in rows_by_split
    }

    if len(class_to_idx) != expected_classes:
        failures.append(
            f"class_to_idx contains {len(class_to_idx)} classes, expected {expected_classes}"
        )
    else:
        checks.append(f"class_to_idx contains {expected_classes} classes")

    class_indices = sorted(int(value) for value in class_to_idx.values())
    if class_indices != list(range(expected_classes)):
        failures.append("class indices are not the contiguous range 0..N-1")
    else:
        checks.append("class indices are contiguous")

    all_paths: dict[str, set[str]] = {}
    all_image_ids: dict[str, set[str]] = {}
    category_to_index: dict[str, set[str]] = defaultdict(set)

    for split, rows in rows_by_split.items():
        expected_count = expected_classes * expected_per_class[split]
        if len(rows) != expected_count:
            failures.append(f"{split} has {len(rows)} rows, expected {expected_count}")
        else:
            checks.append(f"{split} contains {expected_count} rows")

        per_class = Counter(row["class_index"] for row in rows)
        bad_counts = {
            class_index: count
            for class_index, count in per_class.items()
            if count != expected_per_class[split]
        }
        if len(per_class) != expected_classes or bad_counts:
            failures.append(
                f"{split} per-class counts invalid; classes={len(per_class)}, "
                f"bad_counts={len(bad_counts)}"
            )
        else:
            checks.append(
                f"{split} has exactly {expected_per_class[split]} images per class"
            )

        expected_prefix = "val/" if split == "test" else "train_mini/"
        wrong_source = [
            row["image_path"]
            for row in rows
            if not row["image_path"].startswith(expected_prefix)
        ]
        if wrong_source:
            failures.append(
                f"{split} contains {len(wrong_source)} paths from the wrong source"
            )
        else:
            checks.append(f"{split} paths use the correct official source split")

        paths = [row["image_path"] for row in rows]
        image_ids = [row["image_id"] for row in rows]
        if len(set(paths)) != len(paths):
            failures.append(f"{split} contains duplicate image paths")
        if len(set(image_ids)) != len(image_ids):
            failures.append(f"{split} contains duplicate image ids")
        all_paths[split] = set(paths)
        all_image_ids[split] = set(image_ids)

        for row in rows:
            category_to_index[row["original_category_id"]].add(row["class_index"])
            expected_index = class_to_idx.get(row["original_category_id"])
            if expected_index is None or int(expected_index) != int(row["class_index"]):
                failures.append(
                    "CSV/class_to_idx mismatch for category "
                    f"{row['original_category_id']}"
                )
                break

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        path_overlap = all_paths[left] & all_paths[right]
        id_overlap = all_image_ids[left] & all_image_ids[right]
        if path_overlap:
            failures.append(f"{left}/{right} share {len(path_overlap)} image paths")
        if id_overlap:
            failures.append(f"{left}/{right} share {len(id_overlap)} image ids")
        if not path_overlap and not id_overlap:
            checks.append(f"{left}/{right} are disjoint by path and image id")

    inconsistent_categories = {
        category_id: indices
        for category_id, indices in category_to_index.items()
        if len(indices) != 1
    }
    if inconsistent_categories:
        failures.append(
            f"{len(inconsistent_categories)} categories map to multiple class indices"
        )
    else:
        checks.append("every original category maps to one class index")

    unreadable_or_missing = 0
    if args.data_root is None:
        warnings.append("image existence/readability checks skipped: --data-root not supplied")
    else:
        try:
            from PIL import Image
        except ImportError:
            Image = None
            warnings.append("Pillow unavailable; checking file existence only")

        for rows in rows_by_split.values():
            for row in rows:
                image_path = args.data_root / row["image_path"]
                if not image_path.is_file():
                    unreadable_or_missing += 1
                    continue
                if Image is not None:
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception:
                        unreadable_or_missing += 1
        if unreadable_or_missing:
            failures.append(
                f"{unreadable_or_missing} images are missing or unreadable under data root"
            )
        else:
            checks.append("all listed image files exist and are readable")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks_passed": checks,
        "warnings": warnings,
        "failures": failures,
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "unreadable_or_missing_images": unreadable_or_missing,
    }
    report_path = metadata_dir / "audit_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Dataset audit: {report['status']}")
    for check in checks:
        print(f"  PASS: {check}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"Report written to {report_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

