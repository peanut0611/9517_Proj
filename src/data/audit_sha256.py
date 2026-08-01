#!/usr/bin/env python3
"""Detect identical image content within and across dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit image-content duplication using SHA256."
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def load_rows(csv_path, split):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["_split"] = split

    return rows


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    args = parse_args()

    rows = []
    for split in ("train", "val", "test"):
        rows.extend(
            load_rows(args.metadata_dir / f"{split}.csv", split)
        )

    print(f"Hashing {len(rows)} images...")

    by_hash = defaultdict(list)
    missing = []

    for number, row in enumerate(rows, start=1):
        image_path = args.data_root / row["image_path"]

        if not image_path.is_file():
            missing.append(str(image_path))
            continue

        digest = sha256_file(image_path)
        by_hash[digest].append(row)

        if number % 1000 == 0:
            print(f"  processed {number}/{len(rows)}", flush=True)

    if missing:
        print(f"FAIL: {len(missing)} images are missing")
        for path in missing[:10]:
            print(f"  missing: {path}")
        raise SystemExit(1)

    duplicate_groups = {
        digest: group
        for digest, group in by_hash.items()
        if len(group) > 1
    }

    cross_split_groups = {
        digest: group
        for digest, group in duplicate_groups.items()
        if len({row["_split"] for row in group}) > 1
    }

    within_split_groups = {
        digest: group
        for digest, group in duplicate_groups.items()
        if len({row["_split"] for row in group}) == 1
    }

    if within_split_groups:
        print(
            f"FAIL: {len(within_split_groups)} duplicate-content groups "
            "exist within individual splits"
        )

    if cross_split_groups:
        print(
            f"FAIL: {len(cross_split_groups)} duplicate-content groups "
            "exist across splits"
        )

        for digest, group in sorted(cross_split_groups.items()):
            print(f"  SHA256: {digest}")
            for row in group:
                print(
                    f"    {row['_split']}: "
                    f"image_id={row['image_id']} "
                    f"category={row['original_category_id']} "
                    f"path={row['image_path']}"
                )

    if duplicate_groups:
        raise SystemExit(1)

    print("SHA256 AUDIT PASS")
    print("No duplicate image content exists within or across splits.")


if __name__ == "__main__":
    main()
