#!/usr/bin/env python3
"""Extract only the images listed in the frozen split CSV files.

The official iNaturalist archives are scanned sequentially, but only the selected
subset is written to disk. Re-running the script safely skips completed files.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the selected iNaturalist subset from official archives."
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_paths(csv_path: Path) -> set[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["image_path"] for row in csv.DictReader(handle)}


def normalise_member_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {name}")
    return path.as_posix()


def safe_destination(output_root: Path, member_name: str) -> Path:
    destination = output_root.joinpath(*PurePosixPath(member_name).parts)
    root_resolved = output_root.resolve()
    destination_resolved = destination.resolve()
    if os.path.commonpath((root_resolved, destination_resolved)) != str(root_resolved):
        raise ValueError(f"Archive member escapes output root: {member_name}")
    return destination


def extract_selected(
    archive_path: Path,
    selected_paths: set[str],
    output_root: Path,
    label: str,
) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    already_complete = {
        relative_path
        for relative_path in selected_paths
        if (output_root / relative_path).is_file()
        and (output_root / relative_path).stat().st_size > 0
    }
    remaining = selected_paths - already_complete
    print(
        f"{label}: requested={len(selected_paths)}, "
        f"already_complete={len(already_complete)}, remaining={len(remaining)}"
    )
    if not remaining:
        return

    extracted = 0
    scanned = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            scanned += 1
            if scanned % 50_000 == 0:
                print(
                    f"{label}: scanned={scanned}, extracted={extracted}, "
                    f"remaining={len(remaining)}",
                    flush=True,
                )
            if not member.isfile():
                continue

            member_name = normalise_member_name(member.name)
            if member_name not in remaining:
                continue

            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member: {member_name}")

            destination = safe_destination(output_root, member_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

            remaining.remove(member_name)
            extracted += 1

    if remaining:
        examples = sorted(remaining)[:10]
        raise RuntimeError(
            f"{label}: {len(remaining)} requested images were not found. "
            f"Examples: {examples}"
        )
    print(f"{label}: extraction complete ({len(selected_paths)} files)")


def main() -> None:
    args = parse_args()
    train_paths = read_paths(args.metadata_dir / "train.csv")
    validation_paths = read_paths(args.metadata_dir / "val.csv")
    test_paths = read_paths(args.metadata_dir / "test.csv")

    if train_paths & validation_paths:
        raise ValueError("Train and validation path lists overlap.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    extract_selected(
        args.train_archive,
        train_paths | validation_paths,
        args.output_root,
        "official train_mini archive",
    )
    extract_selected(
        args.test_archive,
        test_paths,
        args.output_root,
        "official validation archive",
    )
    print(f"Selected dataset extracted under {args.output_root}")


if __name__ == "__main__":
    main()

