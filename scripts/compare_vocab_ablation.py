#!/usr/bin/env python3
"""Summarise BoVW vocabulary-size ablation runs (SVM only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = [
    ("K=500", "outputs/b1_ablate_k500"),
    ("K=1000", "outputs/b1_full500_sift_bovw_k1000"),
    ("K=2000", "outputs/b1_ablate_k2000"),
]


def load_metrics(run_dir: Path, split: str) -> dict | None:
    path = run_dir / "svm" / f"metrics_{split}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV path (default: outputs/ablation_vocabulary_svm.csv)",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    out_csv = args.out_csv or (root / "outputs" / "ablation_vocabulary_svm.csv")

    rows: list[dict] = []
    print(f"{'run':<10} {'split':<5} {'top1':>8} {'top5':>8} {'macro_f1':>10}")
    print("-" * 48)
    for label, rel in DEFAULT_RUNS:
        run_dir = root / rel
        for split in ("val", "test"):
            metrics = load_metrics(run_dir, split)
            if metrics is None:
                print(f"{label:<10} {split:<5} {'MISSING':>8}")
                continue
            row = {
                "run": label,
                "output_dir": rel,
                "split": split,
                "top1_accuracy": metrics["top1_accuracy"],
                "top5_accuracy": metrics["top5_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
            rows.append(row)
            print(
                f"{label:<10} {split:<5} "
                f"{metrics['top1_accuracy']*100:7.2f}% "
                f"{metrics['top5_accuracy']*100:7.2f}% "
                f"{metrics['macro_f1']*100:9.2f}%"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out_csv}")

    # Recommend K by validation macro-F1 / top1
    val_rows = [r for r in rows if r["split"] == "val"]
    if val_rows:
        best = max(val_rows, key=lambda r: (r["macro_f1"], r["top1_accuracy"]))
        print(
            f"\nBest on validation by macro-F1: {best['run']} "
            f"(top1={best['top1_accuracy']*100:.2f}%, "
            f"macro_f1={best['macro_f1']*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
