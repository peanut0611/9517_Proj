#!/usr/bin/env python3
"""Export success / failure examples from BoVW prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_taxonomy(metadata_dir: Path) -> pd.DataFrame:
    tax = pd.read_csv(metadata_dir / "taxonomy.csv")
    # Expect class_index + species-like fields; be tolerant of column names.
    if "class_index" not in tax.columns:
        raise ValueError(f"taxonomy.csv missing class_index: {list(tax.columns)}")
    return tax


def species_name(row: pd.Series | object) -> str:
    def get(key: str):
        if isinstance(row, pd.Series):
            return row.get(key)
        return getattr(row, key, None)

    for key in ("species", "common_name", "genus"):
        value = get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return str(value)
    class_index = get("class_index")
    return f"class_{int(class_index)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/b1_ablate_k2000/svm/predictions_test.csv"),
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data_metadata"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/member_b/cases_k2000_svm"),
    )
    parser.add_argument("--n-success", type=int, default=10)
    parser.add_argument("--n-failure", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9517)
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)
    tax = load_taxonomy(args.metadata_dir)
    tax = tax.copy()
    tax["class_index"] = tax["class_index"].astype(int)
    name_map = {
        int(row["class_index"]): species_name(row) for _, row in tax.iterrows()
    }

    preds = preds.copy()
    preds["true_name"] = preds["true_class"].map(name_map)
    preds["pred_name"] = preds["predicted_label"].map(name_map)
    preds["confidence"] = preds["confidence"].astype(float)

    correct = preds[preds["is_correct"] == 1].sort_values(
        "confidence", ascending=False
    )
    wrong = preds[preds["is_correct"] == 0].sort_values(
        "confidence", ascending=False
    )

    # High-confidence successes / confident mistakes (useful for discussion).
    success = correct.head(args.n_success)
    failure = wrong.head(args.n_failure)

    # Also sample a few mid-confidence failures for variety.
    mid = wrong.iloc[len(wrong) // 3 : len(wrong) // 3 + max(3, args.n_failure // 3)]
    failure = (
        pd.concat([failure, mid])
        .drop_duplicates(subset=["image_id"])
        .head(args.n_failure)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    success_path = args.out_dir / "success_cases.csv"
    failure_path = args.out_dir / "failure_cases.csv"
    success.to_csv(success_path, index=False)
    failure.to_csv(failure_path, index=False)

    # Markdown summary for report / video.
    lines = [
        "# Case study shortlist — SIFT-BoVW SVM (K=2000, test)",
        "",
        f"Source: `{args.predictions}`",
        "",
        "## High-confidence correct predictions",
        "",
        "| # | true | pred | conf | image_path |",
        "|---|---|---|---:|---|",
    ]
    for i, row in enumerate(success.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.true_name} | {row.pred_name} | {row.confidence:.4f} | `{row.image_path}` |"
        )

    lines += [
        "",
        "## High-confidence errors (model is wrong but confident)",
        "",
        "| # | true | pred | conf | image_path |",
        "|---|---|---|---:|---|",
    ]
    for i, row in enumerate(failure.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.true_name} | {row.pred_name} | {row.confidence:.4f} | `{row.image_path}` |"
        )

    lines += [
        "",
        "## How to use in the report / video",
        "",
        "1. Open the listed paths under `Dataset_v1.0_FROZEN/`.",
        "2. For successes: note distinctive local texture/parts that SIFT can capture.",
        "3. For failures: check same-genus confusion, cluttered background, or tiny subject.",
        "4. Prefer 3–5 successes and 3–5 failures in the final PDF/video.",
        "",
    ]
    md_path = args.out_dir / "CASE_SHORTLIST.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "predictions": str(args.predictions),
        "n_test": int(len(preds)),
        "n_correct": int((preds["is_correct"] == 1).sum()),
        "n_wrong": int((preds["is_correct"] == 0).sum()),
        "success_csv": str(success_path),
        "failure_csv": str(failure_path),
        "markdown": str(md_path),
    }
    (args.out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
