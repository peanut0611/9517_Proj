from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen C1 predictions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def taxonomy_lookup(taxonomy: pd.DataFrame) -> dict[int, dict[str, str]]:
    columns = ["species", "common_name", "genus", "family"]
    return {
        int(row.class_index): {column: str(getattr(row, column)) for column in columns}
        for row in taxonomy.itertuples(index=False)
    }


def attach_taxonomy(
    frame: pd.DataFrame,
    taxonomy: pd.DataFrame,
    class_column: str,
    prefix: str,
) -> pd.DataFrame:
    fields = ["class_index", "species", "common_name", "genus", "family"]
    renamed = taxonomy[fields].rename(
        columns={
            "class_index": class_column,
            **{field: f"{prefix}_{field}" for field in fields if field != "class_index"},
        }
    )
    return frame.merge(renamed, on=class_column, how="left", validate="many_to_one")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.run_dir / "predictions_test.csv")
    taxonomy = pd.read_csv(args.taxonomy).sort_values("class_index")
    matrix = np.load(args.run_dir / "confusion_matrix_test.npy")
    num_classes = len(taxonomy)
    if len(predictions) != 5000 or matrix.shape != (num_classes, num_classes):
        raise ValueError("Frozen prediction or confusion-matrix dimensions are invalid")

    true_counts = predictions["true_class"].value_counts().reindex(range(num_classes), fill_value=0)
    correct_counts = (
        predictions.loc[predictions["true_class"] == predictions["predicted_class"], "true_class"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
    )
    predicted_counts = predictions["predicted_class"].value_counts().reindex(
        range(num_classes), fill_value=0
    )
    class_performance = pd.DataFrame(
        {
            "class_index": range(num_classes),
            "support": true_counts.values,
            "correct": correct_counts.values,
            "accuracy": correct_counts.values / true_counts.values,
            "predicted_count": predicted_counts.values,
        }
    ).merge(taxonomy, on="class_index", how="left", validate="one_to_one")
    class_performance.sort_values(
        ["accuracy", "class_index"], ascending=[False, True]
    ).to_csv(args.output_dir / "class_performance_test.csv", index=False)

    lookup = taxonomy_lookup(taxonomy)
    pairs: list[dict[str, object]] = []
    for left in range(num_classes):
        for right in range(left + 1, num_classes):
            left_to_right = int(matrix[left, right])
            right_to_left = int(matrix[right, left])
            total = left_to_right + right_to_left
            if total == 0:
                continue
            left_tax = lookup[left]
            right_tax = lookup[right]
            pairs.append(
                {
                    "class_a": left,
                    "class_b": right,
                    "a_to_b": left_to_right,
                    "b_to_a": right_to_left,
                    "symmetric_errors": total,
                    "a_species": left_tax["species"],
                    "b_species": right_tax["species"],
                    "a_common_name": left_tax["common_name"],
                    "b_common_name": right_tax["common_name"],
                    "same_genus": left_tax["genus"] == right_tax["genus"],
                    "same_family": left_tax["family"] == right_tax["family"],
                }
            )
    confusion_pairs = pd.DataFrame(pairs).sort_values(
        ["symmetric_errors", "a_to_b", "b_to_a"], ascending=False
    )
    confusion_pairs.head(args.top_k).to_csv(
        args.output_dir / "top_confusion_pairs_test.csv", index=False
    )

    errors = predictions[predictions["true_class"] != predictions["predicted_class"]].copy()
    errors = attach_taxonomy(errors, taxonomy, "true_class", "true")
    errors = attach_taxonomy(errors, taxonomy, "predicted_class", "predicted")
    errors.sort_values("confidence", ascending=False).head(args.top_k).to_csv(
        args.output_dir / "high_confidence_errors_test.csv", index=False
    )

    correct = predictions[predictions["true_class"] == predictions["predicted_class"]].copy()
    correct = attach_taxonomy(correct, taxonomy, "true_class", "true")
    correct.sort_values("confidence", ascending=False).head(args.top_k).to_csv(
        args.output_dir / "high_confidence_correct_test.csv", index=False
    )

    save_json(
        {
            "test_images": len(predictions),
            "correct_images": int(
                (predictions["true_class"] == predictions["predicted_class"]).sum()
            ),
            "incorrect_images": int(
                (predictions["true_class"] != predictions["predicted_class"]).sum()
            ),
            "perfect_accuracy_classes": int((class_performance["accuracy"] == 1).sum()),
            "zero_accuracy_classes": int((class_performance["accuracy"] == 0).sum()),
            "top_confusion_pair": confusion_pairs.iloc[0].to_dict(),
            "selection_rule": "frozen C1 test predictions; no model changes",
        },
        args.output_dir / "error_analysis_summary.json",
    )
    print(f"error analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
