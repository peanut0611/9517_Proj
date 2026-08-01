from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores.argmax(axis=1)
    top_k = min(5, scores.shape[1])
    top5 = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]

    top1 = float((predicted == y_true).mean())
    top5_hits = np.any(top5 == y_true[:, None], axis=1)
    top5_accuracy = float(top5_hits.mean())

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, predicted), 1)

    true_positives = np.diag(matrix).astype(np.float64)
    predicted_counts = matrix.sum(axis=0).astype(np.float64)
    true_counts = matrix.sum(axis=1).astype(np.float64)
    precision_per_class = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros_like(true_positives),
        where=predicted_counts != 0,
    )
    recall_per_class = np.divide(
        true_positives,
        true_counts,
        out=np.zeros_like(true_positives),
        where=true_counts != 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(true_positives),
        where=(precision_per_class + recall_per_class) != 0,
    )

    return (
        {
            "top1_accuracy": top1,
            "top5_accuracy": top5_accuracy,
            "overall_accuracy": top1,
            "macro_precision": float(precision_per_class.mean()),
            "macro_recall": float(recall_per_class.mean()),
            "macro_f1": float(f1_per_class.mean()),
        },
        matrix,
    )


def prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=np.float64)
    # Convert scores to a simple confidence via softmax-like normalization of shifted scores.
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    top_k = min(5, probs.shape[1])
    top5 = np.argsort(-probs, axis=1)[:, :top_k]
    confidences = np.take_along_axis(probs, top5, axis=1)
    predicted = top5[:, 0]

    records: list[dict[str, Any]] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        true_class = int(y_true[index])
        # Prefer original class_index for reporting when available.
        true_report = int(getattr(row, "class_index", true_class))
        pred_local = int(predicted[index])
        records.append(
            {
                "image_path": str(row.image_path),
                "image_id": str(row.image_id),
                "true_class": true_report,
                "true_label": true_class,
                "predicted_label": pred_local,
                "confidence": float(confidences[index, 0]),
                "is_correct": int(pred_local == true_class),
                "top5_labels": json.dumps(top5[index].tolist()),
                "top5_confidences": json.dumps(confidences[index].tolist()),
            }
        )
    return pd.DataFrame.from_records(records)


def save_evaluation_outputs(
    output_dir: Path,
    split: str,
    metrics: dict[str, float],
    matrix: np.ndarray,
    predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"metrics_{split}.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    np.save(output_dir / f"confusion_matrix_{split}.npy", matrix)
    predictions.to_csv(output_dir / f"predictions_{split}.csv", index=False)
