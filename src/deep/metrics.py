from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def classification_metrics(
    targets: torch.Tensor, logits: torch.Tensor, num_classes: int
) -> tuple[dict[str, float], np.ndarray]:
    targets = targets.cpu().long()
    logits = logits.cpu().float()
    predicted = logits.argmax(dim=1)
    top5 = logits.topk(k=min(5, num_classes), dim=1).indices
    top1 = (predicted == targets).float().mean().item()
    top5_accuracy = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets.numpy(), predicted.numpy()), 1)
    true_positives = np.diag(matrix).astype(np.float64)
    predicted_counts = matrix.sum(axis=0)
    true_counts = matrix.sum(axis=1)
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
            "top1_accuracy": float(top1),
            "top5_accuracy": float(top5_accuracy),
            "overall_accuracy": float(top1),
            "macro_precision": float(precision_per_class.mean()),
            "macro_recall": float(recall_per_class.mean()),
            "macro_f1": float(f1_per_class.mean()),
        },
        matrix,
    )


def prediction_frame(
    metadata: list[dict[str, str]],
    targets: torch.Tensor,
    logits: torch.Tensor,
) -> pd.DataFrame:
    probabilities = logits.float().softmax(dim=1)
    confidences, top5 = probabilities.topk(k=min(5, probabilities.shape[1]), dim=1)
    predicted = top5[:, 0]
    records: list[dict[str, Any]] = []
    for index, item in enumerate(metadata):
        target = int(targets[index])
        records.append(
            {
                "image_path": item["image_path"],
                "image_id": item["image_id"],
                "true_class": target,
                "predicted_class": int(predicted[index]),
                "confidence": float(confidences[index, 0]),
                "is_correct": int(predicted[index]) == target,
                "top5_classes": json.dumps(top5[index].tolist()),
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
