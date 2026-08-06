#!/usr/bin/env python3
"""Evaluate saved multiclass scores using the group's common metric contract.

Expected NPZ keys:
    scores:        float array with shape (num_images, num_classes)
    true_labels:   integer array with shape (num_images,)
Optional NPZ keys:
    image_paths:   string array with shape (num_images,)
    class_indices: integer array with shape (num_classes,)

The score values may be logits, probabilities, or decision scores. Only their
within-image ranking is used for Top-1 and Top-5 predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _as_integer_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {values.shape}.")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"{name} must contain integer class indices.")
    return values.astype(np.int64, copy=False)


def validate_inputs(
    scores: np.ndarray,
    true_labels: np.ndarray,
    class_indices: np.ndarray | None = None,
    image_paths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize the arrays used by the shared evaluator."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, C); got {scores.shape}.")
    if scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("scores must contain at least one image and two classes.")
    if not np.issubdtype(scores.dtype, np.number) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite numeric values.")
    scores = scores.astype(np.float64, copy=False)

    true_labels = _as_integer_vector(true_labels, "true_labels")
    if len(true_labels) != len(scores):
        raise ValueError(
            f"true_labels has {len(true_labels)} rows but scores has {len(scores)}."
        )

    if class_indices is None:
        class_indices = np.arange(scores.shape[1], dtype=np.int64)
    else:
        class_indices = _as_integer_vector(class_indices, "class_indices")
    if len(class_indices) != scores.shape[1]:
        raise ValueError(
            f"class_indices has {len(class_indices)} values but scores has "
            f"{scores.shape[1]} columns."
        )
    if len(np.unique(class_indices)) != len(class_indices):
        raise ValueError("class_indices must not contain duplicates.")
    unknown = np.setdiff1d(np.unique(true_labels), class_indices)
    if len(unknown):
        raise ValueError(f"true_labels contains classes absent from scores: {unknown.tolist()}.")

    if image_paths is None:
        image_paths = np.asarray([f"image_{index}" for index in range(len(scores))])
    else:
        image_paths = np.asarray(image_paths).astype(str)
        if image_paths.ndim != 1 or len(image_paths) != len(scores):
            raise ValueError("image_paths must be one-dimensional with one path per score row.")
        if np.any(np.char.str_len(image_paths) == 0):
            raise ValueError("image_paths must not contain empty strings.")

    return scores, true_labels, class_indices, image_paths


def evaluate_scores(
    scores: np.ndarray,
    true_labels: np.ndarray,
    class_indices: np.ndarray | None = None,
    image_paths: np.ndarray | None = None,
    *,
    inference_seconds: float | None = None,
    training_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute predictions, Top-k metrics, macro metrics, and confusion matrix."""
    scores, true_labels, class_indices, image_paths = validate_inputs(
        scores, true_labels, class_indices, image_paths
    )
    if inference_seconds is not None and inference_seconds < 0:
        raise ValueError("inference_seconds must be non-negative.")
    if training_seconds is not None and training_seconds < 0:
        raise ValueError("training_seconds must be non-negative.")

    top_k = min(5, scores.shape[1])
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    top_classes = class_indices[ranking]
    predicted = top_classes[:, 0]
    top1 = float(np.mean(predicted == true_labels))
    top5 = float(np.mean(np.any(top_classes == true_labels[:, None], axis=1)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted, labels=class_indices)
    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_class": true_labels,
            "predicted_class": predicted,
            "top5_classes": [
                json.dumps([int(value) for value in row]) for row in top_classes
            ],
            "top1_score": scores[np.arange(len(scores)), ranking[:, 0]],
            "correct_top1": predicted == true_labels,
            "correct_top5": np.any(top_classes == true_labels[:, None], axis=1),
        }
    )
    metrics: dict[str, Any] = {
        "num_images": int(len(true_labels)),
        "num_classes": int(len(class_indices)),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top5_k": int(top_k),
    }
    if inference_seconds is not None:
        metrics["inference_seconds"] = float(inference_seconds)
        metrics["inference_ms_per_image"] = float(
            1000.0 * inference_seconds / len(true_labels)
        )
    if training_seconds is not None:
        metrics["training_seconds"] = float(training_seconds)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "confusion_matrix": matrix,
        "class_indices": class_indices,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_evaluation(
    result: dict[str, Any],
    output_dir: Path | str,
    *,
    model_name: str,
    split: str,
    input_path: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Save the standardized group evaluation artifacts."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Use --overwrite only for a deliberate rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = dict(result["metrics"])
    metrics["model_name"] = model_name
    metrics["split"] = split
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    result["predictions"].to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(
        result["confusion_matrix"],
        index=result["class_indices"],
        columns=result["class_indices"],
    ).to_csv(output_dir / "confusion_matrix.csv", index_label="true_class")

    matrix = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(np.log1p(matrix), cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_title(f"{model_name} — {split} confusion matrix")
    ax.set_xlabel("Predicted class position")
    ax.set_ylabel("True class position")
    fig.colorbar(image, ax=ax, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    config: dict[str, Any] = {
        "interface_version": "1.0",
        "model_name": model_name,
        "split": split,
        "class_order": [int(value) for value in result["class_indices"]],
        "score_contract": "Higher score means a higher predicted rank.",
    }
    if input_path is not None:
        input_path = Path(input_path)
        config["input_file"] = str(input_path)
        config["input_sha256"] = _sha256(input_path)
    (output_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return output_dir


def load_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load the standard group NPZ without allowing pickled object arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"scores", "true_labels"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}.")
        scores = data["scores"].copy()
        true_labels = data["true_labels"].copy()
        class_indices = data["class_indices"].copy() if "class_indices" in data else None
        image_paths = data["image_paths"].copy() if "image_paths" in data else None
    return scores, true_labels, class_indices, image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared COMP9517 Top-1/Top-5/macro evaluation."
    )
    parser.add_argument("--input", type=Path, required=True, help="Standard score NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--inference-seconds", type=float)
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores, true_labels, class_indices, image_paths = load_npz(args.input)
    result = evaluate_scores(
        scores,
        true_labels,
        class_indices,
        image_paths,
        inference_seconds=args.inference_seconds,
        training_seconds=args.training_seconds,
    )
    output = save_evaluation(
        result,
        args.output_dir,
        model_name=args.model_name,
        split=args.split,
        input_path=args.input,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved standardized evaluation to: {output}")


if __name__ == "__main__":
    main()
