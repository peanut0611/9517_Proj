#!/usr/bin/env python3
"""Export Member B SVM decision scores for Member A's shared evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.traditional.classifiers import decision_scores


def load_paths(metadata_dir: Path, split: str) -> np.ndarray:
    frame = pd.read_csv(metadata_dir / f"{split}.csv")
    if "image_path" not in frame.columns:
        raise ValueError(f"{split}.csv missing image_path")
    return frame["image_path"].astype(str).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/b1_ablate_k2000"),
        help="Experiment directory with features.npz and svm/model.joblib",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data_metadata"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/shared_eval_exports"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="member_b_sift_bovw_svm_k2000",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
    )
    args = parser.parse_args()

    features = np.load(args.run_dir / "features.npz")
    model = joblib.load(args.run_dir / "svm" / "model.joblib")
    timing = json.loads((args.run_dir / "svm" / "timing.json").read_text())
    training_seconds = float(timing["train_sec"])

    split_key = {"val": "val", "test": "test"}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": args.model_name,
        "run_dir": str(args.run_dir),
        "training_seconds_definition": (
            "LinearSVC.fit wall time on BoVW training features only "
            "(excludes SIFT extraction, vocabulary, and encoding)."
        ),
        "inference_seconds_definition": (
            "decision_function / ranking score computation wall time on the "
            "split features only (excludes SIFT and BoVW encoding)."
        ),
        "training_seconds": training_seconds,
        "exports": {},
    }

    for split in args.splits:
        x = features[f"x_{split_key[split]}"]
        y = features[f"y_{split_key[split]}"].astype(np.int64)
        paths = load_paths(args.metadata_dir, split)
        if len(paths) != len(x):
            raise ValueError(
                f"{split}: CSV has {len(paths)} rows but features have {len(x)}"
            )
        if not np.array_equal(y, pd.read_csv(args.metadata_dir / f"{split}.csv")["class_index"].to_numpy()):
            # Full-500 runs keep label == class_index; verify alignment.
            csv_labels = pd.read_csv(args.metadata_dir / f"{split}.csv")["class_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(y, csv_labels):
                raise ValueError(
                    f"{split}: feature labels do not match data_metadata/{split}.csv class_index order"
                )

        scores = decision_scores(model, x).astype(np.float64)
        if scores.shape != (len(x), 500):
            raise ValueError(f"{split}: expected scores shape {(len(x), 500)}, got {scores.shape}")

        timing_split = json.loads(
            (args.run_dir / "svm" / f"timing_{split}.json").read_text()
        )
        inference_seconds = float(timing_split["inference_sec"])

        out_name = f"{args.model_name}_{split}_scores.npz"
        out_path = args.out_dir / out_name
        np.savez_compressed(
            out_path,
            scores=scores,
            true_labels=y.astype(np.int64),
            image_paths=np.asarray(paths, dtype=str),
            class_indices=np.arange(500, dtype=np.int64),
        )
        manifest["exports"][split] = {
            "npz": str(out_path),
            "n_images": int(len(x)),
            "scores_shape": list(scores.shape),
            "inference_seconds": inference_seconds,
            "training_seconds": training_seconds,
        }
        print(
            f"Wrote {out_path}  scores={scores.shape}  "
            f"infer={inference_seconds:.4f}s  train={training_seconds:.1f}s"
        )

    manifest_path = args.out_dir / f"{args.model_name}_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
