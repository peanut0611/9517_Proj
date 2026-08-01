from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .classifiers import build_classifier, decision_scores
from .common import environment_text, load_yaml, resolve_path, save_json, set_seed
from .data import load_split_csv, remap_labels, select_class_subset
from .encode import apply_tfidf, encode_frame, fit_tfidf
from .metrics import classification_metrics, prediction_frame, save_evaluation_outputs
from .sift import extract_and_cache_split, make_sift
from .vocabulary import (
    build_vocabulary,
    sample_training_descriptors,
    save_vocabulary,
)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_frames(config: dict[str, Any], project_root: Path):
    data_cfg = config["data"]
    seed = int(config["experiment"]["seed"])
    metadata_dir = resolve_path(project_root, data_cfg["metadata_dir"])
    data_root = resolve_path(project_root, data_cfg["root"])

    class_indices = select_class_subset(
        metadata_dir=metadata_dir,
        num_classes=data_cfg.get("num_classes"),
        seed=seed,
    )

    train = load_split_csv(
        metadata_dir / "train.csv",
        expected_split="train",
        class_indices=class_indices,
        max_samples=data_cfg.get("max_train_samples"),
        sample_seed=seed,
    )
    val = load_split_csv(
        metadata_dir / "val.csv",
        expected_split="val",
        class_indices=class_indices,
        max_samples=data_cfg.get("max_val_samples"),
        sample_seed=seed,
    )
    test = load_split_csv(
        metadata_dir / "test.csv",
        expected_split="test",
        class_indices=class_indices,
        max_samples=data_cfg.get("max_test_samples"),
        sample_seed=seed,
    )

    train = remap_labels(train, class_indices)
    val = remap_labels(val, class_indices)
    test = remap_labels(test, class_indices)

    if class_indices is None:
        num_classes = int(data_cfg.get("full_num_classes", 500))
    else:
        num_classes = len(class_indices)

    return {
        "data_root": data_root,
        "metadata_dir": metadata_dir,
        "train": train,
        "val": val,
        "test": test,
        "class_indices": None if class_indices is None else sorted(class_indices),
        "num_classes": num_classes,
    }


def train_and_evaluate_classifiers(
    config: dict[str, Any],
    prepared: dict[str, Any],
    output_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    classifier_results: dict[str, Any] = {}
    for clf_cfg in config["classifiers"]:
        name = str(clf_cfg["name"])
        clf_dir = output_dir / name
        clf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training classifier: {name}", flush=True)

        model = build_classifier(
            name=name,
            seed=int(config["experiment"]["seed"]),
            params=clf_cfg.get("params", {}),
        )

        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_sec = time.perf_counter() - t0
        joblib.dump(model, clf_dir / "model.joblib")
        print(f"  fitted in {train_sec:.1f}s", flush=True)

        split_metrics: dict[str, Any] = {}
        for split_name, features, labels, frame in (
            ("val", x_val, y_val, prepared["val"]),
            ("test", x_test, y_test, prepared["test"]),
        ):
            t0 = time.perf_counter()
            scores = decision_scores(model, features)
            infer_sec = time.perf_counter() - t0
            metrics, matrix = classification_metrics(
                labels, scores, num_classes=prepared["num_classes"]
            )
            predictions = prediction_frame(frame, labels, scores)
            save_evaluation_outputs(clf_dir, split_name, metrics, matrix, predictions)
            save_json(
                {
                    "inference_sec": infer_sec,
                    "images": int(len(labels)),
                    "sec_per_image": infer_sec / max(len(labels), 1),
                },
                clf_dir / f"timing_{split_name}.json",
            )
            split_metrics[split_name] = metrics
            print(
                f"  {split_name}: top1={metrics['top1_accuracy']:.4f} "
                f"top5={metrics['top5_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )

        save_json(
            {
                "train_sec": train_sec,
                "n_train": int(len(y_train)),
            },
            clf_dir / "timing.json",
        )
        classifier_results[name] = {
            "train_sec": train_sec,
            "metrics": split_metrics,
        }
    return classifier_results


def run_pipeline(
    config: dict[str, Any],
    project_root: Path,
    from_features: bool = False,
) -> Path:
    set_seed(int(config["experiment"]["seed"]))
    prepared = prepare_frames(config, project_root)

    output_dir = resolve_path(project_root, config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.npz"

    timing: dict[str, float] = {}
    sift_stats: dict[str, Any] = {}
    sampled_descriptors = 0

    if from_features:
        if not features_path.exists():
            raise FileNotFoundError(
                f"--from-features requested but missing: {features_path}"
            )
        print(f"Resuming from cached features: {features_path}", flush=True)
        cached = np.load(features_path)
        x_train = cached["x_train"]
        y_train = cached["y_train"]
        x_val = cached["x_val"]
        y_val = cached["y_val"]
        x_test = cached["x_test"]
        y_test = cached["y_test"]
        timing["resumed_from_features"] = True
        if (output_dir / "vocabulary.joblib").exists():
            sampled_descriptors = -1  # unknown after resume; vocab already saved
    else:
        cache_root = resolve_path(project_root, config["sift"]["cache_dir"])
        cache_root.mkdir(parents=True, exist_ok=True)

        sift = make_sift(
            n_features=int(config["sift"].get("n_features", 0)),
            contrast_threshold=float(config["sift"].get("contrast_threshold", 0.04)),
        )
        max_side = config["sift"].get("max_side", 512)

        for split_name in ("train", "val", "test"):
            t0 = time.perf_counter()
            sift_stats[split_name] = extract_and_cache_split(
                frame=prepared[split_name],
                data_root=prepared["data_root"],
                cache_dir=cache_root / split_name,
                sift=sift,
                max_side=max_side,
                overwrite=bool(config["sift"].get("overwrite_cache", False)),
            )
            timing[f"sift_{split_name}_sec"] = time.perf_counter() - t0

        vocab_cfg = config["vocabulary"]
        vocab_path = output_dir / "vocabulary.joblib"
        t0 = time.perf_counter()
        descriptors = sample_training_descriptors(
            train_frame=prepared["train"],
            cache_dir=cache_root / "train",
            max_descriptors=int(vocab_cfg["max_descriptors"]),
            seed=int(config["experiment"]["seed"]),
        )
        sampled_descriptors = int(descriptors.shape[0])
        vocabulary = build_vocabulary(
            descriptors=descriptors,
            vocabulary_size=int(vocab_cfg["size"]),
            seed=int(config["experiment"]["seed"]),
            batch_size=int(vocab_cfg.get("batch_size", 4096)),
            max_iter=int(vocab_cfg.get("max_iter", 100)),
        )
        save_vocabulary(vocabulary, vocab_path)
        timing["vocabulary_sec"] = time.perf_counter() - t0

        encode_cfg = config.get("encode", {})
        l2_normalize = bool(encode_cfg.get("l2_normalize", True))
        use_tfidf = bool(encode_cfg.get("tfidf", True))

        t0 = time.perf_counter()
        x_train, y_train = encode_frame(
            prepared["train"], cache_root / "train", vocabulary, l2_normalize=False
        )
        x_val, y_val = encode_frame(
            prepared["val"], cache_root / "val", vocabulary, l2_normalize=False
        )
        x_test, y_test = encode_frame(
            prepared["test"], cache_root / "test", vocabulary, l2_normalize=False
        )
        timing["encode_sec"] = time.perf_counter() - t0

        if use_tfidf:
            idf = fit_tfidf(x_train)
            x_train = apply_tfidf(x_train, idf, l2_normalize=l2_normalize)
            x_val = apply_tfidf(x_val, idf, l2_normalize=l2_normalize)
            x_test = apply_tfidf(x_test, idf, l2_normalize=l2_normalize)
            np.save(output_dir / "idf.npy", idf)
        elif l2_normalize:
            for arr_name, arr in (("train", x_train), ("val", x_val), ("test", x_test)):
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                if arr_name == "train":
                    x_train = arr / norms
                elif arr_name == "val":
                    x_val = arr / norms
                else:
                    x_test = arr / norms

        np.savez_compressed(
            features_path,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        print(f"Saved features to {features_path}", flush=True)

    classifier_results = train_and_evaluate_classifiers(
        config=config,
        prepared=prepared,
        output_dir=output_dir,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    experiment_config = {
        "experiment": config["experiment"],
        "data": {
            **config["data"],
            "resolved_root": str(prepared["data_root"]),
            "selected_class_indices": prepared["class_indices"],
            "num_classes_effective": prepared["num_classes"],
            "n_train": int(len(prepared["train"])),
            "n_val": int(len(prepared["val"])),
            "n_test": int(len(prepared["test"])),
        },
        "sift": config["sift"],
        "vocabulary": {
            **config["vocabulary"],
            "sampled_descriptors": sampled_descriptors,
        },
        "encode": config.get("encode", {}),
        "classifiers": config["classifiers"],
        "sift_stats": sift_stats,
        "from_features": from_features,
    }
    save_json(experiment_config, output_dir / "experiment_config.json")
    save_json(timing, output_dir / "timing.json")
    save_json(classifier_results, output_dir / "summary.json")
    (output_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member B: SIFT-BoVW + SVM/RandomForest pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data_metadata/ and Dataset_v1.0_FROZEN/",
    )
    parser.add_argument(
        "--from-features",
        action="store_true",
        help="Skip SIFT/vocabulary/encode and train classifiers from features.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root or project_root_from_here()
    project_root = project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_yaml(config_path)
    output_dir = run_pipeline(
        config, project_root, from_features=bool(args.from_features)
    )
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
