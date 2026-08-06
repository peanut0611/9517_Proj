from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .common import load_yaml, resolve_path, save_json, select_device
from .data import ManifestImageDataset, build_transform, make_loader
from .engine import evaluate_loader
from .metrics import classification_metrics, prediction_frame, save_evaluation_outputs
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Member C run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    run_dir = args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir
    config = load_yaml(run_dir / "config.yaml")
    device = select_device(str(config["training"].get("device", "auto")))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    data_cfg = config["data"]
    model_cfg = config["model"]
    loader_cfg = config["loader"]
    num_classes = int(model_cfg["num_classes"])

    model = build_model(
        str(model_cfg["architecture"]),
        num_classes,
        bool(model_cfg["pretrained"]),
    ).to(device)
    checkpoint_path = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    metadata_dir = resolve_path(project_root, data_cfg["metadata_dir"])
    dataset = ManifestImageDataset(
        metadata_dir / f"{args.split}.csv",
        resolve_path(project_root, data_cfg["root"]),
        build_transform(
            "eval",
            int(data_cfg["image_size"]),
            int(data_cfg["resize_size"]),
            False,
        ),
        num_classes,
        data_cfg.get("max_val_samples") if args.split == "val" else None,
        int(config["experiment"]["seed"]),
    )
    loader = make_loader(
        dataset,
        int(loader_cfg["batch_size"]),
        int(loader_cfg["workers"]),
        False,
        int(config["experiment"]["seed"]),
        bool(loader_cfg.get("pin_memory", device.type == "cuda")),
    )
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    loss, targets, logits, metadata, inference_seconds = evaluate_loader(
        model, loader, criterion, device, amp_enabled
    )
    metrics, matrix = classification_metrics(targets, logits, num_classes)
    metrics["loss"] = loss
    predictions = prediction_frame(metadata, targets, logits)
    save_evaluation_outputs(run_dir, args.split, metrics, matrix, predictions)
    save_json(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "wall_seconds": time.perf_counter() - started,
            "inference_seconds": inference_seconds,
            "images": len(dataset),
            "ms_per_image": inference_seconds / len(dataset) * 1000,
            "images_per_second": len(dataset) / inference_seconds,
        },
        run_dir / f"timing_{args.split}.json",
    )
    print(f"evaluated split={args.split} images={len(dataset)} metrics={metrics}")


if __name__ == "__main__":
    main()
