from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import torch
from torch import nn

from .common import (
    environment_text,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
    select_device,
    set_seed,
    sha256_file,
)
from .data import (
    ManifestImageDataset,
    build_transform,
    make_loader,
    validate_class_mapping,
)
from .engine import evaluate_loader, train_one_epoch
from .metrics import classification_metrics, prediction_frame, save_evaluation_outputs
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Member C scratch ResNet-18.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Temporary override, useful for resume tests.")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "model", "loader", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    if config["model"].get("pretrained", False):
        raise ValueError("Member C training config must use pretrained: false")
    if config["data"]["train_transform"] not in {"standard", "noaug"}:
        raise ValueError("train_transform must be standard or noaug")
    if int(config["training"]["grad_accum_steps"]) < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    scheduler = config["training"].get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("name") != "cosine":
        raise ValueError("training.scheduler must explicitly define name: cosine")
    if int(scheduler.get("t_max", 0)) < 1:
        raise ValueError("training.scheduler.t_max must be positive")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = config["name"].lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config["lr"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    if config["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["t_max"]),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {config['name']}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    stale_epochs: int,
    config: dict[str, Any],
    class_to_idx: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "stale_epochs": stale_epochs,
            "config": config,
            "class_to_idx": class_to_idx,
        },
        path,
    )


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_top1"] for row in rows], label="train Top-1")
    axes[1].plot(epochs, [row["val_top1"] for row in rows], label="val Top-1")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in rows], label="val macro-F1")
    axes[1].set_title("Validation dynamics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    config = load_yaml(args.config)
    validate_config(config)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs

    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic)
    device = select_device(str(config["training"].get("device", "auto")))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"

    output_dir = resolve_path(project_root, config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    (output_dir / "environment.txt").write_text(
        environment_text(device), encoding="utf-8", newline="\n"
    )

    data_cfg = config["data"]
    model_cfg = config["model"]
    loader_cfg = config["loader"]
    train_cfg = config["training"]
    metadata_dir = resolve_path(project_root, data_cfg["metadata_dir"])
    data_root = resolve_path(project_root, data_cfg["root"])
    num_classes = int(model_cfg["num_classes"])
    class_mapping_path = metadata_dir / "class_to_idx.json"
    class_to_idx = validate_class_mapping(class_mapping_path, num_classes)
    config["data"]["class_mapping_sha256"] = sha256_file(class_mapping_path)
    save_yaml(config, output_dir / "config.yaml")

    train_transform = build_transform(
        str(data_cfg["train_transform"]),
        int(data_cfg["image_size"]),
        int(data_cfg["resize_size"]),
        bool(data_cfg.get("color_jitter", False)),
    )
    eval_transform = build_transform(
        "eval",
        int(data_cfg["image_size"]),
        int(data_cfg["resize_size"]),
        False,
    )
    train_dataset = ManifestImageDataset(
        metadata_dir / "train.csv",
        data_root,
        train_transform,
        num_classes,
        data_cfg.get("max_train_samples"),
        seed,
    )
    val_dataset = ManifestImageDataset(
        metadata_dir / "val.csv",
        data_root,
        eval_transform,
        num_classes,
        data_cfg.get("max_val_samples"),
        seed,
    )
    train_loader = make_loader(
        train_dataset,
        int(loader_cfg["batch_size"]),
        int(loader_cfg["workers"]),
        True,
        seed,
        bool(loader_cfg.get("pin_memory", device.type == "cuda")),
    )
    val_loader = make_loader(
        val_dataset,
        int(loader_cfg["batch_size"]),
        int(loader_cfg["workers"]),
        False,
        seed,
        bool(loader_cfg.get("pin_memory", device.type == "cuda")),
    )

    # Reset immediately before model construction so C0/C1 start identically.
    set_seed(seed, deterministic)
    model = build_model(
        str(model_cfg["architecture"]),
        num_classes,
        bool(model_cfg["pretrained"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg["optimizer"])
    epochs = int(train_cfg["epochs"])
    scheduler = build_scheduler(optimizer, train_cfg["scheduler"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_metric = float("-inf")
    resume_stale_epochs: int | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        if "stale_epochs" in checkpoint:
            resume_stale_epochs = int(checkpoint["stale_epochs"])

    history_path = output_dir / "history.csv"
    history: list[dict[str, Any]] = []
    if args.resume is not None and history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                history.append({key: float(value) for key, value in existing.items()})
                history[-1]["epoch"] = int(history[-1]["epoch"])
    patience = int(train_cfg["early_stopping"]["patience"])
    min_delta = float(train_cfg["early_stopping"].get("min_delta", 0.0))
    stale_epochs = resume_stale_epochs if resume_stale_epochs is not None else 0
    if args.resume is not None and resume_stale_epochs is None and history:
        best_history_epoch = max(
            history, key=lambda row: float(row["val_macro_f1"])
        )["epoch"]
        stale_epochs = int(history[-1]["epoch"]) - int(best_history_epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            int(train_cfg["grad_accum_steps"]),
            amp_enabled,
        )
        val_loss, targets, logits, _, val_seconds = evaluate_loader(
            model, val_loader, criterion, device, amp_enabled
        )
        val_metrics, _ = classification_metrics(targets, logits, num_classes)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_top1": train_stats["top1_accuracy"],
            "train_top5": train_stats["top5_accuracy"],
            "val_loss": val_loss,
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_seconds": train_stats["seconds"],
            "val_seconds": val_seconds,
        }
        history.append(row)
        append_history(history_path, row)

        selection_metric = float(val_metrics["macro_f1"])
        improved = selection_metric > best_metric + min_delta
        if improved:
            best_metric = selection_metric
            stale_epochs = 0
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                stale_epochs,
                config,
                class_to_idx,
            )
        else:
            stale_epochs += 1

        scheduler.step()
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            stale_epochs,
            config,
            class_to_idx,
        )
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_top1={row['val_top1']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )
        if stale_epochs >= patience:
            print(f"early_stopping epoch={epoch} patience={patience}")
            break

    total_seconds = time.perf_counter() - total_started
    plot_history(output_dir / "plots" / "training_curves.png", history)

    best_checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    val_loss, targets, logits, metadata, inference_seconds = evaluate_loader(
        model, val_loader, criterion, device, amp_enabled
    )
    metrics, matrix = classification_metrics(targets, logits, num_classes)
    metrics["loss"] = val_loss
    predictions = prediction_frame(metadata, targets, logits)
    save_evaluation_outputs(output_dir, "val", metrics, matrix, predictions)
    save_json(
        {
            "session_wall_seconds": total_seconds,
            "training_total_seconds": sum(
                float(row["train_seconds"]) for row in history
            ),
            "validation_during_training_total_seconds": sum(
                float(row["val_seconds"]) for row in history
            ),
            "validation_inference_seconds": inference_seconds,
            "validation_images": len(val_dataset),
            "validation_ms_per_image": inference_seconds / len(val_dataset) * 1000,
            "validation_images_per_second": len(val_dataset) / inference_seconds,
            "best_epoch": int(best_checkpoint["epoch"]),
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
        output_dir / "timing.json",
    )
    print(f"completed output_dir={output_dir}")


if __name__ == "__main__":
    main()
