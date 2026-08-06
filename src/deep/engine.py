from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top5_correct = 0
    started = time.perf_counter()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
            scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % grad_accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        count = targets.shape[0]
        total_examples += count
        total_loss += float(loss.detach()) * count
        top5 = logits.detach().topk(k=min(5, logits.shape[1]), dim=1).indices
        top1_correct += int((top5[:, 0] == targets).sum())
        top5_correct += int((top5 == targets.unsqueeze(1)).any(dim=1).sum())

    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "loss": total_loss / total_examples,
        "top1_accuracy": top1_correct / total_examples,
        "top5_accuracy": top5_correct / total_examples,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, torch.Tensor, torch.Tensor, list[dict[str, str]], float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    metadata: list[dict[str, str]] = []

    # Warm up kernels outside the measured interval. The loader is recreated by
    # the subsequent for-loop, so every sample is still evaluated exactly once.
    warmup_batch = next(iter(loader))
    warmup_images = warmup_batch["image"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        model(warmup_images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        count = targets.shape[0]
        total_examples += count
        total_loss += float(loss) * count
        targets_parts.append(targets.cpu())
        logits_parts.append(logits.float().cpu())
        metadata.extend(
            {
                "image_path": path,
                "image_id": image_id,
            }
            for path, image_id in zip(batch["image_path"], batch["image_id"])
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        total_loss / total_examples,
        torch.cat(targets_parts),
        torch.cat(logits_parts),
        metadata,
        elapsed,
    )
