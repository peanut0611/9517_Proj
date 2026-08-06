from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture for Member C: {architecture}")
    if pretrained:
        raise ValueError("Member C scratch runs must set pretrained=false")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
