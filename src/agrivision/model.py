"""Backbone factory and checkpoint bundling.

The original project saved a bare ``.pkl`` of the classifier. If the feature
code changed, the old model kept predicting -- just wrongly, and silently.
Every checkpoint here carries the class list, the image size, the normalisation
constants and a schema version, and loading refuses to proceed on a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

SCHEMA_VERSION = 2

SUPPORTED_BACKBONES = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, 240),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 260),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 224),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 224),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        224,
    ),
}


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    num_classes: int = 2
    image_size: int = 224
    dropout: float = 0.3
    class_names: list[str] = field(default_factory=list)
    pretrained: bool = True


def default_image_size(backbone: str) -> int:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(SUPPORTED_BACKBONES)}")
    return SUPPORTED_BACKBONES[backbone][2]


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.backbone!r}. Choose from {sorted(SUPPORTED_BACKBONES)}")

    ctor, weights_enum, _ = SUPPORTED_BACKBONES[cfg.backbone]
    model = ctor(weights=weights_enum if cfg.pretrained else None)

    # Swap the ImageNet head for one sized to our classes.
    if cfg.backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout, inplace=True),
            nn.Linear(in_features, cfg.num_classes),
        )
    elif cfg.backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_features, cfg.num_classes),
        )
    elif cfg.backbone.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, cfg.num_classes)
    else:  # pragma: no cover -- guarded by the membership check above
        raise ValueError(f"No head-replacement rule for {cfg.backbone!r}")

    return model


def classifier_parameters(model: nn.Module, backbone: str):
    """Parameters belonging to the freshly-initialised head."""
    if backbone.startswith("resnet"):
        return model.fc.parameters()
    return model.classifier.parameters()


def set_backbone_frozen(model: nn.Module, backbone: str, frozen: bool) -> None:
    """Freeze or unfreeze everything except the classification head.

    Stage 1 trains only the head, so the large random gradients from an
    untrained head do not wreck the pretrained features. Stage 2 unfreezes.
    """
    head_ids = {id(p) for p in classifier_parameters(model, backbone)}
    for param in model.parameters():
        if id(param) not in head_ids:
            param.requires_grad = not frozen


def param_groups(model: nn.Module, backbone: str, head_lr: float, backbone_lr: float):
    """Discriminative learning rates: the pretrained trunk moves slower than the head."""
    head_ids = {id(p) for p in classifier_parameters(model, backbone)}
    head, trunk = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (head if id(param) in head_ids else trunk).append(param)

    groups = [{"params": head, "lr": head_lr}]
    if trunk:
        groups.append({"params": trunk, "lr": backbone_lr})
    return groups


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: ModelConfig,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics or {},
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, ModelConfig, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain one first:  python -m agrivision.train --data-root <dataset>"
        )

    payload = torch.load(path, map_location=device, weights_only=False)

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema v{version} does not match code schema v{SCHEMA_VERSION}. "
            "Retrain rather than risk silently wrong predictions."
        )

    cfg = ModelConfig(**payload["config"])
    model = build_model(ModelConfig(**{**payload["config"], "pretrained": False}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, cfg, payload.get("metrics", {})
