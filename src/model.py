import torch
import torch.nn as nn
from torchvision import models


def _set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


def build_resnet50(num_classes: int = 7, pretrained: bool = True, freeze: str = "head") -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    apply_freezing(model, freeze)
    return model


def build_efficientnet_b0(num_classes: int = 7, pretrained: bool = True, freeze: str = "last_block") -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    apply_freezing(model, freeze)
    return model


def apply_freezing(model: nn.Module, freeze: str = "head") -> None:
    """
    freeze options:
    - none / whole: train all layers
    - head: freeze backbone and train classifier/fc only
    - last_block: train classifier and final convolutional block/layer
    """
    freeze = freeze.lower()
    _set_requires_grad(model, True)

    if freeze in ["none", "whole", "all_trainable"]:
        return

    _set_requires_grad(model, False)

    if hasattr(model, "fc"):
        _set_requires_grad(model.fc, True)
        if freeze == "last_block" and hasattr(model, "layer4"):
            _set_requires_grad(model.layer4, True)
    elif hasattr(model, "classifier"):
        _set_requires_grad(model.classifier, True)
        if freeze == "last_block" and hasattr(model, "features"):
            _set_requires_grad(model.features[-1], True)
            _set_requires_grad(model.features[-2], True)
    else:
        raise ValueError("Unknown model architecture for freezing strategy")


def build_model(architecture: str = "resnet50", num_classes: int = 7, pretrained: bool = True, freeze: str = "head"):
    architecture = architecture.lower()
    if architecture == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained, freeze=freeze)
    if architecture in ["efficientnet_b0", "efficientnet-b0", "efficientnet"]:
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, freeze=freeze)
    raise ValueError(f"Unsupported architecture: {architecture}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
