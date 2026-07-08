"""
Model + device factory — shared by training and evaluation so they can never
disagree on architecture (same spirit as the one preprocessing contract).

Supports the M1/MPS baseline (resnet18) and the stronger backbones we escalate
to on a Colab GPU (resnet50, convnext_tiny). The checkpoint records which arch
it is, and evaluate.py rebuilds from that field — so a resnet50 checkpoint can
never be loaded into a resnet18 skeleton by mistake.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

ARCHS = ("resnet18", "resnet50", "convnext_tiny")


def get_device() -> torch.device:
    """CUDA (Colab) > MPS (M1) > CPU. One helper, used everywhere."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str = "resnet18", num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Build an ImageNet-pretrained backbone with a fresh num_classes head."""
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        # V2 weights = stronger ImageNet recipe, better features to transfer.
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    return m
