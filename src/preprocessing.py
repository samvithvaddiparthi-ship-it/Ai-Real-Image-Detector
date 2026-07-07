"""
THE preprocessing contract for the AI-vs-real detector.

Golden rule #1 (see CLAUDE.md): resize + normalize is defined ONCE, here, and
imported by BOTH training and inference. v1 died because the deployed model was
trained WITHOUT normalization but the app APPLIED ImageNet normalization at
inference -> train/infer input distributions didn't match -> garbage predictions.

The contract, concretely:
  - Every image is opened and converted to RGB the same way (load_image).
  - Evaluation / inference use the EXACT same transform: get_eval_transform().
    This is the single source of truth the deployed app MUST use.
  - Training adds augmentation on top, but the resize target and the SAME
    normalization statistics. Augmentation is train-only and (per golden rule #2)
    is applied only after the split.

If you ever change IMAGE_SIZE, RESIZE_SIZE, IMAGENET_MEAN, or IMAGENET_STD,
you change it here and it propagates to training AND the app automatically.
"""
from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms

# --- The numbers. Single source of truth. -----------------------------------
# 224 is the native input size for ImageNet-pretrained ResNet18 (our backbone).
IMAGE_SIZE = 224
# Resize the shorter side to 256, then center-crop 224 (standard ImageNet eval).
RESIZE_SIZE = 256
# ImageNet stats — required because the backbone is ImageNet-pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The one normalize op both train and eval share.
_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_image(path_or_img) -> Image.Image:
    """Open an image (path or PIL.Image) and force RGB.

    Forcing RGB is part of the contract: reals include grayscale / CMYK / RGBA
    files, and v1 had a grayscale/color inconsistency between models. Everything
    downstream sees 3-channel RGB, always.
    """
    if isinstance(path_or_img, Image.Image):
        img = path_or_img
    else:
        img = Image.open(path_or_img)
    return img.convert("RGB")


def get_eval_transform() -> transforms.Compose:
    """The inference/eval contract. Deterministic: Resize -> CenterCrop -> Normalize.

    Used by validation, test, held-out-generator eval, AND the deployed app.
    No augmentation, no randomness.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def get_train_transform() -> transforms.Compose:
    """Training transform: light augmentation on top of the SAME normalization.

    Kept deliberately light for the Phase 3 baseline (a random crop + horizontal
    flip). Heavier, real-world augmentation (JPEG recompression, downscale/upscale,
    blur) is Phase 5 — added here, still on top of the same _NORMALIZE.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def preprocess_for_inference(path_or_img) -> torch.Tensor:
    """Convenience for the app: path/PIL -> normalized (1, 3, H, W) tensor.

    Uses get_eval_transform() so the app can never drift from the training-time
    eval contract. Returns a batch of size 1, ready for model(x).
    """
    img = load_image(path_or_img)
    return get_eval_transform()(img).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo _NORMALIZE for visualization (e.g. Grad-CAM overlays in Phase 6).

    Accepts (C, H, W) or (N, C, H, W); returns the same shape clamped to [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
