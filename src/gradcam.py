"""
Grad-CAM — a heatmap of WHERE the model looked to decide "AI-generated".

Standalone module: given a model and a preprocessed tensor, it returns a heatmap
and can overlay it on the original image. Knows nothing about the web/CLI layers.

How it works (classic Grad-CAM): hook the last convolutional block to capture its
feature maps and their gradients w.r.t. the target class logit. Weight each feature
map by its averaged gradient (how much it pushes the target class up), sum, ReLU,
and normalize. Bright = regions that drove the model toward the target class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


def default_target_layer(model, arch: str):
    """The last conv block to attribute against, per architecture."""
    if arch.startswith("resnet"):
        return model.layer4[-1]          # last bottleneck/basic block
    if arch.startswith("convnext"):
        return model.features[-1]        # last ConvNeXt stage
    raise ValueError(f"no Grad-CAM target layer configured for arch {arch!r}")


class GradCAM:
    """Grad-CAM for a single model + target conv layer.

    Registers hooks once at construction; call generate() per image. Not thread-safe
    (shared activation/gradient buffers), which is fine for a single-worker app.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Preprocessed (1,3,H,W) tensor -> HxW heatmap in [0,1] at input resolution."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)                    # forward (grad enabled)
        logits[0, class_idx].backward()                # backprop the target logit

        # weight each feature map by its globally-averaged gradient
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.5, colormap: str = "inferno") -> Image.Image:
    """Blend a [0,1] heatmap over the original image, returned at the image's size."""
    base = image.convert("RGB")
    hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        base.size, Image.BILINEAR)
    colored = colormaps[colormap](np.asarray(hm) / 255.0)[..., :3]  # HxWx3 in [0,1]
    colored_img = Image.fromarray((colored * 255).astype(np.uint8))
    return Image.blend(base, colored_img, alpha)
