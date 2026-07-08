"""
Inference engine — the ONLY place that turns an image into a verdict.

Pure ML/Python: no web, no UI, no CLI imports. Both the CLI and the web server
call into this and nothing else, so the frontend can be replaced without touching
the model pipeline.

Everything that governs a prediction travels inside the production checkpoint:
the weights, the architecture, the class order (0=real, 1=ai), the preprocessing
contract, and the decision threshold. We load them from the artifact rather than
hardcoding — the structural fix for v1, where the app and the training pipeline
silently disagreed on preprocessing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from PIL import Image

from src.model import build_model, get_device
from src.preprocessing import get_eval_transform, load_image

DEFAULT_CKPT = Path("models/resnet50_colab_production.pth")


@dataclass
class Prediction:
    """A single image's result. Plain data — safe to JSON-serialize for the API."""
    verdict: str          # "ai" or "real"
    p_ai: float           # P(image is AI-generated), 0..1 (temperature-calibrated)
    confidence: float     # probability mass on the predicted class, 0..1
    threshold: float      # decision threshold applied (pred=ai iff p_ai >= threshold)
    temperature: float    # calibration temperature (1.0 = uncalibrated)
    model_arch: str       # e.g. "resnet50"
    inference_ms: float   # forward-pass time in milliseconds

    def as_dict(self) -> dict:
        return asdict(self)


class Detector:
    """Loads a production checkpoint once and predicts on images.

    Reusable and stateless per call — construct once, call predict() many times.
    """

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT, device=None):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path}. Download the production model "
                f"or run src/finalize_model.py to create it.")
        self.device = device or get_device()
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.arch: str = ck.get("arch", "resnet18")
        self.class_names: list[str] = ck["class_names"]      # ["real", "ai"]
        self.ai_index: int = self.class_names.index("ai")
        self.threshold: float = ck.get("decision_threshold", 0.5)
        # temperature-scaling calibration (1.0 = none); see src/calibrate.py
        self.temperature: float = ck.get("temperature", 1.0)

        self.model = build_model(self.arch, pretrained=False)
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()

        self.transform = get_eval_transform()  # THE inference contract

    # -- split into preprocess / predict so Grad-CAM can reuse the same tensor --

    def preprocess(self, image) -> tuple[Image.Image, torch.Tensor]:
        """path/PIL -> (RGB PIL image, normalized (1,3,H,W) tensor on device)."""
        pil = load_image(image)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        return pil, tensor

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> Prediction:
        """Run the model on an already-preprocessed tensor."""
        t0 = time.perf_counter()
        logits = self.model(tensor)
        # temperature scaling: soften logits before softmax for calibrated probs
        p_ai = torch.softmax(logits / self.temperature, dim=1)[0, self.ai_index].item()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        verdict = "ai" if p_ai >= self.threshold else "real"
        confidence = p_ai if verdict == "ai" else 1.0 - p_ai
        return Prediction(verdict=verdict, p_ai=p_ai, confidence=confidence,
                          threshold=self.threshold, temperature=self.temperature,
                          model_arch=self.arch, inference_ms=dt_ms)

    def predict(self, image) -> Prediction:
        """Convenience: path/PIL -> Prediction."""
        _, tensor = self.preprocess(image)
        return self.predict_tensor(tensor)
