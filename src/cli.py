"""
Command-line detector. Pure orchestration: it calls the inference engine and
Grad-CAM module and prints/saves the result — no model logic lives here.

Usage:
  python src/cli.py path/to/image.jpg
  python src/cli.py image.jpg --heatmap out.png            # also save a Grad-CAM overlay
  python src/cli.py image.jpg --ckpt models/other.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference import Detector, DEFAULT_CKPT
from src.gradcam import GradCAM, default_target_layer, overlay_heatmap


def main():
    ap = argparse.ArgumentParser(description="AI-vs-real image detector")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="production checkpoint")
    ap.add_argument("--heatmap", default=None,
                    help="if set, write a Grad-CAM overlay PNG to this path")
    args = ap.parse_args()

    detector = Detector(args.ckpt)
    pil, tensor = detector.preprocess(args.image)
    pred = detector.predict_tensor(tensor)

    label = "AI-generated" if pred.verdict == "ai" else "Real photograph"
    print(f"\n  {Path(args.image).name}")
    print(f"  verdict:      {label}")
    print(f"  P(AI):        {pred.p_ai:.1%}")
    print(f"  confidence:   {pred.confidence:.1%}")
    print(f"  threshold:    {pred.threshold:.2f}  (verdict=AI iff P(AI) >= threshold)")
    print(f"  model:        {pred.model_arch}")
    print(f"  inference:    {pred.inference_ms:.0f} ms")

    if args.heatmap:
        cam = GradCAM(detector.model, default_target_layer(detector.model, detector.arch))
        heatmap = cam.generate(tensor, detector.ai_index)
        overlay_heatmap(pil, heatmap).save(args.heatmap)
        print(f"  heatmap:      {args.heatmap}")


if __name__ == "__main__":
    main()
