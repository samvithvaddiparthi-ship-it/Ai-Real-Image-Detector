"""
Web API for the detector. THIN by design: it only marshals HTTP <-> the inference
engine. No model logic lives here — it calls src.inference and src.gradcam, exactly
what the CLI does. Swap this server (or the frontend) without touching the pipeline.

Run:  uvicorn app.server:app --reload --port 8000    (or: python app/server.py)
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.inference import Detector
from src.gradcam import GradCAM, default_target_layer, overlay_heatmap

STATIC_DIR = Path(__file__).parent / "static"
DISPLAY_MAX = 640  # cap returned images so payloads stay small

app = FastAPI(title="AI Image Detector")

# Load the model + Grad-CAM once at startup (the engine, reused per request).
_detector = Detector()
_cam = GradCAM(_detector.model, default_target_layer(_detector.model, _detector.arch))


def _to_data_uri(img: Image.Image) -> str:
    thumb = img.copy()
    thumb.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Image upload -> verdict + probabilities + Grad-CAM overlay (JSON)."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read that file as an image.")

    original, tensor = _detector.preprocess(image)
    prediction = _detector.predict_tensor(tensor)
    heatmap = _cam.generate(tensor, _detector.ai_index)
    overlay = overlay_heatmap(original, heatmap)

    return {
        **prediction.as_dict(),
        "original": _to_data_uri(original),
        "heatmap": _to_data_uri(overlay),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": _detector.arch, "threshold": _detector.threshold}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# static assets (styles.css, app.js) — mounted after routes so /api/* wins
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
