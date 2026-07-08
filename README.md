# AI Image Detector

Binary classifier: **AI-generated vs. real photograph**. ResNet50, fine-tuned on
Defactify (COCO reals + 5 generators), with Midjourney held out for a true
generalization test. Held-out Midjourney AI recall: **71.5%** at precision 92.6%.

## Structure
- `data/raw/{ai,real}/`  — cleaned source images (gitignored)
- `data/splits.csv`      — leakage-safe split by scene_id (mj6 held out)
- `src/`                 — engine: preprocessing, model, training, eval, inference, Grad-CAM
- `app/`                 — web layer (FastAPI + static frontend); UI only, no ML logic
- `models/`              — saved weights (gitignored)
- `notebooks/`           — Colab GPU training notebook
- `reports/`             — diagnosis, metrics, confusion matrices, threshold sweeps

## Architecture
The **inference engine** (`src/inference.py`, `src/gradcam.py`) is fully independent
of any interface. Both the CLI and the web server call into it and nothing else, so
the frontend can be replaced without touching the model pipeline.

Everything that governs a prediction — weights, architecture, class order, the
preprocessing contract, and the decision threshold — travels inside the production
checkpoint. Nothing is hardcoded in the app (the structural fix for v1, where the
deployed app and the training pipeline silently disagreed on preprocessing).

## Golden rule
Preprocessing (resize + normalize) is defined ONCE in `src/preprocessing.py` and
imported by both training and inference. Never redefine transforms separately.

## Running it

Requires the production checkpoint at `models/resnet50_colab_production.pth`
(download `resnet50_colab.pth` from Drive, then
`python src/finalize_model.py --ckpt models/resnet50_colab.pth --threshold 0.35`).

```bash
source venv/bin/activate

# CLI — one image
python src/cli.py path/to/image.jpg
python src/cli.py path/to/image.jpg --heatmap overlay.png   # also save Grad-CAM

# Web app — http://127.0.0.1:8000
python -m uvicorn app.server:app --port 8000
```
