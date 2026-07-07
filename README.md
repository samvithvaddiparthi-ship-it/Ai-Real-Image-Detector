# AI Image Detector

Binary classifier: AI-generated vs. real photograph.

## Status
Rebuild in progress (see reports/diagnosis.md for why v1 failed).

## Structure
- `data/raw/{ai,real}/`  — cleaned source images (gitignored)
- `data/splits/`         — train/val/test, generated from raw (gitignored)
- `src/`                 — training, evaluation, inference code
- `models/`              — saved weights (gitignored)
- `notebooks/`           — Colab GPU notebooks
- `reports/`             — diagnosis, metrics, confusion matrices

## Golden rule
Preprocessing (resize + normalize) is defined ONCE in src/ and imported by
both training and inference. Never redefine transforms separately.
