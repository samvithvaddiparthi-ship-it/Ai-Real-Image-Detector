# AI Image Detector

A binary image classifier that distinguishes **AI-generated images from real photographs**,
built around a question most detectors ignore: *does it still work on a generator it has
never seen?* The model trains on real photos plus five image generators, with **Midjourney
held out entirely** as an unseen-generator test.

It ships as a local **web app** and a **CLI** that return a verdict, a calibrated
probability, and a Grad-CAM heatmap of where the model looked.

- **Backbone:** ResNet50 (ImageNet-pretrained, fine-tuned)
- **In-distribution accuracy:** 96.0% across four seen generators
- **Held-out Midjourney recall:** 71.5% at 92.6% precision (a generator never seen in training)
- **Stack:** PyTorch · torchvision · FastAPI · vanilla HTML/CSS/JS

---

## Quick start — run the app with the trained model

**You do not need to train anything or download any dataset.** The trained model
(`models/resnet50_colab_production.pth`) is included in this repo. Just clone, install
dependencies, and run:

```bash
git clone https://github.com/samvithvaddiparthi-ship-it/Ai-Real-Image-Detector.git
cd Ai-Real-Image-Detector

python3 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m uvicorn app.server:app --port 8000
```

Then open **http://localhost:8000**, drop in any image, and read the result. That's it.

Prefer the command line?

```bash
python src/cli.py path/to/image.jpg
python src/cli.py path/to/image.jpg --heatmap overlay.png   # also save a Grad-CAM overlay
```

> **Note:** Runs on CPU out of the box (no GPU required); it automatically uses Apple MPS
> or CUDA if available. Training and the dataset are **only** needed to reproduce the model
> from scratch — see [Reproducing the model](#reproducing-the-model-optional) at the bottom.
> To simply use the detector, ignore that section entirely.

---

## Why this project exists

The previous version of this detector reported high accuracy but fell apart in the real
world. A post-mortem (`reports/diagnosis.md`) traced it to classic mistakes: training on a
single generator (so it memorized *that model's* fingerprint rather than "AI-ness"), data
leakage across splits, contaminated labels, no honest evaluation, and — fatally — a
**preprocessing mismatch** where the deployed app normalized images differently than the
training pipeline did.

This is a ground-up rebuild designed around five principles that fix those bugs:

1. **One preprocessing contract.** Resize + normalize is defined once (`src/preprocessing.py`)
   and imported by *both* training and inference, so they cannot drift apart.
2. **No data leakage.** Every real photo and its AI counterpart share a `scene_id`; splits
   are made by `scene_id` so no image spans train/val/test.
3. **A held-out generator.** Midjourney v6 is excluded from training entirely and used only
   to measure true generalization to an unseen generator.
4. **Honest evaluation.** Confusion matrices, precision/recall/F1, per-generator accuracy,
   and the held-out-generator score — never single-image spot checks.
5. **Calibrated confidence.** Probabilities are temperature-scaled so the numbers mean what
   they say.

---

## Results

### Model progression

Each model is evaluated on the in-distribution **test** set (four seen generators) and on
the **held-out Midjourney** set. Held-out numbers use the default 0.5 threshold; the
deployed operating point (0.35) is reported separately.

| Model | Test accuracy | Test F1 (AI) | Held-out MJ recall | Held-out MJ precision |
|-------|:---:|:---:|:---:|:---:|
| ResNet18 — baseline | 0.954 | 0.954 | 0.575 | 0.928 |
| ResNet18 — + augmentation | 0.930 | 0.931 | 0.626 | 0.918 |
| ConvNeXt-Tiny | **0.977** | **0.977** | 0.623 | **0.970** |
| **ResNet50 — deployed** | 0.960 | 0.961 | **0.661** | 0.942 |

ResNet50 is the deployed model: strictly better than the augmented ResNet18 on held-out
recall, precision, *and* in-distribution accuracy at once.

### Held-out generalization is the headline

- **In-distribution AI recall:** 98.2%
- **Held-out (unseen generator) AI recall:** 66.1% → **71.5% at the deployed threshold**

Across the rebuild, held-out recall climbed **57.5% → 71.5%** (augmentation → stronger
backbone → threshold calibration) while precision *improved*.

### Per-generator AI detection (ResNet50, test set)

| Generator | Detection recall |
|-----------|:---:|
| DALL·E 3 | 99.5% |
| SDXL | 98.9% |
| Stable Diffusion 2.1 | 97.3% |
| Stable Diffusion 3 | 97.3% |
| Midjourney v6 *(held out)* | 71.5% |

### Operating point & calibration

- **Decision threshold 0.35** (from a full precision/recall sweep): catches 71.5% of
  unseen-generator AI at 92.6% precision (~5.7% of real photos flagged).
- **Calibration:** temperature scaling gives **T = 1.059** — the model was already
  well-calibrated (Expected Calibration Error ≈ 0.006 val / 0.013 test). The threshold is
  remapped to 0.358 in calibrated space (identical verdicts).

---

## Dataset

Derived from the [Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
image dataset (MS-COCO real photographs + five generators).

- **18,514 images** — 9,257 real / 9,257 AI, balanced and paired by scene.
- **Five generators** (~1,850 each): Stable Diffusion 2.1, SDXL, Stable Diffusion 3,
  DALL·E 3, Midjourney v6.

### Leakage-safe split (by `scene_id`)

| Split | Images | Purpose |
|-------|:---:|---|
| Train | 11,848 | training (four generators) |
| Validation | 1,480 | early stopping + calibration |
| Test | 1,480 | in-distribution evaluation |
| Held-out (Midjourney) | 3,706 | unseen-generator generalization test |

> The dataset itself (~1.4 GB) is **not** included in the repo — it is only needed to retrain
> or re-evaluate, not to run the app.

---

## Project structure

```
src/
  preprocessing.py     THE resize+normalize contract (shared by train & inference)
  model.py             backbone factory + device selection (CUDA > MPS > CPU)
  dataset.py           split reader; pins label order (0=real, 1=ai)
  split_dataset.py     leakage-safe split by scene_id -> data/splits.csv
  train.py             training loop: early stopping, checkpointing, metrics
  evaluate.py          confusion matrix, P/R/F1, per-generator, held-out score
  threshold_analysis.py  precision/recall sweep to choose the operating point
  calibrate.py         temperature scaling (calibrated confidence)
  finalize_model.py    stamp the decision threshold into a production checkpoint
  inference.py         the inference engine (Detector) — pure, no UI/web imports
  gradcam.py           Grad-CAM heatmap + overlay
  cli.py               command-line detector
app/
  server.py            FastAPI: thin HTTP <-> engine layer (no ML logic)
  static/              custom frontend (index.html, styles.css, app.js)
notebooks/             GPU training notebook (ResNet50 / ConvNeXt)
reports/               metrics, confusion matrices, threshold sweeps, diagnosis
models/                trained production model (shipped); other checkpoints ignored
```

**Architecture note.** The inference engine (`src/inference.py`, `src/gradcam.py`) has no
knowledge of any interface — the CLI and web server both call into it and nothing else, so
the frontend is fully replaceable without touching the model. Everything that governs a
prediction (weights, architecture, class order, the preprocessing contract, the decision
threshold, and the calibration temperature) travels *inside* the production checkpoint, so
inference can never disagree with training.

---

## Reproducing the model (optional)

Skip this section unless you want to retrain from scratch — it is **not** needed to use the
app. It requires the ~1.4 GB dataset (extracted to `data/raw/` with a `data/splits.csv`
manifest) and, for the stronger backbones, a GPU.

```bash
python src/split_dataset.py                                   # build the leakage-safe split
python src/train.py --arch resnet18 --tag baseline            # train locally (CPU/MPS) — a few minutes/epoch
python src/evaluate.py --ckpt models/resnet18_baseline.pth    # full evaluation
python src/threshold_analysis.py --ckpt models/<model>.pth    # sweep the operating point
python src/calibrate.py --ckpt models/<model>.pth             # temperature scaling
```

The deployed ResNet50 / ConvNeXt-Tiny models are trained on a GPU via
`notebooks/colab_train_stronger_backbone.ipynb`.

---

## Limitations & future work

- Unseen generators are still the hard case — 71.5% recall on Midjourney is a big
  improvement, but its photorealism is genuinely difficult; the largest remaining lever is
  training on more generator diversity.
- Calibration is fit on in-distribution data and is not guaranteed out-of-distribution.
- Possible extensions: batch/folder inference, a programmatic API, and packaging the app as
  a standalone launcher.

---

## Acknowledgements

Data from the Defactify image dataset (MS-COCO reals + five generators). Backbones are
ImageNet-pretrained models from `torchvision`.
