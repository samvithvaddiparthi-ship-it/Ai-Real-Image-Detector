# AI Image Detector

A binary image classifier that distinguishes **AI-generated images from real photographs**,
built around a single question that most detectors ignore: *does it still work on a
generator it has never seen?*

The model is trained on real photos plus five image generators, with **Midjourney held
out entirely** as an unseen-generator test. It ships as a CLI and a local web app that
return a verdict, a calibrated probability, and a Grad-CAM heatmap of where the model
looked.

- **Backbone:** ResNet50 (ImageNet-pretrained, fine-tuned)
- **In-distribution accuracy:** 96.0% (four seen generators)
- **Held-out Midjourney recall:** 71.5% at 92.6% precision (a generator never seen in training)
- **Tech:** PyTorch · torchvision (MPS/CUDA) · FastAPI · vanilla HTML/CSS/JS

---

## Why this project exists

The previous version of this detector reported high accuracy but fell apart in the real
world. A full post-mortem (`reports/diagnosis.md`) traced it to a handful of classic
mistakes: training on a single generator (so it memorized *that model's* fingerprint
rather than "AI-ness"), data leakage across splits, contaminated labels, no honest
evaluation, and — fatally — a **preprocessing mismatch** where the deployed app normalized
images differently than the training pipeline did.

This is a ground-up rebuild designed around five principles that directly fix those bugs:

1. **One preprocessing contract.** Resize + normalize is defined once (`src/preprocessing.py`)
   and imported by *both* training and inference. They cannot drift.
2. **No data leakage.** Every real photo and its AI counterpart share a `scene_id`; splits
   are made by `scene_id` so no image ever spans train/val/test.
3. **A held-out generator.** Midjourney v6 is excluded from training entirely and used only
   to measure true generalization to an unseen generator.
4. **Honest evaluation.** Confusion matrices, precision/recall/F1, per-generator accuracy,
   and the held-out-generator score — never single-image spot checks.
5. **Calibrated confidence.** Probabilities are temperature-scaled so the reported numbers
   mean what they say.

---

## Results

### Model progression

Each model is evaluated on the in-distribution **test** set (four seen generators) and on
the **held-out Midjourney** set. Held-out numbers below use the default 0.5 decision
threshold; the deployed operating point (0.35) is reported separately.

| Model | Test accuracy | Test F1 (AI) | Held-out MJ recall | Held-out MJ precision |
|-------|:---:|:---:|:---:|:---:|
| ResNet18 — baseline (M1) | 0.954 | 0.954 | 0.575 | 0.928 |
| ResNet18 — + augmentation | 0.930 | 0.931 | 0.626 | 0.918 |
| ConvNeXt-Tiny (GPU) | **0.977** | **0.977** | 0.623 | **0.970** |
| **ResNet50 (GPU) — deployed** | 0.960 | 0.961 | **0.661** | 0.942 |

ResNet50 is the deployed model: it is strictly better than the augmented ResNet18 on
held-out recall, precision, *and* in-distribution accuracy at once. ConvNeXt-Tiny scores
highest in-distribution but is too conservative on the unseen generator.

### Held-out generalization is the headline

The whole point of holding out Midjourney is visible in one number:

- **In-distribution AI recall:** 98.2%
- **Held-out (unseen generator) AI recall:** 66.1% → **71.5% at the deployed threshold**

That ~27-point gap is real and honestly measured. The progression across the rebuild took
held-out recall from **57.5% → 71.5%** (via augmentation, then a stronger backbone, then
threshold calibration) while *improving* precision.

### Per-generator AI detection (ResNet50, test set)

| Generator | Detection recall |
|-----------|:---:|
| DALL·E 3 | 99.5% |
| SDXL | 98.9% |
| Stable Diffusion 2.1 | 97.3% |
| Stable Diffusion 3 | 97.3% |
| Midjourney v6 *(held out)* | 71.5% |

### Operating point & calibration

- **Decision threshold:** chosen from a full precision/recall sweep. At **0.35** the model
  catches 71.5% of unseen-generator AI at 92.6% precision (~5.7% of real photos flagged).
- **Calibration:** temperature scaling fit on the validation set gives **T = 1.059** — the
  model was already well-calibrated (Expected Calibration Error ≈ 0.006 val / 0.013 test),
  so calibration confirms the confidence is trustworthy rather than patching a problem.
  The threshold is remapped to 0.358 in calibrated space (identical verdicts).

---

## Dataset

Derived from the [Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
image dataset (MS-COCO real photographs + five generators).

- **18,514 images** — 9,257 real / 9,257 AI, fully balanced and paired by scene.
- **Five generators**, evenly represented (~1,850 each): Stable Diffusion 2.1, SDXL,
  Stable Diffusion 3, DALL·E 3, Midjourney v6.
- Every image verified with a full pixel decode (0 broken / 0 truncated); 0 duplicate reals.

### Leakage-safe split (by `scene_id`)

| Split | Images | Purpose |
|-------|:---:|---|
| Train | 11,848 | training (four generators) |
| Validation | 1,480 | early stopping + calibration |
| Test | 1,480 | in-distribution evaluation |
| Held-out (Midjourney) | 3,706 | unseen-generator generalization test |

The four training generators are split 80/10/10, stratified by generator. Midjourney is
confined entirely to the held-out set.

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
notebooks/
  colab_train_stronger_backbone.ipynb   GPU training (ResNet50 / ConvNeXt)
reports/               metrics, confusion matrices, threshold sweeps, diagnosis
```

**Architecture note.** The inference engine (`src/inference.py`, `src/gradcam.py`) has no
knowledge of any interface. The CLI and the web server both call into it and nothing else,
so the frontend is fully replaceable without touching the model pipeline. Everything that
governs a prediction — weights, architecture, class order, the preprocessing contract, the
decision threshold, and the calibration temperature — travels *inside* the production
checkpoint, so inference can never disagree with training.

---

## Installation

Requires Python 3.12.

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The trained model is not committed (large binary). Obtain
`models/resnet50_colab_production.pth` by either:

- training it (see **Reproducing the pipeline**), or
- placing an existing `resnet50_colab.pth` in `models/` and running:
  ```bash
  python src/calibrate.py --ckpt models/resnet50_colab.pth --raw-threshold 0.35
  ```

---

## Usage

### Web app

```bash
python -m uvicorn app.server:app --port 8000
```

Open **http://localhost:8000**, drop in an image, and get the verdict, calibrated
probability, and Grad-CAM heatmap. Paste an image with ⌘/Ctrl+V.

### Command line

```bash
python src/cli.py path/to/image.jpg
python src/cli.py path/to/image.jpg --heatmap overlay.png   # also save a Grad-CAM overlay
```

### Reproducing the pipeline

```bash
python src/split_dataset.py                                   # build the leakage-safe split
python src/train.py --arch resnet18 --tag baseline            # train (M1/MPS or CPU)
python src/evaluate.py --ckpt models/resnet18_baseline.pth    # full evaluation
python src/threshold_analysis.py --ckpt models/<model>.pth    # sweep the operating point
python src/calibrate.py --ckpt models/<model>.pth             # temperature scaling
```

Stronger backbones (ResNet50 / ConvNeXt-Tiny) are trained on a GPU via
`notebooks/colab_train_stronger_backbone.ipynb`.

---

## Limitations & future work

- **Unseen generators remain the hard case.** 71.5% recall on Midjourney is a large
  improvement, but its photorealism is genuinely difficult; the biggest remaining lever is
  training on *more* generator diversity.
- Calibration is fit on in-distribution data and is not guaranteed out-of-distribution.
- Possible extensions: batch/folder inference, an API for programmatic use, and packaging
  the app as a standalone launcher.

---

## Acknowledgements

Data from the Defactify image dataset (MS-COCO reals + five generators). Backbones are
ImageNet-pretrained models from `torchvision`.
