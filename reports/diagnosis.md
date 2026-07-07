# v1 Diagnosis — why the first AI image detector failed

Based on a full read of `ML_model_latest.ipynb` (367 cells), the old `app.py`,
and `requirements.txt`. Findings are grounded in the actual code, not the
project summary.

## Architecture (all 4 models identical)
- `resnet18` (ImageNet-pretrained), final layer -> `Linear(512, 2)`
- `CrossEntropyLoss`, `Adam`
- ImageFolder alphabetical class order => **label 0 = ai, label 1 = real**

## The four saved models — each trained with DIFFERENT preprocessing
| File | Cell | Preprocessing | Training |
|------|------|---------------|----------|
| `ai_detector_resnet18.pth` | 173 | Resize + **Normalize** | 10 epochs, all layers |
| `ai_detector_resnet18_improved.pth` | 195 | Resize + aug, **NO Normalize** | 10 epochs, all layers (deployed to Mac) |
| `ai_detector_resnet18_fixed.pth` | 270 | Normalize + grayscale | froze backbone, fc only, 3 epochs |
| `ai_detector_resnet18_final.pth` | 279 | Normalize + grayscale + blur + sharpen | layer4 + fc, 2 epochs |

## Root-cause weaknesses (worst first)
1. **Label contamination**: cells 47-48 wrote the same real Unsplash URL 1000x and
   downloaded it INTO the `ai/` folder via img2dataset. Partial cleanup at cell 70.
2. **"Real" class = dataset fingerprints**: CIFAR-10 (32x32 upscaled), beans leaf
   dataset, Unsplash, Bing, OpenImages, DDG. Model can learn "low-res/blurry = real".
3. **"AI" class = Stable Diffusion v1.5 only** (+ augmented copies of itself).
   No generator diversity => only detects SD artifacts.
4. **Data leakage**: augmented copies made before splitting; "extra" data copied
   only into train/, never val/test.
5. **No real evaluation**: no confusion matrix / precision / recall. 40+ manual
   single-image upload cells compensating for meaningless accuracy.

## VS Code app vs. notebook — diverged
- `app.py` loads `improved` (an OLD model). `fixed`/`final` were never deployed.
- **Preprocessing mismatch (critical)**: `improved` trained WITHOUT normalization
  (cell 179), but `app.py` APPLIES ImageNet normalization at inference. Train/infer
  input distributions don't match => degraded predictions.
- `app.py` hardcodes class list, no temperature scaling; `gradcam.py` is empty
  though notebook has full Grad-CAM (cells 196-206).

## Correction to project summary
`raw_data/` was created at a relative path inside `/content` (ephemeral Colab
storage) — it does NOT survive a runtime restart. Durable data is only Google
Drive `ai_image_dataset/ai` and `ai_image_dataset/real`.

## Fix principles for v2
- Define resize+normalize ONCE; import into both training and inference.
- Multi-generator AI data (GenImage etc.), not SD-only.
- Real and AI drawn from comparable resolution/quality distributions.
- Split by source; no leakage; augmentation only after split.
- Proper eval: confusion matrix, precision/recall, per-generator breakdown.
