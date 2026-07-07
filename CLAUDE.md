# AI Image Detector — project context & plan

Binary image classifier: **AI-generated vs. real photograph**. This is a rebuild
(v2). v1 failed to generalize; see `reports/diagnosis.md` for the full autopsy.

## CURRENT STATUS (update as we go)
- **Phase 1 COMPLETE.** Dataset extracted to `data/raw/`:
  - 18,514 images = **9,257 real / 9,257 AI**, balanced, fully paired scenes.
  - ~1,850 images per generator (sd21/sdxl/sd3/dalle3/mj6), even.
  - All images verified with full pixel decode: **0 broken / 0 truncated**.
  - `data/raw/manifest.csv` columns: `scene_id, label_a, generator, path`.
    (real + its AI twin share a `scene_id` — the key to leakage-safe splitting.)
  - Sample grids in `reports/` (sample_grid.png, late_scenes_grid.png).
- **NEXT: Phase 2** — leakage-safe train/val/test split by `scene_id`, hold out
  Midjourney (mj6) entirely for the generalization test, write `src/preprocessing.py`
  (one shared resize+normalize used by BOTH training and inference).

## Working style (important)
- Explain what's happening and *why* at each step; the owner is learning the
  pipeline, not just shipping a model. Don't silently do large or irreversible
  things — narrate, and confirm before big downloads / major design changes.
- Treat any old ChatGPT project summaries as weak hints; verify against real code/data.

## Golden rules (these are the fixes for v1's bugs)
1. **One preprocessing contract.** Resize + normalize is defined ONCE in `src/`
   and imported by BOTH training and inference. v1 broke because the deployed
   model was trained without normalization but the app applied it.
2. **No data leakage.** Dedup; split by source; augment ONLY after splitting.
3. **Hold out a generator.** Train on a subset of generators, keep one entirely
   unseen for a generalization test. Measures true "AI-ness" detection vs.
   memorizing one model's fingerprint (v1's fatal flaw = SD-1.5-only).
4. **Real evaluation.** Confusion matrix, precision/recall/F1, per-generator
   accuracy, held-out-generator score — not single-image spot checks.

## Dataset
- **Defactify (MS-COCO-AI)**: `Rajarshi-Roy-research/Defactify_Image_Dataset` on HF.
  96k images, 7.51 GB, not gated. 5 generators (SD2.1, SDXL, SD3, DALL-E 3,
  Midjourney v6) + COCO reals. Has per-generator labels (`Label_B`) — enables
  held-out-generator testing.
- **Plan:** download, extract a **balanced ~25-30k subset** to `data/raw/`,
  clear the HF cache. Generator diversity matters more than raw count.

## Compute strategy: local-first, Colab as escalation
- Machine: **Apple M1 (8-core, 16 GB)**. MPS works. Benchmark: ~78 img/s training
  ResNet18 => ~1h for a 12-epoch run on 25-30k images. Workable.
- v1's compute pain was Stable Diffusion *image generation* (very GPU-heavy);
  we're not doing that. Training a ResNet classifier is far lighter.
- **Baseline (Phase 3): train locally on M1.** Fast iteration, no Drive/Colab friction.
- **Escalate to Colab (write a clean notebook) only if** we move to a heavier
  backbone (ResNet50/ConvNeXt/ViT), much larger data, or need many fast runs.

## Phased plan
- **Phase 1 — Data:** download Defactify, extract balanced ~25-30k subset,
  inventory + visualize + sanity-check labels.
- **Phase 2 — Pipeline:** clean train/val/test split (no leakage, hold out one
  generator); single shared preprocessing module.
- **Phase 3 — Baseline:** fine-tune ResNet18 on M1/MPS; real training loop
  (class balance, val-based early stopping, checkpointing, logged metrics).
- **Phase 4 — Evaluate:** confusion matrix, P/R/F1, per-generator + held-out-
  generator accuracy.
- **Phase 5 — Harden:** real-world augmentation (JPEG recompression, resize),
  possibly stronger backbone (-> Colab if needed), confidence calibration.
- **Phase 6 — Deploy:** clean inference app (correct normalization contract),
  Grad-CAM heatmaps, packaged to run reliably on the Mac.

## Environment
- `venv/` at project root. torch 2.12.1 + torchvision 0.27.1 (MPS). `requirements.txt` pinned.
- Data/models are gitignored (large).
