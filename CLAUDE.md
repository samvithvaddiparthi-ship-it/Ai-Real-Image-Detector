# AI Image Detector — project context & plan

Binary image classifier: **AI-generated vs. real photograph**. This is a rebuild
(v2). v1 failed to generalize; see `reports/diagnosis.md` for the full autopsy.

## CURRENT STATUS (update as we go)
- **Phase 1 COMPLETE.** Dataset extracted to `data/raw/`:
  - 18,514 images = **9,257 real / 9,257 AI**, balanced, fully paired scenes.
  - ~1,850 images per generator (sd21/sdxl/sd3/dalle3/mj6), even.
  - All images verified with full pixel decode: **0 broken / 0 truncated**.
  - Verified 0 exact-duplicate real files (v1's contamination bug absent).
  - `data/raw/manifest.csv` columns: `scene_id, label_a, generator, path`.
    (real + its AI twin share a `scene_id` — the key to leakage-safe splitting.)
  - Sample grids in `reports/` (sample_grid.png, late_scenes_grid.png).
- **Phase 2 COMPLETE.**
  - `src/preprocessing.py` — THE shared contract (golden rule #1). One
    resize+normalize (ImageNet stats, 224). `get_eval_transform()` is the
    inference contract used by val/test/held-out AND the deployed app;
    `get_train_transform()` = same normalize + light aug (train-only).
  - `src/split_dataset.py` -> `data/splits.csv` (manifest + `split` column).
    Leakage-safe split by `scene_id`; mj6 held out ENTIRELY as `gen_holdout`;
    other 4 generators split 80/10/10 stratified by generator.
    train 11,848 / val 1,480 / test 1,480 / gen_holdout 3,706 (all balanced).
    Runtime asserts confirm: every scene in exactly one split; mj6 only in
    gen_holdout. Deterministic (SEED=42).
  - KNOWN RISK for Phase 5: reals are varied aspect ratios (COCO, min side ~182);
    AI images are often square (512²/270²). Possible "square=AI" shortcut.
- **Phase 3 COMPLETE.** `src/train.py` + `src/dataset.py`.
  - ResNet18 (ImageNet-pretrained), full fine-tune, AdamW lr=1e-4 wd=1e-4,
    CrossEntropyLoss (data already 50/50 -> no class weights). MPS, batch 64.
  - Val-based early stopping (patience 3). Best = epoch 6: val_loss 0.141,
    **val_acc 0.955**. Early-stopped at epoch 9. History: reports/train_history.csv.
  - Label convention PINNED in src/dataset.py: 0=real, 1=ai.
  - Checkpoint models/resnet18_baseline.pth EMBEDS the preprocessing contract
    (image_size, imagenet mean/std, class_names); eval/inference assert-verify it.
  - SSL note: pretrained-weights download needs `SSL_CERT_FILE=$(python -c "import
    certifi;print(certifi.where())")` (macOS system-Python cert issue).
- **Phase 4 COMPLETE.** `src/evaluate.py --ckpt <path>` -> reports/eval_metrics_<stem>.json
  + confusion_<stem>_*.png (stem = e.g. resnet18_baseline).
  - In-distribution TEST (sd21/sdxl/sd3/dalle3): **acc 0.954, F1(ai) 0.954**,
    P(ai) 0.947, R(ai) 0.962, real_recall 0.946. Per-gen AI recall: sdxl 0.984,
    dalle3 0.979, sd21 0.946, sd3 0.940. Strong + even across seen generators.
  - Held-out MIDJOURNEY (mj6, UNSEEN): **acc 0.765, AI recall 0.575**, but
    P(ai) 0.928 and real_recall 0.955. => model is CONSERVATIVE on mj6: misses
    ~42% of mj6 (false negatives), rarely false-positives. This is the real
    generalization gap v1 hid. Failure mode = mj6 photorealism not recognized.
  - HEADLINE: in-dist AI recall 0.962 vs held-out mj6 AI recall 0.575.
- **Phase 5 COMPLETE — Harden.** Raised mj6 recall 0.575 -> 0.661 (backbone) then
  operating point set for deployment.
  - Step 1 (diagnosis, `src/threshold_analysis.py --ckpt`): mj6 p_ai BIMODAL; most
    misses confident-wrong -> capability gap, not calibration. Verdict: retrain.
  - Step 2 (augmentation, `preprocessing.get_train_transform`): RandomJPEG +
    RandomDownscale + GaussianBlur + ColorJitter (train-only; eval UNCHANGED).
    resnet18 mj6 recall 0.575 -> 0.626.
  - Step 3 (stronger backbone on Colab GPU): `src/model.py` shared
    build_model(arch)/get_device (cuda>mps>cpu). Notebook
    notebooks/colab_train_stronger_backbone.ipynb trained resnet50 + convnext_tiny.
    **WINNER = resnet50** (models/resnet50_colab.pth): mj6 recall **0.661**,
    precision 0.942, test_acc 0.960 — strictly better than resnet18_augmented on
    all 3. convnext_tiny: test_acc 0.977, mj6 prec 0.970 but recall only 0.623
    (too conservative). Metrics: reports/eval_metrics_resnet50_colab.json.
  - Step 4 (calibration): swept resnet50 threshold
    (reports/threshold_sweep_resnet50_colab.png). CHOSE **threshold 0.35** ->
    mj6 recall 0.715, precision 0.926, 5.7% reals false-flagged. Stamped into
    **models/resnet50_colab_production.pth** via `src/finalize_model.py`
    (checkpoint now carries weights + arch + preprocessing contract + class_names +
    decision_threshold=0.35 — the single blessed deployment artifact).
- **Phase 6 COMPLETE — Deploy.** Engine/UI cleanly separated (UI has ZERO ML logic).
  - Engine (pure Python): `src/inference.py` (Detector: loads production ckpt, reads
    embedded arch + preprocessing contract + class order + 0.35 threshold, predicts),
    `src/gradcam.py` (Grad-CAM on last conv block; overlay). Both reused by CLI + web.
  - `src/cli.py`: `python src/cli.py img.jpg [--heatmap out.png]`.
  - Web: `app/server.py` (FastAPI, THIN — marshals HTTP<->engine only; /api/predict,
    /api/health) + `app/static/` custom frontend (index.html/styles.css/app.js),
    premium restrained design (NOT gradio — gradio can't hit the design bar). Run:
    `python -m uvicorn app.server:app --port 8000`. Launch cfg: .claude/launch.json.
  - Verified live: mj6 AI image -> "AI-generated" 98.8%; COCO real -> "Real" 0.0%.
    Grad-CAM overlays render; tech panel shows model/threshold/prob/confidence/time.
  - New deps (requirements.txt): fastapi, uvicorn[standard], python-multipart.
  - NOTE: production .pth gitignored; lives in models/ locally + Drive
    (ai_detector/results/). Regenerate: download resnet50_colab.pth, run finalize_model.py.
- **Calibration (post-Phase 6).** `src/calibrate.py`: temperature scaling fit on val.
  Result: **T=1.059** — model was ALREADY well-calibrated (ECE ~0.006 val / 0.013 test),
  so this is nearly a no-op (confirms confidence is trustworthy, not a bug). Threshold
  remapped 0.35 -> 0.358 (calibrated space, same verdicts). T + calibrated threshold
  embedded in production ckpt; inference.py applies logits/T. UI also formats extreme
  probs honestly (">99.9%" / "<0.1%" instead of "100.0%"/"0.0%") and shows T.
- **ALL 6 PHASES COMPLETE.** Final: resnet50, calibrated threshold 0.358 (raw 0.35),
  T=1.059, mj6 recall 0.715 / precision 0.926. Possible future work: more generator
  diversity, OOD calibration, batch/API endpoints, packaging.

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
