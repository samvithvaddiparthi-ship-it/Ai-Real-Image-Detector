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
- **Phase 5 IN PROGRESS — Harden.** Target: raise mj6 recall without wrecking precision.
  - Step 1 (diagnosis, `src/threshold_analysis.py`): mj6 AI p_ai is BIMODAL —
    ~910 confidently AI (p_ai>0.8), but ~538 confidently "real" (p_ai<0.1). Of the
    787 misses, 617 are confident-wrong -> NOT a threshold problem; it's capability.
    Threshold sweep saved to reports/threshold_sweep.png. Verdict: retrain needed.
  - Step 2 (augmentation, in `src/preprocessing.get_train_transform`): added
    RandomJPEG + RandomDownscale + GaussianBlur + ColorJitter (train-only; eval
    contract UNCHANGED). Retrained resnet18 -> models/resnet18_augmented.pth.
    **mj6 AI recall 0.575 -> 0.626 (+93 caught)**, F1 0.710->0.744; small cost on
    seen test (acc 0.954->0.930). Real, modest gain — confirms the confident-wrong
    core is hard. Metrics: reports/eval_metrics_resnet18_augmented.json.
  - Step 3 (stronger backbone -> Colab GPU, CURRENT): refactored to be GPU +
    multi-backbone. `src/model.py` = shared build_model(arch)/get_device (cuda>mps>cpu),
    archs: resnet18/resnet50/convnext_tiny. train.py `--arch`; checkpoint records
    arch; evaluate.py `--ckpt` rebuilds from recorded arch. Checkpoints/history now
    named `<arch>_<tag>`. Colab notebook: notebooks/colab_train_stronger_backbone.ipynb
    (clones repo, mounts Drive, untars data_bundle.tar, trains convnext_tiny+resnet50,
    evals mj6, saves to Drive). Data moved via data_bundle.tar (~1.4GB, gitignored).
    AWAITING Colab results to pick the winner backbone.
- **THEN: threshold calibration** to the chosen operating point, then Phase 6 (deploy:
  inference app honoring the preprocessing contract + Grad-CAM).

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
