"""
Extract a balanced, generator-diverse subset of the Defactify dataset to disk.

Source: Rajarshi-Roy-research/Defactify_Image_Dataset (Hugging Face), streamed
(no full download). It is caption-major: every caption = 6 consecutive rows:
    Label_B: 0 real, 1 SD2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 Midjourney 6

Strategy
--------
For each caption we keep the real image + exactly ONE AI image, round-robining
the AI generator across captions. This gives:
  * 50/50 real-vs-AI balance
  * even coverage of all 5 generators
  * one distinct scene per caption (high scene diversity)

We record caption_id / generator / origin split in manifest.csv so Phase 2 can
build a leakage-safe split (whole scene stays in one split) and hold out a
generator for the generalization test.

Disk/quality notes (revisit in Phase 5):
  * longest side capped at MAX_SIDE (training resizes to 224 anyway)
  * saved uniformly as JPEG q95 to avoid a format-based shortcut
"""

import csv
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image

REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
GEN_NAMES = {0: "real", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "mj6"}
AI_GENERATORS = [1, 2, 3, 4, 5]  # round-robin over these

TARGET_CAPTIONS = 12_000  # ~24k images (real + 1 AI each)
MAX_SIDE = 512
JPEG_Q = 95
SPLITS = ["train", "validation", "test"]  # HF split names

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.csv"


def save_image(img: Image.Image, path: Path) -> None:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    img.save(path, "JPEG", quality=JPEG_Q)


def main() -> None:
    (RAW / "real").mkdir(parents=True, exist_ok=True)
    (RAW / "ai").mkdir(parents=True, exist_ok=True)

    # --- resume support: skip splits already fully extracted ---
    done_splits = set()
    max_cid = -1
    for f in (RAW / "real").glob("*.jpg"):
        split_name, cid = f.stem.split("_")
        done_splits.add(split_name)
        max_cid = max(max_cid, int(cid))
    resuming = max_cid >= 0
    if resuming:
        print(f"Resuming: {len(done_splits)} split(s) done {sorted(done_splits)}, "
              f"last caption_id={max_cid}", flush=True)

    manifest = open(MANIFEST, "a" if resuming else "w", newline="")
    writer = csv.writer(manifest)
    if not resuming:
        writer.writerow(["caption_id", "orig_split", "label_a", "generator", "path"])

    caption_id = max_cid            # next new caption increments to max_cid+1
    prev_caption = object()         # sentinel so first row starts a new caption
    chosen_gen = None
    per_gen = {g: 0 for g in AI_GENERATORS}
    n_real = n_ai = 0

    for split in SPLITS:
        if split in done_splits:    # already extracted in a previous run
            print(f"\n=== skipping already-done split: {split} ===", flush=True)
            continue
        if caption_id + 1 >= TARGET_CAPTIONS:
            break
        print(f"\n=== streaming split: {split} ===", flush=True)
        ds = load_dataset(REPO, split=split, streaming=True)
        for ex in ds:
            cap = ex["Caption"]
            lb = ex["Label_B"]

            if cap != prev_caption:            # new scene
                caption_id += 1
                prev_caption = cap
                if caption_id >= TARGET_CAPTIONS:
                    break
                chosen_gen = AI_GENERATORS[caption_id % len(AI_GENERATORS)]

            if lb == 0:                        # the real image
                p = RAW / "real" / f"{split}_{caption_id:06d}.jpg"
                save_image(ex["Image"], p)
                writer.writerow([caption_id, split, 0, "real", p.relative_to(ROOT)])
                n_real += 1
            elif lb == chosen_gen:             # the one AI image for this scene
                gen = GEN_NAMES[lb]
                p = RAW / "ai" / f"{split}_{caption_id:06d}_{gen}.jpg"
                save_image(ex["Image"], p)
                writer.writerow([caption_id, split, 1, gen, p.relative_to(ROOT)])
                per_gen[lb] += 1
                n_ai += 1

            if (n_real + n_ai) % 1000 == 0 and (n_real + n_ai) > 0:
                print(f"  saved {n_real+n_ai} imgs (real={n_real}, ai={n_ai})", flush=True)

    manifest.close()

    # recount from disk so totals include any resumed splits
    real_total = len(list((RAW / "real").glob("*.jpg")))
    ai_files = list((RAW / "ai").glob("*.jpg"))
    gen_total = {}
    for f in ai_files:
        gen = f.stem.split("_")[-1]
        gen_total[gen] = gen_total.get(gen, 0) + 1

    print("\n===== DONE =====")
    print(f"captions: {caption_id+1}   (added this run: real={n_real}, ai={n_ai})")
    print(f"TOTAL on disk -> real: {real_total}   ai: {len(ai_files)}")
    print("per-generator:", gen_total)
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
