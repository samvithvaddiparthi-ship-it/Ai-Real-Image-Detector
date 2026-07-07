"""
Extract a balanced, generator-diverse subset of the Defactify dataset to disk.

Source: Rajarshi-Roy-research/Defactify_Image_Dataset (Hugging Face), streamed.
Label_B: 0 real, 1 SD2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 Midjourney 6.
Every caption (scene) has one real + one image from each of the 5 generators.

Strategy
--------
We key each scene by its CAPTION TEXT (robust to row ordering, which differs
between train and val/test). For each unique caption we keep:
  * the real image
  * exactly ONE AI image, generator chosen by scene index (even round-robin)

=> 50/50 real/AI balance, even generator coverage, one distinct scene per caption.
Scenes are given a global integer id (scene_id); the real + its AI twin share it,
so Phase 2 can split by scene_id with zero leakage. If the same caption appears in
multiple HF splits it collapses to ONE scene_id (prevents cross-split leakage).

Only images we actually keep are decoded/saved (skipped rows are never decoded),
so a full stream stays reasonably fast.

Disk/quality notes (revisit in Phase 5): longest side capped at MAX_SIDE; saved
uniformly as JPEG q95 to avoid a format-based shortcut.
"""

import csv
from pathlib import Path

from datasets import load_dataset
from PIL import Image

REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
GEN_NAMES = {1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "mj6"}
N_GEN = 5

TARGET_SCENES = 15_000       # ~30k images (real + 1 AI each)
MAX_SIDE = 512
JPEG_Q = 95
SPLITS = ["train", "validation", "test"]

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.csv"


def save_image(img: Image.Image, path: Path) -> None:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    img.save(path, "JPEG", quality=JPEG_Q)


def main() -> None:
    (RAW / "real").mkdir(parents=True, exist_ok=True)
    (RAW / "ai").mkdir(parents=True, exist_ok=True)

    manifest = open(MANIFEST, "w", newline="")
    writer = csv.writer(manifest)
    writer.writerow(["scene_id", "orig_split", "label_a", "generator", "path"])

    scene_of = {}                 # caption text -> scene_id
    real_saved = set()            # scene_ids whose real is on disk
    ai_saved = set()              # scene_ids whose AI is on disk
    n_real = n_ai = 0

    for split in SPLITS:
        if len(scene_of) >= TARGET_SCENES and not (real_saved ^ ai_saved):
            break
        print(f"\n=== streaming split: {split} ===", flush=True)
        for ex in load_dataset(REPO, split=split, streaming=True):
            cap = ex["Caption"]
            lb = ex["Label_B"]

            # assign / look up scene id
            if cap in scene_of:
                sid = scene_of[cap]
            else:
                if len(scene_of) >= TARGET_SCENES:
                    continue      # don't open new scenes past target
                sid = len(scene_of)
                scene_of[cap] = sid
            chosen_gen = (sid % N_GEN) + 1     # which generator to keep for this scene

            if lb == 0 and sid not in real_saved:
                p = RAW / "real" / f"{sid:06d}.jpg"
                save_image(ex["Image"], p)
                writer.writerow([sid, split, 0, "real", p.relative_to(ROOT)])
                real_saved.add(sid); n_real += 1
            elif lb == chosen_gen and sid not in ai_saved:
                gen = GEN_NAMES[lb]
                p = RAW / "ai" / f"{sid:06d}_{gen}.jpg"
                save_image(ex["Image"], p)
                writer.writerow([sid, split, 1, gen, p.relative_to(ROOT)])
                ai_saved.add(sid); n_ai += 1

            if (n_real + n_ai) % 2000 == 0 and (n_real + n_ai) > 0:
                print(f"  scenes={len(scene_of)} real={n_real} ai={n_ai}", flush=True)

    manifest.close()

    # count generators actually saved
    gen_total = {}
    for f in (RAW / "ai").glob("*.jpg"):
        g = f.stem.split("_")[-1]
        gen_total[g] = gen_total.get(g, 0) + 1

    print("\n===== DONE =====")
    print(f"scenes seen: {len(scene_of)}")
    print(f"real: {n_real}   ai: {n_ai}")
    print("per-generator:", gen_total)
    print("complete scenes (real+ai both):", len(real_saved & ai_saved))
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
