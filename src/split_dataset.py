"""
Leakage-safe train/val/test split for the AI-vs-real detector (Phase 2).

Design (leakage-safety + held-out-generator principles):

  * Split by scene_id, never by row. Each scene_id owns exactly one real photo
    and its one AI twin. Splitting by scene guarantees a real photo (and its
    twin) live in exactly ONE split -> no image leaks across train/val/test.

  * Hold out Midjourney (mj6) ENTIRELY as `gen_holdout`. The model never trains
    or validates on any mj6 image. This is the true generalization test: can it
    detect "AI-ness" from a generator it has never seen? (v1's fatal flaw was
    training on SD-1.5 only and memorizing that one model's fingerprint.)

  * The remaining 4 generators (sd21, sdxl, sd3, dalle3) are split 80/10/10 into
    train/val/test, stratified by generator so every split has the same generator
    mix. Because each scene is 1 real + 1 AI, every split is automatically
    class-balanced.

  * Augmentation is NOT applied here. It is train-only and happens at load time
    (get_train_transform), i.e. strictly after this split.

Output: data/splits.csv = manifest.csv + a `split` column
        {train, val, test, gen_holdout}. Deterministic (fixed seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MANIFEST = "data/raw/manifest.csv"
OUT = "data/splits.csv"
HOLDOUT_GENERATOR = "mj6"
VAL_FRAC = 0.10
TEST_FRAC = 0.10
SEED = 42


def main() -> None:
    df = pd.read_csv(MANIFEST)
    rng = np.random.default_rng(SEED)

    # --- Map each scene -> the generator of its AI twin -----------------------
    # (real rows have generator="real"; the twin's generator is what defines the
    #  scene for held-out / stratification purposes.)
    ai = df[df.label_a == 1]
    scene_gen = dict(zip(ai.scene_id, ai.generator))

    scene_split: dict[int, str] = {}

    for gen in sorted(set(scene_gen.values())):
        scenes = np.array([s for s, g in scene_gen.items() if g == gen])
        rng.shuffle(scenes)

        if gen == HOLDOUT_GENERATOR:
            # entire generator held out -> generalization test set
            for s in scenes:
                scene_split[s] = "gen_holdout"
            continue

        # 80/10/10 split of THIS generator's scenes (stratified by generator)
        n = len(scenes)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))
        test_s = scenes[:n_test]
        val_s = scenes[n_test:n_test + n_val]
        train_s = scenes[n_test + n_val:]
        for s in train_s:
            scene_split[s] = "train"
        for s in val_s:
            scene_split[s] = "val"
        for s in test_s:
            scene_split[s] = "test"

    df["split"] = df.scene_id.map(scene_split)
    assert df.split.notna().all(), "some scenes were not assigned a split"

    df.to_csv(OUT, index=False)

    # --- Report so the human can eyeball it -----------------------------------
    print(f"wrote {OUT}  ({len(df)} rows, {df.scene_id.nunique()} scenes)\n")

    print("rows per split (real / ai):")
    tab = df.pivot_table(index="split", columns="label_a", values="scene_id",
                         aggfunc="count", fill_value=0)
    tab.columns = ["real", "ai"]
    tab["total"] = tab.real + tab.ai
    print(tab.reindex(["train", "val", "test", "gen_holdout"]).to_string(), "\n")

    print("AI generator mix per split (should be even across train/val/test,\n"
          "and mj6 should appear ONLY in gen_holdout):")
    gmix = (df[df.label_a == 1]
            .pivot_table(index="split", columns="generator", values="scene_id",
                         aggfunc="count", fill_value=0))
    print(gmix.reindex(["train", "val", "test", "gen_holdout"]).to_string(), "\n")

    # --- Leakage guard: no scene_id may appear in more than one split ---------
    per_scene_splits = df.groupby("scene_id").split.nunique()
    assert (per_scene_splits == 1).all(), "LEAKAGE: a scene_id spans >1 split"
    # mj6 must be exclusively in gen_holdout
    mj6_splits = set(df[df.generator == HOLDOUT_GENERATOR].split.unique())
    assert mj6_splits == {"gen_holdout"}, f"mj6 leaked into {mj6_splits}"
    print("leakage checks passed: every scene in exactly one split; "
          "mj6 confined to gen_holdout.")


if __name__ == "__main__":
    main()
