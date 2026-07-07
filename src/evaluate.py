"""
Phase 4 — real evaluation of the baseline (golden rule #4).

Loads models/resnet18_baseline.pth and reports, for both the in-distribution
TEST split and the held-out MIDJOURNEY split:
  * confusion matrix (real/ai)
  * accuracy, precision, recall, F1
  * per-generator AI detection rate (recall on AI images, by generator)
  * the headline number: held-out-generator (mj6) detection rate — did the model
    learn "AI-ness" or just memorize the 4 training generators? (v1's fatal flaw)

Uses the SAME preprocessing contract as training by loading get_eval_transform
via src.dataset, and it VERIFIES the checkpoint's embedded contract matches the
code — so we can never evaluate (or later deploy) with drifted preprocessing.

Outputs: reports/eval_metrics.json + reports/confusion_{test,gen_holdout}.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

from src.dataset import ImageSplitDataset, CLASS_NAMES
from src.preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

CKPT = Path("models/resnet18_baseline.pth")
REPORTS = Path("reports")


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    # verify the embedded preprocessing contract matches the code (anti-v1 guard)
    pp = ck["preprocessing"]
    assert pp["image_size"] == IMAGE_SIZE, "image_size mismatch train vs eval!"
    assert tuple(pp["imagenet_mean"]) == IMAGENET_MEAN, "normalize mean mismatch!"
    assert tuple(pp["imagenet_std"]) == IMAGENET_STD, "normalize std mismatch!"
    assert ck["class_names"] == CLASS_NAMES, "class order mismatch!"
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    print(f"loaded {CKPT} (epoch {ck['epoch']}, val_acc {ck['val_acc']:.4f}); "
          f"preprocessing contract verified.")
    return m


@torch.no_grad()
def predict_split(model, split, device):
    """Return a DataFrame with true label, predicted label, ai-prob, generator."""
    ds = ImageSplitDataset(split, train=False)  # deterministic eval transform
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    preds, probs = [], []
    for x, _ in dl:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1)[:, 1]  # P(ai), since idx 1 = ai
        probs.append(p.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    out = ds.rows.copy()
    out["y_true"] = out.label_a.astype(int)
    out["y_pred"] = np.concatenate(preds)
    out["p_ai"] = np.concatenate(probs)
    return out


def binary_metrics(y_true, y_pred):
    """Accuracy + precision/recall/F1 with AI (=1) as the positive class."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1,
        zero_division=0)
    return {"accuracy": round(float(acc), 4), "precision_ai": round(float(p), 4),
            "recall_ai": round(float(r), 4), "f1_ai": round(float(f1), 4)}


def save_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return cm


def per_generator_ai_recall(df):
    """AI detection rate (recall) per generator, on AI rows only."""
    ai = df[df.y_true == 1]
    out = {}
    for gen, g in ai.groupby("generator"):
        out[gen] = {"n": int(len(g)),
                    "ai_recall": round(float((g.y_pred == 1).mean()), 4)}
    return out


def main():
    device = get_device()
    model = load_model(device)
    results = {}

    for split, label in [("test", "In-distribution TEST (sd21/sdxl/sd3/dalle3)"),
                         ("gen_holdout", "Held-out MIDJOURNEY (mj6, UNSEEN)")]:
        df = predict_split(model, split, device)
        m = binary_metrics(df.y_true, df.y_pred)
        cm = save_confusion(df.y_true, df.y_pred, label,
                            REPORTS / f"confusion_{split}.png")
        gen_recall = per_generator_ai_recall(df)
        # real-class recall (specificity): fraction of reals correctly called real
        reals = df[df.y_true == 0]
        real_recall = round(float((reals.y_pred == 0).mean()), 4)
        results[split] = {
            "n": int(len(df)), **m,
            "real_recall": real_recall,
            "confusion_matrix": {"rows_true": CLASS_NAMES,
                                 "cols_pred": CLASS_NAMES, "counts": cm.tolist()},
            "per_generator_ai_recall": gen_recall,
        }
        print(f"\n=== {label} ===")
        print(f"  n={len(df)}  acc={m['accuracy']:.4f}  "
              f"P(ai)={m['precision_ai']:.4f}  R(ai)={m['recall_ai']:.4f}  "
              f"F1(ai)={m['f1_ai']:.4f}  real_recall={real_recall:.4f}")
        print(f"  confusion [rows=true {CLASS_NAMES}, cols=pred {CLASS_NAMES}]:")
        print("   ", cm.tolist())
        print("  per-generator AI detection rate:")
        for gen, s in sorted(gen_recall.items()):
            print(f"    {gen:8s} n={s['n']:4d}  recall={s['ai_recall']:.4f}")

    with open(REPORTS / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {REPORTS/'eval_metrics.json'} and confusion_*.png")

    # headline generalization delta
    mj6 = results["gen_holdout"]["per_generator_ai_recall"].get("mj6", {})
    test_ai_r = results["test"]["recall_ai"]
    print(f"\nHEADLINE: in-distribution AI recall={test_ai_r:.4f}  vs  "
          f"held-out mj6 AI recall={mj6.get('ai_recall', float('nan')):.4f}")


if __name__ == "__main__":
    main()
