"""
Phase 5, Step 1 — diagnose the Midjourney misses WITHOUT retraining.

Two questions:
  1. Calibration or capability? For the mj6 AI images, where does the model's
     p_ai (its confidence the image is AI) actually sit? If lots of the missed
     ones cluster just under 0.5, a threshold change recovers them cheaply. If
     they cluster near 0, the model is confidently wrong -> we must retrain.
  2. What does moving the decision threshold cost? Sweep thresholds and report,
     for both the in-distribution TEST set and the held-out mj6 set:
        - ai recall  (fraction of AI caught)
        - ai precision (when it says ai, how often right)
        - real recall (fraction of reals kept correct)
     so we can see the tradeoff curve before picking an operating point.

Reuses the model loader + per-image predictions from src.evaluate (same
preprocessing contract). Outputs a printed report + reports/threshold_sweep.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import get_device, load_model, predict_split

REPORTS = Path("reports")


def band_counts(p, edges):
    """Count how many probabilities fall in each [edge_i, edge_{i+1}) band."""
    return [int(((p >= edges[i]) & (p < edges[i + 1])).sum())
            for i in range(len(edges) - 1)]


def sweep(df, thresholds):
    """For each threshold, compute ai-recall / ai-precision / real-recall."""
    ai = df.y_true == 1
    real = df.y_true == 0
    rows = []
    for t in thresholds:
        pred_ai = df.p_ai >= t
        tp = int((pred_ai & ai).sum())
        fp = int((pred_ai & real).sum())
        fn = int((~pred_ai & ai).sum())
        tn = int((~pred_ai & real).sum())
        ai_recall = tp / (tp + fn) if (tp + fn) else 0.0
        ai_prec = tp / (tp + fp) if (tp + fp) else 0.0
        real_recall = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append((t, ai_recall, ai_prec, real_recall))
    return rows


def main():
    import argparse
    from pathlib import Path as _Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/resnet18_baseline.pth",
                    help="checkpoint to analyze (e.g. models/resnet50_colab.pth)")
    args = ap.parse_args()

    device = get_device()
    model = load_model(device, _Path(args.ckpt))
    test = predict_split(model, "test", device)
    mj6 = predict_split(model, "gen_holdout", device)

    # --- Q1: where do mj6 AI confidences sit? --------------------------------
    mj6_ai = mj6[mj6.y_true == 1]
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
              "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    counts = band_counts(mj6_ai.p_ai.values, edges)
    n = len(mj6_ai)
    missed = mj6_ai[mj6_ai.p_ai < 0.5]
    print(f"\nmj6 AI images: {n}.  Missed at threshold 0.5 (p_ai<0.5): "
          f"{len(missed)} ({len(missed)/n:.1%})")
    print("p_ai distribution of ALL mj6 AI images (how confident model is it's AI):")
    for lab, c in zip(labels, counts):
        bar = "#" * round(40 * c / max(counts))
        flag = "  <- counted as REAL (miss)" if lab < "0.5" else ""
        print(f"  {lab}: {c:5d} {bar}{flag}")
    # how salvageable are the misses? how many sit in 0.3-0.5 vs near 0?
    near0 = int((missed.p_ai < 0.2).sum())
    borderline = int((missed.p_ai >= 0.3).sum())
    print(f"\n  of the {len(missed)} misses: {near0} are confident-wrong (p_ai<0.2), "
          f"{borderline} are borderline (p_ai>=0.3, cheaply recoverable by threshold)")

    # --- Q2: threshold sweep on both sets ------------------------------------
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    print("\nthreshold sweep  [ai_recall / ai_precision / real_recall]")
    print(f"{'thr':>5} | {'TEST (seen)':^28} | {'mj6 (held-out)':^28}")
    print(f"{'':>5} | {'ai_rec  ai_prec  real_rec':^28} | {'ai_rec  ai_prec  real_rec':^28}")
    st = {t: (r1, r2, r3) for t, r1, r2, r3 in sweep(test, thresholds)}
    sm = {t: (r1, r2, r3) for t, r1, r2, r3 in sweep(mj6, thresholds)}
    for t in thresholds:
        a = st[t]; b = sm[t]
        mark = "  <- default" if abs(t - 0.5) < 1e-6 else ""
        print(f"{t:>5} | {a[0]:6.3f}  {a[1]:6.3f}   {a[2]:6.3f}    "
              f"| {b[0]:6.3f}  {b[1]:6.3f}   {b[2]:6.3f}{mark}")

    # --- plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, name in [(axes[0], st, "TEST (seen generators)"),
                           (axes[1], sm, "mj6 (held-out generator)")]:
        ts = thresholds
        ax.plot(ts, [data[t][0] for t in ts], label="ai recall", marker="o", ms=3)
        ax.plot(ts, [data[t][1] for t in ts], label="ai precision", marker="s", ms=3)
        ax.plot(ts, [data[t][2] for t in ts], label="real recall", marker="^", ms=3)
        ax.axvline(0.5, ls="--", c="gray", lw=1)
        ax.set_title(name); ax.set_xlabel("decision threshold  p_ai >= t")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    out_png = REPORTS / f"threshold_sweep_{_Path(args.ckpt).stem}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
