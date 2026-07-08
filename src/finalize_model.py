"""
Stamp the chosen decision threshold into a production checkpoint.

The threshold is a real deployment decision (see reports/threshold_sweep_*.png and
the Phase 5 calibration). Like the preprocessing contract, it must TRAVEL WITH THE
MODEL so the inference app reads it from the artifact instead of hardcoding a guess.

This reads a trained checkpoint, adds `decision_threshold`, and writes a
`*_production.pth` — the single blessed artifact the Phase 6 app loads.

Usage:
  python src/finalize_model.py --ckpt models/resnet50_colab.pth --threshold 0.35
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained checkpoint to bless")
    ap.add_argument("--threshold", type=float, required=True,
                    help="decision threshold on P(ai); pred=ai iff p_ai >= threshold")
    ap.add_argument("--out", default=None,
                    help="output path (default: <ckpt stem>_production.pth)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt.with_name(f"{ckpt.stem}_production.pth")
    assert 0.0 < args.threshold < 1.0, "threshold must be in (0, 1)"

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck["decision_threshold"] = args.threshold
    torch.save(ck, out)

    print(f"blessed {ckpt.name} -> {out.name}")
    print(f"  arch={ck.get('arch')}  class_names={ck.get('class_names')}")
    print(f"  decision_threshold={ck['decision_threshold']}  "
          f"(pred=ai iff p_ai >= {ck['decision_threshold']})")
    print(f"  preprocessing={ck.get('preprocessing')}")


if __name__ == "__main__":
    main()
