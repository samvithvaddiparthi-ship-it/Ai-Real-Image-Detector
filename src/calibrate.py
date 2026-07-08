"""
Temperature-scaling calibration for honest confidence scores.

Neural-net softmax scores are systematically OVERCONFIDENT (a clear-cut image
reads "100%" when the model's real reliability is lower). Temperature scaling
fixes this with a single scalar T fit on the validation set: divide the logits
by T before softmax. Larger T -> softer probabilities. It is monotonic, so it
changes NO verdicts and leaves the recall/precision tradeoff identical — it only
relabels the probability axis so the numbers mean what they say.

Because the deployment threshold (0.35) was chosen on the RAW probabilities, we
remap it into calibrated space so the operating point is byte-for-byte preserved:
    calibrated_threshold = sigmoid(logit(raw_threshold) / T)

Output: overwrites the production checkpoint with `temperature` and the calibrated
`decision_threshold` embedded (raw kept as `raw_decision_threshold` for the record).

Usage:
  python src/calibrate.py --ckpt models/resnet50_colab.pth --raw-threshold 0.35
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import build_model, get_device
from src.dataset import ImageSplitDataset


def collect_logits(model, split, device):
    """Run the model over a split; return (logits, labels) on CPU."""
    ds = ImageSplitDataset(split, train=False)
    dl = DataLoader(ds, batch_size=64, num_workers=4)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels) -> float:
    """Fit T by minimizing NLL of softmax(logits / T) — the standard method."""
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=0.05))


def nll_and_ece(logits, labels, T=1.0, n_bins=15):
    """Negative log-likelihood and Expected Calibration Error at temperature T."""
    probs = F.softmax(logits / T, dim=1)
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels).float()
    nll = F.cross_entropy(logits / T, labels).item()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return nll, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/resnet50_colab.pth",
                    help="trained checkpoint to calibrate")
    ap.add_argument("--raw-threshold", type=float, default=0.35,
                    help="the operating point chosen on RAW probabilities")
    ap.add_argument("--out", default="models/resnet50_colab_production.pth")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck.get("arch", "resnet18"), pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()

    # fit T on validation logits (in-distribution, held out from training)
    val_logits, val_labels = collect_logits(model, "val", device)
    T = fit_temperature(val_logits, val_labels)

    # remap the chosen threshold into calibrated space (identical operating point)
    raw = args.raw_threshold
    cal_threshold = 1.0 / (1.0 + math.exp(-math.log(raw / (1 - raw)) / T))

    print(f"fitted temperature  T = {T:.3f}")
    print(f"{'split':<6} {'NLL (T=1 -> T)':<22} {'ECE (T=1 -> T)':<22}")
    for split in ["val", "test"]:
        lg, lb = (val_logits, val_labels) if split == "val" else collect_logits(model, split, device)
        n0, e0 = nll_and_ece(lg, lb, 1.0)
        n1, e1 = nll_and_ece(lg, lb, T)
        print(f"{split:<6} {n0:.4f} -> {n1:.4f}        {e0:.4f} -> {e1:.4f}")
    print(f"\ndecision threshold: raw {raw} (on raw p_ai)  ->  "
          f"{cal_threshold:.3f} (on calibrated p_ai)  [same verdicts]")

    ck["temperature"] = T
    ck["raw_decision_threshold"] = raw
    ck["decision_threshold"] = cal_threshold
    torch.save(ck, args.out)
    print(f"\nwrote {Path(args.out)} — temperature + calibrated threshold embedded.")


if __name__ == "__main__":
    main()
