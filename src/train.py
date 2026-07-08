"""
Phase 3 — fine-tune ResNet18 (ImageNet-pretrained) as the AI-vs-real baseline.

Runs on Apple M1 / MPS. Consumes data/splits.csv via src.dataset and the ONE
preprocessing contract via src.preprocessing (never redefines preprocessing).

What this does right (the fixes for v1):
  * label meaning is pinned (0=real, 1=ai) in src.dataset, not left to chance.
  * val-based early stopping + best-checkpoint saving (no eyeballing).
  * every checkpoint embeds the preprocessing contract (image size, ImageNet
    mean/std, class names) so inference physically cannot load weights with the
    wrong preprocessing — the exact mismatch that broke v1.
  * metrics logged per epoch to reports/train_history.csv.

Usage:
  python src/train.py                 # full run (defaults below)
  python src/train.py --smoke         # tiny fast pass to prove the loop works
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ImageSplitDataset, CLASS_NAMES
from src.model import build_model, get_device, ARCHS
from src.preprocessing import IMAGE_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD

CKPT_DIR = Path("models")
REPORTS = Path("reports")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy) over a loader. No grad, eval mode."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stop after this many epochs w/o val-loss improvement")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", type=str, default="resnet18", choices=ARCHS,
                    help="backbone; resnet50/convnext_tiny for the Colab GPU run")
    ap.add_argument("--tag", type=str, default="baseline",
                    help="names the checkpoint (models/<arch>_<tag>.pth) and "
                         "history (reports/train_history_<arch>_<tag>.csv)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run: few batches/epoch, 2 epochs, small workers")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="cap batches per epoch (0 = no cap); set by --smoke")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.max_batches = 8
        args.num_workers = 2

    torch.manual_seed(args.seed)
    device = get_device()
    CKPT_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    history_csv = REPORTS / f"train_history_{args.arch}_{args.tag}.csv"
    best_path = CKPT_DIR / f"{args.arch}_{args.tag}.pth"

    # --- data ----------------------------------------------------------------
    train_ds = ImageSplitDataset("train")          # augmenting transform
    val_ds = ImageSplitDataset("val")              # deterministic eval transform
    # class balance sanity: dataset is 50/50 by construction, so plain CE loss.
    tr_labels = train_ds.labels
    n_real = int((tr_labels == 0).sum()); n_ai = int((tr_labels == 1).sum())
    print(f"device={device} | train={len(train_ds)} (real {n_real} / ai {n_ai}) "
          f"| val={len(val_ds)}")

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=False)
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    # --- model / optim -------------------------------------------------------
    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # --- logging setup -------------------------------------------------------
    with open(history_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
             "lr", "secs", "saved_best"])

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # --- train loop ----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for bi, (x, y) in enumerate(train_dl):
            if args.max_batches and bi >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if bi % 20 == 0:
                print(f"  e{epoch} b{bi}/{len(train_dl)} loss={loss.item():.4f}",
                      flush=True)
        train_loss, train_acc = loss_sum / total, correct / total

        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        secs = time.time() - t0

        # checkpoint on val-loss improvement
        saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "class_names": CLASS_NAMES,          # [real, ai]
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                # embed the preprocessing contract so inference can't drift:
                "preprocessing": {
                    "image_size": IMAGE_SIZE,
                    "resize_size": RESIZE_SIZE,
                    "imagenet_mean": IMAGENET_MEAN,
                    "imagenet_std": IMAGENET_STD,
                },
            }, best_path)
            saved = True
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {secs:.0f}s"
              f"{'  <- saved best' if saved else ''}", flush=True)
        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                 f"{val_loss:.4f}", f"{val_acc:.4f}", args.lr,
                 f"{secs:.0f}", int(saved)])

        if epochs_no_improve >= args.patience:
            print(f"early stopping: no val-loss improvement for {args.patience} epochs")
            break

    print(f"\nbest val_loss={best_val_loss:.4f} -> {best_path}")
    print(json.dumps({"best_val_loss": best_val_loss,
                      "best_checkpoint": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
