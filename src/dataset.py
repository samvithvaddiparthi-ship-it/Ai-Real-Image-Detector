"""
Dataset that reads data/splits.csv and serves (image_tensor, label) pairs.

LABEL CONVENTION (fixed here, forever): label_a in the manifest is
    0 = real, 1 = ai
We use it directly, so the model's class indices are 0=real, 1=ai. This is the
single source of truth for what the two output logits mean; training, eval, and
the deployed app all inherit it. (v1 got burned by ImageFolder's alphabetical
ordering silently making 0=ai, 1=real — we pin it explicitly instead.)

Preprocessing is NOT defined here. It is imported from src.preprocessing so the
one contract governs training too (golden rule #1).
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import load_image, get_train_transform, get_eval_transform

CLASS_NAMES = ["real", "ai"]  # index 0 -> real, index 1 -> ai
SPLITS_CSV = "data/splits.csv"


class ImageSplitDataset(Dataset):
    """One split (train/val/test/gen_holdout) from data/splits.csv.

    train=True selects the augmenting transform; otherwise the deterministic
    eval transform. Augmentation is therefore train-only and applied at load
    time, strictly after the split (golden rule #2).
    """

    def __init__(self, split: str, train: bool | None = None,
                 splits_csv: str = SPLITS_CSV):
        df = pd.read_csv(splits_csv)
        self.rows = df[df.split == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"no rows for split={split!r} in {splits_csv}")
        # default: only the 'train' split augments
        use_train_tf = (split == "train") if train is None else train
        self.transform = get_train_transform() if use_train_tf else get_eval_transform()
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        img = load_image(row.path)
        x = self.transform(img)
        y = int(row.label_a)  # 0=real, 1=ai
        return x, y

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.rows.label_a.values, dtype=torch.long)
