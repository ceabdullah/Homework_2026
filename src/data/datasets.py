from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import apply_color_mode

# cv2 spawns its own thread pool, which fights the DataLoader workers.
cv2.setNumThreads(0)


class LesionDataset(Dataset):
    """Rows come from a committed split CSV; `split=None` uses every row.

    `classes` is the model's label space, not the file's -- a HAM-trained
    4-class model evaluated on PAD must index classes identically, so the
    mapping is always driven by the caller.
    """

    def __init__(self, csv_path, split, classes, transform=None, color_mode="raw"):
        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"{csv_path} has no rows for split={split!r}")
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        unknown = set(self.df["label"]) - set(self.cls_to_idx)
        if unknown:
            raise ValueError(f"{csv_path} contains labels outside the model's "
                             f"label space: {unknown}")
        self.transform = transform
        self.color_mode = color_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {row['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_color_mode(img, self.color_mode)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, self.cls_to_idx[row["label"]], str(row["image_id"])


def class_weights(csv_path, split, classes) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the mean weight is 1."""
    df = pd.read_csv(csv_path)
    if split is not None:
        df = df[df["split"] == split]
    counts = np.array([max((df["label"] == c).sum(), 1) for c in classes], dtype=np.float32)
    w = counts.sum() / (len(classes) * counts)
    return torch.tensor(w)
