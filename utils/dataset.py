"""
Dataset and DataLoader builders shared across notebook experiments.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class WasteImageDataset(Dataset):
    """ImageFolder-style dataset with an albumentations transform."""

    def __init__(
        self, root_dir: Path, class_names: list[str], transform: A.Compose
    ) -> None:
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        for name in class_names:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for path in (root_dir / name).glob(ext):
                    self.samples.append((path, self.class_to_idx[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, label


def get_dataloaders(
    train_dir: Path,
    val_dir: Path,
    class_names: List[str],
    train_transform: A.Compose,
    eval_transform: A.Compose,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build train/val WasteImageDataset + DataLoader pair, return (train_loader, val_loader, class_to_idx)."""
    train_dataset = WasteImageDataset(train_dir, class_names, train_transform)
    val_dataset = WasteImageDataset(val_dir, class_names, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_dataset.class_to_idx
