"""
Extract frozen embeddings from a fine-tuned classifier's forward_features()
(Phase 2 of the ConvNeXt -> embedding -> SVM pipeline) and cache them to disk.

The full model (backbone + custom conv head) is frozen (eval(), no_grad()) and
called through forward_features() -- i.e. everything up to the pooled feature
vector, before the final classifier Linear layers. Embeddings are cached per
split (train/inner_val/holdout) so repeated SVM experiments don't need to
re-run the backbone.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.engine import mps_autocast


class _ForwardFeaturesWrapper(nn.Module):
    """Adapts model.forward_features(x) to a plain nn.Module callable (model(x))."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


def build_feature_extractor(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Load a fine-tuned checkpoint's state_dict into `model` (must expose
    forward_features() returning a pooled (B, D) embedding), freeze every
    parameter (eval mode, requires_grad=False), and return a callable wrapper
    around forward_features (i.e. the classifier head is never invoked).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    return _ForwardFeaturesWrapper(model)


@torch.no_grad()
def extract_embeddings(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run feature_extractor over dataloader, return (embeddings [N, D], labels [N])."""
    feature_extractor.eval()
    all_embeddings, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Extract embeddings", leave=False):
        images = images.to(device)
        with mps_autocast(device):
            embeddings = feature_extractor(images)
        all_embeddings.append(embeddings.float().cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def extract_and_cache_embeddings(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like extract_embeddings, but skips recomputation if cache_path (.npz) already exists."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Embeddings sudah ada, load dari '{cache_path}'.")
        data = np.load(cache_path)
        return data["embeddings"], data["labels"]

    embeddings, labels = extract_embeddings(feature_extractor, dataloader, device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, labels=labels)
    print(f"Embeddings ({embeddings.shape}) disimpan ke '{cache_path}'.")

    return embeddings, labels
