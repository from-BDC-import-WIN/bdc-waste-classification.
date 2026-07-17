"""
Generic checkpoint loader + batched inference for the error-diagnosis notebook.
Reconstructs the shared conv-head architecture used by every experiment
notebook (01-04) from its saved config.json, so logits/features can be
recomputed for all trained models without re-deriving the class per model.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

# swin's timm backbone returns channel-last (B, H, W, C); conv backbones
# (mobilenetv2/resnet/convnext) return channel-first (B, C, H, W) already.
CHANNEL_LAST_BACKBONE_PREFIXES = ("swin",)


class WasteClassifier(nn.Module):
    """Backbone + conv head, matching notebooks/experiments/{01..04}*.ipynb."""

    def __init__(
        self, backbone_name: str, num_classes: int, channel_last: bool
    ) -> None:
        super().__init__()
        self.channel_last = channel_last
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool=""
        )
        feat_channels = self.backbone.num_features

        self.custom_conv2d = nn.Conv2d(feat_channels, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.custom_maxpool = nn.MaxPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-d pooled feature vector fed into the classifier head."""
        x = self.backbone(x)
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        x = self.relu(self.custom_conv2d(x))
        x = self.custom_maxpool(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def load_trained_model(
    model_dir: Union[str, Path], device: torch.device
) -> Tuple[WasteClassifier, Dict]:
    """Rebuild a WasteClassifier from model_dir/config.json + model.pt, in eval mode."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r") as f:
        cfg = json.load(f)

    backbone_name = cfg["model"]["backbone"]
    channel_last = backbone_name.startswith(CHANNEL_LAST_BACKBONE_PREFIXES)

    model = WasteClassifier(
        backbone_name, num_classes=cfg["num_classes"], channel_last=channel_last
    )
    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, cfg


@torch.no_grad()
def run_inference(
    model: WasteClassifier,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    normalize_mean: List[float],
    normalize_std: List[float],
    device: torch.device,
    batch_size: int = 32,
    return_features: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Batched inference over image_paths, returns logits (N, C) and optionally penultimate features (N, 256)."""
    mean = torch.tensor(normalize_mean).view(1, 3, 1, 1)
    std = torch.tensor(normalize_std).view(1, 3, 1, 1)

    all_logits, all_features = [], []
    n_batches = (len(image_paths) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(image_paths), batch_size),
        total=n_batches,
        desc="inference",
        leave=False,
    ):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize(image_size)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        batch = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
        batch = (batch - mean) / std
        batch = batch.to(device)

        features = model.forward_features(batch)
        logits = model.classifier(features)

        all_logits.append(logits.cpu().numpy())
        if return_features:
            all_features.append(features.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    if return_features:
        return logits, np.concatenate(all_features, axis=0)
    return logits


def get_or_compute_logits(
    cache_path: Union[str, Path],
    model: WasteClassifier,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    normalize_mean: List[float],
    normalize_std: List[float],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Load cached logits from cache_path if present, otherwise compute + cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    logits = run_inference(
        model,
        image_paths,
        image_size,
        normalize_mean,
        normalize_std,
        device,
        batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, logits)
    return logits


def get_or_compute_features(
    cache_path: Union[str, Path],
    model: WasteClassifier,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    normalize_mean: List[float],
    normalize_std: List[float],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Load cached penultimate features from cache_path if present, otherwise compute + cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    _, features = run_inference(
        model,
        image_paths,
        image_size,
        normalize_mean,
        normalize_std,
        device,
        batch_size,
        return_features=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    return features
