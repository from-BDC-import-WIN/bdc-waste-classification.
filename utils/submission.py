from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image


def preprocess_for_inference(
    image_path: Union[str, Path],
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Load an image, resize to target_size, and apply preprocess_fn."""
    img = Image.open(image_path).convert("RGB").resize(target_size)
    return preprocess_fn(np.array(img))


def load_test_images(
    test_dir: Union[str, Path],
    test_ids: List[str],
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, List[str]]:
    """Load test images by id, trying .jpg then .png. Returns (array, valid_ids), skipping missing files with a warning."""
    test_dir = Path(test_dir)
    images = []
    valid_ids = []

    for test_id in test_ids:
        path = test_dir / f"{test_id}.jpg"
        if not path.exists():
            path = test_dir / f"{test_id}.png"
        if not path.exists():
            print(f"Warning: image not found for id '{test_id}' (.jpg/.png)")
            continue

        images.append(preprocess_for_inference(path, preprocess_fn, target_size))
        valid_ids.append(test_id)

    return np.array(images), valid_ids


def predict_competition_labels(
    model: torch.nn.Module,
    test_array: np.ndarray,
    class_idx_to_comp_label: Dict[int, str],
    batch_size: int = 32,
) -> List[str]:
    """Run model inference over test_array in batches and map predicted class indices to competition labels."""
    model.eval()
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for i in range(0, len(test_array), batch_size):
            batch = torch.from_numpy(test_array[i : i + batch_size]).float().to(device)
            if batch.ndim == 4 and batch.shape[-1] in (1, 3):
                batch = batch.permute(0, 3, 1, 2)

            logits = model(batch)
            class_idx = logits.argmax(dim=1).cpu().numpy()
            preds.extend(class_idx_to_comp_label[idx] for idx in class_idx)

    return preds


def print_label_distribution(preds: List[str], label_names: List[str]) -> None:
    """Print the count and percentage of each label in label_names found in preds."""
    counts = Counter(preds)
    total = len(preds)
    print(f"Label distribution ({total} predictions):")
    for label in label_names:
        count = counts.get(label, 0)
        pct = 100 * count / total if total else 0
        print(f"  {label}: {count} ({pct:.1f}%)")


def build_class_index_mapping(
    class_indices: Dict[str, int],
    english_to_competition: Dict[str, str],
) -> Dict[int, str]:
    """Build a mapping from model class index to competition label, via the english class name."""
    return {idx: english_to_competition[name] for name, idx in class_indices.items()}
