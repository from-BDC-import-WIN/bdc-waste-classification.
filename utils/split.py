"""
Stratified train/val split: copies raw per-class images into an organized
data/processed/{train,val}/{class_name} directory tree.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, Union

SPLIT_NAMES = ["train", "val"]


def split_and_organize_dataset(
    raw_dir: Union[str, Path],
    output_dir: Path,
    class_names: Dict[str, str],
    train_ratio: float,
    seed: int = 42,
) -> None:
    """Split each raw class folder into train/val and copy files into output_dir, per class_names (raw folder name -> english class name)."""
    if output_dir.exists():
        print(f"Folder '{output_dir}' sudah ada. Skip splitting.")
        return

    random.seed(seed)

    for folder_name, english_name in class_names.items():
        source_folder = Path(raw_dir) / folder_name
        if not source_folder.exists():
            print(f"  [WARNING] Folder tidak ditemukan: {source_folder}")
            continue

        for split_name in SPLIT_NAMES:
            (output_dir / split_name / english_name).mkdir(parents=True, exist_ok=True)

        all_files = (
            list(source_folder.glob("*.jpg"))
            + list(source_folder.glob("*.jpeg"))
            + list(source_folder.glob("*.png"))
        )
        random.shuffle(all_files)

        num_train = int(len(all_files) * train_ratio)
        split_map = {
            "train": all_files[:num_train],
            "val": all_files[num_train:],
        }

        for split_name, files in split_map.items():
            dest = output_dir / split_name / english_name
            for f in files:
                shutil.copy2(f, dest / f.name)

        counts = {s: len(v) for s, v in split_map.items()}
        print(f"  {english_name:12}: train={counts['train']}, val={counts['val']}")

    print(f"\nDataset berhasil di-split ke '{output_dir}'")
