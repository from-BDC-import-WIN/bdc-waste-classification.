"""
Two split strategies over raw per-class image folders:
- split_and_organize_dataset: physically copies files into data/processed/{train,val}.
- build_or_load_manifest: index-only stratified train/inner_val/holdout split,
  cached as a CSV manifest (no file duplication). Used by the ConvNeXt
  fine-tune -> embedding -> SVM pipeline, where holdout must stay untouched
  by both fine-tuning and SVM/scaler fitting.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from sklearn.model_selection import train_test_split

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


def _list_raw_files(
    raw_dir: Union[str, Path], class_names: Dict[str, str]
) -> pd.DataFrame:
    """Scan raw_dir/{folder} for images, return DataFrame[filepath, class_name]."""
    rows = []
    for folder_name, english_name in class_names.items():
        folder = Path(raw_dir) / folder_name
        if not folder.exists():
            print(f"  [WARNING] Folder tidak ditemukan: {folder}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in folder.glob(ext):
                rows.append({"filepath": str(path), "class_name": english_name})
    return pd.DataFrame(rows)


def build_or_load_manifest(
    raw_dir: Union[str, Path],
    class_names: Dict[str, str],
    manifest_path: Union[str, Path],
    holdout_ratio: float,
    inner_val_ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified split: all labeled data -> train_dev (1-holdout_ratio) / holdout,
    then train_dev -> train / inner_val (inner_val_ratio of train_dev).

    Cached to manifest_path as a CSV [filepath, class_name, split] where
    split in {"train", "inner_val", "holdout"}. If manifest_path already
    exists it is loaded as-is -- the split is built exactly once, seeded.
    holdout rows must never be used to fit/tune anything (backbone or SVM).
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        print(f"Manifest sudah ada, load dari '{manifest_path}' (split tidak diulang).")
        return pd.read_csv(manifest_path)

    df = _list_raw_files(raw_dir, class_names)
    if df.empty:
        raise RuntimeError(f"Tidak ada gambar ditemukan di '{raw_dir}'.")

    train_dev_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        stratify=df["class_name"],
        random_state=seed,
    )
    train_df, inner_val_df = train_test_split(
        train_dev_df,
        test_size=inner_val_ratio,
        stratify=train_dev_df["class_name"],
        random_state=seed,
    )

    manifest = pd.concat(
        [
            train_df.assign(split="train"),
            inner_val_df.assign(split="inner_val"),
            holdout_df.assign(split="holdout"),
        ],
        ignore_index=True,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest baru dibuat & disimpan ke '{manifest_path}'.")

    return manifest


def print_split_summary(manifest: pd.DataFrame) -> None:
    """Print per-split, per-class row counts (sanity check for the stratified split)."""
    summary = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    print(summary)
    print(f"\nTotal per split:\n{manifest['split'].value_counts()}")
