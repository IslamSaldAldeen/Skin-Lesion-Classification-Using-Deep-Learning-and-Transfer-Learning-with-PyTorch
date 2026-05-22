from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader

from .config import CLASS_NAMES, LABEL_MAP, METADATA_FILE, IMAGE_DIRS, SEED
from .transforms import get_train_transforms, get_eval_transforms


def load_metadata(metadata_file: Path = METADATA_FILE) -> pd.DataFrame:
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_file}\n"
            "Download HAM10000 from Kaggle and place HAM10000_metadata.csv inside data/."
        )
    return pd.read_csv(metadata_file)


def add_image_paths(df: pd.DataFrame, image_dirs: List[Path] = IMAGE_DIRS) -> pd.DataFrame:
    image_lookup = {}
    for image_dir in image_dirs:
        if image_dir.exists():
            for path in image_dir.glob("*.jpg"):
                image_lookup[path.stem] = str(path)

    df = df.copy()
    df["image_path"] = df["image_id"].map(image_lookup)
    missing = df["image_path"].isna().sum()
    if missing:
        raise FileNotFoundError(
            f"Could not find image_path for {missing} rows. Expected image folders in data/: "
            "HAM10000_images_part_1/ and HAM10000_images_part_2/."
        )
    return df


def prepare_dataframe(metadata_file: Path = METADATA_FILE) -> pd.DataFrame:
    df = load_metadata(metadata_file)
    df = add_image_paths(df)
    df["label"] = df["dx"].map(LABEL_MAP).astype(int)
    return df


def stratified_group_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if "lesion_id" not in df.columns:
        train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df["dx"], random_state=seed)
        relative_val = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, train_size=relative_val, stratify=temp_df["dx"], random_state=seed)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # Group split reduces leakage by keeping the same lesion_id in only one split.
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df["lesion_id"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    relative_val = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["lesion_id"]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    num_workers: int = 2,
):
    train_ds = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_ds = SkinLesionDataset(val_df, transform=get_eval_transforms())
    test_ds = SkinLesionDataset(test_df, transform=get_eval_transforms()) if test_df is not None else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None
    return train_loader, val_loader, test_loader


def compute_class_weights(train_df: pd.DataFrame):
    counts = train_df["dx"].value_counts().reindex(CLASS_NAMES).fillna(0).values
    total = counts.sum()
    weights = total / (len(CLASS_NAMES) * counts.clip(min=1))
    return torch.tensor(weights, dtype=torch.float32)
