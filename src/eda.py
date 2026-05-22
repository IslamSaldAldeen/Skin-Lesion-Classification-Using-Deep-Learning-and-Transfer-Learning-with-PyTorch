from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import CLASS_NAMES, OUTPUTS_DIR
from .dataset import prepare_dataframe


def plot_class_distribution(df: pd.DataFrame, output_path: Path = OUTPUTS_DIR / "class_distribution.png"):
    counts = df["dx"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("HAM10000 Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_images(df: pd.DataFrame, output_path: Path = OUTPUTS_DIR / "sample_images.png"):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(18, 4))
    for ax, cls in zip(axes, CLASS_NAMES):
        row = df[df["dx"] == cls].sample(1, random_state=42).iloc[0]
        img = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    df = prepare_dataframe()
    print("Shape:", df.shape)
    print(df.head())
    print("Missing values:")
    print(df.isna().sum())
    print("Class distribution:")
    print(df["dx"].value_counts())
    plot_class_distribution(df)
    plot_sample_images(df)


if __name__ == "__main__":
    main()
