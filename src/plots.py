from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history_csv: Path, output_path: Path):
    df = pd.read_csv(history_csv)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax1.plot(df["epoch"], df["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    if "val_f1_macro" in df.columns:
        ax2.plot(df["epoch"], df["val_f1_macro"], label="Validation macro F1", linestyle="--")
        ax2.set_ylabel("Macro F1")
        ax2.legend(loc="upper right")
    plt.title("Training Curves")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
