"""
generate_gradcam_examples.py

Auto-generates outputs/gradcam_examples.png by running Grad-CAM on
one test image per class (7 classes = 7 panels), using the saved best model.

Usage (from project root):
    python src/generate_gradcam_examples.py --model-path models/best_model.pth
    python src/generate_gradcam_examples.py --model-path models/best_model.pth --arch efficientnet_b0
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CLASS_NAMES, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGE_SIZE
from .dataset import prepare_dataframe, stratified_group_split
from .gradcam import GradCAM, get_target_layer, make_gradcam_overlay
from .model import build_model
from .transforms import get_eval_transforms
from .utils import ensure_dirs, get_device, load_checkpoint


def pick_one_per_class(test_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Returns one representative row per class from the test set."""
    return (
        test_df.groupby("dx", group_keys=False)
               .apply(lambda g: g.sample(1, random_state=seed))
               .reset_index(drop=True)
    )


def generate_gradcam_grid(model, sample_df: pd.DataFrame,
                           device: torch.device,
                           output_path: Path,
                           correct_preds_only: bool = False):
    """
    For each row in sample_df, produces a (original | Grad-CAM overlay) pair.
    Arranges all pairs in a grid and saves to output_path.

    Layout: 7 rows (one per class) × 2 columns (original, overlay).
    """
    transform = get_eval_transforms(IMAGE_SIZE)
    n = len(sample_df)

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.2))
    fig.suptitle(
        "Grad-CAM Visualisation — One Example per Class\n"
        "Highlighted regions most influenced the model's prediction",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row_idx, (_, row) in enumerate(sample_df.iterrows()):
        try:
            pil_img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            for col in range(2):
                axes[row_idx][col].axis("off")
                axes[row_idx][col].text(0.5, 0.5, "image not found",
                                         ha="center", va="center")
            continue

        overlay, pred_class, confidence = make_gradcam_overlay(model, pil_img, device)
        true_class = row["dx"]
        is_correct = (pred_class == true_class)
        pred_color = "#1a7a3f" if is_correct else "#c0392b"

        # Original image
        axes[row_idx][0].imshow(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[row_idx][0].set_title(f"True: {true_class}", fontsize=9, fontweight="bold")
        axes[row_idx][0].axis("off")

        # Grad-CAM overlay
        axes[row_idx][1].imshow(overlay)
        axes[row_idx][1].set_title(
            f"Pred: {pred_class}  ({confidence:.1%})",
            fontsize=9, color=pred_color, fontweight="bold",
        )
        axes[row_idx][1].axis("off")

        # Side label
        axes[row_idx][0].set_ylabel(
            true_class.upper(), rotation=0, labelpad=40,
            fontsize=9, va="center", ha="right",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Grad-CAM examples saved → {output_path}")


def generate_gradcam_misclassified(model, pred_csv: Path,
                                    device: torch.device,
                                    output_path: Path,
                                    n: int = 6):
    """
    Bonus: runs Grad-CAM on the n most-confident wrong predictions.
    Helps understand what regions the model focused on when it was wrong.
    Requires test_predictions.csv to already exist (produced by evaluate.py).
    """
    if not pred_csv.exists():
        print(f"  [SKIP] {pred_csv} not found — run evaluate.py first.")
        return

    pred_df = pd.read_csv(pred_csv)
    wrong   = pred_df[pred_df["true_class"] != pred_df["pred_class"]] \
                .sort_values("confidence", ascending=False) \
                .head(n)

    if len(wrong) == 0:
        print("  [SKIP] No misclassified examples found.")
        return

    cols = 3
    rows = (len(wrong) * 2 + cols - 1) // cols   # 2 images (orig+cam) per example
    rows = len(wrong)

    fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3.2))
    if rows == 1:
        axes = [axes]
    fig.suptitle(
        f"Grad-CAM on Top-{n} Most-Confident Misclassifications\n"
        "Shows what the model 'looked at' when it made a wrong prediction",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for i, (_, row) in enumerate(wrong.iterrows()):
        try:
            pil_img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            for col in range(2):
                axes[i][col].axis("off")
            continue

        overlay, pred_class, confidence = make_gradcam_overlay(model, pil_img, device)

        axes[i][0].imshow(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[i][0].set_title(f"True: {row['true_class']}", fontsize=9, color="#1a7a3f", fontweight="bold")
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay)
        axes[i][1].set_title(f"Pred: {pred_class}  ({confidence:.1%}) ✗",
                              fontsize=9, color="#c0392b", fontweight="bold")
        axes[i][1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Grad-CAM misclassifications saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate gradcam_examples.png")
    parser.add_argument("--model-path",  type=str,
                        default=str(MODELS_DIR / "best_model.pth"))
    parser.add_argument("--arch",        type=str, default="resnet50",
                        choices=["resnet50", "resnet18", "efficientnet_b0"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    ensure_dirs([OUTPUTS_DIR])
    device         = get_device()
    checkpoint_path = Path(args.model_path)

    # Load model
    model = build_model(args.arch, num_classes=len(CLASS_NAMES),
                        pretrained=False, freeze="none").to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Get test data
    df = prepare_dataframe()
    if (DATA_DIR / "test_split.csv").exists():
        test_df = pd.read_csv(DATA_DIR / "test_split.csv")
    else:
        _, _, test_df = stratified_group_split(df)

    # Pick one image per class
    sample_df = pick_one_per_class(test_df, seed=args.seed)
    print(f"[INFO] Selected {len(sample_df)} images (one per class) for Grad-CAM.")

    # Generate main grid  →  outputs/gradcam_examples.png  (required output)
    generate_gradcam_grid(
        model, sample_df, device,
        output_path=OUTPUTS_DIR / "gradcam_examples.png",
    )

    # Generate misclassification grid  →  outputs/gradcam_misclassified.png  (bonus)
    generate_gradcam_misclassified(
        model,
        pred_csv=OUTPUTS_DIR / "test_predictions.csv",
        device=device,
        output_path=OUTPUTS_DIR / "gradcam_misclassified.png",
        n=6,
    )

    print("\n[✓] All Grad-CAM outputs generated.")


if __name__ == "__main__":
    main()
