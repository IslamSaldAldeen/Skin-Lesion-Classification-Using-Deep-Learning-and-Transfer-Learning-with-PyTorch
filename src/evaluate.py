import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .dataset import prepare_dataframe, stratified_group_split, make_loaders
from .metrics import compute_metrics, logits_to_predictions, make_classification_report, make_confusion_matrix
from .model import build_model
from .transforms import get_eval_transforms
from .utils import ensure_dirs, get_device, load_checkpoint, save_json


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, loader, device):
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            preds, probs = logits_to_predictions(logits)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_true, all_pred, all_probs


# ── Confusion matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("HAM10000 Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ── Error analysis visual panel ────────────────────────────────────────────────
def _denormalize(tensor):
    """Undo ImageNet normalisation so pixel values are displayable."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_error_analysis(pred_df: pd.DataFrame, output_path: Path, n_correct: int = 8, n_wrong: int = 8):
    """
    Generates a visual panel showing:
      - Top row(s): correctly classified examples (green border)
      - Bottom row(s): misclassified examples (red border)

    Each cell shows the image, true label, predicted label, and confidence.

    Args:
        pred_df    : DataFrame with columns image_path, true_class, pred_class, confidence, top3
        output_path: where to save the PNG
        n_correct  : number of correct examples to show
        n_wrong    : number of wrong examples to show
    """
    transform = get_eval_transforms(IMAGE_SIZE)

    correct_df = pred_df[pred_df["true_class"] == pred_df["pred_class"]].sample(
        min(n_correct, len(pred_df[pred_df["true_class"] == pred_df["pred_class"]])),
        random_state=42,
    )
    wrong_df = pred_df[pred_df["true_class"] != pred_df["pred_class"]].sample(
        min(n_wrong, len(pred_df[pred_df["true_class"] != pred_df["pred_class"]])),
        random_state=42,
    )

    cols = 4
    total = n_correct + n_wrong
    rows  = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    fig.suptitle(
        "Prediction Examples\n"
        "Green border = Correct  |  Red border = Incorrect",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    combined = list(correct_df.iterrows()) + list(wrong_df.iterrows())

    for ax_idx, (_, row) in enumerate(combined):
        ax = axes[ax_idx]
        is_correct = row["true_class"] == row["pred_class"]
        border_color = "#2ecc71" if is_correct else "#e74c3c"

        # Load image
        try:
            img = Image.open(row["image_path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            img_np = np.array(img) / 255.0
        except Exception:
            img_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

        ax.imshow(img_np)

        # Coloured border via spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)

        conf = row["confidence"] * 100
        title_lines = [
            f"True:  {row['true_class']}",
            f"Pred:  {row['pred_class']}  ({conf:.1f}%)",
        ]
        title_color = "#1a7a3f" if is_correct else "#c0392b"
        ax.set_title("\n".join(title_lines), fontsize=8.5, color=title_color, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(combined):]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Error analysis panel saved → {output_path}")


def plot_class_error_rates(pred_df: pd.DataFrame, output_path: Path):
    """
    Bar chart: per-class error rate, sorted worst → best.
    Highlights medically critical classes (mel, bcc, akiec) in red.
    """
    critical = {"mel", "bcc", "akiec"}
    stats = []
    for cls in CLASS_NAMES:
        subset = pred_df[pred_df["true_class"] == cls]
        if len(subset) == 0:
            continue
        error_rate = (subset["true_class"] != subset["pred_class"]).mean()
        stats.append({"class": cls, "error_rate": error_rate})

    stats_df = pd.DataFrame(stats).sort_values("error_rate", ascending=False)
    colors   = ["#e74c3c" if c in critical else "#5b9bd5" for c in stats_df["class"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stats_df["class"], stats_df["error_rate"] * 100, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Per-Class Error Rate  (Red = Medically Critical)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Error Rate (%)")
    ax.set_ylim(0, min(100, stats_df["error_rate"].max() * 130))
    ax.grid(axis="y", alpha=0.4)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#e74c3c", label="Critical (mel / bcc / akiec)"),
        mpatches.Patch(color="#5b9bd5", label="Other classes"),
    ]
    ax.legend(handles=legend, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Class error rates saved → {output_path}")


# ── Main evaluate function ─────────────────────────────────────────────────────
def evaluate(args):
    ensure_dirs([OUTPUTS_DIR])
    device = get_device()
    checkpoint_path = Path(args.model_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", args.architecture)

    df = prepare_dataframe()
    if (DATA_DIR / "test_split.csv").exists():
        test_df = pd.read_csv(DATA_DIR / "test_split.csv")
    else:
        _, _, test_df = stratified_group_split(df)

    _, _, test_loader = make_loaders(test_df, test_df, test_df, args.batch_size, args.num_workers)
    model = build_model(architecture, num_classes=len(CLASS_NAMES), pretrained=False, freeze="none").to(device)
    load_checkpoint(model, checkpoint_path, device)

    y_true, y_pred, y_probs = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    report  = make_classification_report(y_true, y_pred, CLASS_NAMES)
    cm      = make_confusion_matrix(y_true, y_pred)

    print(report)
    print(metrics)

    (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    save_json(metrics, OUTPUTS_DIR / "test_metrics.json")
    plot_confusion_matrix(cm, OUTPUTS_DIR / "confusion_matrix.png")

    # Build predictions DataFrame
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["pred_label"] = y_pred
    pred_df["pred_class"] = [CLASS_NAMES[i] for i in y_pred]
    pred_df["true_class"] = [CLASS_NAMES[i] for i in y_true]
    pred_df["confidence"] = [max(p) for p in y_probs]
    pred_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    # ── NEW: visual error analysis ─────────────────────────────────────────────
    plot_error_analysis(
        pred_df,
        output_path=OUTPUTS_DIR / "error_analysis_panel.png",
        n_correct=8,
        n_wrong=8,
    )
    plot_class_error_rates(
        pred_df,
        output_path=OUTPUTS_DIR / "class_error_rates.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained HAM10000 classifier")
    parser.add_argument("--model-path",   type=str, default=str(MODELS_DIR / "best_exp2_resnet50_weighted_head.pth"))
    parser.add_argument("--architecture", type=str, default="resnet50")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--num-workers",  type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
