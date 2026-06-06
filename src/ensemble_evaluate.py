import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, OUTPUTS_DIR
from .dataset import make_loaders
from .model import build_model
from .utils import get_device, ensure_dirs


def load_checkpoint_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    architecture = checkpoint.get("architecture", None)
    freeze = checkpoint.get("freeze", "whole")

    if architecture is None:
        raise ValueError(
            f"Checkpoint {model_path} does not contain architecture info. "
            f"Please provide only checkpoints saved by our train.py."
        )

    model = build_model(
        architecture=architecture,
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze=freeze,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, architecture


def ensemble_predict(models, loader, device):
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)

            probs_sum = None

            for model in models:
                logits = model(images)
                probs = F.softmax(logits, dim=1)

                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum += probs

            avg_probs = probs_sum / len(models)
            preds = torch.argmax(avg_probs, dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    return all_true, all_pred


def compute_summary_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main(args):
    ensure_dirs([OUTPUTS_DIR])
    device = get_device()

    print(f"Device: {device}")

    train_split_path = DATA_DIR / "train_split.csv"
    val_split_path = DATA_DIR / "val_split.csv"
    test_split_path = DATA_DIR / "test_split.csv"

    if not train_split_path.exists() or not val_split_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            "Could not find train_split.csv, val_split.csv, and test_split.csv inside data/. "
            "Run training first so the splits are created."
        )

    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)
    test_df = pd.read_csv(test_split_path)

    _, _, test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        args.batch_size,
        args.num_workers,
    )

    models = []
    model_info = []

    for path in args.model_paths:
        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model, architecture = load_checkpoint_model(model_path, device)
        models.append(model)
        model_info.append({"path": str(model_path), "architecture": architecture})

        print(f"Loaded model: {model_path} | architecture={architecture}")

    print(f"\nRunning soft-voting ensemble with {len(models)} models...\n")

    y_true, y_pred = ensemble_predict(models, test_loader, device)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    metrics = compute_summary_metrics(y_true, y_pred)

    print(report)
    print(json.dumps(metrics, indent=2))

    output_txt = OUTPUTS_DIR / f"ensemble_report_{args.ensemble_name}.txt"
    output_json = OUTPUTS_DIR / f"ensemble_metrics_{args.ensemble_name}.json"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Models used:\n")
        for info in model_info:
            f.write(json.dumps(info) + "\n")
        f.write("\nClassification report:\n")
        f.write(report)
        f.write("\nMetrics:\n")
        f.write(json.dumps(metrics, indent=2))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": model_info,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"\nSaved ensemble report to: {output_txt}")
    print(f"Saved ensemble metrics to: {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate soft-voting ensemble")

    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="List of model checkpoint paths to ensemble.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ensemble-name",
        type=str,
        default="v2_ensemble",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())