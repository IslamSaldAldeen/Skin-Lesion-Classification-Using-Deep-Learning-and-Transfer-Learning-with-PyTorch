import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SEED
from .dataset import prepare_dataframe, stratified_group_split, make_loaders, compute_class_weights
from .metrics import compute_metrics, logits_to_predictions
from .model import build_model, count_trainable_parameters
from .utils import AverageMeter, ensure_dirs, get_device, save_checkpoint, save_json, set_seed
from .plots import plot_training_curves


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    meter = AverageMeter()
    all_true, all_pred = [], []
    model.train() if train else model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds, _ = logits_to_predictions(logits)
            meter.update(loss.item(), images.size(0))
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(all_true, all_pred, CLASS_NAMES)
    metrics["loss"] = meter.avg
    return metrics


def get_optimizer(model, name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name.lower() == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name.lower() == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError("optimizer must be adamw or sgd")


def train_experiment(args):
    set_seed(args.seed)
    ensure_dirs([MODELS_DIR, OUTPUTS_DIR])
    device = get_device()

    df = prepare_dataframe()
    train_df, val_df, test_df = stratified_group_split(df, seed=args.seed)
    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_split.csv", index=False)

    train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df, args.batch_size, args.num_workers)
    model = build_model(args.architecture, num_classes=len(CLASS_NAMES), pretrained=True, freeze=args.freeze).to(device)

    class_weights = compute_class_weights(train_df).to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Freeze strategy: {args.freeze}")
    print(f"Weighted loss: {args.weighted_loss}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    best_score = -1.0
    history = []
    best_path = MODELS_DIR / f"best_{args.experiment_name}.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=2))

        score = val_metrics["f1_macro"]
        if score > best_score:
            best_score = score
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "architecture": args.architecture,
                "freeze": args.freeze,
                "class_names": CLASS_NAMES,
                "label_map": {c: i for i, c in enumerate(CLASS_NAMES)},
                "val_metrics": val_metrics,
            }, best_path)
            print(f"Saved new best model to {best_path} | val_macro_f1={best_score:.4f}")

    hist_df = pd.DataFrame(history)
    hist_path = OUTPUTS_DIR / f"history_{args.experiment_name}.csv"
    hist_df.to_csv(hist_path, index=False)
    plot_training_curves(hist_path, OUTPUTS_DIR / f"training_curves_{args.experiment_name}.png")
    save_json(vars(args), OUTPUTS_DIR / f"config_{args.experiment_name}.json")
    print(f"History saved to {hist_path}")
    print(f"Best model saved to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HAM10000 skin lesion classifier")
    parser.add_argument("--experiment-name", type=str, default="exp2_resnet50_weighted_head")
    parser.add_argument("--architecture", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--freeze", type=str, default="head", choices=["head", "last_block", "whole", "none"])
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train_experiment(parse_args())
