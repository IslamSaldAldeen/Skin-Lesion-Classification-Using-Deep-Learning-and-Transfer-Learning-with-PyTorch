import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SEED
from .dataset import (
    prepare_dataframe,
    stratified_group_split,
    make_loaders,
    compute_class_weights,
)
from .losses import FocalLoss
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


def get_criterion(args, train_df, device):
    """
    Select the loss function.

    Supported options:
    - ce
    - weighted_ce
    - focal
    - weighted_focal

    Backward compatibility:
    If the old flag --weighted-loss is used with --loss ce,
    it will automatically behave as weighted_ce.
    """

    selected_loss = args.loss

    # Backward compatibility with old commands
    if args.weighted_loss:
        if selected_loss == "ce":
            selected_loss = "weighted_ce"
        elif selected_loss == "focal":
            selected_loss = "weighted_focal"

    needs_class_weights = selected_loss in ["weighted_ce", "weighted_focal"]
    class_weights = compute_class_weights(train_df).to(device) if needs_class_weights else None

    if selected_loss == "ce":
        criterion = nn.CrossEntropyLoss()

    elif selected_loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    elif selected_loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)

    elif selected_loss == "weighted_focal":
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)

    else:
        raise ValueError(f"Unsupported loss function: {selected_loss}")

    return criterion, selected_loss, class_weights


def get_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None

    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    raise ValueError("scheduler must be none or plateau")


def train_experiment(args):
    set_seed(args.seed)
    ensure_dirs([MODELS_DIR, OUTPUTS_DIR])
    device = get_device()

    df = prepare_dataframe()
    train_df, val_df, test_df = stratified_group_split(df, seed=args.seed)

    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_split.csv", index=False)

    train_loader, val_loader, _ = make_loaders(
        train_df,
        val_df,
        test_df,
        args.batch_size,
        args.num_workers,
    )

    model = build_model(
        args.architecture,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze=args.freeze,
    ).to(device)

    criterion, selected_loss, class_weights = get_criterion(args, train_df, device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = get_scheduler(args, optimizer)

    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Freeze strategy: {args.freeze}")
    print(f"Loss function: {selected_loss}")
    print(f"Old weighted-loss flag: {args.weighted_loss}")
    print(f"Focal gamma: {args.focal_gamma}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Early stopping: {args.early_stopping}")
    print(f"Patience: {args.patience}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    if class_weights is not None:
        print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    best_score = -1.0
    epochs_without_improvement = 0
    history = []

    best_path = MODELS_DIR / f"best_{args.experiment_name}.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        score = val_metrics["f1_macro"]

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "selected_loss": selected_loss,
        }

        history.append(row)
        print(json.dumps(row, indent=2))

        improved = score > best_score + args.min_delta

        if improved:
            best_score = score
            epochs_without_improvement = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "freeze": args.freeze,
                    "class_names": CLASS_NAMES,
                    "label_map": {c: i for i, c in enumerate(CLASS_NAMES)},
                    "val_metrics": val_metrics,
                    "loss": selected_loss,
                    "focal_gamma": args.focal_gamma,
                    "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
                },
                best_path,
            )

            print(f"Saved new best model to {best_path} | val_macro_f1={best_score:.4f}")

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best val_macro_f1={best_score:.4f}"
            )

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(score)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr < old_lr:
                print(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best val_macro_f1={best_score:.4f}"
            )
            break

    hist_df = pd.DataFrame(history)

    hist_path = OUTPUTS_DIR / f"history_{args.experiment_name}.csv"
    hist_df.to_csv(hist_path, index=False)

    plot_training_curves(
        hist_path,
        OUTPUTS_DIR / f"training_curves_{args.experiment_name}.png",
    )

    save_json(vars(args), OUTPUTS_DIR / f"config_{args.experiment_name}.json")

    print(f"History saved to {hist_path}")
    print(f"Best model saved to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HAM10000 skin lesion classifier")

    parser.add_argument("--experiment-name", type=str, default="exp2_resnet50_weighted_head")

    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
    )

    parser.add_argument(
        "--freeze",
        type=str,
        default="head",
        choices=["head", "last_block", "whole", "none"],
    )

    # Old flag kept for compatibility with previous experiments
    parser.add_argument("--weighted-loss", action="store_true")

    # New loss options for Version 2
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "weighted_ce", "focal", "weighted_focal"],
        help="Loss function to use: ce, weighted_ce, focal, or weighted_focal.",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma value for Focal Loss.",
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
    )

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)

    # Early stopping
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop training when validation macro F1 does not improve.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement before early stopping.",
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early stopping patience.",
    )

    # Learning rate scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "plateau"],
        help="Learning rate scheduler. Use plateau for ReduceLROnPlateau.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="Patience for ReduceLROnPlateau.",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Factor for reducing learning rate in ReduceLROnPlateau.",
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train_experiment(parse_args())