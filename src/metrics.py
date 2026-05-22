from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def logits_to_predictions(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds, probs


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def make_classification_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)


def make_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
