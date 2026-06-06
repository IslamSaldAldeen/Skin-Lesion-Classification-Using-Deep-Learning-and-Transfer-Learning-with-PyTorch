import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    It reduces the effect of easy examples and focuses more on
    hard/misclassified examples.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")