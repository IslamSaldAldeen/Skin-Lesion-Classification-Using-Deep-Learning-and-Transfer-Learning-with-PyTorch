from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from .transforms import get_eval_transforms


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, float(probs[0, class_idx].detach().cpu())


def get_target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Could not infer target Grad-CAM layer for this model.")


def make_gradcam_overlay(model, pil_image: Image.Image, device: torch.device):
    transform = get_eval_transforms(IMAGE_SIZE)
    x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    cam, class_idx, confidence = gradcam(x)
    gradcam.remove_hooks()

    original = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    heatmap = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original + 0.45 * heatmap)
    return overlay, CLASS_NAMES[class_idx], confidence


def save_gradcam_example(model, image_path: Path, output_path: Path, device: torch.device):
    pil_image = Image.open(image_path).convert("RGB")
    overlay, predicted_class, confidence = make_gradcam_overlay(model, pil_image, device)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pil_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {predicted_class} ({confidence:.2%})")
    axes[1].axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
