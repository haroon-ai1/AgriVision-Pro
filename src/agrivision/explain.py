"""Grad-CAM heatmaps.

This exists for a specific reason. On PlantVillage a model can hit very high
accuracy while attending to the backdrop or the leaf silhouette. Grad-CAM is how
you check whether the network is actually looking at the lesion, and the overlay
is far more convincing evidence in a README than another accuracy figure.

Implemented directly against forward/backward hooks -- no extra dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .model import load_checkpoint
from .predict import prettify


def find_target_layer(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    """Last convolutional block -- the deepest layer that still has spatial extent."""
    if backbone.startswith("efficientnet") or backbone.startswith("mobilenet"):
        return model.features[-1]
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    # Fallback: the last module that produces a 4-D activation.
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    return conv_layers[-1]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int, float]:
        tensor = tensor.clone().requires_grad_(True)

        # Grad-CAM needs gradients, so no_grad must not be active here.
        with torch.enable_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            if class_index is None:
                class_index = int(logits.argmax(1).item())

            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not fire -- check the target layer.")

        # Channel importance = spatially averaged gradient.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0].cpu().numpy()
        span = cam.max() - cam.min()
        cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

        return cam, class_index, float(probs[0, class_index].detach())


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import cv2

    rgb = np.array(image.convert("RGB").resize((cam.shape[1], cam.shape[0])))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = np.uint8(rgb * (1 - alpha) + heat * alpha)
    return Image.fromarray(blended)


def explain_image(
    predictor_model, cfg, image: Image.Image, device: torch.device, alpha: float = 0.45
) -> tuple[Image.Image, str, float]:
    """Convenience wrapper returning (overlay, predicted_label, confidence)."""
    transform = build_transforms(cfg.image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    target_layer = find_target_layer(predictor_model, cfg.backbone)
    with GradCAM(predictor_model, target_layer) as cam_fn:
        cam, class_index, confidence = cam_fn(tensor)

    return overlay_heatmap(image, cam, alpha), prettify(cfg.class_names[class_index]), confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Grad-CAM overlays for leaf images.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--out-dir", default="artifacts/gradcam")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw in args.inputs:
        path = Path(raw)
        image = Image.open(path).convert("RGB")
        overlay, label, confidence = explain_image(model, cfg, image, device, args.alpha)
        dest = out_dir / f"{path.stem}_gradcam.png"
        overlay.save(dest)
        print(f"{path.name}: {label} ({confidence * 100:.2f}%) -> {dest}")


if __name__ == "__main__":
    main()
