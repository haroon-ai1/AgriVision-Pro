"""Inference: one ``Predictor`` used by the CLI, the desktop GUI and the web app.

Keeping a single implementation means the preprocessing can never drift between
what was trained and what is served -- the bug class that makes a deployed model
quietly worse than the one in the notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .model import load_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prettify(class_name: str) -> str:
    """``Tomato___Late_blight`` -> ``Tomato - Late Blight``."""
    parts = [p for p in class_name.split("___") if p]
    parts = [p.replace("_", " ").strip().title() for p in parts]
    return " - ".join(parts) if parts else class_name


class Predictor:
    def __init__(self, checkpoint: str | Path = "artifacts/best_model.pt", device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.metrics = load_checkpoint(checkpoint, self.device)
        self.class_names = self.cfg.class_names
        self.transform = build_transforms(self.cfg.image_size, train=False)

    @property
    def is_healthy_class(self) -> dict[str, bool]:
        return {c: "healthy" in c.lower() for c in self.class_names}

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1)[0].cpu()

        k = min(top_k, len(self.class_names))
        top_probs, top_idx = torch.topk(probs, k)

        predictions = [
            {
                "class": self.class_names[i],
                "label": prettify(self.class_names[i]),
                "confidence": float(p),
                "healthy": "healthy" in self.class_names[i].lower(),
            }
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]

        best = predictions[0]
        margin = best["confidence"] - (predictions[1]["confidence"] if len(predictions) > 1 else 0.0)

        return {
            "top": best,
            "predictions": predictions,
            "margin": margin,
            # A confident-looking percentage on a near-tie is how users get
            # misled, so surface uncertainty explicitly rather than hiding it.
            "uncertain": best["confidence"] < 0.60 or margin < 0.15,
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs.tolist())},
        }

    @torch.no_grad()
    def predict_batch(self, paths: Iterable[str | Path], batch_size: int = 16) -> list[dict]:
        paths = list(paths)
        results: list[dict] = []

        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            tensors, ok_paths = [], []
            for path in chunk:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as exc:
                    results.append({"path": str(path), "error": str(exc)})
                    continue
                tensors.append(self.transform(img))
                ok_paths.append(path)

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            probs = F.softmax(self.model(batch), dim=1).cpu()

            for path, row in zip(ok_paths, probs):
                idx = int(row.argmax())
                results.append(
                    {
                        "path": str(path),
                        "class": self.class_names[idx],
                        "label": prettify(self.class_names[idx]),
                        "confidence": float(row[idx]),
                    }
                )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify leaf images from the command line.")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[predict] Not found: {p}")

    if not paths:
        raise SystemExit("No images to score.")

    predictor = Predictor(args.checkpoint, args.device)

    if args.as_json:
        print(json.dumps(predictor.predict_batch(paths), indent=2))
        return

    for path in paths:
        result = predictor.predict(path, top_k=args.top_k)
        flag = "  [UNCERTAIN]" if result["uncertain"] else ""
        print(f"\n{path.name}{flag}")
        for i, pred in enumerate(result["predictions"]):
            marker = "->" if i == 0 else "  "
            print(f"  {marker} {pred['label']:<45} {pred['confidence'] * 100:6.2f}%")


if __name__ == "__main__":
    main()
