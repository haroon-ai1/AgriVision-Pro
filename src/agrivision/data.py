"""Dataset discovery, reproducible stratified splits, and transforms.

The dataset is expected in ImageFolder layout::

    <root>/
        Tomato___Late_blight/  img0.jpg ...
        Tomato___healthy/      img0.jpg ...

Splits are computed once, written to ``splits.json`` and reused. This matters:
the original project used a single 80/20 split and reported test accuracy on the
same data used to pick hyper-parameters. Here we hold out a test set that is
touched exactly once, at the end.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet statistics -- the pretrained backbones were trained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitSpec:
    """Fractions for the three-way split. Must sum to 1.0."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def discover_samples(root: str | Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Walk ``root`` and return (path, class_name) pairs plus the sorted class list.

    Non-image files are filtered out by extension rather than being handed to the
    decoder and silently dropped, so the counts you see are the counts you get.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Download PlantVillage and extract it so that each class is a subfolder."
        )

    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class subdirectories found under {root}")

    samples: list[tuple[Path, str]] = []
    skipped = 0
    for class_name in class_names:
        for path in sorted((root / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name))
            elif path.is_file():
                skipped += 1

    if skipped:
        print(f"[data] Skipped {skipped} non-image file(s) by extension.")
    if not samples:
        raise ValueError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} under {root}")

    return samples, class_names


def make_splits(
    samples: Sequence[tuple[Path, str]],
    spec: SplitSpec = SplitSpec(),
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Stratified three-way split: every class keeps its proportions in all three sets."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path, class_name in samples:
        by_class[class_name].append(path)

    rng = random.Random(seed)
    out: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}

    for class_name in sorted(by_class):
        paths = sorted(by_class[class_name])
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * spec.train))
        n_val = int(round(n * spec.val))
        # Give the remainder to test so the three parts always re-sum to n.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        chunks = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split, chunk in chunks.items():
            out[split].extend((str(p), class_name) for p in chunk)

    return out


def load_or_create_splits(
    root: str | Path,
    splits_path: str | Path = "artifacts/splits.json",
    spec: SplitSpec = SplitSpec(),
    seed: int = 42,
    force: bool = False,
) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """Reuse an existing split file if present, otherwise create and persist one."""
    splits_path = Path(splits_path)

    if splits_path.exists() and not force:
        payload = json.loads(splits_path.read_text())
        splits = {k: [tuple(x) for x in v] for k, v in payload["splits"].items()}
        print(f"[data] Reusing splits from {splits_path}")
        return splits, payload["class_names"]

    samples, class_names = discover_samples(root)
    splits = make_splits(samples, spec=spec, seed=seed)

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "seed": seed,
                "spec": asdict(spec),
                "class_names": class_names,
                "counts": {k: len(v) for k, v in splits.items()},
                "splits": splits,
            },
            indent=2,
        )
    )
    print(f"[data] Wrote splits to {splits_path}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return splits, class_names


class RandomizeBackground:
    """Replace a near-uniform background with random colour or noise.

    PlantVillage leaves sit on a flat grey studio backdrop. A network can reach
    very high accuracy by reading that backdrop and the leaf silhouette without
    ever looking at the lesion, and the resulting model collapses on field
    photographs. Applying this during training forces the decision onto leaf
    texture; applying it at evaluation time measures how much of the score was
    resting on the backdrop.
    """

    def __init__(self, p: float = 0.5, saturation_threshold: int = 40, mode: str = "mixed"):
        self.p = p
        self.saturation_threshold = saturation_threshold
        self.mode = mode

    @staticmethod
    def leaf_mask(rgb: np.ndarray, saturation_threshold: int = 40) -> np.ndarray:
        """Boolean mask of the leaf, via HSV saturation + largest connected component."""
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = (hsv[..., 1] > saturation_threshold).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            # No foreground found -- treat the whole frame as leaf rather than
            # returning an empty mask that would blank the image out.
            return np.ones(rgb.shape[:2], dtype=bool)

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        rgb = np.array(img.convert("RGB"))
        mask = self.leaf_mask(rgb, self.saturation_threshold)

        mode = self.mode
        if mode == "mixed":
            mode = random.choice(["colour", "noise"])

        if mode == "colour":
            background = np.full_like(rgb, 0)
            background[:] = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        else:
            background = np.random.randint(0, 256, rgb.shape, dtype=np.uint8)

        out = np.where(mask[..., None], rgb, background)
        return Image.fromarray(out.astype(np.uint8))


def build_transforms(image_size: int = 224, train: bool = False, randomize_background: float = 0.0):
    """Training augmentation is deliberately mild on colour.

    Hue and saturation are the disease signal here, so aggressive colour jitter
    would destroy the very thing being classified.
    """
    if train:
        ops: list[Callable] = []
        if randomize_background > 0:
            ops.append(RandomizeBackground(p=randomize_background))
        ops += [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25, fill=0),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.12, hue=0.03),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
        return T.Compose(ops)

    ops = []
    if randomize_background > 0:
        ops.append(RandomizeBackground(p=randomize_background))
    ops += [
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


class LeafDataset(Dataset):
    """Reads (path, class_name) pairs; maps class names to indices via ``class_names``."""

    def __init__(
        self,
        items: Sequence[tuple[str, str]],
        class_names: Sequence[str],
        transform: Callable | None = None,
    ):
        self.items = list(items)
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    @property
    def targets(self) -> list[int]:
        return [self.class_to_idx[c] for _, c in self.items]

    def __getitem__(self, index: int):
        path, class_name = self.items[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # corrupt file mid-epoch should not kill training
            print(f"[data] Failed to read {path}: {exc}. Substituting a blank image.")
            img = Image.new("RGB", (256, 256), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[class_name]


def class_weights(targets: Sequence[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is stable."""
    counts = np.bincount(np.asarray(targets), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(
    dataset: LeafDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    balanced_sampling: bool = False,
) -> DataLoader:
    sampler = None
    if balanced_sampling:
        targets = np.asarray(dataset.targets)
        counts = np.bincount(targets, minlength=len(dataset.class_names)).astype(np.float64)
        counts[counts == 0] = 1.0
        sample_weights = (1.0 / counts)[targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
