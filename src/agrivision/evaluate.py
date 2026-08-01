"""Final evaluation on the held-out test split.

Beyond the usual report this runs a *background robustness probe*. PlantVillage
images all share a flat studio backdrop, so a model can score extremely well by
reading the backdrop and leaf outline rather than the lesion. Re-scoring the
same test set with randomised backgrounds shows how much of the headline number
survives contact with a different background -- which is the number that
actually predicts field performance.

A large gap between the two is the finding worth reporting, not a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from .data import LeafDataset, build_loader, build_transforms, load_or_create_splits
from .model import load_checkpoint
from .train import pick_device


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all, targets_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images), dim=1)
        probs_all.append(probs.cpu())
        targets_all.append(targets)

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    return probs.argmax(1), targets, probs


def expected_calibration_error(probs: np.ndarray, preds: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """How far the confidence scores are from real accuracy.

    The GUI shows a confidence percentage to the user. If the model is
    overconfident that number is actively misleading, so it is worth measuring.
    """
    confidence = probs.max(1)
    correct = (preds == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.mean()) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = data / row_sums

    n = len(class_names)
    size = max(7.0, n * 0.45)
    fig, ax = plt.subplots(figsize=(size, size * 0.88), dpi=140)
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1 if normalize else None)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))

    if n <= 20:
        for i in range(n):
            for j in range(n):
                value = data[i, j]
                if value > 0.005:
                    ax.text(j, i, f"{value:.2f}" if normalize else f"{int(value)}",
                            ha="center", va="center", fontsize=6,
                            color="white" if value < 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Wrote {path}")


def score(preds, targets, probs, class_names) -> dict:
    n_classes = len(class_names)
    out = {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(probs, preds, targets),
        "mean_confidence": float(probs.max(1).mean()),
    }
    # With exactly 3 classes top-3 is trivially 1.0, so it is not worth reporting.
    if n_classes > 3:
        out["top3_accuracy"] = float(
            top_k_accuracy_score(targets, probs, k=3, labels=list(range(n_classes)))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the held-out test split.")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="artifacts/splits.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip the randomised-background probe.",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    model, cfg, train_metrics = load_checkpoint(args.checkpoint, device)
    class_names = cfg.class_names

    splits, split_class_names = load_or_create_splits(args.data_root, splits_path=args.splits)
    if split_class_names != class_names:
        raise ValueError(
            "Class list in the checkpoint does not match the split file. "
            "The dataset changed since training -- retrain before evaluating."
        )

    items = splits[args.split]
    print(f"[eval] Scoring {len(items)} images from the '{args.split}' split.")

    # --- Clean evaluation -------------------------------------------------
    ds = LeafDataset(items, class_names, build_transforms(cfg.image_size, train=False))
    loader = build_loader(ds, args.batch_size, num_workers=args.num_workers)
    preds, targets, probs = collect_predictions(model, loader, device)

    clean = score(preds, targets, probs, class_names)
    print("\n=== Clean test set ===")
    for k, v in clean.items():
        print(f"  {k:<16} {v:.4f}")

    print("\n" + classification_report(targets, preds, target_names=class_names, zero_division=0, digits=4))

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Most-confused pairs are more actionable than the raw matrix for a README.
    confused = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused.append((int(cm[i, j]), class_names[i], class_names[j]))
    confused.sort(reverse=True)
    if confused:
        print("Most-confused pairs (true -> predicted):")
        for count, true_c, pred_c in confused[:8]:
            print(f"  {count:>5}  {true_c}  ->  {pred_c}")

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_images": len(items),
        "class_names": class_names,
        "clean": clean,
        "per_class": classification_report(
            targets, preds, target_names=class_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "val_metrics_at_training": train_metrics,
    }

    # --- Background robustness probe -------------------------------------
    if not args.skip_robustness:
        print("\n=== Randomised-background probe ===")
        print("Re-scoring the same images with the studio backdrop replaced.")
        rb_ds = LeafDataset(
            items, class_names,
            build_transforms(cfg.image_size, train=False, randomize_background=1.0),
        )
        rb_loader = build_loader(rb_ds, args.batch_size, num_workers=args.num_workers)
        rb_preds, rb_targets, rb_probs = collect_predictions(model, rb_loader, device)
        randomized = score(rb_preds, rb_targets, rb_probs, class_names)

        for k, v in randomized.items():
            delta = v - clean.get(k, 0.0)
            print(f"  {k:<16} {v:.4f}   ({delta:+.4f} vs clean)")

        drop = clean["accuracy"] - randomized["accuracy"]
        report["randomized_background"] = randomized
        report["background_dependence"] = float(drop)

        print(f"\n  Accuracy drop from background swap: {drop:.4f}")
        if drop > 0.15:
            print("  A drop this large means a substantial part of the score came from the")
            print("  backdrop rather than the leaf. Train with --randomize-background 0.5")
            print("  and expect the clean number to fall while field performance improves.")
        else:
            print("  The model is largely reading the leaf rather than the backdrop.")

    report_path = out_dir / f"eval_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] Wrote {report_path}")


if __name__ == "__main__":
    main()
