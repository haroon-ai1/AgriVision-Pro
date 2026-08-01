"""Two-stage transfer-learning trainer.

Stage 1 freezes the pretrained trunk and trains only the new head. Stage 2
unfreezes and fine-tunes everything with a lower trunk learning rate.

Model selection uses macro-F1 on the validation split, never accuracy: with an
imbalanced dataset, accuracy is dominated by the largest classes and will
happily select a model that has given up on the rare ones. The test split is
not touched here at all -- see ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import (
    LeafDataset,
    SplitSpec,
    build_loader,
    build_transforms,
    class_weights,
    load_or_create_splits,
)
from .model import (
    ModelConfig,
    build_model,
    default_image_size,
    param_groups,
    save_checkpoint,
    set_backbone_frozen,
)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs: int,
    head_lr: float,
    backbone_lr: float,
    backbone: str,
    weight_decay: float,
    use_amp: bool,
    stage_name: str,
    best: dict,
    on_improve,
    patience: int,
    history: list,
):
    optimizer = torch.optim.AdamW(
        param_groups(model, backbone, head_lr, backbone_lr), weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        t0 = time.time()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * images.size(0)
            seen += images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()

        scheduler.step()
        val = evaluate_split(model, val_loader, criterion, device)
        record = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "seconds": round(time.time() - t0, 1),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(record)

        print(
            f"[{stage_name}] epoch {epoch:>2}/{epochs}  "
            f"train_loss={record['train_loss']:.4f}  train_acc={record['train_accuracy']:.4f}  "
            f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
            f"val_macroF1={val['macro_f1']:.4f}  ({record['seconds']}s)"
        )

        if val["macro_f1"] > best["macro_f1"] + 1e-5:
            best.update({"macro_f1": val["macro_f1"], "accuracy": val["accuracy"], "stage": stage_name, "epoch": epoch})
            on_improve(best)
            stale = 0
            print(f"    -> new best (macro-F1 {val['macro_f1']:.4f}), checkpoint saved")
        else:
            stale += 1
            if stale >= patience:
                print(f"    -> no improvement in {patience} epochs, stopping this stage early")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a CNN for plant disease classification.")
    parser.add_argument("--data-root", required=True, help="Folder containing one subfolder per class")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to the backbone's native size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3, help="Stage 1: frozen trunk")
    parser.add_argument("--finetune-epochs", type=int, default=12, help="Stage 2: full fine-tune")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random init instead of ImageNet weights. For offline smoke tests only -- "
        "accuracy will be far worse, since the whole point is the pretrained features.",
    )
    parser.add_argument(
        "--randomize-background",
        type=float,
        default=0.35,
        help="Probability of swapping the studio backdrop during training (0 disables).",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Oversample rare classes instead of only reweighting the loss.",
    )
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"[train] device={device}  amp={use_amp}")

    image_size = args.image_size or default_image_size(args.backbone)
    splits, class_names = load_or_create_splits(
        args.data_root,
        splits_path=out_dir / "splits.json",
        spec=SplitSpec(),
        seed=args.seed,
        force=args.force_resplit,
    )
    print(f"[train] {len(class_names)} classes, image_size={image_size}")

    train_ds = LeafDataset(
        splits["train"], class_names,
        build_transforms(image_size, train=True, randomize_background=args.randomize_background),
    )
    val_ds = LeafDataset(splits["val"], class_names, build_transforms(image_size, train=False))

    train_loader = build_loader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, balanced_sampling=args.balanced_sampling,
    )
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    cfg = ModelConfig(
        backbone=args.backbone,
        num_classes=len(class_names),
        image_size=image_size,
        dropout=args.dropout,
        class_names=class_names,
        pretrained=not args.no_pretrained,
    )
    model = build_model(cfg).to(device)

    weights = None if args.balanced_sampling else class_weights(train_ds.targets, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    ckpt_path = out_dir / "best_model.pt"
    history: list[dict] = []
    best = {"macro_f1": -1.0, "accuracy": 0.0, "stage": None, "epoch": -1}

    def on_improve(state):
        save_checkpoint(ckpt_path, model, cfg, metrics={"val": dict(state)},
                        extra={"args": vars(args), "train_size": len(train_ds)})

    print("\n=== Stage 1: training the head, trunk frozen ===")
    set_backbone_frozen(model, args.backbone, frozen=True)
    run_stage(model, train_loader, val_loader, criterion, device, args.head_epochs,
              args.head_lr, args.backbone_lr, args.backbone, args.weight_decay, use_amp,
              "head", best, on_improve, args.patience, history)

    print("\n=== Stage 2: fine-tuning the full network ===")
    set_backbone_frozen(model, args.backbone, frozen=False)
    run_stage(model, train_loader, val_loader, criterion, device, args.finetune_epochs,
              args.head_lr * 0.3, args.backbone_lr, args.backbone, args.weight_decay, use_amp,
              "finetune", best, on_improve, args.patience, history)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[train] Best val macro-F1 {best['macro_f1']:.4f} (stage={best['stage']}, epoch={best['epoch']})")
    print(f"[train] Checkpoint: {ckpt_path}")
    print(f"[train] Next:  python -m agrivision.evaluate --checkpoint {ckpt_path} --data-root {args.data_root}")


if __name__ == "__main__":
    main()
