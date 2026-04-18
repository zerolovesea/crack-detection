from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2.dataset import CrackSegmentationDataset
from v2.metrics import SegmentationMetrics, segmentation_loss
from v2.models import AVAILABLE_MODELS, create_model
from v2.utils import (
    append_metrics_csv,
    overlay_instances,
    plot_curves,
    predict_mask,
    repo_root,
    save_checkpoint,
    save_json,
    select_device,
    timestamp,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = torch.stack([item["image"] for item in batch if isinstance(item["image"], torch.Tensor)])
    masks = torch.stack([item["mask"] for item in batch if isinstance(item["mask"], torch.Tensor)])
    paths = [str(item["path"]) for item in batch]
    return images, masks, paths


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    foreground_weight: float = 3.0,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    metrics = SegmentationMetrics()
    ce_weight = torch.tensor([1.0, foreground_weight], device=device)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks, _ in tqdm(loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, masks, ce_weight=ce_weight)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * images.shape[0]
            metrics.update(logits.detach(), masks.detach())

    result = metrics.compute()
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, result


def make_sample_predictions(
    model: torch.nn.Module,
    image_paths: list[str],
    out_dir: Path,
    device: torch.device,
    image_size: int,
    threshold: float,
    max_samples: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths[:max_samples]:
        image = Image.open(image_path).convert("RGB")
        mask, prob = predict_mask(model, image, device, image_size=image_size, threshold=threshold)
        overlay = overlay_instances(image, mask)
        stem = Path(image_path).stem
        overlay.save(out_dir / f"{stem}_overlay.png")
        Image.fromarray((prob * 255).astype(np.uint8)).save(out_dir / f"{stem}_prob.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crack semantic segmentation models.")
    parser.add_argument("--image-dir", default="data/data-202604/images", help="Training image directory.")
    parser.add_argument("--mask-dir", default="data/data-202604/masks", help="Mask directory. Supports 0001-0.png style multi masks.")
    parser.add_argument("--run-root", default="runs", help="Experiment output root.")
    parser.add_argument("--model", default="unet", help="Model name. Try unet, fcn_resnet50, deeplabv3_resnet50, smp:Unet:resnet50, hf:<model_id>.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--foreground-weight", type=float, default=3.0)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--list-models", action="store_true", help="Print supported model registry and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        for item in AVAILABLE_MODELS:
            print(f"{item.name}: {item.description}")
        return

    seed_everything(args.seed)
    root = repo_root()
    run_dir = root / args.run_root / timestamp()
    weights_dir = run_dir / "weights"
    samples_dir = run_dir / "samples"
    weights_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", vars(args))
    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    full_dataset = CrackSegmentationDataset(args.image_dir, args.mask_dir, image_size=args.image_size, require_masks=True)
    val_size = max(1, int(len(full_dataset) * args.val_ratio)) if len(full_dataset) > 1 else 1
    val_size = min(val_size, len(full_dataset) - 1) if len(full_dataset) > 1 else 1
    train_size = len(full_dataset) - val_size if len(full_dataset) > 1 else len(full_dataset)
    if len(full_dataset) > 1:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = full_dataset
        val_dataset = full_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = create_model(args.model, num_classes=2, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_iou = -1.0
    history: list[dict[str, float]] = []
    metrics_csv = run_dir / "metrics.csv"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer, foreground_weight=args.foreground_weight
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, device, optimizer=None, foreground_weight=args.foreground_weight
        )
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss,
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_map": train_metrics["map"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_loss": val_loss,
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_map": val_metrics["map"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(row)
        append_metrics_csv(metrics_csv, row)

        save_checkpoint(last_path, model, args.model, args.image_size, epoch, val_metrics, vars(args))
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(best_path, model, args.model, args.image_size, epoch, val_metrics, vars(args))

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_mAP={val_metrics['map']:.4f}"
        )

    plot_curves(history, run_dir / "training_curves.png")

    val_image_paths = []
    if hasattr(val_dataset, "indices"):
        val_image_paths = [str(full_dataset.images[i]) for i in val_dataset.indices]  # type: ignore[attr-defined]
    else:
        val_image_paths = [str(p) for p in full_dataset.images]
    make_sample_predictions(model, val_image_paths, samples_dir, device, args.image_size, args.threshold, args.sample_count)

    report = {
        "run_dir": str(run_dir),
        "best_weight": str(best_path),
        "last_weight": str(last_path),
        "num_samples": len(full_dataset),
        "train_samples": train_size,
        "val_samples": val_size,
        "best_val_iou": best_iou,
        "final_metrics": history[-1] if history else {},
    }
    save_json(run_dir / "report.json", report)
    print(f"Training finished. Report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
