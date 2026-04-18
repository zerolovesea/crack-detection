from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2.dataset import CrackSegmentationDataset
from v2.metrics import SegmentationMetrics, segmentation_loss
from v2.train import collate_batch
from v2.utils import load_checkpoint_model, repo_root, save_json, select_device, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained crack segmentation checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint from v2/train.py.")
    parser.add_argument("--image-dir", default="data/data-202604/images")
    parser.add_argument("--mask-dir", default="data/data-202604/masks")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--model", default=None, help="Override checkpoint model name if needed.")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--foreground-weight", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint_model(args.weights, device, args.model)
    image_size = int(args.image_size or checkpoint.get("image_size", 512))

    dataset = CrackSegmentationDataset(args.image_dir, args.mask_dir, image_size=image_size, require_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    ce_weight = torch.tensor([1.0, args.foreground_weight], device=device)
    metrics = SegmentationMetrics()
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for images, masks, _ in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = segmentation_loss(logits, masks, ce_weight=ce_weight)
            total_loss += float(loss.item()) * images.shape[0]
            metrics.update(logits, masks)

    result = metrics.compute()
    result["loss"] = total_loss / max(len(dataset), 1)
    result["num_samples"] = float(len(dataset))

    out_dir = repo_root() / args.output_root / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "validation_metrics.json", result)
    print(f"Validation metrics: {result}")
    print(f"Saved to: {out_dir / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
