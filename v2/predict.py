from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2.dataset import list_images
from v2.utils import load_checkpoint_model, overlay_instances, predict_mask, repo_root, save_json, select_device, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crack masks and pastel instance overlays.")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint from v2/train.py.")
    parser.add_argument("--input", required=True, help="Image file or directory to predict.")
    parser.add_argument("--output-root", default="output", help="Output root. Results go to output/{timestamp}.")
    parser.add_argument("--model", default=None, help="Override checkpoint model name if needed.")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-prob", action="store_true", help="Also save foreground probability maps.")
    parser.add_argument("--save-mask", action="store_true", help="Also save binary masks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint_model(args.weights, device, args.model)
    image_size = int(args.image_size or checkpoint.get("image_size", 512))

    image_paths = list_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No input images found: {args.input}")

    out_dir = repo_root() / args.output_root / timestamp()
    overlay_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"
    prob_dir = out_dir / "probabilities"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")
        mask, prob = predict_mask(model, image, device, image_size=image_size, threshold=args.threshold)
        overlay = overlay_instances(image, mask)

        rel_name = image_path.stem
        overlay_path = overlay_dir / f"{rel_name}_overlay.png"
        overlay.save(overlay_path)

        item = {"image": str(image_path), "overlay": str(overlay_path)}
        if args.save_mask:
            mask_path = mask_dir / f"{rel_name}_mask.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            item["mask"] = str(mask_path)
        if args.save_prob:
            prob_path = prob_dir / f"{rel_name}_prob.png"
            Image.fromarray((prob * 255).astype(np.uint8)).save(prob_path)
            item["probability"] = str(prob_path)
        manifest.append(item)

    save_json(
        out_dir / "manifest.json",
        {
            "weights": str(args.weights),
            "input": str(args.input),
            "threshold": args.threshold,
            "image_size": image_size,
            "count": len(manifest),
            "results": manifest,
        },
    )
    print(f"Saved {len(manifest)} predictions to: {out_dir}")


if __name__ == "__main__":
    main()
