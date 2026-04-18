from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def select_device(device_arg: str = "auto") -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_curves(history: list[dict[str, float]], out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="IoU")
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Dice")
    axes[1].set_title("Validation")
    axes[1].legend()
    axes[2].plot(epochs, [row["val_map"] for row in history], label="mAP")
    axes[2].plot(epochs, [row["val_pixel_accuracy"] for row in history], label="pixel acc")
    axes[2].set_title("Validation")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def tensor_from_pil(image: Image.Image, image_size: int | tuple[int, int]) -> torch.Tensor:
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def connected_components(binary_mask: np.ndarray) -> np.ndarray:
    mask = binary_mask.astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels


LIGHT_COLORS = np.array(
    [
        [180, 235, 255],
        [255, 210, 180],
        [210, 255, 205],
        [245, 210, 255],
        [255, 245, 180],
        [200, 230, 255],
    ],
    dtype=np.float32,
)


def overlay_instances(image: Image.Image, mask: np.ndarray, alpha: float = 0.42, min_area: int = 16) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    labels = connected_components(mask > 0)
    overlay = base.copy()
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if int(region.sum()) < min_area:
            continue
        color = LIGHT_COLORS[(label_id - 1) % len(LIGHT_COLORS)]
        overlay[region] = (1.0 - alpha) * base[region] + alpha * color
    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for label_id in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size < min_area:
            continue
        draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline=(220, 255, 255), width=2)
    return result


def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int | tuple[int, int],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    original_size = image.size
    x = tensor_from_pil(image, image_size).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
    prob_img = Image.fromarray((prob * 255).clip(0, 255).astype(np.uint8)).resize(original_size, Image.Resampling.BILINEAR)
    prob_resized = np.asarray(prob_img, dtype=np.float32) / 255.0
    mask = (prob_resized >= threshold).astype(np.uint8)
    return mask, prob_resized


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    image_size: int,
    epoch: int,
    metrics: dict[str, float],
    args: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "num_classes": 2,
            "image_size": image_size,
            "epoch": epoch,
            "metrics": metrics,
            "args": args,
        },
        path,
    )


def load_checkpoint_model(path: str | Path, device: torch.device, model_name_override: str | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    from v2.models import create_model

    checkpoint = torch.load(path, map_location=device)
    model_name = model_name_override or checkpoint.get("model_name", "unet")
    model = create_model(model_name, num_classes=int(checkpoint.get("num_classes", 2)), pretrained=False).to(device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}
