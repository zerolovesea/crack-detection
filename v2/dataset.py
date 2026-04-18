from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def mask_stem_to_image_stem(mask_stem: str) -> str:
    match = re.match(r"^(?P<stem>.+?)-\d+(?:\s+.*)?$", mask_stem)
    if match:
        return match.group("stem")
    return mask_stem


def build_mask_index(mask_dir: str | Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for mask_path in list_images(mask_dir):
        image_stem = mask_stem_to_image_stem(mask_path.stem)
        index.setdefault(image_stem, []).append(mask_path)
    return {k: sorted(v) for k, v in index.items()}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CrackSegmentationDataset(Dataset):
    """
    Binary semantic segmentation dataset.

    Multiple instance masks such as 0007-0.png, 0007-1.png are merged into one
    foreground mask for training and validation.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        image_size: int | tuple[int, int] = 512,
        require_masks: bool = True,
        transform: Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.require_masks = require_masks
        self.transform = transform

        self.images = list_images(self.image_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found under {self.image_dir}")

        self.mask_index = build_mask_index(self.mask_dir) if self.mask_dir else {}
        if self.require_masks:
            self.images = [p for p in self.images if p.stem in self.mask_index]
            if not self.images:
                raise FileNotFoundError(
                    f"No image/mask pairs found. image_dir={self.image_dir}, mask_dir={self.mask_dir}"
                )

    def __len__(self) -> int:
        return len(self.images)

    def _load_merged_mask(self, image_path: Path, original_size: tuple[int, int]) -> Image.Image:
        merged = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        for mask_path in self.mask_index.get(image_path.stem, []):
            mask = Image.open(mask_path).convert("L")
            if mask.size != original_size:
                mask = mask.resize(original_size, Image.Resampling.NEAREST)
            merged |= (np.asarray(mask) > 0).astype(np.uint8)
        return Image.fromarray(merged * 255, mode="L")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        mask = self._load_merged_mask(image_path, image.size)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        image_tensor = pil_to_tensor(image)
        mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.int64))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "path": str(image_path),
        }
