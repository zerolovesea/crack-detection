import os
import json
import torch
import numpy as np

from PIL import Image, ImageDraw
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from torch.utils.data import Dataset

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None



class CrackDataset(Dataset):
    """
    假设：
      - img_dir 里存 RGB 图片（jpg/png 均可）
      - mask_dir 里存 同名（同 basename）的二值 mask（后缀可以不一样，比如 .png）
      - 如果 img_dir 下有子目录，则每个子目录代表一个类别；mask_dir 也需有同名子目录
        class_id 从 1 开始，0 预留给背景
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        class_map: dict = None
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.class_map = class_map
        self.class_names = []
        self.paired = []

        img_subdirs = [
            d for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d))
        ]

        if img_subdirs:
            img_subdirs.sort()
            if self.class_map is None:
                self.class_map = {name: idx + 1 for idx, name in enumerate(img_subdirs)}
            self.class_names = [k for k, _ in sorted(self.class_map.items(), key=lambda x: x[1])]

            for class_name in img_subdirs:
                if class_name not in self.class_map:
                    continue
                img_dir_c = os.path.join(img_dir, class_name)
                mask_dir_c = os.path.join(mask_dir, class_name)
                if not os.path.isdir(mask_dir_c):
                    print(f"[警告] 找不到 mask 子目录: {mask_dir_c}，跳过类别 {class_name}")
                    continue

                # 该类别下的所有图片
                all_imgs = [
                    f for f in os.listdir(img_dir_c)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                all_imgs.sort()

                # 该类别下的 mask 索引
                mask_map = {}
                for f in os.listdir(mask_dir_c):
                    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    stem = os.path.splitext(f)[0]
                    mask_map[stem] = os.path.join(mask_dir_c, f)

                for f in all_imgs:
                    img_path = os.path.join(img_dir_c, f)
                    stem = os.path.splitext(f)[0]
                    if stem in mask_map:
                        self.paired.append((img_path, mask_map[stem], self.class_map[class_name]))
                    else:
                        print(f"[警告] 找不到 {class_name}/{stem} 的 mask，跳过这张图片")
        else:
            # 平铺目录：只支持单一前景类别
            if self.class_map is None:
                self.class_map = {"foreground": 1}
            self.class_names = [k for k, _ in sorted(self.class_map.items(), key=lambda x: x[1])]

            all_imgs = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            all_imgs.sort()
            img_paths = [os.path.join(img_dir, f) for f in all_imgs]
            assert len(img_paths) > 0, "img_dir 里没有找到图片"

            mask_map = {}
            for f in os.listdir(mask_dir):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = os.path.splitext(f)[0]
                mask_map[stem] = os.path.join(mask_dir, f)

            for img_path in img_paths:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                if stem in mask_map:
                    self.paired.append((img_path, mask_map[stem], 1))
                else:
                    print(f"[警告] 找不到 {stem} 的 mask，跳过这张图片")

        assert len(self.paired) > 0, "没有任何 img-mask 配对样本，请检查文件名是否一致"
        self.num_classes = len(self.class_map) + 1

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_path, mask_path, class_id = self.paired[idx]

        img = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("L").resize(self.img_size)

        img = np.array(img, dtype=np.float32) / 255.0
        # 假设 mask：白=前景，黑=背景
        mask = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)
        mask = mask * np.uint8(class_id)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_path_maybe(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(_repo_root() / path)


def _find_result_jsons(data_poly_dir: str) -> list[str]:
    result = []
    for root, _, files in os.walk(data_poly_dir):
        for f in files:
            if f == "result.json":
                result.append(os.path.join(root, f))
    result.sort()
    return result


def _normalize_polygons(segmentation: Any) -> List[List[float]]:
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []
    if isinstance(segmentation[0], (int, float)):
        return [cast(List[float], segmentation)]
    return [cast(List[float], p) for p in segmentation if isinstance(p, list) and len(p) >= 6]


def _polygons_to_mask(
    polygons: List[List[float]],
    height: int,
    width: int,
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    if cv2 is not None:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(
                [[float(poly[i]) * sx, float(poly[i + 1]) * sy] for i in range(0, len(poly), 2)],
                dtype=np.float32,
            )
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], color=(1,))
        return mask

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        if not poly or len(poly) < 6:
            continue
        pts = [
            (float(poly[i]) * sx, float(poly[i + 1]) * sy)
            for i in range(0, len(poly), 2)
        ]
        draw.polygon(pts, outline=1, fill=1)
    return (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)


class CocoPolygonSegDataset(Dataset):
    """
    读取 data_poly/**/result.json，在线把多边形标注合成为语义分割 mask。
    """

    def __init__(
        self,
        data_poly_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        category_name_allowlist: Optional[List[str]] = None,
    ):
        self.data_poly_dir = data_poly_dir
        self.img_size = img_size

        result_jsons = _find_result_jsons(data_poly_dir)
        if not result_jsons:
            raise FileNotFoundError(f"No result.json found under: {data_poly_dir}")

        self.catname_to_label: Dict[str, int] = {}
        self.label_to_catname: Dict[int, str] = {}

        self.images_by_id: Dict[str, Dict[str, Any]] = {}
        self.ann_by_image: Dict[str, List[dict]] = {}

        for p in result_jsons:
            data = json.loads(Path(p).read_text(encoding="utf-8"))

            categories = {}
            for cat in data.get("categories", []):
                name = str(cat.get("name", cat.get("id")))
                categories[int(cat["id"])] = name
                if category_name_allowlist is not None and name not in category_name_allowlist:
                    continue
                if name not in self.catname_to_label:
                    new_label = len(self.catname_to_label) + 1
                    self.catname_to_label[name] = new_label
                    self.label_to_catname[new_label] = name

            images = {}
            for im in data.get("images", []):
                images[int(im["id"])] = {
                    "file_path": _abs_path_maybe(im["file_name"]),
                    "width": int(im["width"]),
                    "height": int(im["height"]),
                }

            for ann in data.get("annotations", []):
                image_id = int(ann["image_id"])
                cat_id = int(ann["category_id"])
                cat_name = categories.get(cat_id, str(cat_id))

                if category_name_allowlist is not None and cat_name not in category_name_allowlist:
                    continue
                if cat_name not in self.catname_to_label:
                    new_label = len(self.catname_to_label) + 1
                    self.catname_to_label[cat_name] = new_label
                    self.label_to_catname[new_label] = cat_name

                im = images.get(image_id)
                if im is None:
                    continue

                img_path = im["file_path"]
                self.images_by_id[img_path] = im
                self.ann_by_image.setdefault(img_path, []).append(
                    {"category_name": cat_name, "segmentation": ann.get("segmentation", [])}
                )

        self.samples = sorted(self.images_by_id.keys())
        if not self.samples:
            raise RuntimeError("No valid samples found after loading annotations.")

        self.class_names = [self.label_to_catname[k] for k in sorted(self.label_to_catname.keys())]
        self.num_classes = len(self.catname_to_label) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        meta = self.images_by_id[img_path]
        anns = self.ann_by_image.get(img_path, [])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        meta_w = int(meta["width"])
        meta_h = int(meta["height"])
        if (orig_w, orig_h) != (meta_w, meta_h):
            meta_w, meta_h = orig_w, orig_h

        target_w, target_h = self.img_size
        sx = target_w / float(meta_w)
        sy = target_h / float(meta_h)

        if (orig_w, orig_h) != (target_w, target_h):
            resampling = cast(Any, getattr(Image, "Resampling", Image))
            img = img.resize((target_w, target_h), resample=resampling.BILINEAR)

        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        for ann in anns:
            polygons = _normalize_polygons(ann.get("segmentation", []))
            if not polygons:
                continue
            label = int(self.catname_to_label[ann["category_name"]])
            poly_mask = _polygons_to_mask(polygons, height=target_h, width=target_w, sx=sx, sy=sy)
            mask[poly_mask > 0] = label

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t
