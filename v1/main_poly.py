import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import functional as F

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class CocoImageRecord:
    file_path: str
    width: int
    height: int


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs_path_maybe(path: str) -> str:
    # The exported json uses workspace-relative paths like: data_poly/.../images/xxx.png
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def find_result_jsons(data_poly_dir: str) -> List[str]:
    result = []
    for root, _, files in os.walk(data_poly_dir):
        for f in files:
            if f == "result.json":
                result.append(os.path.join(root, f))
    result.sort()
    return result


def load_coco_like(result_json_path: str) -> Tuple[Dict[int, CocoImageRecord], List[dict], Dict[int, str]]:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id: Dict[int, CocoImageRecord] = {}
    for im in data.get("images", []):
        images_by_id[int(im["id"])] = CocoImageRecord(
            file_path=_abs_path_maybe(im["file_name"]),
            width=int(im["width"]),
            height=int(im["height"]),
        )

    categories = {}
    for cat in data.get("categories", []):
        categories[int(cat["id"])] = str(cat.get("name", cat["id"]))

    annotations = data.get("annotations", [])
    return images_by_id, annotations, categories


def polygons_to_mask(
    polygons: List[List[float]],
    height: int,
    width: int,
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    """Rasterize polygons into a binary mask.

    Note: We draw directly at the requested (width, height) and optionally scale
    source coordinates by (sx, sy). This avoids drawing at full-res then resizing.
    """

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
            pts_i = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts_i], color=(1,))
        return mask

    # Fallback: PIL
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


def mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return [x0, y0, x1, y1]


def normalize_polygons(segmentation: Any) -> List[List[float]]:
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []
    if isinstance(segmentation[0], (int, float)):
        return [cast(List[float], segmentation)]
    return [cast(List[float], p) for p in segmentation if isinstance(p, list) and len(p) >= 6]


class CocoPolygonInstanceDataset(Dataset):
    def __init__(
        self,
        data_poly_dir: str,
        img_size: Optional[int] = 512,
        category_name_allowlist: Optional[List[str]] = None,
        min_area: int = 10,
    ):
        self.data_poly_dir = data_poly_dir
        self.img_size = img_size
        self.min_area = min_area

        # Merge all projects
        result_jsons = find_result_jsons(data_poly_dir)
        if not result_jsons:
            raise FileNotFoundError(f"No result.json found under: {data_poly_dir}")

        # Build category name -> label_id (1..K), 0 is background
        self.catname_to_label: Dict[str, int] = {}
        self.label_to_catname: Dict[int, str] = {}

        # Collect per-image annotations
        tmp_ann_by_path: Dict[str, List[dict]] = {}
        tmp_im_meta_by_path: Dict[str, CocoImageRecord] = {}

        for p in result_jsons:
            images_by_id, annotations, categories = load_coco_like(p)

            # categories in this json: id -> name
            for _, name in categories.items():
                if category_name_allowlist is not None and name not in category_name_allowlist:
                    continue
                if name not in self.catname_to_label:
                    new_label = len(self.catname_to_label) + 1
                    self.catname_to_label[name] = new_label
                    self.label_to_catname[new_label] = name

            for ann in annotations:
                image_id = int(ann["image_id"])
                cat_id = int(ann["category_id"])
                cat_name = categories.get(cat_id, str(cat_id))

                if category_name_allowlist is not None and cat_name not in category_name_allowlist:
                    continue
                if cat_name not in self.catname_to_label:
                    # might happen if allowlist is None but categories missing
                    new_label = len(self.catname_to_label) + 1
                    self.catname_to_label[cat_name] = new_label
                    self.label_to_catname[new_label] = cat_name

                im = images_by_id.get(image_id)
                if im is None:
                    continue

                tmp_im_meta_by_path[im.file_path] = im
                tmp_ann_by_path.setdefault(im.file_path, []).append({
                    "category_name": cat_name,
                    "segmentation": ann.get("segmentation", []),
                })

        self.samples: List[str] = sorted(tmp_im_meta_by_path.keys())
        self.im_meta_by_path = tmp_im_meta_by_path
        self.ann_by_path = tmp_ann_by_path

        if not self.samples:
            raise RuntimeError("No valid samples found after loading annotations.")

        self.num_classes = len(self.catname_to_label) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        meta = self.im_meta_by_path[img_path]
        anns = self.ann_by_path.get(img_path, [])

        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        # Trust actual image size; meta is only for reference.
        meta_w = int(meta.width)
        meta_h = int(meta.height)
        if (orig_w, orig_h) != (meta_w, meta_h):
            meta_w, meta_h = orig_w, orig_h

        target_w, target_h = orig_w, orig_h
        if self.img_size is not None:
            target_h = int(self.img_size)
            target_w = int(self.img_size)

        sx = target_w / float(meta_w)
        sy = target_h / float(meta_h)

        if (orig_w, orig_h) != (target_w, target_h):
            resampling = cast(Any, getattr(Image, "Resampling", Image))
            img = img.resize((target_w, target_h), resample=resampling.BILINEAR)

        # Build instance masks
        inst_masks: List[np.ndarray] = []
        labels: List[int] = []
        boxes: List[List[float]] = []
        areas: List[float] = []

        for ann in anns:
            polygons = normalize_polygons(ann.get("segmentation", []))
            if len(polygons) == 0:
                continue

            mask = polygons_to_mask(polygons, height=target_h, width=target_w, sx=sx, sy=sy)
            area = float(mask.sum())
            if area < float(self.min_area):
                continue

            box = mask_to_box(mask)
            if box is None:
                continue

            cat_name = ann["category_name"]
            label = int(self.catname_to_label[cat_name])

            inst_masks.append(mask)
            labels.append(label)
            boxes.append(box)
            areas.append(area)

        # If an image has no instances, still return a valid empty target
        if len(inst_masks) == 0:
            masks_t = torch.zeros((0, target_h, target_w), dtype=torch.uint8)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
        else:
            masks_t = torch.from_numpy(np.stack(inst_masks, axis=0)).to(torch.uint8)
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)

        img_t = F.to_tensor(img)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64),
        }

        return img_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(num_classes: int, pretrained: bool = True):
    # num_classes includes background
    weights = None
    weights_backbone = None
    if pretrained:
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            weights_backbone = None
        except Exception:
            weights = None
            weights_backbone = None

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
    )

    m = cast(Any, model)
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    m.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


@torch.no_grad()
def evaluate_instance_metrics(
    model,
    loader,
    device,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_sum = 0.0
    iou_cnt = 0

    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_masks = tgt["masks"].numpy().astype(bool)

            pred_scores = out["scores"].detach().cpu().numpy() if len(out.get("scores", [])) else np.array([])
            pred_masks = out["masks"].detach().cpu().numpy() if len(out.get("masks", [])) else np.zeros((0, 1, 1, 1))
            if pred_masks.ndim == 4:
                pred_masks = pred_masks[:, 0]
            pred_masks = pred_masks > 0.5

            keep = pred_scores >= float(score_thresh)
            pred_masks = pred_masks[keep]

            if gt_masks.shape[0] == 0:
                total_fp += int(pred_masks.shape[0])
                continue

            if pred_masks.shape[0] == 0:
                total_fn += int(gt_masks.shape[0])
                continue

            # IoU matrix
            iou_mat = np.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=np.float32)
            for i in range(pred_masks.shape[0]):
                pm = pred_masks[i]
                p_area = pm.sum()
                if p_area == 0:
                    continue
                for j in range(gt_masks.shape[0]):
                    gm = gt_masks[j]
                    inter = np.logical_and(pm, gm).sum()
                    union = np.logical_or(pm, gm).sum()
                    if union > 0:
                        iou_mat[i, j] = inter / float(union)

            matched_gt = set()
            matched_pred = set()

            # Greedy matching by best IoU
            while True:
                if iou_mat.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best = float(iou_mat[i, j])
                if best < float(iou_thresh):
                    break
                if j in matched_gt or i in matched_pred:
                    iou_mat[i, j] = 0.0
                    continue
                matched_gt.add(j)
                matched_pred.add(i)
                total_tp += 1
                iou_sum += best
                iou_cnt += 1
                iou_mat[i, :] = 0.0
                iou_mat[:, j] = 0.0

            total_fp += int(pred_masks.shape[0] - len(matched_pred))
            total_fn += int(gt_masks.shape[0] - len(matched_gt))

    precision = total_tp / float(total_tp + total_fp + 1e-9)
    recall = total_tp / float(total_tp + total_fn + 1e-9)
    f1 = (2 * precision * recall) / float(precision + recall + 1e-9)
    mean_iou = iou_sum / float(iou_cnt + 1e-9)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }


def plot_loss_curves(history: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = [h["epoch"] for h in history]
    total = [h["loss_total"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total, label="loss_total", linewidth=2)

    keys = ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]
    for k in keys:
        if k in history[0]:
            plt.plot(epochs, [h.get(k, 0.0) for h in history], label=k, alpha=0.9)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mask R-CNN Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def save_metrics_csv(history: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not history:
        return

    keys = list(history[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for h in history:
            row = []
            for k in keys:
                v = h.get(k, "")
                if isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")


def visualize_inference(model, dataset: Dataset, device, out_path: str, score_thresh: float = 0.5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds = cast(Sized, dataset)
    idx = random.randint(0, len(ds) - 1)
    img_t, _ = dataset[idx]

    model.eval()
    with torch.no_grad():
        out = model([img_t.to(device)])[0]

    img = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    scores = out["scores"].detach().cpu().numpy() if len(out.get("scores", [])) else np.array([])
    masks = out["masks"].detach().cpu().numpy() if len(out.get("masks", [])) else np.zeros((0, 1, 1, 1))
    boxes = out["boxes"].detach().cpu().numpy() if len(out.get("boxes", [])) else np.zeros((0, 4))

    if masks.ndim == 4:
        masks = masks[:, 0]

    keep = scores >= float(score_thresh)
    masks = masks[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    # Draw overlays using numpy (no cv2 dependency here)
    overlay = img.astype(np.float32)
    color = np.array([0, 255, 0], dtype=np.float32)
    alpha = 0.25

    for m in masks:
        mm = (m > 0.5)
        overlay[mm] = (1 - alpha) * overlay[mm] + alpha * color

    overlay = overlay.clip(0, 255).astype(np.uint8)

    # Draw boxes via PIL
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    for b, s in zip(boxes, scores):
        x0, y0, x1, y1 = [float(x) for x in b]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((x0 + 2, y0 + 2), f"{s:.2f}", fill=(255, 0, 0))

    pil.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Train instance segmentation (Mask R-CNN) from data_poly COCO polygon exports")
    parser.add_argument("--data_poly_dir", type=str, default="data_poly", help="Root dir containing project-*/result.json")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save checkpoints and figures")
    parser.add_argument("--img_size", type=int, default=512, help="Resize images/masks to square size for training")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP (if CUDA available)")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for metrics/visualization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print("Using device:", device)

    # A small PyTorch perf hint (has effect mainly on CUDA / newer backends)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dataset = CocoPolygonInstanceDataset(
        data_poly_dir=args.data_poly_dir,
        img_size=args.img_size,
    )

    print(f"Loaded {len(dataset)} images")
    print(f"Classes (incl. background): {dataset.num_classes}")
    for lid in range(1, dataset.num_classes):
        print(f"  {lid}: {dataset.label_to_catname.get(lid, str(lid))}")

    if len(dataset) < 2:
        train_ds = dataset
        val_ds = dataset
    else:
        val_size = int(len(dataset) * float(args.val_ratio))
        if val_size < 1:
            val_size = 1
        if val_size >= len(dataset):
            val_size = len(dataset) - 1
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=dataset.num_classes, pretrained=args.pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr))

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, "maskrcnn_poly_best.pth")
    last_ckpt = os.path.join(args.output_dir, "maskrcnn_poly_last.pth")

    history: List[dict] = []
    best_f1 = -1.0

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_sums = {
            "loss_total": 0.0,
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_mask": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
        }

        t0 = time.time()
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values(), torch.tensor(0.0, device=device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_sums["loss_total"] += float(loss.item())
            for k in epoch_sums.keys():
                if k == "loss_total":
                    continue
                if k in loss_dict:
                    epoch_sums[k] += float(loss_dict[k].item())

        steps = max(1, len(train_loader))
        epoch_avg = {k: v / float(steps) for k, v in epoch_sums.items()}

        metrics = evaluate_instance_metrics(
            model,
            val_loader,
            device=device,
            score_thresh=float(args.score_thresh),
            iou_thresh=0.5,
        )

        row = {
            "epoch": epoch,
            **epoch_avg,
            "val_precision@0.5": metrics["precision"],
            "val_recall@0.5": metrics["recall"],
            "val_f1@0.5": metrics["f1"],
            "val_mean_iou@0.5": metrics["mean_iou"],
            "val_tp": metrics["tp"],
            "val_fp": metrics["fp"],
            "val_fn": metrics["fn"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={row['loss_total']:.4f} "
            f"(cls={row['loss_classifier']:.4f}, box={row['loss_box_reg']:.4f}, mask={row['loss_mask']:.4f}) "
            f"val P/R/F1={row['val_precision@0.5']:.4f}/{row['val_recall@0.5']:.4f}/{row['val_f1@0.5']:.4f} "
            f"mIoU={row['val_mean_iou@0.5']:.4f} "
            f"epoch_time={time.time() - t0:.1f}s"
        )

        if row["val_f1@0.5"] > best_f1:
            best_f1 = row["val_f1@0.5"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  > New best checkpoint saved: {best_ckpt} (val_f1={best_f1:.4f})")

    torch.save(model.state_dict(), last_ckpt)
    print(f"Last checkpoint saved: {last_ckpt}")

    csv_path = os.path.join(args.output_dir, "maskrcnn_poly_metrics.csv")
    save_metrics_csv(history, csv_path)
    print(f"Metrics saved: {csv_path}")

    loss_png = os.path.join(args.output_dir, "maskrcnn_poly_loss.png")
    plot_loss_curves(history, loss_png)
    print(f"Loss plot saved: {loss_png}")

    # Inference visualization using best model
    model2 = build_model(num_classes=dataset.num_classes, pretrained=False).to(device)
    model2.load_state_dict(torch.load(best_ckpt, map_location=device))

    infer_png = os.path.join(args.output_dir, "maskrcnn_poly_infer_example.png")
    visualize_inference(model2, dataset, device, infer_png, score_thresh=float(args.score_thresh))
    print(f"Inference example saved: {infer_png}")


if __name__ == "__main__":
    main()
