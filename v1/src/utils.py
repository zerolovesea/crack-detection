import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def compute_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for i in range(len(dataset)):
        _, mask = dataset[i]
        binc = torch.bincount(mask.view(-1).long(), minlength=num_classes).float()
        counts += binc

    total = counts.sum().item()
    weights = total / (counts + 1e-6)
    weights = weights / weights.mean()
    print("类别像素统计:", counts.tolist())
    print("类别权重:", weights.tolist())
    return weights.to(device)


def get_class_colors(num_classes):
    base = [
        (0, 0, 0),
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    if num_classes <= len(base):
        return base[:num_classes]
    colors = base[:]
    while len(colors) < num_classes:
        colors.append((128, 128, 128))
    return colors


def visualize_predictions(
    model,
    val_dataset: Dataset,
    device,
    img_size=(256, 256),
    out_path="outputs/fcn_vis_overlay_box.png",
    num_samples=5,
):
    """
    可视化：
      左：原图
      中：GT Mask (多类别)
      右：原图 + 预测叠加
    """
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = min(num_samples, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, n, dtype=int)

    plt.figure(figsize=(12, 4 * n))
    num_classes = getattr(val_dataset, "num_classes", None)
    if num_classes is None and hasattr(val_dataset, "dataset"):
        num_classes = getattr(val_dataset.dataset, "num_classes", 2)
    if num_classes is None:
        num_classes = 2
    colors = get_class_colors(num_classes)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 读取图像与标签
            img, gt_mask = val_dataset[idx]
            img_np = img.permute(1, 2, 0).numpy()  # HWC, float0-1
            gt_np = gt_mask.numpy()

            # ---------- 推理 ----------
            img_in = img.unsqueeze(0).to(device)
            logits = model(img_in)
            pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            # ---------- 绿色透明覆盖 ----------
            img_f = img_np.astype(np.float32)
            if img_f.max() > 1.5:
                img_f = img_f / 255.0

            alpha = 0.25
            overlay_f = img_f.copy()
            for cls_id in range(1, num_classes):
                mask = pred_np == cls_id
                if not mask.any():
                    continue
                color = np.array(colors[cls_id], dtype=np.float32) / 255.0
                overlay_f[mask] = (1 - alpha) * img_f[mask] + alpha * color

            # 转成uint8用于OpenCV绘制
            overlay = (overlay_f * 255).clip(0, 255).astype(np.uint8)

            # ---------- 画图 ----------
            row = i

            # 原图
            plt.subplot(n, 3, row * 3 + 1)
            plt.imshow(img_np)
            plt.title("Image")
            plt.axis("off")

            # GT Mask
            plt.subplot(n, 3, row * 3 + 2)
            gt_color = np.zeros_like(overlay)
            for cls_id in range(1, num_classes):
                mask = gt_np == cls_id
                if not mask.any():
                    continue
                gt_color[mask] = colors[cls_id]
            plt.imshow(gt_color)
            plt.title("Ground Truth (Color)")
            plt.axis("off")

            # 叠加预测 + Bounding Box
            plt.subplot(n, 3, row * 3 + 3)
            plt.imshow(overlay)
            plt.title("Overlay (Pred)")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("叠加 + 框选 可视化图已保存到:", out_path)
