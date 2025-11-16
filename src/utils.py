import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def compute_pos_weight(dataset, device):
    total_pos = 0.0
    total_neg = 0.0

    for i in range(len(dataset)):
        _, mask = dataset[i]
        pos = mask.sum().item()
        num = mask.numel()
        neg = num - pos
        total_pos += pos
        total_neg += neg

    pos_weight_value = total_neg / (total_pos + 1e-6)
    print(f"正像素总数: {total_pos:.0f}, 负像素总数: {total_neg:.0f}, pos_weight ≈ {pos_weight_value:.2f}")

    pos_weight = torch.tensor(pos_weight_value, device=device)
    return pos_weight


def visualize_predictions(
    model,
    val_dataset: Dataset,
    device,
    img_size=(256, 256),
    out_path="outputs/fcn_vis_overlay_box.png",
    num_samples=5,
    threshold=0.3,
):
    """
    可视化：
      左：原图
      中：GT Mask
      右：原图 + 透明绿色预测 + 红色Bounding Box
    """
    import cv2  # 需要 pip install opencv-python

    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = min(num_samples, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, n, dtype=int)

    plt.figure(figsize=(12, 4 * n))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 读取图像与标签
            img, gt_mask = val_dataset[idx]
            img_np = img.permute(1, 2, 0).numpy()  # HWC, float0-1
            gt_np = gt_mask.squeeze(0).numpy()

            # ---------- 推理 ----------
            img_in = img.unsqueeze(0).to(device)
            logits = model(img_in)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_np = (probs > threshold).astype(np.uint8)  # 0/1 mask

            # ---------- 绿色透明覆盖 ----------
            img_f = img_np.astype(np.float32)
            if img_f.max() > 1.5:
                img_f = img_f / 255.0

            green = np.zeros_like(img_f)
            green[:, :, 1] = 1.0  # 纯绿色

            alpha = 0.25
            overlay_f = img_f.copy()
            mask = pred_np > 0
            overlay_f[mask] = (1 - alpha) * img_f[mask] + alpha * green[mask]

            # 转成uint8用于OpenCV绘制
            overlay = (overlay_f * 255).clip(0, 255).astype(np.uint8)

            # ---------- 计算 bounding box ----------
            # pred_np 是 0/1，转成 0/255（OpenCV要求）
            pred_cv = (pred_np * 255).astype(np.uint8)

            # 找到所有裂缝的像素点
            ys, xs = np.where(pred_np == 1)

            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                overlay = cv2.rectangle(
                    overlay,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(0, 255, 0 ),  
                    thickness=2
                )

            # ---------- 画图 ----------
            row = i

            # 原图
            plt.subplot(n, 3, row * 3 + 1)
            plt.imshow(img_np)
            plt.title("Image")
            plt.axis("off")

            # GT Mask
            plt.subplot(n, 3, row * 3 + 2)
            plt.imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            plt.title("Ground Truth")
            plt.axis("off")

            # 叠加预测 + Bounding Box
            plt.subplot(n, 3, row * 3 + 3)
            plt.imshow(overlay)
            plt.title("Overlay + Bounding Box")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("叠加 + 框选 可视化图已保存到:", out_path)