import os
import argparse
import torch

from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from v1.src.dataset import CrackDataset, CocoPolygonSegDataset
from v1.src.model import FCNResNet50
from v1.src.utils import compute_class_weights, visualize_predictions


def plot_training_curves(train_losses, train_accuracies, val_losses=None, val_accuracies=None, save_dir='./', filename='training_curves.png'):
    """
    绘制训练损失和准确率的折线图
    
    参数:
    train_losses: list, 训练损失值列表
    train_accuracies: list, 训练准确率列表
    val_losses: list, 验证损失值列表 (可选)
    val_accuracies: list, 验证准确率列表 (可选)
    save_dir: str, 保存目录路径
    filename: str, 保存的文件名
    """
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    
    # 显示图像
    plt.show()

def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        loss = ce_loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        
        # 计算准确率
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels
    return avg_loss, accuracy


def evaluate(model, loader, device, num_classes, class_weights=None):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    total_inter = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_dice_inter = torch.zeros(num_classes, device=device)
    total_dice_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = ce_loss_fn(logits, masks)
            total_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1)
            
            # 计算准确率
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            for cls_id in range(1, num_classes):
                pred_c = preds == cls_id
                mask_c = masks == cls_id
                inter = (pred_c & mask_c).sum()
                union = (pred_c | mask_c).sum()
                total_inter[cls_id] += inter
                total_union[cls_id] += union
                total_dice_inter[cls_id] += 2 * inter
                total_dice_sum[cls_id] += pred_c.sum() + mask_c.sum()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels
    
    valid_classes = list(range(1, num_classes))
    iou_per_class = total_inter[valid_classes] / (total_union[valid_classes] + 1e-6)
    dice_per_class = total_dice_inter[valid_classes] / (total_dice_sum[valid_classes] + 1e-6)
    mean_iou = iou_per_class.mean().item() if len(valid_classes) > 0 else 0.0
    mean_dice = dice_per_class.mean().item() if len(valid_classes) > 0 else 0.0

    return avg_loss, accuracy, mean_iou, mean_dice, iou_per_class.detach().cpu().tolist(), dice_per_class.detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="FCNResNet50 语义分割训练/评估")
    parser.add_argument("--data_source", choices=["data_poly", "mask_dir"], default="data_poly")
    parser.add_argument("--data_poly_dir", type=str, default="data_poly")
    parser.add_argument("--img_dir", type=str, default="data/images")
    parser.add_argument("--mask_dir", type=str, default="data/masks")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    img_size = (int(args.img_size), int(args.img_size))
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    lr = float(args.lr)          # FCN + ResNet 更大，lr 稍微小一点
    val_ratio = float(args.val_ratio)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print("Using device:", device)

    # 数据集 & 划分
    if args.data_source == "data_poly":
        full_dataset = CocoPolygonSegDataset(args.data_poly_dir, img_size=img_size)
    else:
        full_dataset = CrackDataset(args.img_dir, args.mask_dir, img_size=img_size)
    class_weights = compute_class_weights(full_dataset, full_dataset.num_classes, device=device)

    if len(full_dataset) < 2:
        train_dataset = full_dataset
        val_dataset = full_dataset
        train_size = len(full_dataset)
        val_size = len(full_dataset)
    else:
        val_size = int(len(full_dataset) * val_ratio)
        if val_size < 1:
            val_size = 1
        if val_size >= len(full_dataset):
            val_size = len(full_dataset) - 1
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    print(f"总样本数: {len(full_dataset)}, 训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型 & 优化器
    model = FCNResNet50(num_classes=full_dataset.num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0
    best_ckpt_path = os.path.join(output_dir, "fcn_crack_best.pth")
    
    # 用于记录训练曲线的列表
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 训练 + 验证
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            class_weights=class_weights
        )
        val_loss, val_acc, val_iou, val_dice, iou_per_class, dice_per_class = evaluate(
            model, val_loader, device, full_dataset.num_classes, class_weights=class_weights
        )

        # 记录训练曲线数据
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        class_names = getattr(full_dataset, "class_names", [])
        per_class_parts = []
        for idx in range(len(iou_per_class)):
            name = class_names[idx] if idx < len(class_names) else f"C{idx+1}"
            per_class_parts.append(
                f"{name} IoU:{iou_per_class[idx]:.4f} Dice:{dice_per_class[idx]:.4f}"
            )
        per_class_msg = " ".join(per_class_parts)
        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f} "
            f"- val_IoU: {val_iou:.4f} "
            f"- val_Dice: {val_dice:.4f} "
            f"- {per_class_msg}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  > 更新最佳模型，val IoU={best_val_iou:.4f} 已保存到 {best_ckpt_path}")

    # 绘制训练曲线
    plot_training_curves(
        train_losses, train_accuracies, 
        val_losses, val_accuracies,
        save_dir=output_dir, 
        filename='training_curves.png'
    )

    # 保存最终模型
    last_ckpt_path = os.path.join(output_dir, "fcn_crack_last.pth")
    torch.save(model.state_dict(), last_ckpt_path)
    print("最终 FCN 模型已保存到:", last_ckpt_path)

    model = FCNResNet50(num_classes=full_dataset.num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(last_ckpt_path, map_location=device))

    # 可视化
    vis_path = os.path.join(output_dir, "fcn_vis_last.png")
    visualize_predictions(
        model, val_dataset, device,
        img_size=img_size,
        out_path=vis_path,
        num_samples=5,
    )


if __name__ == "__main__":
    main()
