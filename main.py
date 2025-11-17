import os
import torch

from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from src.dataset import CrackDataset
from src.model import FCNResNet50Binary
from src.utils import compute_pos_weight, visualize_predictions


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs + targets).sum(dim=(1, 2, 3)) + eps
    dice = num / den
    return 1 - dice.mean()

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

def train_one_epoch(model, loader, optimizer, device, pos_weight=None, dice_weight=1.0, bce_weight=1.0, threshold=0.3):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        bce = bce_loss_fn(logits, masks)
        dsl = dice_loss_with_logits(logits, masks)

        loss = bce_weight * bce + dice_weight * dsl
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        
        # 计算准确率
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels
    return avg_loss, accuracy


def evaluate(model, loader, device, pos_weight=None, threshold=0.3):
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    total_inter = 0.0
    total_union = 0.0
    total_dice_inter = 0.0
    total_dice_sum = 0.0
    valid_iou_samples = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            bce = bce_loss_fn(logits, masks)
            dsl = dice_loss_with_logits(logits, masks)
            loss = bce + dsl
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            # 计算准确率
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            B = imgs.size(0)
            for b in range(B):
                gt = masks[b]
                pr = preds[b]

                gt_sum = gt.sum().item()
                pr_sum = pr.sum().item()

                if gt_sum == 0 and pr_sum == 0:
                    continue

                inter = (gt * pr).sum().item()
                union = (gt + pr - gt * pr).sum().item()

                total_inter += inter
                total_union += union

                total_dice_inter += 2 * inter
                total_dice_sum += (gt_sum + pr_sum)
                valid_iou_samples += 1

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct_pixels / total_pixels
    
    if valid_iou_samples > 0:
        mean_iou = total_inter / (total_union + 1e-6)
        mean_dice = total_dice_inter / (total_dice_sum + 1e-6)
    else:
        mean_iou = 0.0
        mean_dice = 0.0

    return avg_loss, accuracy, mean_iou, mean_dice


def main():
    img_dir = "data/images"
    mask_dir = "data/masks"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    img_size = (256, 256)
    batch_size = 4
    num_epochs = 10
    lr = 1e-4          # FCN + ResNet 更大，lr 稍微小一点
    val_ratio = 0.2
    threshold = 0.3

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print("Using device:", device)

    # 数据集 & 划分
    full_dataset = CrackDataset(img_dir, mask_dir, img_size=img_size)
    pos_weight = compute_pos_weight(full_dataset, device=device)

    val_size = int(len(full_dataset) * val_ratio)
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
    model = FCNResNet50Binary(pretrained=True).to(device)
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
            pos_weight=pos_weight, dice_weight=1.0, bce_weight=1.0, threshold=threshold
        )
        val_loss, val_acc, val_iou, val_dice = evaluate(
            model, val_loader, device, pos_weight=pos_weight, threshold=threshold
        )

        # 记录训练曲线数据
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f} "
            f"- val_IoU: {val_iou:.4f} "
            f"- val_Dice: {val_dice:.4f}"
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

    model = FCNResNet50Binary(pretrained=False).to(device)
    model.load_state_dict(torch.load(last_ckpt_path, map_location=device))

    # 可视化
    vis_path = os.path.join(output_dir, "fcn_vis_last.png")
    visualize_predictions(
        model, val_dataset, device,
        img_size=img_size,
        out_path=vis_path,
        num_samples=5,
        threshold=threshold
    )


if __name__ == "__main__":
    main()