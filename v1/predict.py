import os
import torch
import numpy as np
import cv2
from PIL import Image
from v1.src.model import FCNResNet50
from v1.src.utils import get_class_colors


def predict_and_visualize(
    model,
    img_path,
    device,
    img_size=(256, 256),
    num_classes=3,
):
    """
    对单张图片进行预测并返回带框的可视化结果
    
    参数:
    model: 训练好的模型
    img_path: 输入图片路径
    device: 设备
    img_size: 图片大小
    num_classes: 类别数 (包含背景)
    
    返回:
    overlay: 带绿色透明覆盖和边界框的图片 (numpy array, uint8)
    """
    model.eval()
    
    # 读取图片
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (width, height)
    
    # 调整大小用于预测
    img_resized = img.resize(img_size)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    
    # 将预测结果resize回原始大小
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # 将原始图片转换为numpy array
    img_original = np.array(img)
    img_f = img_original.astype(np.float32) / 255.0
    
    colors = get_class_colors(num_classes)
    alpha = 0.25
    overlay_f = img_f.copy()
    for cls_id in range(1, num_classes):
        mask = pred_mask_resized == cls_id
        if not mask.any():
            continue
        color = np.array(colors[cls_id], dtype=np.float32) / 255.0
        overlay_f[mask] = (1 - alpha) * img_f[mask] + alpha * color
    
    # 转换为uint8用于绘制边界框
    overlay = (overlay_f * 255).clip(0, 255).astype(np.uint8)
    
    return overlay


def main():
    # 配置参数
    model_path = "outputs/fcn_crack_best.pth"  # 使用最佳模型
    predict_dir = "predict_images"
    results_dir = "results"
    img_size = (256, 256)
    num_classes = 3
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查预测图片目录是否存在
    if not os.path.exists(predict_dir):
        print(f"错误: 预测图片目录 '{predict_dir}' 不存在！")
        print(f"请创建该目录并放入需要预测的图片。")
        return
    
    # 设备配置
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"使用设备: {device}")
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在！")
        print("请先训练模型或指定正确的模型路径。")
        return
    
    print(f"加载模型: {model_path}")
    model = FCNResNet50(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载完成！")
    
    # 获取所有图片文件
    image_files = [
        f for f in os.listdir(predict_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    if len(image_files) == 0:
        print(f"警告: 在 '{predict_dir}' 目录中没有找到图片文件！")
        print("支持的格式: .jpg, .jpeg, .png, .bmp")
        return
    
    print(f"\n找到 {len(image_files)} 张图片，开始预测...\n")
    
    # 对每张图片进行预测
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(predict_dir, img_file)
        
        try:
            print(f"[{i}/{len(image_files)}] 处理: {img_file}")
            
            # 预测并可视化
            result = predict_and_visualize(
                model,
                img_path,
                device,
                img_size=img_size,
                num_classes=num_classes
            )
            
            # 保存结果
            result_path = os.path.join(results_dir, f"result_{img_file}")
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, result_bgr)
            
            print(f"  ✓ 结果已保存: {result_path}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            continue
    
    print(f"\n预测完成！所有结果已保存到 '{results_dir}' 目录。")


if __name__ == "__main__":
    main()
