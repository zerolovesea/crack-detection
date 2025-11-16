import os
import scipy.io as sio
import numpy as np
from PIL import Image

mat_dir = "CrackForest-dataset/groundTruth"
img_dir = "CrackForest-dataset/image"

out_dir = "data/masks"
os.makedirs(out_dir, exist_ok=True)

os.makedirs("data/images", exist_ok=True)

image_basenames = [
    os.path.splitext(f)[0]
    for f in os.listdir(img_dir)
    if f.lower().endswith(".jpg")
]


def mat_to_mask(mat_path: str) -> np.ndarray:
    """从 CrackForest 的 .mat 中提取裂缝 mask（统一 0/255 uint8）"""
    mat = sio.loadmat(mat_path)

    if "groundTruth" not in mat:
        raise ValueError(f"{mat_path} 中没有 groundTruth 字段")

    gt = mat["groundTruth"]  # 一般 shape (1,1)
    cell = gt[0, 0]

    # 情况 1：cell 是 (array1, array2) 这样的 tuple/list
    if isinstance(cell, (tuple, list)):
        if len(cell) < 2:
            raise ValueError("groundTruth[0,0] 长度 < 2，无法取裂缝 mask")
        surface_mask = np.array(cell[0])
        crack_mask = np.array(cell[1])

    # 情况 2：cell 是 numpy.void（MATLAB struct / record），里面同样是两个字段
    elif isinstance(cell, np.void):
        # numpy.void 支持按下标访问各个字段
        if len(cell) < 2:
            raise ValueError("numpy.void 类型但字段数 < 2")
        surface_mask = np.array(cell[0])
        crack_mask = np.array(cell[1])

    else:
        print("Unexpected groundTruth[0,0] type:", type(cell))
        print("Value:", cell)
        raise ValueError("无法识别 groundTruth[0,0] 格式")

    if crack_mask.ndim != 2:
        print("crack_mask shape:", crack_mask.shape)
        raise ValueError("crack_mask 不是 2D 矩阵，无法转为图像")

    # 有些图可能本身就没有裂缝（全 0），这是正常的
    mask = (crack_mask > 0).astype(np.uint8) * 255
    return mask


ok, fail = 0, 0

for base in image_basenames:
    mat_path = os.path.join(mat_dir, base + ".mat")
    if not os.path.exists(mat_path):
        print(f"[跳过] 找不到 {mat_path}")
        continue

    try:
        mask = mat_to_mask(mat_path)
    except Exception as e:
        print(f"[错误] 处理 {mat_path} 时出错：{e}")
        fail += 1
        continue

    out_path = os.path.join(out_dir, base + ".png")
    Image.fromarray(mask).save(out_path)
    ok += 1
    print("Converted:", out_path)

print(f"转换完成，成功 {ok} 个，失败 {fail} 个")