import os
import torch
import numpy as np

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset



class CrackDataset(Dataset):
    """
    假设：
      - img_dir 里存 RGB 图片（jpg/png 均可）
      - mask_dir 里存 同名（同 basename）的二值 mask（后缀可以不一样，比如 .png）
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int] = (256, 256)
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # 所有图片路径
        all_imgs = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        all_imgs.sort()
        self.img_paths = [os.path.join(img_dir, f) for f in all_imgs]
        assert len(self.img_paths) > 0, "img_dir 里没有找到图片"

        # 扫一遍 mask_dir，按 basename 建索引
        self.mask_map = {}
        for f in os.listdir(mask_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(f)[0] # basename without ext
            self.mask_map[stem] = os.path.join(mask_dir, f)

        # 配对
        self.paired = []
        for img_path in self.img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem in self.mask_map:
                self.paired.append((img_path, self.mask_map[stem]))
            else:
                print(f"[警告] 找不到 {stem} 的 mask，跳过这张图片")

        assert len(self.paired) > 0, "没有任何 img-mask 配对样本，请检查文件名是否一致"

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_path, mask_path = self.paired[idx]

        img = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("L").resize(self.img_size)

        img = np.array(img, dtype=np.float32) / 255.0
        # 假设 mask：白=裂缝，黑=背景
        mask = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

