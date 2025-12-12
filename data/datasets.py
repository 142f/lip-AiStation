import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils
import numpy as np


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)
        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1
        self.total_list = self.real_list + self.fake_list

        self.targets = [self.label_dict[path] for path in self.total_list]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]

        # 1. 读取图像 (BGR)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            # 返回全0数据防止崩溃，保持维度一致
            # Shape: (3, 1120, 1120), (3, 5, 3, 224, 224)
            return torch.zeros((3, 1120, 1120)), torch.zeros((3, 5, 3, 224, 224)), label

        # 2. 颜色转换 BGR -> RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # 3. 预定义归一化参数 (利用广播机制)
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)

        # 4. [性能核心] 预分配内存，避免 list append 带来的内存碎片
        # 形状: [Scales=3, Regions=5, H=224, W=224, C=3] (先 HWC 方便 cv2 操作)
        crops_np = np.zeros((3, 5, 224, 224, 3), dtype=np.float32)

        # 5. 快速裁剪 (Scale 0, 1, 2)
        for i in range(5):
            # Scale 0: 原始裁剪
            # Numpy 切片操作极快，无内存拷贝
            patch = img_cv[500:, i*500 : i*500 + 500, :]
            
            if patch.size > 0:
                # Resize
                p0 = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
                crops_np[0, i] = p0
                
                # Scale 1: 基于 p0 再裁剪 (168x168 -> 224)
                # 坐标: 28:196
                p1 = p0[28:196, 28:196, :] 
                crops_np[1, i] = cv2.resize(p1, (224, 224), interpolation=cv2.INTER_LINEAR)
                
                # Scale 2: 基于 p0 再裁剪 (102x102 -> 224)
                # 坐标: 61:163
                p2 = p0[61:163, 61:163, :]
                crops_np[2, i] = cv2.resize(p2, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 6. 统一归一化 + 维度转置
        # (HWC -> CHW) 并转为 Float Tensor
        # 利用 Numpy 向量化计算替代循环，速度提升巨大
        crops_np = (crops_np / 255.0 - mean) / std
        crops_tensor = torch.from_numpy(crops_np.transpose(0, 1, 4, 2, 3)).float()

        # 7. 全局大图处理
        img_global = cv2.resize(img_cv, (1120, 1120), interpolation=cv2.INTER_LINEAR)
        img_global = (img_global / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_global.transpose(2, 0, 1)).float()

        # 返回: Tensor, Tensor, int
        return img_tensor, crops_tensor, label