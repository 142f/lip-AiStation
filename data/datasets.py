import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils
import numpy as np


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val", "test"]
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

        # 1. 读取 (BGR)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            # 返回全0数据防止崩溃，尺寸需对应修改后的逻辑
            return torch.zeros((3, 1120, 1120)), [[torch.zeros((3, 224, 224))]*5]*3, label

        # 2. 颜色转换 BGR -> RGB (此时还是 uint8，速度快)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # --- [优化开始] 先在 Numpy 上裁剪，再转 Tensor ---
        
        # 预定义 Normalize (保持不变)
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        to_tensor = transforms.ToTensor()

        # 优化逻辑：直接操作 numpy 数组进行切片，避免操作巨大的 float tensor
        # 原逻辑：img[:, 500:, i*500 : i*500 + 500] (Tensor操作)
        # 新逻辑：img_cv[500:, i*500 : i*500 + 500, :] (Numpy操作)
        
        crops_list = []
        # 第一组裁剪 (Scale 0)
        for i in range(5):
            # Numpy 切片: [H, W, C]
            # 假设 img_cv 足够大，注意边界检查
            patch_cv = img_cv[500:, i*500 : i*500 + 500, :] 
            
            # 缩放 -> 转Tensor -> 归一化 (这一套流水线处理小图非常快)
            # 注意：cv2.resize 是 (W, H)
            if patch_cv.size == 0: # 防止空切片
                 patch_cv = np.zeros((500, 500, 3), dtype=np.uint8)
            
            patch_cv = cv2.resize(patch_cv, (224, 224), interpolation=cv2.INTER_LINEAR)
            patch_tensor = normalize(to_tensor(patch_cv))
            crops_list.append(patch_tensor)

        crops = [crops_list, [], []]

        # 后续的 crops[1] 和 crops[2] 是基于 crops[0] (也就是 Scale 0) 再次裁剪的
        # 因为 crops[0] 已经是 Tensor 了，这里用 transforms 是可以的，或者也可以继续优化为 numpy 操作
        # 但这部分是在 224x224 上操作，开销不大，维持原样即可
        crop_idx = [(28, 196), (61, 163)]
        crop_resize = transforms.Resize((224, 224))
        
        for i in range(len(crops[0])):
            # 注意：这里 crops[0][i] 已经是 (C, H, W)
            # crop_idx 同样可以直接切片
            crops[1].append(crop_resize(crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(crop_resize(crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))

        # 最后处理大图 (Global Context)
        # 同样先 resize numpy 再转 tensor
        img_global = cv2.resize(img_cv, (1120, 1120), interpolation=cv2.INTER_LINEAR)
        img = normalize(to_tensor(img_global))

        return img, crops, label