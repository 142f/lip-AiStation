import cv2
import torch
from torch.utils.data import Dataset
import utils
import numpy as np

# 建议：避免 DataLoader 多 worker 时 OpenCV 线程爆炸
try:
    cv2.setNumThreads(0)
except Exception:
    pass


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val", "test"]
        self.data_label = opt.data_label
        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)

        self.label_dict = {p: 0 for p in self.real_list}
        self.label_dict.update({p: 1 for p in self.fake_list})

        self.total_list = self.real_list + self.fake_list
        self.targets = [self.label_dict[path] for path in self.total_list]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]

        img_cv = cv2.imread(img_path)  # BGR uint8
        if img_cv is None:
            zero_img = torch.zeros((3, 1120, 1120), dtype=torch.uint8)
            zero_crop = torch.zeros((3, 224, 224), dtype=torch.uint8)
            crops = [[zero_crop.clone() for _ in range(5)] for _ in range(3)]
            return zero_img, crops, label

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        crops_s0, crops_s1, crops_s2 = [], [], []
        for i in range(5):
            patch_cv = img_cv[500:, i * 500: i * 500 + 500, :]
            if patch_cv.size == 0:
                patch_cv = np.zeros((500, 500, 3), dtype=np.uint8)

            patch_cv = cv2.resize(patch_cv, (224, 224), interpolation=cv2.INTER_LINEAR)

            crops_s0.append(torch.from_numpy(patch_cv).permute(2, 0, 1).contiguous())

            s1 = patch_cv[28:196, 28:196, :]
            s2 = patch_cv[61:163, 61:163, :]

            s1 = cv2.resize(s1, (224, 224), interpolation=cv2.INTER_LINEAR)
            s2 = cv2.resize(s2, (224, 224), interpolation=cv2.INTER_LINEAR)

            crops_s1.append(torch.from_numpy(s1).permute(2, 0, 1).contiguous())
            crops_s2.append(torch.from_numpy(s2).permute(2, 0, 1).contiguous())

        crops = [crops_s0, crops_s1, crops_s2]

        img_global = cv2.resize(img_cv, (1120, 1120), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img_global).permute(2, 0, 1).contiguous()  # uint8

        return img, crops, label
