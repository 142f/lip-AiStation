import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils
import numpy as np
import random


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        self.opt = opt
        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)
        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1
        self.total_list = self.real_list + self.fake_list

        self.targets = [self.label_dict[path] for path in self.total_list]

    def _parse_range(self, value, cast=float, default=(0.0, 0.0)):
        try:
            parts = [p.strip() for p in str(value).split(',')]
            if len(parts) != 2:
                return default
            lo = cast(parts[0])
            hi = cast(parts[1])
            if hi < lo:
                lo, hi = hi, lo
            return lo, hi
        except Exception:
            return default

    def _random_jpeg(self, img_rgb, qmin, qmax):
        if qmax <= 0:
            return img_rgb
        quality = int(round(random.uniform(qmin, qmax)))
        quality = max(1, min(100, quality))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), encode_param)
        if not ok:
            return img_rgb
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None:
            return img_rgb
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    def _random_blur(self, img_rgb, sigma_min, sigma_max):
        if sigma_max <= 0:
            return img_rgb
        sigma = random.uniform(sigma_min, sigma_max)
        if sigma <= 1e-6:
            return img_rgb
        k = int(round(sigma * 3) * 2 + 1)
        k = max(3, min(31, k))
        return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=sigma, sigmaY=sigma)

    def _random_resize_degrade(self, img_rgb, scale_min, scale_max, interp):
        if scale_max <= 0:
            return img_rgb
        h, w = img_rgb.shape[:2]
        scale = random.uniform(scale_min, scale_max)
        scale = max(0.1, min(1.0, float(scale)))
        if abs(scale - 1.0) < 1e-6:
            return img_rgb
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        small = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
        return cv2.resize(small, (w, h), interpolation=interp)

    def _spec_mask(self, spec_rgb, num_masks, time_frac, freq_frac):
        # spec_rgb: RGB image region (H, W, 3). Mask rectangles without shifting.
        h, w = spec_rgb.shape[:2]
        if h <= 1 or w <= 1:
            return spec_rgb
        out = spec_rgb.copy()
        mean_val = out.mean(axis=(0, 1), keepdims=True)
        for _ in range(max(0, int(num_masks))):
            max_tw = max(1, int(round(w * max(0.0, min(1.0, time_frac)))))
            max_fh = max(1, int(round(h * max(0.0, min(1.0, freq_frac)))))
            tw = random.randint(1, max_tw)
            fh = random.randint(1, max_fh)
            x0 = random.randint(0, max(0, w - tw))
            y0 = random.randint(0, max(0, h - fh))
            out[y0:y0 + fh, x0:x0 + tw, :] = mean_val
        return out

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

        # 2.5 训练期数据增强（保持音画对齐，不做时间平移/随机大裁剪）
        if getattr(self.opt, 'use_aug', False) and self.data_label == 'train':
            # 解析增强参数
            blur_min, blur_max = self._parse_range(getattr(self.opt, 'blur_sig', '0.0,0.0'), float, (0.0, 0.0))
            jpg_min, jpg_max = self._parse_range(getattr(self.opt, 'jpg_qual', '0,0'), int, (0, 0))
            rz_min, rz_max = self._parse_range(getattr(self.opt, 'rz_scale', '1.0,1.0'), float, (1.0, 1.0))
            interp_name = str(getattr(self.opt, 'rz_interp', 'bilinear')).lower()
            interp = cv2.INTER_LINEAR if interp_name in ('bilinear', 'linear') else cv2.INTER_AREA

            # (a) 轻度分辨率退化（先降采样再回采样）
            if random.random() < 0.5:
                img_cv = self._random_resize_degrade(img_cv, rz_min, rz_max, interp)

            # (b) JPEG 压缩退化（模拟转码）
            if random.random() < 0.7:
                img_cv = self._random_jpeg(img_cv, jpg_min, jpg_max)

            # (c) 轻度模糊（模拟运动/失焦）
            if random.random() < 0.3:
                img_cv = self._random_blur(img_cv, blur_min, blur_max)

            # (d) 频谱区域 SpecAugment 风格遮挡（不改变时序对齐）
            if getattr(self.opt, 'spec_aug', False):
                h, w = img_cv.shape[:2]
                # 按你当前预处理习惯：上半部分是频谱，下半部分是人脸
                split_y = h // 2
                spec = img_cv[:split_y, :, :]
                masked = self._spec_mask(
                    spec,
                    num_masks=getattr(self.opt, 'spec_num_masks', 2),
                    time_frac=getattr(self.opt, 'spec_time_mask', 0.12),
                    freq_frac=getattr(self.opt, 'spec_freq_mask', 0.18),
                )
                img_cv[:split_y, :, :] = masked
        
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