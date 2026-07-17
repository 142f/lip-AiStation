import os
import random
import numpy as np
import torch


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

def get_list(path) -> list:
    r"""Recursively read all files in root path"""
    # 清理路径中的空字符
    path = path.replace('\x00', '')
    image_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] in ['png', 'jpg', 'jpeg']:
                image_list.append(os.path.join(root, f))
    return image_list


def get_clip_normalization(device):
    mean = torch.tensor(CLIP_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def prepare_model_inputs(img, raw_crops, device, mean, std):
    """Move and normalize an AVLip batch, preserving legacy crop input."""
    img = img.to(device, non_blocking=True)
    if img.dtype == torch.uint8:
        img = img.float().div_(255.0)
    img = img.sub_(mean).div_(std)

    if torch.is_tensor(raw_crops):
        crops = raw_crops.to(device, non_blocking=True)
    else:
        crops = [
            [crop.to(device, non_blocking=True) for crop in scale_crops]
            for scale_crops in raw_crops
        ]
    return img, crops

def set_seed(seed=42):
    # 1. Python random
    random.seed(seed)
    
    # 2. Environment variables (Python hash seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 6. CuDNN Determinism (性能会有轻微下降，但保证卷积结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Random seed set to: {seed}")
