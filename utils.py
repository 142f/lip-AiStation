import os
import random
import numpy as np
import torch

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