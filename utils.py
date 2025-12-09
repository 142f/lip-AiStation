import os
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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