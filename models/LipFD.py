import torch
import numpy as np
import torch.nn as nn
from .clip import clip
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import open_clip
from .region_awareness import get_backbone


class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()

        if "@336px" in name:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=3)
        else:
            self.conv1 = nn.Conv2d(
                3, 3, kernel_size=5, stride=5
            )  # (1120, 1120) -> (224, 224)

        if name.startswith("DFN:"):
            print(f"Loading Apple DFN model: {name}")
            self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', 
                pretrained='dfn2b', 
                device='cpu'
            )
        else:
            clean_name = name.replace("CLIP:", "")
            self.encoder, self.preprocess = clip.load(clean_name, device="cpu")
            
        self.backbone = get_backbone()

    def forward(self, x, feature):
        return self.backbone(x, feature)

    def get_features(self, x):
        x = self.conv1(x)
        features = self.encoder.encode_image(x)
        return features


class RALoss(nn.Module):
    def __init__(self, margin=0.25): # 设定及格线 0.25
        super(RALoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU() # 核心：使用 ReLU 实现自动截断

    def forward(self, alphas_max, alphas_org):
        # 逻辑：希望 Max 至少比 Org 大 margin
        # 如果达标：(Org + 0.25) - Max < 0 -> ReLU后为 0 -> 梯度为 0
        diff = (alphas_org + self.margin) - alphas_max
        
        # 计算 Loss，不建议再乘 10 了，保持数值纯粹
        loss = self.relu(diff).mean()
        
        return loss