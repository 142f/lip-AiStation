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
    def __init__(self):
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        total_loss = 0.0
        for i in range(len(alphas_org)):
            # 正确的张量运算方式
            diff = alphas_max[i] - alphas_org[i]  # shape: (batch_size, 1)
            # 添加数值稳定性保护
            diff = torch.clamp(diff, min=0.0, max=100.0)
            # 对整个batch进行向量化计算
            loss_wt = 10 / torch.exp(diff)  # shape: (batch_size, 1)
            # 对batch取平均
            total_loss += loss_wt.mean()

        # 对所有区域组取平均
        return total_loss / len(alphas_org)