from .clip import clip 
from PIL import Image
import torch.nn as nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import open_clip


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14@336px" : 768,
    "DFN:ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        if name.startswith("DFN:"):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', 
                pretrained='dfn2b', 
                device='cpu'
            )
        else:
            clean_name = name.replace("CLIP:", "")
            self.model, self.preprocess = clip.load(clean_name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        
        embed_dim = CHANNELS.get(name.replace("CLIP:", ""), 768)
        self.fc = nn.Linear(embed_dim, num_classes)
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)

