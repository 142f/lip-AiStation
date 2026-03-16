from .clip_models import CLIPModel
from .LipFD import LipFD, RALoss

VALID_NAMES = [
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
    "CLIP:ViT-L/14@336px",
    "DFN:ViT-L/14",
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:])
    elif name.startswith("DFN:"):
        return CLIPModel(name)
    else:
        assert False


def build_model(transformer_name):
    assert transformer_name in VALID_NAMES
    if transformer_name.startswith("CLIP:") or transformer_name.startswith("DFN:"):
        return LipFD(transformer_name)
    else:
        assert False


def get_loss():
    return RALoss()
