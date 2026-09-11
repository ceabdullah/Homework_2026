import timm
import torch.nn as nn

REGISTRY = {
    "resnet50": "resnet50",
    "resnet18": "resnet18",          # pilot / CPU smoke tests only
    "effnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "vit_small": "vit_small_patch16_224",
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a fresh head.

    pretrained=True fetches weights from the HF hub; set it False (or
    --no-pretrained) on a machine without outbound access, and say so in
    the write-up -- it changes the numbers substantially.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; options: {sorted(REGISTRY)}")
    return timm.create_model(REGISTRY[name], pretrained=pretrained, num_classes=num_classes)
