"""
============================================================================
  Backbone — Pretrained feature extractors.
  
  Separated from the F-EDL head so you can swap backbones
  (ResNet-18/50, EfficientNet, Swin, …) without touching the EDL logic.
============================================================================
"""

import torch.nn as nn
import torchvision.models as models


def build_backbone(name: str = "resnet18", pretrained: bool = True, freeze: bool = True):
    """
    Build a backbone feature extractor.

    Args:
        name:       Model name (currently supports 'resnet18', 'resnet50').
        pretrained: Load ImageNet-pretrained weights.
        freeze:     Freeze all backbone parameters (no gradient).

    Returns:
        backbone:     nn.Module with .fc removed (outputs feature vector).
        in_features:  Dimensionality of the feature vector.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # Remove the original classification head
    backbone.fc = nn.Identity()

    # Freeze if requested
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()  # Lock BatchNorm stats

    return backbone, in_features
