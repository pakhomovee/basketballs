"""ResNet-50 + BNNeck re-identification model (Bag of Tricks, Luo et al. 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BNNeck(nn.Module):
    """Batch-norm bottleneck that separates ID-loss and metric-loss feature spaces."""

    def __init__(self, in_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class ReIDModel(nn.Module):
    """ResNet-50 backbone → 2048-d global feature → BNNeck → classifier.

    During training returns (bn_feat, logits) for joint triplet + CE loss.
    During inference returns L2-normalised features for distance computation.
    """

    def __init__(self, num_classes: int, feature_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Drop the original FC layer; keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bnneck = BNNeck(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor):
        feat_map = self.backbone(x)  # (B, 2048, H', W')
        global_feat = self.gap(feat_map)  # (B, 2048, 1, 1)
        global_feat = global_feat.flatten(1)  # (B, 2048)

        bn_feat = self.bnneck(global_feat)
        if self.training:
            logits = self.classifier(bn_feat)
            return global_feat, logits
        # Inference: return L2-normalised BN features
        return F.normalize(bn_feat, p=2, dim=1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised features for a batch of images."""
        was_training = self.training
        self.eval()
        feats = self.forward(x)
        if was_training:
            self.train()
        return feats
