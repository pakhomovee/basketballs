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
    """ResNet-50 backbone → 2048-d global feature → BNNeck."""

    def __init__(self, feature_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        self.backbone = self._build_backbone(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = BNNeck(feature_dim)

    @staticmethod
    def _build_backbone(*, pretrained: bool) -> nn.Module:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        return nn.Sequential(*list(backbone.children())[:-2])

    def _global_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        return self.gap(feat_map).flatten(1)

    def forward(self, x: torch.Tensor):
        global_feat = self._global_features(x)
        bn_feat = self.bnneck(global_feat)
        normed = F.normalize(bn_feat, p=2, dim=1)
        if self.training:
            return global_feat, normed
        return normed

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised features for a batch of images."""
        was_training = self.training
        self.eval()
        feats = self.forward(x)
        if was_training:
            self.train()
        return feats
