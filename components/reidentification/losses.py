"""ReID losses: batch-hard triplet and ArcFace."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TripletLoss", "ArcFaceLoss"]


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with Euclidean distance and margin ranking.

    This is the standard Bag-of-Tricks setup: triplet is applied to the raw
    global features before BNNeck, using hardest positive / hardest negative
    mining within the P×K batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def _pair_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask
        return positive_mask, negative_mask

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(features, features, p=2)
        positive_mask, negative_mask = self._pair_masks(labels)

        hardest_pos = dist.masked_fill(~positive_mask, 0.0).amax(dim=1)
        hardest_neg = dist.masked_fill(~negative_mask, float("inf")).amin(dim=1)

        target = torch.ones_like(hardest_pos)
        return self.ranking_loss(hardest_neg, hardest_pos, target)


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss (ArcFace).

    Operates on L2-normalised features.  The learned weight matrix rows are
    normalised on-the-fly so that the dot-product equals cos(theta).  An
    angular margin *m* is added to the target-class angle, then the result
    is scaled by *s* before cross-entropy.
    """

    def __init__(self, num_classes: int, feature_dim: int = 2048, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.mm(features, W.T).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = cos_theta.acos()
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1)
        logits = self.s * (one_hot * torch.cos(theta + self.m) + (1 - one_hot) * cos_theta)
        return F.cross_entropy(logits, labels)
