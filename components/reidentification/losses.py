"""ReID losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with soft margin.

    For each anchor, selects the hardest positive (max distance) and the
    hardest negative (min distance) within the batch.
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
