"""
Multi-Stage Temporal Convolutional Network (MS-TCN).

Architecture from:
    Abu Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network
    for Action Segmentation", CVPR 2019.

Default hyper-parameters match the paper:
    4 stages, 10 dilated layers per stage, 64 filters, kernel 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, n_filters: int, dropout: float = 0.5):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            n_filters, n_filters, kernel_size=3, padding=dilation, dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_filters: int = 64,
        n_layers: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(input_dim, n_filters, kernel_size=1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2**i, n_filters, dropout=dropout) for i in range(n_layers)]
        )
        self.conv_out = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim, T)

        Returns
        -------
        (B, n_classes, T) logits
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MultiStageTCN(nn.Module):
    """
    Stacks *n_stages* single-stage TCNs.  The first stage receives
    raw features; every subsequent stage receives the softmax output
    of the previous stage and refines it.

    ``forward`` returns a **list** of per-stage logit tensors so that
    each stage's loss can be computed independently.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_stages: int = 4,
        n_filters: int = 64,
        n_layers: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.n_classes = n_classes

        self.stages = nn.ModuleList()
        self.stages.append(SingleStageTCN(input_dim, n_classes, n_filters, n_layers, dropout))
        for _ in range(1, n_stages):
            self.stages.append(SingleStageTCN(n_classes, n_classes, n_filters, n_layers, dropout))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x    : (B, input_dim, T)
        mask : (B, T) boolean – ``True`` for valid positions.

        Returns
        -------
        List of *n_stages* tensors, each (B, n_classes, T).
        """
        outputs: list[torch.Tensor] = []
        out = self.stages[0](x)
        if mask is not None:
            out = out * mask.unsqueeze(1).float()
        outputs.append(out)

        for s in range(1, self.n_stages):
            inp = F.softmax(out, dim=1)
            if mask is not None:
                inp = inp * mask.unsqueeze(1).float()
            out = self.stages[s](inp)
            if mask is not None:
                out = out * mask.unsqueeze(1).float()
            outputs.append(out)

        return outputs
