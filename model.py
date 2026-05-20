from __future__ import annotations

import torch
import torch.nn as nn


class Net(nn.Module):
    """Compact EEGNet-style classifier for 59-channel event EEG epochs.

    The input is the original single-channel tensor with shape
    (batch, 1, 59, 282).  The model builds a two-channel representation inside
    forward(): one branch keeps absolute amplitude after a fixed microvolt-scale
    normalization, while the other uses per-trial z-score normalization.
    """

    def __init__(
        self,
        input_shape=(1, 59, 282),
        num_classes: int = 2,
        temporal_filters: int = 24,
        depth_multiplier: int = 2,
        pointwise_filters: int = 48,
        dropout: float = 0.50,
    ) -> None:
        super().__init__()
        _, channels, samples = input_shape
        hidden = temporal_filters * depth_multiplier

        self.features = nn.Sequential(
            nn.Conv2d(2, temporal_filters, kernel_size=(1, 65), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                hidden,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, hidden, kernel_size=(1, 17), padding=(0, 8), groups=hidden, bias=False),
            nn.Conv2d(hidden, pointwise_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(pointwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            feature_dim = self.features(torch.zeros(1, 2, channels, samples)).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        z_scored = (x - mean) / std
        amplitude = x / 1e-5
        x = torch.cat([z_scored, amplitude], dim=1)
        return self.classifier(self.features(x).flatten(1))
