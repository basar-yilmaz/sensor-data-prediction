"""Per-timestep 2D CNN encoder over raw ToF frames (V3)."""

from __future__ import annotations

import torch
from torch import nn


class TofSpatialEncoder(nn.Module):
    """Encode raw ToF frames ``(B, T, C, 8, 8)`` into ``(B, T, embed_dim)``.

    The five ToF sensors are treated as input channels of a small 2D CNN applied
    independently at each timestep, followed by global average pooling and a
    linear projection to ``embed_dim``.
    """

    def __init__(self, in_channels: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = frames.shape
        flat = frames.reshape(batch_size * timesteps, channels, height, width)
        features = self.cnn(flat)  # (B*T, 32, 1, 1)
        embedded = self.proj(features)  # (B*T, embed_dim)
        return embedded.reshape(batch_size, timesteps, self.embed_dim)
