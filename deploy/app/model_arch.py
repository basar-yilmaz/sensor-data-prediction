"""Verbatim copy of the inference-time model architecture.

The deploy service loads Lightning checkpoints saved by
``bfrb_sensors.training.module.BFRBClassificationModule`` and reconstructs the
underlying ``TemporalConvGRUClassifier`` to consume its state dict. Keeping a
self-contained copy here avoids an import dependency on the training package
and lets the service be deployed without the rest of the training pipeline.

If the training-side model architecture changes, mirror the change here. The
classes are intentionally minimal (no dropout ``training`` flags, no aux
binary head) because inference runs in ``eval()`` mode with no aux loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    binary_logits: torch.Tensor | None = None


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = weights.nan_to_num(0.0)
        weights = weights.unsqueeze(-1)
        return (x * weights).sum(dim=1)


class ConvBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2)
        return x + residual


class TofSpatialEncoder(nn.Module):
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
        features = self.cnn(flat)
        embedded = self.proj(features)
        return embedded.reshape(batch_size, timesteps, self.embed_dim)


class TemporalConvGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        num_conv_blocks: int = 2,
        gru_layers: int = 1,
        use_tof_raw: bool = False,
        tof_embed_dim: int = 32,
        aux_binary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be even (BiGRU uses hidden_dim // 2 per direction), got {hidden_dim}"
            )
        self.tof_encoder = (
            TofSpatialEncoder(in_channels=5, embed_dim=tof_embed_dim, dropout=dropout)
            if use_tof_raw
            else None
        )
        proj_in = input_dim + (tof_embed_dim if use_tof_raw else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock1D(hidden_dim, kernel_size=5, dropout=dropout)
                for _ in range(num_conv_blocks)
            ]
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.binary_head = nn.Linear(hidden_dim, 2) if aux_binary else None

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        x = torch.cat(
            [batch["imu"], batch["imu_derived"], batch["thm"], batch["tof_stats"]],
            dim=-1,
        )
        if self.tof_encoder is not None:
            tof_embed = self.tof_encoder(batch["tof"])
            x = torch.cat([x, tof_embed], dim=-1)
        x = self.input_proj(x)
        for block in self.conv_blocks:
            x = block(x)
        x, _ = self.gru(x)
        pooled = self.pool(x, batch["attention_mask"])
        binary_logits = self.binary_head(pooled) if self.binary_head is not None else None
        return ModelOutput(logits=self.classifier(pooled), binary_logits=binary_logits)
