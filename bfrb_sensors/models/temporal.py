"""Temporal CNN + BiGRU classifier with attention pooling (V2)."""

from __future__ import annotations

import torch
from torch import nn

from bfrb_sensors.models.outputs import ModelOutput


class AttentionPool(nn.Module):
    """Masked attention pooling over the time dimension."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = weights.nan_to_num(0.0)  # guard all-masked rows
        weights = weights.unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, H)


class ConvBlock1D(nn.Module):
    """Residual Conv1D block over the time dimension; preserves (B, T, C)."""

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
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.block(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x + residual


class TemporalConvGRUClassifier(nn.Module):
    """Temporal classifier over concatenated vector features.

    forward expects a batch dict with keys ``imu`` (B,T,7), ``imu_derived`` (B,T,7),
    ``thm`` (B,T,5), ``tof_stats`` (B,T,20) and ``attention_mask`` (B,T). ``input_dim``
    must equal the summed width of the concatenated features (39 by default).
    Returns logits of shape (B, num_classes).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        num_conv_blocks: int = 2,
        gru_layers: int = 1,
        aux_binary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be even (BiGRU uses hidden_dim // 2 per direction), got {hidden_dim}"
            )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        x = self.input_proj(x)
        for block in self.conv_blocks:
            x = block(x)
        x, _ = self.gru(x)
        pooled = self.pool(x, batch["attention_mask"])
        binary_logits = self.binary_head(pooled) if self.binary_head is not None else None
        return ModelOutput(logits=self.classifier(pooled), binary_logits=binary_logits)
