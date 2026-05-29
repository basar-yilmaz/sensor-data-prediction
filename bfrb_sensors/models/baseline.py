from __future__ import annotations

import torch
from torch import nn

from bfrb_sensors.models.outputs import ModelOutput


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.float().unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class BaselineMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        aux_binary: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
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
        encoded = self.encoder(x)
        pooled = masked_mean_pool(encoded, batch["attention_mask"])
        binary_logits = self.binary_head(pooled) if self.binary_head is not None else None
        return ModelOutput(logits=self.classifier(pooled), binary_logits=binary_logits)
