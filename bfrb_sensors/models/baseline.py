import torch
from torch import nn


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.float().unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class BaselineMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            [batch["imu"], batch["imu_derived"], batch["thm"], batch["tof_stats"]],
            dim=-1,
        )
        encoded = self.encoder(x)
        pooled = masked_mean_pool(encoded, batch["attention_mask"])
        return self.classifier(pooled)
