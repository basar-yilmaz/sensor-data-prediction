from __future__ import annotations

import torch

from bfrb_sensors.models.baseline import BaselineMLPClassifier
from bfrb_sensors.models.outputs import ModelOutput


def _batch(batch_size: int = 4, timesteps: int = 6) -> dict[str, torch.Tensor]:
    return {
        "imu": torch.randn(batch_size, timesteps, 7),
        "imu_derived": torch.randn(batch_size, timesteps, 7),
        "thm": torch.randn(batch_size, timesteps, 5),
        "tof": torch.randn(batch_size, timesteps, 5, 8, 8),
        "tof_stats": torch.randn(batch_size, timesteps, 20),
        "attention_mask": torch.ones(batch_size, timesteps, dtype=torch.bool),
    }


def test_baseline_returns_model_output():
    model = BaselineMLPClassifier(input_dim=39, hidden_dim=16, num_classes=3, dropout=0.0)
    out = model(_batch())
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (4, 3)
    assert out.binary_logits is None


def test_temporal_returns_model_output():
    from bfrb_sensors.models.temporal import TemporalConvGRUClassifier

    model = TemporalConvGRUClassifier(input_dim=39, hidden_dim=16, num_classes=3, dropout=0.0)
    out = model(_batch())
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (4, 3)
    assert out.binary_logits is None
