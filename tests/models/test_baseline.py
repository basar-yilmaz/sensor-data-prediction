import torch

from bfrb_sensors.models.baseline import BaselineMLPClassifier, masked_mean_pool


def test_masked_mean_pool_ignores_padded_positions():
    x = torch.tensor([[[1.0], [3.0], [100.0]], [[2.0], [4.0], [6.0]]])
    attention_mask = torch.tensor([[True, True, False], [True, True, True]])

    pooled = masked_mean_pool(x, attention_mask)

    assert torch.equal(pooled, torch.tensor([[2.0], [4.0]]))


def test_baseline_mlp_classifier_returns_class_logits():
    model = BaselineMLPClassifier(input_dim=39, hidden_dim=16, num_classes=18, dropout=0.0)
    batch = {
        "imu": torch.randn(3, 5, 7),
        "imu_derived": torch.randn(3, 5, 7),
        "thm": torch.randn(3, 5, 5),
        "tof": torch.randn(3, 5, 5, 8, 8),
        "tof_stats": torch.randn(3, 5, 20),
        "attention_mask": torch.ones(3, 5, dtype=torch.bool),
    }

    out = model(batch)

    assert out.logits.shape == (3, 18)
