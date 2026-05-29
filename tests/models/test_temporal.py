import torch

from bfrb_sensors.models.temporal import (
    AttentionPool,
    ConvBlock1D,
    TemporalConvGRUClassifier,
)


def test_attention_pool_ignores_padded_positions():
    pool = AttentionPool(hidden_dim=8, dropout=0.0).eval()
    valid = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    a = valid.clone()
    b = valid.clone()
    # Differ only on the padded tail of sample 0.
    b[0, 2] = 999.0

    with torch.no_grad():
        out_a = pool(a, mask)
        out_b = pool(b, mask)

    assert out_a.shape == (2, 8)
    torch.testing.assert_close(out_a, out_b)


def test_conv_block_preserves_shape():
    block = ConvBlock1D(dim=8, kernel_size=5, dropout=0.0).eval()
    x = torch.randn(2, 6, 8)

    with torch.no_grad():
        out = block(x)

    assert out.shape == (2, 6, 8)


def test_temporal_conv_gru_returns_class_logits():
    model = TemporalConvGRUClassifier(
        input_dim=39,
        hidden_dim=16,
        num_classes=18,
        dropout=0.0,
        num_conv_blocks=2,
        gru_layers=1,
    ).eval()
    batch = {
        "imu": torch.randn(3, 5, 7),
        "imu_derived": torch.randn(3, 5, 7),
        "thm": torch.randn(3, 5, 5),
        "tof": torch.randn(3, 5, 5, 8, 8),  # unused by forward; model uses tof_stats
        "tof_stats": torch.randn(3, 5, 20),
        "attention_mask": torch.ones(3, 5, dtype=torch.bool),
    }

    with torch.no_grad():
        out = model(batch)

    assert out.logits.shape == (3, 18)
