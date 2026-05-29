from __future__ import annotations

import torch

from bfrb_sensors.models.tof import TofSpatialEncoder


def test_tof_encoder_output_shape():
    encoder = TofSpatialEncoder(in_channels=5, embed_dim=16, dropout=0.0)
    frames = torch.randn(4, 6, 5, 8, 8)  # (B, T, C, H, W)
    out = encoder(frames)
    assert out.shape == (4, 6, 16)
