import pytest
from omegaconf import OmegaConf

from bfrb_sensors.models.baseline import BaselineMLPClassifier
from bfrb_sensors.models.factory import build_model
from bfrb_sensors.models.temporal import TemporalConvGRUClassifier


def test_build_model_returns_baseline():
    cfg = OmegaConf.create(
        {
            "name": "baseline_mlp",
            "input_dim": 39,
            "hidden_dim": 16,
            "num_classes": 18,
            "dropout": 0.0,
        }
    )
    assert isinstance(build_model(cfg), BaselineMLPClassifier)


def test_build_model_returns_temporal():
    cfg = OmegaConf.create(
        {
            "name": "temporal_conv_gru",
            "input_dim": 39,
            "hidden_dim": 16,
            "num_classes": 18,
            "dropout": 0.0,
            "num_conv_blocks": 2,
            "gru_layers": 1,
        }
    )
    assert isinstance(build_model(cfg), TemporalConvGRUClassifier)


def test_build_model_rejects_unknown_name():
    cfg = OmegaConf.create({"name": "mystery_net"})
    with pytest.raises(ValueError, match="unknown model"):
        build_model(cfg)
